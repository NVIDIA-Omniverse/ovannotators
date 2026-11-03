# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end live-capture benchmark: ovrtx render -> BasicWriter (GPU only).

Drives an ovrtx ``Renderer`` for ``NUM_STEPS`` steps and writes each frame's
``LdrColor`` image to disk via :class:`~ovannotators.writers.basic.BasicWriter`,
exercising the full capture path: GPU render -> ``OvrtxOutput`` extraction ->
writer encode -> ``DiskBackend`` write.

Two scheduling strategies are compared (parametrized as ``mode``):

* ``serial`` — ``renderer.step()`` per frame. ``step()`` blocks immediately
  (it is ``step_async().wait().fetch()``), so the GPU is idle while the CPU
  extracts and writes the frame.
* ``pipelined`` — ``renderer.step_async()`` keeps ``DEPTH`` renders in flight,
  so the GPU renders the next frame(s) while the CPU writes the current one.
  This overlaps the GPU render with the CPU encode/write and is the pattern
  documented in Tutorial 9.

The library itself is unchanged between the two modes — only the renderer call
scheduling differs. The frame is extracted on the GPU (``device="cuda"``),
matching a production zero-copy capture; the image encode uses nvImageCodec when
the GPU extra is present and falls back to Pillow otherwise.

Requires a GPU + ``ovrtx`` (marked ``gpu``; skipped on CI). Output is written to
a throwaway temp directory that is deleted automatically when the benchmark
finishes.

Run locally::

    uv run pytest tests/benchmarks/test_bench_ovrtx_pipeline.py \\
        --override-ini="addopts=" -m gpu --benchmark-only
"""

from __future__ import annotations

import tempfile
from collections import deque
from pathlib import Path
from typing import Callable

import pytest

from ovannotators._internal.optional_deps import ovrtx_available

pytestmark = pytest.mark.gpu

_HAS_OVRTX = ovrtx_available()

# Universal test scene that declares a RenderProduct requesting LdrColor.
SCENE_PATH = Path(__file__).parent.parent / "data" / "test_scene.usda"
RENDER_PRODUCT = "/Render/RenderProduct_001"

# Frames to settle the path tracer before timing (texture streaming + sample
# accumulation), so the measured steps reflect steady-state capture cost.
WARMUP_STEPS = 20

# Number of render+write steps the benchmark drives per round.
NUM_STEPS = 500

# Renders kept in flight by the pipelined mode (overlaps GPU render with the
# CPU encode/write). 2 saturates the GPU here; higher rarely helps.
DEPTH = 2

DELTA_TIME = 1.0 / 60.0


@pytest.fixture(scope="module")
def renderer():
    """Open the test scene and settle the path tracer once for the module."""
    pytest.importorskip("ovrtx")
    pytest.importorskip("ovstage")
    from tests._ovstage_session import OvstageSession

    session = OvstageSession("bench-ovrtx-pipeline")
    session.open_usd(SCENE_PATH.resolve())
    for _ in range(WARMUP_STEPS):
        session.step({RENDER_PRODUCT}, DELTA_TIME)
    try:
        yield session
    finally:
        # Destroy the renderer deterministically while the interpreter is fully
        # alive instead of leaving it to shutdown GC (which can race ovrtx/omni
        # teardown and hang).
        session.close()


def _capture_serial(renderer, extractor, writer, backend) -> int:
    """Serial capture: ``step()`` blocks, so render and write do not overlap."""
    frame_id = 0
    for _ in range(NUM_STEPS):
        outputs = renderer.step({RENDER_PRODUCT}, DELTA_TIME)
        for frame in outputs[RENDER_PRODUCT].frames:
            ldr = extractor.extract(frame, "LdrColor", device="cuda")
            writer.write_frame(frame_id, {"LdrColor": ldr}, backend)
            frame_id += 1
    return frame_id


def _capture_pipelined(renderer, extractor, writer, backend) -> int:
    """Pipelined capture: keep ``DEPTH`` renders in flight via ``step_async``.

    Queues the next render(s) before collecting and writing the oldest in-flight
    frame, so the GPU renders ahead while the CPU encodes/writes. Always drains
    ``inflight`` so no ``step_async`` result is dropped without ``fetch()``.
    """
    inflight: deque = deque()
    issued = frame_id = 0
    while frame_id < NUM_STEPS:
        while issued < NUM_STEPS and len(inflight) < DEPTH:
            inflight.append(renderer.step_async({RENDER_PRODUCT}, DELTA_TIME))
            issued += 1
        op = inflight.popleft()
        outputs = op.wait().fetch()
        for frame in outputs[RENDER_PRODUCT].frames:
            ldr = extractor.extract(frame, "LdrColor", device="cuda")
            writer.write_frame(frame_id, {"LdrColor": ldr}, backend)
            frame_id += 1
    return frame_id


_CAPTURE_FNS: dict[str, Callable] = {
    "serial": _capture_serial,
    "pipelined": _capture_pipelined,
}


@pytest.fixture(params=list(_CAPTURE_FNS), ids=lambda m: m)
def capture_mode(request) -> str:
    """Capture scheduling strategy: ``serial`` or ``pipelined``."""
    return request.param


@pytest.mark.skipif(not _HAS_OVRTX, reason="ovrtx not installed")
class TestOvrtxBasicWriterCapture:
    """Live ovrtx -> BasicWriter capture loop throughput (serial vs pipelined)."""

    def test_capture_ldr_color_500_steps(self, benchmark, renderer, capture_mode):
        """Render and write ``LdrColor`` for ``NUM_STEPS`` steps.

        The whole loop runs inside a ``TemporaryDirectory`` context, so every
        file the writer produces is deleted automatically once the benchmark
        completes (including on failure).
        """
        from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
        from ovannotators.renderer_outputs import OvrtxOutput
        from ovannotators.writers import BasicWriter, BasicWriterConfig

        extractor = OvrtxOutput()
        capture_fn = _CAPTURE_FNS[capture_mode]

        def run() -> None:
            # New throwaway dir each round; deleted when the context exits.
            with tempfile.TemporaryDirectory(prefix="ovann_bench_") as out_dir:
                backend = DiskBackend(
                    DiskBackendConfig(root_dir=out_dir, overwrite=True)
                )
                # Only LdrColor, no colorized companions: isolate the
                # image-write path so the benchmark reflects raw capture
                # throughput.
                writer = BasicWriter(
                    BasicWriterConfig(
                        render_vars=["LdrColor"],
                        image_format="png",
                        colorize=False,
                    )
                )
                written = capture_fn(renderer, extractor, writer, backend)
                writer.finalize(backend)
                assert written == NUM_STEPS

        # One round: NUM_STEPS GPU renders + writes is already a heavy, stable
        # sample, and avoids unbounded disk churn across repeated rounds.
        benchmark.pedantic(run, rounds=1, iterations=1)
