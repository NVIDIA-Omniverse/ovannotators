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

"""Image-encode benchmarks: GPU (nvImageCodec) vs CPU (Pillow) fallback.

These benchmarks quantify the speedup of the GPU image codec over the Pillow
fallback that :func:`ovannotators._internal.codecs.get_image_codec` selects
between, across three pipelines:

* ``pillow-host`` — CPU baseline (Pillow on a host NumPy frame).
* ``nvimgcodec-host`` — GPU encode of a *host* frame; nvImageCodec uploads it to
  the device internally (one host->device copy per frame).
* ``nvimgcodec-device`` — GPU encode of a *GPU-resident* Warp CUDA array. This
  is the faithful stand-in for ``OvrtxOutput(device="cuda")``, which returns a
  zero-copy ``wp.from_dlpack`` view of the renderer buffer. It exercises the
  real on-device, no-host-copy path a live ovrtx capture takes (an actual ovrtx
  ``Renderer`` is too heavy/scene-dependent for a microbenchmark, but the array
  type and codec path are identical).

Two layers are measured for each pipeline: the raw ``codec.encode()`` step
(:class:`TestImageEncode`) and the same encode driven through
``BasicWriter.write_frame`` (:class:`TestBasicWriterImageWrite`).

The ``nvimgcodec-*`` pipelines are skipped automatically when no usable CUDA
device + bindings are present (e.g. on CI), so the suite still runs CPU-only.
Run locally with a GPU to get the side-by-side comparison::

    uv run pytest tests/benchmarks/test_bench_image_codecs.py \\
        --override-ini="addopts=" --benchmark-only \\
        --benchmark-group-by=param:image_format,param:resolution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ovannotators._internal.codecs import (
    ImageCodec,
    NvImageCodecBackend,
    PillowCodecBackend,
    _gpu_image_codec_available,
)
from ovannotators.writers.basic import BasicWriterConfig
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators._internal.codecs import ArrayLike


class _EncodeOnlyBackend:
    """Backend stub that encodes image arrays with a fixed codec, no I/O.

    Faithfully mirrors what the real disk/S3/ovstorage backends do for image
    encodings (``buf.write(get_image_codec().encode(...))``) but discards the
    result, so the benchmark isolates encode (and writer) cost from disk and
    network variance. Non-image encodings are no-ops since these benchmarks only
    exercise the ``LdrColor`` image path.
    """

    def __init__(self, codec: ImageCodec) -> None:
        self._codec = codec

    def makedirs(self, path: str) -> None:
        return None

    def write(self, path: str, data: bytes) -> None:
        return None

    def write_array(self, path: str, array: ArrayLike, encoding: str = "npy") -> None:
        if encoding in ("png", "jpg", "jpeg"):
            self._codec.encode(array, format=encoding)

    def exists(self, path: str) -> bool:
        return False


def _place_frame(frame_np, location: str):
    """Return ``frame_np`` on the requested device.

    ``"host"`` keeps the NumPy array; ``"device"`` returns a GPU-resident Warp
    CUDA array — the same zero-copy view type ``OvrtxOutput(device="cuda")``
    hands to writers (``wp.from_dlpack``). Feeding the device array makes the
    benchmark exercise the real ovrtx -> nvImageCodec on-device path with no
    host upload.
    """
    if location == "host":
        return frame_np
    import warp as wp

    return wp.array(frame_np, dtype=wp.uint8, device="cuda")


@pytest.fixture(
    params=["pillow-host", "nvimgcodec-host", "nvimgcodec-device"],
    ids=lambda p: p,
)
def codec_pipeline(request) -> tuple[ImageCodec, str]:
    """Codec + input-location combos (see module docstring).

    The ``nvimgcodec-*`` variants are skipped when GPU encoding is unavailable,
    so the comparison degrades gracefully to CPU-only environments.
    """
    name = request.param
    if name == "pillow-host":
        return PillowCodecBackend(), "host"
    if not _gpu_image_codec_available():
        pytest.skip("nvImageCodec GPU codec unavailable (needs CUDA device + bindings)")
    location = "device" if name.endswith("device") else "host"
    return NvImageCodecBackend(), location


@pytest.fixture(params=["png", "jpg"], ids=lambda f: f)
def image_format(request) -> str:
    """Parametrized image encoding format."""
    return request.param


class TestImageEncode:
    """Raw ``codec.encode()`` throughput across the codec/location pipelines."""

    def test_encode(
        self, benchmark, codec_pipeline: tuple[ImageCodec, str], image_format, rgb_frame
    ) -> None:
        codec, location = codec_pipeline
        frame = _place_frame(rgb_frame, location)

        # Warm up once so first-call codec/encoder setup is not timed.
        encoded = codec.encode(frame, format=image_format)
        assert isinstance(encoded, (bytes, bytearray)) and len(encoded) > 0

        result = benchmark.pedantic(
            codec.encode,
            args=(frame, image_format),
            rounds=10,
            iterations=1,
        )
        assert len(result) > 0


class TestBasicWriterImageWrite:
    """``BasicWriter`` image write_frame() throughput across the pipelines.

    Writes only the ``LdrColor`` image var with colorization disabled, so the
    measured work is the image encode the writer drives through the backend. The
    ``nvimgcodec-device`` pipeline keeps the Warp array on-device end to end,
    matching a live ``OvrtxOutput(device="cuda")`` capture.
    """

    def test_write_frame_image(
        self, benchmark, codec_pipeline: tuple[ImageCodec, str], image_format, rgb_frame
    ) -> None:
        codec, location = codec_pipeline
        frame = _place_frame(rgb_frame, location)
        backend = _EncodeOnlyBackend(codec)
        writer = WriterRegistry.get("basic")(
            BasicWriterConfig(
                render_vars=["LdrColor"],
                image_format=image_format,
                colorize=False,
            )
        )
        render_vars = {"LdrColor": frame}

        # Warm up the codec/encoder once outside the timed region.
        writer.write_frame(0, render_vars, backend)

        benchmark.pedantic(
            writer.write_frame,
            args=(0, render_vars, backend),
            rounds=10,
            iterations=1,
        )
