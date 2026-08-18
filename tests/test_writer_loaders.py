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

"""GPU-only loader-roundtrip tests for every writer (except BasicWriter).

Each test renders ``test_scene.usda`` once (via the session-scoped
``rendered_scene_vars`` fixture), runs a single writer against a real
:class:`DiskBackend`, then feeds the writer's output through the matching
*vendored* third-party loader script in ``tests/dataset_loaders/``. The loaders
import standard reader libraries (pycocotools, ultralytics, bop_toolkit_lib,
pyarrow/PyAV) that are installed only in a separate venv, so they run as a
subprocess under that interpreter and must exit 0 (they ``raise`` on parse
failure).

These are marked ``gpu`` so CI (which runs ``-m 'not gpu'``) skips them; run
them locally on a GPU box with::

    python3 -m pytest tests/test_writer_loaders.py -o addopts="" -rA -v
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers import (
    BOPWriter,
    COCOWriter,
    KITTIWriter,
    LeRobotWriter,
    YOLOWriter,
)

from ovannotators._internal.optional_deps import ovrtx_available

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not ovrtx_available(),
        reason="ovrtx not installed — skipping rendered-scene loader tests",
    ),
]

# Interpreter with the third-party reader libraries installed (separate from the
# main test env, which intentionally lacks them). Overridable for portability.
READERS_PYTHON = os.environ.get(
    "OVANNOTATORS_READERS_PYTHON", "/tmp/readers_venv/bin/python"
)

# Directory holding the verbatim vendored loader scripts.
LOADERS_DIR = Path(__file__).parent / "dataset_loaders"

# pynvvideocodec ships the NVENC shared libs the lerobot loader's PyAV decode
# path needs on its LD_LIBRARY_PATH.
_PYNVVIDEOCODEC_LIB = "/home/horde/.local/lib/python3.10/site-packages/pynvvideocodec"


def _has_functional_nvvideocodec() -> bool:
    """Check if PyNvVideoCodec can actually encode (mirrors test_rendered_writers)."""
    try:
        from ovannotators._internal.codecs import get_video_encoder
        import numpy as np

        enc = get_video_encoder("", 24, 256, 256)
        enc.write_frame(np.zeros((256, 256, 3), dtype=np.uint8))
        enc.close()
        return True
    except Exception:
        return False


_HAS_NVVIDEOCODEC = _has_functional_nvvideocodec()


def _backend(tmp_path: Path) -> DiskBackend:
    return DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))


def _run_writer(writer, render_vars: dict, tmp_path: Path) -> DiskBackend:
    """Write one frame + finalize, skipping if a required var is absent."""
    missing = [v for v in writer.required_render_vars if v not in render_vars]
    if missing:
        pytest.skip(f"rendered scene missing required vars: {missing}")
    backend = _backend(tmp_path)
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)
    return backend


def _run_loader(name: str, dataset_dir: Path, env: dict | None = None) -> None:
    """Run a vendored loader under the readers venv; assert it parses (exit 0)."""
    script = LOADERS_DIR / f"load_{name}.py"
    if not Path(READERS_PYTHON).exists():
        pytest.skip(f"readers venv interpreter not found at {READERS_PYTHON}")
    proc = subprocess.run(
        [READERS_PYTHON, str(script), str(dataset_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"load_{name}.py failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def test_coco_loader(rendered_scene_vars, tmp_path) -> None:
    _run_writer(COCOWriter(), rendered_scene_vars, tmp_path)
    _run_loader("coco", tmp_path)


def test_kitti_loader(rendered_scene_vars, tmp_path) -> None:
    _run_writer(KITTIWriter(), rendered_scene_vars, tmp_path)
    _run_loader("kitti", tmp_path)


def test_yolo_loader(rendered_scene_vars, tmp_path) -> None:
    _run_writer(YOLOWriter(), rendered_scene_vars, tmp_path)
    _run_loader("yolo", tmp_path)


def test_bop_loader(rendered_scene_vars, tmp_path) -> None:
    _run_writer(BOPWriter(), rendered_scene_vars, tmp_path)
    _run_loader("bop", tmp_path)


@pytest.mark.skipif(not _HAS_NVVIDEOCODEC, reason="PyNvVideoCodec not functional")
def test_lerobot_loader(rendered_scene_vars, tmp_path) -> None:
    _run_writer(LeRobotWriter(), rendered_scene_vars, tmp_path)
    # PyAV's NVENC-encoded video decode needs the pynvvideocodec shared libs.
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{_PYNVVIDEOCODEC_LIB}:{existing}" if existing else _PYNVVIDEOCODEC_LIB
    )
    _run_loader("lerobot", tmp_path, env=env)
