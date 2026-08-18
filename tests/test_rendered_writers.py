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

"""ovrtx-gated end-to-end writer tests against a real rendered scene.

These exercise every writer with the AOVs extracted from ``test_scene.usda``
via ovrtx (the ``rendered_scene_vars`` session fixture). They are skipped when
ovrtx / a GPU is unavailable; per-writer cases additionally skip when the
rendered product did not emit a render var that writer requires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers import (
    BasicWriter,
    BOPWriter,
    COCOWriter,
    COCOWriterConfig,
    KITTIWriter,
    LeRobotWriter,
    YOLOWriter,
)
from ovannotators.writers.cosmos_transfer import CosmosTransferWriter

from ovannotators._internal.optional_deps import ovrtx_available

_OVRTX_AVAILABLE = ovrtx_available()


def _has_functional_nvvideocodec() -> bool:
    """Check if PyNvVideoCodec can actually encode."""
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

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _OVRTX_AVAILABLE,
        reason="ovrtx not installed — skipping rendered-scene writer tests",
    ),
]


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


def test_basic_writer_rendered(rendered_scene_vars, tmp_path) -> None:
    backend = _run_writer(BasicWriter(), rendered_scene_vars, tmp_path)
    assert any(backend.list("0000"))


def test_coco_writer_rendered(rendered_scene_vars, tmp_path) -> None:
    _run_writer(COCOWriter(), rendered_scene_vars, tmp_path)
    assert (tmp_path / "annotations" / "instances_train.json").exists()


def test_coco_writer_panoptic_rendered(rendered_scene_vars, tmp_path) -> None:
    writer = COCOWriter(COCOWriterConfig(include_panoptic=True))
    # The COCO writer's panoptic path expects a 2-D instance map; ovrtx emits a
    # multi-channel (H, W, 4) instance buffer, so collapse to the first channel
    # exactly as the golden panoptic fixture does.
    render_vars = dict(rendered_scene_vars)
    inst_seg = render_vars.get("SemanticInstanceSegmentation")
    if getattr(inst_seg, "ndim", 0) > 2:
        render_vars["SemanticInstanceSegmentation"] = inst_seg[..., 0]
    _run_writer(writer, render_vars, tmp_path)
    assert (tmp_path / "annotations" / "panoptic_train.json").exists()
    assert (tmp_path / "annotations" / "panoptic_train" / "000000.png").exists()


def test_kitti_writer_rendered(rendered_scene_vars, tmp_path) -> None:
    _run_writer(KITTIWriter(), rendered_scene_vars, tmp_path)
    assert (tmp_path / "label_2" / "000000.txt").exists()
    assert (tmp_path / "calib" / "000000.txt").exists()


def test_yolo_writer_rendered(rendered_scene_vars, tmp_path) -> None:
    _run_writer(YOLOWriter(), rendered_scene_vars, tmp_path)
    assert (tmp_path / "data.yaml").exists()


def test_bop_writer_rendered(rendered_scene_vars, tmp_path) -> None:
    _run_writer(BOPWriter(), rendered_scene_vars, tmp_path)
    assert (tmp_path / "000000" / "scene_gt.json").exists()


@pytest.mark.skipif(not _HAS_NVVIDEOCODEC, reason="PyNvVideoCodec not functional")
def test_lerobot_writer_rendered(rendered_scene_vars, tmp_path) -> None:
    _run_writer(LeRobotWriter(), rendered_scene_vars, tmp_path)
    assert (tmp_path / "meta" / "info.json").exists()


@pytest.mark.skipif(not _HAS_NVVIDEOCODEC, reason="PyNvVideoCodec not functional")
def test_cosmos_writer_rendered(rendered_scene_vars, tmp_path) -> None:
    _run_writer(CosmosTransferWriter(), rendered_scene_vars, tmp_path)
    assert (tmp_path / "manifest.json").exists()
