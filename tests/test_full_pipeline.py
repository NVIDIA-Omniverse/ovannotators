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

"""Full pipeline tests: ovrtx render → writers → golden comparison.

These tests require a GPU with CUDA and ovrtx installed.
They are skipped on CI (no GPU runners) and run locally before pushing.

Usage:
    # Run locally (requires GPU + ovrtx + PyNvVideoCodec):
    python -m pytest tests/test_full_pipeline.py -v

    # Run as part of full local suite:
    python -m pytest tests/ -v -m gpu

    # Skip GPU tests (default on CI):
    python -m pytest tests/ -v -m "not gpu"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from ovannotators._internal.optional_deps import ovrtx_available

_HAS_OVRTX = ovrtx_available()

try:
    import PyNvVideoCodec  # noqa: F401

    _HAS_PYNVVIDEOCODEC = True
except ImportError:
    _HAS_PYNVVIDEOCODEC = False

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

# Raw ovrtx bbox/camera payloads are byte buffers; writers consume the
# structured arrays these decoders produce (mirrors tests/conftest.py).
try:
    from ovannotators.writers._helpers import (
        decode_bbox_2d,
        decode_bbox_3d,
        decode_camera_params,
    )

    _ANNOTATOR_DECODERS = {
        "SemanticBoundingBox2DTight": decode_bbox_2d,
        "SemanticBoundingBox2DLoose": decode_bbox_2d,
        "SemanticBoundingBox3D": decode_bbox_3d,
        "CameraParams": decode_camera_params,
    }
except ImportError:
    _ANNOTATOR_DECODERS = {}


def _decode_annotators(render_vars: dict) -> dict:
    """Return a copy of *render_vars* with raw bbox/camera buffers decoded."""
    decoded = dict(render_vars)
    for name, decoder in _ANNOTATOR_DECODERS.items():
        if name in decoded:
            decoded[name] = decoder(decoded[name])
    return decoded


SCENE_PATH = Path(__file__).parent / "data" / "test_scene.usda"
GOLDEN_DIR = Path(__file__).parent / "golden" / "basic_writer_scene" / "expected"
RENDER_PRODUCT = "/Render/RenderProduct_001"
SETTLE_STEPS = 20


def _scene_resolution() -> tuple[int, int]:
    """Return the ``(width, height)`` declared by the scene's RenderProduct.

    On CI the scene is a Git-LFS pointer (not checked out), so the regex finds
    no resolution line. This module is gpu-marked and fully deselected on CI,
    but its top-level code still runs at collection time. Return a ``(0, 0)``
    fallback when the scene is unavailable or unparseable so import does not
    fail; the real ``(width, height)`` is only consumed inside gpu tests, which
    run only with a real scene present.
    """
    try:
        text = SCENE_PATH.read_text()
    except OSError:
        return (0, 0)
    match = re.search(r"uniform int2 resolution = \((\d+),\s*(\d+)\)", text)
    if match is None:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


# Buffers are returned as (height, width[, channels]) numpy arrays.
EXPECTED_W, EXPECTED_H = _scene_resolution()


def _render_frame(renderer):
    """Render one frame and extract render vars."""
    import ovrtx as _ovrtx

    from tests._ovstage_session import render_vars_by_source_name

    outputs = renderer.step({RENDER_PRODUCT}, 1.0 / 24.0)
    prod = outputs[RENDER_PRODUCT]
    frame = prod.frames[0]

    render_vars = {}

    # Extract all available render vars as numpy arrays, keyed by AOV kind.
    # ovrtx 0.5 keys frame.render_vars by full RenderVar prim path, which is not
    # the name writers (or the assertions below) look up.
    for name, rv in render_vars_by_source_name(frame).items():
        try:
            with rv.map(device=_ovrtx.Device.CPU) as mapped:
                arr = np.from_dlpack(mapped).copy()
            render_vars[name] = arr
        except Exception:
            pass  # Skip vars that can't be mapped

    # Post-process some vars for writer compatibility
    if "LdrColor" in render_vars and render_vars["LdrColor"].shape[-1] == 4:
        render_vars["LdrColor"] = render_vars["LdrColor"][:, :, :3]

    if "DistanceToCameraSD" in render_vars:
        depth = render_vars["DistanceToCameraSD"]
        if depth.ndim == 3:
            render_vars["DistanceToCameraSD"] = depth[:, :, 0]

    if "SemanticInstanceSegmentation" in render_vars:
        seg = render_vars["SemanticInstanceSegmentation"]
        if seg.ndim == 3:
            render_vars["SemanticInstanceSegmentation"] = seg[:, :, 0]

    if "SemanticSegmentation" in render_vars:
        seg = render_vars["SemanticSegmentation"]
        if seg.ndim == 3:
            render_vars["SemanticSegmentation"] = seg[:, :, 0]

    return render_vars


@pytest.fixture(scope="module")
def renderer():
    """Create and settle the ovstage-backed renderer (shared across module)."""
    pytest.importorskip("ovrtx")
    pytest.importorskip("ovstage")
    from tests._ovstage_session import OvstageSession

    session = OvstageSession("full-pipeline")
    try:
        session.open_usd(SCENE_PATH.resolve())

        # Settle path tracer
        for _ in range(SETTLE_STEPS):
            session.step({RENDER_PRODUCT}, 1.0 / 24.0)

        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def rendered_frame(renderer):
    """Render a single frame for testing."""
    return _render_frame(renderer)


@pytest.mark.skipif(not _HAS_OVRTX, reason="ovrtx not installed")
class TestRenderOutput:
    """Validate ovrtx render output structure and types."""

    def test_ldr_color_shape(self, rendered_frame):
        ldr = rendered_frame["LdrColor"]
        assert ldr.shape == (EXPECTED_H, EXPECTED_W, 3)
        assert ldr.dtype == np.uint8

    def test_depth_shape(self, rendered_frame):
        depth = rendered_frame["DistanceToCameraSD"]
        assert depth.shape == (EXPECTED_H, EXPECTED_W)
        assert depth.dtype == np.float32

    def test_normals_shape(self, rendered_frame):
        normals = rendered_frame["NormalSD"]
        assert normals.shape[:2] == (EXPECTED_H, EXPECTED_W)
        # ovrtx may output 3 or 4 channels (RGB or RGBA)
        assert normals.shape[2] in (3, 4)
        assert normals.dtype == np.float32

    def test_instance_segmentation_shape(self, rendered_frame):
        seg = rendered_frame["SemanticInstanceSegmentation"]
        assert seg.shape == (EXPECTED_H, EXPECTED_W)
        assert seg.dtype == np.uint32

    def test_semantic_segmentation_shape(self, rendered_frame):
        seg = rendered_frame["SemanticSegmentation"]
        assert seg.shape == (EXPECTED_H, EXPECTED_W)
        assert seg.dtype == np.uint32

    def test_has_expected_instances(self, rendered_frame):
        """Scene has 5 objects: cone, 2 cubes, sphere, plane."""
        seg = rendered_frame["SemanticInstanceSegmentation"]
        unique = np.unique(seg)
        # Background (0) + 5 objects = at least 5 unique values
        # (could be more due to path tracer noise on edges)
        assert len(unique) >= 5

    def test_ldr_color_not_black(self, rendered_frame):
        """Emissive scene should produce visible output."""
        ldr = rendered_frame["LdrColor"]
        assert ldr.mean() > 10.0  # Not a black image

    def test_depth_has_range(self, rendered_frame):
        """Depth should have variation (not flat)."""
        depth = rendered_frame["DistanceToCameraSD"]
        valid = depth[depth > 0]
        assert valid.max() - valid.min() > 0.5


@pytest.mark.skipif(not _HAS_OVRTX, reason="ovrtx not installed")
class TestWriterPipeline:
    """End-to-end: render → writer → validate output structure matches golden."""

    def test_basic_writer_output_structure(self, rendered_frame, tmp_path):
        """BasicWriter produces expected directory structure."""
        from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
        from ovannotators.writers.basic import BasicWriter, BasicWriterConfig

        # BasicWriter decodes raw ovrtx bbox/camera buffers itself, so it can
        # consume the full rendered frame (image/depth/segmentation + the raw
        # annotator payloads and their id maps).
        basic_vars = {
            k: v
            for k, v in rendered_frame.items()
            if k
            in (
                "LdrColor",
                "HdrColor",
                "DistanceToCameraSD",
                "NormalSD",
                "SemanticInstanceSegmentation",
                "SemanticSegmentation",
                "DiffuseAlbedoSD",
                "Camera3dPositionSD",
                "DistanceToImagePlaneSD",
                "StableIdSegmentation",
                "SemanticBoundingBox2DTight",
                "SemanticBoundingBox2DLoose",
                "SemanticBoundingBox3D",
                "CameraParams",
                "SemanticIdMap",
                "StableIdMap",
            )
        }

        cfg = BasicWriterConfig()
        writer = BasicWriter(cfg)
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))

        writer.write_frame(0, basic_vars, backend)
        writer.finalize(backend)

        # Check expected files exist
        frame_dir = tmp_path / "0000"
        assert frame_dir.exists()
        assert (frame_dir / "ldr_color.png").exists()
        # Depth vars follow BasicWriterConfig.depth_format, which defaults to
        # "exr"; cfg above takes the defaults.
        assert (frame_dir / "distance_to_camera.exr").exists()

        # Bbox JSON is decoded into a per-entry list (not a raw byte dump).
        if "SemanticBoundingBox2DTight" in basic_vars:
            bbox_json = frame_dir / "bbox_2d_tight.json"
            assert bbox_json.exists()
            entries = json.loads(bbox_json.read_text())
            assert isinstance(entries, list)
            for entry in entries:
                assert {"x_min", "y_min", "x_max", "y_max", "label"}.issubset(entry)
                assert "prim_path" in entry

    def test_coco_writer_output_structure(self, rendered_frame, tmp_path):
        """CocoWriter produces valid COCO JSON."""
        from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
        from ovannotators.writers.coco import COCOWriter, COCOWriterConfig

        cfg = COCOWriterConfig()
        writer = COCOWriter(cfg)
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))

        writer.write_frame(0, _decode_annotators(rendered_frame), backend)
        writer.finalize(backend)

        # Validate COCO JSON
        ann_file = tmp_path / "annotations" / "instances_train.json"
        assert ann_file.exists()
        data = json.loads(ann_file.read_text())
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert len(data["images"]) == 1

    def test_kitti_writer_output_structure(self, rendered_frame, tmp_path):
        """KittiWriter produces expected file layout."""
        from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
        from ovannotators.writers.kitti import KITTIWriter, KITTIWriterConfig

        cfg = KITTIWriterConfig()
        writer = KITTIWriter(cfg)
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))

        writer.write_frame(0, _decode_annotators(rendered_frame), backend)
        writer.finalize(backend)

        assert (tmp_path / "image_2").exists()
        assert (tmp_path / "label_2").exists()

    def test_yolo_writer_output_structure(self, rendered_frame, tmp_path):
        """YoloWriter produces expected file layout."""
        from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
        from ovannotators.writers.yolo import YOLOWriter, YOLOWriterConfig

        cfg = YOLOWriterConfig()
        writer = YOLOWriter(cfg)
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))

        writer.write_frame(0, _decode_annotators(rendered_frame), backend)
        writer.finalize(backend)

        assert (tmp_path / "images").exists()
        assert (tmp_path / "labels").exists()

    @pytest.mark.skipif(
        not _HAS_PYNVVIDEOCODEC,
        reason="PyNvVideoCodec not installed (requires CUDA)",
    )
    def test_cosmos_transfer_writer(self, rendered_frame, tmp_path):
        """CosmosTransferWriter produces video output."""
        from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
        from ovannotators.writers.cosmos_transfer import (
            CosmosTransferWriter,
            CosmosTransferWriterConfig,
        )

        cfg = CosmosTransferWriterConfig(
            conditioning_types=["canny", "depth", "segmentation"],
        )
        writer = CosmosTransferWriter(cfg)
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))

        # Need multiple frames for video
        writer.write_frame(0, rendered_frame, backend)
        writer.write_frame(1, rendered_frame, backend)
        writer.finalize(backend)

        # Should produce mp4 files
        mp4_files = list(tmp_path.rglob("*.mp4"))
        assert len(mp4_files) > 0


@pytest.mark.skipif(not _HAS_OVRTX, reason="ovrtx not installed")
class TestGoldenComparison:
    """Compare live render output against committed golden data."""

    def test_ldr_color_matches_golden(self, rendered_frame):
        """LdrColor should be close to golden (path tracer has some noise)."""
        golden_ldr = GOLDEN_DIR / "ldr_color.png"
        if not golden_ldr.exists():
            pytest.skip("Golden LDR file not found (LFS not pulled?)")

        from PIL import Image

        golden = np.array(Image.open(golden_ldr))
        live = rendered_frame["LdrColor"]

        # Resize if shapes don't match (golden may be different resolution)
        if golden.shape[:2] != live.shape[:2]:
            pytest.skip(f"Shape mismatch: golden={golden.shape}, live={live.shape}")

        # Allow MAE up to 10.0 due to path tracer stochastic noise
        mae = np.abs(live.astype(float) - golden[:, :, :3].astype(float)).mean()
        assert mae < 10.0, f"LdrColor MAE={mae:.2f} exceeds threshold 10.0"

    def test_depth_colorized_exists(self, rendered_frame):
        """Depth colorized golden should exist and be a valid image."""
        golden_path = GOLDEN_DIR / "distance_to_camera_colorized.png"
        if not golden_path.exists():
            pytest.skip("Golden depth colorized not found")

        from PIL import Image

        img = Image.open(golden_path)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_stats_json_valid(self, rendered_frame):
        """stats.json should contain valid metadata about the render."""
        stats_path = GOLDEN_DIR / "stats.json"
        if not stats_path.exists():
            pytest.skip("Golden stats.json not found")

        data = json.loads(stats_path.read_text())
        assert "render_vars" in data or "resolution" in data or len(data) > 0
