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

"""Golden integration tests for COCO/KITTI driven by ``test_scene.usda`` data.

Run ``UPDATE_GOLDEN=1 pytest tests/test_golden_writers.py`` to regenerate
goldens from the current writer behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
from PIL import Image

from tests._golden_metrics import (
    ImageMetricThresholds,
    assert_image_metrics,
    compute_metrics,
    load_rgb_u8,
    write_golden_image,
)
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers import BOPWriter, YOLOWriter
from ovannotators.writers.coco import COCOWriter, COCOWriterConfig
from ovannotators.writers.cosmos_transfer import (
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
)
from ovannotators.writers.kitti import KITTIWriter
from ovannotators.writers.lerobot import LeRobotWriter, LeRobotWriterConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "golden"
COCO_EXPECTED = GOLDEN_DIR / "coco" / "expected"
KITTI_EXPECTED = GOLDEN_DIR / "kitti" / "expected"
YOLO_EXPECTED = GOLDEN_DIR / "yolo_writer" / "expected"
BOP_EXPECTED = GOLDEN_DIR / "bop_writer" / "expected"
COSMOS_EXPECTED = GOLDEN_DIR / "cosmos_transfer_writer" / "expected"
LEROBOT_EXPECTED = GOLDEN_DIR / "lerobot_writer" / "expected"

UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN", "0") == "1"


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _assert_json_close(actual, expected, tol: float = 1e-3) -> None:
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert set(actual) == set(expected)
        for key in actual:
            _assert_json_close(actual[key], expected[key], tol=tol)
        return
    if isinstance(actual, list):
        assert isinstance(expected, list)
        assert len(actual) == len(expected)
        for a_item, e_item in zip(actual, expected):
            _assert_json_close(a_item, e_item, tol=tol)
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert abs(float(actual) - float(expected)) <= tol
        return
    assert actual == expected


def _h264_probe(path: Path) -> dict[str, object]:
    """Probe an encoded video clip.

    PyNvVideoCodec's ``CreateEncoder`` emits a raw H.264 Annex-B elementary
    stream (not a muxed MP4 container), so the clip begins with a NAL start
    code rather than an ``ftyp`` box.
    """
    payload = path.read_bytes()
    return {
        "size_bytes": len(payload),
        "has_ftyp_box": b"ftyp" in payload[:64],
        "has_moov_box": b"moov" in payload,
        "has_annexb_startcode": (
            payload.startswith(b"\x00\x00\x00\x01")
            or payload.startswith(b"\x00\x00\x01")
        ),
    }


def _to_rgb_u8(frame: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary image frame to an ``(H, W, 3)`` uint8 RGB array."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def _assert_frame_matches_golden(
    frame: np.ndarray,
    golden_path: Path,
    thresholds: ImageMetricThresholds,
) -> None:
    """Compare an in-memory RGB frame against a committed golden PNG.

    Used to give content coverage to video outputs (LeRobot / Cosmos) whose
    encoded MP4 bytes are non-deterministic: the frames fed into the encoder are
    deterministic and can be golden-checked directly.

    Args:
        frame: Produced frame (2-D, RGB, or RGBA; uint8 or float).
        golden_path: Golden PNG location.
        thresholds: Tolerant metric thresholds for pass/fail.
    """
    rgb = _to_rgb_u8(frame)
    if UPDATE_GOLDEN:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(golden_path)
        return

    assert golden_path.exists(), (
        f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
    )
    golden = load_rgb_u8(golden_path)
    assert rgb.shape == golden.shape, (
        f"frame shape mismatch: {rgb.shape} vs {golden.shape}"
    )
    metrics = compute_metrics(rgb, golden)
    assert metrics["mae"] <= thresholds.mae_max, (
        f"MAE {metrics['mae']:.4f} > {thresholds.mae_max:.4f}"
    )
    assert metrics["rmse"] <= thresholds.rmse_max, (
        f"RMSE {metrics['rmse']:.4f} > {thresholds.rmse_max:.4f}"
    )
    assert metrics["psnr"] >= thresholds.psnr_min, (
        f"PSNR {metrics['psnr']:.4f} < {thresholds.psnr_min:.4f}"
    )


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coco_out(tmp_path_factory, rendered_scene_vars):
    required = {"LdrColor", "SemanticBoundingBox2DTight"}
    missing = sorted(required - set(rendered_scene_vars))
    if missing:
        pytest.skip(f"Rendered scene missing COCO vars: {missing}")
    tmp = tmp_path_factory.mktemp("coco")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp), overwrite=True))
    writer = COCOWriter(COCOWriterConfig(split="train", image_format="jpg"))
    render_vars = {
        "LdrColor": rendered_scene_vars["LdrColor"],
        "SemanticBoundingBox2DTight": rendered_scene_vars["SemanticBoundingBox2DTight"],
    }
    if "SemanticIdMap" in rendered_scene_vars:
        render_vars["SemanticIdMap"] = rendered_scene_vars["SemanticIdMap"]
    writer.write_frame(frame_id=0, render_vars=render_vars, backend=backend)
    writer.finalize(backend)
    return tmp


@pytest.fixture(scope="module")
def coco_panoptic_out(tmp_path_factory, rendered_scene_vars):
    required = {
        "LdrColor",
        "SemanticBoundingBox2DTight",
        "SemanticInstanceSegmentation",
    }
    missing = sorted(required - set(rendered_scene_vars))
    if missing:
        pytest.skip(f"Rendered scene missing COCO panoptic vars: {missing}")
    tmp = tmp_path_factory.mktemp("coco_panoptic")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp), overwrite=True))
    writer = COCOWriter(
        COCOWriterConfig(split="train", image_format="jpg", include_panoptic=True)
    )
    inst_seg = rendered_scene_vars["SemanticInstanceSegmentation"]
    if getattr(inst_seg, "ndim", 0) > 2:
        inst_seg = inst_seg[..., 0]

    render_vars = {
        "LdrColor": rendered_scene_vars["LdrColor"],
        "SemanticBoundingBox2DTight": rendered_scene_vars["SemanticBoundingBox2DTight"],
        "SemanticInstanceSegmentation": inst_seg,
    }
    if "SemanticIdMap" in rendered_scene_vars:
        render_vars["SemanticIdMap"] = rendered_scene_vars["SemanticIdMap"]
    writer.write_frame(frame_id=0, render_vars=render_vars, backend=backend)
    writer.finalize(backend)
    return tmp


class TestCOCOGolden:
    def test_instances_json(self, coco_out) -> None:
        result_path = coco_out / "annotations" / "instances_train.json"
        result = json.loads(result_path.read_text())

        golden_path = COCO_EXPECTED / "instances_train.json"

        if UPDATE_GOLDEN:
            COCO_EXPECTED.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(json.dumps(result, indent=2))
            return

        assert golden_path.exists(), (
            f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
        )
        golden = json.loads(golden_path.read_text())
        _assert_coco_equal(result, golden)

    def test_image_file(self, coco_out) -> None:
        result_path = coco_out / "images" / "000000.jpg"
        golden_path = COCO_EXPECTED / "image_000000.jpg"
        if UPDATE_GOLDEN:
            write_golden_image(result_path, golden_path)
            return
        # JPEG output can vary slightly across codec/runtime versions.
        assert_image_metrics(
            result_path,
            golden_path,
            ImageMetricThresholds(mae_max=2.0, rmse_max=4.0, psnr_min=35.0),
        )

    def test_panoptic_output_shape_and_consistency(self, coco_panoptic_out) -> None:
        panoptic_json_path = coco_panoptic_out / "annotations" / "panoptic_train.json"
        assert panoptic_json_path.exists()
        panoptic = json.loads(panoptic_json_path.read_text())
        assert {"images", "annotations", "categories"}.issubset(panoptic.keys())
        assert len(panoptic["images"]) == 1
        assert len(panoptic["annotations"]) == 1

        ann = panoptic["annotations"][0]
        png_path = (
            coco_panoptic_out / "annotations" / "panoptic_train" / ann["file_name"]
        )
        with Image.open(png_path) as image:
            assert image.mode == "RGB"
            assert image.width == panoptic["images"][0]["width"]
            assert image.height == panoptic["images"][0]["height"]

        with Image.open(png_path) as image:
            encoded = np.asarray(image)
        panoptic_ids = (
            encoded[..., 0].astype(np.uint32)
            + (encoded[..., 1].astype(np.uint32) << 8)
            + (encoded[..., 2].astype(np.uint32) << 16)
        )
        png_segment_ids = {int(v) for v in np.unique(panoptic_ids) if int(v) > 0}
        json_segment_ids = {int(seg["id"]) for seg in ann["segments_info"]}
        assert png_segment_ids == json_segment_ids


def _assert_coco_equal(result: dict, golden: dict) -> None:
    assert result["info"] == golden["info"]
    assert result["licenses"] == golden["licenses"]

    assert len(result["categories"]) == len(golden["categories"]), (
        f"categories count: {len(result['categories'])} vs {len(golden['categories'])}"
    )
    for r, g in zip(result["categories"], golden["categories"]):
        assert r == g, f"category mismatch: {r} vs {g}"

    assert len(result["images"]) == len(golden["images"]), (
        f"images count: {len(result['images'])} vs {len(golden['images'])}"
    )
    for r, g in zip(result["images"], golden["images"]):
        assert r["id"] == g["id"]
        assert r["file_name"] == g["file_name"]
        assert r["width"] == g["width"]
        assert r["height"] == g["height"]

    assert len(result["annotations"]) == len(golden["annotations"]), (
        f"annotation count: {len(result['annotations'])} vs {len(golden['annotations'])}"
    )
    for r, g in zip(result["annotations"], golden["annotations"]):
        assert r["id"] == g["id"]
        assert r["image_id"] == g["image_id"]
        assert r["category_id"] == g["category_id"]
        assert r["iscrowd"] == g["iscrowd"]
        assert len(r["bbox"]) == 4
        for rv, gv in zip(r["bbox"], g["bbox"]):
            assert abs(rv - gv) <= 1e-3, f"bbox mismatch: {r['bbox']} vs {g['bbox']}"
        assert abs(r["area"] - g["area"]) <= 1e-3, (
            f"area mismatch: {r['area']} vs {g['area']}"
        )


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kitti_out(tmp_path_factory, rendered_scene_vars):
    required = {
        "LdrColor",
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox3D",
        "CameraParams",
    }
    missing = sorted(required - set(rendered_scene_vars))
    if missing:
        pytest.skip(f"Rendered scene missing KITTI vars: {missing}")
    tmp = tmp_path_factory.mktemp("kitti")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp), overwrite=True))
    writer = KITTIWriter()
    render_vars = {
        "LdrColor": rendered_scene_vars["LdrColor"],
        "SemanticBoundingBox2DTight": rendered_scene_vars["SemanticBoundingBox2DTight"],
        "SemanticBoundingBox3D": rendered_scene_vars["SemanticBoundingBox3D"],
        "CameraParams": rendered_scene_vars["CameraParams"],
    }
    if "SemanticIdMap" in rendered_scene_vars:
        render_vars["SemanticIdMap"] = rendered_scene_vars["SemanticIdMap"]
    writer.write_frame(frame_id=0, render_vars=render_vars, backend=backend)
    writer.finalize(backend)
    return tmp


@pytest.fixture(scope="module")
def yolo_out(tmp_path_factory, rendered_scene_vars):
    writer = YOLOWriter()
    missing = sorted(
        v for v in writer.required_render_vars if v not in rendered_scene_vars
    )
    if missing:
        pytest.skip(f"Rendered scene missing YOLO vars: {missing}")
    tmp = tmp_path_factory.mktemp("yolo_golden")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp), overwrite=True))
    writer.write_frame(0, rendered_scene_vars, backend)
    writer.finalize(backend)
    return tmp


@pytest.fixture(scope="module")
def bop_out(tmp_path_factory, rendered_scene_vars):
    writer = BOPWriter()
    missing = sorted(
        v for v in writer.required_render_vars if v not in rendered_scene_vars
    )
    if missing:
        pytest.skip(f"Rendered scene missing BOP vars: {missing}")
    tmp = tmp_path_factory.mktemp("bop_golden")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp), overwrite=True))
    writer.write_frame(0, rendered_scene_vars, backend)
    writer.finalize(backend)
    return tmp


@pytest.fixture(scope="module")
def cosmos_out(tmp_path_factory, rendered_scene_vars):
    # Default config: full output resolution (so the encoded videos are
    # substantial) and GPU-accelerated Canny. Goldens are GPU-rendered on this
    # host already, so GPU Canny stays deterministic for the comparison.
    writer = CosmosTransferWriter(CosmosTransferWriterConfig())
    missing = sorted(
        v for v in writer.required_render_vars if v not in rendered_scene_vars
    )
    if missing:
        pytest.skip(f"Rendered scene missing Cosmos vars: {missing}")
    tmp = tmp_path_factory.mktemp("cosmos_golden")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp), overwrite=True))
    writer.write_frame(0, rendered_scene_vars, backend)

    # Capture the frames fed into the encoders before finalize() frees them.
    # The encoded MP4 bytes are non-deterministic, but these frames are not, so
    # they provide real content coverage for every conditioning signal.
    frames = {
        "rgb": writer._rgb_frames[0].copy(),
        "canny": writer._compute_canny_frames()[0],
        "depth": writer._colorize_depth_frames()[0],
        "segmentation": writer._colorize_seg_frames()[0],
    }
    try:
        writer.finalize(backend)
    except ImportError as exc:
        pytest.skip(f"video backend unavailable: {exc}")
    return SimpleNamespace(dir=tmp, frames=frames)


def _fake_encode_video(frames, fps):  # noqa: ANN001 - test stub signature
    """Stand-in for NVENC encoding so the golden test needs no GPU codec.

    The LeRobot golden asserts on deterministic metadata (parquet + meta JSON),
    not on the non-deterministic encoded video bytes, so a minimal fake MP4
    payload is sufficient here.
    """
    return b"\x00\x00\x00\x1cftypisom" + b"\x00" * 64


@pytest.fixture(scope="module")
def lerobot_out(tmp_path_factory, rendered_scene_vars):
    pytest.importorskip("pyarrow")
    if "LdrColor" not in rendered_scene_vars:
        pytest.skip("Rendered scene missing LeRobot var: LdrColor")
    tmp = tmp_path_factory.mktemp("lerobot_golden")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp), overwrite=True))
    writer = LeRobotWriter(
        LeRobotWriterConfig(
            fps=30,
            episode_index=0,
            task_description="ovannotators golden scene",
            camera_keys=["top"],
            frames_per_chunk=1000,
        )
    )
    writer.write_frame(
        frame_id=0,
        render_vars={"LdrColor": rendered_scene_vars["LdrColor"]},
        backend=backend,
    )
    # The frame fed to the (GPU, non-deterministic) video encoder. Captured for
    # content coverage; the encoded MP4 itself is faked below.
    video_frame = writer._frame_images["top"][0].copy()
    with patch("ovannotators.writers.lerobot._encode_video", _fake_encode_video):
        writer.finalize(backend)
    return SimpleNamespace(dir=tmp, video_frame=video_frame)


class TestKITTIGolden:
    def test_image_file(self, kitti_out) -> None:
        result_path = kitti_out / "image_2" / "000000.png"
        golden_path = KITTI_EXPECTED / "image_2" / "000000.png"

        if UPDATE_GOLDEN:
            write_golden_image(result_path, golden_path)
            return

        # PNG is lossless, but the path tracer is stochastic: run-to-run
        # noise has a ~1.3-2.7 MAE floor on RGB, so use a tolerance well above
        # the noise while still catching gross regressions (wrong content
        # scores MAE in the tens-to-hundreds).
        assert_image_metrics(
            result_path,
            golden_path,
            ImageMetricThresholds(mae_max=5.0, rmse_max=10.0, psnr_min=28.0),
        )

    def test_label_file(self, kitti_out) -> None:
        result_path = kitti_out / "label_2" / "000000.txt"
        result_text = result_path.read_text()

        golden_path = KITTI_EXPECTED / "label_2" / "000000.txt"

        if UPDATE_GOLDEN:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(result_text)
            return

        assert golden_path.exists(), (
            f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
        )
        _assert_kitti_labels_equal(result_text, golden_path.read_text())

    def test_calib_file(self, kitti_out) -> None:
        result_path = kitti_out / "calib" / "000000.txt"
        result_text = result_path.read_text()

        golden_path = KITTI_EXPECTED / "calib" / "000000.txt"

        if UPDATE_GOLDEN:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(result_text)
            return

        assert golden_path.exists(), (
            f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
        )
        assert result_text == golden_path.read_text(), "calib file mismatch"


class TestYOLOGolden:
    def test_data_yaml(self, yolo_out: Path) -> None:
        result = (yolo_out / "data.yaml").read_text()
        golden = YOLO_EXPECTED / "data.yaml"
        if UPDATE_GOLDEN:
            golden.parent.mkdir(parents=True, exist_ok=True)
            golden.write_text(result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        assert result == golden.read_text()

    def test_labels_txt(self, yolo_out: Path) -> None:
        result = (yolo_out / "labels" / "val" / "000000.txt").read_text()
        golden = YOLO_EXPECTED / "labels" / "val" / "000000.txt"
        if UPDATE_GOLDEN:
            golden.parent.mkdir(parents=True, exist_ok=True)
            golden.write_text(result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        assert result == golden.read_text()

    def test_rendered_image(self, yolo_out: Path) -> None:
        result_path = yolo_out / "images" / "val" / "000000.jpg"
        golden = YOLO_EXPECTED / "images" / "val" / "000000.jpg"
        if UPDATE_GOLDEN:
            write_golden_image(result_path, golden)
            return
        assert_image_metrics(
            result_path,
            golden,
            ImageMetricThresholds(mae_max=2.0, rmse_max=4.0, psnr_min=35.0),
        )


class TestBOPGolden:
    def test_scene_camera(self, bop_out: Path) -> None:
        result = _read_json(bop_out / "000000" / "scene_camera.json")
        golden = BOP_EXPECTED / "scene_camera.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_scene_gt(self, bop_out: Path) -> None:
        result = _read_json(bop_out / "000000" / "scene_gt.json")
        golden = BOP_EXPECTED / "scene_gt.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_scene_gt_info(self, bop_out: Path) -> None:
        result = _read_json(bop_out / "000000" / "scene_gt_info.json")
        golden = BOP_EXPECTED / "scene_gt_info.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_rgb_image(self, bop_out: Path) -> None:
        result_path = bop_out / "000000" / "rgb" / "000000.png"
        golden = BOP_EXPECTED / "rgb_000000.png"
        if UPDATE_GOLDEN:
            write_golden_image(result_path, golden)
            return
        # Path-tracer RGB has a ~1.3-2.7 MAE run-to-run noise floor; tolerate
        # that while still catching gross regressions.
        assert_image_metrics(
            result_path,
            golden,
            ImageMetricThresholds(mae_max=5.0, rmse_max=10.0, psnr_min=28.0),
        )

    def test_depth_image(self, bop_out: Path) -> None:
        result_path = bop_out / "000000" / "depth" / "000000.png"
        golden = BOP_EXPECTED / "depth_000000.png"
        if UPDATE_GOLDEN:
            write_golden_image(result_path, golden)
            return
        assert_image_metrics(
            result_path,
            golden,
            ImageMetricThresholds(mae_max=1.5, rmse_max=3.0, psnr_min=40.0),
        )


class TestCosmosGolden:
    def test_manifest_json(self, cosmos_out) -> None:
        result = _read_json(cosmos_out.dir / "manifest.json")
        golden = COSMOS_EXPECTED / "manifest.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_thumbnail_image(self, cosmos_out) -> None:
        result_path = cosmos_out.dir / "thumbnails" / "000000.jpg"
        golden = COSMOS_EXPECTED / "thumbnails" / "000000.jpg"
        if UPDATE_GOLDEN:
            write_golden_image(result_path, golden)
            return
        assert_image_metrics(
            result_path,
            golden,
            ImageMetricThresholds(mae_max=2.0, rmse_max=4.0, psnr_min=35.0),
        )

    def test_rgb_video_frame(self, cosmos_out) -> None:
        """Content of the RGB video (the frame fed to the encoder)."""
        _assert_frame_matches_golden(
            cosmos_out.frames["rgb"],
            COSMOS_EXPECTED / "frames" / "rgb.png",
            ImageMetricThresholds(mae_max=3.0, rmse_max=6.0, psnr_min=32.0),
        )

    def test_canny_conditioning_frame(self, cosmos_out) -> None:
        """Content of the canny-edge conditioning video."""
        _assert_frame_matches_golden(
            cosmos_out.frames["canny"],
            COSMOS_EXPECTED / "frames" / "canny.png",
            # Edge maps are binary; small render variance can flip a few edge
            # pixels, so tolerate moderate drift while still catching gross
            # regressions (wrong content scores MAE in the tens-to-hundreds).
            ImageMetricThresholds(mae_max=10.0, rmse_max=50.0, psnr_min=14.0),
        )

    def test_depth_conditioning_frame(self, cosmos_out) -> None:
        """Content of the depth-colorized conditioning video."""
        _assert_frame_matches_golden(
            cosmos_out.frames["depth"],
            COSMOS_EXPECTED / "frames" / "depth.png",
            ImageMetricThresholds(mae_max=5.0, rmse_max=12.0, psnr_min=24.0),
        )

    def test_segmentation_conditioning_frame(self, cosmos_out) -> None:
        """Content of the segmentation-colorized conditioning video."""
        _assert_frame_matches_golden(
            cosmos_out.frames["segmentation"],
            COSMOS_EXPECTED / "frames" / "segmentation.png",
            # Segmentation IDs are geometric (deterministic across renders).
            ImageMetricThresholds(mae_max=2.0, rmse_max=6.0, psnr_min=30.0),
        )

    def test_conditioning_media_exists(self, cosmos_out) -> None:
        manifest = _read_json(cosmos_out.dir / "manifest.json")
        clip = manifest["clips"][0]
        assert "canny" in clip["conditioning"]
        expected_paths = [clip["rgb_video"], *clip["conditioning"].values()]
        for rel_path in expected_paths:
            path = cosmos_out.dir / rel_path
            assert path.exists(), f"manifest path missing: {rel_path}"
            if path.suffix == ".mp4":
                probe = _h264_probe(path)
                assert probe["size_bytes"] > 128
                # In no-remux environments, PyNvVideoCodec can emit Annex-B
                # elementary streams instead of MP4 container bytes.
                assert probe["has_ftyp_box"] or probe["has_annexb_startcode"]


class TestLeRobotGolden:
    def test_info_json(self, lerobot_out) -> None:
        result = _read_json(lerobot_out.dir / "meta" / "info.json")
        golden = LEROBOT_EXPECTED / "info.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_tasks_json(self, lerobot_out) -> None:
        result = _read_json(lerobot_out.dir / "meta" / "tasks.json")
        golden = LEROBOT_EXPECTED / "tasks.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_episodes_json(self, lerobot_out) -> None:
        result = _read_json(lerobot_out.dir / "meta" / "episodes.json")
        golden = LEROBOT_EXPECTED / "episodes.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_stats_json(self, lerobot_out) -> None:
        result = _read_json(lerobot_out.dir / "meta" / "stats.json")
        golden = LEROBOT_EXPECTED / "stats.json"
        if UPDATE_GOLDEN:
            _write_json(golden, result)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(result, _read_json(golden))

    def test_parquet_schema_and_rows(self, lerobot_out) -> None:
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        parquet_path = lerobot_out.dir / "data" / "chunk-000" / "episode_000000.parquet"
        assert parquet_path.exists()
        table = pq.read_table(parquet_path)
        assert set(table.column_names) == {
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
        }
        assert len(table) == 1
        assert table["frame_index"].to_pylist() == [0]
        assert table["episode_index"].to_pylist() == [0]
        assert table["timestamp"].to_pylist() == [0.0]

    def test_parquet_rows_match_golden(self, lerobot_out) -> None:
        """Golden-compare the full decoded parquet contents (not just schema).

        Parquet container bytes are not stable across pyarrow versions, so the
        golden is the decoded row payload rather than the binary file.
        """
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        parquet_path = lerobot_out.dir / "data" / "chunk-000" / "episode_000000.parquet"
        rows = pq.read_table(parquet_path).to_pylist()
        golden = LEROBOT_EXPECTED / "parquet_rows.json"
        if UPDATE_GOLDEN:
            _write_json(golden, rows)
            return
        assert golden.exists(), f"Golden missing: {golden}. Run UPDATE_GOLDEN=1 pytest."
        _assert_json_close(rows, _read_json(golden))

    def test_video_frame_matches_golden(self, lerobot_out) -> None:
        """Content of the observation video (the frame fed to the encoder)."""
        _assert_frame_matches_golden(
            lerobot_out.video_frame,
            LEROBOT_EXPECTED / "video_frame.png",
            ImageMetricThresholds(mae_max=3.0, rmse_max=6.0, psnr_min=32.0),
        )

    def test_video_file_emitted(self, lerobot_out) -> None:
        video = (
            lerobot_out.dir
            / "videos"
            / "chunk-000"
            / "observation.images.top"
            / "episode_000000.mp4"
        )
        assert video.exists()
        assert video.stat().st_size > 0


def _assert_kitti_labels_equal(result: str, golden: str, tol: float = 1e-3) -> None:
    def _split_label_and_numeric_fields(line: str) -> tuple[str, list[float]]:
        parts = line.split()
        first_numeric_idx = None
        for idx, part in enumerate(parts):
            try:
                float(part)
                first_numeric_idx = idx
                break
            except ValueError:
                continue
        assert first_numeric_idx is not None, (
            f"no numeric fields in KITTI line: {line!r}"
        )
        label = " ".join(parts[:first_numeric_idx])
        numeric = [float(v) for v in parts[first_numeric_idx:]]
        return label, numeric

    r_lines = [line for line in result.splitlines() if line.strip()]
    g_lines = [line for line in golden.splitlines() if line.strip()]

    assert len(r_lines) == len(g_lines), (
        f"label line count: {len(r_lines)} vs {len(g_lines)}"
    )

    for i, (rl, gl) in enumerate(zip(r_lines, g_lines)):
        r_label, r_values = _split_label_and_numeric_fields(rl)
        g_label, g_values = _split_label_and_numeric_fields(gl)
        assert r_label == g_label, f"line {i}: label {r_label!r} vs {g_label!r}"
        assert len(r_values) == len(g_values), (
            f"line {i}: numeric field count {len(r_values)} vs {len(g_values)}"
        )
        for j, (rf, gf) in enumerate(zip(r_values, g_values), start=1):
            assert abs(rf - gf) <= tol, f"line {i} field {j}: {rf} vs {gf}"
