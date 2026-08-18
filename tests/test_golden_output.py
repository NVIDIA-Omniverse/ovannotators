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

"""Golden output tests — verify each writer produces expected file structure and content."""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np
import pytest

from ovannotators._internal.codecs import read_exr
from ovannotators._internal.optional_deps import ovrtx_available
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers._helpers import (
    OVRTX_BBOX2D_DTYPE,
    OVRTX_BBOX3D_DTYPE,
    OVRTX_CAMERA_PARAMS_DTYPE,
)
from ovannotators.writers.basic import BasicWriter, BasicWriterConfig
from ovannotators.writers.bop import BOPWriter, BOPWriterConfig
from ovannotators.writers.coco import COCOWriter, COCOWriterConfig
from ovannotators.writers.cosmos_transfer import (
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
)
from ovannotators.writers.kitti import KITTIWriter, KITTIWriterConfig
from ovannotators.writers.lerobot import LeRobotWriter, LeRobotWriterConfig
from ovannotators.writers.yolo import YOLOWriter, YOLOWriterConfig


import sys

sys.path.insert(0, str(Path(__file__).parent))

NUM_FRAMES = 5
SEED = 42
H, W = 256, 512
OBJECT_LABELS = ["cone", "cube", "cube", "sphere", "plane"]
NUM_OBJECTS = len(OBJECT_LABELS)  # 5


def _make_fast_render_vars(rng: np.random.Generator, frame_id: int) -> dict:
    """Fast synthetic render vars (no raytracing) matching the 5-object scene schema."""
    # Bright synthetic image (not random noise — use a gradient + colored blocks)
    ldr = np.full((H, W, 3), 180, dtype=np.uint8)  # light gray base
    # Paint colored rectangles for each object region
    ldr[30:90, 30:120] = [220, 40, 30]  # cone region
    ldr[30:90, 150:260] = [30, 200, 40]  # cube1 region
    ldr[30:90, 300:410] = [40, 80, 220]  # cube2 region
    ldr[100:180, 150:280] = [240, 210, 30]  # sphere region
    ldr[200:256, 0:512] = [100, 100, 100]  # plane (ground)
    # Add slight per-frame variation
    ldr = np.clip(ldr.astype(np.int16) + frame_id * 3, 0, 255).astype(np.uint8)

    depth = np.full((H, W), 10.0, dtype=np.float32)
    depth[30:90, 30:120] = 8.0
    depth[30:90, 150:260] = 7.5
    depth[30:90, 300:410] = 9.0
    depth[100:180, 150:280] = 7.0
    depth[200:256, 0:512] = 12.0

    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[..., 1] = 1.0  # default up
    normals[30:180, 30:410, :] = [0, 0, -1]  # facing camera

    inst_seg = np.zeros((H, W), dtype=np.uint32)
    inst_seg[30:90, 30:120] = 1
    inst_seg[30:90, 150:260] = 2
    inst_seg[30:90, 300:410] = 3
    inst_seg[100:180, 150:280] = 4
    inst_seg[200:256, 0:512] = 5

    # Structured AOVs must use real ovrtx payload dtypes.
    bbox2d = np.zeros(NUM_OBJECTS, dtype=OVRTX_BBOX2D_DTYPE)
    bbox2d_specs = [
        # x_min, y_min, x_max, y_max, semantic_id, instance_id
        (30, 30, 120, 90, 1, 1),
        (150, 30, 260, 90, 2, 2),
        (300, 30, 410, 90, 2, 3),
        (150, 100, 280, 180, 3, 4),
        (0, 200, 512, 256, 4, 5),
    ]
    for i, (x0, y0, x1, y1, sem_id, inst_id) in enumerate(bbox2d_specs):
        bbox2d[i]["instance_id"] = (inst_id, 0, 0, 0)
        bbox2d[i]["semantic_id"] = (sem_id, 0, 0, 0)
        bbox2d[i]["x_min"] = x0
        bbox2d[i]["y_min"] = y0
        bbox2d[i]["x_max"] = x1
        bbox2d[i]["y_max"] = y1

    bbox3d = np.zeros(NUM_OBJECTS, dtype=OVRTX_BBOX3D_DTYPE)
    bbox3d_specs = [
        # semantic_id, instance_id, min_xyz, max_xyz, center
        (1, 1, (-0.4, -0.5, -0.4), (0.4, 0.5, 0.4), (-1.5, 0.5, 0.0)),
        (2, 2, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), (0.0, 0.5, 0.0)),
        (2, 3, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), (1.5, 0.5, 0.0)),
        (3, 4, (-0.55, -0.55, -0.55), (0.55, 0.55, 0.55), (0.0, 0.75, -1.0)),
        (4, 5, (-5.0, -0.005, -5.0), (5.0, 0.005, 5.0), (0.0, 0.0, 0.0)),
    ]
    for i, (sem_id, inst_id, lo, hi, center) in enumerate(bbox3d_specs):
        bbox3d[i]["instance_id"] = (inst_id, 0, 0, 0)
        bbox3d[i]["semantic_id"] = (sem_id, 0, 0, 0)
        bbox3d[i]["x_min"], bbox3d[i]["y_min"], bbox3d[i]["z_min"] = lo
        bbox3d[i]["x_max"], bbox3d[i]["y_max"], bbox3d[i]["z_max"] = hi
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = np.asarray(center, dtype=np.float32)
        bbox3d[i]["transform"] = transform

    cam = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
    cam[0]["view_transform"] = np.eye(4, dtype=np.float32)
    cam[0]["projection"] = np.eye(4, dtype=np.float32)
    cam[0]["meters_per_scene_unit"] = 1.0
    cam[0]["resolution"] = (W, H)
    cam[0]["clipping_plane"] = (0.1, 100.0)
    cam[0]["projection_model"] = b"pinhole"
    cam[0]["focal_length"] = 24.0
    cam[0]["fov_x"] = np.radians(60.0).astype(np.float32)

    return {
        "LdrColor": ldr,
        "DistanceToCameraSD": depth,
        "NormalSD": normals,
        "SemanticInstanceSegmentation": inst_seg,
        "SemanticSegmentation": inst_seg.copy(),
        "SemanticBoundingBox2DTight": bbox2d,
        "SemanticBoundingBox3D": bbox3d,
        "CameraParams": cam,
    }


def _run_writer(writer, tmp_path: Path) -> Path:
    """Run a writer with fast synthetic data (no raytracing)."""
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
    rng = np.random.default_rng(SEED)

    for frame_id in range(NUM_FRAMES):
        render_vars = _make_fast_render_vars(rng, frame_id)
        writer.write_frame(frame_id, render_vars, backend)

    writer.finalize(backend)
    return tmp_path


# ---------------------------------------------------------------------------
# BasicWriter
# ---------------------------------------------------------------------------


class TestBasicWriterGolden:
    def test_output_structure(self, tmp_path: Path) -> None:
        writer = BasicWriter(BasicWriterConfig(colorize=True, image_format="png"))
        out = _run_writer(writer, tmp_path)

        for fid in range(NUM_FRAMES):
            frame_dir = out / f"{fid:04d}"
            assert frame_dir.is_dir()
            assert (frame_dir / "ldr_color.png").exists()
            assert (frame_dir / "distance_to_camera.exr").exists()
            assert (frame_dir / "distance_to_camera_colorized.png").exists()
            assert (frame_dir / "normal.exr").exists()
            assert (frame_dir / "normal_colorized.png").exists()
            assert (frame_dir / "instance_segmentation.npy").exists()
            assert (frame_dir / "instance_segmentation_colorized.png").exists()
            assert (frame_dir / "bbox_2d_tight.json").exists()
            assert (frame_dir / "bbox_3d.json").exists()
            assert (frame_dir / "camera_params.json").exists()

    def test_depth_values(self, tmp_path: Path) -> None:
        writer = BasicWriter(BasicWriterConfig(colorize=True))
        out = _run_writer(writer, tmp_path)

        # Depth is now stored as a single-channel half-float EXR.
        depth = read_exr(str(out / "0000" / "distance_to_camera.exr"))
        depth = np.squeeze(depth, axis=-1)
        assert depth.shape == (H, W)
        assert depth.dtype == np.float16
        # Raytraced depth: 0 for misses, positive for hits
        assert depth.min() >= 0.0
        assert depth.max() <= 100.0

    def test_bbox_json_content(self, tmp_path: Path) -> None:
        writer = BasicWriter(BasicWriterConfig(colorize=True))
        out = _run_writer(writer, tmp_path)

        with open(out / "0000" / "bbox_2d_tight.json") as f:
            data = json.load(f)
        assert len(data) == NUM_OBJECTS
        semantic_ids = [entry["semantic_id"] for entry in data]
        assert semantic_ids == [1, 2, 2, 3, 4]

    def test_image_dimensions(self, tmp_path: Path) -> None:
        from PIL import Image

        writer = BasicWriter(BasicWriterConfig(colorize=True, image_format="png"))
        out = _run_writer(writer, tmp_path)

        img = Image.open(out / "0000" / "ldr_color.png")
        assert img.size == (W, H)

    def test_colorized_images_exist(self, tmp_path: Path) -> None:
        from PIL import Image

        writer = BasicWriter(BasicWriterConfig(colorize=True))
        out = _run_writer(writer, tmp_path)

        depth_viz = Image.open(out / "0000" / "distance_to_camera_colorized.png")
        assert depth_viz.size == (W, H)
        assert depth_viz.mode == "RGB"

    def test_no_colorize(self, tmp_path: Path) -> None:
        writer = BasicWriter(BasicWriterConfig(colorize=False))
        out = _run_writer(writer, tmp_path)
        assert not (out / "0000" / "distance_to_camera_colorized.png").exists()
        assert (out / "0000" / "distance_to_camera.exr").exists()


# ---------------------------------------------------------------------------
# COCOWriter
# ---------------------------------------------------------------------------


class TestCOCOWriterGolden:
    def test_output_structure(self, tmp_path: Path) -> None:
        writer = COCOWriter(COCOWriterConfig(split="train"))
        out = _run_writer(writer, tmp_path)

        assert (out / "annotations" / "instances_train.json").exists()
        for fid in range(NUM_FRAMES):
            assert (out / "images" / f"{fid:06d}.jpg").exists()

    def test_annotations_content(self, tmp_path: Path) -> None:
        writer = COCOWriter(COCOWriterConfig(split="train"))
        out = _run_writer(writer, tmp_path)

        with open(out / "annotations" / "instances_train.json") as f:
            coco = json.load(f)

        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == NUM_FRAMES
        # 4 bboxes per frame
        assert len(coco["annotations"]) == NUM_FRAMES * NUM_OBJECTS

    def test_annotation_bbox_values(self, tmp_path: Path) -> None:
        writer = COCOWriter(COCOWriterConfig(split="train"))
        out = _run_writer(writer, tmp_path)

        with open(out / "annotations" / "instances_train.json") as f:
            coco = json.load(f)

        # First annotation should correspond to "red_cube" bbox
        ann = coco["annotations"][0]
        assert "bbox" in ann
        assert ann["bbox"][0] >= 0  # x
        assert ann["bbox"][1] >= 0  # y
        assert ann["bbox"][2] > 0  # width
        assert ann["bbox"][3] > 0  # height

    def test_png_format(self, tmp_path: Path) -> None:
        writer = COCOWriter(COCOWriterConfig(image_format="png"))
        out = _run_writer(writer, tmp_path)
        assert (out / "images" / "000000.png").exists()


# ---------------------------------------------------------------------------
# KITTIWriter
# ---------------------------------------------------------------------------


class TestKITTIWriterGolden:
    def test_output_structure(self, tmp_path: Path) -> None:
        writer = KITTIWriter(KITTIWriterConfig(write_depth=True))
        out = _run_writer(writer, tmp_path)

        for fid in range(NUM_FRAMES):
            assert (out / "image_2" / f"{fid:06d}.png").exists()
            assert (out / "label_2" / f"{fid:06d}.txt").exists()
            assert (out / "calib" / f"{fid:06d}.txt").exists()
            assert (out / "depth" / f"{fid:06d}.png").exists()

    def test_label_format(self, tmp_path: Path) -> None:
        writer = KITTIWriter(KITTIWriterConfig())
        out = _run_writer(writer, tmp_path)

        label_file = out / "label_2" / "000000.txt"
        lines = label_file.read_text().strip().split("\n")
        assert len(lines) == NUM_OBJECTS  # 4 objects

        # Each line: type truncated occluded alpha left top right bottom h w l x y z ry
        parts = lines[0].split()
        assert len(parts) == 15
        assert parts[0]

    def test_calib_format(self, tmp_path: Path) -> None:
        writer = KITTIWriter(KITTIWriterConfig())
        out = _run_writer(writer, tmp_path)

        calib = (out / "calib" / "000000.txt").read_text()
        assert "P2:" in calib

    def test_class_filter(self, tmp_path: Path) -> None:
        writer = KITTIWriter(KITTIWriterConfig(classes=["cube"]))
        out = _run_writer(writer, tmp_path)

        label_file = out / "label_2" / "000000.txt"
        lines = label_file.read_text().strip().split("\n")
        assert all(line.startswith("cube") for line in lines if line.strip())


# ---------------------------------------------------------------------------
# YOLOWriter
# ---------------------------------------------------------------------------


class TestYOLOWriterGolden:
    def test_output_structure(self, tmp_path: Path) -> None:
        writer = YOLOWriter(YOLOWriterConfig(task="detect"))
        out = _run_writer(writer, tmp_path)

        assert (out / "data.yaml").exists()
        # At least some images/labels in train and/or val
        train_imgs = (
            list((out / "images" / "train").glob("*"))
            if (out / "images" / "train").exists()
            else []
        )
        val_imgs = (
            list((out / "images" / "val").glob("*"))
            if (out / "images" / "val").exists()
            else []
        )
        assert len(train_imgs) + len(val_imgs) == NUM_FRAMES

    def test_data_yaml(self, tmp_path: Path) -> None:
        import yaml

        writer = YOLOWriter(YOLOWriterConfig(task="detect"))
        out = _run_writer(writer, tmp_path)

        with open(out / "data.yaml") as f:
            data = yaml.safe_load(f)

        assert "names" in data
        assert "nc" in data
        assert data["nc"] == len(data["names"])
        # Should discover 3 classes
        # 'cube' appears twice but YOLO deduplicates → 4 unique classes
        assert data["nc"] == 4

    def test_label_format(self, tmp_path: Path) -> None:
        writer = YOLOWriter(YOLOWriterConfig(task="detect"))
        out = _run_writer(writer, tmp_path)

        # Find first label file
        label_files = list(out.rglob("labels/**/*.txt"))
        assert len(label_files) > 0

        lines = label_files[0].read_text().strip().split("\n")
        for line in lines:
            parts = line.split()
            assert len(parts) == 5  # class_id cx cy w h
            cx, cy, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0


# ---------------------------------------------------------------------------
# BOPWriter
# ---------------------------------------------------------------------------


class TestBOPWriterGolden:
    def test_output_structure(self, tmp_path: Path) -> None:
        writer = BOPWriter(BOPWriterConfig(scene_id=1))
        out = _run_writer(writer, tmp_path)

        scene_dir = out / "000001"
        assert scene_dir.is_dir()
        assert (scene_dir / "scene_camera.json").exists()
        assert (scene_dir / "scene_gt.json").exists()
        assert (scene_dir / "scene_gt_info.json").exists()

        for fid in range(NUM_FRAMES):
            assert (scene_dir / "rgb" / f"{fid:06d}.png").exists()
            assert (scene_dir / "depth" / f"{fid:06d}.png").exists()

    def test_scene_gt_structure(self, tmp_path: Path) -> None:
        writer = BOPWriter(BOPWriterConfig(scene_id=1))
        out = _run_writer(writer, tmp_path)

        with open(out / "000001" / "scene_gt.json") as f:
            gt = json.load(f)

        # Keys are frame indices as strings
        assert "0" in gt
        assert len(gt) == NUM_FRAMES

        # Each frame has list of objects
        frame0 = gt["0"]
        assert len(frame0) == NUM_OBJECTS  # 4 objects
        assert "obj_id" in frame0[0]
        assert "cam_R_m2c" in frame0[0]
        assert "cam_t_m2c" in frame0[0]

    def test_scene_camera_structure(self, tmp_path: Path) -> None:
        writer = BOPWriter(BOPWriterConfig(scene_id=1))
        out = _run_writer(writer, tmp_path)

        with open(out / "000001" / "scene_camera.json") as f:
            cam = json.load(f)

        assert "0" in cam
        assert "cam_K" in cam["0"]
        assert "depth_scale" in cam["0"]
        assert len(cam["0"]["cam_K"]) == 9

    def test_masks_exist(self, tmp_path: Path) -> None:
        writer = BOPWriter(BOPWriterConfig(scene_id=1))
        out = _run_writer(writer, tmp_path)

        mask_dir = out / "000001" / "mask"
        mask_visib_dir = out / "000001" / "mask_visib"
        # 4 objects * 5 frames = 20 masks each
        assert len(list(mask_dir.glob("*.png"))) == NUM_OBJECTS * NUM_FRAMES
        assert len(list(mask_visib_dir.glob("*.png"))) == NUM_OBJECTS * NUM_FRAMES


# ---------------------------------------------------------------------------
# LeRobotWriter
# ---------------------------------------------------------------------------


try:
    import PyNvVideoCodec  # noqa: F401

    _has_nvvideocodec = True
except (ImportError, OSError):
    _has_nvvideocodec = False


@pytest.mark.skipif(not _has_nvvideocodec, reason="PyNvVideoCodec not available")
class TestLeRobotWriterGolden:
    def test_output_structure(self, tmp_path: Path) -> None:
        writer = LeRobotWriter(
            LeRobotWriterConfig(
                fps=30,
                episode_index=0,
                task_description="golden test",
                camera_keys=["front"],
            )
        )
        out = _run_writer(writer, tmp_path)

        assert (out / "meta" / "info.json").exists()
        assert (out / "meta" / "tasks.json").exists()
        assert (out / "meta" / "episodes.json").exists()
        assert (out / "meta" / "stats.json").exists()
        assert (out / "data" / "chunk-000" / "episode_000000.parquet").exists()

    def test_meta_info(self, tmp_path: Path) -> None:
        writer = LeRobotWriter(
            LeRobotWriterConfig(
                fps=30,
                episode_index=0,
                task_description="golden test",
                camera_keys=["front"],
            )
        )
        out = _run_writer(writer, tmp_path)

        with open(out / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["fps"] == 30
        assert info["total_frames"] == NUM_FRAMES

    def test_video_produced(self, tmp_path: Path) -> None:
        writer = LeRobotWriter(
            LeRobotWriterConfig(
                fps=30,
                episode_index=0,
                task_description="golden test",
                camera_keys=["front"],
            )
        )
        out = _run_writer(writer, tmp_path)

        video = (
            out
            / "videos"
            / "chunk-000"
            / "observation.images.front"
            / "episode_000000.mp4"
        )
        assert video.exists()
        assert video.stat().st_size > 0

    def test_parquet_content(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        writer = LeRobotWriter(
            LeRobotWriterConfig(
                fps=30,
                episode_index=0,
                task_description="golden test",
                camera_keys=["front"],
            )
        )
        out = _run_writer(writer, tmp_path)

        table = pq.read_table(out / "data" / "chunk-000" / "episode_000000.parquet")
        assert len(table) == NUM_FRAMES
        assert "frame_index" in table.column_names
        assert "episode_index" in table.column_names
        assert "timestamp" in table.column_names


# ---------------------------------------------------------------------------
# CosmosTransferWriter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_nvvideocodec, reason="PyNvVideoCodec not available")
class TestCosmosTransferWriterGolden:
    def test_output_structure(self, tmp_path: Path) -> None:
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(
                fps=24,
                conditioning_types=["canny", "depth", "segmentation"],
            )
        )
        out = _run_writer(writer, tmp_path)

        assert (out / "manifest.json").exists()
        assert (out / "videos" / "000000_rgb.mp4").exists()
        assert (out / "conditioning" / "000000_canny.mp4").exists()
        assert (out / "conditioning" / "000000_depth.mp4").exists()
        assert (out / "conditioning" / "000000_segmentation.mp4").exists()
        assert (out / "thumbnails" / "000000.jpg").exists()

    def test_manifest_content(self, tmp_path: Path) -> None:
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(
                fps=24,
                conditioning_types=["canny", "depth", "segmentation"],
            )
        )
        out = _run_writer(writer, tmp_path)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        assert "clips" in manifest
        assert len(manifest["clips"]) == 1
        clip = manifest["clips"][0]
        assert "prompt" not in clip
        assert clip["fps"] == 24
        assert clip["num_frames"] == NUM_FRAMES
        assert "conditioning" in clip
        assert set(clip["conditioning"].keys()) == {"canny", "depth", "segmentation"}

    def test_videos_nonzero_size(self, tmp_path: Path) -> None:
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(
                fps=24,
                conditioning_types=["canny", "depth"],
            )
        )
        out = _run_writer(writer, tmp_path)

        for vid in (out / "videos").glob("*.mp4"):
            assert vid.stat().st_size > 100

        for vid in (out / "conditioning").glob("*.mp4"):
            assert vid.stat().st_size > 100

    def test_subset_conditioning(self, tmp_path: Path) -> None:
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(
                fps=24,
                conditioning_types=["depth"],
            )
        )
        out = _run_writer(writer, tmp_path)

        assert (out / "conditioning" / "000000_depth.mp4").exists()
        assert not (out / "conditioning" / "000000_canny.mp4").exists()
        assert not (out / "conditioning" / "000000_segmentation.mp4").exists()


# ---------------------------------------------------------------------------
# ovrtx integration test (requires GPU + ovrtx; skipped in CI)
# ---------------------------------------------------------------------------

_has_ovrtx = ovrtx_available()


@pytest.mark.skipif(not _has_ovrtx, reason="ovrtx not available")
class TestOvrtxIntegration:
    """Integration test: render scene via ovrtx and write through BasicWriter."""

    SCENE_PATH = Path(__file__).parent / "data" / "test_scene.usda"

    def test_ovrtx_renders_bright_scene(self, tmp_path: Path) -> None:
        """Verify ovrtx renders a properly lit scene with all objects."""
        import ovrtx

        renderer = ovrtx.Renderer()
        try:
            renderer.open_usd(str(self.SCENE_PATH.resolve()))

            # Settle
            for _ in range(20):
                outputs = renderer.step({"/Render/RenderProduct_001"}, 0.0)

            prod = outputs["/Render/RenderProduct_001"]
            frame = prod.frames[0]

            with frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU) as rv:
                ldr = np.from_dlpack(rv).copy()[:, :, :3]

            # The real render resolution is dictated by the scene's
            # RenderProduct, not the synthetic (H, W) used by the fast tests
            # above.
            match = re.search(
                r"uniform int2 resolution = \((\d+),\s*(\d+)\)",
                self.SCENE_PATH.read_text(),
            )
            assert match is not None, "RenderProduct resolution not found in test scene"
            scene_w, scene_h = int(match.group(1)), int(match.group(2))

            assert ldr.shape == (scene_h, scene_w, 3)
            assert ldr.dtype == np.uint8
            # Scene should render something (emissive lit)
            assert ldr.mean() > 5.0, f"Scene too dark: mean={ldr.mean():.1f}"

            with frame.render_vars["SemanticInstanceSegmentation"].map(
                device=ovrtx.Device.CPU
            ) as rv:
                inst_seg = np.from_dlpack(rv).copy()
            if inst_seg.ndim == 3:
                inst_seg = inst_seg[:, :, 0]

            unique_ids = set(np.unique(inst_seg))
            # Should have at least 5 unique instance IDs
            assert len(unique_ids) >= 5, f"Expected >=5 instances, got {unique_ids}"

            # Write through BasicWriter to confirm the end-to-end pipeline using
            # the real, already-extracted render buffers (no fabricated
            # payloads).
            render_vars = {
                "LdrColor": ldr,
                "SemanticInstanceSegmentation": inst_seg,
            }
            writer = BasicWriter(BasicWriterConfig(colorize=True, image_format="png"))
            backend = DiskBackend(
                DiskBackendConfig(root_dir=str(tmp_path), overwrite=True)
            )
            writer.write_frame(0, render_vars, backend)
            writer.finalize(backend)
            assert (tmp_path / "0000" / "ldr_color.png").exists()
        finally:
            # Drop the step-result handles (frame/products) before destroying
            # the renderer, then destroy it deterministically. ovrtx logs
            # leak errors (and can hang at interpreter exit) if a renderer is
            # torn down by GC while step results or mappings are still alive.
            outputs = prod = frame = None  # noqa: F841
            gc.collect()
            del renderer
            gc.collect()
