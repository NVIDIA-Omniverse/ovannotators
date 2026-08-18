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

"""Generate golden output data for all writers using ovrtx rendering.

Renders the test scene (tests/golden/simple_scene/scene.usda) via ovrtx and
feeds the render output through each writer.

Requirements:
    - ovrtx (pip install ovrtx)
    - PyNvVideoCodec (pip install PyNvVideoCodec) for video writers
    - NVIDIA GPU with CUDA runtime

Usage:
    python tests/generate_golden.py [--output-dir /path/to/output] [--settle-steps 20]

Default output: ~/_output/<writer_name>/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

import numpy as np
import ovrtx

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
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


NUM_FRAMES = 5
H, W = 256, 512

SCENE_PATH = Path(__file__).parent / "data" / "test_scene.usda"
RENDER_PRODUCT = "/Render/RenderProduct_001"

# Render var source names (from scene.usda RenderVar prims)
RV_LDR_COLOR = "LdrColor"
RV_DEPTH = "DistanceToCameraSD"
RV_NORMALS = "NormalSD"
RV_INST_SEG = "SemanticInstanceSegmentation"
RV_SEM_SEG = "SemanticSegmentation"

# Object labels from scene semantics
OBJECT_LABELS = ["cone", "cube", "cube", "sphere", "plane"]


def extract_render_vars(frame: ovrtx.FrameOutput) -> dict:
    """Extract numpy arrays from an ovrtx FrameOutput."""
    render_vars = {}

    # LdrColor: RGBA uint8 -> RGB uint8 (drop alpha)
    with frame.render_vars[RV_LDR_COLOR].map(device=ovrtx.Device.CPU) as rv:
        ldr_rgba = np.from_dlpack(rv).copy()
    render_vars["LdrColor"] = ldr_rgba[:, :, :3]

    # HdrColor: RGBA float16 (keep all 4 channels for EXR)
    if "HdrColor" in frame.render_vars:
        with frame.render_vars["HdrColor"].map(device=ovrtx.Device.CPU) as rv:
            hdr = np.from_dlpack(rv).copy()
        render_vars["HdrColor"] = hdr

    # Depth: float32 -> 2D
    with frame.render_vars[RV_DEPTH].map(device=ovrtx.Device.CPU) as rv:
        depth = np.from_dlpack(rv).copy()
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    render_vars["DistanceToCameraSD"] = depth

    # Normals: float32 -> RGB (drop alpha)
    with frame.render_vars[RV_NORMALS].map(device=ovrtx.Device.CPU) as rv:
        normals = np.from_dlpack(rv).copy()
    render_vars["NormalSD"] = normals[:, :, :3]

    # Instance segmentation: uint32 -> 2D
    with frame.render_vars[RV_INST_SEG].map(device=ovrtx.Device.CPU) as rv:
        inst_seg = np.from_dlpack(rv).copy()
    if inst_seg.ndim == 3:
        inst_seg = inst_seg[:, :, 0]
    render_vars["SemanticInstanceSegmentation"] = inst_seg

    # Semantic segmentation: uint32 -> 2D
    with frame.render_vars[RV_SEM_SEG].map(device=ovrtx.Device.CPU) as rv:
        sem_seg = np.from_dlpack(rv).copy()
    if sem_seg.ndim == 3:
        sem_seg = sem_seg[:, :, 0]
    render_vars["SemanticSegmentation"] = sem_seg

    # DiffuseAlbedoSD: uint8 RGBA -> RGB (drop alpha)
    if "DiffuseAlbedoSD" in frame.render_vars:
        with frame.render_vars["DiffuseAlbedoSD"].map(device=ovrtx.Device.CPU) as rv:
            albedo = np.from_dlpack(rv).copy()
        if albedo.ndim == 3 and albedo.shape[2] == 4:
            albedo = albedo[:, :, :3]
        if albedo.size > 0 and albedo.shape[0] > 1:
            render_vars["DiffuseAlbedoSD"] = albedo

    # Camera3dPositionSD: float32
    if "Camera3dPositionSD" in frame.render_vars:
        with frame.render_vars["Camera3dPositionSD"].map(device=ovrtx.Device.CPU) as rv:
            cam_pos = np.from_dlpack(rv).copy()
        if cam_pos.size > 0 and cam_pos.shape[0] > 1:
            render_vars["Camera3dPositionSD"] = cam_pos

    # DistanceToImagePlaneSD: float32 -> 2D
    if "DistanceToImagePlaneSD" in frame.render_vars:
        with frame.render_vars["DistanceToImagePlaneSD"].map(
            device=ovrtx.Device.CPU
        ) as rv:
            dist_plane = np.from_dlpack(rv).copy()
        if dist_plane.ndim == 3:
            dist_plane = dist_plane[:, :, 0]
        if dist_plane.size > 0:
            render_vars["DistanceToImagePlaneSD"] = dist_plane

    # Compute bounding boxes from instance segmentation
    render_vars["SemanticBoundingBox2DTight"] = compute_bbox2d(inst_seg)
    render_vars["SemanticBoundingBox3D"] = compute_bbox3d()
    render_vars["CameraParams"] = make_camera_params()

    return render_vars


def compute_bbox2d(inst_seg: np.ndarray) -> np.ndarray:
    """Compute tight 2D bounding boxes from instance segmentation."""
    bbox2d_dtype = np.dtype(
        [
            ("x_min", np.float32),
            ("y_min", np.float32),
            ("x_max", np.float32),
            ("y_max", np.float32),
            ("semantic_id", np.uint32),
            ("instance_id", np.uint32),
            ("label", "U32"),
            ("occlusion_ratio", np.float32),
        ]
    )
    # Find unique instance IDs (skip 0 = background)
    unique_ids = sorted(set(np.unique(inst_seg)) - {0})
    bboxes = []
    for idx, inst_id in enumerate(unique_ids):
        mask = inst_seg == inst_id
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        label_idx = min(idx, len(OBJECT_LABELS) - 1)
        label = OBJECT_LABELS[label_idx] if label_idx >= 0 else "unknown"
        bboxes.append(
            (
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
                np.uint32(inst_id),
                np.uint32(inst_id),
                label,
                0.0,
            )
        )

    bbox2d = np.zeros(len(bboxes), dtype=bbox2d_dtype)
    for idx, b in enumerate(bboxes):
        bbox2d[idx] = b
    return bbox2d


def compute_bbox3d() -> np.ndarray:
    """Return 3D bounding boxes for the scene objects (from USD positions)."""
    bbox3d_dtype = np.dtype(
        [
            ("center", np.float32, (3,)),
            ("extent", np.float32, (3,)),
            ("rotation", np.float32, (9,)),
            ("semantic_id", np.int32),
            ("instance_id", np.int32),
            ("label", "U32"),
        ]
    )
    # Matches scene.usda object positions/sizes
    objects = [
        ("cube", [-1.5, 0.5, 0.0], [1.0, 1.0, 1.0]),  # RedCube
        ("cube", [0.0, 0.75, -0.5], [0.8, 1.5, 0.8]),  # GreenCube
        ("sphere", [1.5, 0.55, 0.3], [1.1, 1.1, 1.1]),  # BlueSphere
        ("cone", [0.5, 0.6, 1.5], [0.8, 1.2, 0.8]),  # YellowCone
    ]
    bbox3d = np.zeros(len(objects), dtype=bbox3d_dtype)
    for i, (label, center, extent) in enumerate(objects):
        bbox3d[i]["center"] = center
        bbox3d[i]["extent"] = extent
        bbox3d[i]["rotation"] = np.eye(3, dtype=np.float32).ravel()
        bbox3d[i]["semantic_id"] = i + 1
        bbox3d[i]["instance_id"] = i + 1
        bbox3d[i]["label"] = label
    return bbox3d


def make_camera_params() -> np.ndarray:
    """Camera intrinsics/extrinsics matching scene.usda Camera prim."""
    cam_dtype = np.dtype(
        [
            ("K", np.float64, (9,)),
            ("R", np.float64, (9,)),
            ("t", np.float64, (3,)),
            ("width", np.int32),
            ("height", np.int32),
            ("near", np.float64),
            ("far", np.float64),
        ]
    )
    # From scene.usda: focalLength=24, horizontalAperture=20.955, resolution 1280x720
    # fx = focalLength * width / horizontalAperture = 24 * 1280 / 20.955 ≈ 1466.1
    fx = 24.0 * W / 20.955
    fy = fx  # square pixels
    cx = W / 2.0
    cy = H / 2.0

    cam = np.zeros(1, dtype=cam_dtype)
    cam[0]["K"] = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    cam[0]["R"] = np.eye(3, dtype=np.float64).ravel()
    cam[0]["t"] = [4.0, 3.5, 6.0]  # camera translate from USD
    cam[0]["width"] = W
    cam[0]["height"] = H
    cam[0]["near"] = 0.1
    cam[0]["far"] = 100.0
    return cam


def generate_writer(
    name: str, writer, output_dir: Path, all_frames: list[dict]
) -> None:
    """Generate golden output for a single writer."""
    out = output_dir / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    backend = DiskBackend(DiskBackendConfig(root_dir=str(out), overwrite=True))

    for frame_id, render_vars in enumerate(all_frames):
        writer.write_frame(frame_id, render_vars, backend)

    writer.finalize(backend)
    print(f"  ✓ {name}: {sum(1 for _ in out.rglob('*') if _.is_file())} files")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate golden output data for all writers via ovrtx"
    )
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "_output")
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=20,
        help="Number of initial render steps for path tracer convergence",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_path = str(SCENE_PATH.resolve())
    print(f"Scene: {scene_path}")
    print(f"Output: {output_dir}")
    print(f"Settle steps: {args.settle_steps}")

    # Initialize ovrtx renderer
    renderer = ovrtx.Renderer()
    renderer.open_usd(scene_path)

    # Settle: let path tracer converge
    print(f"Settling renderer ({args.settle_steps} steps)...")
    for _ in range(args.settle_steps):
        renderer.step({RENDER_PRODUCT}, 1.0 / 24.0)

    # Render NUM_FRAMES
    print(f"Rendering {NUM_FRAMES} frames...")
    all_frames = []
    for frame_id in range(NUM_FRAMES):
        outputs = renderer.step({RENDER_PRODUCT}, 1.0 / 24.0)
        prod = outputs[RENDER_PRODUCT]
        frame = prod.frames[0]
        render_vars = extract_render_vars(frame)
        all_frames.append(render_vars)
        ldr_mean = render_vars["LdrColor"].mean()
        n_instances = len(np.unique(render_vars["SemanticInstanceSegmentation"])) - 1
        print(
            f"  Frame {frame_id}: LdrColor mean={ldr_mean:.1f}, instances={n_instances}"
        )

    # Write through all writers
    print("\nGenerating writer output:")
    writers = {
        "basic": BasicWriter(BasicWriterConfig(colorize=True, image_format="png")),
        "coco": COCOWriter(COCOWriterConfig(split="train")),
        "kitti": KITTIWriter(KITTIWriterConfig(write_depth=True)),
        "yolo": YOLOWriter(YOLOWriterConfig(task="detect")),
        "bop": BOPWriter(BOPWriterConfig(scene_id=1, depth_scale=1.0)),
        "lerobot": LeRobotWriter(
            LeRobotWriterConfig(
                fps=30,
                episode_index=0,
                task_description="ovrtx test scene",
                camera_keys=["front"],
                frames_per_chunk=1000,
            )
        ),
        "cosmos_transfer": CosmosTransferWriter(
            CosmosTransferWriterConfig(
                fps=24,
                conditioning_types=["canny", "depth", "segmentation"],
            )
        ),
    }

    for name, writer in writers.items():
        try:
            generate_writer(name, writer, output_dir, all_frames)
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print(f"\nDone. Output at: {output_dir}")


if __name__ == "__main__":
    main()
