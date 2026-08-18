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

"""Shared fixtures for performance benchmarks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ovannotators.writers._helpers import (
    OVRTX_BBOX2D_DTYPE,
    OVRTX_BBOX3D_DTYPE,
    OVRTX_CAMERA_PARAMS_DTYPE,
)


def _id128(value: int) -> tuple[int, int, int, int]:
    """Return a 128-bit id as four little-endian uint32 words (low word first)."""
    return (int(value), 0, 0, 0)


def _pack_semantic_id_map(labels: dict[int, str]) -> np.ndarray:
    """Pack ``{semantic_id: label}`` into a ``SemanticIdMap`` uint8 buffer."""
    entry_dtype = np.dtype(
        [("semantic_id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    num = len(labels)
    header_size = num * entry_dtype.itemsize
    items = list(labels.items())
    strings = b""
    offsets: list[int] = []
    lengths: list[int] = []
    for _sid, label in items:
        encoded = label.encode("utf-8")
        offsets.append(header_size + len(strings))
        lengths.append(len(encoded))
        strings += encoded
    entries = np.zeros(num, dtype=entry_dtype)
    for i, (sid, _label) in enumerate(items):
        entries[i]["semantic_id"] = _id128(sid)
        entries[i]["label_length"] = lengths[i]
        entries[i]["label_offset"] = offsets[i]
    raw = entries.tobytes() + strings + int(num).to_bytes(4, "little")
    return np.frombuffer(raw, dtype=np.uint8).copy()


# --- Resolution presets ---
RESOLUTIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}


@pytest.fixture(params=["720p", "1080p"], ids=lambda r: r)
def resolution(request) -> tuple[int, int]:
    """Parametrized resolution fixture (w, h)."""
    return RESOLUTIONS[request.param]


@pytest.fixture
def rgb_frame(resolution) -> np.ndarray:
    """Synthetic RGBA frame at the given resolution."""
    w, h = resolution
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)


@pytest.fixture
def depth_frame(resolution) -> np.ndarray:
    """Synthetic depth frame (float32, 0-100m range) at the given resolution."""
    w, h = resolution
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 100.0, size=(h, w)).astype(np.float32)


@pytest.fixture
def seg_frame(resolution) -> np.ndarray:
    """Synthetic instance segmentation frame (uint32, 0-50 IDs)."""
    w, h = resolution
    rng = np.random.default_rng(42)
    return rng.integers(0, 50, size=(h, w), dtype=np.uint32)


@pytest.fixture
def bbox_2d_data() -> np.ndarray:
    """Synthetic ``SemanticBoundingBox2DTight`` array (50 boxes, OVRTX layout)."""
    rng = np.random.default_rng(42)
    n = 50
    data = np.zeros(n, dtype=OVRTX_BBOX2D_DTYPE)
    x_min = rng.integers(0, 500, n)
    y_min = rng.integers(0, 300, n)
    semantic_ids = rng.integers(1, 20, n)
    for i in range(n):
        data[i]["instance_id"] = _id128(i)
        data[i]["semantic_id"] = _id128(int(semantic_ids[i]))
        data[i]["x_min"] = int(x_min[i])
        data[i]["y_min"] = int(y_min[i])
        data[i]["x_max"] = int(x_min[i]) + int(rng.integers(20, 200))
        data[i]["y_max"] = int(y_min[i]) + int(rng.integers(20, 200))
    return data


@pytest.fixture
def bbox_3d_data(bbox_2d_data) -> np.ndarray:
    """Synthetic ``SemanticBoundingBox3D`` array matching ``bbox_2d_data`` ids."""
    rng = np.random.default_rng(7)
    n = len(bbox_2d_data)
    data = np.zeros(n, dtype=OVRTX_BBOX3D_DTYPE)
    for i in range(n):
        data[i]["instance_id"] = bbox_2d_data[i]["instance_id"]
        data[i]["semantic_id"] = bbox_2d_data[i]["semantic_id"]
        ext = rng.uniform(0.5, 3.0, 3).astype(np.float32)
        data[i]["x_min"], data[i]["y_min"], data[i]["z_min"] = -ext
        data[i]["x_max"], data[i]["y_max"], data[i]["z_max"] = ext
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = rng.uniform(-5, 5, 3).astype(np.float32)
        data[i]["transform"] = transform
    return data


@pytest.fixture
def semantic_id_map() -> np.ndarray:
    """``SemanticIdMap`` buffer covering the benchmark semantic ids (1-19)."""
    return _pack_semantic_id_map({sid: f"object_{sid}" for sid in range(1, 20)})


@pytest.fixture
def camera_params() -> np.ndarray:
    """Synthetic ``CameraParams`` array (OVRTX layout)."""
    data = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
    data[0]["view_transform"] = np.eye(4, dtype=np.float32)
    data[0]["projection"] = np.eye(4, dtype=np.float32)
    data[0]["meters_per_scene_unit"] = 1.0
    data[0]["resolution"] = (1280, 720)
    data[0]["clipping_plane"] = (0.01, 1000.0)
    data[0]["projection_model"] = b"pinhole"
    data[0]["focal_length"] = 24.0
    data[0]["fov_x"] = math.radians(60.0)
    return data


@pytest.fixture
def render_vars_full(
    rgb_frame,
    depth_frame,
    seg_frame,
    bbox_2d_data,
    bbox_3d_data,
    camera_params,
    semantic_id_map,
) -> dict:
    """Full render vars dict (keyed by ovrtx sourceName) with all common AOVs."""
    return {
        "LdrColor": rgb_frame,
        "DistanceToCameraSD": depth_frame,
        "SemanticInstanceSegmentation": seg_frame,
        "SemanticSegmentation": seg_frame,
        "SemanticBoundingBox2DTight": bbox_2d_data,
        "SemanticBoundingBox3D": bbox_3d_data,
        "CameraParams": camera_params,
        "SemanticIdMap": semantic_id_map,
    }
