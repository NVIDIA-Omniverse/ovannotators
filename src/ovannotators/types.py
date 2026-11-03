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

"""Shared plain-data types for ovannotators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp


@dataclass(frozen=True)
class Resolution:
    """Image resolution in pixels."""

    width: int
    height: int


@dataclass(frozen=True)
class BBox2D:
    """Axis-aligned 2-D bounding box in pixel coordinates.

    Attributes:
        x_min: Left edge in pixels.
        y_min: Top edge in pixels.
        x_max: Right edge in pixels.
        y_max: Bottom edge in pixels.
        semantic_id: Semantic class id this box belongs to.
        instance_id: Per-object instance id (links to segmentation masks).
        label: Human-readable class name resolved from the semantic id.
        occlusion_ratio: Fraction of the box occluded in ``[0, 1]``; ``0.0``
            when the renderer provides no occlusion information.
        prim_path: USD prim path of the instance, when a stable-id map is
            available; otherwise ``None``.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    semantic_id: int
    instance_id: int
    label: str
    occlusion_ratio: float = 0.0
    prim_path: str | None = None


@dataclass(frozen=True)
class BBox3D:
    """Oriented 3-D bounding box.

    Attributes:
        center: Shape ``(3,)`` float32 box center.
        extent: Shape ``(3,)`` float32 half-extents along each axis.
        rotation: Shape ``(3, 3)`` float32 rotation matrix (columns are the box
            axes).
        semantic_id: Semantic class id this box belongs to.
        instance_id: Per-object instance id.
        label: Human-readable class name resolved from the semantic id.
        coordinate_system: Frame the pose is expressed in; ``"world"`` by
            default (matching the ovrtx ``SemanticBoundingBox3D`` payload).
        prim_path: USD prim path of the instance, when available; else ``None``.
    """

    center: np.ndarray
    extent: np.ndarray
    rotation: np.ndarray
    semantic_id: int
    instance_id: int
    label: str
    coordinate_system: Literal["camera", "world"] = "world"
    prim_path: str | None = None

    def __post_init__(self) -> None:
        """Normalize array fields to contiguous float32 arrays."""
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float32))
        object.__setattr__(self, "extent", np.asarray(self.extent, dtype=np.float32))
        object.__setattr__(
            self, "rotation", np.asarray(self.rotation, dtype=np.float32)
        )


@dataclass(frozen=True)
class CameraParams:
    """Pinhole camera intrinsics and extrinsics.

    Attributes:
        K: ``(3, 3)`` float64 intrinsic matrix.
        R: ``(3, 3)`` float64 world-to-camera rotation matrix.
        t: ``(3,)`` float64 world-to-camera translation vector.
        resolution: Image resolution in pixels.
        near: Near clipping-plane distance.
        far: Far clipping-plane distance.
        fov_x: Horizontal field of view in radians.
        fov_y: Vertical field of view in radians.
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    resolution: Resolution
    near: float
    far: float
    fov_x: float  # radians
    fov_y: float  # radians

    def __post_init__(self) -> None:
        """Normalize matrix/vector fields to float64 arrays."""
        object.__setattr__(self, "K", np.asarray(self.K, dtype=np.float64))
        object.__setattr__(self, "R", np.asarray(self.R, dtype=np.float64))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64))


@dataclass(frozen=True)
class PointCloudResult:
    """Point-cloud data stored in Warp arrays on CPU or CUDA.

    Attributes:
        points: ``(N, 3)`` float32 world/camera-space point positions.
        normals: Optional ``(N, 3)`` float32 per-point normals.
        colors: Optional ``(N, C)`` per-point colors, preserving input dtype.
        semantic_ids: Optional ``(N,)`` per-point IDs, preserving input dtype.
        count: Optional one-element CUDA int32 array holding the valid point
            count. When present, arrays have capacity ``N`` and only the prefix
            ``[:count[0]]`` is defined. When absent, every row is valid.

    Arrays retain their storage across subsequent helper calls. Use Warp's
    array transfer and interoperability APIs to consume them on another device.
    """

    points: wp.array
    normals: wp.array | None = None
    colors: wp.array | None = None
    semantic_ids: wp.array | None = None
    count: wp.array | None = None
