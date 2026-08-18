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

"""Tests for ovannotators.types dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from ovannotators.types import (
    BBox2D,
    BBox3D,
    CameraParams,
    PointCloudResult,
    Resolution,
)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolution_construction():
    r = Resolution(width=1920, height=1080)
    assert r.width == 1920
    assert r.height == 1080


def test_resolution_frozen():
    r = Resolution(width=640, height=480)
    with pytest.raises(Exception):
        r.width = 100  # type: ignore[misc]


def test_resolution_equality():
    assert Resolution(640, 480) == Resolution(640, 480)
    assert Resolution(640, 480) != Resolution(1920, 1080)


# ---------------------------------------------------------------------------
# BBox2D
# ---------------------------------------------------------------------------


def test_bbox2d_construction_required_fields():
    b = BBox2D(
        x_min=10.0,
        y_min=20.0,
        x_max=100.0,
        y_max=200.0,
        semantic_id=1,
        instance_id=42,
        label="car",
    )
    assert b.x_min == 10.0
    assert b.y_min == 20.0
    assert b.x_max == 100.0
    assert b.y_max == 200.0
    assert b.semantic_id == 1
    assert b.instance_id == 42
    assert b.label == "car"
    assert b.occlusion_ratio == 0.0  # default


def test_bbox2d_with_occlusion():
    b = BBox2D(
        x_min=0,
        y_min=0,
        x_max=10,
        y_max=10,
        semantic_id=1,
        instance_id=1,
        label="test",
        occlusion_ratio=0.5,
    )
    assert b.occlusion_ratio == 0.5


def test_bbox2d_frozen():
    b = BBox2D(
        x_min=0, y_min=0, x_max=10, y_max=10, semantic_id=1, instance_id=1, label="car"
    )
    with pytest.raises(Exception):
        b.label = "truck"  # type: ignore[misc]


def test_bbox2d_equality():
    a = BBox2D(0, 0, 1, 1, 0, 0, "x")
    b = BBox2D(0, 0, 1, 1, 0, 0, "x")
    assert a == b


# ---------------------------------------------------------------------------
# BBox3D
# ---------------------------------------------------------------------------


def test_bbox3d_construction():
    center = [1.0, 2.0, 3.0]
    extent = [0.5, 0.5, 1.0]
    rotation = np.eye(3)
    b = BBox3D(
        center=center,
        extent=extent,
        rotation=rotation,
        semantic_id=1,
        instance_id=1,
        label="car",
        coordinate_system="camera",
    )
    assert b.label == "car"
    assert b.coordinate_system == "camera"
    assert b.center.shape == (3,)
    assert b.extent.shape == (3,)
    assert b.rotation.shape == (3, 3)


def test_bbox3d_coerces_arrays_to_float32():
    b = BBox3D(
        center=[1.0, 2.0, 3.0],
        extent=[0.5, 0.5, 1.0],
        rotation=np.eye(3, dtype=np.float64),
        semantic_id=0,
        instance_id=0,
        label="box",
        coordinate_system="world",
    )
    assert b.center.dtype == np.float32
    assert b.extent.dtype == np.float32
    assert b.rotation.dtype == np.float32


def test_bbox3d_rotation_values_preserved():
    rot = np.eye(3, dtype=np.float32)
    b = BBox3D(
        center=[0, 0, 0],
        extent=[1, 1, 1],
        rotation=rot,
        semantic_id=0,
        instance_id=0,
        label="x",
        coordinate_system="camera",
    )
    np.testing.assert_array_almost_equal(b.rotation, np.eye(3))


def test_bbox3d_world_coordinate_system():
    b = BBox3D(
        center=[0, 0, 0],
        extent=[1, 1, 1],
        rotation=np.eye(3),
        semantic_id=0,
        instance_id=0,
        label="x",
        coordinate_system="world",
    )
    assert b.coordinate_system == "world"


def test_bbox3d_frozen():
    b = BBox3D(
        center=[0, 0, 0],
        extent=[1, 1, 1],
        rotation=np.eye(3),
        semantic_id=0,
        instance_id=0,
        label="x",
        coordinate_system="camera",
    )
    with pytest.raises(Exception):
        b.label = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CameraParams
# ---------------------------------------------------------------------------


def test_camera_params_construction():
    K = np.eye(3, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    cam = CameraParams(
        K=K,
        R=R,
        t=t,
        resolution=Resolution(640, 480),
        near=0.1,
        far=100.0,
        fov_x=1.047,
        fov_y=0.785,
    )
    assert cam.near == 0.1
    assert cam.far == 100.0
    assert cam.fov_x == pytest.approx(1.047)
    assert cam.resolution == Resolution(640, 480)


def test_camera_params_coerces_to_float64():
    K = np.eye(3, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    cam = CameraParams(
        K=K,
        R=R,
        t=t,
        resolution=Resolution(640, 480),
        near=0.1,
        far=100.0,
        fov_x=1.0,
        fov_y=0.75,
    )
    assert cam.K.dtype == np.float64
    assert cam.R.dtype == np.float64
    assert cam.t.dtype == np.float64


def test_camera_params_shapes():
    K = np.eye(3, dtype=np.float64)
    cam = CameraParams(
        K=K,
        R=K,
        t=np.zeros(3),
        resolution=Resolution(640, 480),
        near=0.1,
        far=100.0,
        fov_x=1.0,
        fov_y=0.75,
    )
    assert cam.K.shape == (3, 3)
    assert cam.R.shape == (3, 3)
    assert cam.t.shape == (3,)


def test_camera_params_frozen():
    K = np.eye(3, dtype=np.float64)
    cam = CameraParams(
        K=K,
        R=K,
        t=np.zeros(3),
        resolution=Resolution(640, 480),
        near=0.1,
        far=100.0,
        fov_x=1.0,
        fov_y=0.75,
    )
    with pytest.raises(Exception):
        cam.near = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PointCloudResult
# ---------------------------------------------------------------------------


def test_pointcloud_result_minimal():
    pts = np.zeros((100, 3), dtype=np.float32)
    pc = PointCloudResult(points=pts, normals=None, colors=None, semantic_ids=None)
    assert pc.points.shape == (100, 3)
    assert pc.normals is None
    assert pc.colors is None
    assert pc.semantic_ids is None


def test_pointcloud_result_all_fields():
    pts = np.zeros((50, 3), dtype=np.float32)
    nrm = np.ones((50, 3), dtype=np.float32)
    clr = np.zeros((50, 4), dtype=np.uint8)
    sem = np.arange(50, dtype=np.uint32)
    pc = PointCloudResult(points=pts, normals=nrm, colors=clr, semantic_ids=sem)
    assert pc.normals is not None
    assert pc.colors is not None
    assert pc.semantic_ids is not None
    assert pc.semantic_ids.shape == (50,)


def test_pointcloud_result_frozen():
    pts = np.zeros((10, 3), dtype=np.float32)
    pc = PointCloudResult(points=pts, normals=None, colors=None, semantic_ids=None)
    with pytest.raises(Exception):
        pc.points = pts  # type: ignore[misc]
