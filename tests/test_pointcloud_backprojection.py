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

"""Pointcloud precision, output ownership and resolution regressions."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import warp as wp

from ovannotators.types import CameraParams, Resolution
from ovannotators.writers._helpers import point_cloud_from_depth


def _camera(height, width, fx=1.25, fy=2.5):
    return CameraParams(
        K=np.array([[fx, 0, 0.5], [0, fy, 1.5], [0, 0, 1.0]]),
        R=np.eye(3),
        t=np.zeros(3),
        resolution=Resolution(width, height),
        near=0.1,
        far=100,
        fov_x=1,
        fov_y=1,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("channels", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_formats_masks_and_attributes(dtype, channels, strided):
    source = np.array([[1, 0, 2, 4], [3, 5, 0, 6], [7, 8, 9, 0]], dtype=dtype)
    colors = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
    ids = np.arange(12, dtype=np.uint64).reshape(3, 4)
    depth = source[:, ::-1] if strided else source
    colors = colors[:, ::-1] if strided else colors
    ids = ids[:, ::-1] if strided else ids
    depth_input = depth[..., None] if channels else depth
    before = depth_input.copy()
    cloud = point_cloud_from_depth(
        depth_input,
        _camera(3, 4),
        colors=colors,
        semantic_ids=ids[..., None] if channels else ids,
        device="cpu",
    )
    pixels = [
        (u, v, float(depth[v, u]))
        for v in range(3)
        for u in range(4)
        if depth[v, u] > 0
    ]
    expected = np.array(
        [[(u - 0.5) * z / 1.25, (v - 1.5) * z / 2.5, z] for u, v, z in pixels],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(cloud.points, expected)
    np.testing.assert_array_equal(cloud.colors, colors[depth > 0])
    np.testing.assert_array_equal(cloud.semantic_ids, ids[depth > 0])
    np.testing.assert_array_equal(depth_input, before)
    assert cloud.points.dtype == wp.float32 and cloud.points.is_contiguous
    assert cloud.normals is None


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (2, 3)])
def test_empty_and_invalid_depth(shape):
    depth = np.full(shape, np.nan, dtype=np.float32)
    if depth.size:
        depth[:] = [[np.inf, -np.inf, 0], [-1, np.nan, -0.0]]
    cloud = point_cloud_from_depth(depth, _camera(*shape), device="cpu")
    assert cloud.points.shape == (0, 3)
    assert cloud.colors is None and cloud.semantic_ids is None


def test_current_intrinsics_and_fresh_outputs():
    depth = np.ones((2, 3), np.float32)
    colors = np.zeros((2, 3, 3), np.uint8)
    first = point_cloud_from_depth(depth, _camera(2, 3), colors=colors, device="cpu")
    first.points.fill_(-123)
    first.colors.fill_(255)
    second = point_cloud_from_depth(
        depth, _camera(2, 3, fx=2.5, fy=5), colors=colors, device="cpu"
    )
    np.testing.assert_array_equal(
        second.points[0], np.array([-0.2, -0.3, 1], np.float32)
    )
    assert not np.shares_memory(first.points, second.points)
    assert np.all(colors == 0) and np.all(second.colors.numpy() == 0)


def test_cpu_warp_input_and_zero_focal_fallback():
    depth = wp.array(np.ones((2, 2, 1), np.float32), dtype=wp.float32, device="cpu")
    cloud = point_cloud_from_depth(depth, _camera(2, 2, fx=0, fy=0), device="cpu")
    np.testing.assert_array_equal(
        cloud.points, [[-0.5, -1.5, 1], [0.5, -1.5, 1], [-0.5, -0.5, 1], [0.5, -0.5, 1]]
    )


def test_changing_resolutions_preserves_coordinates():
    for height in range(1, 13):
        cloud = point_cloud_from_depth(
            np.ones((height, 17), np.float32), _camera(height, 17), device="cpu"
        )
        assert cloud.points.shape == (height * 17, 3)
        np.testing.assert_array_equal(
            cloud.points.numpy()[-1],
            np.array([12.4, (height - 2.5) / 2.5, 1], np.float32),
        )


def test_concurrent_cameras_do_not_share_depth_or_intrinsics():
    def capture(index):
        depth = np.full((3, 4), index + 1, np.float32)
        return point_cloud_from_depth(
            depth, _camera(3, 4, fx=1, fy=1), device="cpu"
        ).points

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(capture, range(12)))
    for index, points in enumerate(outputs):
        np.testing.assert_array_equal(
            points[0], [-0.5 * (index + 1), -1.5 * (index + 1), index + 1]
        )
