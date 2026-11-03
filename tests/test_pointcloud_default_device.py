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

"""Device dispatch contracts without requiring a CUDA device."""

from unittest.mock import patch

import numpy as np

from ovannotators.writers._helpers import point_cloud_from_depth


def test_default_dispatches_to_cuda():
    depth, camera, colors, ids, expected = (object() for _ in range(5))
    with patch(
        "ovannotators.writers._pointcloud_gpu._point_cloud_gpu", return_value=expected
    ) as gpu:
        assert point_cloud_from_depth(depth, camera, colors, ids) is expected
        gpu.assert_called_once_with(depth, camera, colors, ids, "cuda", "cpu", False)


def test_explicit_cpu_does_not_dispatch_to_cuda():
    from ovannotators.types import CameraParams, Resolution

    camera = CameraParams(
        K=np.eye(3),
        R=np.eye(3),
        t=np.zeros(3),
        resolution=Resolution(2, 2),
        near=0.1,
        far=100,
        fov_x=1,
        fov_y=1,
    )
    with patch(
        "ovannotators.writers._pointcloud_gpu._point_cloud_gpu",
        side_effect=AssertionError("unexpected GPU"),
    ):
        cloud = point_cloud_from_depth(
            np.ones((2, 2), np.float32), camera, device="cpu"
        )
    assert cloud.points.shape == (4, 3)
