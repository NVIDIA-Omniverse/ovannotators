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

"""End-to-end test for multi-tensor (sensor) render-var extraction.

Renders a lidar scene that produces a multi-tensor ``PointCloud`` render var and
verifies :meth:`OvrtxOutput.extract` returns the nested
``{"tensors": ..., "params": ...}`` structure (vs. a single Warp array for
image AOVs). Unlike the unit tests in ``test_ovrtx_output.py`` (which mock the
ovrtx mapping), this exercises the real ovrtx producer path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ovannotators._internal.optional_deps import ovrtx_available

_OVRTX_AVAILABLE = ovrtx_available()

pytestmark = pytest.mark.skipif(
    not _OVRTX_AVAILABLE,
    reason="ovrtx not installed — skipping GPU sensor extraction test",
)

LIDAR_SCENE = Path(__file__).parent / "data" / "lidar_scene.usda"
LIDAR_PRODUCT = "/World/Render/Products/LidarProduct"
WARMUP_STEPS = 3
STEP_DT_SECONDS = 0.1
EXPECTED_CHANNELS = {"Coordinates", "Intensity", "Counts", "TimeOffsetNs"}


@pytest.fixture(scope="module")
def lidar_pointcloud() -> dict:
    """Render the lidar scene once and extract the ``PointCloud`` render var.

    Returns:
        The value returned by :meth:`OvrtxOutput.extract` for ``PointCloud``.
    """
    import ovrtx

    from ovannotators.renderer_outputs.ovrtx import OvrtxOutput
    from tests._ovstage_session import OvstageSession

    # Motion BVH is required by the lidar sensor pipeline. ovrtx 0.4 replaced
    # the boolean ``enable_motion_bvh`` with the ``motion_bvh`` tri-state enum.
    session = OvstageSession(
        "lidar-pointcloud",
        ovrtx.RendererConfig(motion_bvh=ovrtx.MotionBvh.ENABLE),
    )
    session.open_usd(LIDAR_SCENE)

    # Warm up the sensor before reading the point cloud frame.
    for _ in range(WARMUP_STEPS):
        session.step({LIDAR_PRODUCT}, STEP_DT_SECONDS)

    products = session.step({LIDAR_PRODUCT}, STEP_DT_SECONDS)
    frame = products[LIDAR_PRODUCT].frames[0]
    return OvrtxOutput().extract(frame, "PointCloud")


def test_pointcloud_returns_nested_tensors_and_params(lidar_pointcloud: dict) -> None:
    """A multi-tensor sensor var extracts to a nested tensors+params dict."""
    result = lidar_pointcloud

    assert isinstance(result, dict)
    assert set(result.keys()) == {"tensors", "params"}
    assert isinstance(result["tensors"], dict)
    assert isinstance(result["params"], dict)


def test_pointcloud_exposes_expected_channels(lidar_pointcloud: dict) -> None:
    """The lidar PointCloud channels are exposed by name as Warp arrays."""
    tensors = lidar_pointcloud["tensors"]

    assert EXPECTED_CHANNELS.issubset(tensors.keys())
    # Each channel is a zero-copy Warp view; confirm it converts to NumPy.
    for name in EXPECTED_CHANNELS:
        assert hasattr(tensors[name], "numpy"), f"{name} is not a Warp array"
        tensors[name].numpy()
