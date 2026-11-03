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

"""Compiled CPU pointcloud parity, raster order, formats and input coverage."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import warp as wp

from ovannotators.types import CameraParams, Resolution
from ovannotators.writers._helpers import point_cloud_from_depth


def _camera():
    return CameraParams(
        K=np.array([[137.137, 0, 30.125], [0, 153.789, 34.375], [0, 0, 1]]),
        R=np.eye(3),
        t=np.zeros(3),
        resolution=Resolution(256, 256),
        near=0.1,
        far=100,
        fov_x=1,
        fov_y=1,
    )


def _reference(depth, colors, ids):
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    rows, cols = np.nonzero(valid)
    z = z[valid]
    K = _camera().K.astype(np.float32)
    points = np.column_stack(
        (
            (cols.astype(np.float32) - K[0, 2]) * z / K[0, 0],
            (rows.astype(np.float32) - K[1, 2]) * z / K[1, 1],
            z,
        )
    ).astype(np.float32)
    return points, colors[valid], ids[valid]


@pytest.mark.parametrize("depth_dtype", [np.float16, np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("strided", [False, True])
def test_compiled_matches_reference_for_supported_depth_formats(depth_dtype, strided):
    rng = np.random.default_rng(11)
    depth = rng.uniform(0, 30, (256, 256)).astype(depth_dtype)
    depth[::3, ::5] = 0
    if np.issubdtype(depth_dtype, np.floating):
        depth[0, :4] = [np.nan, np.inf, -np.inf, -1]
    colors = rng.integers(0, 256, (256, 256, 4), dtype=np.uint8)
    ids = rng.integers(0, 2**32, (256, 256), dtype=np.uint64) + np.uint64(2**40)
    if strided:
        depth, colors, ids = depth[:, ::-1], colors[:, ::-1], ids[:, ::-1]
    before = [x.copy() for x in (depth, colors, ids)]
    actual = point_cloud_from_depth(
        depth[..., None], _camera(), colors, ids[..., None], device="cpu"
    )
    expected = _reference(depth, colors, ids)
    for name, values in zip(["points", "colors", "semantic_ids"], expected):
        np.testing.assert_array_equal(getattr(actual, name).numpy(), values)
        assert not np.shares_memory(getattr(actual, name).numpy(), values)
    for original, saved in zip((depth, colors, ids), before):
        np.testing.assert_array_equal(original, saved)
    assert actual.points.is_contiguous


@pytest.mark.parametrize(
    "color_dtype,id_dtype",
    [(np.float32, np.int32), (np.float64, np.uint16), (np.uint16, np.bool_)],
)
def test_compiled_preserves_attribute_dtypes(color_dtype, id_dtype):
    depth = np.ones((256, 256), np.float32)
    colors = np.arange(256 * 256 * 3).reshape(256, 256, 3).astype(color_dtype)
    ids = np.arange(256 * 256).reshape(256, 256).astype(id_dtype)
    actual = point_cloud_from_depth(depth, _camera(), colors, ids, device="cpu")
    expected = _reference(depth, colors, ids)
    for name, values in zip(["points", "colors", "semantic_ids"], expected):
        np.testing.assert_array_equal(getattr(actual, name).numpy(), values)
        assert getattr(actual, name).dtype == wp.dtype_from_numpy(values.dtype)


@pytest.mark.parametrize("attributes", [False, True])
def test_compiled_empty_cloud_and_absent_attributes(attributes):
    depth = np.zeros((256, 256), np.float32)
    colors = np.zeros((256, 256, 4), np.uint8) if attributes else None
    ids = np.zeros((256, 256), np.uint32) if attributes else None
    result = point_cloud_from_depth(depth, _camera(), colors, ids, device="cpu")
    assert result.points.shape == (0, 3)
    if attributes:
        assert result.colors.shape == (0, 4) and result.semantic_ids.shape == (0,)
    else:
        assert result.colors is None and result.semantic_ids is None
    depth[2, 3] = 2
    assert point_cloud_from_depth(
        depth, _camera(), colors, ids, device="cpu"
    ).points.shape == (1, 3)


@pytest.mark.parametrize("shape", [(3, 4), (64, 80), (256, 256)])
def test_compiled_always_launches_on_cpu(monkeypatch, shape):
    devices = []
    launch = wp.launch

    def record_launch(*args, **kwargs):
        devices.append(kwargs["device"])
        return launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", record_launch)
    cloud = point_cloud_from_depth(np.ones(shape, np.float32), _camera(), device="cpu")
    assert devices == ["cpu", "cpu"] and len(cloud.points) == np.prod(shape)


def test_object_attributes_are_rejected_for_warp_storage(monkeypatch):
    def unexpected_launch(*args, **kwargs):
        raise AssertionError(
            "Unsupported object attributes must not launch native kernels"
        )

    monkeypatch.setattr(wp, "launch", unexpected_launch)
    colors = np.full((256, 256, 1), "red", dtype=object)
    with pytest.raises(ValueError, match="Warp-supported numeric dtypes"):
        point_cloud_from_depth(
            np.ones((256, 256), np.float32), _camera(), colors, device="cpu"
        )


def test_compiled_concurrent_cameras_have_fresh_state_and_outputs():
    point_cloud_from_depth(np.ones((256, 256), np.float32), _camera(), device="cpu")

    def capture(index):
        camera = _camera()
        camera.K[0, 0] = camera.K[1, 1] = index + 1
        camera.K[0, 2] = camera.K[1, 2] = 0
        depth = np.full((256, 256), index + 1, np.float32)
        result = point_cloud_from_depth(depth, camera, device="cpu")
        assert np.all(depth == index + 1)
        return result.points

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(capture, range(8)))
    for index, points in enumerate(outputs):
        np.testing.assert_array_equal(points[257], [1, 1, index + 1])
    outputs[0].fill_(-123)
    assert outputs[1].numpy()[257, 2] == 2


def test_unaligned_input_is_normalized_before_native_access():
    depth = np.ndarray(
        (256, 256), dtype=np.float32, buffer=bytearray(256 * 256 * 4 + 1), offset=1
    )
    depth[:] = 2
    assert not depth.flags.aligned
    cloud = point_cloud_from_depth(depth, _camera(), device="cpu")
    np.testing.assert_array_equal(cloud.points[:, 2], 2)


@pytest.mark.parametrize("shape", [(3, 4), (256, 256)])
def test_cpu_result_fields_are_warp_arrays(shape):
    depth = np.ones(shape, np.float32)
    colors = np.zeros((*shape, 4), np.uint8)
    ids = np.arange(depth.size, dtype=np.uint64).reshape(shape)
    cloud = point_cloud_from_depth(depth, _camera(), colors, ids, device="cpu")
    assert cloud.count is None
    for name in ("points", "colors", "semantic_ids"):
        value = getattr(cloud, name)
        assert isinstance(value, wp.array)
        assert value.device.is_cpu
    np.testing.assert_array_equal(cloud.semantic_ids, ids.ravel())


@pytest.mark.parametrize("dtype", [np.dtype(">f4"), np.dtype(">f8"), np.dtype(">u2")])
def test_non_native_byte_order_inputs(dtype):
    depth = np.full((3, 4), 2, dtype=dtype)
    colors = np.arange(36).reshape(3, 4, 3).astype(dtype)
    ids = np.arange(12).reshape(3, 4).astype(">u8")
    cloud = point_cloud_from_depth(depth, _camera(), colors, ids, device="cpu")
    np.testing.assert_array_equal(
        cloud.points.numpy(), _reference(depth, colors, ids)[0]
    )
    np.testing.assert_array_equal(cloud.colors.numpy(), colors.reshape(-1, 3))
    np.testing.assert_array_equal(cloud.semantic_ids.numpy(), ids.ravel())


def test_warp_payloads_are_not_converted_to_numpy(monkeypatch):
    depth = wp.array(np.ones((3, 4, 1), np.float64), dtype=wp.float64, device="cpu")
    colors = wp.array(np.zeros((3, 4, 4), np.uint8), dtype=wp.uint8, device="cpu")
    ids = wp.array(
        np.arange(12, dtype=np.uint64).reshape(3, 4, 1), dtype=wp.uint64, device="cpu"
    )
    numpy = wp.array.numpy

    def metadata_only(array, *args, **kwargs):
        assert array.ndim == 1 and array.dtype == wp.int64
        return numpy(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wp.array, "numpy", metadata_only)
        cloud = point_cloud_from_depth(depth, _camera(), colors, ids, device="cpu")
    np.testing.assert_array_equal(cloud.points.numpy()[:, 2], 1)
    np.testing.assert_array_equal(cloud.semantic_ids.numpy(), np.arange(12))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e40])
def test_invalid_cpu_intrinsics_are_rejected(value):
    camera = _camera()
    camera.K[0, 0] = value
    with pytest.raises(ValueError, match="intrinsics must be finite float32"):
        point_cloud_from_depth(np.ones((3, 4), np.float32), camera, device="cpu")
