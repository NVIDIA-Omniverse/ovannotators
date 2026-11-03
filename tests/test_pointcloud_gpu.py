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

"""Real CUDA pointcloud compaction, numeric parity and output ownership."""

import numpy as np
import pytest
import warp as wp

from ovannotators.types import CameraParams, PointCloudResult, Resolution
from ovannotators.writers._helpers import (
    point_cloud_from_depth,
    point_clouds_from_depth,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device unavailable"),
]


def _cpu_cloud(cloud):
    """Read valid Warp-array prefixes, including empty shapes, on the CPU."""
    size = int(cloud.count.numpy()[0]) if cloud.count is not None else len(cloud.points)

    def transfer(value):
        if value is None:
            return None
        prefix = wp.array(
            ptr=value.ptr,
            shape=(size, *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
            strides=value.strides,
            copy=False,
        )
        prefix._ref = value
        return prefix.to("cpu")

    return PointCloudResult(
        **{
            name: transfer(getattr(cloud, name))
            for name in ("points", "colors", "semantic_ids")
        }
    )


def _camera():
    return CameraParams(
        K=np.array([[137.137, 0, 30.125], [0, 153.789, 34.375], [0, 0, 1]]),
        R=np.eye(3),
        t=np.zeros(3),
        resolution=Resolution(80, 64),
        near=0.1,
        far=100,
        fov_x=1,
        fov_y=1,
    )


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("depth_dtype", [np.float16, np.float32, np.float64])
def test_gpu_matches_cpu_and_preserves_raster_order(resident, depth_dtype):
    rng = np.random.default_rng(11)
    depth = rng.uniform(0, 30, (64, 80, 1)).astype(depth_dtype)
    depth[0, :4, 0] = [np.nan, np.inf, -np.inf, -1]
    depth[::3, ::5] = 0
    rgb = rng.integers(0, 256, (64, 80, 4), dtype=np.uint8)
    ids = rng.integers(0, 2**32, (64, 80, 1), dtype=np.uint64) + np.uint64(2**40)
    expected = point_cloud_from_depth(depth, _camera(), rgb, ids, device="cpu")
    args = [depth, rgb, ids]
    if resident:
        args = [
            wp.array(x, dtype=wp.dtype_from_numpy(x.dtype), device="cuda:0")
            for x in args
        ]
    actual = point_cloud_from_depth(args[0], _camera(), args[1], args[2])
    for name in ["points", "colors", "semantic_ids"]:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        assert not np.shares_memory(getattr(actual, name), getattr(expected, name))
    assert actual.points.is_contiguous and actual.points.dtype == wp.float32
    actual.points.fill_(-123)
    fresh = point_cloud_from_depth(
        args[0], _camera(), args[1], args[2], device="cuda:0"
    )
    np.testing.assert_array_equal(fresh.points, expected.points)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (64, 80)])
def test_gpu_empty_and_invalid_images(shape):
    depth = np.full(shape, np.nan, np.float32)
    result = point_cloud_from_depth(depth, _camera(), device="cuda:0")
    assert result.points.shape == (0, 3)
    assert result.colors is None and result.semantic_ids is None


def test_gpu_zero_focal_lengths_and_shape_errors():
    camera = _camera()
    camera.K[0, 0] = camera.K[1, 1] = 0
    depth = np.ones((64, 80), np.float32)
    expected = point_cloud_from_depth(depth, camera, device="cpu")
    actual = point_cloud_from_depth(depth, camera, device="cuda:0")
    np.testing.assert_array_equal(actual.points, expected.points)
    with pytest.raises(ValueError, match="colors must match"):
        point_cloud_from_depth(
            depth, camera, np.zeros((4, 4, 3), np.uint8), device="cuda:0"
        )


@pytest.mark.parametrize("output_device", ["cpu", "cuda"])
@pytest.mark.parametrize("reuse_gpu_outputs", [False, True])
def test_batch_dynamic_counts_owned_outputs_and_roundtrip(
    output_device, reuse_gpu_outputs
):
    rng = np.random.default_rng(31)
    depths = [
        rng.uniform(1, 20, (64, 80)).astype(np.float32),
        np.zeros((3, 5), np.float32),
    ]
    depths[0][::2, ::3] = np.nan
    colors = [rng.integers(0, 256, (64, 80, 3), dtype=np.uint8), None]
    ids = [np.arange(64 * 80, dtype=np.uint64).reshape(64, 80) + 2**40, None]
    expected = point_clouds_from_depth(
        depths, [_camera()] * 2, colors, ids, device="cpu"
    )
    result = point_clouds_from_depth(
        depths,
        [_camera()] * 2,
        colors,
        ids,
        output_device=output_device,
        reuse_gpu_outputs=reuse_gpu_outputs,
    )
    if output_device == "cuda":
        assert all(c.points.device.is_cuda and c.count.device.is_cuda for c in result)
        assert result[0].points.shape == (64 * 80, 3)
        assert int(result[0].count.numpy()[0]) == len(expected[0].points)
    assert all(isinstance(c, PointCloudResult) for c in result)
    assert all(
        isinstance(getattr(c, name), wp.array)
        for c in result
        for name in ("points", "colors", "semantic_ids")
        if getattr(c, name) is not None
    )
    if output_device == "cpu":
        assert all(c.points.device.is_cpu and c.count is None for c in result)
    saved = [_cpu_cloud(c) for c in result] if output_device == "cuda" else result
    # Same resolution but changed validity; cached staging must not alias old results.
    depths[0][:] = 2
    point_clouds_from_depth(
        depths,
        [_camera()] * 2,
        colors,
        ids,
        output_device=output_device,
        reuse_gpu_outputs=reuse_gpu_outputs,
    )
    retained = [_cpu_cloud(c) for c in result] if output_device == "cuda" else result
    for before, after, reference in zip(saved, retained, expected):
        for name in ["points", "colors", "semantic_ids"]:
            if getattr(reference, name) is not None:
                np.testing.assert_array_equal(
                    getattr(before, name).numpy(), getattr(reference, name).numpy()
                )
                np.testing.assert_array_equal(
                    getattr(after, name).numpy(), getattr(reference, name).numpy()
                )
        assert after.points.shape == reference.points.shape


def test_batch_threads_streams_and_error_recovery():
    from concurrent.futures import ThreadPoolExecutor

    def capture(z):
        depth = np.full((64, 80), z, np.float32)
        result = point_clouds_from_depth([depth] * 2, [_camera()] * 2)
        return wp.clone(result[0].points[:, 2])

    # Warp's current stream is device-global. Concurrent ScopedStream blocks
    # change it outside the helper's lock, so exercise stream scopes separately.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for z, values in zip([3, 7], pool.map(capture, [3, 7])):
            np.testing.assert_array_equal(values, np.full(64 * 80, z, np.float32))
    for z in [3, 7]:
        with wp.ScopedStream(wp.Stream("cuda:0")):
            np.testing.assert_array_equal(capture(z), np.full(64 * 80, z, np.float32))
    with pytest.raises(ValueError, match="colors must match"):
        point_clouds_from_depth(
            [np.ones((64, 80), np.float32)] * 2,
            [_camera()] * 2,
            [None, np.zeros((4, 4, 3), np.uint8)],
        )
    np.testing.assert_array_equal(capture(9), np.full(64 * 80, 9, np.float32))


def test_batch_validation_and_empty_batch():
    assert point_clouds_from_depth([], []) == []
    with pytest.raises(ValueError, match="equal lengths"):
        point_clouds_from_depth([], [_camera()])
    with pytest.raises(ValueError, match="CUDA outputs require"):
        point_cloud_from_depth(
            np.ones((1, 1), np.float32), _camera(), device="cpu", output_device="cuda"
        )
    with pytest.raises(ValueError, match="output_device"):
        point_clouds_from_depth([], [], output_device="invalid")


def test_cpu_pinned_leases_survive_retained_subviews_and_recycle():
    import gc
    from ovannotators.writers import _pointcloud_gpu as implementation

    depth = np.ones((64, 80), np.float32)
    first = point_cloud_from_depth(depth, _camera())
    retained = first.points[::3]
    original = wp.clone(retained)
    del first
    gc.collect()
    depth[:] = 19
    second = point_cloud_from_depth(depth, _camera())
    np.testing.assert_array_equal(retained, original)
    np.testing.assert_array_equal(second.points[:, 2], np.full(64 * 80, 19, np.float32))
    del retained, second
    gc.collect()
    assert all(b[0]["lease"]["count"] == 0 for b in implementation._scratch.values())
    assert (
        sum(b[1] for b in implementation._scratch.values())
        <= implementation._scratch_limit
    )


def test_resident_count_bounds_and_zero_color_channels(monkeypatch):
    depth = np.ones((3, 5), np.float32)
    cloud = point_cloud_from_depth(depth, _camera(), np.empty((3, 5, 0), np.uint8))
    assert cloud.colors.shape == (15, 0)
    from ovannotators.writers import _pointcloud_gpu as implementation

    build = implementation._build

    def invalid_count(*args):
        result = build(*args)
        result.count.fill_(16)
        return result

    monkeypatch.setattr(implementation, "_build", invalid_count)
    with pytest.raises(ValueError, match="valid count exceeds"):
        point_cloud_from_depth(depth, _camera())


def test_pinned_pool_alternates_when_previous_result_is_retained():
    import gc

    depth = np.ones((17, 19), np.float32)
    first = point_cloud_from_depth(depth, _camera())
    first_address = first.points.ptr
    second = point_cloud_from_depth(depth * 2, _camera())
    assert second.points.ptr != first_address
    del first
    gc.collect()
    third = point_cloud_from_depth(depth * 3, _camera())
    assert third.points.ptr == first_address
    np.testing.assert_array_equal(second.points[:, 2], np.full(17 * 19, 2, np.float32))


@pytest.mark.parametrize("width", [1, 7, 8, 9, 15, 16, 17, 257])
def test_group_compaction_partial_groups_and_sparse_order(width):
    rng = np.random.default_rng(width)
    depth = rng.uniform(1, 20, (3, width)).astype(np.float32)
    depth.reshape(-1)[::3] = np.nan
    depth.reshape(-1)[1::5] = 0
    ids = np.arange(depth.size, dtype=np.uint64).reshape(depth.shape) + 2**40
    expected = point_cloud_from_depth(depth, _camera(), semantic_ids=ids, device="cpu")
    result = _cpu_cloud(
        point_cloud_from_depth(depth, _camera(), semantic_ids=ids, output_device="cuda")
    )
    np.testing.assert_array_equal(result.points, expected.points)
    np.testing.assert_array_equal(result.semantic_ids, expected.semantic_ids)


def test_gpu_output_pool_retained_views_recycle_and_eviction():
    import gc
    from ovannotators.writers import _pointcloud_gpu as implementation

    implementation._output_pool.clear()
    depth = np.ones((17, 19), np.float32)
    first = point_cloud_from_depth(
        depth, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    address = first.points.ptr
    view = first.points.flatten()[::3]
    del first
    gc.collect()
    second = point_cloud_from_depth(
        depth * 2, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    assert second.points.ptr != address
    saved = view.numpy().copy()
    # Eviction cannot free a live wrapper or its subview.
    implementation._output_pool.clear()
    point_cloud_from_depth(
        depth * 3, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    np.testing.assert_array_equal(view.numpy(), saved)
    del view, second
    gc.collect()
    implementation._output_pool.clear()
    first = point_cloud_from_depth(
        depth, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    address = first.points.ptr
    second = point_cloud_from_depth(
        depth * 2, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    del first
    gc.collect()
    third = point_cloud_from_depth(
        depth * 3, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    assert third.points.ptr == address
    np.testing.assert_array_equal(
        _cpu_cloud(second).points[:, 2], np.full(17 * 19, 2, np.float32)
    )
    assert (
        sum(v[1] for v in implementation._output_pool.values())
        <= implementation._output_limit
    )


def test_gpu_output_pool_does_not_reuse_until_consumer_stream_finishes(monkeypatch):
    import gc

    from unittest.mock import Mock

    synchronize = Mock(wraps=wp.synchronize_device)
    monkeypatch.setattr(wp, "synchronize_device", synchronize)
    depth = np.ones((128, 128), np.float32)
    cloud = point_cloud_from_depth(
        depth, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    address = cloud.points.ptr
    with wp.ScopedStream(wp.Stream("cuda:0")):
        cloud.points.fill_(99)
    del cloud
    gc.collect()
    # Final lease disposal fences consumers before reusing their storage.
    synchronize.assert_called_once_with(wp.get_device("cuda:0"))
    fresh = point_cloud_from_depth(
        depth * 3, _camera(), output_device="cuda", reuse_gpu_outputs=True
    )
    assert fresh.points.ptr == address
    np.testing.assert_array_equal(
        _cpu_cloud(fresh).points[:, 2], np.full(128 * 128, 3, np.float32)
    )


def test_gpu_count_lease_preserves_all_evicted_channel_owners():
    import gc
    from weakref import ref
    from ovannotators.writers import _pointcloud_gpu as implementation

    implementation._output_pool.clear()
    cloud = point_cloud_from_depth(
        np.ones((17, 19), np.float32),
        _camera(),
        output_device="cuda",
        reuse_gpu_outputs=True,
    )
    owner = ref(next(iter(implementation._output_pool.values()))[0]["points"])
    retained_count = cloud.count
    del cloud
    implementation._output_pool.clear()
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(retained_count.numpy(), [17 * 19])
    del retained_count
    gc.collect()
    assert owner() is None


def test_group_index_limit_is_rejected_before_allocation_or_launch():
    # A non-owning metadata-only array avoids allocating an enormous image.
    # Group/sentinel arithmetic would overflow int32 at this pixel count.
    depth = wp.array(
        ptr=1, shape=(1, 2**31 - 3, 1), dtype=wp.float32, device="cuda:0", copy=False
    )
    with pytest.raises(ValueError, match=r"at most 2\*\*31-4 pixels"):
        point_cloud_from_depth(
            depth, _camera(), output_device="cuda", reuse_gpu_outputs=True
        )


def test_resident_default_has_no_output_pool_lease_and_opt_in_validation():
    from ovannotators.writers import _pointcloud_gpu as implementation

    implementation._output_pool.clear()
    cloud = point_cloud_from_depth(
        np.ones((3, 5), np.float32), _camera(), output_device="cuda"
    )
    assert not implementation._output_pool
    np.testing.assert_array_equal(
        _cpu_cloud(cloud).points[:, 2], np.ones(15, np.float32)
    )
    with pytest.raises(ValueError, match="must be a bool"):
        point_cloud_from_depth(
            np.ones((3, 5), np.float32), _camera(), reuse_gpu_outputs="yes"
        )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_cpu_delivery_releases_temporary_outputs_without_device_fence(
    monkeypatch, batch_size
):
    """CPU delivery keeps NumPy views valid without waiting on other streams."""
    from unittest.mock import Mock

    synchronize = Mock(wraps=wp.synchronize_device)
    monkeypatch.setattr(wp, "synchronize_device", synchronize)
    depth = np.arange(17 * 19, dtype=np.float32).reshape(17, 19)
    depths = [depth * (i + 1) for i in range(batch_size)]
    first = point_clouds_from_depth(depths, [_camera()] * batch_size)
    retained = first[0].points[::2]
    saved = wp.clone(retained)
    second = point_clouds_from_depth([d + 2 for d in depths], [_camera()] * batch_size)
    for original, cloud in zip(depths, first):
        np.testing.assert_array_equal(cloud.points[:, 2], original[original > 0])
    for original, cloud in zip(depths, second):
        np.testing.assert_array_equal(cloud.points[:, 2], (original + 2).ravel())
    del first, second
    np.testing.assert_array_equal(retained, saved)
    synchronize.assert_not_called()
