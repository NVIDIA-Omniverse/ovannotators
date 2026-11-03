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

"""CUDA backprojection with stable pixel compaction and synchronous CPU delivery."""

from collections import OrderedDict
from itertools import count
from threading import RLock
from typing import Any
from weakref import finalize

import numpy as np
import warp as wp
from warp.utils import array_scan

from ovannotators.types import PointCloudResult

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


_PIXELS_PER_GROUP = wp.constant(4)


@wp.kernel
def _valid_groups(
    depth: wp.array3d(dtype=Any),
    flags: wp.array(dtype=wp.int32),
    width: int,
    pixels: int,
):
    group = wp.tid()
    valid = int(0)
    for local in range(_PIXELS_PER_GROUP):
        pixel = group * _PIXELS_PER_GROUP + local
        if pixel < pixels:
            z = wp.float32(depth[pixel // width, pixel % width, 0])
            valid += int(wp.isfinite(z) and z > 0.0)
    flags[group] = valid


@wp.kernel
def _fill_groups(
    depth: wp.array3d(dtype=Any),
    colors: wp.array3d(dtype=Any),
    ids: wp.array3d(dtype=Any),
    offsets: wp.array(dtype=wp.int32),
    width: int,
    pixels: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    channels: int,
    has_ids: bool,
    points: wp.array2d(dtype=wp.float32),
    colors_out: wp.array2d(dtype=Any),
    ids_out: wp.array(dtype=Any),
    count_out: wp.array(dtype=wp.int32),
):
    group = wp.tid()
    if group == 0:
        count_out[0] = offsets[offsets.shape[0] - 1]
    index = offsets[group]
    for local in range(_PIXELS_PER_GROUP):
        pixel = group * _PIXELS_PER_GROUP + local
        if pixel < pixels:
            row = pixel // width
            col = pixel % width
            z = wp.float32(depth[row, col, 0])
            if wp.isfinite(z) and z > 0.0:
                points[index, 0] = (wp.float32(col) - cx) * z / fx
                points[index, 1] = (wp.float32(row) - cy) * z / fy
                points[index, 2] = z
                for channel in range(channels):
                    colors_out[index, channel] = colors[row, col, channel]
                if has_ids:
                    ids_out[index] = ids[row, col, 0]
                index += 1


def _image(value: Any, device: Any) -> wp.array:
    if isinstance(value, wp.array):
        result = value if value.device == device else value.to(device)
    else:
        array = np.asarray(value)
        result = wp.array(array, dtype=wp.dtype_from_numpy(array.dtype), device=device)
    if result.ndim == 2:
        result = result.reshape((*result.shape, 1))
    if result.ndim != 3:
        raise ValueError("GPU pointcloud inputs must have two or three dimensions")
    return result


# Scratch owns no returned arrays. Calls serialize access until queued GPU work
# completes, so another thread/stream cannot overwrite a pending scan or readback.
_scratch_lock = RLock()
_scratch = OrderedDict()
_scratch_limit = 192 * 1024**2
_scratch_generation = count()


def _buffers(target, slot, pixels, channels, color_dtype, id_dtype):
    key = (target.alias, slot, pixels, channels, color_dtype, id_dtype)
    for cached_key in reversed(_scratch):
        if cached_key[:-1] == key and not _scratch[cached_key][0]["lease"]["count"]:
            _scratch.move_to_end(cached_key)
            return _scratch[cached_key][0]
    # Retained Warp views pin their host storage. Never overwrite a live result.
    # Keep busy entries: ordinary `cloud = helper(...)` retains the previous
    # cloud during the next call, so two staging sets alternate after warmup.
    buffers = {
        "flags": wp.empty(
            (pixels + int(_PIXELS_PER_GROUP) - 1) // int(_PIXELS_PER_GROUP) + 1,
            dtype=wp.int32,
            device=target,
        ),
        "offsets": wp.empty(
            (pixels + int(_PIXELS_PER_GROUP) - 1) // int(_PIXELS_PER_GROUP) + 1,
            dtype=wp.int32,
            device=target,
        ),
        "count": wp.empty(1, dtype=wp.int32, device="cpu", pinned=True),
        "points": wp.empty((pixels, 3), dtype=wp.float32, device="cpu", pinned=True),
        "colors": wp.empty(
            (pixels, channels), dtype=color_dtype, device="cpu", pinned=True
        ),
        "ids": wp.empty(pixels, dtype=id_dtype, device="cpu", pinned=True),
    }
    size = sum(b.capacity for b in buffers.values())
    while _scratch and sum(v[1] for v in _scratch.values()) + size > _scratch_limit:
        _scratch.popitem(last=False)
    if size <= _scratch_limit:
        _scratch[(*key, next(_scratch_generation))] = (buffers, size)
    buffers["slot"] = slot
    buffers["lease"] = {"count": 0}
    return buffers


_output_pool = OrderedDict()
_output_limit = 128 * 1024**2
_output_generation = count()


def _output_buffers(target, slot, pixels, channels, color_dtype, id_dtype, lease):
    key = (target.alias, slot, pixels, channels, color_dtype, id_dtype)
    for cached_key in reversed(_output_pool):
        buffers, _ = _output_pool[cached_key]
        if cached_key[:-1] == key and not buffers["lease"]["count"]:
            _output_pool.move_to_end(cached_key)
            return _bind_output_lease(buffers, lease)
    buffers = {
        "points": wp.empty((pixels, 3), dtype=wp.float32, device=target),
        "colors": wp.empty((pixels, channels), dtype=color_dtype, device=target),
        "ids": wp.empty(pixels, dtype=id_dtype, device=target),
        "count": wp.empty(1, dtype=wp.int32, device=target),
    }
    size = sum(b.capacity for b in buffers.values())
    while (
        _output_pool and sum(v[1] for v in _output_pool.values()) + size > _output_limit
    ):
        _output_pool.popitem(last=False)
    if size <= _output_limit:
        _output_pool[(*key, next(_output_generation))] = (buffers, size)
    return _bind_output_lease(buffers, lease)


def _bind_output_lease(buffers, lease):
    # One batch lease holds ALL owning arrays, not just the retained channel.
    # Evicting a busy cache entry cannot free another channel's pending storage.
    buffers["lease"] = lease
    lease["owners"] += tuple(
        buffers[name] for name in ["points", "colors", "ids", "count"]
    )
    return buffers


def _release_gpu(lease):
    with _scratch_lock:
        # Caller-exposed arrays may have been used on another stream. CPU
        # delivery only uses the already-synchronized construction stream.
        if lease["count"] == 1:
            if lease["fence"]:
                wp.synchronize_device(lease["device"])
            lease["owners"] = ()
        lease["count"] -= 1


def _leased_gpu_array(buffers, name):
    owner = buffers[name]
    result = wp.array(
        ptr=owner.ptr,
        shape=owner.shape,
        dtype=owner.dtype,
        device=owner.device,
        strides=owner.strides,
        copy=False,
    )
    # Warp reshape/slice views retain this wrapper; it retains the owning array
    # even if the bounded pool evicts a busy entry.
    result._ref = owner
    buffers["lease"]["count"] += 1
    finalize(result, _release_gpu, buffers["lease"])
    return result


def _release_host(lease):
    with _scratch_lock:
        lease["count"] -= 1


def _owned_prefix(buffers, name, count):
    # Warp subviews retain this wrapper and its owning pinned allocation.
    # Release the staging lease only when every retained subview is gone.
    owner = buffers[name]
    base = wp.array(
        ptr=owner.ptr,
        shape=(count, *owner.shape[1:]),
        dtype=owner.dtype,
        device=owner.device,
        strides=owner.strides,
        copy=False,
    )
    base._ref = owner
    buffers["lease"]["count"] += 1
    finalize(base, _release_host, buffers["lease"])
    return base


def _build(depth, camera, colors, semantic_ids, target, buffers, output_lease):
    K = np.asarray(camera.K, dtype=np.float64)
    fx, fy = float(K[0, 0]) or 1.0, float(K[1, 1]) or 1.0
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if (
        not np.isfinite([fx, fy, cx, cy]).all()
        or max(abs(fx), abs(fy), abs(cx), abs(cy)) > np.finfo(np.float32).max
    ):
        raise ValueError("GPU pointcloud intrinsics must be finite float32 values")
    height, width = depth.shape[:2]
    if depth.shape[2] == 0 or (semantic_ids is not None and semantic_ids.shape[2] == 0):
        raise ValueError("Pointcloud depth and IDs require at least one channel")
    pixels = height * width
    if pixels > np.iinfo(np.int32).max - int(_PIXELS_PER_GROUP) + 1:
        raise ValueError("GPU grouped pointcloud requires at most 2**31-4 pixels")
    channels = colors.shape[2] if colors is not None else 0
    color_input = (
        colors
        if colors is not None
        else wp.empty((0, 0, 0), dtype=wp.uint8, device=target)
    )
    id_input = (
        semantic_ids
        if semantic_ids is not None
        else wp.empty((0, 0, 0), dtype=wp.uint32, device=target)
    )
    for value, label in [(colors, "colors"), (semantic_ids, "IDs")]:
        if value is not None and value.shape[:2] != (height, width):
            raise ValueError(f"Pointcloud {label} must match the depth image shape")
    flags, offsets = buffers["flags"], buffers["offsets"]
    wp.launch(
        _valid_groups,
        dim=flags.shape[0],
        inputs=[depth, flags, width, pixels],
        device=target,
    )
    array_scan(flags, offsets, inclusive=False)
    if output_lease["pooled"]:
        outputs = _output_buffers(
            target,
            buffers["slot"],
            pixels,
            channels,
            color_input.dtype,
            id_input.dtype,
            output_lease,
        )
        points = _leased_gpu_array(outputs, "points")
        color_output = (
            _leased_gpu_array(outputs, "colors") if colors is not None else None
        )
        id_output = (
            _leased_gpu_array(outputs, "ids") if semantic_ids is not None else None
        )
        count = _leased_gpu_array(outputs, "count")
    else:
        # Independently owned resident outputs keep Warp's async allocator path.
        # Conservative pool recycling fences can cost more than fresh allocation.
        points = wp.empty((pixels, 3), dtype=wp.float32, device=target)
        color_output = (
            wp.empty((pixels, channels), dtype=color_input.dtype, device=target)
            if colors is not None
            else None
        )
        id_output = (
            wp.empty(pixels, dtype=id_input.dtype, device=target)
            if semantic_ids is not None
            else None
        )
        count = wp.empty(1, dtype=wp.int32, device=target)
        outputs = {
            "colors": color_output
            if color_output is not None
            else wp.empty((0, 0), dtype=color_input.dtype, device=target),
            "ids": id_output
            if id_output is not None
            else wp.empty(0, dtype=id_input.dtype, device=target),
        }
    # The fill kernel also publishes the count, avoiding a separate tiny D2D copy.
    wp.launch(
        _fill_groups,
        dim=max(1, flags.shape[0] - 1),
        inputs=[
            depth,
            color_input,
            id_input,
            offsets,
            width,
            pixels,
            fx,
            fy,
            cx,
            cy,
            channels,
            semantic_ids is not None,
            points,
            color_output if color_output is not None else outputs["colors"],
            id_output if id_output is not None else outputs["ids"],
            count,
        ],
        device=target,
    )
    return PointCloudResult(
        points=points, colors=color_output, semantic_ids=id_output, count=count
    )


def _deliver(results, buffers, target):
    for cloud, scratch in zip(results, buffers):
        wp.copy(scratch["count"], cloud.count)
    wp.synchronize_stream(wp.get_stream(target))
    counts = [int(s["count"].numpy()[0]) for s in buffers]
    if any(n < 0 or n > c.points.shape[0] for c, n in zip(results, counts)):
        raise ValueError("GPU pointcloud valid count exceeds its capacity")
    for cloud, scratch, size in zip(results, buffers, counts):
        for name in ["points", "colors", "semantic_ids"]:
            value = getattr(cloud, name)
            if value is not None and size:
                host_name = "ids" if name == "semantic_ids" else name
                components = value.shape[1] if value.ndim == 2 else 1
                wp.copy(
                    scratch[host_name].flatten(),
                    value.flatten(),
                    count=size * components,
                )
    wp.synchronize_stream(wp.get_stream(target))
    # Lease pinned storage until every returned Warp array/subview is released. This
    # avoids a second host copy while preserving independent output ownership.
    return [
        PointCloudResult(
            points=_owned_prefix(s, "points", n),
            colors=_owned_prefix(s, "colors", n) if c.colors is not None else None,
            semantic_ids=_owned_prefix(s, "ids", n)
            if c.semantic_ids is not None
            else None,
        )
        for c, s, n in zip(results, buffers, counts)
    ]


def _point_clouds_gpu(
    depths,
    cameras,
    colors,
    semantic_ids,
    device,
    output_device,
    reuse_gpu_outputs=False,
):
    target = wp.get_device(device)
    if not target.is_cuda:
        raise ValueError("GPU pointcloud construction requires a CUDA device")
    with _scratch_lock, wp.ScopedDevice(target):
        buffers, results, input_leases = [], [], []
        output_lease = {
            "count": 0,
            "device": target,
            "owners": (),
            "pooled": output_device == "cpu" or reuse_gpu_outputs,
            # CPU delivery and error cleanup synchronize the only stream that
            # uses these internal arrays before releasing their lease.
            "fence": output_device == "cuda",
        }
        try:
            for i, (depth, camera, color, ids) in enumerate(
                zip(depths, cameras, colors, semantic_ids)
            ):
                depth = _image(depth, target)
                color = _image(color, target) if color is not None else None
                ids = _image(ids, target) if ids is not None else None
                # Uploaded temporary arrays must outlive all queued camera kernels.
                input_leases.append((depth, color, ids))
                pixels = depth.shape[0] * depth.shape[1]
                if pixels > np.iinfo(np.int32).max - int(_PIXELS_PER_GROUP) + 1:
                    raise ValueError(
                        "GPU grouped pointcloud requires at most 2**31-4 pixels"
                    )
                scratch = _buffers(
                    target,
                    i,
                    pixels,
                    color.shape[2] if color is not None else 0,
                    color.dtype if color is not None else wp.uint8,
                    ids.dtype if ids is not None else wp.uint32,
                )
                buffers.append(scratch)
                results.append(
                    _build(depth, camera, color, ids, target, scratch, output_lease)
                )
            if output_device == "cpu":
                return _deliver(results, buffers, target)
            # Ready-on-return avoids borrowing input/frame lifetimes beyond this call.
            wp.synchronize_stream(wp.get_stream(target))
            return results
        except BaseException:
            # Do not unlock scratch while failed/partially submitted GPU work uses it.
            wp.synchronize_stream(wp.get_stream(target))
            raise


def _point_cloud_gpu(
    depth,
    camera,
    colors,
    semantic_ids,
    device,
    output_device="cpu",
    reuse_gpu_outputs=False,
):
    return _point_clouds_gpu(
        [depth],
        [camera],
        [colors],
        [semantic_ids],
        device,
        output_device,
        reuse_gpu_outputs,
    )[0]
