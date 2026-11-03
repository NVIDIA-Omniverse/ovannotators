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

"""Stable CPU pointcloud compaction with fused backprojection and attributes."""

from typing import Any

import numpy as np
import warp as wp
from warp.utils import array_scan

from ovannotators.types import CameraParams, PointCloudResult

# Preserve separate float32 arithmetic operations for CPU/CUDA parity. These
# options apply only to this private module, not other writer kernels.
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def _count_rows(depth: wp.array2d(dtype=Any), counts: wp.array(dtype=wp.int64)):
    row = wp.tid()
    count = wp.int64(0)
    for col in range(depth.shape[1]):
        z = wp.float32(depth[row, col])
        if wp.isfinite(z) and z > 0.0:
            count += wp.int64(1)
    counts[row] = count


@wp.kernel
def _fill_rows(
    depth: wp.array2d(dtype=Any),
    colors: wp.array3d(dtype=Any),
    ids: wp.array2d(dtype=Any),
    offsets: wp.array(dtype=wp.int64),
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    channels: int,
    has_ids: bool,
    points: wp.array2d(dtype=wp.float32),
    colors_out: wp.array2d(dtype=Any),
    ids_out: wp.array(dtype=Any),
):
    row = wp.tid()
    index = offsets[row]
    for col in range(depth.shape[1]):
        z = wp.float32(depth[row, col])
        if wp.isfinite(z) and z > 0.0:
            points[index, 0] = (wp.float32(col) - cx) * z / fx
            points[index, 1] = (wp.float32(row) - cy) * z / fy
            points[index, 2] = z
            for channel in range(channels):
                colors_out[index, channel] = colors[row, col, channel]
            if has_ids:
                ids_out[index] = ids[row, col]
            index += wp.int64(1)


def _cpu_image(value: Any, ndim: int) -> wp.array:
    """Borrow numeric CPU inputs, normalizing NumPy alignment and byte order."""
    if isinstance(value, wp.array):
        result = value.to("cpu")
    else:
        array = np.asarray(value)
        try:
            dtype = wp.dtype_from_numpy(array.dtype.newbyteorder("="))
        except (TypeError, RuntimeError) as error:
            raise ValueError(
                "Pointcloud attributes require Warp-supported numeric dtypes"
            ) from error
        array = np.require(
            array, dtype=array.dtype.newbyteorder("="), requirements=["A"]
        )
        result = wp.array(array, dtype=dtype, device="cpu", copy=False)
    if ndim == 2 and result.ndim == 3:
        if result.shape[2] == 0:
            raise ValueError("Pointcloud depth and IDs require at least one channel")
        result = (
            result[:, :, 0]
            if result.size
            else wp.empty(result.shape[:2], dtype=result.dtype, device="cpu")
        )
    if result.ndim != ndim:
        raise ValueError(f"Pointcloud input must have {ndim} dimensions")
    return result


def _point_cloud_cpu(
    depth: Any,
    camera: CameraParams,
    colors: Any = None,
    semantic_ids: Any = None,
) -> PointCloudResult:
    """Compact and back-project every CPU image with Warp, in raster order."""
    depth = _cpu_image(depth, 2)
    colors = _cpu_image(colors, 3) if colors is not None else None
    semantic_ids = _cpu_image(semantic_ids, 2) if semantic_ids is not None else None
    height, width = depth.shape
    for value, label in [(colors, "colors"), (semantic_ids, "IDs")]:
        if value is not None and value.shape[:2] != depth.shape:
            raise ValueError(f"Pointcloud {label} must match the depth image shape")

    K = np.asarray(camera.K, dtype=np.float64)
    fx, fy = float(K[0, 0]) or 1.0, float(K[1, 1]) or 1.0
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if not np.isfinite([fx, fy, cx, cy]).all() or max(
        abs(fx), abs(fy), abs(cx), abs(cy)
    ) > float(np.finfo(np.float32).max):
        raise ValueError("CPU pointcloud intrinsics must be finite float32 values")

    counts = wp.zeros(height + 1, dtype=wp.int64, device="cpu")
    wp.launch(_count_rows, dim=height, inputs=[depth, counts], device="cpu")
    offsets = wp.empty_like(counts)
    array_scan(counts, offsets, inclusive=False)
    # CPU launches and scans are synchronous; only the compact output size is
    # read on the host, without converting any input or result payload to NumPy.
    size = int(offsets.numpy()[-1])
    points = wp.empty((size, 3), dtype=wp.float32, device="cpu")
    channels = colors.shape[2] if colors is not None else 0
    color_dtype = colors.dtype if colors is not None else wp.uint8
    id_dtype = semantic_ids.dtype if semantic_ids is not None else wp.uint32
    colors_out = wp.empty((size, channels), dtype=color_dtype, device="cpu")
    ids_out = wp.empty(
        size if semantic_ids is not None else 0, dtype=id_dtype, device="cpu"
    )
    if size:
        wp.launch(
            _fill_rows,
            dim=height,
            inputs=[
                depth,
                colors
                if colors is not None
                else wp.empty((0, 0, 0), dtype=color_dtype, device="cpu"),
                semantic_ids
                if semantic_ids is not None
                else wp.empty((0, 0), dtype=id_dtype, device="cpu"),
                offsets,
                fx,
                fy,
                cx,
                cy,
                channels,
                semantic_ids is not None,
                points,
                colors_out,
                ids_out,
            ],
            device="cpu",
        )
    return PointCloudResult(
        points=points,
        colors=colors_out if colors is not None else None,
        semantic_ids=ids_out if semantic_ids is not None else None,
    )
