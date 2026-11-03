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

"""Device-preserving preparation of video frames and NVENC input surfaces."""

import ctypes
import sys
from functools import lru_cache
from typing import Any

import warp as wp


@wp.kernel
def _pack(
    src: wp.array3d(dtype=Any),
    dst: wp.array3d(dtype=wp.uint8),
    scale_float: bool,
    bgra: bool,
):
    y, x, c = wp.tid()
    if c == 3:
        dst[y, x, c] = wp.uint8(255)
    else:
        channel = c
        if bgra:
            channel = 2 - c
        if src.shape[2] == 1:
            channel = 0
        if scale_float:
            dst[y, x, c] = wp.uint8(
                wp.clamp(
                    wp.float64(src[y, x, channel] * type(src[y, x, channel])(255.0)),
                    wp.float64(0.0),
                    wp.float64(255.0),
                )
            )
        else:
            dst[y, x, c] = wp.uint8(src[y, x, channel])


@wp.kernel
def _pack_gbr(src: wp.array3d(dtype=wp.uint8), dst: wp.array3d(dtype=wp.uint8)):
    y, x = wp.tid()
    height = src.shape[0]
    dst[y, x, 0] = src[y, x, 1]
    dst[y + height, x, 0] = src[y, x, 2]
    dst[y + 2 * height, x, 0] = src[y, x, 0]


def device_gbr(frame: Any) -> wp.array:
    """Pack normalized uint8 RGB into full-resolution G/B/R planes on-device.

    The kernel runs on the device's current Warp stream, so ``frame`` must
    already be ready there (see :func:`device_rgb`).

    Args:
        frame: Device RGB array with shape ``(H, W, 3)`` and uint8 samples.

    Returns:
        wp.array: Owned contiguous ``(3 * H, W, 1)`` planar surface.

    Raises:
        ValueError: If the frame is not normalized uint8 RGB.
    """
    src = as_warp(frame)
    if src.ndim != 3 or src.shape[2] != 3 or src.dtype != wp.uint8:
        raise ValueError("Lossless video requires fixed-size uint8 RGB frames")
    h, w, _ = src.shape
    result = wp.empty((3 * h, w, 1), dtype=wp.uint8, device=src.device)
    wp.launch(_pack_gbr, dim=(h, w), inputs=[src, result], device=src.device)
    return result


@wp.kernel
def _gray(src: wp.array3d(dtype=wp.uint8), dst: wp.array2d(dtype=wp.float32)):
    y, x = wp.tid()
    dst[y, x] = (
        0.2989 * float(src[y, x, 0])
        + 0.5870 * float(src[y, x, 1])
        + 0.1140 * float(src[y, x, 2])
    ) / 255.0


@wp.kernel
def _edge_rgb(src: wp.array2d(dtype=wp.uint8), dst: wp.array3d(dtype=wp.uint8)):
    y, x, c = wp.tid()
    dst[y, x, c] = src[y, x] * wp.uint8(255)


def is_device_frame(frame: Any) -> bool:
    """Identify CUDA inputs without materializing their pixels."""
    return hasattr(frame, "__cuda_array_interface__") or (
        hasattr(frame, "__dlpack_device__") and frame.__dlpack_device__()[0] == 2
    )


class _CudaArraySource:
    """Keep the producer alive and supply Warp's sequence requirement for CAI."""

    def __init__(self, frame):
        self.frame = frame

    @property
    def __cuda_array_interface__(self):
        return self.frame.__cuda_array_interface__

    def __len__(self):
        shape = self.__cuda_array_interface__["shape"]
        return shape[0] if shape else 0


@lru_cache(maxsize=1)
def _cuda_driver() -> Any:
    names = (
        ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    )
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _pointer_device_ordinal(ptr: int) -> int | None:
    """Return the CUDA ordinal that owns ``ptr``, or ``None`` if unknown."""
    driver = _cuda_driver()
    if driver is None:
        return None
    ordinal = ctypes.c_int(-1)
    # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL; CUDA_SUCCESS is 0.
    status = driver.cuPointerGetAttribute(
        ctypes.byref(ordinal), ctypes.c_int(9), ctypes.c_uint64(ptr)
    )
    return ordinal.value if status == 0 and ordinal.value >= 0 else None


def _cai_device(ptr: int) -> Any:
    """Resolve the Warp CUDA device that owns a CUDA-array-interface pointer."""
    if wp.get_cuda_device_count() <= 1:
        current = wp.get_device()
        return current if current.is_cuda else wp.get_cuda_device()
    ordinal = _pointer_device_ordinal(ptr)
    if ordinal is None:
        raise ValueError(
            "Cannot determine the CUDA device of a __cuda_array_interface__ "
            "frame; pass device= or provide a DLPack or Warp array"
        )
    return wp.get_cuda_device(ordinal)


def as_scalar_channels(frame: wp.array) -> wp.array:
    """View a vector-typed Warp image as ``(H, W, C)`` scalar channels.

    Renderers may expose per-pixel vectors (e.g. ``vec3``/``vec4`` normals) as
    a 2D array. The returned view aliases the same memory and keeps ``frame``
    alive. Scalar arrays are returned unchanged.
    """
    if not wp.types.type_is_vector(frame.dtype):
        return frame
    scalar = frame.dtype._wp_scalar_type_
    length = frame.dtype._length_
    view = wp.array(
        ptr=frame.ptr,
        dtype=scalar,
        shape=(*frame.shape, length),
        strides=(*frame.strides, wp.types.type_size_in_bytes(scalar)),
        device=frame.device,
        copy=False,
    )
    view._ref = frame
    return view


def as_warp(frame: Any, device: Any = None) -> wp.array:  # noqa: ANN401
    """Import a CUDA array on its owning device, with DLPack stream ordering.

    DLPack carries the owning device. A frame that exposes only
    ``__cuda_array_interface__`` does not, so on a multi-GPU host the owning
    device is queried from the CUDA driver for the frame's pointer.

    Args:
        frame: A Warp array, or an object supporting DLPack or the CUDA array
            interface.
        device: Owning CUDA device for a CUDA-array-interface-only frame.
            Defaults to the device the CUDA driver reports for the pointer, or
            the only CUDA device on a single-GPU host. Ignored for Warp arrays
            and DLPack frames.

    Returns:
        wp.array: A Warp array aliasing the frame's memory.

    Raises:
        TypeError: If the frame is not a Warp array and supports neither DLPack
            nor the CUDA array interface.
        ValueError: If a CUDA-array-interface-only frame's owning device
            cannot be determined on a multi-GPU host and ``device`` is omitted.
    """
    if isinstance(frame, wp.array):
        return frame
    if hasattr(frame, "__dlpack__"):
        return wp.from_dlpack(frame)
    if hasattr(frame, "__cuda_array_interface__"):
        if device is None:
            device = _cai_device(frame.__cuda_array_interface__["data"][0])
        # The Warp array keeps a reference to ``frame``, keeping it alive.
        return wp.array(data=_CudaArraySource(frame), device=device, copy=False)
    raise TypeError(
        "GPU video frame must be a Warp array or support DLPack or "
        "__cuda_array_interface__"
    )


def device_rgb(
    frame: Any,
    *,
    bgra: bool = False,
    crop_even: bool = False,
    synchronize: bool = True,
) -> wp.array:
    """Normalize and copy a frame on its device, owning the resulting pixels.

    A fresh allocation also snapshots borrowed renderer buffers for writers
    that defer encoding. RGB, RGBA, grayscale, float and strided inputs retain
    the same normalization as the CPU writer helpers.

    Args:
        frame: Device or Warp image with shape ``(H, W)`` or ``(H, W, 1/3/4)``.
        bgra: Emit NVENC's B, G, R, A byte order instead of RGB.
        crop_even: Drop a trailing odd row and column.
        synchronize: Wait for all prior work on the frame's device first. Keep
            the default for borrowed frames that another stream may still be
            producing; pass ``False`` for frames already ordered on the
            current Warp stream.

    Returns:
        wp.array: Owned ``(H, W, 3)`` RGB or ``(H, W, 4)`` BGRA uint8 array.

    Raises:
        ValueError: If the frame does not have a supported shape.
    """
    src = as_warp(frame)
    # Borrowed renderer arrays may have been produced on another CUDA stream.
    if synchronize and src.device.is_cuda:
        wp.synchronize_device(src.device)
    if src.ndim == 2:
        src = src.contiguous().reshape((*src.shape, 1))
    if src.ndim != 3 or src.shape[2] not in (1, 3, 4):
        raise ValueError(f"frame shape {src.shape} must be (H,W), (H,W,1/3/4)")
    h, w = src.shape[:2]
    if crop_even:
        h, w = h - h % 2, w - w % 2
    result = wp.empty((h, w, 4 if bgra else 3), dtype=wp.uint8, device=src.device)
    wp.launch(
        _pack,
        dim=result.shape,
        inputs=[src, result, src.dtype in (wp.float16, wp.float32, wp.float64), bgra],
        device=src.device,
    )
    return result


def snapshot_rgb(frame: Any) -> Any:
    """Own RGB pixels on the input device for delayed video encoding."""
    if is_device_frame(frame):
        result = device_rgb(frame)
        # The caller may immediately reuse the borrowed renderer allocation
        # on a different stream after write_frame returns.
        wp.synchronize_stream(wp.get_stream(result.device))
        return result
    from ovannotators.writers._helpers import to_rgb_uint8

    return to_rgb_uint8(frame).copy()


def device_gray(frame: Any) -> wp.array:
    """Compute Canny's grayscale input without a raw-frame host copy."""
    rgb = as_warp(frame)
    result = wp.empty(rgb.shape[:2], dtype=wp.float32, device=rgb.device)
    wp.launch(_gray, dim=result.shape, inputs=[rgb, result], device=rgb.device)
    return result


def device_edges_rgb(edges: wp.array) -> wp.array:
    """Expand a device Canny mask into an RGB video frame."""
    result = wp.empty((*edges.shape, 3), dtype=wp.uint8, device=edges.device)
    wp.launch(_edge_rgb, dim=result.shape, inputs=[edges, result], device=edges.device)
    return result
