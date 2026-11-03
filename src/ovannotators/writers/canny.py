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

"""Canny edge detection using NVIDIA Warp.

This module provides a Warp-based Canny implementation for the Cosmos
Transfer writer. The kernels run on whatever Warp device the data lives on —
CUDA when a GPU is present, otherwise Warp's CPU device — so no separate
host-side implementation is needed.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.config.verbose = False
wp.init()


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _gaussian_blur_x(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    kernel: wp.array(dtype=wp.float32),
    kradius: int,
):
    """Separable Gaussian blur — horizontal pass."""
    x, y = wp.tid()
    h = src.shape[0]
    w = src.shape[1]
    if x >= w or y >= h:
        return
    acc = float(0.0)
    for k in range(-kradius, kradius + 1):
        sx = wp.clamp(x + k, 0, w - 1)
        acc += src[y, sx] * kernel[k + kradius]
    dst[y, x] = acc


@wp.kernel
def _gaussian_blur_y(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    kernel: wp.array(dtype=wp.float32),
    kradius: int,
):
    """Separable Gaussian blur — vertical pass."""
    x, y = wp.tid()
    h = src.shape[0]
    w = src.shape[1]
    if x >= w or y >= h:
        return
    acc = float(0.0)
    for k in range(-kradius, kradius + 1):
        sy = wp.clamp(y + k, 0, h - 1)
        acc += src[sy, x] * kernel[k + kradius]
    dst[y, x] = acc


@wp.kernel
def _sobel_gradients(
    blurred: wp.array2d(dtype=wp.float32),
    mag: wp.array2d(dtype=wp.float32),
    angle: wp.array2d(dtype=wp.float32),
):
    """Compute gradient magnitude and angle via Sobel operators."""
    x, y = wp.tid()
    h = blurred.shape[0]
    w = blurred.shape[1]
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return

    # Sobel X
    gx = (
        -blurred[y - 1, x - 1]
        + blurred[y - 1, x + 1]
        - 2.0 * blurred[y, x - 1]
        + 2.0 * blurred[y, x + 1]
        - blurred[y + 1, x - 1]
        + blurred[y + 1, x + 1]
    )
    # Sobel Y
    gy = (
        -blurred[y - 1, x - 1]
        - 2.0 * blurred[y - 1, x]
        - blurred[y - 1, x + 1]
        + blurred[y + 1, x - 1]
        + 2.0 * blurred[y + 1, x]
        + blurred[y + 1, x + 1]
    )

    mag[y, x] = wp.sqrt(gx * gx + gy * gy)
    angle[y, x] = wp.atan2(gy, gx)


@wp.kernel
def _non_maximum_suppression(
    mag: wp.array2d(dtype=wp.float32),
    angle: wp.array2d(dtype=wp.float32),
    suppressed: wp.array2d(dtype=wp.float32),
):
    """Suppress non-maximum gradient pixels along edge direction."""
    x, y = wp.tid()
    h = mag.shape[0]
    w = mag.shape[1]
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return

    m = mag[y, x]
    a = angle[y, x]

    # Convert to degrees [0, 180)
    deg = a * 57.29577951
    if deg < 0.0:
        deg = deg + 180.0

    n1 = float(0.0)
    n2 = float(0.0)

    if (deg >= 0.0 and deg < 22.5) or (deg >= 157.5 and deg <= 180.0):
        n1 = mag[y, x - 1]
        n2 = mag[y, x + 1]
    elif deg >= 22.5 and deg < 67.5:
        n1 = mag[y - 1, x + 1]
        n2 = mag[y + 1, x - 1]
    elif deg >= 67.5 and deg < 112.5:
        n1 = mag[y - 1, x]
        n2 = mag[y + 1, x]
    else:
        n1 = mag[y - 1, x - 1]
        n2 = mag[y + 1, x + 1]

    if m >= n1 and m >= n2:
        suppressed[y, x] = m
    else:
        suppressed[y, x] = 0.0


@wp.kernel
def _double_threshold(
    suppressed: wp.array2d(dtype=wp.float32),
    strong: wp.array2d(dtype=wp.uint8),
    weak: wp.array2d(dtype=wp.uint8),
    edges: wp.array2d(dtype=wp.uint8),
    high_val: float,
    low_val: float,
):
    """Classify pixels as strong, weak, or suppressed."""
    x, y = wp.tid()
    h = suppressed.shape[0]
    w = suppressed.shape[1]
    if x >= w or y >= h:
        return

    val = suppressed[y, x]
    if val >= high_val:
        strong[y, x] = wp.uint8(1)
        edges[y, x] = wp.uint8(1)
    elif val >= low_val:
        weak[y, x] = wp.uint8(1)
    # else: both remain 0


@wp.kernel
def _hysteresis_iteration(
    strong: wp.array2d(dtype=wp.uint8),
    edges: wp.array2d(dtype=wp.uint8),
    weak: wp.array2d(dtype=wp.uint8),
    changed: wp.array(dtype=wp.int32),
):
    """Single hysteresis pass — promote weak pixels adjacent to edges."""
    x, y = wp.tid()
    h = edges.shape[0]
    w = edges.shape[1]
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return

    if weak[y, x] == wp.uint8(0) or edges[y, x] == wp.uint8(1):
        return

    # Check 8-connected neighbors
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            if edges[y + dy, x + dx] == wp.uint8(1):
                edges[y, x] = wp.uint8(1)
                wp.atomic_add(changed, 0, 1)
                return


@wp.kernel
def _reduce_max(
    values: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    """Device-side maximum reduction into ``out[0]`` (avoids a full D2H copy)."""
    x, y = wp.tid()
    h = values.shape[0]
    w = values.shape[1]
    if x >= w or y >= h:
        return
    wp.atomic_max(out, 0, values[y, x])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def canny(
    gray: np.ndarray | wp.array,
    sigma: float = 1.0,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
) -> np.ndarray | wp.array:
    """Canny edge detection using Warp.

    Accepts either a NumPy array or a Warp ``float32`` array. A Warp array is
    processed on its existing device with no host round-trip, and the result is
    returned on that same device. A NumPy array is processed on Warp's current
    default device (CUDA when present, otherwise CPU; override with
    :func:`warp.ScopedDevice` for multi-GPU control).

    Args:
        gray: 2D float image (H, W) normalized to [0, 1]. Either a NumPy array
            (processed on the current Warp device) or a 2D ``float32`` Warp
            array (processed on its current device).
        sigma: Gaussian blur sigma.
        low_threshold: Low hysteresis threshold (relative to max gradient).
        high_threshold: High hysteresis threshold (relative to max gradient).

    Returns:
        Edge mask matching the input backend: a boolean 2D NumPy array for
        NumPy input, or a ``uint8`` (0/1) 2D Warp array on the input's device
        for Warp input.

    Raises:
        TypeError: If a Warp array input is not ``float32``.
        ValueError: If the input is not 2D.
    """
    input_is_warp = isinstance(gray, wp.array)
    if input_is_warp:
        if gray.ndim != 2:
            raise ValueError(f"canny expects a 2D array, got shape {tuple(gray.shape)}")
        if gray.dtype != wp.float32:
            raise TypeError(
                "canny Warp input must be float32 (got "
                f"{gray.dtype}); cast it to float32 before calling."
            )
        device = gray.device
        h, w = gray.shape
        # Use the caller's array directly — stays on its device, no host copy.
        gray_gpu = gray
    else:
        gray_np = np.ascontiguousarray(gray, dtype=np.float32)
        if gray_np.ndim != 2:
            raise ValueError(f"canny expects a 2D array, got shape {gray_np.shape}")
        h, w = gray_np.shape
        device = wp.get_device()
        gray_gpu = wp.array(gray_np, dtype=wp.float32, shape=(h, w), device=device)

    # 1. Gaussian blur (separable)
    kradius = int(np.ceil(sigma * 3))
    ksize = kradius * 2 + 1
    x = np.arange(ksize, dtype=np.float32) - kradius
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_wp = wp.array(kernel_1d, dtype=wp.float32, device=device)

    temp = wp.zeros((h, w), dtype=wp.float32, device=device)
    blurred = wp.zeros((h, w), dtype=wp.float32, device=device)

    wp.launch(
        _gaussian_blur_x,
        dim=(w, h),
        inputs=[gray_gpu, temp, kernel_wp, kradius],
        device=device,
    )
    wp.launch(
        _gaussian_blur_y,
        dim=(w, h),
        inputs=[temp, blurred, kernel_wp, kradius],
        device=device,
    )

    # 2. Sobel gradients
    mag = wp.zeros((h, w), dtype=wp.float32, device=device)
    angle = wp.zeros((h, w), dtype=wp.float32, device=device)
    wp.launch(_sobel_gradients, dim=(w, h), inputs=[blurred, mag, angle], device=device)

    # 3. Non-maximum suppression
    suppressed = wp.zeros((h, w), dtype=wp.float32, device=device)
    wp.launch(
        _non_maximum_suppression,
        dim=(w, h),
        inputs=[mag, angle, suppressed],
        device=device,
    )

    # 4. Double threshold — reduce the max on-device to avoid a full D2H copy.
    mag_max_wp = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(_reduce_max, dim=(w, h), inputs=[mag, mag_max_wp], device=device)
    wp.synchronize_device(device)
    mag_max = float(mag_max_wp.numpy()[0])
    if mag_max < 1e-5:
        empty = wp.zeros((h, w), dtype=wp.uint8, device=device)
        return empty if input_is_warp else np.zeros((h, w), dtype=bool)

    high_val = high_threshold * mag_max
    low_val = low_threshold * mag_max

    strong = wp.zeros((h, w), dtype=wp.uint8, device=device)
    weak = wp.zeros((h, w), dtype=wp.uint8, device=device)
    edges = wp.zeros((h, w), dtype=wp.uint8, device=device)

    wp.launch(
        _double_threshold,
        dim=(w, h),
        inputs=[suppressed, strong, weak, edges, high_val, low_val],
        device=device,
    )

    # 5. Iterative hysteresis
    changed = wp.array([1], dtype=wp.int32, device=device)
    max_iterations = 100

    for _ in range(max_iterations):
        changed.zero_()
        wp.launch(
            _hysteresis_iteration,
            dim=(w, h),
            inputs=[strong, edges, weak, changed],
            device=device,
        )
        wp.synchronize_device(device)
        if changed.numpy()[0] == 0:
            break

    # Preserve the input backend/device: Warp in → Warp (uint8 0/1) out,
    # NumPy in → boolean NumPy out.
    if input_is_warp:
        return edges
    return edges.numpy().astype(bool)
