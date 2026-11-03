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

"""Warp-accelerated segmentation shading.

"Shaded segmentation" combines a colorized segmentation map with the surface
normals so each instance keeps its semantic palette color but gains
Lambertian shading cues from the geometry. This gives Cosmos Transfer a
control signal that is both semantically labelled (per-instance color) and
geometrically grounded (shaded relief), which the plain flat-color
segmentation lacks.

The shading is a single fused Warp kernel (normalize normal -> N·L diffuse
term -> ambient mix -> modulate the palette color) so it runs on the GPU when
a CUDA device is present and on Warp's CPU device otherwise.
"""

from __future__ import annotations

import numpy as np
import warp as wp

try:  # Warp is required project-wide; init once so kernels launch on CPU/CUDA.
    wp.config.verbose = False
    wp.init()
except Exception:  # pragma: no cover - init is idempotent / already done elsewhere
    pass


def _shade_device() -> wp.context.Device:
    """Return the current default Warp device (CUDA if present, else CPU).

    Respects :func:`warp.ScopedDevice`/:func:`warp.set_device`, so multi-GPU
    callers can target a specific device without this module hard-coding one.
    """
    return wp.get_device()


@wp.kernel
def _shade_seg_kernel(
    seg_rgb: wp.array3d(dtype=wp.uint8),
    normals: wp.array3d(dtype=wp.float32),
    out_rgb: wp.array3d(dtype=wp.uint8),
    light_x: float,
    light_y: float,
    light_z: float,
    ambient: float,
):
    """Modulate a per-instance palette color by a Lambertian shading term.

    The light direction is assumed pre-normalized on the host. Pixels whose
    normal is a (near) zero vector — typically background / no-hit — are left
    unshaded (intensity ``1.0``) so the flat palette color passes through.
    """
    x, y = wp.tid()
    h = seg_rgb.shape[0]
    w = seg_rgb.shape[1]
    if x >= w or y >= h:
        return

    nx = normals[y, x, 0]
    ny = normals[y, x, 1]
    nz = normals[y, x, 2]
    length = wp.sqrt(nx * nx + ny * ny + nz * nz)

    intensity = float(1.0)
    if length > 1.0e-6:
        inv = 1.0 / length
        diffuse = (nx * light_x + ny * light_y + nz * light_z) * inv
        diffuse = wp.clamp(diffuse, 0.0, 1.0)
        intensity = ambient + (1.0 - ambient) * diffuse

    for c in range(3):
        v = float(seg_rgb[y, x, c]) * intensity
        out_rgb[y, x, c] = wp.uint8(wp.clamp(v, 0.0, 255.0))


def _normalize_light(
    light_dir: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return a unit-length light direction, defaulting to +Z if degenerate."""
    v = np.asarray(light_dir, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return (0.0, 0.0, 1.0)
    v = v / norm
    return (float(v[0]), float(v[1]), float(v[2]))


def shade_segmentation(
    seg_rgb: np.ndarray,
    normals: np.ndarray,
    light_dir: tuple[float, float, float] = (0.3, 0.5, 0.81),
    ambient: float = 0.35,
) -> np.ndarray:
    """Shade a colorized segmentation map with surface normals via Warp.

    For every pixel the per-instance palette color is multiplied by
    ``ambient + (1 - ambient) * max(N·L, 0)``, where ``N`` is the unit surface
    normal and ``L`` the (normalized) light direction. Background pixels with a
    zero-vector normal keep their flat color. The fused operation runs as a
    single Warp kernel launch on CUDA when available, otherwise on Warp's CPU
    device.

    Args:
        seg_rgb: Colorized segmentation as an ``(H, W, 3)`` ``uint8`` array.
        normals: Surface normals as an ``(H, W, C)`` float array with ``C >= 3``
            (extra channels, e.g. alpha, are ignored). Must share ``seg_rgb``'s
            spatial shape.
        light_dir: Light direction ``(x, y, z)``; normalized internally.
        ambient: Ambient term in ``[0, 1]``; the floor brightness applied where
            a surface faces fully away from the light.

    Returns:
        np.ndarray: Shaded segmentation as an ``(H, W, 3)`` ``uint8`` array.

    Raises:
        ValueError: If ``seg_rgb`` is not ``(H, W, 3)``, ``normals`` has fewer
            than 3 channels, or the two inputs disagree on ``(H, W)``.
    """
    seg = np.ascontiguousarray(seg_rgb)
    if seg.ndim != 3 or seg.shape[2] < 3:
        raise ValueError(f"seg_rgb must be (H, W, 3); got shape {tuple(seg.shape)}")
    seg = seg[..., :3].astype(np.uint8, copy=False)

    n = np.asarray(normals)
    if n.ndim != 3 or n.shape[2] < 3:
        raise ValueError(f"normals must be (H, W, C>=3); got shape {tuple(n.shape)}")
    if n.shape[:2] != seg.shape[:2]:
        raise ValueError(
            "seg_rgb and normals must share (H, W); got "
            f"{seg.shape[:2]} vs {n.shape[:2]}"
        )

    h, w = seg.shape[:2]
    light = _normalize_light(light_dir)
    ambient = float(np.clip(ambient, 0.0, 1.0))
    normals_f32 = np.ascontiguousarray(n[..., :3], dtype=np.float32)

    device = _shade_device()
    seg_wp = wp.array(seg, dtype=wp.uint8, shape=(h, w, 3), device=device)
    normals_wp = wp.array(normals_f32, dtype=wp.float32, shape=(h, w, 3), device=device)
    out_wp = wp.empty((h, w, 3), dtype=wp.uint8, device=device)
    wp.launch(
        _shade_seg_kernel,
        dim=(w, h),
        inputs=[
            seg_wp,
            normals_wp,
            out_wp,
            light[0],
            light[1],
            light[2],
            ambient,
        ],
        device=device,
    )
    return out_wp.numpy()
