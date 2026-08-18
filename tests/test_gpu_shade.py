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

"""Tests for Warp-accelerated segmentation shading (``_gpu_shade``)."""

from __future__ import annotations

import numpy as np
import pytest

from ovannotators.writers._gpu_shade import shade_segmentation


def _seg_color(h: int, w: int) -> np.ndarray:
    """Flat colored segmentation block (single instance) for shading tests."""
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    rgb[..., 1] = 120
    rgb[..., 2] = 40
    return rgb


class TestShadeSegmentation:
    def test_output_shape_and_dtype(self):
        seg = _seg_color(48, 64)
        normals = np.zeros((48, 64, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        out = shade_segmentation(seg, normals)
        assert out.shape == (48, 64, 3)
        assert out.dtype == np.uint8

    def test_facing_light_brighter_than_facing_away(self):
        seg = _seg_color(16, 16)
        toward = np.zeros((16, 16, 3), dtype=np.float32)
        toward[..., 2] = 1.0
        away = np.zeros((16, 16, 3), dtype=np.float32)
        away[..., 2] = -1.0
        lit = shade_segmentation(seg, toward, light_dir=(0, 0, 1), ambient=0.3)
        unlit = shade_segmentation(seg, away, light_dir=(0, 0, 1), ambient=0.3)
        assert lit.astype(int).sum() > unlit.astype(int).sum()

    def test_full_light_preserves_color(self):
        seg = _seg_color(8, 8)
        normals = np.zeros((8, 8, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        out = shade_segmentation(seg, normals, light_dir=(0, 0, 1), ambient=0.0)
        # Diffuse == 1 everywhere → palette color passes through unchanged.
        np.testing.assert_array_equal(out, seg)

    def test_ambient_floor_for_backfaces(self):
        seg = _seg_color(8, 8)
        away = np.zeros((8, 8, 3), dtype=np.float32)
        away[..., 2] = -1.0
        out = shade_segmentation(seg, away, light_dir=(0, 0, 1), ambient=0.5)
        expected = (seg.astype(np.float64) * 0.5).astype(np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_zero_normals_unshaded(self):
        seg = _seg_color(8, 8)
        normals = np.zeros((8, 8, 3), dtype=np.float32)  # background / no-hit
        out = shade_segmentation(seg, normals, ambient=0.2)
        # Zero-length normals leave the flat color unchanged (intensity 1.0).
        np.testing.assert_array_equal(out, seg)

    def test_extra_normal_channels_ignored(self):
        seg = _seg_color(8, 8)
        normals = np.zeros((8, 8, 4), dtype=np.float32)
        normals[..., 2] = 1.0
        normals[..., 3] = 999.0  # spurious alpha channel
        out = shade_segmentation(seg, normals, light_dir=(0, 0, 1), ambient=0.0)
        np.testing.assert_array_equal(out, seg)

    def test_shape_mismatch_raises(self):
        seg = _seg_color(8, 8)
        normals = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="share"):
            shade_segmentation(seg, normals)

    def test_bad_normals_channels_raise(self):
        seg = _seg_color(8, 8)
        normals = np.zeros((8, 8, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="normals"):
            shade_segmentation(seg, normals)

    def test_bad_seg_shape_raises(self):
        seg = np.zeros((8, 8), dtype=np.uint8)
        normals = np.zeros((8, 8, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="seg_rgb"):
            shade_segmentation(seg, normals)


class TestNormalizeLight:
    def test_degenerate_light_falls_back_to_plus_z(self):
        """A zero-length light direction defaults to +Z instead of dividing by 0."""
        from ovannotators.writers._gpu_shade import _normalize_light

        assert _normalize_light((0.0, 0.0, 0.0)) == (0.0, 0.0, 1.0)
