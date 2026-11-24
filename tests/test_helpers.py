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

"""Comprehensive tests for ovannotators/writers/_helpers.py.

All bounding-box / camera helpers operate on the strict ovrtx payload dtypes
(:data:`OVRTX_BBOX2D_DTYPE`, :data:`OVRTX_BBOX3D_DTYPE`,
:data:`OVRTX_CAMERA_PARAMS_DTYPE`); legacy float-array fallbacks are no longer
supported.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from ovannotators.types import CameraParams, PointCloudResult, Resolution
from ovannotators.writers._helpers import (
    OVRTX_BBOX2D_DTYPE,
    OVRTX_BBOX3D_DTYPE,
    OVRTX_CAMERA_PARAMS_DTYPE,
    OVRTX_INSTANCE_MAP_DTYPE,
    OVRTX_INSTANCE_TOKEN_DTYPE,
    OVRTX_OCCLUSION_DTYPE,
    OVRTX_STABLE_SEMANTIC_ID_DTYPE,
    decode_bbox_2d,
    decode_bbox_3d,
    decode_camera_params,
    decode_instance_map,
    decode_instance_token_map,
    decode_occlusion,
    decode_semantic_id_map,
    decode_stable_id_semantic_id_map,
    extract_camera_params,
    id_label_map_from_render_vars,
    id_prim_path_map_from_render_vars,
    instance_masks,
    iter_bboxes_2d,
    iter_bboxes_3d,
    mask_to_polygon,
    mask_to_rle,
    occlusion_by_stable_id,
    point_cloud_from_depth,
    pose_in_opencv_camera,
    rotation_and_scale,
    rotation_without_scale,
    to_rgb_uint8,
    world_to_opencv_camera,
)


def _id128(value: int) -> tuple[int, int, int, int]:
    return (int(value), 0, 0, 0)


def _make_id_map(labels: dict[int, str]) -> np.ndarray:
    entry_dtype = np.dtype(
        [("semantic_id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    num = len(labels)
    header_size = num * entry_dtype.itemsize
    items = list(labels.items())
    strings = b""
    offsets: list[int] = []
    lengths: list[int] = []
    for _sid, label in items:
        encoded = label.encode("utf-8")
        offsets.append(header_size + len(strings))
        lengths.append(len(encoded))
        strings += encoded
    entries = np.zeros(num, dtype=entry_dtype)
    for i, (sid, _label) in enumerate(items):
        entries[i]["semantic_id"] = _id128(sid)
        entries[i]["label_length"] = lengths[i]
        entries[i]["label_offset"] = offsets[i]
    raw = entries.tobytes() + strings + int(num).to_bytes(4, "little")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def test_decode_stable_id_semantic_id_map() -> None:
    raw = np.array(
        [(_id128(101), _id128(7)), (_id128(202), _id128(9))],
        dtype=OVRTX_STABLE_SEMANTIC_ID_DTYPE,
    ).view(np.uint8)

    decoded = decode_stable_id_semantic_id_map(raw)

    assert decoded.dtype == OVRTX_STABLE_SEMANTIC_ID_DTYPE
    assert decoded["stable_id"][:, 0].tolist() == [101, 202]
    assert decoded["semantic_id"][:, 0].tolist() == [7, 9]


def test_decode_stable_id_semantic_id_map_rejects_partial_record() -> None:
    with pytest.raises(ValueError, match="multiple"):
        decode_stable_id_semantic_id_map(b"partial")


def test_decode_instance_map_preserves_invalid_sentinel() -> None:
    raw = np.array([5, 8, 0xFFFFFFFF], dtype="<u4").view(np.uint8)

    decoded = decode_instance_map(raw)

    assert decoded.dtype == OVRTX_INSTANCE_MAP_DTYPE
    assert decoded["instance_index"].tolist() == [0, 1, 2]
    assert decoded["stable_semantic_index"].tolist() == [5, 8, 0xFFFFFFFF]


def test_decode_instance_map_rejects_unaligned_buffer() -> None:
    with pytest.raises(ValueError, match="alignment"):
        decode_instance_map(b"123")


def _make_bbox3d(boxes):
    arr = np.zeros(len(boxes), dtype=OVRTX_BBOX3D_DTYPE)
    for i, (iid, sid, lo, hi, translate) in enumerate(boxes):
        arr[i]["instance_id"] = _id128(iid)
        arr[i]["semantic_id"] = _id128(sid)
        arr[i]["x_min"], arr[i]["y_min"], arr[i]["z_min"] = lo
        arr[i]["x_max"], arr[i]["y_max"], arr[i]["z_max"] = hi
        transform = np.eye(4, dtype=np.float32)
        transform[3, :3] = translate
        arr[i]["transform"] = transform
    return arr


def _make_camera_params():
    arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
    arr[0]["view_transform"] = np.eye(4, dtype=np.float32)
    arr[0]["projection"] = np.eye(4, dtype=np.float32)
    arr[0]["meters_per_scene_unit"] = 1.0
    arr[0]["resolution"] = (64, 48)
    arr[0]["clipping_plane"] = (0.01, 1000.0)
    arr[0]["fov_x"] = np.radians(60.0)
    return arr


# ---------------------------------------------------------------------------
# iter_bboxes_2d
# ---------------------------------------------------------------------------


class TestIterBboxes2D:
    def test_none_yields_nothing(self):
        assert list(iter_bboxes_2d(None)) == []

    def test_empty_array_yields_nothing(self):
        assert list(iter_bboxes_2d(np.zeros(0, dtype=OVRTX_BBOX2D_DTYPE))) == []

    def test_ovrtx_dtype_fields(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX2D_DTYPE)
        arr[0]["instance_id"] = _id128(7)
        arr[0]["semantic_id"] = _id128(3)
        arr[0]["x_min"], arr[0]["y_min"], arr[0]["x_max"], arr[0]["y_max"] = (
            10,
            20,
            50,
            60,
        )
        b = list(iter_bboxes_2d(arr))[0]
        assert b.x_min == 10.0
        assert b.y_min == 20.0
        assert b.x_max == 50.0
        assert b.y_max == 60.0
        assert b.semantic_id == 3
        assert b.instance_id == 7

    def test_label_fallback_to_stringified_id(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX2D_DTYPE)
        arr[0]["semantic_id"] = _id128(5)
        b = list(iter_bboxes_2d(arr))[0]
        assert b.label == "5"

    def test_label_resolved_from_map(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX2D_DTYPE)
        arr[0]["semantic_id"] = _id128(3)
        b = list(iter_bboxes_2d(arr, {3: "car"}))[0]
        assert b.label == "car"

    def test_multiple_rows(self):
        arr = np.zeros(2, dtype=OVRTX_BBOX2D_DTYPE)
        arr[0]["semantic_id"] = _id128(1)
        arr[1]["semantic_id"] = _id128(2)
        assert len(list(iter_bboxes_2d(arr))) == 2


# ---------------------------------------------------------------------------
# iter_bboxes_3d
# ---------------------------------------------------------------------------


class TestIterBboxes3D:
    def test_none_yields_nothing(self):
        assert list(iter_bboxes_3d(None)) == []

    def test_empty_array_yields_nothing(self):
        assert list(iter_bboxes_3d(np.zeros(0, dtype=OVRTX_BBOX3D_DTYPE))) == []

    def test_translation_is_read_from_last_row(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX3D_DTYPE)
        arr[0]["x_min"], arr[0]["y_min"], arr[0]["z_min"] = -1, -1, -1
        arr[0]["x_max"], arr[0]["y_max"], arr[0]["z_max"] = 1, 1, 1
        transform = np.eye(4, dtype=np.float32)
        transform[3, :3] = (500, 500, 500)
        arr[0]["transform"] = transform
        b = list(iter_bboxes_3d(arr))[0]
        np.testing.assert_allclose(b.center, [500, 500, 500])

    def test_ovrtx_dtype_center_extent_from_transform(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX3D_DTYPE)
        arr[0]["instance_id"] = _id128(20)
        arr[0]["semantic_id"] = _id128(10)
        arr[0]["x_min"], arr[0]["y_min"], arr[0]["z_min"] = -1, -1, -2
        arr[0]["x_max"], arr[0]["y_max"], arr[0]["z_max"] = 1, 1, 2
        transform = np.eye(4, dtype=np.float32)
        transform[3, :3] = (5, 2, 10)
        arr[0]["transform"] = transform
        b = list(iter_bboxes_3d(arr, {10: "box"}))[0]
        np.testing.assert_allclose(b.center, [5, 2, 10])
        np.testing.assert_allclose(b.extent, [1, 1, 2])
        np.testing.assert_allclose(b.rotation, np.eye(3))
        assert b.semantic_id == 10
        assert b.instance_id == 20
        assert b.label == "box"
        assert b.coordinate_system == "world"

    def test_rotation_applied_to_center(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX3D_DTYPE)
        arr[0]["x_min"], arr[0]["y_min"], arr[0]["z_min"] = 0, 0, 0
        arr[0]["x_max"], arr[0]["y_max"], arr[0]["z_max"] = 2, 0, 0
        # 90-degree rotation about Z, translation (0, 0, 0). The transform
        # multiplies row vectors, so its 3x3 block is the transpose of the
        # column-vector rotation matrix.
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot.T
        arr[0]["transform"] = transform
        b = list(iter_bboxes_3d(arr))[0]
        # local center (1, 0, 0) rotates to (0, 1, 0)
        np.testing.assert_allclose(b.center, [0, 1, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# extract_camera_params
# ---------------------------------------------------------------------------


class TestExtractCameraParams:
    def _params(self, fov_x: float, width: int, height: int) -> np.ndarray:
        arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
        arr[0]["view_transform"] = np.eye(4, dtype=np.float32)
        arr[0]["projection"] = np.eye(4, dtype=np.float32)
        arr[0]["resolution"] = (width, height)
        arr[0]["clipping_plane"] = (0.1, 100.0)
        arr[0]["fov_x"] = fov_x
        return arr

    def test_resolution_and_clipping(self):
        params = extract_camera_params(self._params(math.radians(60), 1920, 1080))
        assert params.resolution.width == 1920
        assert params.resolution.height == 1080
        assert abs(params.near - 0.1) < 1e-5
        assert abs(params.far - 100.0) < 1e-3

    def test_intrinsics_from_fov(self):
        # fov_x = 90deg -> fx = (w/2) / tan(45deg) = w/2
        params = extract_camera_params(self._params(math.radians(90), 640, 480))
        assert abs(params.K[0, 0] - 320.0) < 1e-3
        assert abs(params.K[0, 2] - 320.0) < 1e-6  # cx = w/2
        assert abs(params.K[1, 2] - 240.0) < 1e-6  # cy = h/2

    def test_fov_y_derived_from_aspect(self):
        # Square pixels: fov_y = 2*atan((h/2)/fx). For fov_x=90deg, w=640,
        # h=480 -> fx=320 -> fov_y = 2*atan(240/320).
        params = extract_camera_params(self._params(math.radians(90), 640, 480))
        assert abs(params.fov_x - math.radians(90)) < 1e-6
        assert abs(params.fov_y - 2.0 * math.atan(240.0 / 320.0)) < 1e-6

    def test_extrinsics_read_row_vector_world_to_camera(self):
        arr = self._params(math.radians(60), 640, 480)
        # World-to-camera view matrix for row vectors: translation is the
        # last row and the 3x3 block is the transpose of the column rotation.
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rot.T
        view[3, :3] = (1, 2, 3)
        arr[0]["view_transform"] = view
        params = extract_camera_params(arr)
        np.testing.assert_allclose(params.R, rot, atol=1e-6)
        np.testing.assert_allclose(params.t, [1, 2, 3], atol=1e-6)
        # R @ world + t reproduces [world, 1] @ view for an off-origin point.
        world = np.array([4.0, -5.0, 6.0])
        expected = (np.append(world, 1.0) @ view.astype(np.float64))[:3]
        np.testing.assert_allclose(params.R @ world + params.t, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# SemanticIdMap decoding + label-map resolution
# ---------------------------------------------------------------------------


class TestSemanticIdMap:
    def test_roundtrip(self):
        buf = _make_id_map({1: "car", 2: "person"})
        assert decode_semantic_id_map(buf) == {1: "car", 2: "person"}

    def test_none_returns_empty(self):
        assert decode_semantic_id_map(None) == {}

    def test_too_short_returns_empty(self):
        assert decode_semantic_id_map(b"\x00") == {}

    def test_roundtrip_from_warp_array(self):
        buf = _make_id_map({1: "car", 2: "person"})
        assert decode_semantic_id_map(wp.array(buf, device="cpu")) == {
            1: "car",
            2: "person",
        }

    def test_id_label_map_from_render_vars_decodes_buffer(self):
        rv = {"SemanticIdMap": _make_id_map({4: "tree"})}
        assert id_label_map_from_render_vars(rv) == {4: "tree"}

    def test_id_label_map_from_render_vars_passthrough_dict(self):
        rv = {"SemanticIdMap": {9: "sign"}}
        assert id_label_map_from_render_vars(rv) == {9: "sign"}

    def test_id_label_map_from_render_vars_absent(self):
        assert id_label_map_from_render_vars({}) is None


# ---------------------------------------------------------------------------
# Raw payload decoders
# ---------------------------------------------------------------------------


class TestPayloadDecoders:
    def test_decode_bbox_2d_from_raw(self):
        arr = np.zeros(2, dtype=OVRTX_BBOX2D_DTYPE)
        arr[0]["x_min"] = 1
        arr[1]["x_min"] = 2
        raw = arr.tobytes() + (2).to_bytes(4, "little")
        decoded = decode_bbox_2d(np.frombuffer(raw, dtype=np.uint8))
        assert len(decoded) == 2
        assert decoded[0]["x_min"] == 1 and decoded[1]["x_min"] == 2

    def test_decode_bbox_2d_passthrough(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX2D_DTYPE)
        decoded = decode_bbox_2d(arr)
        decoded_np = decoded.numpy() if hasattr(decoded, "numpy") else decoded
        assert decoded_np.dtype == OVRTX_BBOX2D_DTYPE
        assert len(decoded_np) == 1

    def test_decode_bbox_3d_from_raw(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX3D_DTYPE)
        arr[0]["x_max"] = 3.0
        raw = arr.tobytes() + (1).to_bytes(4, "little")
        decoded = decode_bbox_3d(np.frombuffer(raw, dtype=np.uint8))
        decoded_np = decoded.numpy() if hasattr(decoded, "numpy") else decoded
        assert len(decoded_np) == 1 and decoded_np[0]["x_max"] == 3.0

    def test_decode_camera_params_from_raw(self):
        arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
        arr[0]["fov_x"] = 1.25
        decoded = decode_camera_params(np.frombuffer(arr.tobytes(), dtype=np.uint8))
        decoded_np = decoded.numpy() if hasattr(decoded, "numpy") else decoded
        assert abs(float(decoded_np[0]["fov_x"]) - 1.25) < 1e-6

    def test_decode_bbox_2d_empty(self):
        assert len(decode_bbox_2d(b"")) == 0

    def test_decode_bbox_2d_from_warp_raw(self):
        arr = np.zeros(1, dtype=OVRTX_BBOX2D_DTYPE)
        arr[0]["x_min"] = 7
        raw = np.frombuffer(
            arr.tobytes() + (1).to_bytes(4, "little"), dtype=np.uint8
        ).copy()
        decoded = decode_bbox_2d(wp.array(raw, device="cpu"))
        assert len(decoded) == 1
        assert decoded[0]["x_min"] == 7


# ---------------------------------------------------------------------------
# mask_to_rle
# ---------------------------------------------------------------------------


class TestMaskToRle:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        rle = mask_to_rle(mask)
        assert rle["size"] == [10, 10]
        # All zeros → single count covering everything
        assert sum(rle["counts"]) == 100

    def test_full_mask(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        rle = mask_to_rle(mask)
        assert rle["size"] == [5, 5]
        # Starts with 0 zeros, then 25 ones
        assert rle["counts"][0] == 0
        assert rle["counts"][1] == 25

    def test_single_pixel(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[2, 1] = 1
        rle = mask_to_rle(mask)
        assert rle["size"] == [4, 4]
        # Fortran order: column-major
        total = sum(rle["counts"])
        assert total == 16

    def test_boolean_mask(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        rle = mask_to_rle(mask)
        assert rle["size"] == [3, 3]
        total = sum(rle["counts"])
        assert total == 9

    def test_vertical_stripe(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:, 3] = 1  # Full column
        rle = mask_to_rle(mask)
        # In Fortran order, column 3 is contiguous
        assert sum(rle["counts"]) == 100

    def test_warp_mask(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 2] = 1
        rle = mask_to_rle(wp.array(mask, device="cpu"))
        assert rle["size"] == [4, 4]
        assert sum(rle["counts"]) == 16


# ---------------------------------------------------------------------------
# mask_to_polygon
# ---------------------------------------------------------------------------


def _point_in_polygon(
    poly_norm: list[float], height: int, width: int, py: int, px: int
) -> bool:
    """Even-odd ray-cast test for a normalized flat polygon (pixel centers)."""
    pts = np.asarray(poly_norm, dtype=float).reshape(-1, 2) * np.array([width, height])
    x, y = px + 0.5, py + 0.5
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class TestMaskToPolygon:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_to_polygon(mask) == []

    def test_single_pixel_is_valid_polygon(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1
        poly = mask_to_polygon(mask)
        # A 1-px object yields a small closed contour (never a degenerate
        # <3-vertex polygon, which YOLO-seg cannot consume).
        assert len(poly) >= 6
        assert len(poly) % 2 == 0
        assert all(0.0 <= v <= 1.0 for v in poly)

    def test_rectangle_is_four_corners(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:40, 50:150] = 1
        poly = mask_to_polygon(mask)
        # A simplified axis-aligned rectangle reduces to its 4 corners.
        assert len(poly) // 2 == 4
        assert all(0.0 <= v <= 1.0 for v in poly)

    def test_blob_traces_many_vertices(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        yy, xx = np.ogrid[:200, :200]
        mask[((yy - 100) ** 2 + (xx - 100) ** 2) < 50**2] = 1
        poly = mask_to_polygon(mask)
        # A circle's tight contour has many vertices, far more than a bbox.
        assert len(poly) // 2 > 8

    def test_concave_contour_excludes_notch(self):
        """A concave L-shape keeps its notch outside the polygon.

        This is the key advantage over a convex hull, which would wrongly
        include the empty notch.
        """
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[10:50, 10:25] = 1  # vertical bar
        mask[35:50, 10:50] = 1  # horizontal foot
        poly = mask_to_polygon(mask)
        assert _point_in_polygon(poly, 60, 60, 20, 15)  # inside the bar
        assert _point_in_polygon(poly, 60, 60, 45, 45)  # inside the foot
        assert not _point_in_polygon(poly, 60, 60, 15, 40)  # empty notch
        assert not _point_in_polygon(poly, 60, 60, 5, 5)  # outside

    def test_two_pixel_line_is_valid_polygon(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10, 10] = 1
        mask[10, 11] = 1
        poly = mask_to_polygon(mask)
        assert len(poly) // 2 >= 3

    def test_no_simplification_keeps_full_contour(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:40, 50:150] = 1
        full = mask_to_polygon(mask, simplify_tol=0.0)
        simplified = mask_to_polygon(mask, simplify_tol=1.0)
        assert len(full) > len(simplified)

    def test_device_agnostic_cpu_matches(self):
        """The contour runs on the Warp CPU device too (no CUDA hard-coding)."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:30, 8:35] = 1
        with wp.ScopedDevice("cpu"):
            poly = mask_to_polygon(mask)
        assert len(poly) // 2 == 4
        assert all(0.0 <= v <= 1.0 for v in poly)

    def test_channel_axis_is_squeezed(self):
        """An ``(H, W, 1)`` mask is accepted (trailing channel squeezed)."""
        mask = np.zeros((100, 200, 1), dtype=np.uint8)
        mask[20:40, 50:150, 0] = 1
        assert len(mask_to_polygon(mask)) // 2 == 4


class TestDouglasPeucker:
    def test_eps_zero_returns_input(self):
        from ovannotators.writers._helpers import _douglas_peucker

        pts = np.array([[0.0, 0.0], [1.0, 5.0], [10.0, 0.0]])
        out = _douglas_peucker(pts, 0.0)
        assert np.array_equal(out, pts)

    def test_splits_and_keeps_significant_vertex(self):
        from ovannotators.writers._helpers import _douglas_peucker

        pts = np.array([[0.0, 0.0], [0.0, 5.0], [10.0, 0.0]])
        out = _douglas_peucker(pts, 1.0)
        assert len(out) == 3  # the 5-px deviation is kept

    def test_zero_length_segment_uses_radial_distance(self):
        from ovannotators.writers._helpers import _douglas_peucker

        # First and last vertices coincide -> the segment is degenerate, so the
        # algorithm falls back to radial distance from that anchor.
        pts = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
        out = _douglas_peucker(pts, 1.0)
        assert len(out) == 3


class TestSimplifyRing:
    def test_degenerate_ring_returns_unchanged(self):
        from ovannotators.writers._helpers import _simplify_ring

        ring = np.zeros((4, 2), dtype=np.float64)  # all vertices coincide
        out = _simplify_ring(ring, 1.0)
        assert np.array_equal(out, ring)


# ---------------------------------------------------------------------------
# Fallback branches, structured-array decoders, and defensive guards
# ---------------------------------------------------------------------------


def test_to_numpy_asarray_and_typeerror():
    from ovannotators.writers import _helpers

    assert _helpers._to_numpy([1, 2, 3]).tolist() == [1, 2, 3]

    class _Bad:
        def __array__(self, *args, **kwargs):
            raise ValueError("nope")

    with pytest.raises(TypeError):
        _helpers._to_numpy(_Bad())


def test_decode_semantic_id_map_edge_cases():
    from ovannotators.writers import _helpers

    # num_entries <= 0
    assert _helpers.decode_semantic_id_map((0).to_bytes(4, "little")) == {}
    # needed > len(raw): claims 5 entries but buffer too short
    assert _helpers.decode_semantic_id_map((5).to_bytes(4, "little")) == {}
    # offset + length out of range → entry skipped
    entry_dtype = np.dtype(
        [("semantic_id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    entries = np.zeros(1, dtype=entry_dtype)
    entries[0]["semantic_id"] = (9, 0, 0, 0)
    entries[0]["label_length"] = 100
    entries[0]["label_offset"] = 1000
    raw = entries.tobytes() + (1).to_bytes(4, "little")
    assert _helpers.decode_semantic_id_map(raw) == {}


def test_to_bytes_generic_fallback():
    from ovannotators.writers import _helpers

    # A plain iterable of ints falls through to ``bytes(data)``.
    assert _helpers._to_bytes([1, 2, 3]) == bytes([1, 2, 3])


def test_decode_bbox3d_structured_and_short():
    from ovannotators.writers import _helpers

    structured = _make_bbox3d([(1, 1, (-1, -1, -1), (1, 1, 1), (0, 0, 0))])
    out = _helpers.decode_bbox_3d(structured)
    assert _helpers._to_numpy(out).dtype == OVRTX_BBOX3D_DTYPE
    # Buffer shorter than 4 bytes → empty array.
    empty = _helpers.decode_bbox_3d(b"\x00")
    assert _helpers._to_numpy(empty).size == 0


def test_decode_camera_params_structured_passthrough():
    from ovannotators.writers import _helpers

    arr = _make_camera_params()
    out = _helpers.decode_camera_params(arr)
    assert _helpers._to_numpy(out).dtype == OVRTX_CAMERA_PARAMS_DTYPE


# ---------------------------------------------------------------------------
# point_cloud_from_depth
# ---------------------------------------------------------------------------


class TestPointCloudFromDepth:
    def _camera(self) -> CameraParams:
        # fx = fy = 1, principal point at (1, 1) for a 3x3 image.
        K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        return CameraParams(
            K=K,
            R=np.eye(3),
            t=np.zeros(3),
            resolution=Resolution(3, 3),
            near=0.1,
            far=10.0,
            fov_x=1.0,
            fov_y=1.0,
        )

    def test_backprojects_center_pixel(self):
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 5.0  # only the centre pixel is valid
        pc = point_cloud_from_depth(depth, self._camera(), device="cpu")
        assert isinstance(pc, PointCloudResult)
        # Centre pixel (u=cx, v=cy) back-projects to (0, 0, z).
        np.testing.assert_allclose(pc.points, [[0.0, 0.0, 5.0]])
        assert pc.normals is None
        assert pc.colors is None

    def test_drops_invalid_depth(self):
        depth = np.full((2, 2), np.nan, dtype=np.float32)
        depth[0, 0] = -1.0  # non-positive is also dropped
        pc = point_cloud_from_depth(depth, self._camera(), device="cpu")
        assert pc.points.shape == (0, 3)

    def test_carries_colors_and_semantic_ids(self):
        depth = np.ones((2, 2), dtype=np.float32)
        colors = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
        sem = np.arange(4, dtype=np.uint32).reshape(2, 2)
        pc = point_cloud_from_depth(
            depth, self._camera(), colors=colors, semantic_ids=sem, device="cpu"
        )
        assert pc.points.shape == (4, 3)
        assert pc.colors.shape == (4, 3)
        assert pc.semantic_ids.shape == (4,)

    def test_squeezes_3d_depth_and_semantic_channel_axis(self):
        """A trailing channel axis on depth and semantic ids is squeezed."""
        depth = np.ones((2, 2, 1), dtype=np.float32)  # (H, W, 1) depth
        sem = np.arange(4, dtype=np.uint32).reshape(2, 2, 1)  # (H, W, 1) ids
        pc = point_cloud_from_depth(
            depth, self._camera(), semantic_ids=sem, device="cpu"
        )
        assert pc.points.shape == (4, 3)
        assert pc.semantic_ids.shape == (4,)


def test_mask_to_rle_empty_mask():
    from ovannotators.writers import _helpers

    out = _helpers.mask_to_rle(np.zeros((0, 0), dtype=np.uint8))
    assert out == {"counts": [], "size": [0, 0]}


# ---------------------------------------------------------------------------
# id map / instance-mask helpers
# ---------------------------------------------------------------------------


class TestIdAndInstanceHelpers:
    def test_prim_path_map_passthrough_dict(self):
        """A pre-decoded ``StableIdMap`` dict is returned unchanged."""
        mapping = {1: "/World/Cube", 2: "/World/Sphere"}
        assert id_prim_path_map_from_render_vars({"StableIdMap": mapping}) == mapping

    def test_instance_masks_empty_ids_returns_empty(self):
        """No requested instance ids short-circuits to an empty mapping."""
        seg = np.zeros((4, 4), dtype=np.int32)
        assert instance_masks(seg, []) == {}

    def test_instance_masks_squeezes_channel_axis(self):
        """A trailing ``(H, W, 1)`` channel axis on the seg image is squeezed."""
        seg = np.zeros((4, 4, 1), dtype=np.int32)
        seg[1:3, 1:3, 0] = 7
        masks = instance_masks(seg, [7])
        assert set(masks) == {7}
        assert masks[7].shape == (4, 4)
        assert masks[7].sum() == 4  # the 2x2 region set to id 7


# ---------------------------------------------------------------------------
# to_rgb_uint8
# ---------------------------------------------------------------------------


class TestToRgbUint8:
    def test_rgba_drops_alpha(self):
        rgba = np.zeros((3, 4, 4), dtype=np.uint8)
        rgba[..., 3] = 128
        rgba[..., 0] = 10
        out = to_rgb_uint8(rgba)
        assert out.shape == (3, 4, 3)
        assert out[0, 0].tolist() == [10, 0, 0]

    def test_grayscale_replicated_across_channels(self):
        gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
        out = to_rgb_uint8(gray)
        assert out.shape == (2, 2, 3)
        assert out[1, 1].tolist() == [3, 3, 3]

    def test_single_channel_axis_squeezed(self):
        out = to_rgb_uint8(np.ones((2, 2, 1), dtype=np.uint8) * 5)
        assert out.shape == (2, 2, 3)
        assert out[0, 0].tolist() == [5, 5, 5]

    def test_float_scaled_and_clipped(self):
        """Floats are treated as [0, 1] and scaled to the 8-bit range."""
        arr = np.array([[[0.0, 0.5, 1.0]], [[-1.0, 2.0, 0.25]]], dtype=np.float32)
        out = to_rgb_uint8(arr)
        assert out.dtype == np.uint8
        assert out[0, 0].tolist() == [0, 127, 255]
        assert out[1, 0].tolist() == [0, 255, 63]

    def test_warp_array_input(self):
        """Warp arrays are converted to host NumPy before the RGB fixup."""
        arr = wp.array(np.zeros((2, 2, 4), dtype=np.uint8), device="cpu")
        out = to_rgb_uint8(arr)
        assert out.shape == (2, 2, 3) and out.dtype == np.uint8

    def test_output_is_contiguous(self):
        """Codecs require contiguous buffers; the alpha-slice view is copied."""
        out = to_rgb_uint8(np.zeros((2, 2, 4), dtype=np.uint8))
        assert out.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# OpenCV camera frame
# ---------------------------------------------------------------------------


class TestOpenCvCameraFrame:
    def _camera(self, translation):
        arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
        view = np.eye(4, dtype=np.float32)
        view[3, :3] = translation
        arr[0]["view_transform"] = view
        arr[0]["resolution"] = (640, 480)
        arr[0]["fov_x"] = math.radians(60)
        return extract_camera_params(arr)

    def test_usd_camera_frame_is_flipped_to_opencv(self):
        # A point 20 units in front of a USD camera (-Z) has +Z depth in OpenCV.
        cam = self._camera((0, 0, -20))
        r, t = world_to_opencv_camera(cam)
        np.testing.assert_allclose(r, np.diag([1, -1, -1]))
        np.testing.assert_allclose(r @ np.zeros(3) + t, [0, 0, 20])

    def test_pose_rotates_and_translates_into_camera(self):
        cam = self._camera((0, 0, -20))
        rotation, center = pose_in_opencv_camera(
            cam, np.eye(3), np.array([5.0, 2.0, 10.0])
        )
        np.testing.assert_allclose(center, [5, -2, 10])
        np.testing.assert_allclose(rotation, np.diag([1, -1, -1]))

    def test_pose_composes_rotations_in_world_to_camera_order(self):
        # Rotations about different axes do not commute, so a reversed
        # multiplication changes the answer. Camera rotated 90 degrees about Y
        # (world to USD camera), object rotated 90 degrees about X.
        def rot_y(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def rot_x(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        r_cam, r_obj = rot_y(math.pi / 2), rot_x(math.pi / 2)
        arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = r_cam.T  # row-vector convention
        arr[0]["view_transform"] = view
        arr[0]["resolution"] = (640, 480)
        arr[0]["fov_x"] = math.radians(60)
        cam = extract_camera_params(arr)

        flip = np.diag([1.0, -1.0, -1.0])
        expected = flip @ r_cam @ r_obj
        assert not np.allclose(expected, flip @ r_obj @ r_cam)
        rotation, _ = pose_in_opencv_camera(cam, r_obj, np.zeros(3))
        np.testing.assert_allclose(rotation, expected, atol=1e-6)

    def test_rotation_without_scale_normalizes_each_axis(self):
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        scaled = rot * np.array([2.0, 3.0, 4.0])  # scale each axis column
        np.testing.assert_allclose(rotation_without_scale(scaled), rot, atol=1e-12)
        np.testing.assert_allclose(rotation_without_scale(rot), rot, atol=1e-12)

    def test_rotation_and_scale_returns_unit_axes_and_per_axis_scale(self):
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        unit, scale = rotation_and_scale(rot * np.array([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(unit, rot, atol=1e-12)
        np.testing.assert_allclose(scale, [2.0, 3.0, 4.0], atol=1e-12)

    def test_rotation_without_scale_keeps_a_zero_axis(self):
        out = rotation_without_scale(np.zeros((3, 3)))
        np.testing.assert_array_equal(out, np.zeros((3, 3)))


# ovrtx 0.7 named-tensor annotator layout
# ---------------------------------------------------------------------------
# Shapes and values follow a real ovrtx 0.7 capture of tests/data/test_scene.usda.


def _ids(*low_words: int) -> np.ndarray:
    ids = np.zeros((len(low_words), 4), dtype=np.uint32)
    ids[:, 0] = low_words
    return ids


def _label_table(ids: np.ndarray, labels: list[bytes]) -> dict:
    offsets = np.cumsum([0, *(len(label) for label in labels)]).astype(np.uint64)
    data = np.frombuffer(b"".join(labels), dtype=np.uint8)
    return {
        "tensors": {"id": ids, "label_offsets": offsets, "label_data": data},
        "params": {},
    }


class TestOvrtx07Decoders:
    def test_bbox_2d_keeps_valid_rows_and_splits_extent(self):
        """Rows past ``valid_count`` are dropped; extent is min/max corners."""
        extent = np.array(
            [[[361, 196], [485, 320]], [[578, 307], [701, 449]], [[9, 9], [9, 9]]],
            dtype=np.int32,
        )
        payload = {
            "tensors": {
                "id": _ids(11, 12, 0),
                "semantic_id": _ids(2, 3, 0),
                "extent": extent,
                "valid_count": np.array([2], dtype=np.uint32),
            },
            "params": {},
        }
        decoded = decode_bbox_2d(payload)
        assert decoded.dtype == OVRTX_BBOX2D_DTYPE
        assert decoded["instance_id"][:, 0].tolist() == [11, 12]
        assert decoded["semantic_id"][:, 0].tolist() == [2, 3]
        assert decoded["x_min"].tolist() == [361, 578]
        assert decoded["y_min"].tolist() == [196, 307]
        assert decoded["x_max"].tolist() == [485, 701]
        assert decoded["y_max"].tolist() == [320, 449]

    @pytest.mark.parametrize(
        ("count", "expected_ids"),
        [(0, []), (None, [11, 12, 13]), (4, [11, 12, 13])],
        ids=["zero", "missing", "over-capacity"],
    )
    def test_bbox_2d_valid_count_boundaries(self, count, expected_ids):
        tensors = {
            "id": _ids(11, 12, 13),
            "semantic_id": _ids(2, 3, 4),
            "extent": np.zeros((3, 2, 2), dtype=np.int32),
        }
        if count is not None:
            tensors["valid_count"] = np.array([count], dtype=np.uint32)
        decoded = decode_bbox_2d({"tensors": tensors, "params": {}})
        assert decoded["instance_id"][:, 0].tolist() == expected_ids

    def test_bbox_3d_unpacks_extent_and_transform(self):
        """3-D extent splits into per-axis min/max; object_to_world is kept."""
        extent = np.array(
            [[[-50, -50, -50], [50, 50, 50]], [[0, 0, 0], [0, 0, 0]]], dtype=np.float32
        )
        transform = np.stack(
            [np.eye(4, dtype=np.float32), np.zeros((4, 4), np.float32)]
        )
        transform[0, 3, :3] = [100.0, 0.0, -150.0]
        payload = {
            "tensors": {
                "id": _ids(7, 0),
                "semantic_id": _ids(4, 0),
                "extent": extent,
                "object_to_world": transform,
                "valid_count": np.array([1], dtype=np.uint32),
            },
            "params": {},
        }
        decoded = decode_bbox_3d(payload)
        assert decoded.dtype == OVRTX_BBOX3D_DTYPE
        assert len(decoded) == 1
        assert [decoded[f"{a}_min"][0] for a in "xyz"] == [-50.0] * 3
        assert [decoded[f"{a}_max"][0] for a in "xyz"] == [50.0] * 3
        np.testing.assert_array_equal(decoded["transform"][0], transform[0])

    def test_id_string_maps_use_label_offsets(self):
        """Labels are sliced from ``label_data`` by consecutive offsets."""
        payload = _label_table(_ids(2, 3), [b"class: sphere; ", b"class: cone; "])
        assert decode_semantic_id_map(payload) == {
            2: "class: sphere;",
            3: "class: cone;",
        }
        assert id_label_map_from_render_vars({"SemanticIdMap": payload}) == {
            2: "class: sphere;",
            3: "class: cone;",
        }
        paths = _label_table(_ids(5), [b"/Replicator/Cube_Xform/Cube"])
        assert id_prim_path_map_from_render_vars({"StableIdMap": paths}) == {
            5: "/Replicator/Cube_Xform/Cube"
        }

    def test_stable_id_semantic_id_map_pairs_tensors(self):
        """``stable_id`` and ``semantic_id`` tensors zip into record pairs."""
        payload = {
            "tensors": {"stable_id": _ids(10, 11), "semantic_id": _ids(2, 3)},
            "params": {},
        }
        decoded = decode_stable_id_semantic_id_map(payload)
        assert decoded.dtype == OVRTX_STABLE_SEMANTIC_ID_DTYPE
        assert decoded["stable_id"][:, 0].tolist() == [10, 11]
        assert decoded["semantic_id"][:, 0].tolist() == [2, 3]

    def test_camera_params_maps_named_params(self):
        """Named params land in the legacy record fields."""
        view = np.arange(16, dtype=np.float32).reshape(4, 4)
        proj = np.eye(4, dtype=np.float32) * 2
        f32 = lambda v: np.array([v], dtype=np.float32)  # noqa: E731
        params = {
            "view_matrix": view,
            "projection_matrix": proj,
            "meters_per_scene_unit": f32(0.01),
            "render_product_resolution": np.array([1280, 720], dtype=np.uint32),
            "near_far": np.array([1.0, 1.0e6], dtype=np.float32),
            "camera_model": np.frombuffer(b"pinhole", dtype=np.uint8),
            "focal_length": f32(1.814756),
            "fov_x_radians": f32(1.0471977),
            "ftheta_cx": f32(3.0),
            "ftheta_cy": f32(4.0),
            **{f"ftheta_poly_{c}": f32(i) for i, c in enumerate("abcdef")},
            "lens_p": np.zeros(2, np.float32),
            "lens_s": np.zeros(4, np.float32),
        }
        decoded = np.asarray(decode_camera_params({"tensors": {}, "params": params}))
        assert decoded.dtype == OVRTX_CAMERA_PARAMS_DTYPE
        record = decoded[0]
        np.testing.assert_array_equal(record["view_transform"], view)
        np.testing.assert_array_equal(record["projection"], proj)
        assert record["resolution"].tolist() == [1280, 720]
        assert record["clipping_plane"].tolist() == [1.0, 1.0e6]
        assert record["projection_model"] == b"pinhole"
        assert record["fisheye_optical_centre"].tolist() == [3.0, 4.0]
        assert record["fisheye_polynomial"].tolist() == [0, 1, 2, 3, 4, 5]
        assert record["fov_x"] == pytest.approx(1.0471977)
        assert record["focal_length"] == pytest.approx(1.814756)
        assert record["meters_per_scene_unit"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Occlusion decoding
# ---------------------------------------------------------------------------


class TestOcclusion:
    def test_decode_occlusion_reads_12_byte_records(self):
        arr = np.zeros(2, dtype=OVRTX_OCCLUSION_DTYPE)
        arr["instance_id"] = [6, 7]
        arr["ratio"] = [0.25, -1.0]
        decoded = decode_occlusion(arr.tobytes())
        assert decoded["instance_id"].tolist() == [6, 7]
        np.testing.assert_allclose(decoded["ratio"], [0.25, -1.0])

    def test_decode_occlusion_rejects_partial_record(self):
        with pytest.raises(ValueError, match="multiple of 12"):
            decode_occlusion(b"\x00" * 13)

    def test_decode_instance_token_map_checks_header_count(self):
        tokens = np.zeros(3, dtype=OVRTX_INSTANCE_TOKEN_DTYPE)
        tokens[0]["instance_id"] = 2
        assert len(decode_instance_token_map(tokens.tobytes())) == 3
        tokens[0]["instance_id"] = 5
        with pytest.raises(ValueError, match="header count"):
            decode_instance_token_map(tokens.tobytes())

    # Values from an ovrtx 0.7 capture of the DOPE occlusion scene.
    _OCCLUSION_TENSORS = {
        "tensors": {
            "instance_id": np.array([3, 4, 2, 5, 1], dtype=np.uint32),
            "render_tile_id": np.zeros(5, dtype=np.uint32),
            "occlusion_ratio": np.array(
                [0.0, 0.0, 0.3272, 0.0, 0.0682], dtype=np.float32
            ),
        },
        "params": {},
    }
    _TOKEN_TENSORS = {
        "tensors": {
            "instance_id": np.array([4, 2, 3, 5, 1], dtype=np.uint32),
            "prim_token": np.array(
                [62930689, 62931201, 62931713, 62932225, 62932737], dtype=np.uint64
            ),
            "prototype_index": np.zeros(5, dtype=np.uint32),
            "instance_index": np.zeros(5, dtype=np.uint32),
        },
        "params": {
            "min_instance_id": np.array([1], dtype=np.uint32),
            "max_instance_id": np.array([5], dtype=np.uint32),
            "render_tile_count": np.array([1], dtype=np.uint32),
        },
    }

    def test_decode_occlusion_reads_named_tensors(self):
        decoded = decode_occlusion(self._OCCLUSION_TENSORS)
        assert decoded.dtype == OVRTX_OCCLUSION_DTYPE
        assert decoded["instance_id"].tolist() == [3, 4, 2, 5, 1]
        assert decoded["tile_id"].tolist() == [0] * 5
        np.testing.assert_allclose(decoded["ratio"], [0.0, 0.0, 0.3272, 0.0, 0.0682])

    def test_decode_instance_token_map_rebuilds_header_from_params(self):
        decoded = decode_instance_token_map(self._TOKEN_TENSORS)
        assert decoded.dtype == OVRTX_INSTANCE_TOKEN_DTYPE
        assert int(decoded[0]["instance_id"]) == 5
        assert int(decoded[0]["prim_token"]) == (5 << 32) | 1
        assert decoded["instance_id"][1:].tolist() == [4, 2, 3, 5, 1]
        assert decoded["prim_token"][1:].tolist() == [
            62930689,
            62931201,
            62931713,
            62932225,
            62932737,
        ]

    def test_occlusion_resolves_from_undecoded_ovrtx_07_vars(self):
        # Instance 2 -> map index 1 -> table entry 2; instance 1 -> index 0 -> entry 4.
        stable_ids = np.array(
            [[n, 0, 0, 0] for n in (10, 20, 30, 40, 50)], dtype=np.uint32
        )
        render_vars = {
            "OcclusionSD": self._OCCLUSION_TENSORS,
            "InstanceIdTokenMapSD": self._TOKEN_TENSORS,
            "InstanceMap": np.array([4, 2, 3, 1, 0], dtype=np.uint32),
            "StableIdSemanticIdMap": {
                "tensors": {"stable_id": stable_ids, "semantic_id": stable_ids},
                "params": {},
            },
        }
        result = occlusion_by_stable_id(render_vars)
        assert result[(30, 0, 0, 0)] == pytest.approx(0.3272)
        assert result[(50, 0, 0, 0)] == pytest.approx(0.0682)
        assert len(result) == 5

    def _vars(self, ratios, instance_ids, table):
        occlusion = np.zeros(len(ratios), dtype=OVRTX_OCCLUSION_DTYPE)
        occlusion["instance_id"] = instance_ids
        occlusion["ratio"] = ratios
        tokens = np.zeros(len(table) + 1, dtype=OVRTX_INSTANCE_TOKEN_DTYPE)
        tokens[0]["instance_id"] = len(table)
        tokens[0]["prim_token"] = min(instance_ids)
        instance_map = np.zeros(len(table), dtype=OVRTX_INSTANCE_MAP_DTYPE)
        instance_map["instance_index"] = np.arange(len(table))
        instance_map["stable_semantic_index"] = table
        stable = np.zeros(2, dtype=OVRTX_STABLE_SEMANTIC_ID_DTYPE)
        stable["stable_id"] = [(11, 0, 0, 0), (22, 0, 0, 0)]
        return {
            "OcclusionSD": occlusion,
            "InstanceIdTokenMapSD": tokens,
            "InstanceMap": instance_map,
            "StableIdSemanticIdMap": stable,
        }

    def test_returns_none_when_a_required_var_is_missing(self):
        render_vars = self._vars([0.1], [4], [0])
        del render_vars["InstanceMap"]
        assert occlusion_by_stable_id(render_vars) is None

    def test_maps_renderer_instances_to_stable_ids(self):
        # Instances 4 and 5 -> table entries 1 and 0.
        render_vars = self._vars([0.1, 0.4], [4, 5], [1, 0])
        assert occlusion_by_stable_id(render_vars) == {
            (22, 0, 0, 0): pytest.approx(0.1),
            (11, 0, 0, 0): pytest.approx(0.4),
        }

    def test_averages_instances_sharing_a_stable_id(self):
        render_vars = self._vars([0.2, 0.6], [4, 5], [0, 0])
        result = occlusion_by_stable_id(render_vars)
        assert result == {(11, 0, 0, 0): pytest.approx(0.4)}

    def test_skips_negative_ratios_and_invalid_map_entries(self):
        render_vars = self._vars([-1.0, 0.5], [4, 5], [0, 0xFFFFFFFF])
        assert occlusion_by_stable_id(render_vars) == {}
