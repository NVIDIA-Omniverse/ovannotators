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

"""Tests for :class:`ovannotators.writers.dope.DOPEWriter`."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ovannotators.exceptions import ConfigurationError, MissingRenderVarError
from ovannotators.writers import WriterRegistry
from ovannotators.writers._helpers import (
    OVRTX_BBOX3D_DTYPE,
    OVRTX_CAMERA_PARAMS_DTYPE,
    OVRTX_INSTANCE_MAP_DTYPE,
    OVRTX_INSTANCE_TOKEN_DTYPE,
    OVRTX_OCCLUSION_DTYPE,
    OVRTX_STABLE_SEMANTIC_ID_DTYPE,
)
from ovannotators.writers.dope import DOPEWriter, DOPEWriterConfig

WIDTH, HEIGHT = 200, 100


class _Backend:
    def __init__(self) -> None:
        self.written: dict[str, bytes] = {}

    def write(self, path: str, data: bytes) -> None:
        self.written[path] = data

    def write_array(self, path: str, array, encoding: str = "png") -> None:
        self.written[path] = encoding.encode()

    def makedirs(self, path: str) -> None:
        pass


def _camera(translation=(0.0, 0.0, -20.0), model: bytes = b"pinhole") -> np.ndarray:
    """Camera 20 units from the origin, axes aligned, looking down -Z.

    ``view_transform`` is world-to-camera for row vectors. The projection has a
    90 degree horizontal field of view and, for row vectors, ``w = -z``.
    """
    arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
    view = np.eye(4)
    view[3, :3] = translation
    projection = np.zeros((4, 4))
    projection[0, 0], projection[1, 1] = 1.0, 2.0
    projection[2, 2], projection[2, 3] = -1.0, -1.0
    arr[0]["view_transform"] = view
    arr[0]["projection"] = projection
    arr[0]["resolution"] = (WIDTH, HEIGHT)
    arr[0]["fov_x"] = math.radians(90.0)
    arr[0]["clipping_plane"] = (0.1, 100.0)
    arr[0]["projection_model"] = model
    return arr


def _box(
    instance_id=(1, 0, 0, 0),
    lo=(-1.0, -1.0, -1.0),
    hi=(1.0, 1.0, 1.0),
    transform=None,
) -> np.ndarray:
    arr = np.zeros(1, dtype=OVRTX_BBOX3D_DTYPE)
    arr[0]["instance_id"] = instance_id
    arr[0]["semantic_id"] = (7, 0, 0, 0)
    arr[0]["x_min"], arr[0]["y_min"], arr[0]["z_min"] = lo
    arr[0]["x_max"], arr[0]["y_max"], arr[0]["z_max"] = hi
    arr[0]["transform"] = np.eye(4) if transform is None else transform
    return arr


def _frame(box=None, camera=None, **extra) -> dict:
    return {
        "LdrColor": np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8),
        "SemanticBoundingBox3D": _box() if box is None else box,
        "CameraParams": _camera() if camera is None else camera,
        **extra,
    }


def _write(frame: dict, config: DOPEWriterConfig | None = None, frame_id: int = 3):
    backend = _Backend()
    DOPEWriter(config).write_frame(frame_id, frame, backend)
    return backend


def _only_object(backend: _Backend, stem: str = "000003") -> dict:
    return json.loads(backend.written[f"{stem}.json"])["objects"][0]


def test_registered_under_dope() -> None:
    assert DOPEWriter.name == "dope"
    assert WriterRegistry.get("dope") is DOPEWriter
    assert DOPEWriter.required_render_vars == {
        "LdrColor",
        "SemanticBoundingBox3D",
        "CameraParams",
    }


def test_missing_required_var_raises() -> None:
    frame = _frame()
    del frame["CameraParams"]
    with pytest.raises(MissingRenderVarError):
        DOPEWriter().write_frame(0, frame, _Backend())


def test_writes_image_and_annotation_named_by_frame() -> None:
    backend = _write(_frame())
    assert sorted(backend.written) == ["000003.json", "000003.png"]
    data = json.loads(backend.written["000003.json"])
    assert data["keypoint_order"] == [
        "LUF", "RUF", "RDF", "LDF", "LUB", "RUB", "RDB", "LDB", "Center",
    ]  # fmt: skip
    assert set(data["camera_data"]) == {
        "intrinsics",
        "camera_view_matrix",
        "camera_projection_matrix",
    }
    assert data["camera_data"]["camera_view_matrix"][3] == [0, 0, -20, 1]


def test_pose_and_cuboid_for_a_box_at_the_origin() -> None:
    obj = _only_object(_write(_frame()))
    assert obj["class"] == "7"
    assert obj["visibility"] == 1.0
    # USD camera frame: the origin is 20 units ahead along -Z.
    np.testing.assert_allclose(obj["location"], [0, 0, -20], atol=1e-6)
    np.testing.assert_allclose(obj["quaternion_xyzw"], [0, 0, 0, 1], atol=1e-6)
    # Front face (z=+1) is 19 units away, back face 21. Hand-projected corners.
    cuboid = obj["projected_cuboid"]
    assert cuboid[0] == [95, 45]  # LUF: x=-1, y=+1, front
    assert cuboid[1] == [105, 45]  # RUF
    assert cuboid[2] == [105, 55]  # RDF
    assert cuboid[3] == [95, 55]  # LDF
    assert cuboid[8] == [100, 50]  # Center
    assert obj["truncation_ratio"] == 0.0


def test_location_is_the_object_origin_not_the_box_centre() -> None:
    # Local bounds are offset from the origin, so centre != origin.
    transform = np.eye(4)
    transform[3, :3] = (2.0, 0.0, 0.0)
    box = _box(lo=(0.0, -1.0, -1.0), hi=(2.0, 1.0, 1.0), transform=transform)
    obj = _only_object(_write(_frame(box)))
    np.testing.assert_allclose(obj["location"], [2, 0, -20], atol=1e-6)
    # Centre keypoint is the box centre, one unit further along +X.
    centre_x = obj["projected_cuboid"][8][0]
    origin_x = round((2.0 / 20.0 * 0.5 + 0.5) * WIDTH)
    assert centre_x > origin_x


def test_rotation_becomes_a_unit_quaternion_and_scale_is_removed() -> None:
    # 90 degrees about +Y, uniform scale 2. Row-vector block is the transpose
    # of the column-vector rotation.
    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    transform = np.eye(4)
    transform[:3, :3] = 2.0 * rot.T
    obj = _only_object(_write(_frame(_box(transform=transform))))
    half = math.sqrt(0.5)
    np.testing.assert_allclose(obj["quaternion_xyzw"], [0, half, 0, half], atol=1e-6)
    assert obj["quaternion_xyzw"][3] >= 0.0


def test_object_behind_the_camera_is_skipped() -> None:
    camera = _camera(translation=(0.0, 0.0, 20.0))  # origin now behind the camera
    backend = _write(_frame(camera=camera))
    assert backend.written == {}


def test_mirrored_box_transform_is_skipped() -> None:
    # A negative scale on one axis mirrors the box: its normalized rotation has
    # determinant -1, so there is no valid quaternion for it.
    mirrored = _box(transform=np.diag([1.0, 1.0, -1.0, 1.0]))
    assert _write(_frame(box=mirrored)).written == {}


def test_empty_frame_is_skipped_by_default_and_kept_when_asked() -> None:
    camera = _camera(translation=(0.0, 0.0, 20.0))
    assert _write(_frame(camera=camera)).written == {}
    kept = _write(_frame(camera=camera), DOPEWriterConfig(skip_empty_frames=False))
    assert json.loads(kept.written["000003.json"])["objects"] == []


def test_frame_padding_and_image_format() -> None:
    backend = _write(
        _frame(), DOPEWriterConfig(frame_padding=3, image_format="jpg"), frame_id=7
    )
    assert sorted(backend.written) == ["007.jpg", "007.json"]


def test_rejects_unknown_image_format() -> None:
    with pytest.raises(ConfigurationError, match="image_format"):
        DOPEWriter(DOPEWriterConfig(image_format="exr"))  # type: ignore[arg-type]


def test_rejects_non_pinhole_camera() -> None:
    with pytest.raises(ConfigurationError, match="pinhole"):
        _write(_frame(camera=_camera(model=b"fisheyePolynomial")))


def test_finalize_is_idempotent() -> None:
    writer = DOPEWriter()
    backend = _Backend()
    writer.finalize(backend)
    writer.finalize(backend)
    assert backend.written == {}


# --- visibility from OcclusionSD --------------------------------------------

# Captured from ovrtx 0.5.0 on test_scene.usda: renderer instances 6..10, the
# stable ids they resolve to, and the occluded fractions the renderer reported.
_STABLE_IDS = [
    (3826580222, 3872662258, 3284804802, 0),
    (2862958192, 3724311983, 1859604465, 0),
    (1378685019, 3272213611, 3766051977, 0),
    (2399567550, 4125116989, 3625674830, 0),
    (3319052530, 131194618, 2467051485, 0),
]


def _occlusion_vars() -> dict:
    occlusion = np.zeros(5, dtype=OVRTX_OCCLUSION_DTYPE)
    occlusion["instance_id"] = [6, 7, 8, 9, 10]
    occlusion["ratio"] = [0.0, 0.0683, 0.0, 0.0, 0.3272]
    tokens = np.zeros(6, dtype=OVRTX_INSTANCE_TOKEN_DTYPE)
    tokens[0]["instance_id"] = 5  # record count
    tokens[0]["prim_token"] = (10 << 32) | 6  # max and min renderer instance id
    instance_map = np.zeros(5, dtype=OVRTX_INSTANCE_MAP_DTYPE)
    instance_map["instance_index"] = np.arange(5)
    instance_map["stable_semantic_index"] = [0, 2, 4, 3, 1]
    stable = np.zeros(5, dtype=OVRTX_STABLE_SEMANTIC_ID_DTYPE)
    stable["stable_id"] = _STABLE_IDS
    return {
        "OcclusionSD": occlusion,
        "InstanceIdTokenMapSD": tokens,
        "InstanceMap": instance_map,
        "StableIdSemanticIdMap": stable,
    }


def test_visibility_is_one_minus_the_occlusion_ratio() -> None:
    # Renderer instance 10 -> map index 4 -> table entry 1 -> stable id 2862958192.
    box = _box(instance_id=_STABLE_IDS[1])
    obj = _only_object(_write(_frame(box, **_occlusion_vars())))
    assert obj["visibility"] == pytest.approx(1.0 - 0.3272, abs=1e-3)


def test_visibility_is_one_without_the_occlusion_vars() -> None:
    assert _only_object(_write(_frame()))["visibility"] == 1.0


def test_fully_occluded_object_is_dropped_at_the_default_threshold() -> None:
    extra = _occlusion_vars()
    extra["OcclusionSD"]["ratio"][4] = 1.0
    box = _box(instance_id=_STABLE_IDS[1])
    assert _write(_frame(box, **extra)).written == {}
    kept = _write(_frame(box, **extra), DOPEWriterConfig(visibility_threshold=-1.0))
    assert _only_object(kept)["visibility"] == 0.0
