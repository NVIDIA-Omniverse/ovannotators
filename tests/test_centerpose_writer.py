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

"""Tests for :class:`ovannotators.writers.centerpose.CenterPoseWriter`."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ovannotators.exceptions import ConfigurationError, MissingRenderVarError
from ovannotators.writers import WriterRegistry
from ovannotators.writers.centerpose import CenterPoseWriter, CenterPoseWriterConfig
from ovannotators.writers.dope import DOPEWriter
from tests.test_dope_writer import (
    HEIGHT,
    WIDTH,
    _Backend,
    _box,
    _camera,
    _frame,
    _occlusion_vars,
    _STABLE_IDS,
)

_ORDER = ["Center", "LDB", "LDF", "LUB", "LUF", "RDB", "RDF", "RUB", "RUF"]


def _write(
    frame: dict, config: CenterPoseWriterConfig | None = None, frame_id: int = 3
):
    backend = _Backend()
    CenterPoseWriter(config).write_frame(frame_id, frame, backend)
    return backend


def _only_object(backend: _Backend, stem: str = "000003") -> dict:
    return json.loads(backend.written[f"{stem}.json"])["objects"][0]


def test_registered_under_centerpose() -> None:
    assert CenterPoseWriter.name == "centerpose"
    assert WriterRegistry.get("centerpose") is CenterPoseWriter
    assert CenterPoseWriter.required_render_vars == {
        "LdrColor",
        "SemanticBoundingBox3D",
        "CameraParams",
    }


def test_missing_required_var_raises() -> None:
    frame = _frame()
    del frame["SemanticBoundingBox3D"]
    with pytest.raises(MissingRenderVarError):
        CenterPoseWriter().write_frame(0, frame, _Backend())


def test_annotation_has_centerpose_fields_and_not_dope_fields() -> None:
    backend = _write(_frame())
    assert sorted(backend.written) == ["000003.json", "000003.png"]
    data = json.loads(backend.written["000003.json"])
    assert data["keypoint_order"] == _ORDER
    assert data["camera_data"]["width"] == WIDTH
    assert data["camera_data"]["height"] == HEIGHT
    obj = data["objects"][0]
    assert set(obj) == {
        "label",
        "prim_path",
        "visibility",
        "location",
        "quaternion_xyzw",
        "scale",
        "keypoints_3d",
        "projected_cuboid",
        "truncation_ratio",
    }
    assert obj["label"] == "7"


def test_cuboid_and_camera_frame_keypoints_for_a_box_at_the_origin() -> None:
    obj = _only_object(_write(_frame()))
    # Hand-projected: front face (z=+1) is 19 units away, back face 21.
    assert obj["projected_cuboid"] == [
        [100, 50],  # Center
        [95, 55], [95, 55],  # LDB, LDF
        [95, 45], [95, 45],  # LUB, LUF
        [105, 55], [105, 55],  # RDB, RDF
        [105, 45], [105, 45],  # RUB, RUF
    ]  # fmt: skip
    keypoints = np.asarray(obj["keypoints_3d"])
    # USD camera frame: the box origin is 20 units ahead along -Z.
    np.testing.assert_allclose(keypoints[0], [0, 0, -20], atol=1e-6)  # Center
    np.testing.assert_allclose(keypoints[1], [-1, -1, -21], atol=1e-6)  # LDB
    np.testing.assert_allclose(keypoints[4], [-1, 1, -19], atol=1e-6)  # LUF
    np.testing.assert_allclose(obj["location"], [0, 0, -20], atol=1e-6)


def test_scale_is_local_size_times_world_scale() -> None:
    transform = np.eye(4)
    transform[0, 0], transform[1, 1], transform[2, 2] = 1.0, 2.0, 3.0
    obj = _only_object(_write(_frame(_box(transform=transform))))
    # Local box is 2 units on each axis.
    np.testing.assert_allclose(obj["scale"], [2.0, 4.0, 6.0], atol=1e-6)
    # Rotation is scale-free.
    np.testing.assert_allclose(obj["quaternion_xyzw"], [0, 0, 0, 1], atol=1e-6)


def test_agrees_with_dope_on_the_same_frame() -> None:
    frame = _frame(_box(instance_id=_STABLE_IDS[1]), **_occlusion_vars())
    centerpose = _only_object(_write(frame))
    dope_backend = _Backend()
    DOPEWriter().write_frame(3, frame, dope_backend)
    dope = _only_object(dope_backend)
    for key in ("location", "quaternion_xyzw", "visibility", "truncation_ratio"):
        assert centerpose[key] == pytest.approx(dope[key])
    assert centerpose["label"] == dope["class"]
    # The same keypoints, in each format's order.
    by_name = dict(zip(_ORDER, centerpose["projected_cuboid"]))
    dope_order = ["LUF", "RUF", "RDF", "LDF", "LUB", "RUB", "RDB", "LDB", "Center"]
    assert dope["projected_cuboid"] == [by_name[name] for name in dope_order]


def test_visibility_is_one_minus_the_occlusion_ratio() -> None:
    frame = _frame(_box(instance_id=_STABLE_IDS[1]), **_occlusion_vars())
    assert _only_object(_write(frame))["visibility"] == pytest.approx(
        1.0 - 0.3272, abs=1e-3
    )


def test_object_behind_the_camera_is_skipped() -> None:
    camera = _camera(translation=(0.0, 0.0, 20.0))
    assert _write(_frame(camera=camera)).written == {}


def test_mirrored_box_transform_is_skipped() -> None:
    mirrored = _box(transform=np.diag([1.0, 1.0, -1.0, 1.0]))
    assert _write(_frame(box=mirrored)).written == {}


def test_empty_frame_is_kept_when_asked() -> None:
    camera = _camera(translation=(0.0, 0.0, 20.0))
    kept = _write(
        _frame(camera=camera), CenterPoseWriterConfig(skip_empty_frames=False)
    )
    assert json.loads(kept.written["000003.json"])["objects"] == []


def test_rejects_unknown_image_format_and_non_pinhole_camera() -> None:
    with pytest.raises(ConfigurationError, match="image_format"):
        CenterPoseWriter(CenterPoseWriterConfig(image_format="exr"))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="pinhole"):
        _write(_frame(camera=_camera(model=b"fisheyePolynomial")))
