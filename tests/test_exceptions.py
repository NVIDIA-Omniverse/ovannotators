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

"""Tests for the ovannotators exception hierarchy."""

from __future__ import annotations

import pytest

from ovannotators.exceptions import (
    BackendReadError,
    BackendWriteError,
    ConfigurationError,
    ExtractionError,
    MissingRenderVarError,
    OVAnnotatorsError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# OVAnnotatorsError — base
# ---------------------------------------------------------------------------


def test_ovgta_error_is_exception():
    e = OVAnnotatorsError("something went wrong")
    assert isinstance(e, Exception)
    assert str(e) == "something went wrong"


# ---------------------------------------------------------------------------
# ConfigurationError
# ---------------------------------------------------------------------------


def test_configuration_error_inherits_ovgta():
    e = ConfigurationError("bad config")
    assert isinstance(e, OVAnnotatorsError)
    assert "bad config" in str(e)


# ---------------------------------------------------------------------------
# ExtractionError
# ---------------------------------------------------------------------------


def test_extraction_error_attributes():
    e = ExtractionError(render_var_name="DistanceToCameraSD", frame_id=5)
    assert e.render_var_name == "DistanceToCameraSD"
    assert e.frame_id == 5


def test_extraction_error_message_contains_var_name():
    e = ExtractionError(render_var_name="LdrColor")
    assert "LdrColor" in str(e)


def test_extraction_error_message_contains_frame_id():
    e = ExtractionError(render_var_name="DistanceToCameraSD", frame_id=7)
    assert "7" in str(e)


def test_extraction_error_no_frame_id_default():
    e = ExtractionError(render_var_name="NormalSD")
    assert e.frame_id is None
    assert "None" not in str(e)


def test_extraction_error_custom_message():
    e = ExtractionError(render_var_name="DistanceToCameraSD", message="custom msg")
    assert str(e) == "custom msg"


def test_extraction_error_inherits_ovgta():
    e = ExtractionError(render_var_name="x")
    assert isinstance(e, OVAnnotatorsError)


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


def test_validation_error_attributes():
    e = ValidationError(render_var_name="NormalSD", frame_id=3)
    assert e.render_var_name == "NormalSD"
    assert e.frame_id == 3


def test_validation_error_message_contains_var():
    e = ValidationError(render_var_name="SemanticSegmentation")
    assert "SemanticSegmentation" in str(e)


def test_validation_error_no_frame_id():
    e = ValidationError(render_var_name="x")
    assert e.frame_id is None
    assert "None" not in str(e)


def test_validation_error_custom_message():
    e = ValidationError(render_var_name="x", message="override")
    assert str(e) == "override"


def test_validation_error_inherits_ovgta():
    e = ValidationError(render_var_name="x")
    assert isinstance(e, OVAnnotatorsError)


# ---------------------------------------------------------------------------
# MissingRenderVarError
# ---------------------------------------------------------------------------


def test_missing_render_var_error_attributes():
    e = MissingRenderVarError(
        writer_name="coco", missing_render_var="DistanceToCameraSD"
    )
    assert e.writer_name == "coco"
    assert e.missing_render_var == "DistanceToCameraSD"


def test_missing_render_var_error_message():
    e = MissingRenderVarError(
        writer_name="kitti", missing_render_var="SemanticBoundingBox3D"
    )
    assert "kitti" in str(e)
    assert "SemanticBoundingBox3D" in str(e)


def test_missing_render_var_error_custom_message():
    e = MissingRenderVarError(writer_name="w", missing_render_var="v", message="my msg")
    assert str(e) == "my msg"


def test_missing_render_var_error_inherits_validation_error():
    e = MissingRenderVarError(writer_name="w", missing_render_var="v")
    assert isinstance(e, ValidationError)
    assert isinstance(e, OVAnnotatorsError)
    # render_var_name mirrors the missing var for ValidationError compatibility.
    assert e.render_var_name == "v"


def test_missing_render_var_error_includes_frame_id():
    e = MissingRenderVarError(writer_name="w", missing_render_var="v", frame_id=7)
    assert e.frame_id == 7
    assert "frame 7" in str(e)


# ---------------------------------------------------------------------------
# BackendWriteError
# ---------------------------------------------------------------------------


def test_backend_write_error_attributes():
    e = BackendWriteError(path="images/001.png")
    assert e.path == "images/001.png"


def test_backend_write_error_message():
    e = BackendWriteError(path="images/001.png")
    assert "images/001.png" in str(e)


def test_backend_write_error_custom_message():
    e = BackendWriteError(path="x", message="disk full")
    assert str(e) == "disk full"


def test_backend_write_error_inherits_ovgta():
    e = BackendWriteError(path="x")
    assert isinstance(e, OVAnnotatorsError)


# ---------------------------------------------------------------------------
# BackendReadError
# ---------------------------------------------------------------------------


def test_backend_read_error_attributes():
    e = BackendReadError(path="missing/data.npy")
    assert e.path == "missing/data.npy"


def test_backend_read_error_message():
    e = BackendReadError(path="missing/data.npy")
    assert "missing/data.npy" in str(e)


def test_backend_read_error_custom_message():
    e = BackendReadError(path="x", message="not found")
    assert str(e) == "not found"


def test_backend_read_error_inherits_ovgta():
    e = BackendReadError(path="x")
    assert isinstance(e, OVAnnotatorsError)


# ---------------------------------------------------------------------------
# Hierarchy: all exceptions are catchable as OVAnnotatorsError
# ---------------------------------------------------------------------------


def test_all_exceptions_catchable_as_ovgta_error():
    errors = [
        ConfigurationError("x"),
        ExtractionError("v"),
        ValidationError("v"),
        MissingRenderVarError("w", "v"),
        BackendWriteError("p"),
        BackendReadError("p"),
    ]
    for err in errors:
        with pytest.raises(OVAnnotatorsError):
            raise err
