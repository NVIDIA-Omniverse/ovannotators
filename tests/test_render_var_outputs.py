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

"""Tests for the renderer_outputs layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ovannotators.exceptions import ConfigurationError, ValidationError
from ovannotators.renderer_outputs._registry import RendererOutputRegistry
from ovannotators.renderer_outputs.ovrtx import OvrtxOutput


# ---------------------------------------------------------------------------
# RendererOutputRegistry
# ---------------------------------------------------------------------------


def test_registry_list_names_contains_ovrtx():
    assert "ovrtx" in RendererOutputRegistry.list_names()


def test_registry_get_ovrtx_returns_class():
    cls = RendererOutputRegistry.get("ovrtx")
    assert cls is OvrtxOutput


def test_registry_get_unknown_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="Unknown renderer output"):
        RendererOutputRegistry.get("_nonexistent_renderer_xyz")


def test_registry_create_returns_instance():
    instance = RendererOutputRegistry.create("ovrtx")
    assert isinstance(instance, OvrtxOutput)


def test_registry_get_error_lists_available():
    try:
        RendererOutputRegistry.get("_nonexistent_renderer_xyz")
    except ConfigurationError as e:
        assert "ovrtx" in str(e)


def test_registry_register_and_retrieve():
    class _FakeRenderer:
        renderer_name = "_test_renderer_abc"

        def extract(self, frame_output, var_name):
            return np.zeros((1,))

        def extract_all(self, frame_output, var_names=None):
            if var_names is None:
                var_names = []
            return {n: np.zeros((1,)) for n in var_names}

        def available_vars(self, frame_output):
            return []

    try:
        RendererOutputRegistry.register(_FakeRenderer)
        assert RendererOutputRegistry.get("_test_renderer_abc") is _FakeRenderer
        assert "_test_renderer_abc" in RendererOutputRegistry.list_names()
    finally:
        RendererOutputRegistry._registry.pop("_test_renderer_abc", None)


def test_registry_register_returns_class():
    class _FakeRenderer2:
        renderer_name = "_test_renderer_def"

        def extract(self, frame_output, var_name):
            pass

        def extract_all(self, frame_output, var_names=None):
            pass

        def available_vars(self, frame_output):
            pass

    try:
        result = RendererOutputRegistry.register(_FakeRenderer2)
        assert result is _FakeRenderer2
    finally:
        RendererOutputRegistry._registry.pop("_test_renderer_def", None)


def test_registry_list_names_is_sorted():
    names = RendererOutputRegistry.list_names()
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# OvrtxOutput
# ---------------------------------------------------------------------------


def test_ovrtx_output_renderer_name():
    assert OvrtxOutput.renderer_name == "ovrtx"


def test_ovrtx_output_instance_renderer_name():
    output = OvrtxOutput()
    assert output.renderer_name == "ovrtx"


def test_ovrtx_extract_raises_import_error_when_ovrtx_absent():
    output = OvrtxOutput()
    mock_frame = MagicMock()
    with patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", False):
        with pytest.raises(ImportError, match="ovrtx is not installed"):
            output.extract(mock_frame, "LdrColor")


def test_ovrtx_extract_raises_validation_error_when_var_missing():
    output = OvrtxOutput()
    mock_frame = MagicMock()
    mock_frame.render_vars.get.return_value = None

    with (
        patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True),
        patch("ovannotators.renderer_outputs.ovrtx.Device", MagicMock()),
    ):
        with pytest.raises(ValidationError) as exc_info:
            output.extract(mock_frame, "missing_var")

    assert exc_info.value.render_var_name == "missing_var"


def test_ovrtx_extract_uses_warp_from_dlpack_for_cpu():
    """extract() should return the zero-copy Warp view for CPU, without cloning."""
    output = OvrtxOutput()
    mock_raw_var = MagicMock()
    mock_mapping = MagicMock()
    mock_mapping.__enter__ = MagicMock(return_value=mock_mapping)
    mock_mapping.__exit__ = MagicMock(return_value=False)
    mock_mapping.__len__.return_value = 1
    mock_mapping.params = {}
    mock_raw_var.map.return_value = mock_mapping

    mock_frame = MagicMock()
    mock_frame.render_vars.get.return_value = mock_raw_var

    mock_device = MagicMock()
    fake_wp = MagicMock()

    with (
        patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True),
        patch("ovannotators.renderer_outputs.ovrtx.Device", mock_device),
        patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp,
    ):
        mock_wp.from_dlpack.return_value = fake_wp
        result = output.extract(mock_frame, "LdrColor", device="cpu")

    mock_wp.from_dlpack.assert_called_once_with(mock_mapping)
    # Zero-copy: the view is returned directly and the buffer is not cloned.
    mock_wp.clone.assert_not_called()
    # CPU mappings are valid on map() return; no GPU sync is issued.
    mock_mapping.wait.assert_not_called()
    assert result is fake_wp


def test_ovrtx_available_vars_returns_keys():
    output = OvrtxOutput()
    mock_frame = MagicMock()
    mock_frame.render_vars.keys.return_value = [
        "LdrColor",
        "DistanceToCameraSD",
        "NormalSD",
    ]

    result = output.available_vars(mock_frame)
    assert set(result) == {"LdrColor", "DistanceToCameraSD", "NormalSD"}


def test_ovrtx_extract_all_calls_extract_per_var():
    output = OvrtxOutput()
    mock_frame = MagicMock()
    fake = np.zeros((2, 2), dtype=np.float32)

    with patch.object(output, "extract", return_value=fake) as mock_extract:
        result = output.extract_all(mock_frame, ["LdrColor", "DistanceToCameraSD"])

    assert set(result.keys()) == {"LdrColor", "DistanceToCameraSD"}
    assert mock_extract.call_count == 2


def test_ovrtx_extract_all_returns_dict_keyed_by_var_name():
    output = OvrtxOutput()
    mock_frame = MagicMock()
    counter = [0]

    def _fake_extract(frame, var_name, device="cpu", decode=True):
        arr = np.full((2, 2), counter[0], dtype=np.float32)
        counter[0] += 1
        return arr

    with patch.object(output, "extract", side_effect=_fake_extract):
        result = output.extract_all(mock_frame, ["a", "b", "c"])

    assert "a" in result
    assert "b" in result
    assert "c" in result


def test_ovrtx_extract_all_without_var_names_uses_available_vars():
    output = OvrtxOutput()
    mock_frame = MagicMock()
    fake = np.zeros((2, 2), dtype=np.float32)

    with (
        patch.object(
            output, "available_vars", return_value=["a", "b"]
        ) as mock_available,
        patch.object(output, "extract", return_value=fake) as mock_extract,
    ):
        result = output.extract_all(mock_frame)

    assert set(result.keys()) == {"a", "b"}
    mock_available.assert_called_once_with(mock_frame)
    mock_extract.assert_any_call(mock_frame, "a", device="cuda", decode=True)
    mock_extract.assert_any_call(mock_frame, "b", device="cuda", decode=True)
