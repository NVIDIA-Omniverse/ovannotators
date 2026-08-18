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

"""Tests for OvrtxOutput — mock-based for maximum coverage."""

from __future__ import annotations

import re
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ovannotators.exceptions import ExtractionError, ValidationError
from ovannotators.renderer_outputs.ovrtx import OvrtxOutput

try:
    import ovrtx

    _HAS_OVRTX = True
except ImportError:
    _HAS_OVRTX = False


class _MockDevice(Enum):
    """Stand-in for ovrtx.Device when the real package isn't installed."""

    CPU = "cpu"
    CUDA = "cuda"


class TestOvrtxOutput:
    """Tests for OvrtxOutput with mocked ovrtx frame output."""

    def test_renderer_name(self):
        out = OvrtxOutput()
        assert out.renderer_name == "ovrtx"

    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_missing_var_raises_validation_error(self):
        out = OvrtxOutput()
        mock_frame = MagicMock()
        mock_rv = MagicMock()
        mock_rv.get.return_value = None
        mock_frame.render_vars = mock_rv
        with pytest.raises(ValidationError) as exc_info:
            out.extract(mock_frame, "nonexistent_var")
        assert exc_info.value.render_var_name == "nonexistent_var"
        # An absent var is a validation problem, not an extraction/decode failure.
        assert not isinstance(exc_info.value, ExtractionError)

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_map_failure_raises_extraction_error(self):
        out = OvrtxOutput()
        mock_frame = MagicMock()
        raw_var = MagicMock()
        raw_var.map.side_effect = RuntimeError("malformed buffer")
        mock_frame.render_vars = {"LdrColor": raw_var}
        with pytest.raises(ExtractionError) as exc_info:
            out.extract(mock_frame, "LdrColor")
        assert exc_info.value.render_var_name == "LdrColor"
        assert "malformed buffer" in str(exc_info.value)

    def test_extract_invalid_device_raises(self):
        out = OvrtxOutput()
        with patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True):
            with pytest.raises(ValueError, match="Unsupported device"):
                out.extract(MagicMock(), "x", device="tpu")  # type: ignore[arg-type]

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_returns_warp_array(self):
        """CPU extract returns the zero-copy Warp view without cloning."""
        out = OvrtxOutput()
        mock_mapping = MagicMock()
        mock_mapping.__len__.return_value = 1
        mock_mapping.params = {}

        @contextmanager
        def mock_map(device):
            yield mock_mapping

        mock_var = MagicMock()
        mock_var.map = mock_map

        mock_rv = MagicMock()
        mock_rv.get.return_value = mock_var
        mock_rv.keys.return_value = ["LdrColor"]
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        with patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp:
            fake_warp = MagicMock()
            mock_wp.from_dlpack.return_value = fake_warp
            result = out.extract(mock_frame, "LdrColor", device="cpu")
            # Zero-copy: the from_dlpack view is returned directly, not cloned.
            assert result is fake_warp
            mock_wp.from_dlpack.assert_called_once_with(mock_mapping)
            mock_wp.clone.assert_not_called()
            # CPU mappings are valid on map() return; no wait() needed.
            mock_mapping.wait.assert_not_called()

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_cuda_uses_warp_from_dlpack(self):
        out = OvrtxOutput()
        mock_mapping = MagicMock()
        mock_mapping.__len__.return_value = 1
        mock_mapping.params = {}

        @contextmanager
        def mock_map(device):
            yield mock_mapping

        mock_var = MagicMock()
        mock_var.map = mock_map

        mock_rv = MagicMock()
        mock_rv.get.return_value = mock_var
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        with patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp:
            fake_warp = MagicMock()
            mock_wp.from_dlpack.return_value = fake_warp
            result = out.extract(mock_frame, "DistanceToCameraSD", device="cuda")
            assert result is fake_warp
            mock_wp.from_dlpack.assert_called_once_with(mock_mapping)
            mock_wp.array.assert_not_called()
            mock_wp.clone.assert_not_called()
            # CUDA mappings must be synchronized before the view is read.
            mock_mapping.wait.assert_called_once_with()

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_cpu_uses_warp_from_dlpack(self):
        """Verify CPU extraction stays in Warp and avoids NumPy wrapping."""
        out = OvrtxOutput()
        mock_mapping = MagicMock()
        mock_mapping.__len__.return_value = 1
        mock_mapping.params = {}

        @contextmanager
        def mock_map(device):
            yield mock_mapping

        mock_var = MagicMock()
        mock_var.map = mock_map

        mock_rv = MagicMock()
        mock_rv.get.return_value = mock_var
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        with patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp:
            out.extract(mock_frame, "DistanceToCameraSD", device="cpu")
            mock_wp.from_dlpack.assert_called_once_with(mock_mapping)
            mock_wp.array.assert_not_called()

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_multitensor_returns_nested_dict(self):
        """Multi-tensor/composite vars return a nested tensors+params dict."""
        out = OvrtxOutput()
        coord_tensor = object()
        intensity_tensor = object()
        frame_param = object()
        mock_mapping = MagicMock()
        mock_mapping.__len__.return_value = 2
        mock_mapping.keys.return_value = ["Coordinates", "Intensity"]
        tensor_views = {"Coordinates": coord_tensor, "Intensity": intensity_tensor}
        mock_mapping.__getitem__.side_effect = lambda name: tensor_views[name]
        mock_mapping.params = {"frameId": frame_param}

        @contextmanager
        def mock_map(device):
            yield mock_mapping

        mock_var = MagicMock()
        mock_var.map = mock_map

        mock_rv = MagicMock()
        mock_rv.get.return_value = mock_var
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        coord_warp = object()
        intensity_warp = object()
        frame_np = object()
        with (
            patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp,
            patch("ovannotators.renderer_outputs.ovrtx.np") as mock_np,
        ):
            mock_wp.from_dlpack.side_effect = lambda t: {
                coord_tensor: coord_warp,
                intensity_tensor: intensity_warp,
            }[t]
            mock_np.from_dlpack.side_effect = lambda p: {frame_param: frame_np}[p]
            result = out.extract(mock_frame, "PointCloud", device="cpu")

        assert result == {
            "tensors": {"Coordinates": coord_warp, "Intensity": intensity_warp},
            "params": {"frameId": frame_np},
        }
        mock_mapping.wait.assert_not_called()

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_multitensor_cuda_waits(self):
        """CUDA multi-tensor extraction syncs once before tensor access."""
        out = OvrtxOutput()
        coord_tensor = object()
        frame_param = object()
        mock_mapping = MagicMock()
        mock_mapping.__len__.return_value = 1
        mock_mapping.keys.return_value = ["Coordinates"]
        mock_mapping.__getitem__.side_effect = lambda name: coord_tensor
        mock_mapping.params = {"frameId": frame_param}

        @contextmanager
        def mock_map(device):
            yield mock_mapping

        mock_var = MagicMock()
        mock_var.map = mock_map

        mock_rv = MagicMock()
        mock_rv.get.return_value = mock_var
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        with (
            patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp,
            patch("ovannotators.renderer_outputs.ovrtx.np") as mock_np,
        ):
            mock_wp.from_dlpack.return_value = object()
            mock_np.from_dlpack.return_value = object()
            result = out.extract(mock_frame, "PointCloud", device="cuda")

        # Single tensor but with params -> nested dict, and wait() ran first.
        assert set(result.keys()) == {"tensors", "params"}
        mock_mapping.wait.assert_called_once_with()

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_all_returns_dict(self):
        out = OvrtxOutput()
        arrays = {
            "LdrColor": np.zeros((64, 64, 4), dtype=np.uint8),
            "DistanceToCameraSD": np.ones((64, 64), dtype=np.float32),
        }

        def make_mock_var(arr):
            mock_mapping = MagicMock()
            mock_mapping.__len__.return_value = 1
            mock_mapping.params = {}

            @contextmanager
            def mock_map(device):
                yield mock_mapping

            v = MagicMock()
            v.map = mock_map
            return v

        mock_rv = MagicMock()

        def side_effect(k):
            return make_mock_var(arrays[k]) if k in arrays else None

        mock_rv.get.side_effect = side_effect
        mock_rv.keys.return_value = ["LdrColor", "DistanceToCameraSD"]
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        with patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp:
            mock_wp.from_dlpack.side_effect = [
                arrays["LdrColor"],
                arrays["DistanceToCameraSD"],
            ]
            result = out.extract_all(mock_frame, ["LdrColor", "DistanceToCameraSD"])
            assert isinstance(result, dict)
            assert set(result.keys()) == {"LdrColor", "DistanceToCameraSD"}
            assert result["LdrColor"].shape == (64, 64, 4)
            assert result["DistanceToCameraSD"].shape == (64, 64)

    def test_extract_all_forwards_device(self):
        out = OvrtxOutput()
        mock_frame = MagicMock()
        with patch.object(out, "extract", return_value=MagicMock()) as mock_extract:
            out.extract_all(mock_frame, ["a", "b"], device="cuda")
            mock_extract.assert_any_call(mock_frame, "a", device="cuda", decode=True)
            mock_extract.assert_any_call(mock_frame, "b", device="cuda", decode=True)

    def test_extract_all_without_var_names_uses_available_vars(self):
        out = OvrtxOutput()
        mock_frame = MagicMock()
        with (
            patch.object(
                out, "available_vars", return_value=["a", "b"]
            ) as mock_available,
            patch.object(out, "extract", return_value=MagicMock()) as mock_extract,
        ):
            result = out.extract_all(mock_frame, device="cuda")
            assert set(result.keys()) == {"a", "b"}
            mock_available.assert_called_once_with(mock_frame)
            mock_extract.assert_any_call(mock_frame, "a", device="cuda", decode=True)
            mock_extract.assert_any_call(mock_frame, "b", device="cuda", decode=True)

    def test_extract_all_without_var_names_skips_unprocessable_var(self):
        out = OvrtxOutput()
        mock_frame = MagicMock()

        def _extract(frame, var_name, device="cpu", decode=True):
            if var_name == "bad":
                raise RuntimeError("cannot map")
            return MagicMock()

        with (
            patch.object(out, "available_vars", return_value=["a", "bad", "b"]),
            patch.object(out, "extract", side_effect=_extract),
        ):
            result = out.extract_all(mock_frame)

        assert set(result.keys()) == {"a", "b"}

    def test_extract_all_with_explicit_var_names_raises_on_failure(self):
        out = OvrtxOutput()
        mock_frame = MagicMock()

        with patch.object(out, "extract", side_effect=RuntimeError("cannot map")):
            with pytest.raises(RuntimeError, match="cannot map"):
                out.extract_all(mock_frame, ["bad"])

    def test_available_vars(self):
        out = OvrtxOutput()
        mock_frame = MagicMock()
        mock_frame.render_vars.keys.return_value = [
            "LdrColor",
            "DistanceToCameraSD",
            "NormalSD",
        ]

        result = out.available_vars(mock_frame)
        assert set(result) == {"LdrColor", "DistanceToCameraSD", "NormalSD"}

    def test_extract_ovrtx_not_installed(self):
        """Test import error path when ovrtx is not available."""
        out = OvrtxOutput()
        with patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", False):
            with pytest.raises(ImportError, match="ovrtx"):
                out.extract(MagicMock(), "x")


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_OVRTX, reason="ovrtx not installed")
class TestOvrtxOutputIntegration:
    """Integration coverage for OvrtxOutput against real ovrtx frames."""

    _SCENE_PATH = Path(__file__).parent / "data" / "test_scene.usda"
    _RENDER_PRODUCT = "/Render/RenderProduct_001"

    def test_extract_ldr_color_from_real_frame_cpu(self):
        renderer = ovrtx.Renderer()
        renderer.open_usd(str(self._SCENE_PATH.resolve()))
        outputs = renderer.step({self._RENDER_PRODUCT}, 1.0 / 60.0)
        frame = outputs[self._RENDER_PRODUCT].frames[0]

        extracted = OvrtxOutput().extract(frame, "LdrColor", device="cpu")
        array = extracted.numpy() if hasattr(extracted, "numpy") else extracted

        match = re.search(
            r"uniform int2 resolution = \((\d+),\s*(\d+)\)",
            self._SCENE_PATH.read_text(),
        )
        assert match is not None, "RenderProduct resolution not found in test scene"
        width, height = int(match.group(1)), int(match.group(2))

        assert array.ndim == 3
        assert array.shape[:2] == (height, width)
        assert array.shape[2] in (3, 4)
        assert array.dtype == np.uint8


# ---------------------------------------------------------------------------
# extract() — annotator decoding and structured-error re-raise
# ---------------------------------------------------------------------------


def test_annotator_decoder_catalog_covers_runtime_verified_maps() -> None:
    import ovannotators.renderer_outputs.ovrtx as ovrtx_output

    with patch.object(ovrtx_output, "_ANNOTATOR_DECODERS", None):
        decoders = ovrtx_output._annotator_decoders()

    assert {
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox2DLoose",
        "SemanticBoundingBox3D",
        "SemanticIdMap",
        "StableIdMap",
        "StableIdSemanticIdMap",
        "InstanceMap",
        "CameraParams",
    } <= decoders.keys()


class TestOvrtxExtractDecodeAndErrors:
    """Cover the annotator-decode and error-passthrough branches of extract()."""

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_decodes_annotator_var(self):
        """An annotator var routes the mapped result through its decoder."""
        out = OvrtxOutput()
        mock_mapping = MagicMock()
        mock_mapping.__len__.return_value = 1
        mock_mapping.params = {}

        @contextmanager
        def mock_map(device):
            yield mock_mapping

        mock_var = MagicMock()
        mock_var.map = mock_map
        mock_rv = MagicMock()
        mock_rv.get.return_value = mock_var
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        sentinel = object()
        with (
            patch("ovannotators.renderer_outputs.ovrtx.wp") as mock_wp,
            patch(
                "ovannotators.renderer_outputs.ovrtx._annotator_decoders",
                return_value={"CameraParams": lambda r: ("decoded", r)},
            ),
        ):
            mock_wp.from_dlpack.return_value = sentinel
            result = out.extract(mock_frame, "CameraParams", device="cpu", decode=True)

        assert result == ("decoded", sentinel)

    @patch("ovannotators.renderer_outputs.ovrtx.Device", _MockDevice)
    @patch("ovannotators.renderer_outputs.ovrtx._OVRTX_AVAILABLE", True)
    def test_extract_reraises_structured_error(self):
        """A structured (de)mapping error propagates without being re-wrapped."""
        out = OvrtxOutput()

        @contextmanager
        def mock_map(device):
            raise ExtractionError(render_var_name="LdrColor")
            yield  # unreachable; required for the generator contextmanager

        mock_var = MagicMock()
        mock_var.map = mock_map
        mock_rv = MagicMock()
        mock_rv.get.return_value = mock_var
        mock_frame = MagicMock()
        mock_frame.render_vars = mock_rv

        with pytest.raises(ExtractionError) as exc_info:
            out.extract(mock_frame, "LdrColor", device="cpu")
        # Re-raised verbatim: same render var, not wrapped with a new message.
        assert exc_info.value.render_var_name == "LdrColor"


def test_ovrtx_module_imports_device_when_ovrtx_available():
    """The module-level ``from ovrtx import Device`` runs when ovrtx imports.

    Under the GPU-less import shim ``import ovrtx`` fails, so this reloads the
    module with a fake ``ovrtx`` importable (mirroring real CI with ovrtx
    installed) to exercise the success branch of the top-level import guard.
    """
    import builtins
    import importlib
    import sys
    import types

    import ovannotators.renderer_outputs.ovrtx as ovrtx_mod
    from ovannotators._internal import optional_deps

    fake_ovrtx = types.ModuleType("ovrtx")
    fake_ovrtx.Renderer = object
    fake_ovrtx.Device = _MockDevice

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ovrtx" or name.startswith("ovrtx."):
            return sys.modules["ovrtx"]
        return real_import(name, *args, **kwargs)

    optional_deps.ovrtx_available.cache_clear()
    try:
        with (
            patch.dict(sys.modules, {"ovrtx": fake_ovrtx}),
            patch("builtins.__import__", fake_import),
        ):
            importlib.reload(ovrtx_mod)
            assert ovrtx_mod.ovrtx is fake_ovrtx
            assert ovrtx_mod.Device is _MockDevice
    finally:
        # Restore the module to its real (ovrtx-absent) state for other tests.
        optional_deps.ovrtx_available.cache_clear()
        importlib.reload(ovrtx_mod)
