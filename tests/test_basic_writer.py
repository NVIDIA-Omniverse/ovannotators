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

"""Tests for ovannotators/writers/basic.py.

Covers the JSON encoder, the turbo colormap lookup (matplotlib and grey
fallbacks), the colorizers, the JSON/camera-params payload branches, and the
BasicWriter end-to-end output paths.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers import basic
from ovannotators.writers._helpers import OVRTX_CAMERA_PARAMS_DTYPE
from ovannotators.writers.basic import BasicWriter, BasicWriterConfig


class _RecordingBackend:
    def __init__(self):
        self.written = {}
        self.arrays = {}
        self.dirs = set()

    def write(self, path, data):
        self.written[path] = data

    def write_array(self, path, array, encoding="npy"):
        self.arrays[path] = (array, encoding)

    def read(self, path):
        return self.written[path]

    def list(self, prefix=""):
        return [p for p in {**self.written, **self.arrays} if p.startswith(prefix)]

    def exists(self, path):
        return path in self.written or path in self.arrays

    def makedirs(self, path):
        self.dirs.add(path)


# ===========================================================================
# basic.py — _NumpyEncoder
# ===========================================================================


def test_numpy_encoder_structured_and_scalars():
    enc = basic._NumpyEncoder()
    # 0-d structured array → dict
    dt = np.dtype([("a", "<i4"), ("b", "<f4")])
    zero_d = np.array((3, 1.5), dtype=dt)
    assert enc.default(zero_d) == {"a": 3, "b": pytest.approx(1.5)}
    # 1-d structured array → list of dicts
    one_d = np.zeros(2, dtype=dt)
    one_d[0] = (1, 2.0)
    rows = enc.default(one_d)
    assert isinstance(rows, list) and rows[0]["a"] == 1
    # plain ndarray → list
    assert enc.default(np.array([1, 2, 3])) == [1, 2, 3]
    # numpy integer / floating / bytes scalars
    assert enc.default(np.int64(7)) == 7
    assert enc.default(np.float32(2.5)) == pytest.approx(2.5)
    assert enc.default(np.bytes_(b"hi")) == "hi"


def test_numpy_encoder_falls_through_for_unknown():
    enc = basic._NumpyEncoder()
    with pytest.raises(TypeError):
        enc.default(object())


# ===========================================================================
# basic.py — _turbo_lut matplotlib path
# ===========================================================================


def _install_fake_matplotlib(monkeypatch, *, with_colormaps: bool):
    """Inject a minimal fake ``matplotlib.cm`` module."""

    def _cmap(samples):
        # Return (N, 4) RGBA in [0, 1].
        n = np.asarray(samples).shape[0]
        return np.ones((n, 4), dtype=np.float64) * 0.5

    cm = types.ModuleType("matplotlib.cm")
    if with_colormaps:
        cm.colormaps = {"turbo": _cmap}
    else:
        cm.get_cmap = lambda name: _cmap
    mpl = types.ModuleType("matplotlib")
    mpl.cm = cm
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.cm", cm)


def test_turbo_lut_matplotlib_colormaps(monkeypatch):
    _install_fake_matplotlib(monkeypatch, with_colormaps=True)
    basic._turbo_lut.cache_clear()
    try:
        lut = basic._turbo_lut()
        assert lut.shape == (4096, 3)
    finally:
        basic._turbo_lut.cache_clear()


def test_turbo_lut_matplotlib_get_cmap(monkeypatch):
    _install_fake_matplotlib(monkeypatch, with_colormaps=False)
    basic._turbo_lut.cache_clear()
    try:
        lut = basic._turbo_lut()
        assert lut.shape == (4096, 3)
    finally:
        basic._turbo_lut.cache_clear()


def test_turbo_lut_grey_fallback_without_matplotlib(monkeypatch):
    # Force the ImportError branch even if matplotlib is later installed.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    basic._turbo_lut.cache_clear()
    try:
        lut = basic._turbo_lut()
        assert lut.shape == (4096, 3)
        # Grey ramp: R == G == B.
        assert np.allclose(lut[:, 0], lut[:, 1]) and np.allclose(lut[:, 1], lut[:, 2])
    finally:
        basic._turbo_lut.cache_clear()


# ===========================================================================
# basic.py — Warp colorization (CPU/GPU)
# ===========================================================================


def test_colorize_depth_squeezes_3d_input():
    depth = np.linspace(1.0, 10.0, 64, dtype=np.float32).reshape(8, 8, 1)
    rgb = basic._colorize_depth(depth, near_bright=True)
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8


def test_colorize_depth_all_infinite_returns_zeros():
    all_inf = np.full((5, 5), np.inf, dtype=np.float32)
    out = basic._colorize_depth(all_inf)
    assert out.shape == (5, 5, 3) and np.all(out == 0)


def test_colorize_normals_zero_vectors_map_to_black():
    normals = np.zeros((4, 4, 3), dtype=np.float32)
    normals[0, 0] = (1.0, 0.0, 0.0)
    rgb = basic._colorize_normals(normals)
    assert rgb.shape == (4, 4, 3)
    # Zero-vector normals map to black.
    assert tuple(rgb[1, 1]) == (0, 0, 0)


def test_colorizers_return_expected_shapes():
    depth = np.linspace(1.0, 5.0, 16, dtype=np.float32).reshape(4, 4)
    assert basic._colorize_depth(depth).shape == (4, 4, 3)
    normals = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32) - 0.5
    assert basic._colorize_normals(normals).shape == (4, 4, 3)
    seg = np.array([[0, 1], [2, 3]], dtype=np.int32)
    assert basic._colorize_segmentation(seg).shape == (2, 2, 3)
    # A trailing channel axis (H, W, 1) is squeezed before colorization.
    seg_3d = seg[..., np.newaxis]
    assert basic._colorize_segmentation(seg_3d).shape == (2, 2, 3)


# ===========================================================================
# basic.py — writer helpers
# ===========================================================================


def test_basic_writer_skips_absent_requested_var():
    writer = basic.BasicWriter(basic.BasicWriterConfig(render_vars=["NotPresentVar"]))
    backend = _RecordingBackend()
    writer.write_frame(0, {"LdrColor": np.zeros((2, 2, 3), np.uint8)}, backend)
    # The requested var is absent (skipped via `continue`), so no var files are
    # written — only the frame directory is created.
    assert backend.arrays == {}
    assert "0000" in backend.dirs


def test_basic_write_var_npz_structured_and_plain():
    writer = basic.BasicWriter()
    backend = _RecordingBackend()
    dt = np.dtype([("x", "<i4"), ("y", "<f4")])
    structured = np.zeros(3, dtype=dt)
    writer._write_var("0000", "gmo", "gmo", "npz", structured, backend, {})
    plain = np.arange(6).reshape(2, 3)
    writer._write_var("0000", "plain", "plain", "npz", plain, backend, {})
    # Both wrote .npz payloads.
    assert "0000/gmo.npz" in backend.written
    assert "0000/plain.npz" in backend.written
    # Structured payload round-trips with its field names.
    loaded = np.load(io.BytesIO(backend.written["0000/gmo.npz"]))
    assert set(loaded.files) == {"x", "y"}


def test_to_json_serializable_scalar_structured():
    dt = np.dtype([("a", "<i4"), ("b", "<f4")])
    zero_d = np.array((5, 2.0), dtype=dt)
    out = basic._to_json_serializable(zero_d)
    assert out == {"a": 5, "b": pytest.approx(2.0)}


# ---------------------------------------------------------------------------
# JSON / camera-params payload branches
# ---------------------------------------------------------------------------


class TestBasicJsonBranches:
    def test_array_is_empty_without_shape(self):
        """Objects lacking a ``shape`` attribute are treated as non-empty."""
        assert basic._array_is_empty(object()) is False

    def test_to_json_serializable_one_d_structured(self):
        """A 1-D structured array becomes a list of per-row field dicts."""
        rows = np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "<i4"), ("b", "<f4")])
        assert basic._to_json_serializable(rows) == [
            {"a": 1, "b": pytest.approx(2.0)},
            {"a": 3, "b": pytest.approx(4.0)},
        ]

    def test_format_camera_params_singular_view_transform(self):
        """A non-invertible view transform falls back to identity extrinsics."""
        arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
        arr[0]["view_transform"] = np.zeros((4, 4), dtype=np.float32)  # singular
        arr[0]["projection"] = np.eye(4, dtype=np.float32)
        arr[0]["resolution"] = (64, 48)
        out = basic._format_camera_params(arr)
        assert out["extrinsics"]["R_world_to_camera"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
        assert out["extrinsics"]["t_world_to_camera"] == [0, 0, 0]

    def test_json_payload_generic_array(self):
        """A non-bbox, non-camera var falls through to a generic array dump."""
        payload = basic._json_payload("SomeOtherVar", np.array([1, 2, 3]), {})
        assert payload == [1, 2, 3]

    def test_json_payload_camera_params_from_raw_bytes(self):
        """A raw ``CameraParams`` byte buffer is decoded via the bytes path."""
        struct = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
        struct[0]["view_transform"] = np.eye(4, dtype=np.float32)
        struct[0]["projection"] = np.eye(4, dtype=np.float32)
        struct[0]["resolution"] = (64, 48)
        raw = struct.view(np.uint8)  # not OVRTX_CAMERA_PARAMS_DTYPE anymore
        assert isinstance(basic._json_payload("CameraParams", raw, {}), dict)


# ---------------------------------------------------------------------------
# BasicWriter end-to-end output paths
# ---------------------------------------------------------------------------


class TestBasicWriterOutputs:
    def test_jpg_format(self, tmp_path: Path):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(colorize=True, image_format="jpg"))
        rng = np.random.default_rng(42)
        render_vars = {
            "LdrColor": rng.integers(0, 255, (100, 100, 3), dtype=np.uint8),
            "DistanceToCameraSD": rng.random((100, 100)).astype(np.float32) * 10,
        }
        writer.write_frame(0, render_vars, backend)
        writer.finalize(backend)
        assert (tmp_path / "0000" / "ldr_color.jpg").exists()

    def test_depth_format_npy(self, tmp_path: Path):
        """depth_format='npy' is opt-in; EXR is the default."""
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(depth_format="npy"))
        writer.write_frame(
            0, {"DistanceToCameraSD": np.ones((100, 100), np.float32)}, backend
        )
        writer.finalize(backend)
        assert (tmp_path / "0000" / "distance_to_camera.npy").exists()

    def test_depth_format_exr(self, tmp_path: Path):
        """depth_format='exr' on a single-channel depth map writes an EXR file."""
        pytest.importorskip("OpenEXR")
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(depth_format="exr"))
        writer.write_frame(
            0, {"DistanceToCameraSD": np.ones((100, 100), np.float32) * 5.0}, backend
        )
        assert (tmp_path / "0000" / "distance_to_camera.exr").exists()

    def test_semantic_segmentation_colorized(self, tmp_path: Path):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(colorize=True))
        seg = np.zeros((100, 100), dtype=np.uint32)
        seg[20:50, 20:50] = 1
        seg[60:90, 60:90] = 2
        writer.write_frame(0, {"SemanticSegmentation": seg}, backend)
        writer.finalize(backend)
        assert (tmp_path / "0000" / "semantic_segmentation_colorized.png").exists()

    def test_empty_render_vars(self, tmp_path: Path):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig())
        writer.write_frame(0, {}, backend)
        writer.finalize(backend)
        assert (tmp_path / "0000").is_dir()

    def test_only_normals(self, tmp_path: Path):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(colorize=True))
        normals = (
            np.random.default_rng(42).random((100, 100, 3)).astype(np.float32) * 2 - 1
        )
        writer.write_frame(0, {"NormalSD": normals}, backend)
        writer.finalize(backend)
        assert (tmp_path / "0000" / "normal.exr").exists()
        assert (tmp_path / "0000" / "normal_colorized.png").exists()

    def test_render_vars_filter(self, tmp_path: Path):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(render_vars=["DistanceToCameraSD"]))
        render_vars = {
            "LdrColor": np.zeros((100, 100, 3), dtype=np.uint8),
            "DistanceToCameraSD": np.ones((100, 100), dtype=np.float32),
            "NormalSD": np.ones((100, 100, 3), dtype=np.float32),
        }
        writer.write_frame(0, render_vars, backend)
        writer.finalize(backend)
        assert (tmp_path / "0000" / "distance_to_camera.exr").exists()
        assert not (tmp_path / "0000" / "ldr_color.png").exists()
        assert not (tmp_path / "0000" / "normal.exr").exists()
