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
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers import basic
from ovannotators.writers._helpers import OVRTX_CAMERA_PARAMS_DTYPE
from ovannotators.writers._segmentation_palette import SegmentationColorizer
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


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_writer_colorizes_cpu_warp_normals(monkeypatch, channels, dtype):
    import warp as wp

    monkeypatch.setattr(basic, "_warp_device", lambda: wp.get_device("cpu"))
    normals = np.zeros((2, 2, channels), dtype=dtype)
    normals[0, 0, :3] = (1.0, 0.0, -1.0)
    normals[1, 1, :3] = (-1.0, 1.0, 0.0)
    backend = _RecordingBackend()
    writer = BasicWriter(BasicWriterConfig(colorize=True, video=False))
    writer.write_frame(0, {"NormalSD": wp.array(normals, device="cpu")}, backend)
    writer.finalize(backend)

    colorized, encoding = backend.arrays["0000/normal_colorized.png"]
    assert encoding == "png"
    expected = np.array(
        [[[255, 127, 0], [0, 0, 0]], [[0, 0, 0], [0, 255, 127]]], dtype=np.uint8
    )
    np.testing.assert_array_equal(colorized, expected)


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
    colorizer = SegmentationColorizer()
    assert colorizer.colorize(seg, "instance_id").shape == (2, 2, 3)
    # A trailing channel axis (H, W, 1) is accepted and keeps the same colors.
    seg_3d = seg[..., np.newaxis]
    np.testing.assert_array_equal(
        colorizer.colorize(seg_3d, "instance_id"),
        colorizer.colorize(seg, "instance_id"),
    )


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


_NAMED_AOV_GROUPS = (
    ("image", basic._ADDITIONAL_IMAGE_VARS),
    ("exr", basic._ADDITIONAL_EXR_VARS),
    ("npy", basic._ADDITIONAL_RAW_VARS),
)
_NAMED_AOVS = sorted(
    (name, enc) for enc, names in _NAMED_AOV_GROUPS for name in sorted(names)
)


class TestNamedAovMapping:
    def test_all_named_aovs_are_present(self):
        """The three groups hold 98 distinct names and none is dropped."""
        assert [len(g) for _, g in _NAMED_AOV_GROUPS] == [6, 55, 37]
        assert len({name for name, _ in _NAMED_AOVS}) == 98
        for name, enc in _NAMED_AOVS:
            assert name in basic._VAR_FILE_MAP
            assert basic._VAR_FILE_MAP[name][1] == enc

    def test_filename_stems_do_not_collide(self):
        """No two source names share a file stem, so no frame file is overwritten."""
        stems = [stem for stem, _ in basic._VAR_FILE_MAP.values()]
        assert len(stems) == len(set(stems))

    @pytest.mark.parametrize(("name", "enc"), _NAMED_AOVS)
    def test_named_aov_writes_expected_file(self, tmp_path: Path, name: str, enc: str):
        """Each named AOV lands at ``<stem>.<ext>`` for its encoding."""
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        stem = basic._VAR_FILE_MAP[name][0]
        array = {
            "image": np.full((2, 3, 4), 7, dtype=np.uint8),
            "exr": np.full((2, 3, 4), 0.5, dtype=np.float32),
            "npy": np.full((2, 3, 4), 7, dtype=np.uint32),
        }[enc]
        BasicWriter(BasicWriterConfig(colorize=False)).write_frame(
            0, {name: array}, backend
        )
        ext = {"image": "png", "exr": "exr", "npy": "npy"}[enc]
        assert sorted(p.name for p in (tmp_path / "0000").iterdir()) == [
            f"{stem}.{ext}"
        ]


class TestBasicWriterOutputs:
    def test_segmentation_palette_defaults(self):
        config = BasicWriterConfig()

        assert config.segmentation_palette_accessibility == "none"
        assert config.semantic_color_key == "semantic_id"
        assert config.segmentation_palette_path is None

    def test_invalid_segmentation_palette_accessibility_raises(self):
        with pytest.raises(
            ValueError, match="segmentation_palette_accessibility.*'achromatopsia'"
        ):
            BasicWriterConfig(segmentation_palette_accessibility="achromatopsia")

    @pytest.mark.parametrize(
        "mode", ["none", "protanopia", "deuteranopia", "tritanopia"]
    )
    def test_valid_segmentation_palette_accessibility_accepted(self, mode):
        config = BasicWriterConfig(segmentation_palette_accessibility=mode)
        assert config.segmentation_palette_accessibility == mode

    def test_legacy_segmentation_palette_option_is_removed(self):
        with pytest.raises(TypeError, match="segmentation_palette"):
            BasicWriterConfig(segmentation_palette="legacy")

    def test_instance_colors_remain_stable_across_frames(self):
        backend = _RecordingBackend()
        writer = BasicWriter(BasicWriterConfig(colorize=True))
        writer.write_frame(
            0,
            {"SemanticInstanceSegmentation": np.array([[0, 10, 20]])},
            backend,
        )
        writer.write_frame(
            1,
            {"SemanticInstanceSegmentation": np.array([[5, 0, 20]])},
            backend,
        )

        first = backend.arrays["0000/instance_segmentation_colorized.png"][0]
        second = backend.arrays["0001/instance_segmentation_colorized.png"][0]
        np.testing.assert_array_equal(first[0, 2], second[0, 2])

    def test_semantic_label_colors_share_assignments(self):
        backend = _RecordingBackend()
        writer = BasicWriter(
            BasicWriterConfig(colorize=True, semantic_color_key="semantic_label")
        )
        writer.write_frame(
            0,
            {
                "SemanticSegmentation": np.array([[1, 2, 3]], dtype=np.uint32),
                "SemanticIdMap": {
                    1: "class:car",
                    2: "class:car",
                    3: "class:person",
                },
            },
            backend,
        )

        colors = backend.arrays["0000/semantic_segmentation_colorized.png"][0]
        np.testing.assert_array_equal(colors[0, 0], colors[0, 1])
        assert not np.array_equal(colors[0, 0], colors[0, 2])

    def test_stable_id_segmentation_is_colorized(self):
        backend = _RecordingBackend()
        writer = BasicWriter(BasicWriterConfig(colorize=True))
        writer.write_frame(
            0,
            {
                "StableIdSegmentation": np.array(
                    [[[0, 0, 0, 0], [17, 0, 0, 0]]], dtype=np.uint32
                )
            },
            backend,
        )

        assert "0000/stable_id_segmentation_colorized.png" in backend.arrays

    def test_segmentation_palette_persists_typed_assignments(self, tmp_path):
        state_path = tmp_path / "segmentation-palette.json"
        config = BasicWriterConfig(colorize=True, segmentation_palette_path=state_path)
        backend = _RecordingBackend()
        writer = BasicWriter(config)
        writer.write_frame(
            0,
            {
                "SemanticInstanceSegmentation": np.array([[0, 7]], dtype=np.uint32),
                "SemanticSegmentation": np.array([[0, 7]], dtype=np.uint32),
            },
            backend,
        )
        writer.finalize(backend)

        state = json.loads(state_path.read_text())
        assert [item["key_type"] for item in state["assignments"]] == [
            "instance_id",
            "semantic_id",
        ]

        restored_backend = _RecordingBackend()
        restored = BasicWriter(config)
        restored.write_frame(
            0,
            {"SemanticInstanceSegmentation": np.array([[0, 7]], dtype=np.uint32)},
            restored_backend,
        )
        expected = backend.arrays["0000/instance_segmentation_colorized.png"][0]
        actual = restored_backend.arrays["0000/instance_segmentation_colorized.png"][0]
        np.testing.assert_array_equal(actual, expected)

    def test_observed_aov_encodings_preserve_data(self, tmp_path: Path):
        """Image passes are viewable; IDs and structured buffers stay lossless."""
        from ovannotators._internal.codecs import read_exr

        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        color = np.full((3, 4, 4), 123, dtype=np.uint8)
        lighting = np.full((3, 4, 4), 2.5, dtype=np.float16)
        position = np.full((3, 4, 4), 1.0e12, dtype=np.float32)
        ids = np.full((3, 4, 4), 2**32 - 1, dtype=np.uint32)
        records = np.array([(1, 2)], dtype=[("a", "u4"), ("b", "u4")])
        BasicWriter(BasicWriterConfig(colorize=False)).write_frame(
            0,
            {
                "DiffuseAlbedo": color,
                "DiffuseAlbedoSD": color,
                "PtGlobalIllumination": lighting,
                "PtWorldPos": position,
                "StableId": ids,
                "BoundingBox2DTightSD": records,
            },
            backend,
        )
        frame = tmp_path / "0000"
        assert (frame / "diffuse_albedo_native.png").exists()
        assert (frame / "diffuse_albedo.png").exists()
        np.testing.assert_array_equal(
            read_exr(str(frame / "pt_global_illumination.exr")), lighting
        )
        np.testing.assert_array_equal(
            read_exr(str(frame / "pt_world_pos.exr")), position
        )
        np.testing.assert_array_equal(np.load(frame / "stable_id.npy"), ids)
        np.testing.assert_array_equal(
            np.load(frame / "bounding_box_2d_tight_sd.npy"), records
        )

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

    def test_exr_compression_validated_at_config_creation(self):
        """A typo fails when the config is built, not on the first EXR write."""
        with pytest.raises(ValueError, match="Unsupported EXR compression"):
            BasicWriterConfig(exr_compression="bogus")
        with pytest.raises(TypeError, match="must be a string"):
            BasicWriterConfig(exr_compression=3)  # type: ignore[arg-type]
        assert BasicWriterConfig(exr_compression="PIZ").exr_compression == "PIZ"

    def test_exr_compression_keyword_omitted_when_unset(self):
        """A backend without the keyword keeps working when no scheme is set."""
        backend = _RecordingBackend()  # write_array has no exr_compression
        BasicWriter(BasicWriterConfig(colorize=False)).write_frame(
            0,
            {
                "HdrColor": np.ones((4, 4, 4), np.float16),
                "DistanceToCameraSD": np.ones((4, 4, 1), np.float32),
            },
            backend,
        )
        assert {enc for _, enc in backend.arrays.values()} == {"exr"}
        assert len(backend.arrays) == 2

    def test_exr_compression_keyword_forwarded_only_to_exr(self):
        """A set scheme reaches EXR writes and not depth written as NPY."""
        calls: list[tuple[str, str, dict]] = []

        class _KwBackend(_RecordingBackend):
            def write_array(self, path, array, encoding="npy", **kwargs):
                calls.append((path, encoding, kwargs))

        config = BasicWriterConfig(
            colorize=False, depth_format="npy", exr_compression="piz"
        )
        BasicWriter(config).write_frame(
            0,
            {
                "HdrColor": np.ones((4, 4, 4), np.float16),
                "DistanceToCameraSD": np.ones((4, 4, 1), np.float32),
            },
            _KwBackend(),
        )
        by_encoding = {enc: kwargs for _, enc, kwargs in calls}
        assert by_encoding == {"exr": {"exr_compression": "piz"}, "npy": {}}

    def test_exr_compression_applies_to_depth_and_hdr(self, tmp_path: Path):
        """One writer setting reaches both depth-format and direct EXR AOVs."""
        import Imath
        import OpenEXR

        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(exr_compression="piz", colorize=False))
        writer.write_frame(
            0,
            {
                "HdrColor": np.ones((16, 16, 4), np.float16),
                "DistanceToCameraSD": np.ones((16, 16, 1), np.float32),
            },
            backend,
        )
        expected = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
        for name in ("hdr_color.exr", "distance_to_camera.exr"):
            inp = OpenEXR.InputFile(str(tmp_path / "0000" / name))
            assert inp.header()["compression"] == expected
            inp.close()

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


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------


class _FakeVideoEncoder:
    """Stand-in for ``NvVideoEncoder`` recording the frames it was handed."""

    instances: ClassVar[list[_FakeVideoEncoder]] = []

    def __init__(self, fps: int, w: int, h: int):
        self.fps = fps
        self.size = (w, h)
        self.frames: list[np.ndarray] = []
        self.closed = False
        _FakeVideoEncoder.instances.append(self)

    def write_frame(self, array):
        self.frames.append(np.asarray(array))

    def close(self) -> bytes:
        self.closed = True
        # Stand in for a muxed MP4 payload; the byte count encodes frame count
        # so tests can assert frames reached the encoder.
        return b"\x00" * len(self.frames) if self.frames else b""


@pytest.fixture
def fake_video_encoder(monkeypatch):
    """Patch the codec factory so video paths run without PyNvVideoCodec."""
    from ovannotators._internal import codecs

    _FakeVideoEncoder.instances = []

    def _factory(path, fps, w, h, **kwargs):
        return _FakeVideoEncoder(fps, w, h)

    monkeypatch.setattr(codecs, "get_video_encoder", _factory)
    return _FakeVideoEncoder


def _frames(n: int, shape=(64, 64, 3)) -> list[dict]:
    rng = np.random.default_rng(7)
    return [
        {
            "LdrColor": rng.integers(0, 255, shape, dtype=np.uint8),
            "DistanceToCameraSD": rng.random(shape[:2]).astype(np.float32) * 10,
        }
        for _ in range(n)
    ]


class TestBasicWriterVideo:
    def test_video_disabled_by_default(self, tmp_path: Path, fake_video_encoder):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig())
        for i, rv in enumerate(_frames(3)):
            writer.write_frame(i, rv, backend)
        writer.finalize(backend)
        assert fake_video_encoder.instances == []
        assert not (tmp_path / "videos").exists()

    def test_writes_one_mp4_per_video_capable_var(
        self, tmp_path: Path, fake_video_encoder
    ):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(video=True, video_fps=12))
        for i, rv in enumerate(_frames(3)):
            writer.write_frame(i, rv, backend)
        writer.finalize(backend)

        # RGB and the colorized depth visualization each get their own file.
        assert (tmp_path / "videos" / "ldr_color.mp4").exists()
        assert (tmp_path / "videos" / "distance_to_camera_colorized.mp4").exists()
        # Every frame reached its encoder, at the configured fps and resolution.
        assert len(fake_video_encoder.instances) == 2
        for enc in fake_video_encoder.instances:
            assert len(enc.frames) == 3
            assert enc.fps == 12
            assert enc.size == (64, 64)
            assert enc.closed
        # Per-frame files are still written alongside the videos.
        assert (tmp_path / "0000" / "ldr_color.png").exists()

    def test_video_frames_are_rgb_uint8(self, tmp_path: Path, fake_video_encoder):
        """RGBA input is reduced to the 3-channel uint8 layout codecs expect."""
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"]))
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., 0] = 200
        writer.write_frame(0, {"LdrColor": rgba}, backend)
        writer.finalize(backend)
        frame = fake_video_encoder.instances[0].frames[0]
        assert frame.shape == (32, 32, 3)
        assert frame.dtype == np.uint8
        assert frame[0, 0, 0] == 200

    def test_video_vars_filter(self, tmp_path: Path, fake_video_encoder):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(
            BasicWriterConfig(video=True, video_vars=["DistanceToCameraSD"])
        )
        for i, rv in enumerate(_frames(2)):
            writer.write_frame(i, rv, backend)
        writer.finalize(backend)
        assert (tmp_path / "videos" / "distance_to_camera_colorized.mp4").exists()
        assert not (tmp_path / "videos" / "ldr_color.mp4").exists()

    def test_colorized_video_without_colorized_pngs(
        self, tmp_path: Path, fake_video_encoder
    ):
        """colorize=False suppresses the PNGs but still feeds the depth video."""
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(video=True, colorize=False))
        for i, rv in enumerate(_frames(2)):
            writer.write_frame(i, rv, backend)
        writer.finalize(backend)
        assert not (tmp_path / "0000" / "distance_to_camera_colorized.png").exists()
        assert (tmp_path / "videos" / "distance_to_camera_colorized.mp4").exists()

    def test_video_dir_configurable(self, tmp_path: Path, fake_video_encoder):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(
            BasicWriterConfig(video=True, video_vars=["LdrColor"], video_dir="")
        )
        writer.write_frame(0, _frames(1)[0], backend)
        writer.finalize(backend)
        assert (tmp_path / "ldr_color.mp4").exists()

    def test_odd_resolution_cropped_to_even(self, tmp_path: Path, fake_video_encoder):
        """H.264 4:2:0 needs even dimensions, so odd frames are cropped."""
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"]))
        writer.write_frame(0, {"LdrColor": np.zeros((33, 31, 3), np.uint8)}, backend)
        writer.finalize(backend)
        assert fake_video_encoder.instances[0].size == (30, 32)  # (w, h)
        assert fake_video_encoder.instances[0].frames[0].shape == (32, 30, 3)

    def test_resolution_change_mid_capture_skips_frame(
        self, tmp_path: Path, fake_video_encoder, caplog
    ):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"]))
        writer.write_frame(0, {"LdrColor": np.zeros((32, 32, 3), np.uint8)}, backend)
        with caplog.at_level("WARNING"):
            writer.write_frame(
                1, {"LdrColor": np.zeros((16, 16, 3), np.uint8)}, backend
            )
        writer.finalize(backend)
        # The mismatched frame is dropped rather than corrupting the stream.
        assert len(fake_video_encoder.instances) == 1
        assert len(fake_video_encoder.instances[0].frames) == 1
        assert "does not match" in caplog.text

    def test_empty_var_produces_no_video(self, tmp_path: Path, fake_video_encoder):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(video=True))
        writer.write_frame(0, {"LdrColor": np.zeros((0, 0, 3), np.uint8)}, backend)
        writer.finalize(backend)
        assert fake_video_encoder.instances == []
        assert not (tmp_path / "videos").exists()

    def test_finalize_is_idempotent(self, tmp_path: Path, fake_video_encoder):
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"]))
        writer.write_frame(0, _frames(1)[0], backend)
        writer.finalize(backend)
        writer.finalize(backend)
        # A second finalize must not re-close the encoder or rewrite the file.
        assert len(fake_video_encoder.instances) == 1
        assert (tmp_path / "videos" / "ldr_color.mp4").read_bytes() == b"\x00"


@pytest.mark.gpu
@pytest.mark.integration
def test_basic_writer_video_end_to_end(tmp_path: Path):
    """Real NVENC encode produces a non-trivial, MP4-signatured file."""
    pytest.importorskip("PyNvVideoCodec")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
    writer = BasicWriter(BasicWriterConfig(video=True, video_vars=["LdrColor"]))
    rng = np.random.default_rng(3)
    # 256x256, not something smaller: NVENC rejects frames below roughly 160
    # wide at initialization (NV_ENC_ERR_INVALID_PARAM), so a tiny test frame
    # fails in the encoder rather than in anything this test is checking.
    for i in range(8):
        writer.write_frame(
            i,
            {"LdrColor": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)},
            backend,
        )
    writer.finalize(backend)
    data = (tmp_path / "videos" / "ldr_color.mp4").read_bytes()
    assert len(data) > 0
    assert data[4:8] == b"ftyp"
