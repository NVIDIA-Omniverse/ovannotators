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

"""Unit tests for CosmosTransferWriter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ovannotators.exceptions import MissingRenderVarError
from ovannotators.writers._registry import WriterRegistry
from ovannotators.writers._segmentation_palette import (
    DistinctColorPalette,
    SegmentationColorizer,
    _simulate_accessibility_oklab,
    _srgb_to_oklab,
)
from ovannotators.writers.cosmos_transfer import (
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend():
    """Mock backend that records all write calls."""
    backend = MagicMock()
    backend.written = {}

    def record_write(path, data):
        backend.written[path] = data

    backend.write.side_effect = record_write
    return backend


@pytest.fixture
def sample_render_vars():
    """Produce a synthetic frame of render vars at 64x48 (ovrtx sourceNames)."""
    h, w = 48, 64
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32) * 10.0
    seg = np.random.randint(0, 10, (h, w), dtype=np.int32)
    return {
        "LdrColor": rgb,
        "DistanceToCameraSD": depth,
        "SemanticInstanceSegmentation": seg,
    }


@pytest.fixture
def small_config():
    """Config for fast tests (output resolution follows the input frames)."""
    return CosmosTransferWriterConfig(
        fps=10,
        clip_index=0,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_writer_registered(self):
        assert "cosmos_transfer" in WriterRegistry.list_names()

    def test_get_returns_class(self):
        cls = WriterRegistry.get("cosmos_transfer")
        assert cls is CosmosTransferWriter

    def test_name_attribute(self):
        assert CosmosTransferWriter.name == "cosmos_transfer"

    def test_required_render_vars(self):
        expected = frozenset(
            {"LdrColor", "DistanceToCameraSD", "SemanticInstanceSegmentation"}
        )
        assert CosmosTransferWriter.required_render_vars == expected


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = CosmosTransferWriterConfig()
        assert cfg.fps == 24
        assert cfg.clip_index == 0
        assert cfg.conditioning_types == ["canny", "depth", "segmentation"]
        assert cfg.canny_source == "rgb"
        assert cfg.canny_sigma == 1.0
        assert cfg.segmentation_palette_accessibility == "none"
        assert cfg.segmentation_color_key == "instance_id"
        assert cfg.segmentation_palette_path is None

    def test_custom(self):
        cfg = CosmosTransferWriterConfig(
            fps=30,
            clip_index=5,
            conditioning_types=["canny"],
        )
        assert cfg.fps == 30
        assert cfg.clip_index == 5
        assert cfg.conditioning_types == ["canny"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("segmentation_palette_accessibility", "achromatopsia"),
            ("segmentation_color_key", "prim_path"),
        ],
    )
    def test_invalid_segmentation_palette_options(self, field, value):
        with pytest.raises(ValueError, match=field):
            CosmosTransferWriterConfig(**{field: value})

    def test_legacy_segmentation_palette_option_is_removed(self):
        with pytest.raises(TypeError, match="segmentation_palette"):
            CosmosTransferWriterConfig(segmentation_palette="legacy")

    @pytest.mark.parametrize(
        ("color_key", "render_var"),
        [
            ("instance_id", "SemanticInstanceSegmentation"),
            ("stable_id", "StableIdSegmentation"),
            ("semantic_id", "SemanticSegmentation"),
            ("semantic_label", "SemanticSegmentation"),
        ],
    )
    def test_color_key_selects_required_segmentation_var(self, color_key, render_var):
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(segmentation_color_key=color_key)
        )

        assert render_var in writer.required_render_vars
        assert (
            len(
                writer.required_render_vars
                & {
                    "SemanticInstanceSegmentation",
                    "StableIdSegmentation",
                    "SemanticSegmentation",
                }
            )
            == 1
        )


class TestConstructorValidation:
    def test_unknown_conditioning_type_raises(self):
        cfg = CosmosTransferWriterConfig(conditioning_types=["canny", "bogus"])
        with pytest.raises(ValueError, match="conditioning_types"):
            CosmosTransferWriter(cfg)

    def test_unsupported_canny_source_raises(self):
        cfg = CosmosTransferWriterConfig(canny_source="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="canny_source"):
            CosmosTransferWriter(cfg)

    def test_valid_config_narrows_required_vars(self):
        cfg = CosmosTransferWriterConfig(conditioning_types=["canny"])
        writer = CosmosTransferWriter(cfg)
        assert writer.required_render_vars == frozenset({"LdrColor"})


# ---------------------------------------------------------------------------
# Frame processing helpers
# ---------------------------------------------------------------------------


class TestFrameProcessing:
    def test_resize_noop(self):
        writer = CosmosTransferWriter(CosmosTransferWriterConfig())
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        result = writer._resize_frame(frame, 64, 48)
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8

    def test_resize_upscale(self):
        writer = CosmosTransferWriter(CosmosTransferWriterConfig())
        frame = np.ones((48, 64, 3), dtype=np.uint8) * 128
        result = writer._resize_frame(frame, 128, 96)
        assert result.shape == (96, 128, 3)
        assert result.dtype == np.uint8

    def test_canny_computation(self):
        """Canny should produce 3-channel uint8 edge maps."""
        cfg = CosmosTransferWriterConfig()
        writer = CosmosTransferWriter(cfg)
        # Create a frame with a clear edge
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, 32:, :] = 255  # sharp vertical edge
        writer._rgb_frames = [frame]
        canny_frames = writer._compute_canny_frames()
        assert len(canny_frames) == 1
        assert canny_frames[0].shape == (48, 64, 3)
        assert canny_frames[0].dtype == np.uint8
        # Should have some edge pixels
        assert canny_frames[0].max() == 255

    def test_canny_from_normals(self):
        """canny_source='normals' derives edges from the NormalSD field."""
        cfg = CosmosTransferWriterConfig(canny_source="normals")
        writer = CosmosTransferWriter(cfg)
        # Two half-spaces with orthogonal surface normals → seam at x=32.
        normals = np.zeros((48, 64, 4), dtype=np.float32)
        normals[:, :32, 0] = 1.0  # +X facing
        normals[:, 32:, 1] = 1.0  # +Y facing
        writer._normal_frames = [normals]
        canny = writer._compute_canny_frames()
        assert len(canny) == 1
        assert canny[0].shape == (48, 64, 3)
        assert canny[0].dtype == np.uint8
        assert canny[0].max() == 255

    def test_canny_from_segmentation(self):
        """canny_source='segmentation' derives edges from object boundaries.

        Uses large instance IDs (not small contiguous labels) to confirm
        boundaries are detected regardless of the raw ID magnitude.
        """
        cfg = CosmosTransferWriterConfig(canny_source="segmentation")
        writer = CosmosTransferWriter(cfg)
        seg = np.zeros((48, 64), dtype=np.int64)
        seg[:, 32:] = 3_319_052_530  # large instance id on one side
        writer._seg_frames = [seg]
        canny = writer._compute_canny_frames()
        assert len(canny) == 1
        assert canny[0].shape == (48, 64, 3)
        assert canny[0].dtype == np.uint8
        assert canny[0].max() == 255

    def test_canny_source_normals_missing_normalsd_raises(
        self, mock_backend, sample_render_vars
    ):
        """Selecting normals as the canny source requires NormalSD."""
        cfg = CosmosTransferWriterConfig(canny_source="normals")
        writer = CosmosTransferWriter(cfg)
        with pytest.raises(MissingRenderVarError) as exc_info:
            writer.write_frame(0, sample_render_vars, mock_backend)
        assert exc_info.value.missing_render_var == "NormalSD"

    def test_canny_source_segmentation_captures_seg_without_seg_conditioning(
        self, mock_backend, sample_render_vars
    ):
        """Seg frames are captured for seg-sourced canny even when the
        segmentation conditioning video is not requested."""
        cfg = CosmosTransferWriterConfig(
            conditioning_types=["canny"],
            canny_source="segmentation",
        )
        writer = CosmosTransferWriter(cfg)
        writer.write_frame(0, sample_render_vars, mock_backend)
        assert len(writer._seg_frames) == 1
        assert len(writer._normal_frames) == 0

    def test_canny_from_shaded_segmentation(self):
        """canny_source='shaded_segmentation' derives edges from the
        normal-shaded segmentation (object boundary + shading relief)."""
        cfg = CosmosTransferWriterConfig(canny_source="shaded_segmentation")
        writer = CosmosTransferWriter(cfg)
        seg = np.zeros((48, 64), dtype=np.int64)
        seg[:, 32:] = 7
        normals = np.zeros((48, 64, 3), dtype=np.float32)
        normals[..., 2] = 1.0  # all facing camera
        writer._seg_frames = [seg]
        writer._normal_frames = [normals]
        canny = writer._compute_canny_frames()
        assert len(canny) == 1
        assert canny[0].shape == (48, 64, 3)
        assert canny[0].dtype == np.uint8
        assert canny[0].max() == 255

    def test_shaded_seg_captures_seg_and_normals(
        self, mock_backend, sample_render_vars
    ):
        """shaded_segmentation conditioning captures both seg and normals."""
        cfg = CosmosTransferWriterConfig(
            conditioning_types=["shaded_segmentation"],
        )
        writer = CosmosTransferWriter(cfg)
        render_vars = {
            **sample_render_vars,
            "NormalSD": np.zeros((48, 64, 3), dtype=np.float32),
        }
        writer.write_frame(0, render_vars, mock_backend)
        assert len(writer._seg_frames) == 1
        assert len(writer._normal_frames) == 1

    def test_shaded_seg_missing_normalsd_raises(self, mock_backend, sample_render_vars):
        """shaded_segmentation conditioning requires NormalSD."""
        cfg = CosmosTransferWriterConfig(
            conditioning_types=["shaded_segmentation"],
        )
        writer = CosmosTransferWriter(cfg)
        with pytest.raises(MissingRenderVarError) as exc_info:
            writer.write_frame(0, sample_render_vars, mock_backend)
        assert exc_info.value.missing_render_var == "NormalSD"

    def test_shaded_seg_shading_modulates_brightness(self):
        """Surfaces facing the light are brighter than surfaces facing away,
        while both keep the same per-instance palette color (hue)."""
        cfg = CosmosTransferWriterConfig(
            shaded_seg_light_dir=(0.0, 0.0, 1.0),
        )
        writer = CosmosTransferWriter(cfg)
        seg = np.ones((48, 64), dtype=np.int64)  # single instance everywhere
        normals = np.zeros((48, 64, 3), dtype=np.float32)
        normals[:, :32, 2] = 1.0  # left: facing the light
        normals[:, 32:, 2] = -1.0  # right: facing away
        writer._seg_frames = [seg]
        writer._normal_frames = [normals]
        shaded = writer._compute_shaded_seg_frames()[0]
        assert shaded.shape == (48, 64, 3)
        assert shaded.dtype == np.uint8
        lit = shaded[:, :32].astype(np.float64).mean()
        unlit = shaded[:, 32:].astype(np.float64).mean()
        assert lit > unlit
        # Facing-away region is the ambient floor (~0.35) of the lit region.
        assert unlit == pytest.approx(lit * 0.35, rel=0.05)

    def test_shaded_seg_frames_cached(self):
        """The shaded frames are computed once and reused (canny + video)."""
        cfg = CosmosTransferWriterConfig()
        writer = CosmosTransferWriter(cfg)
        seg = np.ones((48, 64), dtype=np.int64)
        normals = np.zeros((48, 64, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        writer._seg_frames = [seg]
        writer._normal_frames = [normals]
        first = writer._compute_shaded_seg_frames()
        second = writer._compute_shaded_seg_frames()
        assert first is second

    def test_depth_colorization_grayscale(self):
        cfg = CosmosTransferWriterConfig()
        writer = CosmosTransferWriter(cfg)
        depth = np.linspace(0, 5, 48 * 64).reshape(48, 64).astype(np.float32)
        writer._depth_frames = [depth]
        colored = writer._colorize_depth_frames()
        assert colored[0].shape == (48, 64, 3)
        # Grayscale: all channels should be equal
        np.testing.assert_array_equal(colored[0][..., 0], colored[0][..., 1])

    def test_depth_colorization_squeezes_channel_axis(self):
        """A depth frame with a trailing channel axis (H, W, 1) is squeezed."""
        cfg = CosmosTransferWriterConfig()
        writer = CosmosTransferWriter(cfg)
        depth = np.linspace(0, 5, 48 * 64).reshape(48, 64, 1).astype(np.float32)
        writer._depth_frames = [depth]
        colored = writer._colorize_depth_frames()
        assert colored[0].shape == (48, 64, 3)

    def test_depth_constant_value(self):
        """Constant depth should not crash (div-by-zero guard)."""
        cfg = CosmosTransferWriterConfig()
        writer = CosmosTransferWriter(cfg)
        depth = np.full((48, 64), 5.0, dtype=np.float32)
        writer._depth_frames = [depth]
        colored = writer._colorize_depth_frames()
        assert colored[0].shape == (48, 64, 3)
        # Constant depth → all pixels should be the same color
        # (zero normalized range collapses to a single luma level)
        first_pixel = colored[0][0, 0]
        assert np.all(colored[0] == first_pixel)

    def test_depth_normalization_anchored_on_first_frame(self):
        """Depth bounds come from the first frame, not the whole clip.

        A transient near object appearing only in a later frame must not change
        how the first frame is encoded (the mapping is causal / clip-content
        independent), otherwise a static object's brightness would depend on
        unrelated future frames.
        """
        cfg = CosmosTransferWriterConfig()

        def frame(near=None):
            z = np.full((48, 64), 5.0, dtype=np.float32)
            z[:, 32:] = 20.0  # static background (right half)
            if near is not None:
                z[40:, 50:] = near  # transient near object (background corner)
            return z

        wa = CosmosTransferWriter(cfg)
        wa._depth_frames = [frame(), frame()]
        out_a = wa._colorize_depth_frames()

        wb = CosmosTransferWriter(cfg)
        wb._depth_frames = [frame(), frame(0.5)]  # later frame gets a 0.5 m object
        out_b = wb._colorize_depth_frames()

        # Shared first frame encodes identically despite differing later frames.
        np.testing.assert_array_equal(out_a[0], out_b[0])

    def test_depth_static_region_constant_across_frames(self):
        """A static region keeps constant brightness as other geometry moves."""
        cfg = CosmosTransferWriterConfig()

        def frame(near):
            z = np.full((48, 64), 5.0, dtype=np.float32)  # static object (left)
            z[:, 32:] = near  # right half moves over time
            return z

        writer = CosmosTransferWriter(cfg)
        writer._depth_frames = [frame(20.0), frame(12.0), frame(8.0)]
        out = writer._colorize_depth_frames()

        static_region = [o[:, :32] for o in out]
        np.testing.assert_array_equal(static_region[0], static_region[1])
        np.testing.assert_array_equal(static_region[1], static_region[2])

    def test_seg_colorization(self):
        cfg = CosmosTransferWriterConfig()
        writer = CosmosTransferWriter(cfg)
        seg = np.zeros((48, 64), dtype=np.int32)
        seg[10:20, 10:30] = 1
        seg[30:40, 40:60] = 2
        writer._seg_frames = [seg]
        colored = writer._colorize_seg_frames()
        assert len(colored) == 1
        assert colored[0].shape == (48, 64, 3)
        assert colored[0].dtype == np.uint8
        # Background should be black
        assert colored[0][0, 0].sum() == 0
        # Object region should have color
        assert colored[0][15, 20].sum() > 0


# ---------------------------------------------------------------------------
# Required render vars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_var",
    ["LdrColor", "DistanceToCameraSD", "SemanticInstanceSegmentation"],
)
def test_write_frame_missing_required_var_raises(
    mock_backend, sample_render_vars, small_config, missing_var
):
    writer = CosmosTransferWriter(small_config)
    incomplete = {k: v for k, v in sample_render_vars.items() if k != missing_var}

    with pytest.raises(MissingRenderVarError) as exc_info:
        writer.write_frame(0, incomplete, mock_backend)

    assert exc_info.value.missing_render_var == missing_var


# ---------------------------------------------------------------------------
# End-to-end write_frame + finalize
# ---------------------------------------------------------------------------


def _mock_get_video_encoder(path, fps, w, h, **kwargs):
    """Return a mock video encoder for unit tests (NVENC needs >64px)."""
    enc = MagicMock()
    enc.close.return_value = b"\x00\x00\x00\x1c" + b"ftypisom" + b"\x00" * 100
    return enc


@patch(
    "ovannotators._internal.codecs.get_video_encoder",
    _mock_get_video_encoder,
)
class TestEndToEnd:
    def test_single_frame(self, mock_backend, sample_render_vars, small_config):
        """Single frame should produce RGB video, conditioning, manifest."""
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        written = mock_backend.written
        assert "videos/000000_rgb.mp4" in written
        assert "conditioning/000000_canny.mp4" in written
        assert "conditioning/000000_depth.mp4" in written
        assert "conditioning/000000_segmentation.mp4" in written
        assert "thumbnails/000000.jpg" in written
        assert "manifest.json" in written

    def test_manifest_content(self, mock_backend, sample_render_vars, small_config):
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.write_frame(1, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        manifest = json.loads(mock_backend.written["manifest.json"])
        assert manifest["version"] == "cosmos-transfer-2.5"
        assert len(manifest["clips"]) == 1
        clip = manifest["clips"][0]
        assert clip["clip_id"] == 0
        assert clip["num_frames"] == 2
        assert clip["fps"] == 10
        assert clip["resolution"] == [64, 48]
        assert "prompt" not in clip
        assert "canny" in clip["conditioning_types"]
        assert "depth" in clip["conditioning_types"]
        assert "segmentation" in clip["conditioning_types"]

    def test_multi_frame(self, mock_backend, sample_render_vars, small_config):
        """Multiple frames should all be encoded."""
        writer = CosmosTransferWriter(small_config)
        for i in range(5):
            writer.write_frame(i, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        manifest = json.loads(mock_backend.written["manifest.json"])
        assert manifest["clips"][0]["num_frames"] == 5

    def test_finalize_idempotent(self, mock_backend, sample_render_vars, small_config):
        """Calling finalize twice should not duplicate output."""
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)
        call_count = mock_backend.write.call_count
        writer.finalize(mock_backend)
        assert mock_backend.write.call_count == call_count

    def test_no_frames(self, mock_backend, small_config):
        """Empty writer should not write anything."""
        writer = CosmosTransferWriter(small_config)
        writer.finalize(mock_backend)
        assert mock_backend.write.call_count == 0

    def test_canny_only(self, mock_backend, sample_render_vars):
        """With only canny conditioning, should skip depth and seg videos."""
        cfg = CosmosTransferWriterConfig(
            conditioning_types=["canny"],
        )
        writer = CosmosTransferWriter(cfg)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        written = mock_backend.written
        assert "conditioning/000000_canny.mp4" in written
        assert "conditioning/000000_depth.mp4" not in written
        assert "conditioning/000000_segmentation.mp4" not in written

    def test_custom_clip_index(self, mock_backend, sample_render_vars):
        """Custom clip index should be reflected in paths and manifest."""
        cfg = CosmosTransferWriterConfig(
            clip_index=42,
        )
        writer = CosmosTransferWriter(cfg)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        assert "videos/000042_rgb.mp4" in mock_backend.written
        manifest = json.loads(mock_backend.written["manifest.json"])
        assert manifest["clips"][0]["clip_id"] == 42

    def test_cut_produces_sequential_clips(
        self, mock_backend, sample_render_vars, small_config
    ):
        """cut() closes the current clip and starts a new one with the next id."""
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.write_frame(1, sample_render_vars, mock_backend)
        first_id = writer.cut(mock_backend)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        assert first_id == 0
        written = mock_backend.written
        assert "videos/000000_rgb.mp4" in written
        assert "videos/000001_rgb.mp4" in written
        assert "conditioning/000000_depth.mp4" in written
        assert "conditioning/000001_depth.mp4" in written

        manifest = json.loads(written["manifest.json"])
        assert [c["clip_id"] for c in manifest["clips"]] == [0, 1]
        assert manifest["clips"][0]["num_frames"] == 2
        assert manifest["clips"][1]["num_frames"] == 1

    def test_cut_returns_clip_id(self, mock_backend, sample_render_vars, small_config):
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        assert writer.cut(mock_backend) == 0
        writer.write_frame(0, sample_render_vars, mock_backend)
        assert writer.cut(mock_backend) == 1

    def test_cut_no_frames_is_noop(self, mock_backend, small_config):
        """cut() with nothing buffered writes nothing and returns None."""
        writer = CosmosTransferWriter(small_config)
        assert writer.cut(mock_backend) is None
        assert mock_backend.write.call_count == 0

    def test_cut_then_write_skips_empty_clip(
        self, mock_backend, sample_render_vars, small_config
    ):
        """Back-to-back cuts do not emit an empty clip between them."""
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        assert writer.cut(mock_backend) == 0
        # No frames written since the last cut → no-op, id stays available.
        assert writer.cut(mock_backend) is None
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        manifest = json.loads(mock_backend.written["manifest.json"])
        assert [c["clip_id"] for c in manifest["clips"]] == [0, 1]

    def test_cut_respects_starting_clip_index(self, mock_backend, sample_render_vars):
        """cut() advances from the configured starting clip_index."""
        cfg = CosmosTransferWriterConfig(clip_index=5)
        writer = CosmosTransferWriter(cfg)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.cut(mock_backend)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        assert "videos/000005_rgb.mp4" in mock_backend.written
        assert "videos/000006_rgb.mp4" in mock_backend.written
        manifest = json.loads(mock_backend.written["manifest.json"])
        assert [c["clip_id"] for c in manifest["clips"]] == [5, 6]

    def test_video_bytes_nonempty(self, mock_backend, sample_render_vars, small_config):
        """All video outputs should be non-empty bytes."""
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        for key in ["videos/000000_rgb.mp4", "conditioning/000000_canny.mp4"]:
            assert len(mock_backend.written[key]) > 0
            assert isinstance(mock_backend.written[key], bytes)

    def test_thumbnail_is_jpeg(self, mock_backend, sample_render_vars, small_config):
        """Thumbnail should be valid JPEG bytes."""
        writer = CosmosTransferWriter(small_config)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)

        thumb = mock_backend.written["thumbnails/000000.jpg"]
        # JPEG magic bytes: FF D8 FF
        assert thumb[:2] == b"\xff\xd8"

    def test_depth_only_conditioning(self, mock_backend, sample_render_vars):
        cfg = CosmosTransferWriterConfig(conditioning_types=["depth"])
        writer = CosmosTransferWriter(cfg)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)
        written = mock_backend.written
        assert "conditioning/000000_depth.mp4" in written
        assert "conditioning/000000_canny.mp4" not in written
        assert "conditioning/000000_segmentation.mp4" not in written

    def test_segmentation_only_conditioning(self, mock_backend, sample_render_vars):
        cfg = CosmosTransferWriterConfig(conditioning_types=["segmentation"])
        writer = CosmosTransferWriter(cfg)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)
        written = mock_backend.written
        assert "conditioning/000000_segmentation.mp4" in written
        assert "conditioning/000000_canny.mp4" not in written
        assert "conditioning/000000_depth.mp4" not in written

    def test_shaded_segmentation_conditioning(self, mock_backend, sample_render_vars):
        cfg = CosmosTransferWriterConfig(
            conditioning_types=["shaded_segmentation"],
        )
        writer = CosmosTransferWriter(cfg)
        render_vars = {
            **sample_render_vars,
            "NormalSD": (np.random.rand(48, 64, 3) * 2.0 - 1.0).astype(np.float32),
        }
        writer.write_frame(0, render_vars, mock_backend)
        writer.finalize(mock_backend)
        written = mock_backend.written
        assert "conditioning/000000_shaded_segmentation.mp4" in written
        assert "conditioning/000000_canny.mp4" not in written
        assert "conditioning/000000_depth.mp4" not in written
        manifest = json.loads(written["manifest.json"])
        assert "shaded_segmentation" in manifest["clips"][0]["conditioning_types"]

    def test_hevc_video_codec(self, mock_backend, sample_render_vars):
        cfg = CosmosTransferWriterConfig()
        writer = CosmosTransferWriter(cfg)
        writer.write_frame(0, sample_render_vars, mock_backend)
        writer.finalize(mock_backend)
        assert len(mock_backend.written["videos/000000_rgb.mp4"]) > 0


class TestCannyThresholds:
    def test_thresholds_change_edge_output(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        # Soft gradient so threshold choice changes which edges survive.
        frame[:, :, :] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None]
        frame[:, 32:, :] = np.clip(frame[:, 32:, :].astype(int) + 40, 0, 255)

        def _canny(sigma, low, high):
            cfg = CosmosTransferWriterConfig(
                conditioning_types=["canny"],
                canny_sigma=sigma,
                canny_low_threshold=low,
                canny_high_threshold=high,
            )
            writer = CosmosTransferWriter(cfg)
            writer._rgb_frames = [frame]
            return writer._compute_canny_frames()[0]

        default_edges = _canny(1.0, 0.1, 0.2)
        strict_edges = _canny(1.0, 0.4, 0.8)
        assert default_edges.shape == strict_edges.shape
        # Stricter hysteresis thresholds should not produce more edge pixels.
        assert int((strict_edges > 0).sum()) <= int((default_edges > 0).sum())
        assert not np.array_equal(default_edges, strict_edges)


class TestCosmosTransferGeometryGuards:
    """Resolution/resize/shade geometry guards exercised in isolation."""

    def test_output_resolution_empty_writer(self):
        """A writer with no accumulated frames reports a zero resolution."""
        assert CosmosTransferWriter(
            CosmosTransferWriterConfig()
        )._output_resolution() == (0, 0)

    def test_resize_rgb_nearest_upsamples(self):
        out = CosmosTransferWriter._resize_rgb_nearest(
            np.zeros((4, 4, 3), np.uint8), 8, 8
        )
        assert out.shape == (8, 8, 3)

    def test_resize_normals_resamples_per_channel(self):
        out = CosmosTransferWriter._resize_normals(
            np.zeros((4, 4, 3), np.float32), 8, 8
        )
        assert out.shape == (8, 8, 3)

    def test_resize_normals_noop_when_already_target_size(self):
        """Normals already at the target size are returned without resampling."""
        out = CosmosTransferWriter._resize_normals(np.ones((8, 8, 3), np.float32), 8, 8)
        assert out.shape == (8, 8, 3)
        assert np.array_equal(out, np.ones((8, 8, 3), np.float32))

    def test_resize_depth_squeezes_and_resamples(self):
        writer = CosmosTransferWriter(CosmosTransferWriterConfig())
        out = writer._resize_depth(np.ones((4, 4, 2), np.float32), 8, 8)
        assert out.shape == (8, 8)

    def test_shaded_seg_snaps_mismatched_normals(self):
        """Normals not matching the seg grid are resized before shading."""
        writer = CosmosTransferWriter(CosmosTransferWriterConfig())
        writer._resolution = (8, 8)
        writer._seg_frames = [np.zeros((8, 8), dtype=np.int32)]
        writer._normal_frames = [np.zeros((4, 4, 3), dtype=np.float32)]  # mismatched
        frames = writer._compute_shaded_seg_frames()
        assert frames[0].shape == (8, 8, 3)

    def test_segmentation_canny_source_resizes_gray(self):
        """A seg frame with an extra channel and off-grid size is squeezed+resized."""
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(canny_source="segmentation")
        )
        writer._resolution = (8, 8)
        writer._seg_frames = [np.arange(16, dtype=np.int32).reshape(4, 4, 1)]
        grays = writer._canny_source_gray_frames()
        assert grays[0].shape == (8, 8)


@pytest.mark.parametrize("channels", [False, True])
def test_seg_colors_survive_visibility_changes(channels):
    """Disappearance, reappearance and new lower IDs preserve old colors."""
    writer = CosmosTransferWriter()
    frames = [
        np.array([row], dtype=np.uint32)
        for row in (
            [0, 10, 2**32 - 1],
            [0, 0, 2**32 - 1],
            [5, 10, 2**32 - 1],
            [0, 10, 2**32 - 1],
        )
    ]
    writer._seg_frames = [f[..., None] if channels else f for f in frames]
    colored = writer._colorize_seg_frames()
    for frame in colored[1:]:
        np.testing.assert_array_equal(frame[0, 2], colored[0][0, 2])
    np.testing.assert_array_equal(colored[2][0, 1], colored[0][0, 1])
    np.testing.assert_array_equal(colored[3], colored[0])
    assert not colored[1][0, :2].any()
    assert np.unique(colored[2].reshape(-1, 3), axis=0).shape[0] == 3


def test_seg_colors_use_complete_multichannel_identity():
    """128-bit IDs differing outside channel zero must not collapse."""
    writer = CosmosTransferWriter()
    first = np.array(
        [[[0, 0, 0, 0], [0, 0, 0, 7], [42, 1, 0, 0], [42, 2, 0, 0]]], dtype=np.uint32
    )
    writer._seg_frames = [first, first[:, [0, 0, 3, 1]]]
    before, after = writer._colorize_seg_frames()
    assert not before[0, 0].any()
    assert np.unique(before.reshape(-1, 3), axis=0).shape[0] == 4
    np.testing.assert_array_equal(after[0, 2], before[0, 3])
    np.testing.assert_array_equal(after[0, 3], before[0, 1])


@pytest.mark.parametrize("first_channel_collision", [False, True])
def test_seg_colors_match_full_id_sort(first_channel_collision):
    """The fast lookup and collision fallback preserve full-ID palette order."""
    ids = np.array(
        [
            [0, 0, 0, 0],
            [7, 11, 0, 1],
            [19, 3, 0, 2],
            [31, 5, 0, 3],
        ],
        dtype=np.uint32,
    )
    if first_channel_collision:
        ids[2, 0] = ids[1, 0]
    rng = np.random.default_rng(101)
    frame = ids[rng.integers(0, len(ids), size=(64, 64))]

    unique, inverse = np.unique(frame.reshape(-1, 4), axis=0, return_inverse=True)
    foreground = np.any(unique, axis=1)
    palette = DistinctColorPalette().colors(int(foreground.sum()))
    unique_colors = np.zeros((len(unique), 3), dtype=np.uint8)
    unique_colors[foreground] = palette
    expected = unique_colors[inverse].reshape(*frame.shape[:2], 3)

    actual = CosmosTransferWriter()._seg_to_color(frame)
    np.testing.assert_array_equal(actual, expected)


def test_distinct_seg_palette_supports_thousands_of_unique_instances():
    writer = CosmosTransferWriter()
    colored = writer._seg_to_color(np.arange(1001, dtype=np.uint32)[None, :])

    assert not colored[0, 0].any()
    assert np.unique(colored[0, 1:], axis=0).shape[0] == 1000


def test_distinct_seg_palette_sequence_is_deterministic_and_incremental():
    expected_prefix = np.array(
        [
            [99, 0, 255],
            [0, 144, 0],
            [255, 0, 88],
            [99, 33, 44],
            [44, 255, 0],
            [55, 177, 255],
            [232, 166, 0],
            [232, 0, 255],
            [255, 155, 221],
            [0, 110, 155],
        ],
        dtype=np.uint8,
    )
    one_shot = DistinctColorPalette().colors(256)
    incremental_palette = DistinctColorPalette()
    incremental_palette.colors(10)
    incremental = incremental_palette.colors(256)

    np.testing.assert_array_equal(one_shot[:10], expected_prefix)
    np.testing.assert_array_equal(incremental, one_shot)


_GOLDEN_PALETTE_PREFIX = {
    "none": [
        [99, 0, 255], [0, 144, 0], [255, 0, 88], [99, 33, 44],
        [44, 255, 0], [55, 177, 255], [232, 166, 0], [232, 0, 255],
        [255, 155, 221], [0, 110, 155], [166, 99, 22], [155, 0, 155],
        [11, 243, 221], [0, 77, 0], [0, 33, 166], [166, 121, 177],
    ],
    "protanopia": [
        [110, 0, 221], [88, 155, 0], [133, 22, 0], [144, 166, 255],
        [221, 221, 33], [255, 0, 110], [99, 0, 110], [133, 210, 188],
        [199, 133, 155], [77, 121, 255], [88, 88, 11], [188, 0, 144],
        [232, 177, 88], [110, 255, 166], [166, 33, 77], [210, 88, 177],
    ],
    "deuteranopia": [
        [66, 0, 255], [232, 22, 0], [99, 22, 77], [188, 155, 255],
        [232, 199, 0], [77, 121, 155], [133, 188, 144], [155, 22, 0],
        [0, 33, 166], [11, 144, 255], [22, 243, 243], [110, 255, 166],
        [232, 44, 133], [255, 110, 0], [144, 33, 188], [155, 0, 99],
    ],
    "tritanopia": [
        [166, 33, 121], [33, 177, 88], [11, 77, 0], [255, 121, 255],
        [66, 255, 0], [133, 33, 255], [232, 0, 0], [199, 210, 121],
        [166, 133, 44], [99, 33, 44], [221, 22, 232], [0, 133, 0],
        [110, 88, 0], [177, 144, 255], [133, 199, 210], [88, 0, 155],
    ],
}  # fmt: skip


@pytest.mark.parametrize("accessibility", sorted(_GOLDEN_PALETTE_PREFIX))
def test_distinct_seg_palette_golden_colors(accessibility):
    """Pin the palette so float32 drift across platforms or NumPy builds fails CI.

    Persisted palette files are validated against the regenerated palette, so a
    different color sequence would make existing state files unloadable.
    """
    colors = DistinctColorPalette(accessibility).colors(16)
    np.testing.assert_array_equal(colors, _GOLDEN_PALETTE_PREFIX[accessibility])


@pytest.mark.parametrize("accessibility", sorted(_GOLDEN_PALETTE_PREFIX))
def test_palette_state_file_with_golden_colors_loads(tmp_path, accessibility):
    """A state file holding the pinned colors loads and keeps its assignments."""
    golden = _GOLDEN_PALETTE_PREFIX[accessibility][:3]
    path = tmp_path / "palette.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "algorithm": "distinct-v1",
                "accessibility": accessibility,
                "colors": golden,
                "assignments": [
                    {"key_type": "instance_id", "key": [10 + i], "slot": i}
                    for i in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    colorizer = SegmentationColorizer(accessibility=accessibility, path=path)
    seg = np.array([[10, 11, 12]], dtype=np.uint32)
    np.testing.assert_array_equal(
        colorizer.colorize(seg, "instance_id")[0], np.array(golden, dtype=np.uint8)
    )


def test_distinct_seg_palette_improves_early_perceptual_separation():
    distinct = DistinctColorPalette().colors(10).astype(np.float32) / 255.0

    def minimum_distance(colors):
        lab = _srgb_to_oklab(colors)
        distances = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return distances.min()

    assert minimum_distance(distinct) > 0.2


@pytest.mark.parametrize("accessibility", ["protanopia", "deuteranopia", "tritanopia"])
def test_distinct_seg_palette_supports_accessibility_modes(accessibility):
    standard = DistinctColorPalette().colors(16).astype(np.float32) / 255.0
    accessible = (
        DistinctColorPalette(accessibility).colors(16).astype(np.float32) / 255.0
    )

    def minimum_simulated_distance(colors):
        lab = _simulate_accessibility_oklab(colors, accessibility)
        distances = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return distances.min()

    assert not np.array_equal(accessible, standard)
    assert minimum_simulated_distance(accessible) > minimum_simulated_distance(standard)


def test_semantic_label_color_key_shares_colors_by_label(mock_backend):
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(
            conditioning_types=["segmentation"],
            segmentation_color_key="semantic_label",
        )
    )
    render_vars = {
        "LdrColor": np.zeros((1, 3, 4), dtype=np.uint8),
        "DistanceToCameraSD": np.ones((1, 3), dtype=np.float32),
        "SemanticSegmentation": np.array([[1, 2, 3]], dtype=np.uint32),
        "SemanticIdMap": {1: "class:car", 2: "class:car", 3: "class:person"},
    }

    writer.write_frame(0, render_vars, mock_backend)
    colored = writer._colorize_seg_frames()[0]

    np.testing.assert_array_equal(colored[0, 0], colored[0, 1])
    assert not np.array_equal(colored[0, 0], colored[0, 2])


def test_semantic_label_color_key_requires_labels_for_new_ids(mock_backend):
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(
            conditioning_types=["segmentation"],
            segmentation_color_key="semantic_label",
        )
    )
    base = {
        "LdrColor": np.zeros((1, 1, 4), dtype=np.uint8),
        "DistanceToCameraSD": np.ones((1, 1), dtype=np.float32),
        "SemanticSegmentation": np.array([[1]], dtype=np.uint32),
        "SemanticIdMap": {1: "class:car"},
    }
    writer.write_frame(0, base, mock_backend)
    writer.write_frame(
        1,
        {key: value for key, value in base.items() if key != "SemanticIdMap"},
        mock_backend,
    )

    conflicting = dict(base)
    conflicting["SemanticIdMap"] = {1: "class:truck"}
    with pytest.raises(ValueError, match="changed label"):
        writer.write_frame(2, conflicting, mock_backend)

    missing = dict(base)
    missing.pop("SemanticIdMap")
    missing["SemanticSegmentation"] = np.array([[2]], dtype=np.uint32)
    with pytest.raises(MissingRenderVarError) as exc_info:
        writer.write_frame(3, missing, mock_backend)
    assert exc_info.value.missing_render_var == "SemanticIdMap"


def test_stable_id_palette_state_persists_assignments(
    tmp_path, mock_backend, monkeypatch
):
    state_path = tmp_path / "segmentation-palette.json"
    config = CosmosTransferWriterConfig(
        conditioning_types=["segmentation"],
        segmentation_color_key="stable_id",
        segmentation_palette_path=state_path,
    )
    writer = CosmosTransferWriter(config)
    encoded = []

    def encode(frames, width, height):
        encoded.append([frame.copy() for frame in frames])
        return b"video"

    monkeypatch.setattr(writer, "_encode_video", encode)
    stable_ids = np.array(
        [[[0, 0, 0, 0], [11, 0, 0, 0], [22, 0, 0, 0]]], dtype=np.uint32
    )
    writer.write_frame(
        0,
        {
            "LdrColor": np.zeros((1, 3, 4), dtype=np.uint8),
            "DistanceToCameraSD": np.ones((1, 3), dtype=np.float32),
            "StableIdSegmentation": stable_ids,
        },
        mock_backend,
    )
    writer.finalize(mock_backend)

    state = json.loads(state_path.read_text())
    assert state["assignments"] == [
        {"key": [11, 0, 0, 0], "key_type": "stable_id", "slot": 0},
        {"key": [22, 0, 0, 0], "key_type": "stable_id", "slot": 1},
    ]
    assert not list(tmp_path.glob("*.tmp"))

    restored = CosmosTransferWriter(config)
    recolored = restored._seg_to_color(stable_ids[:, [0, 2, 1]])
    original = encoded[1][0]
    np.testing.assert_array_equal(recolored[0, 1], original[0, 2])
    np.testing.assert_array_equal(recolored[0, 2], original[0, 1])


def test_palette_state_rejects_incompatible_key_mode(tmp_path):
    state_path = tmp_path / "segmentation-palette.json"
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(segmentation_palette_path=state_path)
    )
    writer._seg_to_color(np.array([[1]], dtype=np.uint32))
    writer._seg_colorizer.save()

    with pytest.raises(ValueError, match="key type"):
        CosmosTransferWriter(
            CosmosTransferWriterConfig(
                segmentation_color_key="semantic_id",
                segmentation_palette_path=state_path,
            )
        )


def test_palette_state_rejects_modified_colors(tmp_path):
    state_path = tmp_path / "segmentation-palette.json"
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(segmentation_palette_path=state_path)
    )
    writer._seg_to_color(np.array([[1]], dtype=np.uint32))
    writer._seg_colorizer.save()
    state = json.loads(state_path.read_text())
    state["colors"][0] = [1, 2, 3]
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="colors do not match"):
        CosmosTransferWriter(
            CosmosTransferWriterConfig(segmentation_palette_path=state_path)
        )


@pytest.mark.parametrize("shaded_first", [False, True])
def test_seg_and_shaded_share_stable_colors(shaded_first):
    """Both paths retain the same colors regardless of invocation order."""
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(shaded_seg_light_dir=(0.0, 0.0, 1.0))
    )
    writer._seg_frames = [
        np.array([[0, 10, 20]], dtype=np.uint32),
        np.array([[0, 0, 20]], dtype=np.uint32),
    ]
    normals = np.zeros((1, 3, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    writer._normal_frames = [normals, normals]
    if shaded_first:
        shaded = writer._compute_shaded_seg_frames()
        plain = writer._colorize_seg_frames()
    else:
        plain = writer._colorize_seg_frames()
        shaded = writer._compute_shaded_seg_frames()
    for unshaded, lit in zip(plain, shaded, strict=True):
        np.testing.assert_array_equal(unshaded, lit)
    np.testing.assert_array_equal(shaded[0][0, 2], shaded[1][0, 2])


def test_seg_palette_remains_stable_between_encoded_clips(mock_backend, monkeypatch):
    """cut() preserves dataset-session colors across clip boundaries."""
    writer = CosmosTransferWriter(
        CosmosTransferWriterConfig(conditioning_types=["segmentation"])
    )
    encoded = []

    def encode(frames, width, height):
        encoded.append([f.copy() for f in frames])
        return b"video"

    monkeypatch.setattr(writer, "_encode_video", encode)
    for index, ids in enumerate(([0, 10, 20], [0, 0, 20])):
        writer.write_frame(
            index,
            {
                "LdrColor": np.zeros((1, 3, 4), dtype=np.uint8),
                "DistanceToCameraSD": np.ones((1, 3), dtype=np.float32),
                "SemanticInstanceSegmentation": np.array([ids], dtype=np.uint32),
            },
            mock_backend,
        )
    assert writer.cut(mock_backend) == 0
    np.testing.assert_array_equal(encoded[1][0][0, 2], encoded[1][1][0, 2])
    writer.write_frame(
        0,
        {
            "LdrColor": np.zeros((1, 3, 4), dtype=np.uint8),
            "DistanceToCameraSD": np.ones((1, 3), dtype=np.float32),
            "SemanticInstanceSegmentation": np.array([[0, 0, 20]], dtype=np.uint32),
        },
        mock_backend,
    )
    writer.finalize(mock_backend)
    np.testing.assert_array_equal(encoded[3][0][0, 2], encoded[1][0][0, 2])
    assert len(json.loads(mock_backend.written["manifest.json"])["clips"]) == 2
