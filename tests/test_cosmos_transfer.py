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
from ovannotators.writers.cosmos_transfer import (
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
    _SEG_PALETTE,
)
from ovannotators.writers._registry import WriterRegistry


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

    def test_custom(self):
        cfg = CosmosTransferWriterConfig(
            fps=30,
            clip_index=5,
            conditioning_types=["canny"],
        )
        assert cfg.fps == 30
        assert cfg.clip_index == 5
        assert cfg.conditioning_types == ["canny"]


# ---------------------------------------------------------------------------
# LUT generation
# ---------------------------------------------------------------------------


class TestLUTs:
    def test_seg_palette_shape(self):
        assert _SEG_PALETTE.shape == (256, 3)
        assert _SEG_PALETTE.dtype == np.uint8

    def test_seg_palette_background_black(self):
        np.testing.assert_array_equal(_SEG_PALETTE[0], [0, 0, 0])

    def test_seg_palette_nonzero_entries(self):
        # All non-zero indices should have non-zero colors
        for i in range(1, 256):
            assert _SEG_PALETTE[i].sum() > 0


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
