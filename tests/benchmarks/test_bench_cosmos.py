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

"""Benchmarks for Cosmos Transfer writer internals.

Focuses on the expensive operations: canny edge detection, depth
colorization, segmentation colorization, and video encoding.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import PyNvVideoCodec  # noqa: F401

    _HAS_PYNVVIDEOCODEC = True
except ImportError:
    _HAS_PYNVVIDEOCODEC = False

from ovannotators.writers._segmentation_palette import DistinctColorPalette
from ovannotators.writers.canny import canny as _run_canny
from ovannotators.writers.cosmos_transfer import (
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
)

# ---------------------------------------------------------------------------
# Canny edge detection
# ---------------------------------------------------------------------------


class TestCannyPerformance:
    """Benchmark Warp Canny at various resolutions."""

    @pytest.fixture(params=["480p", "720p", "1080p"], ids=lambda r: r)
    def gray_frame(self, request) -> np.ndarray:
        resolutions = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
        w, h = resolutions[request.param]
        rng = np.random.default_rng(42)
        return rng.uniform(0, 1, size=(h, w))

    def test_canny_default_sigma(self, benchmark, gray_frame):
        kwargs = {"sigma": 1.0, "low_threshold": 0.1, "high_threshold": 0.2}
        _run_canny(gray_frame, **kwargs)
        benchmark.pedantic(
            _run_canny,
            args=(gray_frame,),
            kwargs=kwargs,
            rounds=10,
            iterations=1,
        )

    def test_canny_high_sigma(self, benchmark, gray_frame):
        kwargs = {"sigma": 2.0, "low_threshold": 0.1, "high_threshold": 0.2}
        _run_canny(gray_frame, **kwargs)
        benchmark.pedantic(
            _run_canny,
            args=(gray_frame,),
            kwargs=kwargs,
            rounds=10,
            iterations=1,
        )


# ---------------------------------------------------------------------------
# Depth colorization
# ---------------------------------------------------------------------------


class TestDepthColorization:
    """Benchmark grayscale depth encoding at various resolutions."""

    @pytest.fixture(params=["720p", "1080p"], ids=lambda r: r)
    def depth(self, request) -> np.ndarray:
        resolutions = {"720p": (1280, 720), "1080p": (1920, 1080)}
        w, h = resolutions[request.param]
        rng = np.random.default_rng(42)
        return rng.uniform(0.1, 100.0, size=(h, w)).astype(np.float32)

    def test_colorize_depth_frames(self, benchmark, depth):
        """Benchmark depth → grayscale (luma) RGB colorization."""
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(conditioning_types=["depth"])
        )
        writer._depth_frames = [depth]
        result = writer._colorize_depth_frames()
        assert result[0].shape == (*depth.shape, 3)

        benchmark.pedantic(
            writer._colorize_depth_frames,
            rounds=10,
            iterations=1,
        )


# ---------------------------------------------------------------------------
# Segmentation colorization
# ---------------------------------------------------------------------------


class TestSegColorization:
    """Benchmark distinct palette generation for segmentation."""

    @pytest.fixture(params=["720p", "1080p"], ids=lambda r: r)
    def seg(self, request) -> np.ndarray:
        resolutions = {"720p": (1280, 720), "1080p": (1920, 1080)}
        w, h = resolutions[request.param]
        rng = np.random.default_rng(42)
        return rng.integers(0, 50, size=(h, w), dtype=np.uint32)

    def test_assign_persistent_palette(self, benchmark, seg):
        """Benchmark seg ID → RGB via palette."""
        writer = CosmosTransferWriter(
            CosmosTransferWriterConfig(conditioning_types=["segmentation"])
        )
        result = writer._seg_to_color(seg)
        assert result.shape == (*seg.shape, 3)

        benchmark.pedantic(
            writer._seg_to_color,
            args=(seg,),
            rounds=10,
            iterations=1,
        )

    @pytest.mark.parametrize("count", [1000, 5000])
    def test_generate_distinct_colors(self, benchmark, count):
        """Benchmark a fresh session palette at representative scales."""
        benchmark(lambda: DistinctColorPalette().colors(count))


# ---------------------------------------------------------------------------
# Frame resize
# ---------------------------------------------------------------------------


class TestResize:
    """Benchmark Pillow-based frame resizing."""

    def test_resize_720p_to_1080p(self, benchmark):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        writer = CosmosTransferWriter()
        result = writer._resize_frame(frame, 1920, 1080)
        assert result.shape == (1080, 1920, 3)

        benchmark.pedantic(
            writer._resize_frame,
            args=(frame, 1920, 1080),
            rounds=10,
            iterations=1,
        )

    def test_resize_1080p_to_720p(self, benchmark):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        writer = CosmosTransferWriter()
        result = writer._resize_frame(frame, 1280, 720)
        assert result.shape == (720, 1280, 3)

        benchmark.pedantic(
            writer._resize_frame,
            args=(frame, 1280, 720),
            rounds=10,
            iterations=1,
        )


# ---------------------------------------------------------------------------
# Video encoding (PyNvVideoCodec)
# ---------------------------------------------------------------------------


class TestVideoEncoding:
    """Benchmark video encoding via PyNvVideoCodec."""

    @pytest.fixture
    def frames_720p(self) -> list[np.ndarray]:
        rng = np.random.default_rng(42)
        return [
            rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(30)
        ]

    @pytest.mark.skipif(
        not _HAS_PYNVVIDEOCODEC,
        reason="PyNvVideoCodec not installed (requires CUDA)",
    )
    def test_encode_30_frames_720p(self, benchmark, frames_720p):
        """Encode 30 frames (1 second @ 30fps) at 720p."""
        from ovannotators._internal.codecs import get_video_encoder

        def encode(frames):
            enc = get_video_encoder(
                path="",
                fps=30,
                w=1280,
                h=720,
            )
            for f in frames:
                enc.write_frame(f)
            return enc.close()

        benchmark.pedantic(encode, args=(frames_720p,), rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# End-to-end Cosmos writer (single frame)
# ---------------------------------------------------------------------------


class TestCosmosEndToEnd:
    """Benchmark full Cosmos writer frame accumulation + finalize."""

    @pytest.mark.skipif(
        not _HAS_PYNVVIDEOCODEC,
        reason="PyNvVideoCodec not installed (requires CUDA)",
    )
    def test_single_frame_720p(self, benchmark):
        """Single frame through full Cosmos pipeline (accumulate + finalize)."""
        from unittest.mock import MagicMock

        rng = np.random.default_rng(42)
        # Use RGB (3-channel) for LdrColor since JPEG encoding in finalize rejects RGBA
        rgb = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        depth = rng.uniform(0.1, 50.0, size=(720, 1280)).astype(np.float32)
        seg = rng.integers(0, 30, size=(720, 1280), dtype=np.uint32)

        render_vars = {
            "LdrColor": rgb,
            "DistanceToCameraSD": depth,
            "SemanticInstanceSegmentation": seg,
        }
        backend = MagicMock()

        def run(rv):
            cfg = CosmosTransferWriterConfig(
                conditioning_types=["canny", "depth", "segmentation"],
            )
            writer = CosmosTransferWriter(cfg)
            writer.write_frame(0, rv, backend)
            writer.finalize(backend)

        benchmark.pedantic(run, args=(render_vars,), rounds=3, iterations=1)
