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

"""Tests for Warp-based Canny edge detection (runs on CPU or CUDA)."""

from __future__ import annotations

import numpy as np
import pytest

from ovannotators.writers.canny import canny


class TestCanny:
    """Test Warp-based Canny edge detection."""

    def test_basic_output_shape(self):
        gray = np.random.default_rng(42).uniform(0, 1, size=(100, 100))
        edges = canny(gray)
        assert edges.shape == (100, 100)
        assert edges.dtype == bool

    def test_all_black_no_edges(self):
        gray = np.zeros((100, 100), dtype=np.float64)
        edges = canny(gray)
        assert edges.sum() == 0

    def test_all_white_no_edges(self):
        gray = np.ones((100, 100), dtype=np.float64)
        edges = canny(gray, low_threshold=0.3, high_threshold=0.5)
        # Uniform image should have very few/no real edges
        # (border artifacts from convolution may produce a handful)
        assert edges.sum() < 100  # effectively no meaningful edges

    def test_sharp_edge_detected(self):
        # Create a clear vertical edge
        gray = np.zeros((100, 100), dtype=np.float64)
        gray[:, 50:] = 1.0
        edges = canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        # Should detect edges around column 50
        assert edges.sum() > 0
        # Most edges should be near column 50
        edge_cols = np.where(edges)[1]
        assert np.abs(edge_cols.mean() - 50) < 5

    def test_720p_resolution(self):
        gray = np.random.default_rng(42).uniform(0, 1, size=(720, 1280))
        edges = canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        assert edges.shape == (720, 1280)
        # Random noise should produce some edges
        assert edges.sum() > 0

    def test_sigma_affects_output(self):
        gray = np.random.default_rng(42).uniform(0, 1, size=(200, 200))
        edges_low = canny(gray, sigma=0.5)
        edges_high = canny(gray, sigma=3.0)
        # Higher sigma = more blur = fewer edges
        assert edges_low.sum() > edges_high.sum()

    def test_threshold_affects_output(self):
        gray = np.random.default_rng(42).uniform(0, 1, size=(200, 200))
        edges_low = canny(gray, low_threshold=0.05, high_threshold=0.1)
        edges_high = canny(gray, low_threshold=0.3, high_threshold=0.6)
        # Higher thresholds = fewer edges
        assert edges_low.sum() > edges_high.sum()

    def test_warp_array_input_stays_on_device(self):
        """A Warp float32 input is processed on its device and returned there."""
        import warp as wp

        gray_np = np.zeros((100, 100), dtype=np.float32)
        gray_np[:, 50:] = 1.0
        gray_wp = wp.array(gray_np, dtype=wp.float32, device=wp.get_device())

        edges = canny(gray_wp)
        # Warp in → Warp out, on the same device, no host conversion.
        assert isinstance(edges, wp.array)
        assert edges.dtype == wp.uint8
        assert str(edges.device) == str(gray_wp.device)
        assert tuple(edges.shape) == (100, 100)

    def test_warp_and_numpy_inputs_match(self):
        """Warp and NumPy inputs produce identical edge masks."""
        import warp as wp

        gray = np.random.default_rng(7).uniform(0, 1, size=(128, 96)).astype(np.float32)
        edges_np = canny(gray)
        edges_wp = canny(wp.array(gray, dtype=wp.float32, device=wp.get_device()))
        assert np.array_equal(edges_wp.numpy().astype(bool), edges_np)

    def test_warp_non_float32_raises(self):
        """A non-float32 Warp input is rejected (avoids silent bit-reinterpret)."""
        import warp as wp

        bad = wp.array(
            np.zeros((16, 16), dtype=np.uint32), dtype=wp.uint32, device=wp.get_device()
        )
        with pytest.raises(TypeError):
            canny(bad)

    def test_non_2d_input_raises(self):
        """A non-2D input is rejected for both NumPy and Warp paths."""
        with pytest.raises(ValueError):
            canny(np.zeros((4, 4, 3), dtype=np.float32))

    def test_warp_input_non_2d_raises(self):
        """A non-2D Warp array is rejected before any kernel launch."""
        import warp as wp

        arr = wp.zeros((2, 2, 2), dtype=wp.float32)
        with pytest.raises(ValueError, match="2D array"):
            canny(arr)

    def test_cosmos_writer_uses_canny(self):
        """Integration: the Cosmos writer runs the canny path end-to-end."""
        from unittest.mock import MagicMock, patch

        from ovannotators.writers.cosmos_transfer import (
            CosmosTransferWriter,
            CosmosTransferWriterConfig,
        )

        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        depth = rng.uniform(0.1, 50, size=(720, 1280)).astype(np.float32)
        seg = rng.integers(0, 30, size=(720, 1280), dtype=np.uint32)

        cfg = CosmosTransferWriterConfig(
            conditioning_types=["canny"],
        )
        writer = CosmosTransferWriter(cfg)
        backend = MagicMock()

        writer.write_frame(
            0,
            {
                "LdrColor": rgb,
                "DistanceToCameraSD": depth,
                "SemanticInstanceSegmentation": seg,
            },
            backend,
        )

        # Mock the video encoder to avoid PyNvVideoCodec constructor issues
        mock_enc = MagicMock()
        mock_enc.close.return_value = b"fake_video_bytes"
        with patch(
            "ovannotators._internal.codecs.get_video_encoder", return_value=mock_enc
        ):
            writer.finalize(backend)

        # Should have written canny video
        write_calls = [str(c) for c in backend.write.call_args_list]
        assert any("canny" in c for c in write_calls)
