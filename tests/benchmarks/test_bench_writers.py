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

"""Benchmarks for writer write_frame() performance.

Measures per-frame throughput for each writer with realistic data.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ovannotators.writers._registry import WriterRegistry


# ---------------------------------------------------------------------------
# Writer benchmarks
# ---------------------------------------------------------------------------


class TestWriterPerFrame:
    """Benchmark write_frame() for each writer."""

    @pytest.fixture
    def mock_backend(self):
        """Mock backend that accepts writes without I/O."""
        backend = MagicMock()
        backend.makedirs = MagicMock()
        backend.write = MagicMock()
        backend.write_array = MagicMock()
        backend.exists = MagicMock(return_value=False)
        return backend

    def test_basic_writer(self, benchmark, mock_backend, render_vars_full):
        writer = WriterRegistry.get("basic")()
        benchmark(writer.write_frame, 0, render_vars_full, mock_backend)

    def test_coco_writer(
        self, benchmark, mock_backend, rgb_frame, bbox_2d_data, semantic_id_map
    ):
        writer = WriterRegistry.get("coco")()
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticIdMap": semantic_id_map,
        }
        benchmark(writer.write_frame, 0, render_vars, mock_backend)

    def test_kitti_writer(
        self,
        benchmark,
        mock_backend,
        rgb_frame,
        bbox_2d_data,
        bbox_3d_data,
        camera_params,
        semantic_id_map,
    ):
        writer = WriterRegistry.get("kitti")()
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticBoundingBox3D": bbox_3d_data,
            "CameraParams": camera_params,
            "SemanticIdMap": semantic_id_map,
        }
        benchmark(writer.write_frame, 0, render_vars, mock_backend)

    def test_yolo_writer(
        self, benchmark, mock_backend, rgb_frame, bbox_2d_data, semantic_id_map
    ):
        writer = WriterRegistry.get("yolo")()
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticIdMap": semantic_id_map,
        }
        benchmark(writer.write_frame, 0, render_vars, mock_backend)

    def test_bop_writer(self, benchmark, mock_backend, render_vars_full):
        writer = WriterRegistry.get("bop")()
        benchmark(writer.write_frame, 0, render_vars_full, mock_backend)

    def test_lerobot_writer(self, benchmark, mock_backend, rgb_frame):
        # Fresh writer per round so accumulated frames don't grow unbounded.
        render_vars = {"LdrColor": rgb_frame}

        def setup():
            return (WriterRegistry.get("lerobot")(), 0, render_vars, mock_backend), {}

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b), setup=setup, rounds=20
        )

    def test_cosmos_writer(self, benchmark, mock_backend, render_vars_full):
        # Fresh writer per round so accumulated frames don't grow unbounded.
        def setup():
            return (
                (
                    WriterRegistry.get("cosmos_transfer")(),
                    0,
                    render_vars_full,
                    mock_backend,
                ),
                {},
            )

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b), setup=setup, rounds=20
        )


class TestWriterFinalize:
    """Benchmark finalize() with accumulated frame data."""

    @pytest.fixture
    def mock_backend(self):
        backend = MagicMock()
        backend.makedirs = MagicMock()
        backend.write = MagicMock()
        backend.write_array = MagicMock()
        backend.exists = MagicMock(return_value=False)
        backend.read = MagicMock(return_value=b"[]")
        backend.list = MagicMock(return_value=[])
        return backend

    def test_coco_finalize_100_frames(
        self, benchmark, mock_backend, rgb_frame, bbox_2d_data, semantic_id_map
    ):
        """COCO finalize with 100 accumulated frames."""
        writer = WriterRegistry.get("coco")()
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticIdMap": semantic_id_map,
        }
        for i in range(100):
            writer.write_frame(i, render_vars, mock_backend)
        benchmark(writer.finalize, mock_backend)

    def test_kitti_finalize(self, benchmark, mock_backend, render_vars_full):
        """KITTI finalize is a no-op; measures its baseline overhead."""
        writer = WriterRegistry.get("kitti")()
        writer.write_frame(0, render_vars_full, mock_backend)
        benchmark(writer.finalize, mock_backend)

    def test_yolo_finalize_100_frames(
        self, benchmark, mock_backend, rgb_frame, bbox_2d_data, semantic_id_map
    ):
        """YOLO finalize (data.yaml) after 100 accumulated frames."""
        writer = WriterRegistry.get("yolo")()
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticIdMap": semantic_id_map,
        }
        for i in range(100):
            writer.write_frame(i, render_vars, mock_backend)
        benchmark(writer.finalize, mock_backend)

    def test_bop_finalize_20_frames(self, benchmark, mock_backend, render_vars_full):
        """BOP finalize (scene JSON) after 20 accumulated frames."""
        writer = WriterRegistry.get("bop")()
        for i in range(20):
            writer.write_frame(i, render_vars_full, mock_backend)
        benchmark(writer.finalize, mock_backend)

    def test_lerobot_finalize(self, benchmark, mock_backend, rgb_frame):
        """LeRobot finalize (video encode + parquet + meta).

        Requires pyarrow and a video encoder; skipped when unavailable.
        """
        pytest.importorskip("pyarrow")
        from ovannotators.writers.lerobot import LeRobotWriterConfig

        # Pre-check: try a full run to see if video encoder is available.
        # Skip before entering benchmark to avoid empty stats.
        writer = WriterRegistry.get("lerobot")(
            LeRobotWriterConfig(fps=10, frames_per_chunk=1000)
        )
        for i in range(4):
            writer.write_frame(i, {"LdrColor": rgb_frame}, mock_backend)
        try:
            writer.finalize(mock_backend)
        except (ImportError, FileNotFoundError, RuntimeError) as exc:
            pytest.skip(f"video/parquet backend unavailable: {exc}")

        def run():
            w = WriterRegistry.get("lerobot")(
                LeRobotWriterConfig(fps=10, frames_per_chunk=1000)
            )
            for i in range(4):
                w.write_frame(i, {"LdrColor": rgb_frame}, mock_backend)
            w.finalize(mock_backend)

        benchmark.pedantic(run, rounds=2, iterations=1)
