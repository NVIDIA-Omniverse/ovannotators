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

"""Comprehensive tests for LeRobotWriter — targeting full finalize() coverage."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import numpy as np
import pytest

from ovannotators.exceptions import MissingRenderVarError
from ovannotators.writers.lerobot import LeRobotWriter, LeRobotWriterConfig


class MockBackend:
    """In-memory backend for testing."""

    def __init__(self):
        self.written: dict[str, bytes] = {}
        self.dirs: set[str] = set()

    def write(self, path: str, data: bytes) -> None:
        self.written[path] = data

    def makedirs(self, path: str) -> None:
        self.dirs.add(path)

    def read(self, path: str) -> bytes:
        return self.written[path]


# ---------------------------------------------------------------------------
# Config / Init
# ---------------------------------------------------------------------------


class TestLeRobotConfig:
    def test_default_config(self):
        cfg = LeRobotWriterConfig()
        assert cfg.fps == 30
        assert cfg.episode_index == 0
        assert cfg.camera_keys == ["top"]
        assert cfg.frames_per_chunk == 1000

    def test_custom_config(self):
        cfg = LeRobotWriterConfig(
            fps=60,
            episode_index=5,
            task_description="pick and place",
            camera_keys=["front", "wrist"],
            frames_per_chunk=500,
        )
        assert cfg.fps == 60
        assert cfg.camera_keys == ["front", "wrist"]


# ---------------------------------------------------------------------------
# write_frame
# ---------------------------------------------------------------------------


class TestLeRobotWriteFrame:
    def test_missing_ldrcolor_raises(self):
        writer = LeRobotWriter()
        backend = MockBackend()
        with pytest.raises(MissingRenderVarError):
            writer.write_frame(0, {"CameraParams": np.zeros(25)}, backend)

    def test_camera_params_not_required(self):
        # CameraParams is not consumed by the writer; LdrColor alone is enough.
        writer = LeRobotWriter()
        backend = MockBackend()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        writer.write_frame(0, {"LdrColor": img}, backend)
        assert len(writer._rows) == 1

    def test_accumulates_frames(self):
        writer = LeRobotWriter(LeRobotWriterConfig(camera_keys=["top"]))
        backend = MockBackend()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cam = np.zeros(25, dtype=np.float64)

        for i in range(5):
            writer.write_frame(i, {"LdrColor": img, "CameraParams": cam}, backend)

        assert len(writer._rows) == 5
        assert writer._frame_index == 5

    def test_rgba_to_rgb(self):
        writer = LeRobotWriter()
        backend = MockBackend()
        # RGBA input → should be stripped to RGB
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        cam = np.zeros(25, dtype=np.float64)
        writer.write_frame(0, {"LdrColor": img, "CameraParams": cam}, backend)
        stored = writer._frame_images["top"][0]
        assert stored.shape == (64, 64, 3)

    def test_float_to_uint8(self):
        writer = LeRobotWriter()
        backend = MockBackend()
        img = np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)
        cam = np.zeros(25, dtype=np.float64)
        writer.write_frame(0, {"LdrColor": img, "CameraParams": cam}, backend)
        stored = writer._frame_images["top"][0]
        assert stored.dtype == np.uint8

    def test_grayscale_to_3ch(self):
        writer = LeRobotWriter()
        backend = MockBackend()
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cam = np.zeros(25, dtype=np.float64)
        writer.write_frame(0, {"LdrColor": img, "CameraParams": cam}, backend)
        stored = writer._frame_images["top"][0]
        assert stored.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# finalize — full path (with PyNvVideoCodec)
# ---------------------------------------------------------------------------


def _mock_encode_video(frames, fps):
    """Mock video encoder for unit tests (NVENC needs >64px)."""
    return b"\x00\x00\x00\x1c" + b"ftypisom" + b"\x00" * 100  # fake MP4


@patch("ovannotators.writers.lerobot._encode_video", _mock_encode_video)
class TestLeRobotFinalize:
    def _make_writer_with_frames(self, n_frames=3, fps=30, camera_keys=None):
        """Helper: create writer with n_frames already accumulated."""
        if camera_keys is None:
            camera_keys = ["top"]
        cfg = LeRobotWriterConfig(
            fps=fps,
            camera_keys=camera_keys,
            task_description="test task",
            frames_per_chunk=1000,
        )
        writer = LeRobotWriter(cfg)
        backend = MockBackend()
        for i in range(n_frames):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cam = np.zeros(25, dtype=np.float64)
            writer.write_frame(i, {"LdrColor": img, "CameraParams": cam}, backend)
        return writer, backend

    def test_finalize_writes_video(self):
        writer, backend = self._make_writer_with_frames(5)
        writer.finalize(backend)
        # Should have written a video file
        video_keys = [k for k in backend.written if k.endswith(".mp4")]
        assert len(video_keys) == 1
        assert "observation.images.top" in video_keys[0]
        # Verify it's real MP4 bytes
        mp4_data = backend.written[video_keys[0]]
        assert len(mp4_data) > 0

    def test_finalize_writes_parquet(self):
        writer, backend = self._make_writer_with_frames(3)
        writer.finalize(backend)
        parquet_keys = [k for k in backend.written if k.endswith(".parquet")]
        assert len(parquet_keys) == 1
        assert "chunk-000" in parquet_keys[0]

    def test_finalize_writes_meta_json(self):
        writer, backend = self._make_writer_with_frames(3)
        writer.finalize(backend)
        assert "meta/info.json" in backend.written
        assert "meta/tasks.json" in backend.written
        assert "meta/episodes.json" in backend.written
        assert "meta/stats.json" in backend.written

    def test_finalize_info_json_content(self):
        writer, backend = self._make_writer_with_frames(5, fps=24)
        writer.finalize(backend)
        info = json.loads(backend.written["meta/info.json"])
        assert info["fps"] == 24
        assert info["total_frames"] == 5
        assert info["total_episodes"] == 1
        assert "observation.images.top" in info["features"]

    def test_finalize_tasks_json_content(self):
        writer, backend = self._make_writer_with_frames(2)
        writer.finalize(backend)
        tasks = json.loads(backend.written["meta/tasks.json"])
        assert tasks[0]["task"] == "test task"

    def test_finalize_episodes_json_content(self):
        writer, backend = self._make_writer_with_frames(10)
        writer.finalize(backend)
        episodes = json.loads(backend.written["meta/episodes.json"])
        assert episodes[0]["length"] == 10

    def test_finalize_idempotent(self):
        writer, backend = self._make_writer_with_frames(3)
        writer.finalize(backend)
        n_writes = len(backend.written)
        writer.finalize(backend)  # second call should be no-op
        assert len(backend.written) == n_writes

    def test_finalize_empty_writer(self):
        writer = LeRobotWriter()
        backend = MockBackend()
        writer.finalize(backend)  # no frames → should not write anything
        assert len(backend.written) == 0

    def test_finalize_multi_camera(self):
        writer, backend = self._make_writer_with_frames(
            3, camera_keys=["front", "wrist"]
        )
        writer.finalize(backend)
        video_keys = [k for k in backend.written if k.endswith(".mp4")]
        assert len(video_keys) == 2
        assert any("front" in k for k in video_keys)
        assert any("wrist" in k for k in video_keys)

    def test_finalize_multiple_chunks(self):
        cfg = LeRobotWriterConfig(
            fps=30,
            camera_keys=["top"],
            frames_per_chunk=5,  # small chunk size
        )
        writer = LeRobotWriter(cfg)
        backend = MockBackend()
        for i in range(12):
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cam = np.zeros(25, dtype=np.float64)
            writer.write_frame(i, {"LdrColor": img, "CameraParams": cam}, backend)
        writer.finalize(backend)
        # 12 frames / 5 per chunk = 3 chunks
        video_keys = [k for k in backend.written if k.endswith(".mp4")]
        assert len(video_keys) == 3
        parquet_keys = [k for k in backend.written if k.endswith(".parquet")]
        assert len(parquet_keys) == 3

    def test_finalize_creates_directories(self):
        writer, backend = self._make_writer_with_frames(3)
        writer.finalize(backend)
        assert "meta" in backend.dirs
        assert any("videos" in d for d in backend.dirs)
        assert any("data" in d for d in backend.dirs)

    def test_parquet_row_schema(self):
        writer, backend = self._make_writer_with_frames(2, fps=10)
        writer.finalize(backend)
        import pyarrow.parquet as pq

        parquet_keys = [k for k in backend.written if k.endswith(".parquet")]
        table = pq.read_table(io.BytesIO(backend.written[parquet_keys[0]]))
        assert "index" in table.column_names
        assert "episode_index" in table.column_names
        assert "frame_index" in table.column_names
        assert "timestamp" in table.column_names
        assert "task_index" in table.column_names
        # Verify timestamp calculation
        timestamps = table["timestamp"].to_pylist()
        assert timestamps[0] == 0.0
        assert abs(timestamps[1] - 0.1) < 1e-6  # 1/10 fps

    def test_finalize_custom_episode_index(self):
        cfg = LeRobotWriterConfig(fps=30, camera_keys=["top"], episode_index=5)
        writer = LeRobotWriter(cfg)
        backend = MockBackend()
        for i in range(3):
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            writer.write_frame(i, {"LdrColor": img}, backend)
        writer.finalize(backend)
        # Episode index appears in parquet + video filenames ...
        assert any("episode_000005.parquet" in k for k in backend.written)
        assert any("episode_000005.mp4" in k for k in backend.written)
        # ... and in every parquet row.
        import pyarrow.parquet as pq

        parquet_key = next(k for k in backend.written if k.endswith(".parquet"))
        table = pq.read_table(io.BytesIO(backend.written[parquet_key]))
        assert set(table["episode_index"].to_pylist()) == {5}
