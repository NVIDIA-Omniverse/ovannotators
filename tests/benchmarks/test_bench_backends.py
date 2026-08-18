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

"""Benchmarks for backend I/O operations.

Tests disk backend write throughput with realistic payloads.
"""

from __future__ import annotations


import numpy as np
import pytest

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig


class TestDiskBackendWrite:
    """Benchmark disk backend write operations."""

    @pytest.fixture
    def backend(self, tmp_path) -> DiskBackend:
        return DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))

    def test_write_png_720p(self, benchmark, backend):
        """Write a 720p RGBA PNG to disk."""
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(720, 1280, 4), dtype=np.uint8)
        counter = [0]

        def write(f):
            backend.write_array(f"frame_{counter[0]:06d}.png", frame, encoding="png")
            counter[0] += 1

        benchmark(write, frame)

    def test_write_npy_depth(self, benchmark, backend):
        """Write a 720p float32 depth map as .npy."""
        rng = np.random.default_rng(42)
        depth = rng.uniform(0, 100, size=(720, 1280)).astype(np.float32)
        counter = [0]

        def write(d):
            backend.write_array(f"depth_{counter[0]:06d}.npy", d, encoding="npy")
            counter[0] += 1

        benchmark(write, depth)

    def test_write_json_metadata(self, benchmark, backend):
        """Write a typical metadata JSON blob."""
        import json

        meta = {
            "frame_id": 0,
            "timestamp": 0.0333,
            "objects": [
                {"id": i, "class": f"obj_{i}", "bbox": [10, 20, 100, 150]}
                for i in range(50)
            ],
        }
        data = json.dumps(meta).encode()
        counter = [0]

        def write(d):
            backend.write(f"meta_{counter[0]:06d}.json", d)
            counter[0] += 1

        benchmark(write, data)

    def test_write_jpg_720p(self, benchmark, backend):
        """Write a 720p RGB JPEG to disk."""
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        counter = [0]

        def write(f):
            backend.write_array(f"frame_{counter[0]:06d}.jpg", f, encoding="jpg")
            counter[0] += 1

        benchmark(write, frame)


class TestDiskBackendThroughput:
    """Measure sustained write throughput (multi-frame bursts)."""

    def test_burst_100_frames_jpg(self, benchmark, tmp_path):
        """Write 100 JPEG frames as fast as possible."""
        backend = DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))
        rng = np.random.default_rng(42)
        frames = [
            rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(10)
        ]

        def burst():
            for i in range(100):
                backend.write_array(
                    f"burst_{i:06d}.jpg", frames[i % 10], encoding="jpg"
                )

        benchmark.pedantic(burst, rounds=3, iterations=1)
