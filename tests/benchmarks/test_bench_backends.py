# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for backend serialization and advisory filesystem throughput."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ovannotators._internal.codecs import PillowCodecBackend
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig


class _EncodeOnlyDiskBackend(DiskBackend):
    """Exercise ``DiskBackend.write_array`` serialization without filesystem I/O."""

    def __init__(self) -> None:
        super().__init__(DiskBackendConfig(root_dir=".", overwrite=True))
        self.last_size = 0

    def write(self, path: str, data: bytes) -> None:
        self.last_size = len(data)


class TestDiskBackendSerialization:
    """Measure array serialization independently from shared-runner storage."""

    @pytest.fixture
    def backend(self) -> _EncodeOnlyDiskBackend:
        return _EncodeOnlyDiskBackend()

    def test_encode_png_720p(self, benchmark, backend):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(720, 1280, 4), dtype=np.uint8)
        backend.write_array("frame.png", frame, encoding="png")

        benchmark.pedantic(
            backend.write_array,
            args=("frame.png", frame, "png"),
            rounds=10,
            iterations=1,
        )

        assert backend.last_size > 0

    def test_serialize_npy_depth(self, benchmark, backend):
        rng = np.random.default_rng(42)
        depth = rng.uniform(0, 100, size=(720, 1280)).astype(np.float32)
        backend.write_array("depth.npy", depth, encoding="npy")

        benchmark.pedantic(
            backend.write_array,
            args=("depth.npy", depth, "npy"),
            rounds=20,
            iterations=1,
        )

        assert backend.last_size > depth.nbytes

    def test_encode_jpg_720p(self, benchmark, backend):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        backend.write_array("frame.jpg", frame, encoding="jpg")

        benchmark.pedantic(
            backend.write_array,
            args=("frame.jpg", frame, "jpg"),
            rounds=20,
            iterations=1,
        )

        assert backend.last_size > 0


class TestDiskBackendThroughput:
    """Measure advisory filesystem throughput with pre-serialized payloads."""

    @pytest.fixture
    def backend(self, tmp_path) -> DiskBackend:
        return DiskBackend(DiskBackendConfig(root_dir=str(tmp_path), overwrite=True))

    def test_write_json_metadata(self, benchmark, backend):
        meta = {
            "frame_id": 0,
            "timestamp": 0.0333,
            "objects": [
                {"id": i, "class": f"obj_{i}", "bbox": [10, 20, 100, 150]}
                for i in range(50)
            ],
        }
        data = json.dumps(meta).encode()

        benchmark.pedantic(
            backend.write,
            args=("metadata.json", data),
            rounds=20,
            iterations=1,
        )

    def test_burst_100_preencoded_jpg(self, benchmark, backend):
        rng = np.random.default_rng(42)
        codec = PillowCodecBackend()
        payloads = [
            codec.encode(
                rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8),
                "jpg",
            )
            for _ in range(10)
        ]

        def burst() -> None:
            for i in range(100):
                backend.write(f"burst_{i:06d}.jpg", payloads[i % len(payloads)])

        benchmark.pedantic(burst, rounds=5, iterations=1)
