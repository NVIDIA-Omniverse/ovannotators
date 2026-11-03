# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for writer transformations and meaningful finalization work."""

from __future__ import annotations

import numpy as np
import pytest

from ovannotators.backends._encoding import encode_array
from ovannotators.writers._registry import WriterRegistry
from ovannotators.writers.cosmos_transfer import CosmosTransferWriter


class _NullBackend:
    """Accept backend operations without I/O or call-history bookkeeping."""

    def makedirs(self, path: str) -> None:
        return None

    def write(self, path: str, data: bytes) -> None:
        return None

    def write_array(self, path: str, array, encoding: str = "npy") -> None:
        return None

    def exists(self, path: str) -> bool:
        return False

    def read(self, path: str) -> bytes:
        return b"[]"

    def list(self, prefix: str = "") -> list[str]:
        return []


@pytest.fixture
def null_backend() -> _NullBackend:
    return _NullBackend()


class _EncodingNullBackend(_NullBackend):
    """Encode arrays like a real backend, then discard the bytes.

    Use this for writers that encode some outputs themselves (BOP encodes each
    object mask once and writes the bytes to both mask paths). With
    ``_NullBackend`` the encodes a writer delegates to ``write_array`` would be
    free and the ones it does itself would not, so the benchmark would move
    whenever encoding moved between the writer and the backend.
    """

    def write_array(self, path: str, array, encoding: str = "npy") -> None:
        encode_array(path, array, encoding)


@pytest.fixture
def encoding_backend() -> _EncodingNullBackend:
    return _EncodingNullBackend()


class TestWriterPerFrame:
    """Benchmark one independent ``write_frame`` operation per round."""

    def test_basic_writer(self, benchmark, null_backend, render_vars_full):
        def setup():
            return (
                (WriterRegistry.get("basic")(), 0, render_vars_full, null_backend),
                {},
            )

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b),
            setup=setup,
            rounds=10,
            iterations=1,
        )

    def test_coco_writer(
        self, benchmark, null_backend, rgb_frame, bbox_2d_data, semantic_id_map
    ):
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticIdMap": semantic_id_map,
        }

        def setup():
            return (
                (WriterRegistry.get("coco")(), 0, render_vars, null_backend),
                {},
            )

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b),
            setup=setup,
            rounds=50,
            iterations=1,
        )

    def test_kitti_writer(
        self,
        benchmark,
        null_backend,
        rgb_frame,
        bbox_2d_data,
        bbox_3d_data,
        camera_params,
        semantic_id_map,
    ):
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticBoundingBox3D": bbox_3d_data,
            "CameraParams": camera_params,
            "SemanticIdMap": semantic_id_map,
        }

        def setup():
            return (
                (WriterRegistry.get("kitti")(), 0, render_vars, null_backend),
                {},
            )

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b),
            setup=setup,
            rounds=50,
            iterations=1,
        )

    def test_yolo_writer(
        self, benchmark, null_backend, rgb_frame, bbox_2d_data, semantic_id_map
    ):
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticIdMap": semantic_id_map,
        }

        def setup():
            return (
                (WriterRegistry.get("yolo")(), 0, render_vars, null_backend),
                {},
            )

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b),
            setup=setup,
            rounds=50,
            iterations=1,
        )

    def test_bop_writer_encoded(self, benchmark, encoding_backend, render_vars_full):
        def setup():
            return (
                (WriterRegistry.get("bop")(), 0, render_vars_full, encoding_backend),
                {},
            )

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b),
            setup=setup,
            rounds=10,
            iterations=1,
        )

    def test_lerobot_writer(self, benchmark, null_backend, rgb_frame):
        render_vars = {"LdrColor": rgb_frame}

        def setup():
            return (
                (WriterRegistry.get("lerobot")(), 0, render_vars, null_backend),
                {},
            )

        benchmark.pedantic(
            lambda w, f, rv, b: w.write_frame(f, rv, b),
            setup=setup,
            rounds=20,
            iterations=1,
        )


class TestSegmentationColorization:
    """Benchmark the four-channel instance-ID to palette conversion."""

    def test_twelve_ids_at_512_square(self, benchmark):
        ids = np.zeros((12, 4), dtype=np.uint32)
        ids[1:, 0] = np.arange(1, 12, dtype=np.uint32)
        ids[1:, 1] = ids[1:, 0] * 17
        rng = np.random.default_rng(101)
        frame = ids[rng.integers(0, len(ids), size=(512, 512))]
        writer = CosmosTransferWriter()

        benchmark.pedantic(
            writer._seg_to_color,
            args=(frame,),
            rounds=10,
            iterations=1,
        )


class TestWriterFinalize:
    """Benchmark serialization using fresh populated writer state each round."""

    def test_coco_finalize_serialization_100_frames(
        self, benchmark, null_backend, rgb_frame, bbox_2d_data, semantic_id_map
    ):
        render_vars = {
            "LdrColor": rgb_frame,
            "SemanticBoundingBox2DTight": bbox_2d_data,
            "SemanticIdMap": semantic_id_map,
        }

        def setup():
            writer = WriterRegistry.get("coco")()
            for i in range(100):
                writer.write_frame(i, render_vars, null_backend)
            return (writer, null_backend), {}

        benchmark.pedantic(
            lambda w, b: w.finalize(b),
            setup=setup,
            rounds=10,
            iterations=1,
        )

    def test_bop_finalize_1_frame(self, benchmark, null_backend, render_vars_full):
        def setup():
            writer = WriterRegistry.get("bop")()
            writer.write_frame(0, render_vars_full, null_backend)
            return (writer, null_backend), {}

        benchmark.pedantic(
            lambda w, b: w.finalize(b),
            setup=setup,
            rounds=10,
            iterations=1,
        )

    def test_lerobot_finalize(self, benchmark, null_backend, rgb_frame):
        pytest.importorskip("pyarrow")
        from ovannotators.writers.lerobot import LeRobotWriterConfig

        writer = WriterRegistry.get("lerobot")(
            LeRobotWriterConfig(fps=10, frames_per_chunk=1000)
        )
        for i in range(4):
            writer.write_frame(i, {"LdrColor": rgb_frame}, null_backend)
        try:
            writer.finalize(null_backend)
        except (ImportError, FileNotFoundError, RuntimeError) as exc:
            pytest.skip(f"video/parquet backend unavailable: {exc}")

        def setup():
            current = WriterRegistry.get("lerobot")(
                LeRobotWriterConfig(fps=10, frames_per_chunk=1000)
            )
            for i in range(4):
                current.write_frame(i, {"LdrColor": rgb_frame}, null_backend)
            return (current, null_backend), {}

        benchmark.pedantic(
            lambda w, b: w.finalize(b),
            setup=setup,
            rounds=3,
            iterations=1,
        )
