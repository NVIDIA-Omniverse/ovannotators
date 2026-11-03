# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark loose files against WebDataset shards of the same encoded samples.

Sizes are configurable with ``OVANNOTATORS_WDS_BENCH_DATASET_MIB`` (default
48) and ``OVANNOTATORS_WDS_BENCH_SHARD_SIZES`` (default ``16 MiB,128 MiB``).
Each benchmark records samples/s, CPU time, tracemalloc peak, temporary disk,
storage requests, hardware and reader configuration in ``extra_info``. Use
``python -m tests.benchmarks.webdataset_harness`` for an isolated-process
report that also includes peak RSS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.webdataset_harness import (
    READER_CONFIG,
    UNITS,
    build_corpus,
    hardware,
    measure,
    package,
    read_loose,
    read_tarfile,
    read_webdataset,
    settings_from_env,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.cpu_only]

DATASET_MIB, SHARD_SIZES = settings_from_env()


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    root = tmp_path_factory.mktemp("wds-bench")
    return {
        unit: build_corpus(root / unit / "loose", unit, DATASET_MIB) for unit in UNITS
    }


@pytest.fixture(scope="module")
def packed(corpora, tmp_path_factory):
    root = tmp_path_factory.mktemp("wds-shards")
    cache = {}

    def get(unit, shard_size):
        if (unit, shard_size) not in cache:
            name = f"{unit}-{shard_size.replace(' ', '')}"
            cache[unit, shard_size] = package(
                corpora[unit], root / name, shard_size, root / "tmp" / name
            )
        return cache[unit, shard_size]

    return get


def _record(benchmark, corpus, result, **extra):
    median = benchmark.stats.stats.median
    benchmark.extra_info.update(
        {
            "samples": corpus.samples,
            "encoded_bytes": corpus.bytes,
            "samples_per_s": corpus.samples / median,
            "mib_per_s": corpus.bytes / 2**20 / median,
            "cpu_s": result["cpu_s"],
            "py_peak_bytes": result["py_peak_bytes"],
            "objects_opened": result.get("objects_opened"),
            "hardware": hardware(),
            "reader_config": READER_CONFIG,
            **extra,
        }
    )


def _check(result, corpus):
    assert result["samples"] == corpus.samples
    assert result["bytes"] >= corpus.bytes


@pytest.mark.parametrize("unit", list(UNITS))
def test_read_loose(benchmark, corpora, unit):
    corpus = corpora[unit]
    result = measure(lambda: read_loose(corpus.root))
    assert result["bytes"] == corpus.bytes
    _check(result, corpus)
    benchmark.pedantic(
        read_loose, args=(corpus.root,), rounds=3, iterations=1, warmup_rounds=1
    )
    _record(benchmark, corpus, result, layout="loose")


@pytest.mark.parametrize("shard_size", SHARD_SIZES)
@pytest.mark.parametrize("unit", list(UNITS))
@pytest.mark.parametrize(
    "reader", [read_tarfile, read_webdataset], ids=["tarfile", "webdataset"]
)
def test_read_shards(benchmark, corpora, packed, unit, shard_size, reader):
    corpus = corpora[unit]
    shards = packed(unit, shard_size)["shards"]
    result = measure(lambda: reader(shards))
    _check(result, corpus)
    benchmark.pedantic(reader, args=(shards,), rounds=3, iterations=1, warmup_rounds=1)
    _record(benchmark, corpus, result, layout="shards", shards=len(shards))


@pytest.mark.parametrize("shard_size", SHARD_SIZES)
@pytest.mark.parametrize("unit", list(UNITS))
def test_package(benchmark, corpora, packed, tmp_path, unit, shard_size):
    corpus = corpora[unit]
    probed = packed(unit, shard_size)
    destination = tmp_path / "out"
    result = measure(lambda: package(corpus, destination, shard_size))
    result["objects_opened"] = corpus.files
    benchmark.pedantic(
        package, args=(corpus, destination, shard_size), rounds=3, iterations=1
    )
    assert Path(destination, "bench/manifest.json").is_file()
    _record(
        benchmark,
        corpus,
        result,
        layout="package",
        shards=len(probed["shards"]),
        shard_bytes=probed["shard_bytes"],
        temp_peak_bytes=probed["temp_peak_bytes"],
        publish_requests=probed["requests"],
        loose_write_requests=corpus.write_requests,
    )
