# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for paired benchmark regression decisions and reporting."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "compare_benchmarks.py"
    spec = importlib.util.spec_from_file_location("compare_benchmarks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_benchmarks = _load_module()

STABLE_BENCHMARK = (
    "tests/benchmarks/test_bench_cosmos.py::"
    "TestCannyPerformance::test_canny_default_sigma[720p]"
)
NOISY_BENCHMARK = (
    "tests/benchmarks/test_bench_backends.py::"
    "TestDiskBackendThroughput::test_burst_100_frames_jpg"
)


def _result(
    *benchmarks: tuple[str, float, float, int],
    commit: str = "a" * 40,
) -> dict:
    return {
        "commit_info": {"id": commit},
        "machine_info": {
            "system": "Linux",
            "python_version": "3.10.21",
            "node": "paired-runner",
            "cpu": {"brand_raw": "Test CPU"},
        },
        "benchmarks": [
            {
                "fullname": name,
                "stats": {
                    "mean": median,
                    "median": median,
                    "stddev": stddev,
                    "rounds": rounds,
                },
            }
            for name, median, stddev, rounds in benchmarks
        ],
    }


def test_stable_paired_regression_blocks() -> None:
    baseline = _result((STABLE_BENCHMARK, 1.0, 0.02, 12))
    current = _result((STABLE_BENCHMARK, 1.6, 0.03, 12), commit="b" * 40)

    markdown, failed = compare_benchmarks.compare(
        current, baseline, warn_threshold=0.05, fail_threshold=0.50
    )

    assert failed
    assert "1 blocking regression(s)" in markdown
    assert "❌ block" in markdown
    assert "Python 3.10.21, Test CPU, node paired-runner" in markdown


def test_noisy_filesystem_regression_warns_without_blocking() -> None:
    baseline = _result((NOISY_BENCHMARK, 0.690, 0.067, 3))
    current = _result((NOISY_BENCHMARK, 4.408, 4.190, 3))

    markdown, failed = compare_benchmarks.compare(
        current, baseline, warn_threshold=0.05, fail_threshold=0.50
    )

    assert not failed
    assert "advisory" in markdown
    assert "fewer than" not in markdown
    assert "+538.8%" in markdown


def test_unstable_blocking_benchmark_is_downgraded_to_advisory() -> None:
    baseline = _result((STABLE_BENCHMARK, 1.0, 0.02, 12))
    current = _result((STABLE_BENCHMARK, 1.8, 0.50, 12))

    markdown, failed = compare_benchmarks.compare(
        current, baseline, warn_threshold=0.05, fail_threshold=0.50
    )

    assert not failed
    assert "advisory: relative stddev above 20%" in markdown


def test_low_sample_blocking_benchmark_is_downgraded_to_advisory() -> None:
    baseline = _result((STABLE_BENCHMARK, 1.0, 0.02, 5))
    current = _result((STABLE_BENCHMARK, 1.8, 0.03, 5))

    markdown, failed = compare_benchmarks.compare(
        current, baseline, warn_threshold=0.05, fail_threshold=0.50
    )

    assert not failed
    assert "advisory: fewer than 10 rounds" in markdown


def test_comparison_uses_median_instead_of_outlier_skewed_mean() -> None:
    baseline = _result((STABLE_BENCHMARK, 1.0, 0.02, 12))
    current = _result((STABLE_BENCHMARK, 1.01, 0.02, 12))
    current["benchmarks"][0]["stats"]["mean"] = 2.0

    markdown, failed = compare_benchmarks.compare(
        current, baseline, warn_threshold=0.05, fail_threshold=0.50
    )

    assert markdown == ""
    assert not failed


def test_non_finite_baseline_medians_are_not_compared() -> None:
    current = _result((STABLE_BENCHMARK, 1.6, 0.03, 12))

    for invalid_median in (float("nan"), float("inf"), float("-inf")):
        baseline = _result((STABLE_BENCHMARK, invalid_median, 0.02, 12))
        markdown, failed = compare_benchmarks.compare(
            current, baseline, warn_threshold=0.05, fail_threshold=0.50
        )

        assert markdown == ""
        assert not failed


def test_compare_emits_no_comment_without_comparable_regressions() -> None:
    baseline = _result((STABLE_BENCHMARK, 1.0, 0.02, 12))
    current = _result((STABLE_BENCHMARK, 0.9, 0.02, 12))

    markdown, failed = compare_benchmarks.compare(
        current, baseline, warn_threshold=0.05, fail_threshold=0.50
    )

    assert markdown == ""
    assert not failed
