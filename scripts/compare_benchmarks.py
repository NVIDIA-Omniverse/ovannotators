# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare paired benchmark results and enforce stable regression checks."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path
from typing import Any

SECONDS_TO_MS = 1000.0
MIN_BLOCKING_ROUNDS = 10
MAX_RELATIVE_STDDEV = 0.20

BLOCKING_PATTERNS = (
    "tests/benchmarks/test_bench_cosmos.py::TestCannyPerformance::*",
    "tests/benchmarks/test_bench_image_codecs.py::*",
    "tests/benchmarks/test_bench_writers.py::TestWriterPerFrame::test_basic_writer*",
    "tests/benchmarks/test_bench_writers.py::TestWriterPerFrame::test_bop_writer*",
    "tests/benchmarks/test_bench_writers.py::TestWriterPerFrame::test_kitti_writer*",
    "tests/benchmarks/test_bench_writers.py::TestWriterPerFrame::test_lerobot_writer*",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _benchmark_key(item: dict[str, Any]) -> str:
    return str(item.get("fullname") or item.get("name") or item.get("group"))


def _benchmarks_by_key(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not data:
        return {}
    return {_benchmark_key(item): item for item in data.get("benchmarks", [])}


def _stat(item: dict[str, Any], name: str) -> float:
    return float(item.get("stats", {}).get(name, math.nan))


def _median_ms(item: dict[str, Any]) -> float:
    return _stat(item, "median") * SECONDS_TO_MS


def _rounds(item: dict[str, Any]) -> int:
    return int(item.get("stats", {}).get("rounds", 0))


def _relative_stddev(item: dict[str, Any]) -> float:
    mean = _stat(item, "mean")
    stddev = _stat(item, "stddev")
    if not math.isfinite(mean) or mean <= 0 or not math.isfinite(stddev):
        return math.inf
    return stddev / mean


def _machine(data: dict[str, Any] | None) -> str:
    if not data:
        return "unknown"
    machine = data.get("machine_info", {})
    cpu = machine.get("cpu", {})
    system = machine.get("system", "unknown-os")
    python = machine.get("python_version", "unknown-python")
    processor = cpu.get("brand_raw") or machine.get("processor") or "unknown-cpu"
    node = machine.get("node", "unknown-node")
    return f"{system}, Python {python}, {processor}, node {node}"


def _commit(data: dict[str, Any] | None) -> str:
    if not data:
        return "unknown"
    commit_info = data.get("commit_info", {})
    commit_id = (
        commit_info.get("id") or data.get("metadata", {}).get("sha") or "unknown"
    )
    return str(commit_id)[:12]


def _policy(key: str) -> str:
    if any(fnmatch.fnmatchcase(key, pattern) for pattern in BLOCKING_PATTERNS):
        return "blocking"
    return "advisory"


def _quality_reason(
    current_item: dict[str, Any], baseline_item: dict[str, Any]
) -> str | None:
    current_rounds = _rounds(current_item)
    baseline_rounds = _rounds(baseline_item)
    if min(current_rounds, baseline_rounds) < MIN_BLOCKING_ROUNDS:
        return f"fewer than {MIN_BLOCKING_ROUNDS} rounds"
    current_cv = _relative_stddev(current_item)
    baseline_cv = _relative_stddev(baseline_item)
    if max(current_cv, baseline_cv) > MAX_RELATIVE_STDDEV:
        return f"relative stddev above {MAX_RELATIVE_STDDEV:.0%}"
    return None


def _format_delta(delta: float) -> str:
    if math.isnan(delta):
        return "n/a"
    return f"{delta:+.1%}"


def _format_cv(item: dict[str, Any]) -> str:
    value = _relative_stddev(item)
    return "n/a" if not math.isfinite(value) else f"{value:.1%}"


def compare(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    warn_threshold: float,
    fail_threshold: float | None,
) -> tuple[str, bool]:
    current_benchmarks = _benchmarks_by_key(current)
    baseline_benchmarks = _benchmarks_by_key(baseline)
    if not current_benchmarks or not baseline_benchmarks:
        return "", False

    warnings: list[tuple[str, float, str, str | None]] = []
    failed = False
    for key, current_item in sorted(current_benchmarks.items()):
        baseline_item = baseline_benchmarks.get(key)
        if baseline_item is None:
            continue
        current_median = _stat(current_item, "median")
        baseline_median = _stat(baseline_item, "median")
        if (
            not math.isfinite(current_median)
            or not math.isfinite(baseline_median)
            or baseline_median <= 0
        ):
            continue
        delta = (current_median - baseline_median) / baseline_median
        if delta <= warn_threshold:
            continue
        policy = _policy(key)
        quality_reason = _quality_reason(current_item, baseline_item)
        if (
            policy == "blocking"
            and quality_reason is None
            and fail_threshold is not None
            and delta > fail_threshold
        ):
            failed = True
        warnings.append((key, delta, policy, quality_reason))

    if not warnings:
        return "", False

    blocked_count = sum(
        1
        for _key, delta, policy, reason in warnings
        if policy == "blocking"
        and reason is None
        and fail_threshold is not None
        and delta > fail_threshold
    )
    lines = [
        "## 🚦 Paired benchmark comparison",
        "",
        f"**Summary:** {len(warnings)} warning(s) above {warn_threshold:.0%}; "
        f"{blocked_count} blocking regression(s).",
        "",
        f"Current: `{_commit(current)}` ({_machine(current)})",
        f"Baseline: `{_commit(baseline)}` ({_machine(baseline)})",
        "",
        "| Benchmark | Policy | Baseline median | Current median | Delta | Rounds (base/current) | CV (base/current) | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, delta, policy, quality_reason in warnings:
        current_item = current_benchmarks[key]
        baseline_item = baseline_benchmarks[key]
        is_blocked = (
            policy == "blocking"
            and quality_reason is None
            and fail_threshold is not None
            and delta > fail_threshold
        )
        if is_blocked:
            result = "❌ block"
        elif quality_reason is not None and policy == "blocking":
            result = f"⚠️ advisory: {quality_reason}"
        else:
            result = "⚠️ advisory"
        lines.append(
            f"| `{key}` | {policy} | {_median_ms(baseline_item):.3f} ms | "
            f"{_median_ms(current_item):.3f} ms | {_format_delta(delta)} | "
            f"{_rounds(baseline_item)}/{_rounds(current_item)} | "
            f"{_format_cv(baseline_item)}/{_format_cv(current_item)} | {result} |"
        )

    if fail_threshold is not None:
        lines.extend(
            [
                "",
                f"Blocking threshold: {fail_threshold:.0%} paired-median regression. "
                f"Blocking samples require at least {MIN_BLOCKING_ROUNDS} rounds and "
                f"relative standard deviation at or below {MAX_RELATIVE_STDDEV:.0%}.",
            ]
        )
    return "\n".join(lines) + "\n", failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warn-threshold", type=float, default=0.05)
    parser.add_argument("--fail-threshold", type=float, default=None)
    args = parser.parse_args()

    current = _load_json(args.current)
    if current is None:
        raise FileNotFoundError(f"Current benchmark JSON not found: {args.current}")
    baseline = _load_json(args.baseline)
    markdown, failed = compare(
        current,
        baseline,
        warn_threshold=args.warn_threshold,
        fail_threshold=args.fail_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
