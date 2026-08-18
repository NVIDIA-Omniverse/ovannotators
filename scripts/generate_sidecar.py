#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a structured JSON sidecar from JUnit XML test results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _case_name(testcase: ET.Element) -> str:
    classname = testcase.get("classname", "")
    name = testcase.get("name", "")
    return f"{classname}::{name}" if classname else name


def parse_junit_xml(xml_path: Path) -> dict[str, Any]:
    """Parse JUnit XML and extract summary, failures, and skips."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results: dict[str, Any] = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "failures": [],
        "skips": [],
    }

    for testsuite in root.iter("testsuite"):
        results["total"] += int(testsuite.get("tests", 0) or 0)
        results["errors"] += int(testsuite.get("errors", 0) or 0)

    # Some emitters put counts only on a top-level testsuite.
    if results["total"] == 0 and root.tag == "testsuite":
        results["total"] = int(root.get("tests", 0) or 0)
        results["errors"] = int(root.get("errors", 0) or 0)

    for testcase in root.iter("testcase"):
        time_taken = float(testcase.get("time", 0) or 0)
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")

        if failure is not None or error is not None:
            node = failure if failure is not None else error
            results["failed"] += 1
            results["failures"].append(
                {
                    "test": _case_name(testcase),
                    "time": time_taken,
                    "type": node.get("type", ""),
                    "message": node.get("message", ""),
                    "text": (node.text or "")[:4000],
                }
            )
        elif skipped is not None:
            results["skipped"] += 1
            results["skips"].append(
                {
                    "test": _case_name(testcase),
                    "reason": skipped.get("message", skipped.text or ""),
                }
            )
        else:
            results["passed"] += 1

    if results["total"] == 0:
        results["total"] = results["passed"] + results["failed"] + results["skipped"]

    return results


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_context() -> dict[str, Any]:
    """Return commit, branch, recent commits, and changed files when available."""
    context: dict[str, Any] = {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
    }
    recent = _git(["log", "--oneline", "-5", "--format=%H %s"])
    if recent:
        context["recent_commits"] = [
            {"sha": line.split(" ", 1)[0], "message": line.split(" ", 1)[1]}
            for line in recent.splitlines()
            if " " in line
        ]
    for base in ("origin/main", "main"):
        changed = _git(["diff", "--name-only", f"{base}...HEAD"])
        if changed:
            context["changed_files"] = changed.splitlines()
            break
    return context


def implicate_files(failures: list[dict[str, Any]], changed_files: list[str]) -> None:
    """Annotate failures whose tracebacks mention changed files."""
    for failure in failures:
        text = " ".join(
            str(failure.get(key, "")) for key in ("test", "message", "text")
        )
        implicated = [f for f in changed_files if f in text or Path(f).name in text]
        if implicated:
            failure["implicated_files"] = implicated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit_xml", type=Path, help="Path to JUnit XML file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON path")
    args = parser.parse_args()

    if not args.junit_xml.exists():
        print(f"Error: {args.junit_xml} not found", file=sys.stderr)
        sys.exit(1)

    results = parse_junit_xml(args.junit_xml)
    git_context = get_git_context()
    implicate_files(results["failures"], git_context.get("changed_files", []))

    sidecar = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.junit_xml),
        "git": git_context,
        "summary": {
            "total": results["total"],
            "passed": results["passed"],
            "failed": results["failed"],
            "skipped": results["skipped"],
            "errors": results["errors"],
        },
        "failures": results["failures"],
        "skips": results["skips"],
    }

    output = args.output or args.junit_xml.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Sidecar written to {output}")


if __name__ == "__main__":
    main()
