# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the conformance requirement pins in step with the writers' targets.

``LEROBOT_VERSION`` and ``LEROBOT_LANCEDB_VERSION`` are recorded in published
profile metadata and are the default ``expected_version`` of the canonical
loader checks, so a loader pin bump must change them together with the
isolated conformance requirements that CI installs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ovannotators.profile_writers.lerobot_lance import (
    LEROBOT_LANCEDB_COMMIT,
    LEROBOT_LANCEDB_VERSION,
)
from ovannotators.profile_writers.lerobot_v3 import LEROBOT_COMMIT, LEROBOT_VERSION

REQUIREMENTS = Path(__file__).parent / "requirements"


def _pin(text: str, package: str) -> str:
    match = re.search(
        rf"^{re.escape(package)}(?:\[[^\]]*\])?==([^\s;#]+)", text, re.MULTILINE
    )
    assert match, f"{package} is not pinned"
    return match.group(1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename", ["lerobot-v3-conformance.txt", "lerobot-lance-conformance.txt"]
)
def test_lerobot_pin_matches_the_writer_target(filename: str):
    text = (REQUIREMENTS / filename).read_text(encoding="utf-8")
    assert _pin(text, "lerobot") == LEROBOT_VERSION


@pytest.mark.unit
def test_lerobot_lancedb_pin_matches_the_writer_target():
    text = (REQUIREMENTS / "lerobot-lance-conformance.txt").read_text(encoding="utf-8")
    assert _pin(text, "lerobot-lancedb") == LEROBOT_LANCEDB_VERSION


@pytest.mark.unit
@pytest.mark.parametrize(
    ("filename", "version", "commit"),
    [
        ("lerobot-v3-conformance.txt", LEROBOT_VERSION, LEROBOT_COMMIT),
        (
            "lerobot-lance-conformance.txt",
            LEROBOT_LANCEDB_VERSION,
            LEROBOT_LANCEDB_COMMIT,
        ),
    ],
)
def test_recorded_source_commit_matches_the_writer_target(
    filename: str, version: str, commit: str
):
    text = (REQUIREMENTS / filename).read_text(encoding="utf-8")
    assert f"Source tag v{version} resolves to {commit}." in text


@pytest.mark.unit
def test_no_test_hard_codes_the_current_loader_targets():
    """A loader pin bump must not break tests that should follow the target."""
    tests = Path(__file__).resolve().parent
    offenders = []
    for path in sorted(tests.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for version in (LEROBOT_VERSION, LEROBOT_LANCEDB_VERSION):
            if re.search(rf"[\"'=]{re.escape(version)}[\"']", text):
                offenders.append(f"{path.relative_to(tests)}: {version}")
    assert offenders == [], (
        "use LEROBOT_VERSION / LEROBOT_LANCEDB_VERSION instead of literals"
    )
