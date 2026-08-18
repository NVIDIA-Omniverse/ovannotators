# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the README minimal capture snippet."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _extract_minimal_example_python(readme_text: str) -> str:
    """Extract the python snippet under the README Minimal Example section.

    Args:
        readme_text: Full README markdown content.

    Returns:
        The Python code block contents for the Minimal Example section.

    Raises:
        AssertionError: If the Minimal Example python block is missing.
    """
    match = re.search(
        r"### Minimal Example.*?```python\n(.*?)\n```",
        readme_text,
        flags=re.DOTALL,
    )
    assert match is not None, "README Minimal Example python snippet not found"
    return match.group(1)


@pytest.mark.gpu
def test_readme_minimal_example_runs(monkeypatch) -> None:
    """Execute the README minimal example and verify it writes frame outputs.

    The output is written to the repo's ``_out/basic_minimal`` directory (the
    same path the README documents) and deliberately left in place after the
    run so it can be inspected. ``DiskBackend`` is configured with
    ``overwrite=True``, so re-runs replace the previous output cleanly.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to isolate cwd.
    """
    repo_root = Path(__file__).resolve().parents[1]
    readme_path = repo_root / "README.md"
    snippet = _extract_minimal_example_python(readme_path.read_text(encoding="utf-8"))

    scene_path = repo_root / "tests" / "data" / "test_scene.usda"
    out_dir = repo_root / "_out" / "basic_minimal"
    snippet = snippet.replace(
        '"tests/data/test_scene.usda"',
        repr(str(scene_path.resolve())),
    )
    snippet = snippet.replace(
        '"_out/basic_minimal"',
        repr(str(out_dir.resolve())),
    )

    monkeypatch.chdir(repo_root)
    exec(snippet, {}, {})  # noqa: S102

    frame_images = sorted(out_dir.glob("*/ldr_color.png"))
    assert len(frame_images) == 3
