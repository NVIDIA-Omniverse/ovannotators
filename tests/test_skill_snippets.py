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

"""Guards for the agent-skills system under ``skills/``.

Enforces the convention documented in ``skills/README.md``:

* every ``> **Source:**`` reference in a ``SKILL.md`` resolves to a real
  ``# [snippet:name]`` / ``# [/snippet:name]`` marker pair in the referenced
  file, so skill code never goes stale silently;
* each skill's frontmatter ``name`` matches its directory and its root holds
  only Agent Skills spec fields;
* the ``.agents/skills`` and ``.claude/skills`` mirrors are symlinks resolving
  to the canonical ``skills/`` tree;
* each ``evals/evals.json`` for the NVIDIA skill catalog names its skill, has
  unique ids, includes both a loading case and a negative routing case, and
  grades positive cases on the outcome rather than on reading the skill.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "skills"

# A "> **Source:**" line: first backticked span is the file path, every
# backticked span after the word "snippet"/"snippets" is a snippet name.
_SOURCE_LINE = re.compile(r"^>\s*\*\*Source:\*\*\s*(.+)$", re.MULTILINE)
_BACKTICKED = re.compile(r"`([^`]+)`")


def _skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/SKILL.md"))


def _iter_source_references():
    """Yield ``(skill_md, rel_path, snippet_name)`` for every Source reference."""
    for skill_md in _skill_files():
        text = skill_md.read_text(encoding="utf-8")
        for line in _SOURCE_LINE.findall(text):
            tokens = _BACKTICKED.findall(line)
            if not tokens:
                continue
            rel_path, snippet_names = tokens[0], tokens[1:]
            for name in snippet_names:
                yield skill_md, rel_path, name


def test_skills_directory_present():
    assert _skill_files(), "no skills/*/SKILL.md found"


@pytest.mark.parametrize("skill_md", _skill_files(), ids=lambda p: p.parent.name)
def test_skill_name_matches_directory(skill_md: Path):
    """The frontmatter ``name:`` must equal the skill's directory name."""
    text = skill_md.read_text(encoding="utf-8")
    match = re.search(r"^name:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match, f"{skill_md} has no 'name:' frontmatter field"
    assert match.group(1) == skill_md.parent.name


# Root frontmatter keys in the Agent Skills spec; anything else goes under metadata.
_SPEC_ROOT_KEYS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}


@pytest.mark.parametrize("skill_md", _skill_files(), ids=lambda p: p.parent.name)
def test_skill_frontmatter_root_keys(skill_md: Path):
    """Only spec fields sit at the frontmatter root; validators ignore the rest."""
    text = skill_md.read_text(encoding="utf-8")
    frontmatter = text.split("---\n", 2)[1]
    keys = set(re.findall(r"^([A-Za-z][\w-]*):", frontmatter, re.MULTILINE))
    assert keys <= _SPEC_ROOT_KEYS, (
        f"move {sorted(keys - _SPEC_ROOT_KEYS)} under metadata"
    )


def test_source_references_resolve():
    """Every ``> **Source:**`` reference points at a real snippet marker."""
    references = list(_iter_source_references())
    assert references, "expected at least one > **Source:** reference in skills/"

    errors: list[str] = []
    for skill_md, rel_path, name in references:
        target = REPO_ROOT / rel_path
        rel = skill_md.relative_to(REPO_ROOT)
        if not target.is_file():
            errors.append(f"{rel}: missing source file '{rel_path}'")
            continue
        body = target.read_text(encoding="utf-8")
        if f"# [snippet:{name}]" not in body:
            errors.append(f"{rel}: no '# [snippet:{name}]' in '{rel_path}'")
        if f"# [/snippet:{name}]" not in body:
            errors.append(f"{rel}: no '# [/snippet:{name}]' in '{rel_path}'")
    assert not errors, "Unresolved skill snippet references:\n" + "\n".join(errors)


@pytest.mark.parametrize("mirror", [".agents/skills", ".claude/skills"])
def test_skill_mirror_is_symlink_to_canonical(mirror: str):
    """The agent-runtime mirrors are symlinks resolving to ``skills/``."""
    link = REPO_ROOT / mirror
    assert link.is_symlink(), f"{mirror} should be a symlink"
    assert link.resolve() == SKILLS_DIR.resolve(), f"{mirror} must resolve to skills/"


def _eval_datasets() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/evals/evals.json"))


@pytest.mark.parametrize("dataset", _eval_datasets(), ids=lambda p: p.parents[1].name)
def test_eval_dataset_shape(dataset: Path):
    """Catalog eval datasets name their skill and carry positive and negative cases."""
    skill = dataset.parents[1].name
    data = json.loads(dataset.read_text(encoding="utf-8"))
    assert data["skill_name"] == skill
    ids = [case["id"] for case in data["evals"]]
    assert len(ids) == len(set(ids)), f"duplicate eval ids in {dataset}"
    for case in data["evals"]:
        assert case["prompt"] and case["expected_output"] and case["assertions"]
        assert case["expected_skill"] in (skill, None)
        if case["expected_skill"]:
            # The baseline arm has no skill, so outcome checks must not require it.
            text = " ".join(
                [case["prompt"], case["expected_output"], *case["assertions"]]
            )
            assert "SKILL.md" not in text, (
                f"{case['id']} grades skill use, not the outcome"
            )
    expected = [case["expected_skill"] for case in data["evals"]]
    assert skill in expected, "needs at least one case that loads the skill"
    assert None in expected, "needs at least one negative routing case"
    assert (dataset.parents[1] / "skill-card.md").is_file()
