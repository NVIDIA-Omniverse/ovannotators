<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# ovannotators Skills Directory

This directory contains structured skill files for AI coding agents (Cursor,
Claude Code, etc.) working on the `ovannotators` codebase. Each skill is a
self-contained guide for one ovannotators task. The format follows the
[Agent Skills specification](https://agentskills.io/specification).

## Structure

Each subdirectory contains a single `SKILL.md` file:

```
skills/
  new-writer/SKILL.md      # Add an output-format writer
  new-backend/SKILL.md     # Add a storage backend
  README.md                # This file
```

The same tree is exposed read-only to other agent runtimes via symlinks at the
repo root: `.agents/skills -> ../skills` and `.claude/skills -> ../skills`.
`skills/` is the single source of truth; never edit the mirrors directly.

## SKILL.md Format

```markdown
---
name: skill-name                 # matches the directory name
description: >
  What this skill covers. Use when [trigger phrases].
license: Apache-2.0
version: "0.1.0"
author: NVIDIA ovannotators
tags:
  - ovannotators
  - topic
tools:
  - Read
  - Grep
---

# Skill Title

## When to Use         Activation conditions and trigger phrases.
## Inputs              Spec resolution (URL/file/prompt) and what to gather.
## Prerequisites       Repo/API assumptions before following the skill.
## Instructions        Numbered, agent-directed steps.
## Output Format       Expected response or implementation summary.
## Scripts             Executable helpers (or "none").
## Key Types / Functions   Quick reference of the API surface involved.
## Limitations         Known constraints.
## Troubleshooting     Gotchas to watch for.
## References          Related skills and source-of-truth conventions.
```

## Code Snippet References

Skills reference live, tested code instead of duplicating snippets inline, so
examples stay accurate as the API evolves. Test code is the single source of
truth; skills reference it -- never the other way around.

### Marker format in test files

Wrap illustrative code under `tests/docs/python/` in markers:

```python
# [snippet:doc-writer-class]
class ManifestWriter:
    ...
# [/snippet:doc-writer-class]
```

Snippet names are kebab-case, prefixed with `doc-`, and unique across the
`tests/docs/` tree.

### Reference format in SKILL.md

Replace inline code blocks with a blockquote directive:

```markdown
> **Source:** `tests/docs/python/test_custom_writer.py` snippet `doc-writer-class`
```

Agents read the referenced file between the `# [snippet:name]` and
`# [/snippet:name]` markers to get the current code. Small fragments (a one-line
`pyproject.toml` entry-point, an import) may stay inline -- don't over-engineer.

## Adding a New Skill

1. **Write the tested snippets first** under `tests/docs/python/`, wrapping each
   illustrative region in `doc-*` snippet markers, and update the inventory in
   `tests/docs/AGENTS.md`.
2. Create a directory under `skills/` named in kebab-case; set the `SKILL.md`
   `name` to match.
3. Write `SKILL.md` following the format above. **Do not write inline code
   blocks for API usage** -- reference the test snippets via `> **Source:**`.
4. `tests/test_skill_snippets.py` enforces that every `> **Source:**` reference
   resolves and that the `.agents`/`.claude` mirrors point at `skills/`.

## Updating Skills

When you change the ovannotators API in a way that affects a skill, update the
tested snippet and the corresponding `SKILL.md` in the same change.
