<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# AGENTS.md - Documentation Tests

This directory keeps every code example used by the agent skills under
`skills/` tested. **Test snippets are the single source of truth** -- skills
reference them, never the other way around.

## Why This Exists

Hand-written code blocks go stale: they drift from real API signatures, miss
imports, and teach patterns that no longer work. By requiring all skill code to
live in tested files with snippet markers, the examples break when the API
changes -- which is the point.

## Data flow: test -> skill

```
test file (tests/docs/python/*.py)
    contains snippet markers: # [snippet:doc-name] ... # [/snippet:doc-name]
        |
        v
skill (../../skills/*/SKILL.md, mirrored read-only via .agents/skills + .claude/skills)
    references snippets via `> **Source:**` directives
```

Skills are the consumer of these snippets -- they are what agents read to learn
how to extend ovannotators. Keep them in sync: when you add or change a snippet,
update every skill that references it.

## Layout

```
tests/docs/
├── AGENTS.md                       # This file
├── CLAUDE.md                       # Redirects to AGENTS.md
└── python/
    ├── test_custom_writer.py       # Snippets for skills/new-writer
    ├── test_custom_backend.py      # Snippets for skills/new-backend
    └── test_config_serialization.py  # Config save/load snippets (both skills)
```

ovannotators is a Python-only library, so (unlike the OVRTX docs-tests this is
modeled on) there is no `c/` or `usd/` subtree, and these tests run inside the
repo's normal pytest suite (`testpaths = ["tests"]`) rather than a separate
sub-project. They are deliberately CPU-only so they run under the default
`-m 'not gpu'` selection without a renderer or GPU.

## Rules

### Snippets are the source of truth
1. **Never write inline code blocks in a `SKILL.md` for API usage.** Substantial
   examples must live in a test here with snippet markers, then be referenced
   from the skill via `> **Source:**`.
2. **Small fragments are OK inline.** A one-line `pyproject.toml` entry-point or
   an import line can stay inline in a skill -- don't over-engineer.

### Naming conventions
- Snippet names are kebab-case, prefixed with `doc-`, e.g. `doc-writer-class`.
- Names must be unique across the entire `tests/docs/` tree.

### Marker format
```python
# [snippet:doc-writer-class]
class ManifestWriter:
    ...
# [/snippet:doc-writer-class]
```

### Reference format in `SKILL.md`
```markdown
> **Source:** `tests/docs/python/test_custom_writer.py` snippet `doc-writer-class`
```

## Running Tests

```bash
# Just the doc-snippet tests
uv run pytest tests/docs/python -q

# The whole suite (these tests are included automatically)
uv run pytest -q
```

`tests/test_skill_snippets.py` additionally verifies that every `> **Source:**`
reference under `skills/` resolves to a real snippet marker here, and that the
`.agents/skills` and `.claude/skills` mirrors point at `skills/`.

## Current Snippet Inventory

**Agents MUST keep this table up to date.** When you add, rename, move, or
remove a snippet, update this table in the same change.

| Snippet | Source file | Used in skill |
|---------|-------------|---------------|
| `doc-writer-config` | `python/test_custom_writer.py` | `new-writer` |
| `doc-writer-class` | `python/test_custom_writer.py` | `new-writer` |
| `doc-writer-register` | `python/test_custom_writer.py` | `new-writer` |
| `doc-writer-test` | `python/test_custom_writer.py` | `new-writer` |
| `doc-backend-config` | `python/test_custom_backend.py` | `new-backend` |
| `doc-backend-class` | `python/test_custom_backend.py` | `new-backend` |
| `doc-backend-register` | `python/test_custom_backend.py` | `new-backend` |
| `doc-backend-test` | `python/test_custom_backend.py` | `new-backend` |
| `doc-config-save-load` | `python/test_config_serialization.py` | `new-writer`, `new-backend` |
| `doc-config-registry` | `python/test_config_serialization.py` | `new-writer`, `new-backend` |
