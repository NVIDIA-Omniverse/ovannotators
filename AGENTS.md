<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# AGENTS.md - AI Agent Guide for ovannotators

This file gives AI coding agents the minimum context to work effectively in this
repository. Use it as a starting map, then go to `skills/` for task-level
implementation guidance.

## What This Repo Is

`ovannotators` is a standalone, receive-only Python library that turns
self-describing SDG renderer output (RGB, depth, segmentation, bounding boxes,
normals, camera params) into ML-ready datasets, outside the renderer process. It
ships built-in writers (COCO, KITTI, YOLO, BOP, LeRobot, Cosmos-Transfer, plus a
flat `BasicWriter`) and pluggable storage backends (disk, S3, HTTP storage API).

## Start Here

- `README.md` — product context, install (`uv sync`), and the minimal example.
- `skills/README.md` — the skill format and the tested-snippet convention.
- `docs/public/api.rst` — the public API reference.

## Repo Layout (High-Level)

- `source/ovannotators/` — package source
  - `writers/` — output-format writers (`_base.py` protocol, `_registry.py`)
  - `backends/` — storage destinations (`_base.py` protocol, `_registry.py`)
  - `renderer_outputs/` — extract render vars from renderer frames
- `tests/` — pytest suite; `tests/docs/python/` holds tested skill snippets
- `docs/public/` — Sphinx docs
- `skills/` — task-oriented agent skills (`*/SKILL.md`)
- `scripts/` — tooling that ships with the public repo

## Common Workflows

```bash
uv sync                 # set up the environment
uv run pytest           # default suite (no GPU; excludes -m gpu and benchmarks)
uv run pytest --cov     # with coverage
```

Python 3.10+. CI runs the default CPU suite on Linux and Windows. GPU-gated
end-to-end tests require `ovrtx` + CUDA and are marked both `e2e` and `gpu`.

## Use Skills for Task-Specific Work

When a request maps to a known ovannotators workflow, go directly to the skill:

- Add/implement/debug an output-format writer (COCO, KITTI, YOLO, custom, ...)
  -> `skills/new-writer/SKILL.md`
- Add/implement/debug a storage backend (GCS, Azure Blob, HTTP API, custom, ...)
  -> `skills/new-backend/SKILL.md`
- Create, classify, triage, or audit test coverage
  -> `skills/test-creation/SKILL.md`, `skills/test-classification/SKILL.md`,
     `skills/test-triage/SKILL.md`, `skills/test-coverage/SKILL.md`

The same skills are mirrored read-only at `.agents/skills/` and `.claude/skills/`
(symlinks to `skills/`).

## Agent Expectations

- Prefer small, targeted edits over broad refactors unless asked.
- Preserve the NVIDIA Apache-2.0 SPDX headers on new files; use Google-style
  docstrings.
- If you introduce a new repeated workflow, add/update a skill under `skills/`.

### Snippet and skill rules

These rules are mandatory. Test code is the single source of truth; skills
reference it — never the other way around.

- **Adding tests/examples:** wrap every illustrative region in
  `# [snippet:doc-name]` / `# [/snippet:doc-name]` markers under
  `tests/docs/python/`. Names are kebab-case, `doc-`-prefixed, and unique. Add a
  `> **Source:**` reference from the matching skill and update the inventory in
  `tests/docs/AGENTS.md`.
- **Adding skills:** every code example in a `SKILL.md` must come from a snippet
  marker — never write inline code blocks for API usage. Small fragments (a
  one-line entry-point or import) may stay inline. See `skills/README.md`.
- `tests/test_skill_snippets.py` enforces that references resolve and that the
  `.agents`/`.claude` mirrors point at `skills/`.

## Notes

- `ovannotators` is Apache-2.0 but does not accept external code contributions;
  see `CONTRIBUTING.md`.
