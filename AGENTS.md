<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# AGENTS.md - AI Agent Guide for ovannotators

This file provides the minimum repository context for coding agents. Follow the
reading order below before editing, then use the complete skill catalog for the
task at hand.

## Read in This Order

1. [`README.md`](README.md) — product purpose, requirements, and first run.
2. [`examples/README.md`](examples/README.md) — runnable tasks and observable
   results.
3. [`skills/README.md`](skills/README.md) — complete task-to-skill catalog and
   skill authoring rules.

Canonical entry pages:

- [Documentation map](docs/README.md)
- [Python API reference](docs/public/api.rst)
- [Troubleshooting](docs/public/troubleshooting.rst)
- [Contributing policy](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Release history](CHANGELOG.md)

## What This Repo Owns

`ovannotators` is a receive-only Python library. It converts self-describing
renderer output into ML-ready datasets and owns:

- renderer-frame extraction into canonical render-variable mappings;
- dataset writers and writer registration;
- storage backends and backend registration; and
- serialization, output layout, and writer-side validation.

Companion libraries own adjacent concerns:

- [`ovrtx`](https://github.com/NVIDIA-Omniverse/ovrtx) owns rendering and the
  self-describing frame payload. Do not move renderer lifecycle or GPU readback
  behavior into this library.
- `ovstage` owns USD stage creation, population, and time updates. An attached
  renderer borrows the stage; `ovannotators` does not manage it.
- Scene randomization is out of scope. Examples drive `ovrtx` and `ovstage`
  directly and do not depend on a scene-randomization frontend.

## Scope and Dependency Admission

Preserve a small, purpose-fit dependency surface. A new third-party dependency
must support a capability within this repository's ownership; adding a
dependency does not itself approve an adjacent feature or expand the product
boundary.

- Prefer existing dependencies and standard-library facilities when they meet
  the requirement.
- Prefer an optional extra for an integration that is not needed by every
  `ovannotators` user. An integration such as LanceDB is allowed when it fits
  the library's dataset-writing or storage boundary, but should not become a
  core dependency without a demonstrated core requirement.
- A PR adding a runtime dependency must link an accepted issue and record the
  capability rationale, alternatives considered, license/security and
  maintenance assessment, packaging choice, supported-platform impact, and
  validation plan.
- Treat a new public registry, configuration surface, output-layout change, or
  companion-library concern as a scope decision. Link the relevant accepted
  issue or state why the PR is permitted to serve as the change brief under
  the repository issue policy.

## Repository Layout

- `src/ovannotators/` — package source
  - `writers/` — output-format writers and registry
  - `backends/` — storage destinations and registry
  - `renderer_outputs/` — renderer-frame extraction
- `tests/` — default pytest suite and tested documentation snippets
- `examples/` — curated runnable examples
- `docs/public/` — Sphinx documentation
- `skills/` — canonical task-oriented agent skills
- `scripts/` — tooling that ships in the public repository

## Build and Validate

Prerequisite: install Python 3.11+ and `uv`. The default lane is CPU-only:

```bash
uv sync --extra dev-cpu
uv run pytest
uv run pytest --cov
uv run sphinx-build -W -b html docs/public _build/docs
uv build
```

The warning-free Sphinx build and successful source and wheel builds are the
public checkout's publication-validation commands.

The default suite excludes `gpu`, `flaky`, and `experimental` tests. A change to
renderer-facing code or a GPU dependency requires an NVIDIA GPU, CUDA, the real
renderer packages, and the separate GPU procedure in
[`skills/gpu-verification/SKILL.md`](skills/gpu-verification/SKILL.md).

Tests passing is not runtime validation. For a renderer-facing workflow, run
the [minimal capture](examples/README.md#minimal-renderer-capture), confirm its
success message, inspect files for all three frame groups under
`_out/basic_minimal`, and open at least one generated image or metadata file.
For CPU-only writer work, execute the relevant tutorial or a focused writer →
backend flow and inspect its emitted dataset structure.

## Working Rules

- Prefer small, targeted edits over broad refactors unless asked.
- Preserve NVIDIA Apache-2.0 SPDX headers on new files and use Google-style
  docstrings.
- Document public modules, public classes, and public functions or methods with
  Google-style docstrings. Document a private helper only when its behavior,
  invariant, side effect, unit, performance or device constraint, or failure
  mode is not clear from its name, type signature, and local implementation.
  Do not add boilerplate docstrings to tests, pytest fixtures, nested helpers,
  dunder methods, or straightforward private helpers. Ruff enforces public API
  docstring presence and basic quality; do not use a percentage-based
  docstring-coverage gate.
- Record changes that alter what a caller can do, see, or must pass under the
  top `Unreleased` version in `CHANGELOG.md`.
- Use [`skills/README.md`](skills/README.md) as the complete workflow catalog;
  do not maintain a second list here.
- The `.agents/skills` and `.claude/skills` directories are read-only aliases
  of `skills/`; edit only the canonical tree.

## Changelog Rule

From 0.2.0 onward, a change that alters public API, registry names, config
fields, CLI or install extras, output layout, defaults, supported versions,
deprecations, or user-visible bug behavior carries its own `CHANGELOG.md` entry.
Put it under the top `## [<version>] - Unreleased` heading in the appropriate
Added, Changed, Deprecated, Fixed, or Removed section. Internal refactors,
test-only changes, CI, and repository tooling do not require an entry unless a
library user can observe the change.

## Snippet and Skill Rules

Test code is the source of truth for skill examples.

- Put illustrative code under `tests/docs/python/` and wrap each region in
  `# [snippet:doc-name]` / `# [/snippet:doc-name]` markers. Names are unique,
  kebab-case, and `doc-`-prefixed. Update `tests/docs/AGENTS.md`.
- Reference those markers from `SKILL.md`; do not duplicate substantial API
  examples in skill files. A one-line import or entry point may remain inline.
- `tests/test_skill_snippets.py` verifies source references, frontmatter names,
  and the skill aliases.

## Contribution Boundary

The public repository is Apache-2.0 but does not accept external code
contributions. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
