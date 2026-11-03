<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# ovannotators Agent Skills

Use this catalog to select the task-specific workflow before changing the
repository. Each skill is self-contained and follows the
[Agent Skills specification](https://agentskills.io/specification).

## Complete Catalog

| Skill | Use it for |
| --- | --- |
| [`gpu-verification`](gpu-verification/SKILL.md) | Run renderer, CUDA, NVENC, and GPU-gated validation that CPU CI cannot cover. |
| [`new-backend`](new-backend/SKILL.md) | Add, implement, or debug a storage backend and its registration. |
| [`new-writer`](new-writer/SKILL.md) | Add, implement, or debug a dataset-format writer and its registration. |
| [`test-classification`](test-classification/SKILL.md) | Choose pytest taxonomy markers and the correct CI or GPU lane. |
| [`test-coverage`](test-coverage/SKILL.md) | Run coverage checks, interpret reports, and investigate changed-file gaps. |
| [`test-creation`](test-creation/SKILL.md) | Create or update meaningful tests using repository conventions. |
| [`test-triage`](test-triage/SKILL.md) | Diagnose test or CI failures and choose the appropriate response. |

`skills/` is the single source of truth. `.agents/skills` and `.claude/skills`
are read-only aliases for agent runtimes; never edit the aliases directly.

For connection setup, authentication, reads, writes, and route diagnosis in the
underlying storage client, see the [ovstorage skills catalog](https://github.com/NVIDIA-Omniverse/ovstorage/tree/main/skills).

## Skill Structure

Each kebab-case subdirectory contains one `SKILL.md`. Its YAML frontmatter
names the skill, describes activation conditions, and may declare metadata and
tools. The body defines inputs, prerequisites, instructions, validation,
limitations, troubleshooting, and references as needed by that workflow.

## Tested Code References

Skills reference live, tested code instead of copying substantial API examples.
Test code is the source of truth.

Put illustrative regions under `tests/docs/python/` between matching markers:

```text
# [snippet:doc-example-name]
illustrative code
# [/snippet:doc-example-name]
```

Names must be unique, kebab-case, and prefixed with `doc-`. A skill points to a
region with this directive:

```text
> **Source:** `tests/docs/python/test_example.py` snippet `doc-example-name`
```

Small fragments such as a one-line import or entry-point declaration may stay
inline. `tests/test_skill_snippets.py` verifies every source reference and both
runtime aliases.

## Add a Skill

1. Write and test the illustrative snippet under `tests/docs/python/`; add its
   markers and update `tests/docs/AGENTS.md`.
2. Create `skills/<skill-name>/SKILL.md` with a frontmatter `name` that matches
   the directory.
3. Describe activation conditions, required context, implementation steps,
   output, validation, limitations, and troubleshooting.
4. Reference tested snippets instead of duplicating API examples.
5. Add the skill to the complete catalog above and run
   `uv run pytest tests/test_skill_snippets.py`.

## Update a Skill

When an API change affects a skill, update the tested snippet and the skill in
the same change. Run the focused snippet tests before the default suite.
