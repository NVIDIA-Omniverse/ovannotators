<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# ovannotators Agent Skills

Use this catalog to select the task-specific workflow before changing the
repository. Each skill is self-contained and follows the
[Agent Skills specification](https://agentskills.io/specification).

## Complete Catalog

Start with `ovannotators-dataset-quickstart` if you have not produced a dataset yet.

| Skill | Use it for |
| --- | --- |
| [`ovannotators-dataset-quickstart`](ovannotators-dataset-quickstart/SKILL.md) | Produce and inspect a small labeled COCO dataset, on CPU by default. |
| [`ovannotators-dataset-capture`](ovannotators-dataset-capture/SKILL.md) | Choose a writer, profile, frame source, and backend, then capture and verify a dataset. |
| [`new-backend`](new-backend/SKILL.md) | Add, implement, or debug a storage backend and its registration. |
| [`new-writer`](new-writer/SKILL.md) | Add, implement, or debug a dataset-format writer and its registration. |

`skills/` is the single source of truth. `.agents/skills` and `.claude/skills`
are read-only aliases for agent runtimes; never edit the aliases directly.

For connection setup, authentication, reads, writes, and route diagnosis in the
underlying storage client, see the [ovstorage skills catalog](https://github.com/NVIDIA-Omniverse/ovstorage/tree/main/skills).

## Skill Structure

Each kebab-case subdirectory contains one `SKILL.md`. Its YAML frontmatter
names the skill, describes activation conditions, and may declare a license.
Put `author`, `version`, and `tags` under `metadata:`; validators ignore them at
the root. The body defines inputs, prerequisites, instructions, validation,
limitations, troubleshooting, and references as needed by that workflow.

Skills listed in the [NVIDIA skill catalog](https://github.com/NVIDIA/skills)
also carry its onboarding files next to `SKILL.md`:

| File | Content | Author |
| --- | --- | --- |
| `evals/evals.json` | Evaluation tasks, including cases where the skill must not load | Skill owner |
| `evals/EVAL.md`, `evals/config.yml` | How to grade the tasks; evaluator settings | Skill owner |
| `skill-card.md` | Release record: owner, license, use case, risks, output | Skill owner; NVSkills CI adds evaluation results |
| `BENCHMARK.md` | Measured with-skill and baseline results | SkillEvaluator |
| `skill.oms.sig` | Signature over the skill directory | NVSkills CI |

Change a signed skill only together with a new evaluation and signature: any
edit to a file in the directory invalidates `skill.oms.sig`.

SkillEvaluator runs each task once, in a sandbox with no GPU and no one to
answer questions. Write every prompt so that it can be completed there: give
the details the skill would otherwise ask for, and for a GPU workflow either
ask for the steps or test the fallback when the renderer cannot start.

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
