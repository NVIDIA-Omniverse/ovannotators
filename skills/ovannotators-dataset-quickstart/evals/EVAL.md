<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Evaluating ovannotators-dataset-quickstart

`evals.json` holds eight cases: four first-run requests and four negative routing cases.
Two negative cases belong to `ovannotators-dataset-capture`: a user who has
already run the quickstart, and a newcomer who names a format. Run each case with and without the skill on Codex and Claude Code in
identical environments, one attempt per case.

The evaluation environment is CPU-only. The CPU cases are graded on runnable
behavior: the script must print `Wrote 4 files to _out/dataset_quickstart`, and
the verification must report `3 images, 6 annotations`. The GPU scene case is
graded on the plan and commands, not on a render. The GPU fallback case is
executed: it passes when the agent finds that rendering is unavailable, by a
GPU check or a renderer failure, runs the CPU lane, and reports why the GPU
lane could not run.

Check negative cases from skill-load traces, not from the agent's claim that it
did not use a skill. Baseline runs receive the same public docs and package but
no skill instructions.

Reproduce the CPU lane locally from the repository root:

```bash
uv run pytest tests/docs/python/test_dataset_capture.py -q
```

The GPU lane belongs to release validation on an RTX host, outside this
dataset. Generate `BENCHMARK.md` from measured results only. NVSkills CI
completes the evaluation sections of `skill-card.md` and produces
`skill.oms.sig`.

See the [dataset contract](https://docs.nvidia.com/skills/skillevaluator/eval-datasets)
and the [live evaluation guide](https://docs.nvidia.com/skills/skillevaluator/tier3-live-evaluation).
