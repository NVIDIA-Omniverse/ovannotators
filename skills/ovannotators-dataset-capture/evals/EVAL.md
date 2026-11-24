<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Evaluating ovannotators-dataset-capture

`evals.json` holds eight cases: five capture requests and three negative routing
cases. Run each case with and without the skill on Codex and Claude Code in
identical environments, one attempt per case.

The evaluation environment is CPU-only. The KITTI, own-arrays and WebDataset
cases are graded on runnable behavior against `pip install ovannotators`,
using the synthetic stand-in frames their prompts allow. The
S3 and ovrtx case is graded on the wiring, because the sandbox has no bucket
or renderer. The LeRobot v3 case is graded on the writer setup and episode loop it shows;
installing the `lerobot` and `video` extras, and having an NVIDIA GPU for
NVENC encoding, are not required.

Grade discovery by whether the answer calls `describe_requirements` and
creates the writer from `requirements.config`, not by exact wording. Check
negative cases from skill-load traces, not from the agent's claim that it did
not use a skill. Baseline runs receive the same public docs and package but no
skill instructions.

Reproduce the referenced snippets locally from the repository root:

```bash
uv run pytest tests/docs/python/test_dataset_capture.py tests/docs/python/test_requirements.py tests/docs/python/test_webdataset.py -q
```

Generate `BENCHMARK.md` from measured results only. NVSkills CI completes the
evaluation sections of `skill-card.md` and produces `skill.oms.sig`.

See the [dataset contract](https://docs.nvidia.com/skills/skillevaluator/eval-datasets)
and the [live evaluation guide](https://docs.nvidia.com/skills/skillevaluator/tier3-live-evaluation).
