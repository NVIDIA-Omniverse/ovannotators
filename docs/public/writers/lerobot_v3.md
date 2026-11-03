<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# LeRobot v3: Coming Soon

The `lerobot_v3` profile writer records robot-learning episodes through the
official LeRobot v3 dataset API. It and the `lerobot_lance` profile writer,
which builds on the same `lerobot` stack, are not part of ovannotators 0.2.0.

Both writers depend on `lerobot` 0.6.1, which requires a Pillow version earlier
than 13. ovannotators 0.2.0 requires Pillow 13 (`>=13.0.0,<14`), which
includes later 13.x releases; 13.0.0 is the first release that fixes several
high-severity CVEs. The writers return in a later release
after LeRobot supports Pillow 13; see
[huggingface/lerobot#5024](https://github.com/huggingface/lerobot/pull/5024).

Until then, the legacy [lerobot](lerobot.md) writer emits a single-episode
v2.0-style layout. It does not record robot state or action and does not claim
v3 conformance.
