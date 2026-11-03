<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# LeRobot: Frame-Based Video Export

Registry name: `lerobot`. Class: `LeRobotWriter`. Install `[lerobot,video]`;
video encoding requires an NVENC-capable NVIDIA GPU.

## Purpose and Inputs

Use for a simple single-episode RGB sequence with LeRobot-style metadata.
This writer requires only `LdrColor`. A profile for robot state/action signals, explicit
episodes, and the official v3 dataset API is [coming soon](lerobot_v3.md). This writer emits
the earlier v2.0-style layout and does not claim v3 conformance.

## Output and Data Types

Frames are buffered until `finalize()`: host frames in host memory, CUDA frames
in GPU memory on their source device (about `H*W*3` bytes per frame). For each
chunk, output includes `data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet` and
`videos/chunk-{chunk:03d}/observation.images.{camera_key}/episode_{episode_index:06d}.mp4`.
Videos are H.264 RGB sequences converted from input to `uint8 (H,W,3)`.

| Parquet column | Type | Meaning |
| --- | --- | --- |
| `index` | int64 | Counter starting at zero for this writer |
| `episode_index` | int64 | Configured episode ID |
| `frame_index` | int64 | Sequence position |
| `timestamp` | float64 | `frame_index / fps`, seconds |
| `task_index` | int64 | Zero |

`meta/info.json`, `tasks.json`, `episodes.json`, and `stats.json` describe a
single synthetic episode/task; stats are empty. `info.json` declares
`codebase_version="v2.0"`. Image features are declared in metadata and stored
as video, not as Parquet image columns.

Every `camera_keys` entry receives the same `LdrColor` image; this option does
not accept independent camera observations. Source `frame_id` does not set
indices or timestamps. Robot states/actions and source timestamps are not
serialized. Treat this as the implementation's limited frame export, and verify
compatibility with your target loader.

```{warning}
`frames_per_chunk` splits one episode across several chunk directories. When an
episode has more frames than `frames_per_chunk`, each `chunk-{chunk:03d}`
directory holds a slice of that episode's frames and uses the same
`episode_{episode_index:06d}` filename. Loaders that expect one file per episode
see the episode spread across chunks. Set `frames_per_chunk` at least as large as
the longest sequence to keep the episode in one chunk.
```

See [LeRobot dataset documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
for the v3 layout, which differs from the v2.0 metadata this
writer emits. Use it for background only; v3 output is
[coming soon](lerobot_v3.md).

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `fps` | `float` | `30` |
| `episode_index` | `int` | `0` |
| `task_description` | `str` | `''` |
| `camera_keys` | `list[str]` | `['top']` |
| `frames_per_chunk` | `int` | `1000` |

## Usage

```python
from ovannotators.writers import LeRobotWriter, LeRobotWriterConfig

writer = LeRobotWriter(LeRobotWriterConfig())
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

See {ref}`Tutorial 4 <writer-tutorial-lerobot>`.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
