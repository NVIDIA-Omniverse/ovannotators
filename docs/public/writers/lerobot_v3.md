<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# LeRobot v3: Timed Robot-Learning Episodes

The `lerobot_v3` profile (`LeRobotV3Writer` in `ovannotators.profile_writers`)
writes robot-learning episodes in the
[LeRobot v3.0 dataset format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3),
which `lerobot==0.7.0` loads. It requires the `[lerobot,video]` extras and an
NVIDIA GPU with NVENC to encode camera video.

## Purpose and Inputs

Use this writer for episodes that pair camera video with robot state and action
vectors. Each episode must declare these streams:

- One or more cameras: required `rgb` streams of `(H, W, 3)` uint8 images with
  a nominal rate that divides `fps` evenly. Camera stream IDs must not contain
  `image`.
- One `state` stream and one `action` stream, each matching its
  `LeRobotV3FeatureSpec`: a one-dimensional `(N,)` vector of the declared data
  type, with a name for each element.

Each sample must provide:

- The state vector in `observations` and the action vector in
  `policy_signals`.
- Every camera on its own frames only. A camera at a lower rate than `fps`
  supplies an image every `fps / rate` frames, starting with the first frame.
  On the other frames, omit it and set
  `missing_data[<camera>] = "not_sampled_at_source_rate"`.
- A timestamp that increases strictly. Sample `k` must be within
  `timestamp_tolerance_ns` of the episode's `source_origin_ns` plus `k / fps`
  seconds.

## Output and Data Types

| Path | Contents |
| --- | --- |
| `data/chunk-NNN/file-NNN.parquet` | One row per frame |
| `videos/observation.images.<camera>/chunk-NNN/file-NNN.mp4` | Camera video, several episodes per file |
| `meta/` | `info.json`, `stats.json`, `tasks.parquet`, and `episodes/chunk-NNN/file-NNN.parquet` |
| `_ovannotators/` | Checksum manifest, conformance record, and the `_SUCCESS` completion marker |

- State and action columns are named after their stream IDs. These IDs must not
  contain `/` or reuse a generated column name such as `index`, `task`, or
  `timestamp`.
- A camera stream ID becomes `observation.images.<name>`. A leading `camera.` is
  removed and `/` becomes `.`, so `camera.front` becomes
  `observation.images.front`.
- `timestamp` is `frame_index / fps`. The source time is kept in the int64
  columns `source.timestamp_ns`, `source.frame_id`, and
  `source.camera.<name>.timestamp_ns`.
- A camera slower than `fps` repeats its latest image on the frames between its
  samples.

## Finalizing and Validating

`finalize()` builds the dataset in `staging_dir` and then publishes it. The
dataset is complete only when `_ovannotators/_SUCCESS` exists.

- The backend must support `write_if_absent`. The disk and S3 backends do, as do
  ovstorage routes that support fail-if-exists writes.
- Publish each dataset to its own output directory. A retry reuses files that
  already match, and any existing file with different contents stops
  publication.
- With `DiskBackend`, keep `staging_dir` and the output directory separate.
- If `end_episode()` fails, the episode stays active and can be retried. If
  `finalize()` fails, the staging directory is kept and `finalize()` can be
  called again.

To check a published dataset:

- `writer.validate(backend)` checks the `_SUCCESS` marker and the SHA-256
  checksum of every file.
- `validate_lerobot_v3_canonical(root, repo_id)` opens the dataset with the
  official `LeRobotDataset` loader. Run it in a separate Python 3.12 or newer
  environment with `lerobot[dataset]==0.7.0` installed. The loader source this
  writer targets is in the
  [LeRobot v0.7.0 datasets package](https://github.com/huggingface/lerobot/tree/v0.7.0/src/lerobot/datasets).

## Configuration

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fps` | `int` | Required | Dataset frame rate |
| `camera_stream_ids` | `tuple[str, ...]` | Required | Camera streams to write as video |
| `state` | `LeRobotV3FeatureSpec` | Required | State vector declaration |
| `action` | `LeRobotV3FeatureSpec` | Required | Action vector declaration |
| `repo_id` | `str \| None` | `None` | `namespace/dataset` name to pass to `validate_lerobot_v3_canonical`; not written to the dataset |
| `chunks_size` | `int` | `1000` | Maximum files per chunk directory |
| `video_files_size_in_mb` | `int` | `200` | Target size of each MP4 file |
| `data_files_size_in_mb` | `int` | `100` | Target size of each Parquet file |
| `video_codec` | `'h264' \| 'hevc' \| 'av1'` | `'h264'` | AV1 needs an Ada Lovelace or newer GPU |
| `video_lossless` | `bool` | `False` | Lossless encoding; HEVC only |
| `video_preset` | `str \| None` | `None` | NVENC preset `"P1"` to `"P7"` |
| `video_gop` | `int \| None` | `2` | Keyframe interval in frames; `None` uses the NVENC default |
| `timestamp_tolerance_ns` | `int` | `100000` | Allowed timestamp error, 0 to 100,000 ns |
| `staging_dir` | `str \| Path \| None` | `None` | Local build directory that must not exist yet; defaults to a new directory under the system temporary directory |
| `keep_staging` | `bool` | `False` | Keep `staging_dir` after a successful `finalize()` |

If you set `staging_dir`, put it in a directory that only you can write to, or
in a sticky directory such as the system temporary directory.

## Usage

This example configures one camera and a one-joint robot. Choose feature names,
units, and representation to match your producer.

```python
from ovannotators.profile_writers import (
    LeRobotV3FeatureSpec, LeRobotV3Writer, LeRobotV3WriterConfig,
)

writer = LeRobotV3Writer(LeRobotV3WriterConfig(
    fps=30,
    repo_id="my-team/arm-episodes",
    camera_stream_ids=("camera.front",),
    state=LeRobotV3FeatureSpec(
        stream_id="observation.state",
        feature_names=("joint.position",),
        units=("rad",),
        representation="absolute",
        shape=(1,),
    ),
    action=LeRobotV3FeatureSpec(
        stream_id="action",
        feature_names=("joint.target",),
        units=("rad",),
        representation="absolute",
        shape=(1,),
    ),
))
```

Then follow the {ref}`episode workflow <writer-episode-workflow>`: pass
`EpisodeEnvelope` and `SampleEnvelope` objects, not the frame writer's
`render_vars` dictionary. The [shared writer workflow](index.md) describes the
input conventions and lifecycle.

### State and Action Feature Declarations

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `stream_id` | `str` | Required | Stream ID and output column name |
| `feature_names` | `tuple[str, ...]` | Required | One name per vector element |
| `units` | `tuple[str, ...]` | Required | One unit for all elements, or one per element |
| `representation` | `str` | Required | Meaning of the values, such as `"absolute"` |
| `shape` | `tuple[int, ...]` | Required | `(N,)`, one dimension |
| `dtype` | `str` | `'float32'` | NumPy boolean, integer, or floating-point type of at most 8 bytes |
| `coordinate_frame` | `str \| None` | `None` | Frame the values are expressed in, if any |
