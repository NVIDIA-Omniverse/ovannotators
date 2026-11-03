<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Basic: direct per-frame output

Registry name: `basic`. Class: `BasicWriter`. Core install; add `[video]` and
an NVENC-capable NVIDIA GPU only when enabling video.

## Purpose and inputs

Use this format to inspect frames or build a custom reader for renderer data.
There is no external dataset schema. No render variable is mandatory; `render_vars`
selects which supplied variables to write, or `None` writes all present variables.

## Output and data types

Files live in `{frame_id:04d}/` by default. Names use lowercase snake case.

| Input | Default output | Meaning |
| --- | --- | --- |
| `LdrColor`, `DiffuseAlbedoSD` | `ldr_color.png`, `diffuse_albedo.png` | Encoded color images |
| `HdrColor`, normals, positions, motion | `.exr` | Floating-point image channels |
| Depth variables | `.exr` (or `.npy`) | Raw depth values in producer units |
| Instance/stable/semantic segmentation | `.npy` | Original integer ID arrays |
| Boxes, camera parameters, ID maps | `.json` | Decoded fields, numbers, strings, and lists |
| `PointCloud` | `point_cloud.npz` | One named member per tensor |
| Unrecognized variables | `.npy` | Original NumPy-compatible array |

Optional `*_colorized.png` files visualize depth, normals, and segmentation.
These previews do not preserve quantitative values. With `video=True`,
`finalize()` closes H.264 streams under `videos/`; `video_vars` selects RGB,
albedo, or supported colorized streams. Raw frame files are still written.

Read `.npy` and `.npz` using NumPy, JSON using `json`, images using Pillow,
and EXR with the package EXR reader. Empty image buffers are skipped;
empty annotation containers are serialized. This writer does not convert stage
units or impose train/validation splits.

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `render_vars` | `list[str] \| None` | `None` |
| `depth_format` | `Literal['npy', 'exr']` | `'exr'` |
| `image_format` | `Literal['png', 'jpg']` | `'png'` |
| `frame_id_digits` | `int` | `4` |
| `colorize` | `bool` | `True` |
| `depth_near_bright` | `bool` | `True` |
| `video` | `bool` | `False` |
| `video_fps` | `float` | `24` |
| `video_vars` | `list[str] \| None` | `None` |
| `video_dir` | `str` | `'videos'` |
| `segmentation_palette_accessibility` | `Literal['none', 'protanopia', 'deuteranopia', 'tritanopia']` | `'none'` |
| `semantic_color_key` | `Literal['semantic_id', 'semantic_label']` | `'semantic_id'` |
| `segmentation_palette_path` | `str \| Path \| None` | `None` |
| `exr_compression` | `str \| None` | `None` |

`exr_compression=None` uses lossless ZIP; see the
[API reference](../api.rst) for the accepted scheme names. Lossy schemes can
change float values. `semantic_color_key="semantic_label"` gives IDs with the
same label one color. Set `segmentation_palette_path` to persist segmentation
colors across writer sessions; do not share one file between concurrent writers.

## Usage

```python
from ovannotators.writers import BasicWriter, BasicWriterConfig

writer = BasicWriter(BasicWriterConfig())
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

For a complete CPU-only example, see {ref}`Tutorial 8 <writer-tutorial-numpy>`.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
