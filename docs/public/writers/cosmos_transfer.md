<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Cosmos Transfer: RGB and Visual Conditioning

This writer is registered as `cosmos_transfer` using the `CosmosTransferWriter`
class. Install the `[video]` extra. H.264 encoding requires an NVENC-capable
NVIDIA GPU.
Canny processing uses Warp on CPU or CUDA.

## Purpose and Inputs

Use this writer to export matching RGB clips and visual control signals for
Cosmos Transfer workflows. It prepares media and does not run models or training
jobs.

- `LdrColor` is always required.
- Depth conditioning requires `DistanceToCameraSD`.
- Segmentation requires the render variable selected by `segmentation_color_key`
  (`SemanticInstanceSegmentation` for the default `instance_id`). Shaded
  segmentation additionally requires `NormalSD`.
- When `canny` is in `conditioning_types`, `canny_source` determines extra inputs:
  `rgb` adds none, `normals` adds `NormalSD`, `segmentation` adds the segmentation
  variable, and `shaded_segmentation` adds both.
- Default settings produce Canny (from RGB), depth, and segmentation, requiring
  `LdrColor`, `DistanceToCameraSD`, and `SemanticInstanceSegmentation`.
- Check the configured writer's `required_render_vars` for the resolved set of
  variables.

## Output and Data Types

Each clip emits `videos/{clip_id:06d}_rgb.mp4`, selected
`conditioning/{clip_id:06d}_{type}.mp4`, and `thumbnails/{clip_id:06d}.jpg`.
All video inputs become `uint8 (H,W,3)` frames at the first RGB frame's
resolution and configured constant FPS.

| Conditioning | Representation |
| --- | --- |
| `canny` | Black background and white edges, from RGB, normals, segmentation, or shaded segmentation |
| `depth` | 8-bit inverse-depth grayscale repeated in RGB; near is bright, with clip-global percentile bounds and dithering |
| `segmentation` | Deterministic per-instance RGB palette; background black |
| `shaded_segmentation` | Palette shaded with normals and configured light direction |

The lossy H.264 conditioning videos do not preserve metric depth or integer
segmentation IDs. Use [Basic](basic.md) for raw arrays.

`manifest.json` has `version="cosmos-transfer-2.5"` and a `clips` list.
Each record has integer `clip_id` and `num_frames`, numeric `fps`,
`[width,height]` `resolution`, relative `rgb_video` and `thumbnail` paths, a
`conditioning` map of type to path, a `conditioning_types` list, and a
`video_encoding` object with `codec`, `lossless`, and `pixel_format`.

Frames are buffered until `cut(backend)` or `finalize(backend)`; CUDA RGB frames
stay in GPU memory on their source device (about `H*W*3` bytes per frame), so
call `cut()` to bound long sequences. `cut()` returns
the emitted clip ID, updates the manifest, and advances the index; empty cuts
return `None`. Use one writer per camera/destination. FPS describes constant
cadence; source frame IDs are not retained as timing. For explicitly timed
multiview episodes, use [cosmos_multiview](cosmos_multiview.md).

Reference: [Cosmos Transfer 2.5](https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html).

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `fps` | `float` | `24` |
| `clip_index` | `int` | `0` |
| `conditioning_types` | `list[str]` | `['canny', 'depth', 'segmentation']` |
| `canny_source` | `Literal['rgb', 'normals', 'segmentation', 'shaded_segmentation']` | `'rgb'` |
| `canny_sigma` | `float` | `1.0` |
| `canny_low_threshold` | `float` | `0.1` |
| `canny_high_threshold` | `float` | `0.2` |
| `shaded_seg_light_dir` | `tuple[float, float, float]` | `(0.3, 0.5, 0.81)` |
| `segmentation_palette_accessibility` | `Literal['none', 'protanopia', 'deuteranopia', 'tritanopia']` | `'none'` |
| `segmentation_color_key` | `Literal['instance_id', 'stable_id', 'semantic_id', 'semantic_label']` | `'instance_id'` |
| `segmentation_palette_path` | `str \| Path \| None` | `None` |

`segmentation_color_key` picks the identity that receives a color and the render
variable it reads: `SemanticInstanceSegmentation` for `instance_id`,
`StableIdSegmentation` for `stable_id`, and `SemanticSegmentation` for the
semantic modes (`semantic_label` also uses `SemanticIdMap`). Colors stay fixed
for the writer's lifetime, including across `cut()`. Set
`segmentation_palette_path` to reuse exact colors in later sessions; the file is
bound to one accessibility mode and one color key, and concurrent writers must
not share it.

## Usage

```python
from ovannotators.writers import CosmosTransferWriter, CosmosTransferWriterConfig

writer = CosmosTransferWriter(CosmosTransferWriterConfig())
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

See {ref}`Tutorial 5 <writer-tutorial-cosmos>` for clip capture and media previews.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
