<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# BOP: Object Pose Data

This writer is registered as `bop` using the `BOPWriter` class. It is
included with the core installation and supports CPU processing.

## Purpose and Inputs

Use for object pose estimation datasets. This writer requires `LdrColor`,
`DistanceToCameraSD`, `SemanticBoundingBox2DTight`, `SemanticBoundingBox3D`,
`SemanticInstanceSegmentation`, and `CameraParams`.

## Output and Data Types

A scene lives under `{scene_id:06d}/` with `rgb/`, `depth/`, `mask/`, and
`mask_visib/`. RGB/depth filenames have six-digit frame IDs. Mask filenames are
`{frame_id:06d}_{obj_id:06d}.png` in this implementation.

Depth PNGs are `uint16`: `stored = clip(depth_m * 1000 / depth_scale, 0, 65535)`
cast to integer. Decode as `stored * depth_scale` millimetres. Input is radial
camera distance; ensure that matches your downstream consumer's depth convention.
Mask PNGs contain 0 or 255.

`finalize()` writes three JSON objects keyed by string frame IDs:

| File | Fields |
| --- | --- |
| `scene_camera.json` | `cam_K` (9 numbers), `cam_R_w2c` (9 numbers), `cam_t_w2c` (3 numbers, mm), `depth_scale` |
| `scene_gt.json` | List of integer `obj_id`, `cam_R_m2c` (9 numbers), `cam_t_m2c` (3 numbers, mm) |
| `scene_gt_info.json` | Integer pixel boxes `[x,y,w,h]`, pixel counts, numeric `visib_fract` |

Supply model-to-camera box rotations and centres in metres; the writer copies
rotation and multiplies centre by 1000. `object_id_map` maps labels to model IDs;
unmapped labels fall back to instance IDs. Provide corresponding object models
separately. Masks use object IDs rather than ground-truth list indices, and
repeated model IDs in one frame can collide.

Full and visible masks are identical; visibility metadata is derived from those
same masks, so it does not measure hidden object extent. Check these conventions
against your BOP loader before evaluation.

References: [BOP dataset format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)
and [BOP datasets](https://bop.felk.cvut.cz/datasets/).

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `scene_id` | `int` | `0` |
| `depth_scale` | `float` | `1.0` |
| `object_id_map` | `dict[str, int] \| None` | `None` |

## Usage

```python
from ovannotators.writers import BOPWriter, BOPWriterConfig

writer = BOPWriter(BOPWriterConfig())
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
