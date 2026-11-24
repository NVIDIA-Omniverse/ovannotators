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
`{frame_id:06d}_{gt_index:06d}.png`, where `gt_index` is the zero-based index
of the instance in that frame's `scene_gt.json` list. Multiple instances may
share a model `obj_id` and retain separate masks.

Depth PNGs are `uint16`: `stored = clip(depth_m * 1000 / depth_scale, 0, 65535)`
cast to integer. Decode as `stored * depth_scale` millimetres. Input is radial
camera distance; ensure that matches your downstream consumer's depth convention.
Mask PNGs contain 0 or 255.

`finalize()` writes three JSON objects keyed by string frame IDs, then the
scene marker `mask_layout.json`, `{"mask_suffix": "gt_index"}`, which records
that mask suffixes are ground-truth indices:

| File | Fields |
| --- | --- |
| `scene_camera.json` | `cam_K` (9 numbers), `cam_R_w2c` (9 numbers), `cam_t_w2c` (3 numbers, mm), `depth_scale` |
| `scene_gt.json` | List of integer `obj_id`, `cam_R_m2c` (9 numbers), `cam_t_m2c` (3 numbers, mm) |
| `scene_gt_info.json` | Integer pixel boxes `[x,y,w,h]`, pixel counts, numeric `visib_fract` |

Supply world-space 3D boxes and the camera in scene units, with a finite,
positive `CameraParams.meters_per_scene_unit`. Both camera and object
translations are multiplied by `meters_per_scene_unit * 1000` for millimetre
output. Depth inputs remain in metres. The writer
transforms each box into the OpenCV camera frame (+Z forward, +Y down), the frame
`cam_K` projects from, and writes `cam_R_m2c` and `cam_t_m2c` there, with
`cam_t_m2c` the 3D box centre in millimetres. `cam_R_m2c` is a rotation matrix when the box transform has no shear or
reflection. Object scale is removed by normalizing each axis, so scale stays in
the model coordinates, not the pose. A box with a zero-length axis has no
rotation, so the writer raises `ValueError` naming the instance. `cam_R_w2c` and `cam_t_w2c` use the
same frame. The box centre stands in for the model origin, so use model
coordinates that match it. `object_id_map` maps labels to model IDs;
unmapped labels fall back to instance IDs. Provide corresponding object models
separately.

Datasets written with model IDs in mask filenames need migration before export
through `BOPFrameAdapter`: rename each mask suffix to its entry's zero-based
ground-truth index, using temporary names to avoid rename collisions. If several
instances shared one model ID, regenerate the frame's masks from the source;
an overwritten mask cannot be recovered from the dataset. Then add
`mask_layout.json` with the content above. `BOPFrameAdapter` refuses a scene
that has masks but no marker, because a model ID can also be a valid index and
the masks would otherwise be paired with the wrong objects.

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
