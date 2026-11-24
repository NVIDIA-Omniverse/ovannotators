<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# DOPE: Object Pose Keypoints

This writer is registered as `dope` using the `DOPEWriter` class. It is
included with the core installation and supports CPU processing. It writes the
layout that Isaac Sim's `PoseWriter(format="dope")` produces, so datasets work
with existing DOPE training workflows.

## Purpose and Inputs

Use for 6D object pose estimation with
[DOPE](https://github.com/NVlabs/Deep_Object_Pose), which regresses the
projected corners of each object's 3D cuboid. This writer requires `LdrColor`,
`SemanticBoundingBox3D`, and `CameraParams`. `SemanticIdMap` supplies class
names and `StableIdMap` supplies prim paths.

Visibility needs `OcclusionSD`, `InstanceIdTokenMapSD`, `InstanceMap`, and
`StableIdSemanticIdMap`. Without them every object has `visibility` 1.0, so add
those render vars to the render product when you filter by visibility.

## Output and Data Types

Each frame writes `{frame_id:06d}.png` (or `.jpg`) and `{frame_id:06d}.json` at
the backend root. The JSON has `camera_data`, `keypoint_order`, and `objects`.

| Field | Content |
| --- | --- |
| `camera_data` | `intrinsics` (`fx`, `fy`, `cx`, `cy`), `camera_view_matrix` and `camera_projection_matrix` (4x4, row-vector convention, rounded to five places) |
| `keypoint_order` | `LUF, RUF, RDF, LDF, LUB, RUB, RDB, LDB, Center`, where L/R, D/U and B/F are the -/+ ends of the box's local X, Y and Z |
| `class` | Label from `SemanticIdMap`, or the semantic ID when absent |
| `prim_path` | USD prim path from `StableIdMap`, or `null` |
| `visibility` | `1 - occlusion_ratio`, three decimals |
| `location` | Object origin in the camera frame, scene units |
| `quaternion_xyzw` | Object-to-camera rotation, unit length, `w >= 0`; object scale is removed |
| `projected_cuboid` | Nine integer pixel keypoints in `keypoint_order` |
| `truncation_ratio` | Fraction of the keypoints' bounding box outside the image, 0 to 1 |

The camera frame follows USD, as in Isaac Sim: the camera looks down -Z with +Y
up. `location` is the object's origin, which differs from the box centre when
the object's local bounds are not centred on its origin; the `Center` keypoint
is the box centre.

An object is skipped when `visibility` is at or below `visibility_threshold`,
when its scale is zero, or when any keypoint is behind the camera plane. Only
pinhole cameras are supported; other projection models raise
`ConfigurationError`. Isaac Sim's optional debug overlay images and
`use_subfolders` are not implemented.

Visibility comes from `OcclusionSD`, which is keyed by renderer instance ID. The
writer maps those IDs to the stable IDs on the semantic boxes through
`InstanceIdTokenMapSD`, `InstanceMap`, and `StableIdSemanticIdMap`. When several
renderer instances share a stable ID their ratios are averaged. An object with
no valid ratio keeps `visibility` 1.0.

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `visibility_threshold` | `float` | `0.0` |
| `skip_empty_frames` | `bool` | `True` |
| `frame_padding` | `int` | `6` |
| `image_format` | `"png" \| "jpg"` | `"png"` |

## Usage

```python
from ovannotators.writers import DOPEWriter, DOPEWriterConfig

writer = DOPEWriter(DOPEWriterConfig(visibility_threshold=0.2))
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
