<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# KITTI: Camera Object Detection

This writer is registered as `kitti` using the `KITTIWriter` class. It is
included with the core installation and supports CPU processing.

## Purpose and Inputs

Use for camera-based 2D/3D object detection. This requires `LdrColor`,
`SemanticBoundingBox2DTight`, `SemanticBoundingBox3D`, and `CameraParams`.
`write_depth=True` also requires `DistanceToCameraSD`.

## Output and Data Types

Each frame writes `image_2/{frame_id:06d}.png`, `label_2/{frame_id:06d}.txt`,
and `calib/{frame_id:06d}.txt`. Label rows are whitespace-separated:

```text
type truncated occluded alpha left top right bottom height width length x y z rotation_y
```

Field details:

- **Labels and Categories:** `type` is a sanitized label string. `occluded` is
  an integer status; other fields are numeric. Truncation and occlusion are
  both zero placeholders.
- **2D Bounding Boxes:** Box values use pixel coordinates.
- **3D Bounding Boxes and Dimensions:** Dimensions and locations use the
  supplied camera-space length units (supply metres for KITTI). Dimensions
  equal twice the 3D box half-extents. Location refers to bottom-centre with
  camera Y down.
- **Orientation and Angles:** Angles are in radians. `alpha` represents the
  observation angle, and `rotation_y` represents yaw.
- **Missing Data:** Missing matched 3D boxes produce sentinel values.

Calibration repeats the intrinsic projection in `P0` through `P3`, uses identity
rectification, and writes identity/zero transforms for LiDAR and IMU. It does
not represent a measured stereo/LiDAR rig. No point clouds are generated.

Optional `depth/{frame_id:06d}.png` stores `uint16` radial distance:
`stored = clip(depth_m * depth_scale, 0, 65535)` cast to integer (truncated).
Decode as `stored / depth_scale` metres; values beyond the range are clipped.
The default scale is 256. `classes` filters original labels before sanitization.
All files are written per frame; `finalize()` adds no aggregate metadata.

Reference: [KITTI object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php).

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `classes` | `list[str] \| None` | `None` |
| `write_depth` | `bool` | `False` |
| `depth_scale` | `float` | `256.0` |

## Usage

```python
from ovannotators.writers import KITTIWriter, KITTIWriterConfig

writer = KITTIWriter(KITTIWriterConfig())
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
