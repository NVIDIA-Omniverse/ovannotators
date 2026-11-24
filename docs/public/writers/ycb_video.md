<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# YCB-Video: Object Pose in Video Frames

This writer is registered as `ycb_video` using the `YCBVideoWriter` class. It is
included with the core installation and supports CPU processing. It writes the
layout of Isaac Sim's `YCBVideoWriter`, so datasets work with existing
YCB-Video training workflows such as PoseCNN and DenseFusion.

## Purpose and Inputs

Use for 6D object pose estimation on the
[YCB-Video](https://rse-lab.cs.washington.edu/projects/posecnn/) layout. This
writer requires `LdrColor`, `DistanceToImagePlaneSD`, `SemanticSegmentation`,
`SemanticIdMap`, `SemanticBoundingBox2DTight`, `SemanticBoundingBox3D`, and
`CameraParams`.

The configuration needs `class_name_to_index_map`, which maps each semantic
label to a YCB-Video class index from 1 to 255. Labels not in the map are
background: they get label 0 and no pose or box.

## Output and Data Types

```text
data/
├── YCB_Video/
│   ├── data/{video_id}/
│   │   ├── {frame_id:06d}-color.png
│   │   ├── {frame_id:06d}-depth.png
│   │   ├── {frame_id:06d}-label.png
│   │   ├── {frame_id:06d}-box.txt
│   │   └── {frame_id:06d}-meta.mat
│   └── train.txt
└── models/{model_name}/points.xyz
```

| File | Content |
| --- | --- |
| `-color.png` | RGB image |
| `-depth.png` | `uint16` PNG: `round(depth_m * factor_depth)`, clipped to 65535; 0 where there is no depth |
| `-label.png` | `uint8` PNG: class index per pixel, 0 for background |
| `-box.txt` | One `class_name x1 y1 x2 y2` line per mapped object, integer pixels, top-left origin |
| `-meta.mat` | MATLAB v5 file, below |
| `train.txt` | One `{video_id}/{frame_id:06d}` line per written frame, written by `finalize()` |

`-meta.mat` holds `cls_indexes` (`1 x N` `uint8`), `factor_depth` (`uint16`),
`intrinsic_matrix` (`3 x 3`), `poses` (`3 x 4 x N`), and `center` (`N x 2`).
`poses` are object-to-camera `[R | t]` in the OpenCV camera frame (+Z forward, +Y
down) with `t` in metres. The writer converts scene units to metres with the
camera's `meters_per_scene_unit`. `center` is each object's origin projected to
pixels. The file is written without scipy and `scipy.io.loadmat` reads it.

An object is skipped when its label is not in the map, its scale is zero, or its
origin is behind the camera. A frame with no mapped object is not written and is
not listed in `train.txt`.

### Differences from Isaac Sim

- `-depth.png` is physical 16-bit depth consistent with `factor_depth`. Isaac
  Sim writes an 8-bit normalised inverse-depth image for visualisation.
- `poses` are in the camera frame in metres, as YCB-Video consumers read them.
- Isaac Sim's per-output switches (`rgb`, `pose`, and so on) are not
  implemented; every output is written.
- `train.txt` lists the frames actually written rather than `num_frames`.

### Model points

Isaac Sim exports model vertices from USD with a separate helper. ovannotators
does not read USD, so you supply the points. Call
`write_model_points(model_name, points, backend)` with `(N, 3)` vertices in
metres, in the same frame as the object origin the poses refer to. The file is
`data/models/{model_name}/points.xyz`; `model_name` is the YCB-Video model name,
for example `002_master_chef_can`, and need not match the scene's name.

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `class_name_to_index_map` | `dict[str, int]` | required |
| `factor_depth` | `int` | `10000` |
| `video_id` | `str` | `"0000"` |

The default `factor_depth` limits depth to 6.5535 m, and deeper pixels are
clipped to 65535. The writer logs a warning the first time it clips. Use a
smaller value for a deeper scene.

## Usage

```python
from ovannotators.writers import YCBVideoWriter, YCBVideoWriterConfig

writer = YCBVideoWriter(
    YCBVideoWriterConfig(class_name_to_index_map={"002_master_chef_can": 1})
)
writer.write_frame(0, render_vars, backend)
writer.write_model_points("002_master_chef_can", points, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above, `points` is an `(N, 3)`
array, and `backend` is a configured storage backend. Use a fresh writer and
output root for each dataset.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
