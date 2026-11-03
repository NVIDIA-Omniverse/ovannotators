<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# YOLO: Detection and Polygon Segmentation

Registry name: `yolo`. Class: `YOLOWriter`. Core install, CPU-capable.

## Purpose and Inputs

Use for Ultralytics-style detection or instance polygon training. This requires
`LdrColor` and `SemanticBoundingBox2DTight`; `task="segment"` also requires
`SemanticInstanceSegmentation`. Supply semantic mappings for meaningful names.

## Output and Data Types

Images and labels share six-digit stems in `images/{train,val}/` and
`labels/{train,val}/`. Each text row starts with an integer, zero-based class ID.
Detection rows are `class_id cx cy width height`, with floating-point coordinates
normalized by image width/height. Segmentation rows are
`class_id x1 y1 x2 y2 ...`, also normalized. Polygon extraction retains the
largest outer contour and simplifies it; holes and disconnected smaller regions
are not retained. If no polygon is available, the writer falls back to a
box row, so inspect segment labels before training.

`finalize()` writes `data.yaml` containing split paths, `nc`, and `names`.
Without `classes`, IDs follow first-seen labels. With an explicit list, unknown
labels map to class zero; the list does not filter objects.

Validation membership is deterministic: every
`max(1, round(1 / max(val_split, 1e-6)))`-th frame goes to validation, including
frame zero. This is not a random split; correlated frames may span both splits.
Detection coordinates are computed from supplied boxes without clipping them.

References: [Ultralytics detection format](https://docs.ultralytics.com/datasets/detect/)
and [segmentation format](https://docs.ultralytics.com/datasets/segment/).

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `task` | `Literal['detect', 'segment']` | `'detect'` |
| `val_split` | `float` | `0.1` |
| `image_format` | `Literal['jpg', 'png']` | `'jpg'` |
| `classes` | `list[str] \| None` | `None` |

## Usage

```python
from ovannotators.writers import YOLOWriter, YOLOWriterConfig

writer = YOLOWriter(YOLOWriterConfig())
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
