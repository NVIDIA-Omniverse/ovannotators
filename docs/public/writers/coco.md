<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# COCO: Detection and Segmentation

This writer is registered as `coco` using the `COCOWriter` class. It is
included with the core installation and supports CPU processing.

## Purpose and Inputs

This writer supports image object detection, instance masks, and panoptic
annotations. Supply `LdrColor` and `SemanticBoundingBox2DTight` inputs.
Optionally include `SemanticIdMap` to resolve class labels; without it,
categories are named by their semantic ID. If you enable instance or panoptic
segmentation, also include `SemanticInstanceSegmentation`.

## Output and Data Types

Images are named `images/{frame_id:06d}.jpg` by default. `finalize()` writes
`annotations/instances_{split}.json` with `images`, `annotations`, and `categories`.
Image IDs are `frame_id + 1`; annotation and category IDs are positive integers.
Categories are assigned in first-seen label order. Each annotation contains
`bbox=[x_min,y_min,width,height]` in floating-point pixels, numeric `area`,
`category_id`, `image_id`, and `iscrowd=0`. Instance segmentation uses
uncompressed run-length encoding (`size` and integer `counts`).

Panoptic mode adds `annotations/panoptic_{split}.json` and RGB PNGs in
`annotations/panoptic_{split}/`. Decode each pixel's segment ID as
`R + 256*G + 65536*B`; zero denotes background. `segments_info` maps these
IDs to categories, areas, and pixel boxes.

`split` names annotations; it does not partition images. `bbox_mode="loose"`
pads tight boxes by 1% per side with a minimum of one pixel rather than reading
a separate loose-box variable. Annotations accumulate in memory until finalization.
This writer does not emit keypoints or captions.

Reference: [COCO data format](https://cocodataset.org/#format-data).

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `split` | `str` | `'train'` |
| `image_format` | `Literal['jpg', 'png']` | `'jpg'` |
| `include_segmentation` | `bool` | `False` |
| `include_panoptic` | `bool` | `False` |
| `bbox_mode` | `Literal['tight', 'loose']` | `'tight'` |

## Usage

```python
from ovannotators.writers import COCOWriter, COCOWriterConfig

writer = COCOWriter(COCOWriterConfig())
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
```

Here `render_vars` contains the inputs listed above and `backend` is a configured
storage backend. Use a fresh writer and output root for each dataset.

See {ref}`Tutorial 1 <writer-tutorial-coco>` for a renderer workflow.

See the [shared writer workflow](index.md) for input conventions and lifecycle.
