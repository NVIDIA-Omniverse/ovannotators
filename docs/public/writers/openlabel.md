<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# OpenLABEL: 2D Image Annotations

This writer is registered as `openlabel` using the `OpenLABELWriter` class. It is
included with the core installation and supports CPU processing.

## Purpose and Inputs

Use for ASAM OpenLABEL 1.0 image datasets with 2D boxes or instance polygons.
The writer requires `LdrColor` and `SemanticBoundingBox2DTight`.
`include_segmentation=True` also requires `SemanticInstanceSegmentation`.
`SemanticIdMap` supplies class names, and `StableIdMap` can supply object names.

## Output and Data Types

Images and JSON labels share six-digit stems in `data/` and `labels/`.
Each JSON file has `openlabel.metadata.schema_version="1.0.0"` and an `objects`
mapping keyed by instance ID. Each object's bounding-box entry in `bbox` stores
a `val` array of `[center_x, center_y, width, height]` in image pixels.
Optional `poly2d` values use
`MODE_POLY2D_ABSOLUTE` pixel coordinates. The layout loads through FiftyOne's
`OpenLABELImageDataset` importer.

The polygon is the largest simplified outer contour of the instance mask.
Holes and disconnected smaller regions are not retained. The writer emits
2D image labels; it does not emit temporal tracks, 3D cuboids, or multi-camera
stream alignment.

References: [ASAM OpenLABEL](https://www.asam.net/standards/detail/openlabel/)
and [FiftyOne's OpenLABEL image importer](https://docs.voxel51.com/user_guide/import_datasets.html#openlabel-image).

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `image_format` | `Literal['png', 'jpg']` | `'png'` |
| `include_segmentation` | `bool` | `False` |

## Usage

Construct `OpenLABELWriter(OpenLABELWriterConfig(...))`, call `write_frame()`
for each frame, then `finalize()`. The complete test in
[`tests/test_openlabel_writer.py`](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/tests/test_openlabel_writer.py)
checks the output and its FiftyOne import. Use a separate backend root for each
dataset writer.
