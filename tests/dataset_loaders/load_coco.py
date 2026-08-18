#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and inspect a COCO dataset produced by the ovannotators COCOWriter.

Reader: pycocotools

Usage:
    /tmp/readers_venv/bin/python load_coco.py [DATASET_DIR]

DATASET_DIR defaults to ~/Downloads/test_output_5/coco and should contain:
    images/                       {frame_id:06d}.jpg
    annotations/instances_*.json
"""

from __future__ import annotations

import os
import sys

from pycocotools.coco import COCO


def load(dataset_dir: str) -> None:
    # Find the instances_*.json annotation file.
    ann_dir = os.path.join(dataset_dir, "annotations")
    ann_files = [
        f
        for f in os.listdir(ann_dir)
        if f.startswith("instances_") and f.endswith(".json")
    ]
    if not ann_files:
        raise FileNotFoundError(f"no instances_*.json in {ann_dir}")
    ann_path = os.path.join(ann_dir, ann_files[0])

    coco = COCO(ann_path)

    img_ids = sorted(coco.getImgIds())
    cats = coco.loadCats(coco.getCatIds())
    print(f"annotation file : {ann_path}")
    print(f"images          : {len(img_ids)}")
    print(f"annotations     : {len(coco.getAnnIds())}")
    print(f"categories      : {[(c['id'], c['name']) for c in cats]}")

    # Per-image annotation rundown.
    for iid in img_ids:
        im = coco.loadImgs(iid)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid))
        img_path = os.path.join(dataset_dir, "images", im["file_name"])
        exists = "ok" if os.path.exists(img_path) else "MISSING"
        print(
            f"\nimage id={iid} file={im['file_name']} "
            f"({im['width']}x{im['height']}) [{exists}] -> {len(anns)} anns"
        )
        for a in anns:
            bbox = [round(v, 1) for v in a["bbox"]]
            has_seg = bool(a.get("segmentation"))
            print(
                f"    cat={a['category_id']} bbox(xywh)={bbox} "
                f"area={round(a['area'], 1)} iscrowd={a['iscrowd']} seg={has_seg}"
            )


if __name__ == "__main__":
    default = os.path.expanduser("~/Downloads/test_output_5/coco")
    load(sys.argv[1] if len(sys.argv) > 1 else default)
