#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and inspect an Ultralytics YOLO dataset produced by the YOLOWriter.

Reader: ultralytics

Usage:
    /tmp/readers_venv/bin/python load_yolo.py [DATASET_DIR]

DATASET_DIR defaults to ~/Downloads/test_output_5/yolo and should contain:
    images/{train,val}/   labels/{train,val}/   data.yaml

Note: the writer's data.yaml uses `path: .`, which Ultralytics resolves against
its global datasets dir rather than the yaml location.  This script writes a
temporary copy of data.yaml with `path` set to the absolute dataset root so the
Ultralytics loader resolves correctly.
"""

from __future__ import annotations

import os
import sys
import tempfile

import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


def load(dataset_dir: str) -> None:
    dataset_dir = os.path.abspath(dataset_dir)
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    data = yaml.safe_load(open(yaml_path))
    print(f"data.yaml       : {yaml_path}")
    print(f"nc / names      : {data['nc']} / {data['names']}")

    # Rewrite `path` to the absolute dataset root for reliable resolution.
    data["path"] = dataset_dir
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=tempfile.gettempdir()
    )
    yaml.safe_dump(data, tmp)
    tmp.close()

    res = check_det_dataset(tmp.name, autodownload=False)
    print(f"resolved train  : {res['train']}")
    print(f"resolved val    : {res['val']}")

    for split in ("train", "val"):
        path = res.get(split)
        if not path or not os.path.isdir(path):
            continue
        ds = YOLODataset(img_path=path, data=res, task="detect")
        print(f"\n[{split}] {len(ds.im_files)} images")
        for im_file, label in zip(ds.im_files, ds.labels):
            cls = label["cls"].ravel().astype(int).tolist()
            print(f"    {os.path.basename(im_file)}: {len(cls)} boxes  classes={cls}")

    os.unlink(tmp.name)


if __name__ == "__main__":
    default = os.path.expanduser("~/Downloads/test_output_5/yolo")
    load(sys.argv[1] if len(sys.argv) > 1 else default)
