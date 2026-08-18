#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and inspect a KITTI dataset produced by the ovannotators KITTIWriter.

Reader: plain text parser + Pillow (KITTI has no single canonical pip reader).

Usage:
    /tmp/readers_venv/bin/python load_kitti.py [DATASET_DIR]

DATASET_DIR defaults to ~/Downloads/test_output_5/kitti and should contain:
    image_2/   label_2/   calib/   [depth/]

IMPORTANT: standard KITTI readers split each label row on whitespace and expect
exactly 15 columns:
    type truncated occluded alpha left top right bottom h w l x y z rotation_y
This writer puts the raw semantic label (e.g. 'class: plane;') in the `type`
field; the embedded space yields 16 columns and breaks naive parsers.  To stay
robust, this loader parses the 14 trailing NUMERIC fields and treats everything
before them as the (possibly space-containing) type string.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image

NUMERIC_FIELDS = [
    "truncated",
    "occluded",
    "alpha",
    "left",
    "top",
    "right",
    "bottom",
    "h",
    "w",
    "l",
    "x",
    "y",
    "z",
    "rotation_y",
]


def parse_label_file(path: str) -> list[dict]:
    objs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # Robust parse: last 14 tokens are numeric; the rest is the type string.
        nums = parts[-14:]
        obj_type = " ".join(parts[:-14])
        row = {"type": obj_type}
        row.update({k: float(v) for k, v in zip(NUMERIC_FIELDS, nums)})
        # Flag rows that a strict 15-column reader would choke on.
        row["_strict_15col"] = len(parts) == 15
        objs.append(row)
    return objs


def parse_calib(path: str) -> dict:
    calib = {}
    for line in open(path):
        if ":" in line:
            key, vals = line.split(":", 1)
            calib[key.strip()] = np.array([float(x) for x in vals.split()])
    return calib


def load(dataset_dir: str) -> None:
    label_files = sorted(glob.glob(os.path.join(dataset_dir, "label_2", "*.txt")))
    if not label_files:
        raise FileNotFoundError(f"no label_2/*.txt under {dataset_dir}")
    print(f"dataset dir : {dataset_dir}")
    print(f"frames      : {len(label_files)}")

    has_depth = os.path.isdir(os.path.join(dataset_dir, "depth"))

    for lf in label_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        objs = parse_label_file(lf)

        img = Image.open(os.path.join(dataset_dir, "image_2", f"{stem}.png"))
        calib = parse_calib(os.path.join(dataset_dir, "calib", f"{stem}.txt"))
        P2 = calib["P2"].reshape(3, 4)

        print(f"\nframe {stem}: image {img.size} {img.mode}, {len(objs)} objects")
        print(
            f"    P2 (3x4) fx={P2[0, 0]:.2f} fy={P2[1, 1]:.2f} "
            f"cx={P2[0, 2]:.1f} cy={P2[1, 2]:.1f}"
        )
        strict = all(o["_strict_15col"] for o in objs)
        if not strict:
            raise ValueError(
                f"frame {stem}: rows are NOT 15-column (type contains a space) -- "
                "naive KITTI parsers will fail"
            )

        if has_depth:
            dep = np.array(
                Image.open(os.path.join(dataset_dir, "depth", f"{stem}.png"))
            )
            print(
                f"    depth {dep.dtype} {dep.shape} "
                f"metric range (m): {dep.min() / 256:.2f} - {dep.max() / 256:.2f}"
            )

        for o in objs:
            box2d = [round(o[k], 1) for k in ("left", "top", "right", "bottom")]
            dims = [round(o[k], 2) for k in ("h", "w", "l")]
            loc = [round(o[k], 2) for k in ("x", "y", "z")]
            print(
                f"      type='{o['type']}' 2d={box2d} hwl={dims} "
                f"xyz={loc} ry={o['rotation_y']:.3f} alpha={o['alpha']:.3f}"
            )


if __name__ == "__main__":
    default = os.path.expanduser("~/Downloads/test_output_5/kitti")
    load(sys.argv[1] if len(sys.argv) > 1 else default)
