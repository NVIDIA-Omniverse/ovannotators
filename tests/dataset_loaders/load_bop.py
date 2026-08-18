#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and inspect a BOP dataset produced by the ovannotators BOPWriter.

Reader: bop_toolkit_lib  (pip package: bop-toolkit-lib)

Usage:
    /tmp/readers_venv/bin/python load_bop.py [DATASET_DIR]

DATASET_DIR defaults to ~/Downloads/test_output_5/bop and should contain one or
more scene dirs ({scene_id:06d}/) each with:
    rgb/ depth/ mask/ mask_visib/
    scene_camera.json  scene_gt.json  scene_gt_info.json
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
from bop_toolkit_lib import inout


def load_scene(scene_dir: str) -> None:
    sc = inout.load_scene_camera(os.path.join(scene_dir, "scene_camera.json"))
    gt = inout.load_scene_gt(os.path.join(scene_dir, "scene_gt.json"))
    gti = inout.load_json(
        os.path.join(scene_dir, "scene_gt_info.json"), keys_to_int=True
    )

    frames = sorted(sc.keys())
    print(f"scene dir   : {scene_dir}")
    print(f"frames      : {frames}")

    for f in frames:
        cam = sc[f]
        objs = gt[f]
        rgb_path = os.path.join(scene_dir, "rgb", f"{f:06d}.png")
        depth_path = os.path.join(scene_dir, "depth", f"{f:06d}.png")
        rgb = inout.load_im(rgb_path)
        depth = inout.load_depth(depth_path)
        depth_mm = depth * cam["depth_scale"]

        n_mask = len(glob.glob(os.path.join(scene_dir, "mask", f"{f:06d}_*.png")))
        n_maskv = len(
            glob.glob(os.path.join(scene_dir, "mask_visib", f"{f:06d}_*.png"))
        )

        print(f"\nframe {f}:")
        print(f"    rgb {rgb.shape}  depth {depth.shape} {depth.dtype}")
        print(
            f"    depth metric range (mm): "
            f"{float(depth_mm.min()):.1f} - {float(depth_mm.max()):.1f}"
        )
        print(
            f"    fx={cam['cam_K'][0, 0]:.2f} fy={cam['cam_K'][1, 1]:.2f} "
            f"cx={cam['cam_K'][0, 2]:.1f} cy={cam['cam_K'][1, 2]:.1f}"
        )
        print(
            f"    GT objects: {len(objs)}  mask files: {n_mask}  "
            f"mask_visib files: {n_maskv}"
        )
        for o, info in zip(objs, gti[f]):
            R = o["cam_R_m2c"]
            orthonormal = np.allclose(R @ R.T, np.eye(3), atol=1e-3)
            print(
                f"      obj_id={o['obj_id']} t_m2c(mm)={o['cam_t_m2c'].ravel()} "
                f"R_orthonormal={orthonormal} "
                f"bbox_visib={info['bbox_visib']} visib_fract={info['visib_fract']}"
            )


def load(dataset_dir: str) -> None:
    scene_dirs = sorted(
        d
        for d in glob.glob(os.path.join(dataset_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "scene_camera.json"))
    )
    if not scene_dirs:
        raise FileNotFoundError(f"no BOP scene dirs under {dataset_dir}")
    for sd in scene_dirs:
        load_scene(sd)


if __name__ == "__main__":
    default = os.path.expanduser("~/Downloads/test_output_5/bop")
    load(sys.argv[1] if len(sys.argv) > 1 else default)
