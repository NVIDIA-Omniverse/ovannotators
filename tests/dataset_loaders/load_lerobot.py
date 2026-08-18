#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and inspect a LeRobot dataset produced by the ovannotators LeRobotWriter.

Readers: pyarrow (parquet) + PyAV (video) + json (meta).

Usage:
    /tmp/readers_venv/bin/python load_lerobot.py [DATASET_DIR]

DATASET_DIR defaults to ~/Downloads/test_output_5/lerobot and should contain:
    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/observation.images.{cam}/episode_XXXXXX.mp4
    meta/{info,tasks,episodes,stats}.json

NOTE: This writer emits a LeRobot **v2.0**-style layout and omits the
`codebase_version` key, so the current official `LeRobotDataset` loader
(codebase v3.0) cannot read it directly.  This script first *attempts* the
official loader, then always falls back to reading the parquet/video/meta
directly -- which is the reliable path for this data.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pyarrow.parquet as pq


def try_official_loader(dataset_dir: str) -> bool:
    """Attempt the official LeRobotDataset loader; return True on success."""
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id="local/inspect", root=os.path.abspath(dataset_dir))
        print(
            f"  [official loader OK] frames={ds.num_frames} "
            f"episodes={ds.num_episodes} fps={ds.fps}"
        )
        return True
    except Exception as e:  # noqa: BLE001 - we want the fallback regardless
        print(
            f"  [official loader failed: {type(e).__name__}] {str(e).splitlines()[0]}"
        )
        print("  -> falling back to direct parquet/video/meta read")
        return False


def read_meta(dataset_dir: str) -> None:
    meta_dir = os.path.join(dataset_dir, "meta")
    info = json.load(open(os.path.join(meta_dir, "info.json")))
    tasks = json.load(open(os.path.join(meta_dir, "tasks.json")))
    episodes = json.load(open(os.path.join(meta_dir, "episodes.json")))
    print("\nmeta/info.json:")
    print(
        f"    dataset_type={info.get('dataset_type')} fps={info.get('fps')} "
        f"codebase_version={info.get('codebase_version', '<MISSING>')}"
    )
    print(
        f"    total_frames={info.get('total_frames')} "
        f"total_episodes={info.get('total_episodes')} "
        f"total_videos={info.get('total_videos')}"
    )
    print(f"    features={list(info.get('features', {}).keys())}")
    print(f"meta/tasks.json   : {tasks}")
    print(f"meta/episodes.json: {episodes}")


def read_parquet(dataset_dir: str) -> None:
    for pf in sorted(
        glob.glob(os.path.join(dataset_dir, "data", "**", "*.parquet"), recursive=True)
    ):
        t = pq.read_table(pf)
        cols = {n: str(t.schema.field(n).type) for n in t.column_names}
        d = t.to_pydict()
        print(f"\nparquet {os.path.relpath(pf, dataset_dir)}: {t.num_rows} rows")
        print(f"    schema: {cols}")
        print(f"    frame_index: {d.get('frame_index')}")
        print(f"    timestamp  : {[round(x, 4) for x in d.get('timestamp', [])]}")


def read_videos(dataset_dir: str) -> None:
    vids = sorted(
        glob.glob(os.path.join(dataset_dir, "videos", "**", "*.mp4"), recursive=True)
    )
    for v in vids:
        rel = os.path.relpath(v, dataset_dir)
        try:
            import av

            c = av.open(v)
            st = c.streams.video[0]
            n = sum(1 for _ in c.decode(video=0))
            print(
                f"\nvideo {rel}: codec={st.codec_context.name} "
                f"{st.codec_context.width}x{st.codec_context.height} frames={n}"
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"\nvideo {rel}: decode failed ({type(e).__name__}: {e}) "
                f"[size={os.path.getsize(v)} bytes]"
            )


def load(dataset_dir: str) -> None:
    print(f"dataset dir : {dataset_dir}")
    print("\n== official LeRobotDataset loader ==")
    try_official_loader(dataset_dir)
    print("\n== direct read ==")
    read_meta(dataset_dir)
    read_parquet(dataset_dir)
    read_videos(dataset_dir)


if __name__ == "__main__":
    default = os.path.expanduser("~/Downloads/test_output_5/lerobot")
    load(sys.argv[1] if len(sys.argv) > 1 else default)
