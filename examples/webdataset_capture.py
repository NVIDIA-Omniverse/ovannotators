# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebDataset capture example: package two COCO episodes and read them back.

This example needs no renderer or GPU. It synthesizes small renderer-style
frames with NumPy, captures two episodes through ``COCOWriter`` into
episode-preserving WebDataset shards on disk, then verifies the published
dataset the way a training job should consume it:

    python examples/webdataset_capture.py

The optional ``webdataset`` reader package is used for a final read-back when
it is installed; otherwise only the standard-library ``tarfile`` path runs.
"""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
from ovannotators import DatasetCapture
from ovannotators.backends import (
    DiskBackend,
    DiskBackendConfig,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
)
from ovannotators.writers import COCOWriter, COCOWriterConfig

out_dir = Path("_out/webdataset_capture")
dataset_id = "example-run"

# SemanticBoundingBox2DTight record layout produced by ovrtx: 128-bit instance
# and semantic IDs as four little-endian uint32 words, then pixel extents.
bbox2d_dtype = np.dtype(
    [
        ("instance_id", "<u4", (4,)),
        ("semantic_id", "<u4", (4,)),
        ("x_min", "<i4"),
        ("y_min", "<i4"),
        ("x_max", "<i4"),
        ("y_max", "<i4"),
    ]
)
rng = np.random.default_rng(7)


def synthetic_frame(frame_index: int) -> dict[str, np.ndarray]:
    """Return a small renderer-style frame with one moving bounding box."""
    boxes = np.zeros(1, dtype=bbox2d_dtype)
    boxes[0]["instance_id"] = (1, 0, 0, 0)
    boxes[0]["semantic_id"] = (1, 0, 0, 0)
    x = 8 + 4 * frame_index
    boxes[0]["x_min"], boxes[0]["y_min"] = x, 10
    boxes[0]["x_max"], boxes[0]["y_max"] = x + 20, 40
    return {
        "LdrColor": rng.integers(0, 256, (48, 64, 4), dtype=np.uint8),
        "SemanticBoundingBox2DTight": boxes,
    }


# A dataset namespace cannot be overwritten; start from a clean directory.
shutil.rmtree(out_dir, ignore_errors=True)
storage = DiskBackend(DiskBackendConfig(root_dir=out_dir))
packager = WebDatasetPackager(
    storage,
    WebDatasetPackagerConfig(
        dataset_id=dataset_id, max_shard_bytes="64 MiB", max_episodes_per_shard=1
    ),
)
writer = COCOWriter(COCOWriterConfig(image_format="png"))
episodes = {"episode001": 3, "episode002": 2}
with DatasetCapture(writer=writer, packager=packager) as dataset:
    for episode_id, num_frames in episodes.items():
        with dataset.episode(episode_id) as episode:
            for frame_index in range(num_frames):
                episode.write_frame(frame_index, synthetic_frame(frame_index))

# The manifest is published last: its presence marks a complete dataset.
manifest_path = out_dir / dataset_id / "manifest.json"
manifest = json.loads(manifest_path.read_text())
shard_paths = []
for shard in manifest["shards"]:
    path = out_dir / shard["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != shard["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {shard['path']}")
    shard_paths.append(path)

sample_count = 0
for path in shard_paths:
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if member.name.endswith(".metadata.json"):
                identity = json.load(archive.extractfile(member))
                key = member.name.removesuffix(".metadata.json")
                record = json.load(archive.extractfile(key + ".json"))
                assert record["image"]["file_name"] == key + ".png"
                print(
                    f"{path.name}: episode {identity['episode_id']} sample "
                    f"{identity['sample_id']} with "
                    f"{len(record['annotations'])} annotation(s)"
                )
                sample_count += 1
if sample_count != manifest["samples"]:
    raise RuntimeError("Shard contents do not match the manifest sample count")

try:
    import webdataset as wds
except ImportError:
    print("Install the optional `webdataset` package to also try its reader.")
else:
    reader = wds.WebDataset(
        [f"file:{p.resolve().as_posix()}" for p in shard_paths], shardshuffle=False
    )
    decoded = list(reader.decode("pil"))
    print(f"webdataset reader decoded {len(decoded)} samples")

print(
    f"Packaged {manifest['samples']} samples from {len(manifest['episodes'])} "
    f"episodes into {len(shard_paths)} shards under {(out_dir / dataset_id).as_posix()}"
)
