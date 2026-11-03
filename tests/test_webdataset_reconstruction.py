# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rebuild native Basic/COCO datasets from WebDataset shards and compare them."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ovannotators import (
    BasicWriter,
    BasicWriterConfig,
    COCOWriter,
    COCOWriterConfig,
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
    export_finalized_dataset,
)
from ovannotators.writers._helpers import OVRTX_BBOX2D_DTYPE
from tests._webdataset_reconstruction import (
    ReconstructionError,
    assert_trees_equal,
    coco_documents_equivalent,
    coco_shared_document,
    compare_trees,
    read_packaged_dataset,
    _native_target,
    reconstruct_native,
)

pytestmark = [pytest.mark.integration, pytest.mark.cpu_only]

READERS_PYTHON = os.environ.get(
    "OVANNOTATORS_READERS_PYTHON", "/tmp/readers_venv/bin/python"
)
COCO_LOADER = Path(__file__).parent / "dataset_loaders" / "load_coco.py"
COCO_BUILDERS = {"instances": coco_shared_document, "panoptic": coco_shared_document}
COCO_COMPARATORS = {"annotations/*.json": coco_documents_equivalent}


def _packager(root, **kwargs):
    return WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=root)),
        WebDatasetPackagerConfig(dataset_id="ds", **kwargs),
    )


def _render_vars(label_id=1):
    """Return one COCO frame's render vars with a single tight 2D box."""
    boxes = np.zeros(1, dtype=OVRTX_BBOX2D_DTYPE)
    boxes[0]["instance_id"] = (1, 0, 0, 0)
    boxes[0]["semantic_id"] = (label_id, 0, 0, 0)
    boxes[0]["x_max"] = 8
    boxes[0]["y_max"] = 6
    return {
        "LdrColor": np.full((6, 8, 3), 127, dtype=np.uint8),
        "SemanticBoundingBox2DTight": boxes,
    }


def _basic_values(frame):
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    color[..., 0] = frame
    return {
        "LdrColor": color,
        "DistanceToCameraSD": np.array([[1.25, np.inf, frame]], dtype=np.float32),
    }


def _basic_natives(tmp_path, frames):
    config = BasicWriterConfig(depth_format="npy", colorize=False)
    natives = {}
    for identity, ids in frames.items():
        natives[identity] = tmp_path / f"native-{identity}"
        disk = DiskBackend(DiskBackendConfig(root_dir=natives[identity]))
        writer = BasicWriter(config)
        for frame in ids:
            writer.write_frame(frame, _basic_values(frame), disk)
        writer.finalize(disk)
    return BasicWriter(config), natives


def _coco_capture(tmp_path, frames, **config):
    """Capture COCO samples and write the same frames natively per episode."""
    writer = COCOWriter(COCOWriterConfig(image_format="png", **config))

    def values(label):
        result = _render_vars(label)
        if config.get("include_panoptic"):
            result["SemanticInstanceSegmentation"] = np.ones((6, 8), dtype=np.uint32)
        return result

    with DatasetCapture(
        writer, _packager(tmp_path / "out", max_episodes_per_shard=1)
    ) as dataset:
        for identity, items in frames.items():
            with dataset.episode(identity) as episode:
                for frame, label in items:
                    episode.write_frame(frame, values(label))
    natives = {}
    for identity, items in frames.items():
        natives[identity] = tmp_path / f"native-{identity}"
        disk = DiskBackend(DiskBackendConfig(root_dir=natives[identity]))
        native_writer = COCOWriter(COCOWriterConfig(image_format="png", **config))
        for frame, label in items:
            native_writer.write_frame(frame, values(label), disk)
        native_writer.finalize(disk)
    return natives


def test_basic_spanned_export_reconstructs_byte_identical_frames(tmp_path):
    writer, natives = _basic_natives(tmp_path, {"a": [0, 3, 4], "b": [1]})
    export_finalized_dataset(
        writer,
        natives,
        _packager(
            tmp_path / "out", max_samples_per_shard=2, allow_episode_spanning=True
        ),
    )
    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    assert len(dataset.manifest["shards"]) == 2
    assert [s.key for s in dataset.samples] == [
        "ds/episode-a/sample-0",
        "ds/episode-a/sample-3",
        "ds/episode-a/sample-4",
        "ds/episode-b/sample-1",
    ]
    roots = reconstruct_native(dataset, tmp_path / "rebuilt")
    assert sorted(roots) == ["a", "b"]
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity])
    assert (roots["a"] / "0003/distance_to_camera.npy").is_file()


@pytest.mark.parametrize("include_panoptic", [False, True])
def test_coco_capture_reconstructs_images_and_equivalent_annotations(
    tmp_path, include_panoptic
):
    frames = {"a": [(3, 1), (5, 2)], "b": [(0, 2), (1, 3)]}
    natives = _coco_capture(
        tmp_path,
        frames,
        include_panoptic=include_panoptic,
        include_segmentation=include_panoptic,
    )
    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    roots = reconstruct_native(
        dataset, tmp_path / "rebuilt", shared_builders=COCO_BUILDERS
    )
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity], comparators=COCO_COMPARATORS)
        assert (
            compare_trees(
                native, roots[identity], ignore=["annotations/*.json"]
            ).changed
            == {}
        )
    # Episode b's native category IDs differ from the dataset-wide IDs, so the
    # documents are equivalent only after the semantic remap.
    native_doc = (natives["b"] / "annotations/instances_train.json").read_bytes()
    rebuilt_doc = (roots["b"] / "annotations/instances_train.json").read_bytes()
    assert native_doc != rebuilt_doc
    assert coco_documents_equivalent(native_doc, rebuilt_doc) is None
    rebuilt = json.loads(rebuilt_doc)
    assert [i["file_name"] for i in rebuilt["images"]] == ["000000.png", "000001.png"]
    assert {c["name"]: c["id"] for c in rebuilt["categories"]} == {"2": 2, "3": 3}


@pytest.mark.skipif(
    not Path(READERS_PYTHON).is_file(),
    reason=f"pycocotools reader interpreter not found at {READERS_PYTHON}; "
    "set OVANNOTATORS_READERS_PYTHON (CI has no such interpreter, so this "
    "test runs only on machines that provide one)",
)
def test_reconstructed_coco_passes_native_loader(tmp_path):
    _coco_capture(tmp_path, {"a": [(3, 1), (5, 2)]})
    roots = reconstruct_native(
        read_packaged_dataset(tmp_path / "out", "ds"),
        tmp_path / "rebuilt",
        shared_builders=COCO_BUILDERS,
    )
    result = subprocess.run(
        [READERS_PYTHON, str(COCO_LOADER), str(roots["a"])],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "images          : 2" in result.stdout
    assert "MISSING" not in result.stdout


def test_reader_rejects_tampered_shard_and_auxiliary(tmp_path):
    writer, natives = _basic_natives(tmp_path, {"a": [0]})
    export_finalized_dataset(writer, natives, _packager(tmp_path / "out"))
    shard = tmp_path / "out/ds/shard-000000.tar"
    original = shard.read_bytes()
    shard.write_bytes(original[:-1] + b"\x01")
    with pytest.raises(ReconstructionError, match="sha256 mismatch"):
        read_packaged_dataset(tmp_path / "out", "ds")
    shard.write_bytes(original)
    (tmp_path / "out/ds/auxiliary/source-paths.json").write_text("{}")
    with pytest.raises(ReconstructionError, match="Byte count mismatch"):
        read_packaged_dataset(tmp_path / "out", "ds")


def test_reconstruction_requires_shared_builders_and_safe_paths(tmp_path):
    _coco_capture(tmp_path, {"a": [(0, 1)]})
    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    with pytest.raises(ReconstructionError, match="No shared builder"):
        reconstruct_native(dataset, tmp_path / "missing-builder")
    dataset.source_paths["episodes"]["a"]["samples"]["0"]["png"] = "../escape.png"
    with pytest.raises(ReconstructionError, match="Unsafe native path"):
        reconstruct_native(dataset, tmp_path / "unsafe", shared_builders=COCO_BUILDERS)
    assert not (tmp_path / "escape.png").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "/abs.png",
        "../x.png",
        "a/../../x.png",
        "C:\\outside.bin",
        "..\\x.png",
        "\\\\host\\s\\x",
        "",
    ],
)
def test_native_target_rejects_paths_that_could_leave_the_destination(
    tmp_path, relative
):
    with pytest.raises(ReconstructionError, match="Unsafe native path"):
        _native_target(tmp_path, relative)
    assert _native_target(tmp_path, "annotations/a.json") == (
        tmp_path / "annotations" / "a.json"
    )


def test_tree_comparison_reports_missing_extra_and_semantic_changes(tmp_path):
    natives = _coco_capture(tmp_path, {"a": [(0, 1)]})
    roots = reconstruct_native(
        read_packaged_dataset(tmp_path / "out", "ds"),
        tmp_path / "rebuilt",
        shared_builders=COCO_BUILDERS,
    )
    rebuilt = roots["a"]
    document = json.loads((rebuilt / "annotations/instances_train.json").read_text())
    document["categories"][0]["name"] = "renamed"
    (rebuilt / "annotations/instances_train.json").write_text(json.dumps(document))
    (rebuilt / "images/000000.png").write_bytes(b"not a png")
    (rebuilt / "images/extra.png").write_bytes(b"")
    (natives["a"] / "images/only-native.png").write_bytes(b"")
    difference = compare_trees(natives["a"], rebuilt, comparators=COCO_COMPARATORS)
    assert difference.missing == ("images/only-native.png",)
    assert difference.extra == ("images/extra.png",)
    assert difference.changed == {
        "annotations/instances_train.json": (
            "COCO documents differ in ['annotations', 'categories']"
        ),
        "images/000000.png": "bytes differ",
    }
    with pytest.raises(AssertionError, match="missing: images/only-native.png"):
        assert_trees_equal(natives["a"], rebuilt, comparators=COCO_COMPARATORS)
