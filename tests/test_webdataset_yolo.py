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

"""Package YOLO output as WebDataset samples and rebuild the native dataset."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import webdataset as wds

from ovannotators import (
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
    YOLOFrameAdapter,
    YOLOWriter,
    YOLOWriterConfig,
    export_finalized_dataset,
)
from ovannotators.exceptions import ConfigurationError, PackagingError
from ovannotators.writers._helpers import OVRTX_BBOX2D_DTYPE
from tests._webdataset_reconstruction import (
    assert_trees_equal,
    read_packaged_dataset,
    reconstruct_native,
)

pytestmark = [pytest.mark.integration, pytest.mark.cpu_only]

READERS_PYTHON = os.environ.get(
    "OVANNOTATORS_READERS_PYTHON", "/tmp/readers_venv/bin/python"
)
YOLO_LOADER = Path(__file__).parent / "dataset_loaders" / "load_yolo.py"

# Episode b restarts at frame 0 and discovers classes in a different order.
# Empty object lists are negative examples. val_split=0.5 puts even frames in
# val, so every episode holds both splits.
EPISODES = {
    "a": {
        0: [("cat", 1, 1, 9, 7)],
        1: [],
        2: [("dog", 4, 2, 15, 11), ("cat", 0, 0, 5, 5)],
    },
    "b": {
        0: [("dog", 2, 2, 10, 10)],
        1: [("bird", 3, 1, 8, 6), ("cat", 9, 4, 16, 12)],
        3: [],
    },
}


def _render_vars(frame, objects):
    boxes = np.zeros(len(objects), dtype=OVRTX_BBOX2D_DTYPE)
    labels = {}
    seg = np.zeros((12, 16), dtype=np.uint32)
    for index, (label, x0, y0, x1, y1) in enumerate(objects):
        semantic = {"cat": 1, "dog": 2, "bird": 3}[label]
        labels[semantic] = label
        boxes[index]["instance_id"] = (index + 1, 0, 0, 0)
        boxes[index]["semantic_id"] = (semantic, 0, 0, 0)
        boxes[index]["x_min"], boxes[index]["y_min"] = x0, y0
        boxes[index]["x_max"], boxes[index]["y_max"] = x1, y1
        seg[y0:y1, x0:x1] = index + 1
    color = np.zeros((12, 16, 3), dtype=np.uint8)
    color[..., 1] = 10 * frame
    return {
        "LdrColor": color,
        "SemanticBoundingBox2DTight": boxes,
        "SemanticIdMap": labels,
        "SemanticInstanceSegmentation": seg,
    }


def _config(**kwargs):
    return YOLOWriterConfig(val_split=0.5, image_format="png", **kwargs)


def _packager(root, **kwargs):
    return WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=root)),
        WebDatasetPackagerConfig(dataset_id="ds", **kwargs),
    )


def _natives(tmp_path, episodes=EPISODES, **config):
    natives = {}
    for identity, frames in episodes.items():
        natives[identity] = tmp_path / f"native-{identity}"
        disk = DiskBackend(DiskBackendConfig(root_dir=natives[identity]))
        writer = YOLOWriter(_config(**config))
        for frame, objects in frames.items():
            writer.write_frame(frame, _render_vars(frame, objects), disk)
        writer.finalize(disk)
    return natives


def _capture(root, episodes=EPISODES, **config):
    with DatasetCapture(YOLOWriter(_config(**config)), _packager(root)) as dataset:
        for identity, frames in episodes.items():
            with dataset.episode(identity) as episode:
                for frame, objects in frames.items():
                    episode.write_frame(frame, _render_vars(frame, objects))


def _names(native):
    line = (native / "data.yaml").read_text().splitlines()[-1]
    return ast.literal_eval(line.removeprefix("names: "))


def yolo_data_yaml(records, role):
    """Rebuild an episode's data.yaml from the per-sample class records."""
    assert role == "data"
    tables = {json.dumps(sample.json()["classes"]) for sample in records.samples}
    assert len(tables) == 1, "samples of one episode disagree on classes"
    classes = sorted(json.loads(tables.pop()), key=lambda c: c["source_id"])
    names = [c["name"] for c in classes]
    return (
        f"train: images/train\nval: images/val\nnc: {len(names)}\nnames: {names!r}\n"
    ).encode()


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_capture_keeps_negatives_splits_and_stable_class_meanings(tmp_path, task):
    _capture(tmp_path / "out", task=task)
    natives = _natives(tmp_path, task=task)
    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    manifest = dataset.manifest
    assert manifest["metadata"]["sample_schema"] == "yolo-frame-v1"
    table = manifest["metadata"]["classes"]
    assert table == [
        {"id": 0, "name": "cat"},
        {"id": 1, "name": "dog"},
        {"id": 2, "name": "bird"},
    ]
    assert manifest["metadata"]["split_counts"] == {"train": 3, "val": 3}
    assert _names(natives["b"]) == ["dog", "bird", "cat"]

    urls = [
        f"file:{(tmp_path / 'out' / s['path']).resolve().as_posix()}"
        for s in manifest["shards"]
    ]
    samples = list(wds.WebDataset(urls, shardshuffle=False).decode("pil"))
    assert [s["__key__"] for s in samples] == [
        "ds/episode-a/sample-0",
        "ds/episode-a/sample-1",
        "ds/episode-a/sample-2",
        "ds/episode-b/sample-0",
        "ds/episode-b/sample-1",
        "ds/episode-b/sample-3",
    ]
    dataset_names = [c["name"] for c in table]
    for sample in samples:
        episode = sample["__key__"].split("/")[1].removeprefix("episode-")
        frame = sample["json"]["frame_index"]
        split = "val" if frame % 2 == 0 else "train"
        assert sample["json"]["split"] == split
        assert sample["metadata.json"]["metadata"]["split"] == split
        assert sample["png"].size == (16, 12)
        native = natives[episode] / f"labels/{split}/{frame:06d}.txt"
        native_rows = [r.split(" ", 1) for r in native.read_text().splitlines()]
        rows = [r.split(" ", 1) for r in sample["txt"].splitlines()]
        assert len(rows) == len(EPISODES[episode][frame]) == sample["json"]["objects"]
        # Only the class column changes; it keeps the same class name.
        assert [r[1] for r in rows] == [r[1] for r in native_rows]
        native_names = _names(natives[episode])
        assert [dataset_names[int(r[0])] for r in rows] == [
            native_names[int(r[0])] for r in native_rows
        ]
        assert [dataset_names[int(r[0])] for r in rows] == [
            o[0] for o in EPISODES[episode][frame]
        ]
        if episode == "a":
            assert "native.txt" not in sample
        else:
            assert sample["native.txt"] == native.read_text()
            assert {c["name"]: c["id"] for c in sample["json"]["classes"]} == {
                "dog": 1,
                "bird": 2,
                "cat": 0,
            }
    negatives = [s for s in samples if s["json"]["objects"] == 0]
    assert [s["__key__"] for s in negatives] == [
        "ds/episode-a/sample-1",
        "ds/episode-b/sample-3",
    ]
    assert all(s["txt"] == "" for s in negatives)
    if task == "segment":
        assert any(len(r.split()) > 5 for s in samples for r in s["txt"].splitlines())


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_reconstruction_matches_native_output(tmp_path, task):
    _capture(tmp_path / "out", task=task)
    natives = _natives(tmp_path, task=task)
    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    paths = dataset.source_paths["episodes"]
    assert paths["b"]["shared"] == {"data": "data.yaml"}
    assert paths["b"]["samples"]["1"] == {
        "png": "images/train/000001.png",
        "native.txt": "labels/train/000001.txt",
    }
    assert paths["a"]["samples"]["1"]["txt"] == "labels/train/000001.txt"
    roots = reconstruct_native(
        dataset, tmp_path / "rebuilt", shared_builders={"data": yolo_data_yaml}
    )
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity])


def test_finalized_export_matches_capture_samples(tmp_path):
    _capture(tmp_path / "captured")
    natives = _natives(tmp_path)
    writer = YOLOWriter(_config())
    export_finalized_dataset(writer, natives, _packager(tmp_path / "exported"))
    captured = read_packaged_dataset(tmp_path / "captured", "ds")
    exported = read_packaged_dataset(tmp_path / "exported", "ds")
    assert [s.key for s in captured.samples] == [s.key for s in exported.samples]
    for left, right in zip(captured.samples, exported.samples):
        assert dict(left.members) == dict(right.members)
    metadata = dict(exported.manifest["metadata"])
    assert metadata.pop("provenance") == "finalized-export"
    assert {
        k: v for k, v in captured.manifest["metadata"].items() if k != "provenance"
    } == metadata
    roots = reconstruct_native(
        exported, tmp_path / "rebuilt", shared_builders={"data": yolo_data_yaml}
    )
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity])


def test_explicit_classes_keep_native_labels(tmp_path):
    classes = ["bird", "cat", "dog"]
    _capture(tmp_path / "out", classes=classes)
    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    assert [c["name"] for c in dataset.manifest["metadata"]["classes"]] == classes
    assert all("native.txt" not in s.members for s in dataset.samples)
    natives = _natives(tmp_path, classes=classes)
    roots = reconstruct_native(
        dataset, tmp_path / "rebuilt", shared_builders={"data": yolo_data_yaml}
    )
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity])
    with pytest.raises(PackagingError, match="explicit classes"):
        export_finalized_dataset(
            YOLOWriter(_config(classes=["cat", "dog", "bird"])),
            natives,
            _packager(tmp_path / "mismatch"),
        )
    assert not (tmp_path / "mismatch/ds/manifest.json").exists()
    with pytest.raises(ConfigurationError, match="unique"):
        DatasetCapture(
            YOLOWriter(_config(classes=["cat", "cat"])), _packager(tmp_path / "dup")
        )


def test_builtin_selection_and_subclass_requires_explicit_adapter(tmp_path):
    class CustomYOLO(YOLOWriter):
        pass

    with pytest.raises(
        ConfigurationError, match=r"exact built-in writer types .*YOLOWriter"
    ):
        DatasetCapture(CustomYOLO(_config()), _packager(tmp_path / "sub"))
    capture = DatasetCapture(
        CustomYOLO(_config()), _packager(tmp_path / "ok"), adapter=YOLOFrameAdapter()
    )
    assert isinstance(capture.adapter, YOLOFrameAdapter)
    capture.abort()
    with pytest.raises(ConfigurationError, match="requires a YOLOWriter"):
        DatasetCapture(object(), _packager(tmp_path / "x"), adapter=YOLOFrameAdapter())


def test_rejected_episode_does_not_leak_classes_or_split_counts(tmp_path):
    with DatasetCapture(
        YOLOWriter(_config()), _packager(tmp_path / "out", max_samples_per_shard=1)
    ) as dataset:
        with pytest.raises(PackagingError, match="exceeds shard limits"):
            with dataset.episode("big") as episode:
                episode.write_frame(0, _render_vars(0, [("dog", 0, 0, 4, 4)]))
                episode.write_frame(1, _render_vars(1, []))
        with dataset.episode("small") as episode:
            episode.write_frame(1, _render_vars(1, [("cat", 0, 0, 4, 4)]))
    metadata = read_packaged_dataset(tmp_path / "out", "ds").manifest["metadata"]
    assert metadata["classes"] == [{"id": 0, "name": "cat"}]
    assert metadata["split_counts"] == {"train": 1, "val": 0}


def _corrupt(native, case):
    if case == "missing data.yaml":
        (native / "data.yaml").unlink()
    elif case == "foreign data.yaml":
        (native / "data.yaml").write_text("names: {0: cat}\n")
    elif case == "both splits":
        shutil.copy(
            native / "images/val/000000.png", native / "images/train/000000.png"
        )
    elif case == "orphan label":
        (native / "labels/train/000005.txt").write_text("")
    elif case == "unexpected file":
        (native / "images/val/cover.png").write_bytes(b"")
    elif case == "unexpected split":
        (native / "images/test").mkdir()
    elif case == "missing label":
        (native / "labels/val/000000.txt").unlink()
    elif case == "bad class":
        (native / "labels/val/000000.txt").write_text("7 0.5 0.5 0.1 0.1\n")
    elif case == "bad row":
        (native / "labels/val/000000.txt").write_text("0 0.5 0.5 0.1\n")
    elif case == "bad number":
        (native / "labels/val/000000.txt").write_text("0 0.5 x 0.1 0.1\n")
    elif case == "duplicate names":
        (native / "data.yaml").write_text(
            "train: images/train\nval: images/val\nnc: 2\nnames: ['cat', 'cat']\n"
        )
    elif case == "no newline":
        (native / "labels/val/000000.txt").write_text("0 0.5 0.5 0.1 0.1")
    elif case == "not text":
        (native / "labels/val/000000.txt").write_bytes(b"\xff\n")
    elif case == "no images":
        shutil.rmtree(native / "images")
        shutil.rmtree(native / "labels")


@pytest.mark.parametrize(
    "case, match",
    [
        ("missing data.yaml", "Missing YOLO data.yaml"),
        ("foreign data.yaml", "not a YOLOWriter data.yaml"),
        ("both splits", "both splits"),
        ("orphan label", "without matching image"),
        ("unexpected file", "Unsupported YOLO file"),
        ("unexpected split", "Unexpected YOLO images entries"),
        ("missing label", "Missing YOLO label"),
        ("bad class", "label row"),
        ("bad row", "label row"),
        ("bad number", "label row"),
        ("duplicate names", "Duplicate class names"),
        ("no newline", "final newline"),
        ("not text", "UTF-8"),
        ("no images", "No YOLO images"),
    ],
)
def test_export_rejects_output_it_cannot_package(tmp_path, case, match):
    natives = _natives(tmp_path, {"a": EPISODES["a"]})
    _corrupt(natives["a"], case)
    with pytest.raises(PackagingError, match=match):
        export_finalized_dataset(
            YOLOWriter(_config()), natives, _packager(tmp_path / "out")
        )
    assert not (tmp_path / "out/ds/manifest.json").exists()


@pytest.mark.parametrize(
    ("row", "accepted"),
    [
        ("0 0.1 0.1 0.9 0.9 0.5 0.1", True),  # triangle
        ("0 0.5 0.5 0.2 0.2", True),  # the writer's box fallback for no polygon
        ("0 0.1 0.1", False),  # one point
        ("0 0.1 0.1 0.9 0.9 0.5", False),  # odd coordinate count
    ],
)
def test_segment_label_rows_are_polygons_or_the_box_fallback(tmp_path, row, accepted):
    natives = _natives(tmp_path, {"a": EPISODES["a"]}, task="segment")
    (natives["a"] / "labels/val/000000.txt").write_text(row + "\n")

    def export():
        export_finalized_dataset(
            YOLOWriter(_config(task="segment")), natives, _packager(tmp_path / "out")
        )

    if accepted:
        export()
        assert (tmp_path / "out/ds/manifest.json").exists()
    else:
        with pytest.raises(PackagingError, match="Unsupported YOLO segment label row"):
            export()


def _loader_summary(root):
    result = subprocess.run(
        [READERS_PYTHON, str(YOLO_LOADER), str(root)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    # Keep class names, image counts and per-image classes; drop paths/progress.
    return [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("nc / names", "[", "    "))
    ]


@pytest.mark.skipif(
    not Path(READERS_PYTHON).is_file(),
    reason=f"ultralytics reader interpreter not found at {READERS_PYTHON}; "
    "set OVANNOTATORS_READERS_PYTHON",
)
@pytest.mark.parametrize("task", ["detect", "segment"])
def test_reconstructed_yolo_passes_native_loader(tmp_path, task):
    _capture(tmp_path / "out", task=task)
    natives = _natives(tmp_path, task=task)
    roots = reconstruct_native(
        read_packaged_dataset(tmp_path / "out", "ds"),
        tmp_path / "rebuilt",
        shared_builders={"data": yolo_data_yaml},
    )
    summaries = {}
    for identity, native in natives.items():
        summaries[identity] = _loader_summary(roots[identity])
        assert summaries[identity] == _loader_summary(native)
    assert "nc / names      : 3 / ['dog', 'bird', 'cat']" in summaries["b"]
    assert "[train] 2 images" in summaries["b"]
    assert "    000001.png: 0 boxes  classes=[]" in summaries["a"]
    assert "    000002.png: 2 boxes  classes=[1, 0]" in summaries["a"]


def test_frame_validation_requires_one_image_per_frame(tmp_path):
    adapter, writer = YOLOFrameAdapter(), YOLOWriter(_config())
    with pytest.raises(PackagingError, match="needs one 000004.png image"):
        adapter.validate_frame_output(4, writer, tmp_path, None)
