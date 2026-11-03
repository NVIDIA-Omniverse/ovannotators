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

"""Package KITTI output as WebDataset samples and rebuild the native tree."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ovannotators import (
    AdapterContext,
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    KITTIFrameAdapter,
    KITTIWriter,
    KITTIWriterConfig,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
    export_finalized_dataset,
)
from ovannotators.exceptions import ConfigurationError, PackagingError
from ovannotators.writers._helpers import OVRTX_BBOX2D_DTYPE, OVRTX_BBOX3D_DTYPE
from tests._webdataset_reconstruction import (
    assert_trees_equal,
    read_packaged_dataset,
    reconstruct_native,
)

pytestmark = [pytest.mark.integration, pytest.mark.cpu_only]

# No default path: a fixed location such as /tmp could hold an executable that
# another local user planted, and the test would run it.
READERS_PYTHON = os.environ.get("OVANNOTATORS_READERS_PYTHON", "")
KITTI_LOADER = Path(__file__).parent / "dataset_loaders" / "load_kitti.py"
COMPONENTS = ["calib.txt", "depth.png", "image_2.png", "label_2.txt", "metadata.json"]


def _packager(root, **kwargs):
    return WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=root)),
        WebDatasetPackagerConfig(dataset_id="ds", **kwargs),
    )


def _frame(base, frame, *, negative=False):
    values = dict(base)
    color = base["LdrColor"].copy()
    color[0, 0, 0] = frame
    values["LdrColor"] = color
    depth = base["DistanceToCameraSD"].copy()
    depth[0, 0] = 1.0 + frame
    values["DistanceToCameraSD"] = depth
    if negative:
        values["SemanticBoundingBox2DTight"] = np.zeros(0, OVRTX_BBOX2D_DTYPE)
        values["SemanticBoundingBox3D"] = np.zeros(0, OVRTX_BBOX3D_DTYPE)
    return values


def _natives(tmp_path, base, frames, config):
    """Write each episode directly with KITTIWriter as the native reference."""
    natives = {}
    for identity, items in frames.items():
        natives[identity] = tmp_path / f"native-{identity}"
        disk = DiskBackend(DiskBackendConfig(root_dir=natives[identity]))
        writer = KITTIWriter(config)
        for frame, negative in items:
            writer.write_frame(frame, _frame(base, frame, negative=negative), disk)
        writer.finalize(disk)
    return natives


def _tree_digest(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# Episodes restart at frame 0; frame 1 of episode a is a negative example.
FRAMES = {"a": [(0, False), (1, True), (4, False)], "b": [(0, False), (2, False)]}


def test_capture_reconstructs_byte_identical_kitti_tree(tmp_path, sample_render_vars):
    config = KITTIWriterConfig(write_depth=True)
    with DatasetCapture(
        KITTIWriter(config), _packager(tmp_path / "out", max_episodes_per_shard=1)
    ) as dataset:
        assert isinstance(dataset.adapter, KITTIFrameAdapter)
        for identity, items in FRAMES.items():
            with dataset.episode(identity) as episode:
                for frame, negative in items:
                    episode.write_frame(
                        frame, _frame(sample_render_vars, frame, negative=negative)
                    )
    natives = _natives(tmp_path, sample_render_vars, FRAMES, config)

    packaged = read_packaged_dataset(tmp_path / "out", "ds")
    assert [s.key for s in packaged.samples] == [
        "ds/episode-a/sample-0",
        "ds/episode-a/sample-1",
        "ds/episode-a/sample-4",
        "ds/episode-b/sample-0",
        "ds/episode-b/sample-2",
    ]
    assert all(sorted(s.members) == COMPONENTS for s in packaged.samples)
    negative = packaged.samples[1]
    assert negative.members["label_2.txt"] == b""
    assert negative.metadata["metadata"] == {
        "writer": "kitti",
        "frame_index": 1,
        "depth_scale": 256.0,
    }
    metadata = packaged.manifest["metadata"]
    assert metadata["sample_schema"] == "kitti-frame-v1"
    assert metadata["writer"] == "kitti"
    assert metadata["writer_config"]["depth_scale"] == 256.0
    assert packaged.source_paths["episodes"]["b"]["samples"]["2"] == {
        "image_2.png": "image_2/000002.png",
        "label_2.txt": "label_2/000002.txt",
        "calib.txt": "calib/000002.txt",
        "depth.png": "depth/000002.png",
    }

    roots = reconstruct_native(packaged, tmp_path / "rebuilt")
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity])
    # Restarted indices keep distinct content per episode.
    assert (roots["a"] / "image_2/000000.png").read_bytes() == (
        natives["a"] / "image_2/000000.png"
    ).read_bytes()
    assert (roots["a"] / "label_2/000004.txt").read_text().startswith("car ")


def test_export_finalized_without_depth_leaves_sources_unchanged(
    tmp_path, sample_render_vars
):
    config = KITTIWriterConfig(classes=["person"])
    natives = _natives(tmp_path, sample_render_vars, FRAMES, config)
    before = {k: _tree_digest(v) for k, v in natives.items()}
    export_finalized_dataset(
        KITTIWriter(config),
        natives,
        _packager(
            tmp_path / "out", max_samples_per_shard=2, allow_episode_spanning=True
        ),
    )
    assert {k: _tree_digest(v) for k, v in natives.items()} == before

    packaged = read_packaged_dataset(tmp_path / "out", "ds")
    assert len(packaged.manifest["shards"]) == 3
    assert packaged.manifest["metadata"]["provenance"] == "finalized-export"
    assert all("depth.png" not in s.members for s in packaged.samples)
    assert "depth_scale" not in packaged.samples[0].metadata["metadata"]
    labels = packaged.samples[0].members["label_2.txt"].decode()
    assert labels.startswith("person ") and "car" not in labels
    roots = reconstruct_native(packaged, tmp_path / "rebuilt")
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity])


@pytest.mark.skipif(
    not READERS_PYTHON or not Path(READERS_PYTHON).is_file(),
    reason="KITTI reader interpreter not found; set OVANNOTATORS_READERS_PYTHON "
    "to a Python with the KITTI loader dependencies",
)
def test_reconstructed_kitti_passes_native_loader(tmp_path, sample_render_vars):
    frames = {"a": [(0, False), (3, True)]}
    config = KITTIWriterConfig(write_depth=True)
    natives = _natives(tmp_path, sample_render_vars, frames, config)
    export_finalized_dataset(KITTIWriter(config), natives, _packager(tmp_path / "out"))
    roots = reconstruct_native(
        read_packaged_dataset(tmp_path / "out", "ds"), tmp_path / "rebuilt"
    )

    def load(root):
        result = subprocess.run(
            [READERS_PYTHON, str(KITTI_LOADER), str(root)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.split("\n", 1)[1]

    rebuilt = load(roots["a"])
    assert "frames      : 2" in rebuilt
    assert "frame 000003: image (640, 480) RGBA, 0 objects" in rebuilt
    assert "depth uint16 (480, 640)" in rebuilt
    assert rebuilt == load(natives["a"])


def test_builtin_selection_and_writer_validation(tmp_path):
    class CustomKITTI(KITTIWriter):
        pass

    packager = _packager(tmp_path / "out")
    with pytest.raises(ConfigurationError, match="KITTIWriter"):
        DatasetCapture(CustomKITTI(), packager)
    with pytest.raises(ConfigurationError, match="requires a KITTIWriter"):
        DatasetCapture(object(), packager, adapter=KITTIFrameAdapter())
    capture = DatasetCapture(CustomKITTI(), packager, adapter=KITTIFrameAdapter())
    assert type(capture.adapter.create_writer(CustomKITTI())) is CustomKITTI
    capture.abort()


def test_missing_frame_file_is_rejected(tmp_path):
    root = tmp_path / "native"
    for directory in ("image_2", "label_2"):
        (root / directory).mkdir(parents=True)
    (root / "image_2/000000.png").write_bytes(b"png")
    (root / "label_2/000000.txt").write_bytes(b"")
    context = AdapterContext("ds", "a", tmp_path / "generated")
    with pytest.raises(PackagingError, match="no calib file for frame 0"):
        KITTIFrameAdapter().validate_frame_output(0, KITTIWriter(), root, context)


def _replace_with_file(path):
    shutil.rmtree(path)
    path.write_text("")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: (r / "depth").mkdir(), "expected"),
        (lambda r: (r / "notes.txt").write_text(""), "expected"),
        (lambda r: (r / "calib/000000.txt").unlink(), "calib frames differ"),
        (lambda r: (r / "label_2/0.txt").write_text(""), "Unexpected KITTI entry"),
        (lambda r: (r / "image_2/000001.jpg").write_bytes(b""), "Unexpected KITTI"),
        (lambda r: (r / "label_2/000001.txt").mkdir(), "Unexpected KITTI entry"),
        (lambda r: _replace_with_file(r / "calib"), "calib in .* is not a directory"),
    ],
)
def test_discover_frames_rejects_inconsistent_output(tmp_path, mutate, match):
    root = tmp_path / "native"
    for directory, extension in (
        ("image_2", "png"),
        ("label_2", "txt"),
        ("calib", "txt"),
    ):
        (root / directory).mkdir(parents=True)
        for frame in (0, 1_000_000):
            (root / directory / f"{frame:06d}.{extension}").write_bytes(b"")
    adapter = KITTIFrameAdapter()
    assert adapter.discover_frames(root, KITTIWriter()) == [0, 1_000_000]
    mutate(root)
    with pytest.raises(PackagingError, match=match):
        adapter.discover_frames(root, KITTIWriter())


def test_export_rejects_depth_mismatch_and_empty_output(tmp_path, sample_render_vars):
    natives = _natives(tmp_path, sample_render_vars, {"a": [(0, False)]}, None)
    with pytest.raises(PackagingError, match="write_depth=True"):
        export_finalized_dataset(
            KITTIWriter(KITTIWriterConfig(write_depth=True)),
            natives,
            _packager(tmp_path / "out"),
        )
    empty = tmp_path / "empty"
    for directory in ("image_2", "label_2", "calib"):
        (empty / directory).mkdir(parents=True)
    with pytest.raises(PackagingError, match="No KITTI frames"):
        export_finalized_dataset(
            KITTIWriter(), {"a": empty}, _packager(tmp_path / "o2")
        )
    assert not (tmp_path / "out/ds/manifest.json").exists()


def test_collect_rejects_frames_not_captured(tmp_path, sample_render_vars):
    natives = _natives(
        tmp_path, sample_render_vars, {"a": [(0, False), (1, False)]}, None
    )
    context = AdapterContext("ds", "a", tmp_path, frame_ids=[0])
    with pytest.raises(PackagingError, match="do not match captured frames"):
        KITTIFrameAdapter().collect_samples(natives["a"], KITTIWriter(), context)


def test_sample_metadata_is_json(tmp_path, sample_render_vars):
    natives = _natives(tmp_path, sample_render_vars, {"a": [(7, False)]}, None)
    context = AdapterContext("ds", "a", tmp_path, frame_ids=[7])
    (sample,) = KITTIFrameAdapter().collect_samples(
        natives["a"], KITTIWriter(), context
    )
    assert sample.key == 7
    assert json.loads(json.dumps(sample.metadata)) == {
        "writer": "kitti",
        "frame_index": 7,
    }
