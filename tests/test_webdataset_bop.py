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

"""Package BOP frames as WebDataset samples and rebuild the native scenes."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import webdataset as wds

from ovannotators import (
    BOPFrameAdapter,
    BOPWriter,
    BOPWriterConfig,
    COCOWriter,
    ConfigurationError,
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    PackagingError,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
    export_finalized_dataset,
)
from ovannotators.capture_adapters import AdapterContext
from tests._webdataset_reconstruction import (
    EpisodeRecords,
    assert_trees_equal,
    compare_trees,
    read_packaged_dataset,
    reconstruct_native,
)

pytestmark = [pytest.mark.integration, pytest.mark.cpu_only]

READERS_PYTHON = os.environ.get(
    "OVANNOTATORS_READERS_PYTHON", "/tmp/readers_venv/bin/python"
)
BOP_LOADER = Path(__file__).parent / "dataset_loaders" / "load_bop.py"
CONFIG = {"scene_id": 3, "depth_scale": 10.0, "object_id_map": {"car": 42}}
ROLES = ("scene_camera", "scene_gt", "scene_gt_info")


def bop_shared_document(records: EpisodeRecords, role: str) -> bytes:
    """Rebuild a BOP scene document from per-frame ``json`` records."""
    document = {}
    for sample in records.samples:
        record = sample.json()
        if record[role] is not None:
            document[str(record["frame_id"])] = record[role]
    return json.dumps(document, indent=2).encode()


def bop_documents_equivalent(expected: bytes, actual: bytes) -> str | None:
    if json.loads(expected) == json.loads(actual):
        return None
    return "BOP scene documents differ"


BUILDERS = dict.fromkeys(ROLES, bop_shared_document)
COMPARATORS = {"*/scene_*.json": bop_documents_equivalent}


def _packager(root, **kwargs):
    return WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=root)),
        WebDatasetPackagerConfig(dataset_id="ds", **kwargs),
    )


@pytest.fixture
def frame_values(sample_render_vars):
    def values(frame):
        result = dict(sample_render_vars)
        color = sample_render_vars["LdrColor"][:120, :160].copy()
        color[..., 0] = frame
        result["LdrColor"] = color
        result["DistanceToCameraSD"] = sample_render_vars["DistanceToCameraSD"][
            :120, :160
        ] + np.float32(frame)
        result["SemanticInstanceSegmentation"] = sample_render_vars[
            "SemanticInstanceSegmentation"
        ][:120, :160]
        return result

    return values


def _natives(tmp_path, frames, values, **config):
    natives = {}
    for identity, ids in frames.items():
        natives[identity] = tmp_path / f"native-{identity}"
        disk = DiskBackend(DiskBackendConfig(root_dir=natives[identity]))
        writer = BOPWriter(BOPWriterConfig(**config))
        for frame in ids:
            writer.write_frame(frame, values(frame), disk)
        writer.finalize(disk)
    return natives


def _scene(root, name):
    return json.loads((root / "000003" / f"{name}.json").read_text())


def test_capture_restarted_frames_match_native_and_reconstruct(tmp_path, frame_values):
    frames = {"a": [0, 2, 5], "b": [0, 1]}
    with DatasetCapture(
        BOPWriter(BOPWriterConfig(**CONFIG)),
        _packager(
            tmp_path / "out", max_samples_per_shard=2, allow_episode_spanning=True
        ),
    ) as capture:
        for identity, ids in frames.items():
            with capture.episode(identity) as episode:
                for frame in ids:
                    episode.write_frame(frame, frame_values(frame))
    natives = _natives(tmp_path, frames, frame_values, **CONFIG)

    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    assert [s.key for s in dataset.samples] == [
        "ds/episode-a/sample-0",
        "ds/episode-a/sample-2",
        "ds/episode-a/sample-5",
        "ds/episode-b/sample-0",
        "ds/episode-b/sample-1",
    ]
    metadata = dataset.manifest["metadata"]
    assert metadata["sample_schema"] == "bop-frame-v1"
    assert metadata["writer"] == "bop"
    assert metadata["writer_config"]["depth_scale"] == 10.0
    assert metadata["object_ids"] == [2, 42]
    for sample in dataset.samples:
        native = natives[sample.episode_id] / "000003"
        stem = f"{int(sample.sample_id):06d}"
        assert sorted(sample.members) == [
            "depth.png",
            "json",
            "mask.000002.png",
            "mask.000042.png",
            "mask_visib.000002.png",
            "mask_visib.000042.png",
            "metadata.json",
            "rgb.png",
        ]
        assert sample.members["rgb.png"] == (native / f"rgb/{stem}.png").read_bytes()
        assert (
            sample.members["depth.png"] == (native / f"depth/{stem}.png").read_bytes()
        )
        assert (
            sample.members["mask_visib.000042.png"]
            == (native / f"mask_visib/{stem}_000042.png").read_bytes()
        )
        record = sample.json()
        assert (record["schema_version"], record["scene_id"]) == (1, 3)
        assert record["frame_id"] == int(sample.sample_id)
        for role in ROLES:
            assert (
                record[role]
                == _scene(natives[sample.episode_id], role)[sample.sample_id]
            )
        assert record["scene_camera"]["depth_scale"] == 10.0
        assert record["masks"] == [
            {
                "obj_id": 42,
                "mask": "mask.000042.png",
                "mask_visib": "mask_visib.000042.png",
            },
            {
                "obj_id": 2,
                "mask": "mask.000002.png",
                "mask_visib": "mask_visib.000002.png",
            },
        ]
        assert sample.metadata["metadata"] == {
            "writer": "bop",
            "frame_index": int(sample.sample_id),
            "scene_id": 3,
        }

    episodes = dataset.source_paths["episodes"]
    assert episodes["a"]["shared"] == {role: f"000003/{role}.json" for role in ROLES}
    assert episodes["a"]["samples"]["5"]["mask.000042.png"] == (
        "000003/mask/000005_000042.png"
    )
    roots = reconstruct_native(dataset, tmp_path / "rebuilt", shared_builders=BUILDERS)
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity], comparators=COMPARATORS)
        # The rebuilt documents also keep the writer's key order and layout.
        assert compare_trees(native, roots[identity]).changed == {}

    decoded = list(
        wds.WebDataset(
            [
                f"file:{(tmp_path / 'out' / s['path']).resolve().as_posix()}"
                for s in dataset.manifest["shards"]
            ],
            shardshuffle=False,
        ).decode("rgb8")
    )
    assert decoded[0]["rgb.png"].shape == (120, 160, 3)
    assert decoded[0]["mask.000042.png"].shape == (120, 160, 3)


def test_export_finalized_handles_optional_documents_and_masks(tmp_path, frame_values):
    natives = _natives(tmp_path, {"a": [0, 1], "b": [4]}, frame_values, **CONFIG)
    scene_a = natives["a"] / "000003"
    shutil.rmtree(scene_a / "mask")
    (scene_a / "mask_visib/000001_000002.png").unlink()
    (scene_a / "scene_gt_info.json").unlink()
    gt = _scene(natives["b"], "scene_gt")
    gt["4"][0]["obj_id"] = 2
    (natives["b"] / "000003/scene_gt.json").write_text(json.dumps(gt, indent=2))
    (natives["b"] / "000003/mask/000004_000042.png").unlink()
    (natives["b"] / "000003/mask_visib/000004_000042.png").unlink()

    writer = BOPWriter(BOPWriterConfig(**CONFIG))
    assert BOPFrameAdapter().discover_frames(natives["a"], writer) == [0, 1]
    export_finalized_dataset(writer, natives, _packager(tmp_path / "out"))

    dataset = read_packaged_dataset(tmp_path / "out", "ds")
    assert dataset.manifest["metadata"]["provenance"] == "finalized-export"
    assert dataset.manifest["metadata"]["object_ids"] == [2, 42]
    samples = {s.key: s for s in dataset.samples}
    frame1 = samples["ds/episode-a/sample-1"]
    assert sorted(frame1.members) == [
        "depth.png",
        "json",
        "mask_visib.000042.png",
        "metadata.json",
        "rgb.png",
    ]
    record = frame1.json()
    assert record["scene_gt_info"] is None
    assert record["masks"] == [
        {"obj_id": 42, "mask": None, "mask_visib": "mask_visib.000042.png"},
        {"obj_id": 2, "mask": None, "mask_visib": None},
    ]
    # Two GT entries share obj_id 2 and therefore one native mask file.
    shared = samples["ds/episode-b/sample-4"].json()["masks"]
    assert [m["mask"] for m in shared] == ["mask.000002.png", "mask.000002.png"]
    assert dataset.source_paths["episodes"]["a"]["shared"] == {
        "scene_camera": "000003/scene_camera.json",
        "scene_gt": "000003/scene_gt.json",
    }
    roots = reconstruct_native(dataset, tmp_path / "rebuilt", shared_builders=BUILDERS)
    for identity, native in natives.items():
        assert_trees_equal(native, roots[identity], comparators=COMPARATORS)


def _context(tmp_path, frames):
    generated = tmp_path / "generated"
    generated.mkdir(exist_ok=True)
    return AdapterContext("ds", "a", generated, frame_ids=frames)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: (s.parent / "notes.txt").write_text("x"), "outside any frame"),
        (lambda s: (s.parent / "000004").mkdir(), "Missing 000005/scene_camera"),
        (lambda s: (s / "depth/000000.png").unlink(), "Missing 000003/depth"),
        (
            lambda s: (s / "scene_camera.json").write_text('{"00": {}}'),
            "Unsupported BOP frame key",
        ),
        (lambda s: (s / "scene_camera.json").write_text("[]"), "not a frame mapping"),
        (lambda s: (s / "scene_camera.json").write_text("{}"), "No BOP frames"),
        (
            lambda s: (s / "scene_camera.json").write_text("{not json"),
            "scene_camera.json is not valid UTF-8 JSON",
        ),
        (
            lambda s: (s / "scene_gt.json").write_bytes(b"\xff\xfe"),
            "scene_gt.json is not valid UTF-8 JSON",
        ),
        (
            lambda s: (s / "scene_camera.json").write_text('{"0": 5}'),
            "frame 0 is not an object",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text('{"0": [5]}'),
            "frame 0 is not a list of objects",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text('{"0": {"obj_id": 1}}'),
            "frame 0 is not a list of objects",
        ),
        (
            lambda s: (s / "scene_gt_info.json").write_text('{"0": null}'),
            "scene_gt_info.json frame 0 is not a list of objects",
        ),
        (
            lambda s: (s / "scene_gt_info.json").write_text('{"0": [1]}'),
            "scene_gt_info.json frame 0 is not a list of objects",
        ),
        (
            lambda s: (s / "scene_gt_info.json").write_text('{"0": {"bbox_obj": []}}'),
            "scene_gt_info.json frame 0 is not a list of objects",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text('{"0": {}}'),
            "frame 0 is not a list of objects",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text('{"0": 0}'),
            "frame 0 is not a list of objects",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text("{}"),
            r"scene_gt.json has no entry for frames \[0\]",
        ),
        (
            lambda s: (s / "scene_gt_info.json").write_text("{}"),
            r"scene_gt_info.json has no entry for frames \[0\]",
        ),
        (
            lambda s: (s / "scene_camera.json").write_text(
                '{"0": {"depth_scale": 10.0}}'
            ),
            "frame 0 lacks cam_K",
        ),
        (
            lambda s: (s / "scene_camera.json").write_text(
                '{"0": {"cam_K": [1, 2], "depth_scale": 10.0}}'
            ),
            "frame 0 lacks cam_K",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text('{"0": [{"obj_id": -1}]}'),
            "has obj_id -1",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text('{"9": []}'),
            r"frames \[9\] are absent",
        ),
        (
            lambda s: (s / "scene_gt.json").write_text('{"0": [{"obj_id": "x"}]}'),
            "has obj_id 'x'",
        ),
        (
            lambda s: (s / "scene_gt_info.json").write_text(
                '{"0": [{"visib_fract": NaN}]}'
            ),
            "non-finite",
        ),
    ],
)
def test_collect_rejects_inconsistent_native_output(
    tmp_path, frame_values, mutate, message
):
    root = _natives(tmp_path, {"a": [0]}, frame_values, **CONFIG)["a"]
    mutate(root / "000003")
    config = dict(CONFIG, scene_id=5) if "000005" in message else CONFIG
    writer = BOPWriter(BOPWriterConfig(**config))
    with pytest.raises(PackagingError, match=message):
        BOPFrameAdapter().collect_samples(root, writer, _context(tmp_path, [0]))


def test_writer_configuration_must_match_native_output(tmp_path, frame_values):
    root = _natives(tmp_path, {"a": [0, 1]}, frame_values, **CONFIG)["a"]
    adapter = BOPFrameAdapter()
    mismatched = BOPWriter(BOPWriterConfig(scene_id=3, depth_scale=1.0))
    with pytest.raises(PackagingError, match="depth_scale 10.0 does not match"):
        adapter.discover_frames(root, mismatched)
    writer = BOPWriter(BOPWriterConfig(**CONFIG))
    with pytest.raises(PackagingError, match="do not match the captured frames"):
        adapter.collect_samples(root, writer, _context(tmp_path, [0]))
    with pytest.raises(PackagingError, match="no rgb image for frame 7"):
        adapter.validate_frame_output(7, writer, root, _context(tmp_path, []))
    with pytest.raises(ConfigurationError, match="requires a BOPWriter"):
        DatasetCapture(COCOWriter(), _packager(tmp_path / "out"), adapter=adapter)


def test_failed_episode_keeps_dataset_object_ids(tmp_path, frame_values):
    with DatasetCapture(
        BOPWriter(BOPWriterConfig(**CONFIG)), _packager(tmp_path / "out")
    ) as capture:
        with capture.episode("a") as episode:
            episode.write_frame(0, frame_values(0))
        values = frame_values(1)
        values["SemanticBoundingBox3D"] = values["SemanticBoundingBox3D"].copy()
        values["SemanticBoundingBox3D"]["transform"][0, 0, 3] = np.nan
        with pytest.raises(PackagingError, match="non-finite"):
            with capture.episode("b") as episode:
                episode.write_frame(1, values)
    manifest = json.loads((tmp_path / "out/ds/manifest.json").read_text())
    assert manifest["episodes"] == ["a"]
    assert manifest["metadata"]["object_ids"] == [2, 42]


def _bop_reader_available():
    if not Path(READERS_PYTHON).is_file():
        return False
    probe = subprocess.run(
        [READERS_PYTHON, "-c", "import bop_toolkit_lib.inout"],
        capture_output=True,
        timeout=120,
    )
    return probe.returncode == 0


@pytest.mark.skipif(
    not _bop_reader_available(),
    reason=f"bop_toolkit_lib reader interpreter not found at {READERS_PYTHON}; "
    "set OVANNOTATORS_READERS_PYTHON",
)
def test_reconstructed_bop_passes_native_loader(tmp_path, frame_values):
    natives = _natives(tmp_path, {"a": [0, 2]}, frame_values, **CONFIG)
    export_finalized_dataset(
        BOPWriter(BOPWriterConfig(**CONFIG)), natives, _packager(tmp_path / "out")
    )
    roots = reconstruct_native(
        read_packaged_dataset(tmp_path / "out", "ds"),
        tmp_path / "rebuilt",
        shared_builders=BUILDERS,
    )
    outputs = []
    for root in (natives["a"], roots["a"]):
        result = subprocess.run(
            [READERS_PYTHON, str(BOP_LOADER), str(root)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.replace(str(root), "<root>"))
    assert outputs[0] == outputs[1]
    assert "frames      : [0, 2]" in outputs[1]
    assert "GT objects: 2  mask files: 2  mask_visib files: 2" in outputs[1]
