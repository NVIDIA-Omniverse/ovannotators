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

"""Verify WebDataset interoperability, shard policies and managed capture."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import webdataset as wds
from PIL import Image

from ovannotators import (
    BasicFrameAdapter,
    BasicWriter,
    BasicWriterConfig,
    COCOWriter,
    COCOWriterConfig,
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    SampleFiles,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
    config_from_dict,
    config_to_dict,
    export_finalized_dataset,
)
from ovannotators.exceptions import (
    BackendWriteError,
    ConfigurationError,
    MissingRenderVarError,
    PackagingError,
)
from ovannotators.writers._helpers import OVRTX_BBOX2D_DTYPE

pytestmark = [pytest.mark.integration, pytest.mark.cpu_only]


def _backend(tmp_path, **kwargs):
    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / "out"))
    return WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="test", **kwargs)
    )


def _samples(tmp_path, count=2, size=10):
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * size)
    return [SampleFiles(i, {"bin": source}) for i in range(count)]


def _file_url(path):
    # webdataset opens URLs; a bare Windows drive path reads as a URL scheme.
    return f"file:{path.resolve().as_posix()}"


def _manifest(tmp_path):
    return json.loads((tmp_path / "out/test/manifest.json").read_text())


def _members(tmp_path):
    result = {}
    for path in sorted((tmp_path / "out/test").glob("*.tar")):
        with tarfile.open(path) as archive:
            for member in archive:
                assert member.name not in result
                result[member.name] = archive.extractfile(member).read()
    return result


@pytest.mark.parametrize(
    "size, expected",
    [
        ("1 GB", 10**9),
        ("1 GiB", 2**30),
        ("1.5 MiB", 1572864),
        (" 20 KB ", 20000),
        (10240, 10240),
    ],
)
def test_human_sizes_and_config_roundtrip(size, expected):
    cfg = WebDatasetPackagerConfig(max_shard_bytes=size, dataset_id="test")
    assert cfg.max_shard_bytes == expected
    assert config_from_dict(WebDatasetPackagerConfig, config_to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_shard_bytes": "1"},
        {"max_shard_bytes": "1 gb"},
        {"max_shard_bytes": "1.0001 B"},
        {"max_shard_bytes": -1},
        {"max_shard_bytes": True},
        {"max_shard_bytes": 10.0},
        {"max_samples_per_shard": 0},
        {"max_samples_per_shard": True},
        {"max_episodes_per_shard": -1},
        {"allow_episode_spanning": 1},
        {"oversized_episode_policy": "ignore"},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ConfigurationError):
        WebDatasetPackagerConfig(**kwargs)


def test_episode_preservation_count_rotation_and_collision_free_extraction(tmp_path):
    backend = _backend(tmp_path, max_samples_per_shard=3)
    with backend:
        backend.pack_episode("first", _samples(tmp_path))
        backend.pack_episode("second", _samples(tmp_path))
    manifest = _manifest(tmp_path)
    assert [s["samples"] for s in manifest["shards"]] == [2, 2]
    assert [s["episodes"] for s in manifest["shards"]] == [["first"], ["second"]]
    members = _members(tmp_path)
    assert len(members) == 8
    destination = tmp_path / "extracted"
    for path in (tmp_path / "out/test").glob("*.tar"):
        with tarfile.open(path) as archive:
            for member in archive:
                target = destination / member.name
                assert not target.exists()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    assert (destination / "test/episode-first/sample-0.bin").read_bytes() == b"x" * 10
    assert manifest["samples"] == 4
    backend.finalize()
    assert len(list((tmp_path / "out/test").glob("*.tar"))) == 2
    with pytest.raises(PackagingError, match="complete"):
        backend.pack_episode("third", [])


def test_episode_count_rotation(tmp_path):
    backend = _backend(tmp_path, max_episodes_per_shard=1)
    with backend:
        backend.pack_episode("a", _samples(tmp_path, 1))
        backend.pack_episode("b", _samples(tmp_path, 1))
    assert [s["episodes"] for s in _manifest(tmp_path)["shards"]] == [["a"], ["b"]]


def test_exact_byte_limit_includes_headers_and_record_padding(tmp_path):
    backend = _backend(tmp_path, max_shard_bytes=10240)
    with backend:
        backend.pack_episode("a", _samples(tmp_path, 1, 6500))
        backend.pack_episode("b", _samples(tmp_path, 1, 6500))
    assert [s["bytes"] for s in _manifest(tmp_path)["shards"]] == [10240, 10240]
    for path in (tmp_path / "out/test").glob("*.tar"):
        assert path.stat().st_size == 10240


def test_spanning_rotates_between_complete_samples(tmp_path):
    backend = _backend(tmp_path, max_samples_per_shard=2, allow_episode_spanning=True)
    with backend:
        backend.pack_episode("a", _samples(tmp_path, 5))
    assert [s["samples"] for s in _manifest(tmp_path)["shards"]] == [2, 2, 1]
    assert len(_members(tmp_path)) == 10


@pytest.mark.parametrize("spanning", [False, True])
def test_oversized_byte_policy(tmp_path, spanning):
    samples = _samples(tmp_path, 1, 20000)
    backend = _backend(tmp_path, max_shard_bytes=10240, allow_episode_spanning=spanning)
    with pytest.raises(PackagingError, match="exceeds shard limits"):
        backend.pack_episode("a", samples)
    backend.abort()
    assert not (tmp_path / "out/test/manifest.json").exists()
    backend = _backend(
        tmp_path,
        max_shard_bytes=10240,
        allow_episode_spanning=spanning,
        oversized_episode_policy="dedicated_shard",
    )
    with backend:
        backend.pack_episode("a", samples)
        backend.pack_episode("b", _samples(tmp_path, 1))
    shards = _manifest(tmp_path)["shards"]
    assert [s["oversized"] for s in shards] == [True, False]
    assert shards[0]["bytes"] > 10240
    assert shards[1]["episodes"] == ["b"]


def test_preservation_sample_limit_is_not_silently_ignored(tmp_path):
    backend = _backend(tmp_path, max_samples_per_shard=1)
    with pytest.raises(PackagingError):
        backend.pack_episode("a", _samples(tmp_path, 2))
    backend.abort()
    backend = _backend(
        tmp_path, max_samples_per_shard=1, oversized_episode_policy="dedicated_shard"
    )
    with backend:
        backend.pack_episode("a", _samples(tmp_path, 2))
    assert _manifest(tmp_path)["shards"][0]["oversized"] is True


def test_duplicate_identity_and_rejected_episode_can_be_retried(tmp_path):
    backend = _backend(tmp_path)
    samples = _samples(tmp_path, 1)
    with pytest.raises(PackagingError, match="Duplicate sample"):
        backend.pack_episode("a", samples * 2)
    with backend:
        backend.pack_episode("a", samples)
        with pytest.raises(PackagingError, match="Duplicate episode"):
            backend.pack_episode("a", samples)
    assert _manifest(tmp_path)["samples"] == 1
    with pytest.raises(PackagingError, match="namespace already exists"):
        _backend(tmp_path)


@pytest.mark.parametrize(
    "identifier", ["../bad", "a.b", "Upper", "", "a" * 41, True, -1]
)
def test_unsafe_identifiers(tmp_path, identifier):
    backend = _backend(tmp_path)
    with pytest.raises(PackagingError):
        backend.pack_episode(identifier, [])
    backend.abort()


@pytest.mark.parametrize("suffix", ["../bin", "a/b", "metadata.json", "UPPER", ""])
def test_unsafe_or_reserved_component_suffix(tmp_path, suffix):
    backend = _backend(tmp_path)
    sample = _samples(tmp_path, 1)[0]
    with pytest.raises(PackagingError):
        backend.pack_episode("a", [SampleFiles(0, {suffix: sample.files["bin"]})])
    backend.abort()


def test_stage_copies_bytes_and_metadata_before_source_changes(tmp_path):
    backend = _backend(tmp_path)
    sample = _samples(tmp_path, 1)[0]
    with backend:
        backend.pack_episode("a", [SampleFiles(0, sample.files, {"units": "meters"})])
        Path(sample.files["bin"]).write_bytes(b"changed")
    members = _members(tmp_path)
    assert members["test/episode-a/sample-0.bin"] == b"x" * 10
    assert json.loads(members["test/episode-a/sample-0.metadata.json"])["metadata"] == {
        "units": "meters"
    }


def test_failed_publication_aborts_without_manifest(tmp_path, monkeypatch):
    backend = _backend(tmp_path, max_samples_per_shard=1)
    original = backend.storage.write

    def fail(path, payload):
        if "shard-000001" in path:
            raise BackendWriteError(path=path, message="injected failure")
        original(path, payload)

    monkeypatch.setattr(backend.storage, "write", fail)
    with pytest.raises(BackendWriteError):
        with backend:
            backend.pack_episode("a", _samples(tmp_path, 1))
            backend.pack_episode("b", _samples(tmp_path, 1))
    assert (tmp_path / "out/test/shard-000000.tar").exists()
    assert not (tmp_path / "out/test/manifest.json").exists()
    assert not backend._root.exists()


def _render_vars(label_id=1):
    boxes = np.zeros(1, dtype=OVRTX_BBOX2D_DTYPE)
    boxes[0]["instance_id"] = (1, 0, 0, 0)
    boxes[0]["semantic_id"] = (label_id, 0, 0, 0)
    boxes[0]["x_max"] = 8
    boxes[0]["y_max"] = 6
    return {
        "LdrColor": np.full((6, 8, 3), 127, dtype=np.uint8),
        "SemanticBoundingBox2DTight": boxes,
    }


def test_coco_capture_official_reader_and_stable_categories(tmp_path):
    backend = _backend(tmp_path, max_episodes_per_shard=1)
    with DatasetCapture(
        COCOWriter(COCOWriterConfig(image_format="png")), backend
    ) as dataset:
        with dataset.episode("a") as episode:
            episode.write_frame(0, _render_vars(1))
        with dataset.episode("b") as episode:
            episode.write_frame(0, _render_vars(2))
            episode.write_frame(2, _render_vars(1))
    manifest = _manifest(tmp_path)
    urls = [_file_url(tmp_path / "out" / s["path"]) for s in manifest["shards"]]
    samples = list(wds.WebDataset(urls, shardshuffle=False).decode("pil"))
    assert len(samples) == 3
    assert len({s["__key__"] for s in samples}) == 3
    assert all(s["png"].size == (8, 6) for s in samples)
    assert [s["json"]["annotations"][0]["category_id"] for s in samples] == [1, 2, 1]
    assert [s["metadata.json"]["metadata"]["frame_index"] for s in samples] == [0, 0, 2]
    assert all(
        s["json"]["image"]["file_name"] == s["__key__"] + ".png" for s in samples
    )
    assert samples[0]["json"]["annotations"][0]["bbox"] == [0.0, 0.0, 8.0, 6.0]
    assert len(manifest["metadata"]["categories"]) == 2
    image_ids = [s["json"]["image"]["id"] for s in samples]
    annotation_ids = [s["json"]["annotations"][0]["id"] for s in samples]
    assert len(set(image_ids)) == 3 and len(set(annotation_ids)) == 3
    assert all(
        s["json"]["annotations"][0]["image_id"] == s["json"]["image"]["id"]
        for s in samples
    )
    dataset.finalize()


def test_coco_panoptic_samples_remap_categories(tmp_path):
    backend = _backend(tmp_path)
    writer = COCOWriter(
        COCOWriterConfig(include_panoptic=True, include_segmentation=True)
    )
    with DatasetCapture(writer, backend) as dataset:
        for identity, label in [("a", 1), ("b", 2)]:
            with dataset.episode(identity) as episode:
                values = _render_vars(label)
                values["SemanticInstanceSegmentation"] = np.ones(
                    (6, 8), dtype=np.uint32
                )
                episode.write_frame(0, values)
    samples = list(
        wds.WebDataset(
            _file_url(tmp_path / "out/test/shard-000000.tar"), shardshuffle=False
        ).decode("pil")
    )
    assert [
        s["json"]["panoptic"]["segments_info"][0]["category_id"] for s in samples
    ] == [1, 2]
    assert all(s["panoptic.png"].size == (8, 6) for s in samples)
    assert all(
        s["json"]["annotations"][0]["segmentation"]["size"] == [6, 8] for s in samples
    )


def test_basic_capture_numeric_fidelity_and_unchanged_disk_lifecycle(tmp_path):
    values = {
        "LdrColor": np.zeros((3, 4, 3), dtype=np.uint8),
        "DistanceToCameraSD": np.array([[1.25, np.inf]], dtype=np.float32),
    }
    writer = BasicWriter(BasicWriterConfig(depth_format="npy", colorize=False))
    backend = _backend(tmp_path)
    with DatasetCapture(writer, backend) as dataset:
        with dataset.episode("a") as episode:
            episode.write_frame(0, values)
    members = _members(tmp_path)
    assert np.array_equal(
        np.load(io.BytesIO(members["test/episode-a/sample-0.distance_to_camera.npy"])),
        values["DistanceToCameraSD"],
    )
    image = Image.open(io.BytesIO(members["test/episode-a/sample-0.ldr_color.png"]))
    assert image.size == (4, 3)
    disk = DiskBackend(DiskBackendConfig(root_dir=tmp_path / "plain"))
    writer.write_frame(0, values, disk)
    assert (tmp_path / "plain/0000/ldr_color.png").exists()


@pytest.mark.parametrize(
    "config, values",
    [
        (BasicWriterConfig(), {}),
        (
            BasicWriterConfig(render_vars=[]),
            {"LdrColor": np.zeros((3, 4, 3), dtype=np.uint8)},
        ),
        (
            BasicWriterConfig(render_vars=["DepthSD"]),
            {"LdrColor": np.zeros((3, 4, 3), dtype=np.uint8)},
        ),
        (
            BasicWriterConfig(colorize=False),
            {"DepthSD": np.empty((0, 0), dtype=np.float32)},
        ),
    ],
)
def test_basic_empty_frame_fails_during_write_and_discards_episode(
    tmp_path, config, values
):
    backend = _backend(tmp_path)
    with pytest.raises(PackagingError) as exited:
        with DatasetCapture(BasicWriter(config), backend) as dataset:
            with dataset.episode("empty") as episode:
                with pytest.raises(PackagingError) as raised:
                    episode.write_frame(7, values)
                assert str(raised.value) == (
                    "BasicWriter produced no files for frame 7 in episode empty; "
                    "check render_vars and the writer selection"
                )
                assert episode._frames == []
                assert backend.storage.list() == []
                assert not Path(episode._tmp.name).exists()
    assert str(exited.value) == "Cannot complete failed episode empty"
    assert not backend._root.exists()


def test_capture_exception_does_not_publish_episode_or_manifest(tmp_path):
    backend = _backend(tmp_path)
    dataset = DatasetCapture(COCOWriter(), backend)
    with pytest.raises(RuntimeError):
        with dataset:
            with dataset.episode("a") as episode:
                episode.write_frame(0, _render_vars())
                raise RuntimeError("producer failed")
    assert backend.storage.list() == []
    assert not Path(episode._tmp.name).exists()
    assert not backend._root.exists()


def test_capture_nested_duplicate_indices_and_context_reuse(tmp_path):
    backend = _backend(tmp_path)
    with DatasetCapture(COCOWriter(), backend) as dataset:
        with dataset.episode("a") as episode:
            episode.write_frame(0, _render_vars())
            with pytest.raises(PackagingError, match="strictly increasing"):
                episode.write_frame(0, _render_vars())
            with pytest.raises(PackagingError):
                dataset.episode("nested")
        with pytest.raises(PackagingError, match="Duplicate episode"):
            dataset.episode("a")
    with pytest.raises(PackagingError):
        with dataset:
            pass
    with pytest.raises(PackagingError):
        episode.write_frame(1, _render_vars())


def test_unsupported_writer_and_video_are_rejected(tmp_path):
    backend = _backend(tmp_path)
    with pytest.raises(ConfigurationError, match="exact built-in writer types") as info:
        DatasetCapture(object(), backend)
    # Every registered writer is named, so adding an adapter needs no edit here.
    assert "BasicWriter" in str(info.value) and "COCOWriter" in str(info.value)
    with pytest.raises(ConfigurationError, match="video"):
        DatasetCapture(BasicWriter(BasicWriterConfig(video=True)), backend)
    backend.abort()


def test_independent_dataset_namespaces_extract_without_collisions(tmp_path):
    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / "out"))
    samples = _samples(tmp_path, 1)
    members = set()
    for identity in ("first", "second"):
        with WebDatasetPackager(
            storage, WebDatasetPackagerConfig(dataset_id=identity)
        ) as backend:
            backend.pack_episode("a", samples)
        with tarfile.open(tmp_path / "out" / identity / "shard-000000.tar") as archive:
            names = set(archive.getnames())
            assert not members.intersection(names)
            members.update(names)
    assert len(members) == 4


@pytest.mark.parametrize("identifier", ["con", "aux", "com1", "lpt9"])
def test_windows_reserved_identifiers(tmp_path, identifier):
    backend = _backend(tmp_path)
    with pytest.raises(PackagingError, match="Windows"):
        backend.pack_episode(identifier, [])
    backend.abort()


def test_trailing_dot_component_is_rejected(tmp_path):
    backend = _backend(tmp_path)
    sample = _samples(tmp_path, 1)[0]
    with pytest.raises(PackagingError, match="suffix"):
        backend.pack_episode("a", [SampleFiles(0, {"bin.": sample.files["bin"]})])
    backend.abort()


def test_nonfinite_metadata_and_missing_source_do_not_commit_episode(tmp_path):
    backend = _backend(tmp_path)
    sample = _samples(tmp_path, 1)[0]
    with pytest.raises(ValueError):
        backend.pack_episode(
            "a", [SampleFiles(0, sample.files, {"value": float("nan")})]
        )
    with pytest.raises(FileNotFoundError):
        backend.pack_episode("a", [SampleFiles(0, {"bin": tmp_path / "missing"})])
    with backend:
        backend.pack_episode("a", [sample])
    assert _manifest(tmp_path)["samples"] == 1


def test_writer_failure_aborts_even_if_producer_catches_it(tmp_path):
    backend = _backend(tmp_path)
    with pytest.raises(PackagingError, match="failed episode a"):
        with DatasetCapture(COCOWriter(), backend) as dataset:
            with dataset.episode("a") as episode:
                with pytest.raises(MissingRenderVarError):
                    episode.write_frame(0, {})
    assert backend.storage.list() == []
    assert not backend._root.exists()
    assert not Path(episode._tmp.name).exists()


def test_plain_backend_cannot_be_used_as_packaging_session(tmp_path):
    with pytest.raises(ConfigurationError, match="WebDatasetPackager"):
        DatasetCapture(COCOWriter(), DiskBackend(DiskBackendConfig(tmp_path)))


def test_object_destination_receives_complete_shards_and_manifest_last(tmp_path):
    import hashlib

    class ObjectDestination:
        def __init__(self):
            self.objects = {}
            self.order = []

        def list(self, prefix=""):
            return [name for name in self.objects if name.startswith(prefix)]

        def exists(self, path):
            return path in self.objects

        def write(self, path, data):
            assert isinstance(data, bytes)
            self.objects[path] = data
            self.order.append(path)

    destination = ObjectDestination()
    with WebDatasetPackager(
        destination, WebDatasetPackagerConfig(dataset_id="objects")
    ) as backend:
        backend.pack_episode("a", _samples(tmp_path, 1))
    assert destination.order == ["objects/shard-000000.tar", "objects/manifest.json"]
    manifest = json.loads(destination.objects["objects/manifest.json"])
    shard = destination.objects["objects/shard-000000.tar"]
    assert manifest["shards"][0]["sha256"] == hashlib.sha256(shard).hexdigest()
    assert manifest["shards"][0]["bytes"] == len(shard)
    with tarfile.open(fileobj=io.BytesIO(shard)) as archive:
        assert archive.extractfile("objects/episode-a/sample-0.bin").read() == b"x" * 10


def test_manifest_failure_does_not_claim_completion(tmp_path, monkeypatch):
    backend = _backend(tmp_path)
    original = backend.storage.write

    def fail(path, payload):
        if path.endswith("manifest.json"):
            raise BackendWriteError(path=path, message="manifest failure")
        original(path, payload)

    monkeypatch.setattr(backend.storage, "write", fail)
    with pytest.raises(BackendWriteError, match="manifest failure"):
        with backend:
            backend.pack_episode("a", _samples(tmp_path, 1))
    assert (tmp_path / "out/test/shard-000000.tar").exists()
    assert not (tmp_path / "out/test/manifest.json").exists()
    assert not backend._root.exists()


def test_namespace_member_names_fit_ustar_at_maximum_lengths(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / "out"))
    with WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="d" * 40)
    ) as backend:
        backend.pack_episode("e" * 40, [SampleFiles("s" * 40, {"f" * 40: source})])
    shard = tmp_path / "out" / ("d" * 40) / "shard-000000.tar"
    with tarfile.open(shard) as archive:
        assert len(archive.getmembers()) == 2
        assert archive.extractfile(archive.getmembers()[0]).read() == b"payload"


def test_packaging_errors_have_plain_messages(tmp_path):
    backend = _backend(tmp_path)
    backend.pack_episode("a", _samples(tmp_path, 1))
    with pytest.raises(PackagingError) as raised:
        backend.pack_episode("a", _samples(tmp_path, 1))
    assert str(raised.value) == "Duplicate episode: a"
    with pytest.raises(PackagingError) as raised:
        backend.pack_episode("b", [SampleFiles(0, {"x:y": tmp_path / "source.bin"})])
    assert str(raised.value) == "Invalid or reserved component suffix: 'x:y'"
    backend.abort()


def test_empty_episode_is_rejected_and_packager_stays_open(tmp_path):
    backend = _backend(tmp_path)
    with backend:
        with pytest.raises(PackagingError, match="Episode a contains no samples"):
            backend.pack_episode("a", [])
        backend.pack_episode("a", _samples(tmp_path, 1))
    assert _manifest(tmp_path)["episodes"] == ["a"]


def test_capture_empty_episode_is_rejected_and_capture_continues(tmp_path):
    backend = _backend(tmp_path)
    with DatasetCapture(COCOWriter(), backend) as dataset:
        with pytest.raises(PackagingError, match="Episode a contains no frames"):
            with dataset.episode("a"):
                pass
        with dataset.episode("b") as episode:
            episode.write_frame(0, _render_vars())
    assert _manifest(tmp_path)["episodes"] == ["b"]


def test_capture_retries_episode_rejected_by_shard_limits(tmp_path):
    backend = _backend(tmp_path, max_samples_per_shard=1)
    writer = BasicWriter(BasicWriterConfig())
    values = {"LdrColor": np.zeros((3, 4, 3), dtype=np.uint8)}
    with DatasetCapture(writer, backend) as dataset:
        with dataset.episode("first") as episode:
            episode.write_frame(0, values)
        with pytest.raises(PackagingError, match="exceeds shard limits"):
            with dataset.episode("big") as episode:
                episode.write_frame(0, values)
                episode.write_frame(1, values)
        with dataset.episode("big") as episode:
            episode.write_frame(0, values)
    manifest = _manifest(tmp_path)
    assert manifest["episodes"] == ["big", "first"]
    assert [s["episodes"] for s in manifest["shards"]] == [["first"], ["big"]]


def test_coco_rejected_episode_retry_leaves_contiguous_ids_and_no_orphans(tmp_path):
    backend = _backend(tmp_path, max_samples_per_shard=1)
    with DatasetCapture(COCOWriter(), backend) as dataset:
        with dataset.episode("first") as episode:
            episode.write_frame(0, _render_vars(1))
        with pytest.raises(PackagingError, match="exceeds shard limits"):
            with dataset.episode("big") as episode:
                episode.write_frame(0, _render_vars(2))
                episode.write_frame(1, _render_vars(2))
        state = dataset._session.state
        assert state["categories"] == {"1": 1}
        assert state["next_image_id"] == 2
        assert state["next_annotation_id"] == 2
        with dataset.episode("big") as episode:
            episode.write_frame(0, _render_vars(1))
    urls = [
        _file_url(tmp_path / "out" / s["path"]) for s in _manifest(tmp_path)["shards"]
    ]
    samples = list(wds.WebDataset(urls, shardshuffle=False).decode("pil"))
    assert [s["json"]["image"]["id"] for s in samples] == [1, 2]
    assert [s["json"]["annotations"][0]["id"] for s in samples] == [1, 2]
    assert len(_manifest(tmp_path)["metadata"]["categories"]) == 1


def test_capture_continues_after_producer_handles_episode_error(tmp_path):
    backend = _backend(tmp_path)
    with DatasetCapture(COCOWriter(), backend) as dataset:
        with pytest.raises(RuntimeError):
            with dataset.episode("a") as episode:
                episode.write_frame(0, _render_vars())
                raise RuntimeError("skip this episode")
        assert not Path(episode._tmp.name).exists()
        with dataset.episode("a") as episode:
            episode.write_frame(0, _render_vars())
    assert _manifest(tmp_path)["samples"] == 1


def test_basic_non_portable_file_name_fails_during_write(tmp_path):
    backend = _backend(tmp_path)
    writer = BasicWriter(BasicWriterConfig())
    values = {"Custom:Aov": np.zeros((2, 2), dtype=np.float32)}
    with DatasetCapture(writer, backend) as dataset:
        with pytest.raises(PackagingError, match="custom:aov.npy"):
            with dataset.episode("a") as episode:
                episode.write_frame(0, values)
        with dataset.episode("a") as episode:
            episode.write_frame(0, {"LdrColor": np.zeros((3, 4, 3), dtype=np.uint8)})
    assert _manifest(tmp_path)["samples"] == 1


def test_member_size_limit_is_checked_before_staging(tmp_path, monkeypatch):
    from ovannotators.backends import webdataset

    monkeypatch.setattr(webdataset, "_MAX_MEMBER_BYTES", 5)
    backend = _backend(tmp_path, oversized_episode_policy="dedicated_shard")
    with pytest.raises(PackagingError, match="8 GiB USTAR member limit"):
        backend.pack_episode("a", _samples(tmp_path, 1, 6))
    assert backend.is_open
    backend.abort()


def test_invalid_manifest_metadata_keeps_packager_open(tmp_path):
    backend = _backend(tmp_path)
    backend.pack_episode("a", _samples(tmp_path, 1))
    with pytest.raises(PackagingError, match="finite JSON"):
        backend.finalize({"value": object()})
    assert backend.is_open
    backend.finalize({"value": 1})
    assert _manifest(tmp_path)["metadata"] == {"value": 1}


def test_manifest_records_generator_and_per_episode_counts(tmp_path):
    backend = _backend(tmp_path, max_samples_per_shard=3, allow_episode_spanning=True)
    with backend:
        backend.pack_episode("a", _samples(tmp_path, 2))
        backend.pack_episode("b", _samples(tmp_path, 2))
    manifest = _manifest(tmp_path)
    assert manifest["generator"]["name"] == "ovannotators"
    assert manifest["generator"]["version"]
    assert [s["episode_samples"] for s in manifest["shards"]] == [
        {"a": 2, "b": 1},
        {"b": 1},
    ]


def _coco_native(tmp_path, identity, label, **config):
    root = tmp_path / f"native-{identity}"
    disk = DiskBackend(DiskBackendConfig(root_dir=root))
    writer = COCOWriter(COCOWriterConfig(image_format="png", **config))
    values = _render_vars(label)
    if config.get("include_panoptic"):
        values["SemanticInstanceSegmentation"] = np.ones((6, 8), dtype=np.uint32)
    writer.write_frame(0, values, disk)
    writer.finalize(disk)
    return root


def _source_paths(root, manifest):
    return json.loads((root / manifest["auxiliary"][0]["path"]).read_bytes())


def test_finalized_basic_export_packages_native_files_and_paths(tmp_path):
    values = {"LdrColor": np.zeros((3, 4, 3), dtype=np.uint8)}
    writer = BasicWriter(BasicWriterConfig())
    native = tmp_path / "native"
    disk = DiskBackend(DiskBackendConfig(root_dir=native))
    writer.write_frame(7, values, disk)
    writer.finalize(disk)
    export_finalized_dataset(writer, {"a": native}, _backend(tmp_path))
    manifest = _manifest(tmp_path)
    assert manifest["metadata"]["sample_schema"] == "basic-frame-v1"
    assert manifest["metadata"]["provenance"] == "finalized-export"
    assert manifest["samples"] == 1
    reference = manifest["auxiliary"][0]
    payload = (tmp_path / "out" / reference["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
    paths = json.loads(payload)["episodes"]
    assert paths["a"]["samples"]["7"]["ldr_color.png"] == "0007/ldr_color.png"
    assert (
        _members(tmp_path)["test/episode-a/sample-7.ldr_color.png"]
        == (native / "0007/ldr_color.png").read_bytes()
    )


def test_finalized_coco_export_reconciles_categories_and_keeps_sources(tmp_path):
    writer = COCOWriter(COCOWriterConfig(image_format="png"))
    episodes = {
        "a": _coco_native(tmp_path, "a", 1),
        "b": _coco_native(tmp_path, "b", 2),
    }

    def files():
        return {
            identity: sorted(
                (path.relative_to(root).as_posix(), path.read_bytes())
                for path in root.rglob("*")
                if path.is_file()
            )
            for identity, root in episodes.items()
        }

    before = files()
    export_finalized_dataset(writer, episodes, _backend(tmp_path))
    assert files() == before
    manifest = _manifest(tmp_path)
    assert [c["name"] for c in manifest["metadata"]["categories"]] == ["1", "2"]
    record = json.loads(_members(tmp_path)["test/episode-b/sample-0.json"])
    assert record["annotations"][0]["category_id"] == 2
    assert record["image"]["id"] == 2
    paths = _source_paths(tmp_path / "out", manifest)["episodes"]
    assert paths["b"]["samples"]["0"] == {"png": "images/000000.png"}
    assert paths["b"]["shared"] == {"instances": "annotations/instances_train.json"}


def test_capture_and_finalized_export_emit_equivalent_coco_datasets(tmp_path):
    writer = COCOWriter(COCOWriterConfig(image_format="png"))
    frames = {"a": [(3, 1), (5, 2)], "b": [(0, 2)]}
    capture_root = tmp_path / "captured"
    captured = WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=capture_root)),
        WebDatasetPackagerConfig(dataset_id="same"),
    )
    with DatasetCapture(writer, captured) as dataset:
        for identity, items in frames.items():
            with dataset.episode(identity) as episode:
                for frame, label in items:
                    episode.write_frame(frame, _render_vars(label))

    natives = {}
    for identity, items in frames.items():
        natives[identity] = tmp_path / f"native-{identity}"
        disk = DiskBackend(DiskBackendConfig(root_dir=natives[identity]))
        native_writer = COCOWriter(COCOWriterConfig(image_format="png"))
        for frame, label in items:
            native_writer.write_frame(frame, _render_vars(label), disk)
        native_writer.finalize(disk)
    exported_root = tmp_path / "exported"
    exported = WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=exported_root)),
        WebDatasetPackagerConfig(dataset_id="same"),
    )
    export_finalized_dataset(writer, natives, exported)

    def members(root):
        with tarfile.open(root / "same/shard-000000.tar") as archive:
            return {
                member.name: archive.extractfile(member).read() for member in archive
            }

    def manifest(root):
        data = json.loads((root / "same/manifest.json").read_text())
        data["metadata"].pop("provenance")
        for shard in data["shards"]:
            shard.pop("sha256")
        data.pop("auxiliary")
        return data

    def paths(root):
        data = _source_paths(
            root, json.loads((root / "same/manifest.json").read_text())
        )
        data.pop("provenance")
        return data

    assert members(capture_root) == members(exported_root)
    assert len(members(capture_root)) == 9
    assert manifest(capture_root) == manifest(exported_root)
    assert paths(capture_root) == paths(exported_root)


def test_export_rejects_unsupported_writer_before_publication(tmp_path):
    backend = _backend(tmp_path)
    with pytest.raises(ConfigurationError, match="adapter"):
        export_finalized_dataset(object(), {}, backend)
    assert backend.storage.list() == []
    backend.abort()


def test_auxiliary_failure_cannot_publish_completion(tmp_path, monkeypatch):
    backend = _backend(tmp_path)
    original = backend.storage.write

    def fail_auxiliary(path, payload):
        if path.endswith("source-paths.json"):
            raise BackendWriteError(path=path, message="auxiliary failure")
        original(path, payload)

    monkeypatch.setattr(backend.storage, "write", fail_auxiliary)
    with pytest.raises(BackendWriteError, match="auxiliary failure"):
        with DatasetCapture(BasicWriter(), backend) as capture:
            with capture.episode("a") as episode:
                episode.write_frame(
                    0, {"LdrColor": np.zeros((2, 2, 3), dtype=np.uint8)}
                )
    assert not (tmp_path / "out/test/manifest.json").exists()
    assert not backend._root.exists()


@pytest.mark.parametrize(
    "native_config, export_config, message",
    [
        ({"include_panoptic": True, "include_segmentation": True}, {}, "has panoptic"),
        ({}, {"include_panoptic": True}, "lacks panoptic"),
        ({}, {"split": "val"}, "Missing instances_val.json"),
    ],
)
def test_export_rejects_config_that_does_not_match_native_output(
    tmp_path, native_config, export_config, message
):
    native = _coco_native(tmp_path, "a", 1, **native_config)
    writer = COCOWriter(COCOWriterConfig(image_format="png", **export_config))
    backend = _backend(tmp_path)
    with pytest.raises(PackagingError, match=message):
        export_finalized_dataset(writer, {"a": native}, backend)
    assert not (tmp_path / "out/test/manifest.json").exists()
    assert not backend.is_open


def test_export_rejects_missing_or_empty_episode_root(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PackagingError, match="No BasicWriter frame directories"):
        export_finalized_dataset(
            BasicWriter(), {"a": tmp_path / "empty"}, _backend(tmp_path)
        )
    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / "out"))
    with pytest.raises(PackagingError, match="does not exist"):
        export_finalized_dataset(
            BasicWriter(),
            {"a": tmp_path / "missing"},
            WebDatasetPackager(storage, WebDatasetPackagerConfig(dataset_id="other")),
        )


def test_source_paths_keep_symlink_names_and_reject_escapes(tmp_path):
    native = tmp_path / "native"
    disk = DiskBackend(DiskBackendConfig(root_dir=native))
    BasicWriter().write_frame(
        0, {"LdrColor": np.zeros((2, 2, 3), dtype=np.uint8)}, disk
    )
    (native / "0000/alias.png").symlink_to("ldr_color.png")
    export_finalized_dataset(BasicWriter(), {"a": native}, _backend(tmp_path))
    paths = _source_paths(tmp_path / "out", _manifest(tmp_path))["episodes"]["a"]
    assert paths["samples"]["0"] == {
        "alias.png": "0000/alias.png",
        "ldr_color.png": "0000/ldr_color.png",
    }
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")
    (native / "0000/escape.png").symlink_to(outside)
    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / "out"))
    with pytest.raises(PackagingError, match="escape.png is outside the native root"):
        export_finalized_dataset(
            BasicWriter(),
            {"a": native},
            WebDatasetPackager(storage, WebDatasetPackagerConfig(dataset_id="other")),
        )


def test_builtin_writer_subclass_requires_explicit_adapter(tmp_path):
    class TunedBasicWriter(BasicWriter):
        pass

    writer = TunedBasicWriter(BasicWriterConfig())
    backend = _backend(tmp_path)
    with pytest.raises(ConfigurationError, match="BasicFrameAdapter"):
        DatasetCapture(writer, backend)
    with DatasetCapture(writer, backend, adapter=BasicFrameAdapter()) as dataset:
        with dataset.episode("a") as episode:
            assert type(episode._writer) is TunedBasicWriter
            episode.write_frame(0, {"LdrColor": np.zeros((2, 2, 3), dtype=np.uint8)})
    assert _manifest(tmp_path)["samples"] == 1


def test_adapter_must_be_capture_adapter_instance(tmp_path):
    backend = _backend(tmp_path)
    with pytest.raises(ConfigurationError, match="CaptureAdapter instance"):
        DatasetCapture(BasicWriter(), backend, adapter=object())
    backend.abort()


def test_unserializable_writer_description_fails_before_capture(tmp_path):
    from ovannotators import CaptureAdapter, SampleFiles

    class OpaqueWriter:
        config = {"handle": object()}

    class OpaqueAdapter(CaptureAdapter):
        sample_schema = "opaque-v1"

        def create_writer(self, template):
            return template

        def collect_samples(self, root, writer, context):
            return [SampleFiles(0, {"bin": root / "x"})]

    backend = _backend(tmp_path)
    with pytest.raises(ConfigurationError, match="describe_writer"):
        DatasetCapture(OpaqueWriter(), backend, adapter=OpaqueAdapter())
    assert backend.storage.list() == []
    backend.abort()


@pytest.mark.parametrize("attribute", ["sample_schema", "sample_unit"])
def test_adapter_missing_manifest_metadata_fails_before_capture(tmp_path, attribute):
    from ovannotators import CaptureAdapter, SampleFiles

    class Writer:
        config = None

    class Adapter(CaptureAdapter):
        sample_schema = "custom-v1"

        def create_writer(self, template):
            return template

        def collect_samples(self, root, writer, context):
            return [SampleFiles(0, {"bin": root / "x"})]

    # Without the check this surfaced as an AttributeError at finalize, after
    # every episode had been captured.
    if attribute == "sample_schema":
        del Adapter.sample_schema
    else:
        Adapter.sample_unit = ""
    backend = _backend(tmp_path)
    with pytest.raises(ConfigurationError, match=attribute):
        DatasetCapture(Writer(), backend, adapter=Adapter())
    assert backend.storage.list() == []
    backend.abort()


def test_finalized_coco_export_rejects_out_of_order_frames(tmp_path):
    writer = COCOWriter(COCOWriterConfig(image_format="png"))
    root = tmp_path / "native"
    disk = DiskBackend(DiskBackendConfig(root_dir=root))
    for frame in (5, 3):
        writer.write_frame(frame, _render_vars(1), disk)
    writer.finalize(disk)
    backend = _backend(tmp_path)
    with pytest.raises(PackagingError, match="increasing frame order"):
        export_finalized_dataset(writer, {"a": root}, backend)
    assert not backend.is_open


def test_export_aborts_packager_when_no_adapter_applies(tmp_path):
    class Custom:
        config = None

    backend = _backend(tmp_path)
    with pytest.raises(ConfigurationError):
        export_finalized_dataset(Custom(), {"a": tmp_path}, backend)
    assert not backend.is_open


def test_finalized_coco_export_rejects_dangling_annotation_references(tmp_path):
    writer = COCOWriter(COCOWriterConfig(image_format="png"))
    root = _coco_native(tmp_path, "a", 1)
    instances = root / "annotations/instances_train.json"
    document = json.loads(instances.read_text())
    document["annotations"][0]["category_id"] = 99
    instances.write_text(json.dumps(document))
    backend = _backend(tmp_path)
    with pytest.raises(PackagingError, match="unknown category_id 99"):
        export_finalized_dataset(writer, {"a": root}, backend)
    document["annotations"][0]["category_id"] = document["categories"][0]["id"]
    document["annotations"][0]["image_id"] = 99
    instances.write_text(json.dumps(document))
    with pytest.raises(PackagingError):
        export_finalized_dataset(writer, {"a": root}, _backend(tmp_path / "second"))


def test_rejected_coco_episode_does_not_leak_categories(tmp_path):
    backend = _backend(tmp_path, max_samples_per_shard=1)
    with DatasetCapture(COCOWriter(), backend) as dataset:
        with pytest.raises(PackagingError, match="exceeds shard limits"):
            with dataset.episode("big") as episode:
                episode.write_frame(0, _render_vars(2))
                episode.write_frame(1, _render_vars(2))
        with dataset.episode("small") as episode:
            episode.write_frame(0, _render_vars(1))
    manifest = _manifest(tmp_path)
    assert manifest["metadata"]["categories"] == [
        {"id": 1, "name": "1", "supercategory": ""}
    ]
    record = json.loads(_members(tmp_path)["test/episode-small/sample-0.json"])
    assert record["image"]["id"] == 1
    assert record["annotations"][0]["id"] == 1
