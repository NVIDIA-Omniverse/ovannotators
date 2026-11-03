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

"""Verify episode and clip sample units of the experimental adapter contract."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import webdataset as wds

from ovannotators import (
    AdapterContext,
    BasicWriter,
    BasicWriterConfig,
    CaptureAdapter,
    ClipSpan,
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    SampleFiles,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
    export_finalized_dataset,
)
from ovannotators.exceptions import (
    BackendWriteError,
    ConfigurationError,
    PackagingError,
)

pytestmark = [pytest.mark.integration, pytest.mark.cpu_only]


class ClipWriter:
    """Buffer text frames and write paired rgb/depth streams per clip."""

    name = "fake-clip"

    def __init__(self, config=None):
        self.config = config
        self._frames: list[tuple[int, str]] = []
        self._clips: list[dict] = []
        self._next = 0
        self.fail_cut = False

    def write_frame(self, frame_id, render_vars, backend):
        self._frames.append((frame_id, render_vars["text"]))

    def cut(self, backend):
        if self.fail_cut:
            raise RuntimeError("encoder failed")
        if not self._frames:
            return None
        index = self._next
        ids = [frame for frame, _ in self._frames]
        backend.write(
            f"videos/{index:06d}_rgb.txt",
            "|".join(text for _, text in self._frames).encode(),
        )
        stream = io.BytesIO()
        np.save(stream, np.asarray(ids, dtype=np.int64))
        backend.write(f"conditioning/{index:06d}_depth.npy", stream.getvalue())
        self._clips.append({"clip_id": index, "frame_ids": ids})
        backend.write("manifest.json", json.dumps({"clips": self._clips}).encode())
        self._frames.clear()
        self._next += 1
        return index

    def finalize(self, backend):
        self.cut(backend)


class ClipAdapter(CaptureAdapter):
    sample_schema = "fake-clip-v1"
    sample_unit = "clip"
    # Like CosmosTransferWriter, the writer rewrites manifest.json per cut.
    allow_native_overwrite = True

    def create_writer(self, template):
        return ClipWriter()

    def cut_clip(self, writer, backend, root, context):
        index = writer.cut(backend)
        if index is None:
            raise PackagingError("writer had no frames to cut")
        return index

    def discover_clips(self, root, writer):
        manifest = json.loads((root / "manifest.json").read_text())
        return [
            ClipSpan(clip["clip_id"], tuple(clip["frame_ids"]))
            for clip in manifest["clips"]
        ]

    def collect_samples(self, root, writer, context):
        return [
            SampleFiles(
                span.clip_index,
                {
                    "rgb.txt": root / f"videos/{span.clip_index:06d}_rgb.txt",
                    "depth.npy": root / f"conditioning/{span.clip_index:06d}_depth.npy",
                },
                {"clip_index": span.clip_index, "frame_ids": list(span.frame_ids)},
            )
            for span in context.clips
        ]


class EpisodeWriter:
    """Write one episode-wide text stream and frame index on finalization."""

    name = "fake-episode"

    def __init__(self, config=None):
        self.config = config
        self._frames: list[tuple[int, str]] = []

    def write_frame(self, frame_id, render_vars, backend):
        self._frames.append((frame_id, render_vars["text"]))

    def finalize(self, backend):
        backend.write(
            "episode.txt", "|".join(text for _, text in self._frames).encode()
        )
        stream = io.BytesIO()
        np.save(stream, np.asarray([f for f, _ in self._frames], dtype=np.int64))
        backend.write("frames.npy", stream.getvalue())


class EpisodeAdapter(CaptureAdapter):
    sample_schema = "fake-episode-v1"
    sample_unit = "episode"

    def __init__(self, extra_sample=False):
        self.extra_sample = extra_sample

    def create_writer(self, template):
        return EpisodeWriter()

    def discover_frames(self, root, writer):
        return [int(v) for v in np.load(root / "frames.npy")]

    def collect_samples(self, root, writer, context):
        files = {"txt": root / "episode.txt", "frames.npy": root / "frames.npy"}
        samples = [SampleFiles("episode", files, {"frame_ids": context.frame_ids})]
        if self.extra_sample:
            samples.append(SampleFiles("extra", files))
        return samples


def _packager(tmp_path, name="out", **kwargs):
    storage = DiskBackend(DiskBackendConfig(root_dir=tmp_path / name))
    return WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="test", **kwargs)
    )


def _manifest(tmp_path, name="out"):
    return json.loads((tmp_path / name / "test/manifest.json").read_text())


def _members(tmp_path, name="out"):
    result = {}
    for path in sorted((tmp_path / name / "test").glob("*.tar")):
        with tarfile.open(path) as archive:
            for member in archive:
                result[member.name] = archive.extractfile(member).read()
    return result


def _source_paths(tmp_path, name="out"):
    reference = _manifest(tmp_path, name)["auxiliary"][0]
    return json.loads((tmp_path / name / reference["path"]).read_bytes())


def _capture_clips(tmp_path, clips, name="out", **kwargs):
    """Capture one episode per entry; each entry lists clips of frame texts."""
    with DatasetCapture(
        ClipWriter(), _packager(tmp_path, name, **kwargs), adapter=ClipAdapter()
    ) as capture:
        for episode_id, episode_clips in clips.items():
            frame = 0
            with capture.episode(episode_id) as episode:
                for number, clip in enumerate(episode_clips):
                    for text in clip:
                        episode.write_frame(frame, {"text": text})
                        frame += 2
                    if number < len(episode_clips) - 1:
                        assert episode.end_clip() == number


def test_clip_boundaries_pair_streams_per_clip(tmp_path):
    _capture_clips(tmp_path, {"a": [["x", "y"], ["z"], ["u", "v", "w"]]})
    manifest = _manifest(tmp_path)
    assert manifest["metadata"]["sample_unit"] == "clip"
    assert manifest["samples"] == 3
    samples = {
        s["__key__"]: s
        for s in wds.WebDataset(
            [
                f"file:{(tmp_path / 'out' / s['path']).resolve().as_posix()}"
                for s in manifest["shards"]
            ],
            shardshuffle=False,
        )
    }
    expected = {0: ("x|y", [0, 2]), 1: ("z", [4]), 2: ("u|v|w", [6, 8, 10])}
    for index, (text, frames) in expected.items():
        sample = samples[f"test/episode-a/sample-{index}"]
        assert sample["rgb.txt"].decode() == text
        assert np.load(io.BytesIO(sample["depth.npy"])).tolist() == frames
        meta = json.loads(sample["metadata.json"])["metadata"]
        assert meta == {"clip_index": index, "frame_ids": frames}
    clips = _source_paths(tmp_path)["episodes"]["a"]["clips"]
    assert clips == [
        {"clip_index": 0, "sample": "0", "frame_ids": [0, 2]},
        {"clip_index": 1, "sample": "1", "frame_ids": [4]},
        {"clip_index": 2, "sample": "2", "frame_ids": [6, 8, 10]},
    ]
    assert _source_paths(tmp_path)["episodes"]["a"]["samples"]["1"] == {
        "rgb.txt": "videos/000001_rgb.txt",
        "depth.npy": "conditioning/000001_depth.npy",
    }


def test_trailing_end_clip_does_not_add_empty_clip(tmp_path):
    with DatasetCapture(
        ClipWriter(), _packager(tmp_path), adapter=ClipAdapter()
    ) as capture:
        with capture.episode("a") as episode:
            episode.write_frame(0, {"text": "x"})
            assert episode.end_clip() == 0
    assert _manifest(tmp_path)["samples"] == 1


def test_empty_clip_is_rejected_and_episode_continues(tmp_path):
    with DatasetCapture(
        ClipWriter(), _packager(tmp_path), adapter=ClipAdapter()
    ) as capture:
        with capture.episode("a") as episode:
            with pytest.raises(PackagingError, match="empty clip"):
                episode.end_clip()
            episode.write_frame(0, {"text": "x"})
            episode.end_clip()
            with pytest.raises(PackagingError, match="empty clip"):
                episode.end_clip()
            episode.write_frame(1, {"text": "y"})
    assert _manifest(tmp_path)["samples"] == 2


def test_clip_episodes_stay_together_unless_spanning(tmp_path):
    clips = {"a": [["x"], ["y"]], "b": [["z"]]}
    _capture_clips(tmp_path, clips, max_samples_per_shard=2)
    assert [s["episode_samples"] for s in _manifest(tmp_path)["shards"]] == [
        {"a": 2},
        {"b": 1},
    ]
    _capture_clips(
        tmp_path, clips, "span", max_samples_per_shard=1, allow_episode_spanning=True
    )
    assert [s["episode_samples"] for s in _manifest(tmp_path, "span")["shards"]] == [
        {"a": 1},
        {"a": 1},
        {"b": 1},
    ]
    with pytest.raises(PackagingError, match="exceeds shard limits"):
        _capture_clips(
            tmp_path, {"a": [["x"], ["y"]]}, "strict", max_samples_per_shard=1
        )
    assert not (tmp_path / "strict/test/manifest.json").exists()


def test_frame_and_episode_adapters_reject_clip_boundaries(tmp_path):
    writer = BasicWriter(BasicWriterConfig())
    with DatasetCapture(writer, _packager(tmp_path)) as capture:
        with capture.episode("a") as episode:
            episode.write_frame(0, {"LdrColor": np.zeros((2, 2, 3), np.uint8)})
            with pytest.raises(PackagingError, match="frame samples"):
                episode.end_clip()
            episode.write_frame(1, {"LdrColor": np.zeros((2, 2, 3), np.uint8)})
    assert _manifest(tmp_path)["samples"] == 2
    with DatasetCapture(
        EpisodeWriter(), _packager(tmp_path, "ep"), adapter=EpisodeAdapter()
    ) as capture:
        with capture.episode("a") as episode:
            episode.write_frame(0, {"text": "x"})
            with pytest.raises(PackagingError, match="episode samples"):
                episode.end_clip()
    with pytest.raises(PackagingError, match="open episode"):
        episode.end_clip()


def test_failed_cut_discards_episode_and_capture_continues(tmp_path):
    class FailingAdapter(ClipAdapter):
        def cut_clip(self, writer, backend, root, context):
            writer.fail_cut = context.episode_id == "bad"
            return super().cut_clip(writer, backend, root, context)

    with DatasetCapture(
        ClipWriter(), _packager(tmp_path), adapter=FailingAdapter()
    ) as capture:
        with pytest.raises(RuntimeError, match="encoder failed"):
            with capture.episode("bad") as episode:
                episode.write_frame(0, {"text": "x"})
                episode.end_clip()
        with capture.episode("good") as episode:
            episode.write_frame(0, {"text": "x"})
    assert _manifest(tmp_path)["episodes"] == ["good"]


def test_duplicate_or_invalid_clip_index_discards_episode(tmp_path):
    class RepeatingAdapter(ClipAdapter):
        def cut_clip(self, writer, backend, root, context):
            super().cut_clip(writer, backend, root, context)
            return 0

    with DatasetCapture(
        ClipWriter(), _packager(tmp_path), adapter=RepeatingAdapter()
    ) as capture:
        with pytest.raises(PackagingError, match="Duplicate clip index"):
            with capture.episode("a") as episode:
                episode.write_frame(0, {"text": "x"})
                episode.end_clip()
                episode.write_frame(1, {"text": "y"})
                episode.end_clip()
        with capture.episode("b") as episode:
            episode.write_frame(0, {"text": "x"})
    assert _manifest(tmp_path)["episodes"] == ["b"]


@pytest.mark.parametrize(
    ("clips", "message"),
    [
        ([], "at least one clip"),
        ([ClipSpan(0, ())], "contains no frames"),
        ([ClipSpan(-1, (0,))], "Invalid clip index"),
        ([ClipSpan(True, (0,))], "Invalid clip index"),
        ([ClipSpan(0, (2,)), ClipSpan(1, (2,))], "strictly increasing"),
        ([ClipSpan(0, (0,)), ClipSpan(0, (1,))], "Duplicate clip index"),
        ([(0, (1,))], "ClipSpan instances"),
    ],
)
def test_export_rejects_invalid_discovered_clips(tmp_path, clips, message):
    class Discovered(ClipAdapter):
        def discover_clips(self, root, writer):
            return clips

    native = tmp_path / "native"
    native.mkdir()
    packager = _packager(tmp_path)
    with pytest.raises(PackagingError, match=message):
        export_finalized_dataset(
            ClipWriter(), {"a": native}, packager, adapter=Discovered()
        )
    assert not packager.is_open


def test_clip_sample_count_must_match_clips(tmp_path):
    class Dropping(ClipAdapter):
        def collect_samples(self, root, writer, context):
            return super().collect_samples(root, writer, context)[:1]

    with DatasetCapture(
        ClipWriter(), _packager(tmp_path), adapter=Dropping()
    ) as capture:
        with pytest.raises(PackagingError, match="1 samples for 2 clips"):
            with capture.episode("a") as episode:
                episode.write_frame(0, {"text": "x"})
                episode.end_clip()
                episode.write_frame(1, {"text": "y"})


def test_clip_export_matches_capture(tmp_path):
    _capture_clips(tmp_path, {"a": [["x", "y"], ["z"]]}, "captured")
    native = tmp_path / "native"
    disk = DiskBackend(DiskBackendConfig(root_dir=native, overwrite=True))
    writer = ClipWriter()
    for frame, text in [(0, "x"), (2, "y")]:
        writer.write_frame(frame, {"text": text}, disk)
    writer.cut(disk)
    writer.write_frame(4, {"text": "z"}, disk)
    writer.finalize(disk)
    before = {p: p.read_bytes() for p in native.rglob("*") if p.is_file()}
    export_finalized_dataset(
        ClipWriter(),
        {"a": native},
        _packager(tmp_path, "exported"),
        adapter=ClipAdapter(),
    )
    assert _members(tmp_path, "exported") == _members(tmp_path, "captured")
    assert (
        _source_paths(tmp_path, "exported")["episodes"]
        == _source_paths(tmp_path, "captured")["episodes"]
    )
    assert _manifest(tmp_path, "exported")["metadata"]["provenance"] == (
        "finalized-export"
    )
    assert before == {p: p.read_bytes() for p in native.rglob("*") if p.is_file()}


def test_clip_export_requires_discover_clips(tmp_path):
    class NoDiscovery(ClipAdapter):
        discover_clips = CaptureAdapter.discover_clips

    native = tmp_path / "native"
    native.mkdir()
    with pytest.raises(PackagingError, match="does not support finalized"):
        export_finalized_dataset(
            ClipWriter(), {"a": native}, _packager(tmp_path), adapter=NoDiscovery()
        )


def test_episode_unit_packs_one_sample_per_episode(tmp_path):
    with DatasetCapture(
        EpisodeWriter(),
        _packager(tmp_path, max_samples_per_shard=1),
        adapter=EpisodeAdapter(),
    ) as capture:
        for episode_id, texts in {"a": ["x", "y"], "b": ["z"]}.items():
            with capture.episode(episode_id) as episode:
                for frame, text in enumerate(texts):
                    episode.write_frame(frame, {"text": text})
    manifest = _manifest(tmp_path)
    assert manifest["metadata"]["sample_unit"] == "episode"
    assert manifest["samples"] == 2
    assert [s["episodes"] for s in manifest["shards"]] == [["a"], ["b"]]
    members = _members(tmp_path)
    assert members["test/episode-a/sample-episode.txt"] == b"x|y"
    meta = json.loads(members["test/episode-a/sample-episode.metadata.json"])
    assert meta["metadata"] == {"frame_ids": [0, 1]}
    assert _source_paths(tmp_path)["episodes"]["a"] == {
        "shared": {},
        "samples": {"episode": {"txt": "episode.txt", "frames.npy": "frames.npy"}},
    }


def test_episode_unit_rejects_multiple_samples(tmp_path):
    with DatasetCapture(
        EpisodeWriter(), _packager(tmp_path), adapter=EpisodeAdapter(True)
    ) as capture:
        with pytest.raises(PackagingError, match="expected exactly one"):
            with capture.episode("a") as episode:
                episode.write_frame(0, {"text": "x"})


@pytest.mark.parametrize("spanning", [False, True])
def test_oversized_episode_sample_cannot_be_split(tmp_path, spanning):
    big = "x" * 20_000

    def capture(name, policy):
        with DatasetCapture(
            EpisodeWriter(),
            _packager(
                tmp_path,
                name,
                max_shard_bytes=10240,
                allow_episode_spanning=spanning,
                oversized_episode_policy=policy,
            ),
            adapter=EpisodeAdapter(),
        ) as dataset:
            with dataset.episode("a") as episode:
                episode.write_frame(0, {"text": big})
                episode.write_frame(1, {"text": big})

    with pytest.raises(PackagingError, match="exceeds shard limits"):
        capture("strict", "error")
    capture("dedicated", "dedicated_shard")
    shards = _manifest(tmp_path, "dedicated")["shards"]
    assert len(shards) == 1
    assert shards[0]["oversized"] and shards[0]["samples"] == 1


def test_episode_export_matches_capture(tmp_path):
    with DatasetCapture(
        EpisodeWriter(), _packager(tmp_path, "captured"), adapter=EpisodeAdapter()
    ) as capture:
        with capture.episode("a") as episode:
            episode.write_frame(3, {"text": "x"})
            episode.write_frame(5, {"text": "y"})
    native = tmp_path / "native"
    disk = DiskBackend(DiskBackendConfig(root_dir=native))
    writer = EpisodeWriter()
    writer.write_frame(3, {"text": "x"}, disk)
    writer.write_frame(5, {"text": "y"}, disk)
    writer.finalize(disk)
    export_finalized_dataset(
        EpisodeWriter(),
        {"a": native},
        _packager(tmp_path, "exported"),
        adapter=EpisodeAdapter(),
    )
    assert _members(tmp_path, "exported") == _members(tmp_path, "captured")
    assert (
        _source_paths(tmp_path, "exported")["episodes"]
        == _source_paths(tmp_path, "captured")["episodes"]
    )


def test_native_rewrite_requires_explicit_overwrite(tmp_path):
    class NoOverwrite(ClipAdapter):
        allow_native_overwrite = False

    with DatasetCapture(
        ClipWriter(), _packager(tmp_path), adapter=NoOverwrite()
    ) as capture:
        with pytest.raises(BackendWriteError, match="overwrite=False"):
            with capture.episode("a") as episode:
                episode.write_frame(0, {"text": "x"})
                episode.end_clip()
                episode.write_frame(1, {"text": "y"})
                episode.end_clip()
        with capture.episode("b") as episode:
            episode.write_frame(0, {"text": "x"})
    assert _manifest(tmp_path)["episodes"] == ["b"]


def test_sample_unit_is_validated_before_capture(tmp_path):
    class Unknown(EpisodeAdapter):
        sample_unit = "window"

    class BadOverwrite(EpisodeAdapter):
        allow_native_overwrite = "yes"

    class MissingCut(CaptureAdapter):
        sample_schema = "missing-cut-v1"
        sample_unit = "clip"

        def create_writer(self, template):
            return ClipWriter()

        def collect_samples(self, root, writer, context: AdapterContext):
            return []

    with pytest.raises(ConfigurationError, match="Unsupported sample_unit"):
        DatasetCapture(EpisodeWriter(), _packager(tmp_path), adapter=Unknown())
    with pytest.raises(ConfigurationError, match="allow_native_overwrite"):
        DatasetCapture(EpisodeWriter(), _packager(tmp_path), adapter=BadOverwrite())
    with pytest.raises(ConfigurationError, match="must override cut_clip"):
        DatasetCapture(ClipWriter(), _packager(tmp_path), adapter=MissingCut())
    with pytest.raises(PackagingError, match="does not support clips"):
        CaptureAdapter.cut_clip(
            MissingCut(), ClipWriter(), None, Path(), AdapterContext("d", "e", Path())
        )
