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

"""Shared WebDataset regression matrix for capture adapters.

An adapter test module describes its writer and samples once with
:class:`AdapterCase` and inherits every matrix check by subclassing
:class:`WebDatasetAdapterMatrix` and overriding its ``case`` fixture::

    class TestKittiMatrix(WebDatasetAdapterMatrix):
        @pytest.fixture
        def case(self):
            return AdapterCase(name="kitti", ...)

The matrix covers byte/sample/episode limits, default episode preservation and
opt-in spanning, oversized units under both policies, negative and empty
examples, shared metadata, duplicate identities, case/normalization collisions,
long identities at the USTAR limit, writer finalization and storage transfer
failures, decoded fidelity, extraction of all shards into one directory, and
unchanged ordinary disk writing.

The matrix is frame-unit only. It asserts ``metadata["sample_unit"] == "frame"``,
counts one sample per frame, and never calls ``end_clip``. Episode or clip
adapters cannot reuse it unchanged: they need an extension such as a
``sample_unit`` field on :class:`AdapterCase`, per-unit expected sample counts,
and a clip-boundary hook in the capture helper.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest
import webdataset as wds

from ovannotators import (
    DatasetCapture,
    DiskBackend,
    DiskBackendConfig,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
)
from ovannotators.capture_adapters import CaptureAdapter
from ovannotators.exceptions import BackendWriteError, PackagingError

RenderVars = dict[str, Any]

#: Default episodes: identity mapped to frame IDs with gaps. The episode
#: position is the ``variant`` passed to :attr:`AdapterCase.frame`.
DEFAULT_EPISODES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("a", (0, 3)),
    ("b", (1, 2)),
    ("c", (0, 5)),
)


@dataclass(frozen=True)
class AdapterCase:
    """Describe one adapter's writer and samples for the shared matrix.

    Attributes:
        name: Short label used in failure messages.
        make_writer: Return a fresh writer configuration template.
        frame: Return render vars for ``(variant, frame_id)``. Different
            variants should exercise dataset-wide state, such as categories.
        large_frame: Return render vars whose sample alone exceeds one
            10240-byte tar record, used for oversized-unit checks.
        check_decoded: Assert one sample decoded by the official reader
            (``decode`` spec) matches the render vars that produced it.
        negative_frame: A valid frame with no labels, or an empty frame.
        negative_is_error: Whether ``negative_frame`` must be rejected by
            ``write_frame`` instead of producing a sample.
        check_negative: Assert the decoded negative sample is well formed.
        check_manifest: Assert adapter-specific shared manifest metadata.
        sample_schema: Expected ``metadata.sample_schema``.
        adapter: Explicit adapter, or ``None`` for built-in selection.
        decode: Official-reader decode specification.
    """

    name: str
    make_writer: Callable[[], Any]
    frame: Callable[[int, int], RenderVars]
    large_frame: Callable[[int], RenderVars]
    check_decoded: Callable[[Mapping[str, Any], RenderVars], None]
    negative_frame: RenderVars
    sample_schema: str
    negative_is_error: bool = False
    check_negative: Callable[[Mapping[str, Any]], None] | None = None
    check_manifest: Callable[[Mapping[str, Any]], None] | None = None
    adapter: CaptureAdapter | None = None
    decode: str = "rgb8"


class RecordingStorage(DiskBackend):
    """Disk backend that counts storage calls and can fail selected writes.

    Attributes:
        calls: Number of calls per storage method.
        written: Paths written, in order.
        bytes_written: Total payload bytes accepted by ``write``.
        fail_write: Predicate selecting write paths that raise
            ``BackendWriteError`` instead of writing.
    """

    def __init__(
        self, root: Path, fail_write: Callable[[str], bool] | None = None
    ) -> None:
        super().__init__(DiskBackendConfig(root_dir=root))
        self.calls: Counter[str] = Counter()
        self.written: list[str] = []
        self.bytes_written = 0
        self.fail_write = fail_write

    def write(self, path: str, data: bytes) -> None:
        self.calls["write"] += 1
        if self.fail_write is not None and self.fail_write(path):
            raise BackendWriteError(path=path, message="injected transfer failure")
        super().write(path, data)
        self.written.append(path)
        self.bytes_written += len(data)

    def exists(self, path: str) -> bool:
        self.calls["exists"] += 1
        return super().exists(path)

    def list(self, prefix: str = "") -> list[str]:
        self.calls["list"] += 1
        return super().list(prefix)


@dataclass
class CaptureResult:
    """Published dataset produced by :func:`capture_dataset`."""

    root: Path
    dataset_id: str
    storage: RecordingStorage
    rendered: dict[tuple[str, int], RenderVars] = field(default_factory=dict)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.root / self.dataset_id / "manifest.json").read_text())

    @property
    def shard_paths(self) -> list[Path]:
        return [self.root / shard["path"] for shard in self.manifest["shards"]]

    def members(self) -> dict[str, bytes]:
        """Return every shard member, asserting names are unique."""
        result: dict[str, bytes] = {}
        for path in self.shard_paths:
            with tarfile.open(path, format=tarfile.USTAR_FORMAT) as archive:
                for member in archive:
                    assert member.name not in result, member.name
                    result[member.name] = archive.extractfile(member).read()
        return result

    def decoded(self, decode: str) -> list[dict[str, Any]]:
        """Read every sample with the official WebDataset reader."""
        urls = [f"file:{path.resolve().as_posix()}" for path in self.shard_paths]
        return list(wds.WebDataset(urls, shardshuffle=False).decode(decode))

    def source_paths(self) -> dict[str, Any]:
        reference = self.manifest["auxiliary"][0]
        payload = (self.root / reference["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
        return json.loads(payload)


def capture_dataset(
    case: AdapterCase,
    root: Path,
    *,
    episodes: Sequence[tuple[str, Sequence[int]]] = DEFAULT_EPISODES,
    dataset_id: str = "matrix",
    fail_write: Callable[[str], bool] | None = None,
    **limits: Any,
) -> CaptureResult:
    """Capture ``episodes`` through ``DatasetCapture`` with the given limits."""
    storage = RecordingStorage(root, fail_write)
    packager = WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id=dataset_id, **limits)
    )
    result = CaptureResult(root, dataset_id, storage)
    with DatasetCapture(case.make_writer(), packager, adapter=case.adapter) as dataset:
        for variant, (identity, frames) in enumerate(episodes):
            with dataset.episode(identity) as episode:
                for frame in frames:
                    values = case.frame(variant, frame)
                    result.rendered[(identity, frame)] = values
                    episode.write_frame(frame, values)
    return result


def _sample_locations(result: CaptureResult) -> list[tuple[int, str, str]]:
    """Return ``(shard index, episode, sample)`` for every member group."""
    seen = []
    for index, path in enumerate(result.shard_paths):
        with tarfile.open(path) as archive:
            for member in archive:
                _, episode, sample = member.name.split("/")
                location = (index, episode, sample.split(".", 1)[0])
                if location not in seen:
                    seen.append(location)
    return seen


def _assert_whole_episodes(result: CaptureResult) -> None:
    owners: dict[str, set[int]] = {}
    for shard, episode, _ in _sample_locations(result):
        owners.setdefault(episode, set()).add(shard)
    split = {episode: shards for episode, shards in owners.items() if len(shards) > 1}
    assert not split, f"episodes split across shards: {split}"


def _expected_samples(episodes: Sequence[tuple[str, Sequence[int]]]) -> int:
    return sum(len(frames) for _, frames in episodes)


def _collision_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


class WebDatasetAdapterMatrix:
    """Shared WebDataset regressions; subclass and override ``case``."""

    @pytest.fixture
    def case(self) -> AdapterCase:
        raise NotImplementedError("Override the case fixture in the subclass")

    # -- limits and episode preservation ----------------------------------

    def test_byte_limit_rotates_between_whole_episodes(self, case, tmp_path):
        single = capture_dataset(
            case, tmp_path / "single", max_episodes_per_shard=1, dataset_id="single"
        )
        combined = capture_dataset(case, tmp_path / "combined", dataset_id="combined")
        limit = max(shard["bytes"] for shard in single.manifest["shards"])
        assert combined.manifest["shards"][0]["bytes"] > limit, (
            f"{case.name} samples are too small to force byte rotation"
        )
        result = capture_dataset(case, tmp_path / "limited", max_shard_bytes=limit)
        shards = result.manifest["shards"]
        assert len(shards) > 1
        assert all(shard["bytes"] <= limit for shard in shards)
        assert all(path.stat().st_size <= limit for path in result.shard_paths)
        assert not any(shard["oversized"] for shard in shards)
        assert result.manifest["samples"] == _expected_samples(DEFAULT_EPISODES)
        _assert_whole_episodes(result)

    @pytest.mark.parametrize(
        "limits, expected",
        [
            ({"max_samples_per_shard": 4}, [["a", "b"], ["c"]]),
            ({"max_episodes_per_shard": 2}, [["a", "b"], ["c"]]),
        ],
    )
    def test_count_limits_rotate_between_whole_episodes(
        self, case, tmp_path, limits, expected
    ):
        result = capture_dataset(case, tmp_path, **limits)
        shards = result.manifest["shards"]
        assert [shard["episodes"] for shard in shards] == expected
        assert sum(shard["samples"] for shard in shards) == 6
        _assert_whole_episodes(result)

    def test_default_preserves_episode_and_spanning_is_opt_in(self, case, tmp_path):
        episodes = [("long", (0, 1, 2, 3, 4))]
        with pytest.raises(PackagingError, match="exceeds shard limits"):
            capture_dataset(
                case, tmp_path / "preserve", episodes=episodes, max_samples_per_shard=2
            )
        assert not (tmp_path / "preserve/matrix/manifest.json").exists()
        result = capture_dataset(
            case,
            tmp_path / "span",
            episodes=episodes,
            max_samples_per_shard=2,
            allow_episode_spanning=True,
        )
        shards = result.manifest["shards"]
        assert [shard["samples"] for shard in shards] == [2, 2, 1]
        assert all(shard["episodes"] == ["long"] for shard in shards)
        order = [sample for _, _, sample in _sample_locations(result)]
        assert order == [f"sample-{frame}" for frame in episodes[0][1]]

    @pytest.mark.parametrize("spanning", [False, True])
    def test_oversized_unit_policy(self, case, tmp_path, spanning):
        def large_episode(variant, frame):
            return case.large_frame(frame)

        oversized = replace(case, frame=large_episode)
        episodes = [("big", (0, 1))]
        with pytest.raises(PackagingError, match="exceeds shard limits"):
            capture_dataset(
                oversized,
                tmp_path / "error",
                episodes=episodes,
                max_shard_bytes=10240,
                allow_episode_spanning=spanning,
            )
        result = capture_dataset(
            oversized,
            tmp_path / "dedicated",
            episodes=episodes,
            max_shard_bytes=10240,
            allow_episode_spanning=spanning,
            oversized_episode_policy="dedicated_shard",
        )
        shards = result.manifest["shards"]
        assert [shard["samples"] for shard in shards] == ([1, 1] if spanning else [2])
        assert all(shard["oversized"] and shard["bytes"] > 10240 for shard in shards)
        assert len(result.decoded(case.decode)) == 2

    # -- negative, empty and shared metadata ------------------------------

    def test_negative_example(self, case, tmp_path):
        if case.negative_is_error:
            with pytest.raises(PackagingError):
                capture_dataset(
                    replace(case, frame=lambda variant, frame: case.negative_frame),
                    tmp_path,
                    episodes=[("neg", (0,))],
                )
            assert not (tmp_path / "matrix/manifest.json").exists()
            return
        values = [case.frame(0, 0), case.negative_frame]
        result = capture_dataset(
            replace(case, frame=lambda variant, frame: values[frame]),
            tmp_path,
            episodes=[("neg", (0, 1))],
        )
        samples = result.decoded(case.decode)
        assert len(samples) == 2
        if case.check_negative is not None:
            case.check_negative(samples[1])

    def test_empty_episode_is_rejected_and_capture_continues(self, case, tmp_path):
        storage = RecordingStorage(tmp_path)
        packager = WebDatasetPackager(
            storage, WebDatasetPackagerConfig(dataset_id="matrix")
        )
        with DatasetCapture(
            case.make_writer(), packager, adapter=case.adapter
        ) as dataset:
            with pytest.raises(PackagingError, match="no frames"):
                with dataset.episode("empty"):
                    pass
            with dataset.episode("full") as episode:
                episode.write_frame(0, case.frame(0, 0))
        manifest = json.loads((tmp_path / "matrix/manifest.json").read_text())
        assert manifest["episodes"] == ["full"]

    def test_shared_metadata_and_per_sample_identity(self, case, tmp_path):
        result = capture_dataset(case, tmp_path)
        manifest = result.manifest
        metadata = manifest["metadata"]
        assert metadata["sample_schema"] == case.sample_schema
        assert metadata["sample_unit"] == "frame"
        assert metadata["provenance"] == "capture"
        assert manifest["episodes"] == ["a", "b", "c"]
        assert result.storage.written[-1] == "matrix/manifest.json"
        paths = result.source_paths()["episodes"]
        assert sorted(paths) == ["a", "b", "c"]
        for sample in result.decoded(case.decode):
            identity = sample["metadata.json"]
            episode, key = sample["__key__"].split("/")[1:]
            assert identity["dataset_id"] == "matrix"
            assert f"episode-{identity['episode_id']}" == episode
            assert f"sample-{identity['sample_id']}" == key
            assert identity["sample_id"] in paths[identity["episode_id"]]["samples"]
        if case.check_manifest is not None:
            case.check_manifest(manifest)

    # -- identity -----------------------------------------------------------

    def test_duplicate_identities_are_rejected(self, case, tmp_path):
        storage = RecordingStorage(tmp_path)
        packager = WebDatasetPackager(
            storage, WebDatasetPackagerConfig(dataset_id="matrix")
        )
        with DatasetCapture(
            case.make_writer(), packager, adapter=case.adapter
        ) as dataset:
            with dataset.episode(1) as episode:
                episode.write_frame(0, case.frame(0, 0))
            for duplicate in (1, "1"):
                with pytest.raises(PackagingError, match="Duplicate episode"):
                    dataset.episode(duplicate)
            with pytest.raises(PackagingError, match="strictly increasing"):
                with dataset.episode("2") as episode:
                    episode.write_frame(3, case.frame(0, 3))
                    episode.write_frame(3, case.frame(0, 3))
        assert json.loads((tmp_path / "matrix/manifest.json").read_text())[
            "episodes"
        ] == ["1"]

    @pytest.mark.parametrize("identity", ["A", "Episode", "é", "é", "ａ", "a​"])
    def test_case_and_normalization_variants_are_rejected(
        self, case, tmp_path, identity
    ):
        packager = WebDatasetPackager(
            RecordingStorage(tmp_path), WebDatasetPackagerConfig(dataset_id="matrix")
        )
        with DatasetCapture(
            case.make_writer(), packager, adapter=case.adapter
        ) as dataset:
            with pytest.raises(PackagingError, match="Identifiers"):
                dataset.episode(identity)
            with dataset.episode("a") as episode:
                episode.write_frame(0, case.frame(0, 0))

    def test_member_names_are_unique_after_case_folding(self, case, tmp_path):
        names = list(capture_dataset(case, tmp_path).members())
        folded = Counter(_collision_key(name) for name in names)
        assert [name for name, count in folded.items() if count > 1] == []
        assert all(name.isascii() for name in names)

    def test_long_identities_fit_ustar(self, case, tmp_path):
        result = capture_dataset(
            case,
            tmp_path,
            episodes=[("e" * 40, (999999,))],
            dataset_id="d" * 40,
        )
        with tarfile.open(result.shard_paths[0], format=tarfile.USTAR_FORMAT) as tar:
            names = tar.getnames()
        assert names and all(
            name.startswith(f"{'d' * 40}/episode-{'e' * 40}/sample-999999.")
            for name in names
        )
        assert len(result.decoded(case.decode)) == 1

    # -- failures -------------------------------------------------------------

    def test_writer_finalization_failure_discards_episode(
        self, case, tmp_path, monkeypatch
    ):
        writer_type = type(case.make_writer())
        original = writer_type.finalize
        failures = [RuntimeError("injected finalization failure")]

        def finalize(self, backend):
            if failures:
                raise failures.pop()
            return original(self, backend)

        monkeypatch.setattr(writer_type, "finalize", finalize)
        packager = WebDatasetPackager(
            RecordingStorage(tmp_path), WebDatasetPackagerConfig(dataset_id="matrix")
        )
        with DatasetCapture(
            case.make_writer(), packager, adapter=case.adapter
        ) as dataset:
            with pytest.raises(RuntimeError, match="injected"):
                with dataset.episode("a") as episode:
                    episode.write_frame(0, case.frame(0, 0))
            assert packager.is_open
            with dataset.episode("a") as episode:
                episode.write_frame(0, case.frame(0, 0))
        manifest = json.loads((tmp_path / "matrix/manifest.json").read_text())
        assert manifest["episodes"] == ["a"] and manifest["samples"] == 1

    @pytest.mark.parametrize(
        "target", ["shard-000000.tar", "shard-000002.tar", "auxiliary/", "manifest"]
    )
    def test_storage_transfer_failure_aborts_without_manifest(
        self, case, tmp_path, target
    ):
        with pytest.raises(BackendWriteError, match="injected transfer failure"):
            capture_dataset(
                case,
                tmp_path,
                max_episodes_per_shard=1,
                fail_write=lambda path: target in path,
            )
        assert not (tmp_path / "matrix/manifest.json").exists()
        published = sorted(p.name for p in (tmp_path / "matrix").glob("*.tar"))
        expected = {
            "shard-000000.tar": [],
            "shard-000002.tar": ["shard-000000.tar", "shard-000001.tar"],
        }
        assert published == expected.get(
            target, ["shard-000000.tar", "shard-000001.tar", "shard-000002.tar"]
        )

    # -- fidelity and layout --------------------------------------------------

    def test_decoded_fidelity(self, case, tmp_path):
        result = capture_dataset(case, tmp_path)
        samples = result.decoded(case.decode)
        assert len(samples) == _expected_samples(DEFAULT_EPISODES)
        for sample in samples:
            identity = sample["metadata.json"]
            values = result.rendered[
                (identity["episode_id"], int(identity["sample_id"]))
            ]
            case.check_decoded(sample, values)

    def test_extract_all_shards_into_one_directory(self, case, tmp_path):
        result = capture_dataset(case, tmp_path / "out", max_episodes_per_shard=1)
        assert len(result.shard_paths) == 3
        destination = tmp_path / "extracted"
        destination.mkdir()
        extracted = 0
        for path in result.shard_paths:
            with tarfile.open(path) as archive:
                for member in archive:
                    assert member.isfile() and not member.name.startswith("/")
                    target = destination / member.name
                    assert destination.resolve() in target.resolve().parents
                    assert not target.exists(), f"collision: {member.name}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
                    extracted += 1
        files = [p for p in destination.rglob("*") if p.is_file()]
        assert len(files) == extracted == len(result.members())
        assert len({_collision_key(str(p)) for p in files}) == len(files)
        assert len({(p.parent, p.name.split(".", 1)[0]) for p in files}) == 6

    def test_ordinary_disk_writing_is_unchanged(self, case, tmp_path):
        result = capture_dataset(case, tmp_path / "packaged")
        members = result.members()
        paths = result.source_paths()["episodes"]
        for variant, (identity, frames) in enumerate(DEFAULT_EPISODES):
            native = tmp_path / "native" / identity
            disk = DiskBackend(DiskBackendConfig(root_dir=native))
            writer = case.make_writer()
            for frame in frames:
                writer.write_frame(frame, case.frame(variant, frame), disk)
            writer.finalize(disk)
            for path in paths[identity]["shared"].values():
                assert (native / path).is_file()
            for key, components in paths[identity]["samples"].items():
                for component, relative in components.items():
                    member = f"matrix/episode-{identity}/sample-{key}.{component}"
                    assert members[member] == (native / relative).read_bytes()
