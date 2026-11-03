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

"""Test-only native reconstruction and fidelity checks for WebDataset output.

This is a validation utility for adapter tests, not library API: it proves that
a completed WebDataset dataset retains enough information to rebuild a format
writer's native directory. ``source-paths.json`` stays informational for users.

Workflow::

    dataset = read_packaged_dataset(storage_root, "dataset-id")
    roots = reconstruct_native(dataset, tmp_path / "rebuilt", shared_builders={
        "instances": coco_shared_document, "panoptic": coco_shared_document})
    assert_trees_equal(native_root, roots["episode-id"], comparators={
        "annotations/*.json": coco_documents_equivalent})

``read_packaged_dataset(storage_root, dataset_id, *, cross_check=True)``
    Reads ``<storage_root>/<dataset_id>/manifest.json`` from a DiskBackend root,
    verifies every shard and auxiliary file against the recorded bytes and
    sha256, groups tar members by sample key, checks per-shard and per-episode
    sample counts and ``metadata.json`` identity, and (by default) checks that
    the official ``webdataset`` reader yields the same keys and member bytes.
    Returns a ``PackagedDataset``; raises ``ReconstructionError``.

``reconstruct_native(dataset, destination, *, shared_builders=None)``
    Writes ``destination/<episode_id>/<native path>`` for every component listed
    in the ``source-paths.json`` ``samples`` mapping. Generated components and
    ``metadata.json`` have no native path and are not written. Every ``shared``
    role needs a ``SharedBuilder`` ``(EpisodeRecords, role) -> bytes`` that
    regenerates the document, typically from per-sample JSON records. Returns
    ``{episode_id: native_root}``.

``compare_trees(expected, actual, *, comparators=None, ignore=())``
    Returns a ``TreeDifference`` (missing/extra/changed relative paths). Files
    are byte-equal unless a ``comparators`` glob (``fnmatch`` on the POSIX
    relative path, first match wins) supplies a ``Comparator``
    ``(expected_bytes, actual_bytes) -> str | None`` returning a reason on
    mismatch. ``assert_trees_equal`` raises ``AssertionError`` with the report.

COCO reference hooks: ``coco_shared_document`` regenerates ``instances`` and
``panoptic`` documents; ``coco_documents_equivalent`` compares COCO documents
after replacing renumbered image/category IDs with file and category names.
Sample records do not carry ``info``/``licenses``, so the reconstructed
document omits them and the comparator ignores those keys by default.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping

import webdataset as wds


class ReconstructionError(AssertionError):
    """Packaged output is inconsistent or cannot be reconstructed."""


@dataclass(frozen=True)
class PackagedSample:
    """One sample's tar members, grouped by WebDataset key."""

    key: str
    episode_id: str
    sample_id: str
    shard: str
    members: Mapping[str, bytes]

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.members["metadata.json"])

    def json(self, component: str = "json") -> Any:
        return json.loads(self.members[component])


@dataclass(frozen=True)
class EpisodeRecords:
    """Inputs a ``SharedBuilder`` uses to regenerate one shared document."""

    episode_id: str
    samples: tuple[PackagedSample, ...]
    sample_paths: Mapping[str, Mapping[str, str]]
    shared_paths: Mapping[str, str]
    dataset_metadata: Mapping[str, Any]


SharedBuilder = Callable[[EpisodeRecords, str], bytes]
Comparator = Callable[[bytes, bytes], "str | None"]


@dataclass(frozen=True)
class PackagedDataset:
    """Verified completed dataset: manifest, source paths and grouped samples."""

    root: Path
    manifest: Mapping[str, Any]
    source_paths: Mapping[str, Any]
    samples: tuple[PackagedSample, ...]

    def episode(self, episode_id: str) -> EpisodeRecords:
        entry = self.source_paths["episodes"][episode_id]
        return EpisodeRecords(
            episode_id=episode_id,
            samples=tuple(s for s in self.samples if s.episode_id == episode_id),
            sample_paths=entry["samples"],
            shared_paths=entry["shared"],
            dataset_metadata=self.manifest["metadata"],
        )


def _verified_bytes(root: Path, record: Mapping[str, Any]) -> bytes:
    path = root / record["path"]
    if not path.is_file():
        raise ReconstructionError(f"Missing published file: {record['path']}")
    payload = path.read_bytes()
    if len(payload) != record["bytes"]:
        raise ReconstructionError(f"Byte count mismatch: {record['path']}")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ReconstructionError(f"sha256 mismatch: {record['path']}")
    return payload


def _split_member(name: str, dataset_id: str) -> tuple[str, str, str, str]:
    """Return (key, episode_id, sample_id, component) for a member name."""
    parts = name.split("/")
    if len(parts) != 3 or parts[0] != dataset_id:
        raise ReconstructionError(f"Member outside dataset namespace: {name}")
    episode_dir, basename = parts[1], parts[2]
    stem, dot, component = basename.partition(".")
    if (
        not dot
        or not episode_dir.startswith("episode-")
        or not stem.startswith("sample-")
    ):
        raise ReconstructionError(f"Unexpected member name: {name}")
    return (
        f"{dataset_id}/{episode_dir}/{stem}",
        episode_dir.removeprefix("episode-"),
        stem.removeprefix("sample-"),
        component,
    )


def read_packaged_dataset(
    storage_root: str | Path, dataset_id: str, *, cross_check: bool = True
) -> PackagedDataset:
    root = Path(storage_root)
    manifest_path = root / dataset_id / "manifest.json"
    if not manifest_path.is_file():
        raise ReconstructionError(f"No completion manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "webdataset" or manifest.get("schema_version") != 1:
        raise ReconstructionError("Unsupported manifest format or schema version")
    if manifest["dataset_id"] != dataset_id:
        raise ReconstructionError("Manifest dataset_id does not match its namespace")

    auxiliary = {
        PurePosixPath(record["path"]).name: _verified_bytes(root, record)
        for record in manifest.get("auxiliary", [])
    }
    if "source-paths.json" not in auxiliary:
        raise ReconstructionError("Manifest does not reference source-paths.json")
    source_paths = json.loads(auxiliary["source-paths.json"])

    samples: list[PackagedSample] = []
    seen: set[str] = set()
    for shard in manifest["shards"]:
        _verified_bytes(root, shard)
        grouped: dict[str, dict[str, bytes]] = {}
        identity: dict[str, tuple[str, str]] = {}
        current = None
        with tarfile.open(root / shard["path"]) as archive:
            for member in archive:
                if not member.isfile():
                    raise ReconstructionError(f"Non-file tar member: {member.name}")
                key, episode, sample, component = _split_member(member.name, dataset_id)
                if key != current:
                    # WebDataset groups adjacent members; a key that reappears
                    # later would be read as a second, partial sample.
                    if key in seen:
                        raise ReconstructionError(f"Duplicate or split sample: {key}")
                    seen.add(key)
                    current = key
                    grouped[key] = {}
                    identity[key] = (episode, sample)
                if component in grouped[key]:
                    raise ReconstructionError(f"Duplicate member: {member.name}")
                grouped[key][component] = archive.extractfile(member).read()
        episode_counts: dict[str, int] = {}
        for key, members in grouped.items():
            episode, sample = identity[key]
            if "metadata.json" not in members:
                raise ReconstructionError(f"Sample without metadata.json: {key}")
            meta = json.loads(members["metadata.json"])
            if (meta["dataset_id"], meta["episode_id"], meta["sample_id"]) != (
                dataset_id,
                episode,
                sample,
            ):
                raise ReconstructionError(f"metadata.json identity mismatch: {key}")
            episode_counts[episode] = episode_counts.get(episode, 0) + 1
            samples.append(PackagedSample(key, episode, sample, shard["path"], members))
        if (
            len(grouped) != shard["samples"]
            or sorted(episode_counts) != shard["episodes"]
        ):
            raise ReconstructionError(f"Shard record mismatch: {shard['path']}")
        if episode_counts != shard["episode_samples"]:
            raise ReconstructionError(f"Episode sample counts: {shard['path']}")
    if len(samples) != manifest["samples"]:
        raise ReconstructionError("Manifest sample count does not match shards")
    if sorted({s.episode_id for s in samples}) != manifest["episodes"]:
        raise ReconstructionError("Manifest episodes do not match shard members")
    if sorted(source_paths["episodes"]) != manifest["episodes"]:
        raise ReconstructionError("source-paths.json episodes do not match manifest")

    if cross_check:
        urls = [
            f"file:{(root / shard['path']).resolve().as_posix()}"
            for shard in manifest["shards"]
        ]
        official = [
            (
                record["__key__"],
                {k: v for k, v in record.items() if not k.startswith("__")},
            )
            for record in wds.WebDataset(urls, shardshuffle=False, empty_check=False)
        ]
        ours = [(s.key, dict(s.members)) for s in samples]
        if official != ours:
            raise ReconstructionError("Official webdataset reader disagrees")
    return PackagedDataset(root, manifest, source_paths, tuple(samples))


def _native_target(base: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    # A value such as "C:\\x" is one POSIX part but an absolute path on Windows.
    windows = PureWindowsPath(relative)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in path.parts
        or ".." in windows.parts
        or not path.parts
    ):
        raise ReconstructionError(f"Unsafe native path: {relative!r}")
    target = base.joinpath(*path.parts)
    if not target.resolve().is_relative_to(base.resolve()):
        raise ReconstructionError(f"Unsafe native path: {relative!r}")
    return target


def _write_once(target: Path, payload: bytes) -> None:
    if target.exists():
        if target.read_bytes() != payload:
            raise ReconstructionError(f"Conflicting bytes for {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def reconstruct_native(
    dataset: PackagedDataset,
    destination: str | Path,
    *,
    shared_builders: Mapping[str, SharedBuilder] | None = None,
) -> dict[str, Path]:
    builders = dict(shared_builders or {})
    roots: dict[str, Path] = {}
    for episode_id in dataset.manifest["episodes"]:
        records = dataset.episode(episode_id)
        base = _native_target(Path(destination), episode_id)
        if base.exists():
            raise ReconstructionError(f"Destination already exists: {base}")
        base.mkdir(parents=True)
        by_id = {s.sample_id: s for s in records.samples}
        if sorted(by_id) != sorted(records.sample_paths):
            raise ReconstructionError(
                f"source-paths.json samples differ from episode {episode_id}"
            )
        for sample_id, paths in records.sample_paths.items():
            members = by_id[sample_id].members
            for component, relative in paths.items():
                if component not in members:
                    raise ReconstructionError(
                        f"Mapped component {component} missing from {sample_id}"
                    )
                _write_once(_native_target(base, relative), members[component])
        for role, relative in records.shared_paths.items():
            if role not in builders:
                raise ReconstructionError(f"No shared builder for role {role!r}")
            _write_once(_native_target(base, relative), builders[role](records, role))
        roots[episode_id] = base
    return roots


@dataclass(frozen=True)
class TreeDifference:
    """Relative paths that differ between an expected and an actual tree."""

    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    changed: Mapping[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.missing or self.extra or self.changed)

    def report(self) -> str:
        lines = [f"missing: {p}" for p in self.missing]
        lines += [f"extra: {p}" for p in self.extra]
        lines += [f"changed: {p}: {why}" for p, why in self.changed.items()]
        return "\n".join(lines)


def _files(root: Path) -> dict[str, Path]:
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def compare_trees(
    expected: str | Path,
    actual: str | Path,
    *,
    comparators: Mapping[str, Comparator] | None = None,
    ignore: Iterable[str] = (),
) -> TreeDifference:
    patterns = tuple(ignore)

    def kept(files: dict[str, Path]) -> dict[str, Path]:
        return {
            k: v
            for k, v in files.items()
            if not any(fnmatch.fnmatchcase(k, p) for p in patterns)
        }

    left, right = kept(_files(Path(expected))), kept(_files(Path(actual)))
    changed: dict[str, str] = {}
    for relative in sorted(left.keys() & right.keys()):
        a, b = left[relative].read_bytes(), right[relative].read_bytes()
        comparator = next(
            (
                c
                for p, c in (comparators or {}).items()
                if fnmatch.fnmatchcase(relative, p)
            ),
            None,
        )
        reason = (
            comparator(a, b)
            if comparator is not None
            else (None if a == b else "bytes differ")
        )
        if reason is not None:
            changed[relative] = reason
    return TreeDifference(
        tuple(sorted(left.keys() - right.keys())),
        tuple(sorted(right.keys() - left.keys())),
        changed,
    )


def assert_trees_equal(expected: str | Path, actual: str | Path, **kwargs: Any) -> None:
    difference = compare_trees(expected, actual, **kwargs)
    if difference:
        raise AssertionError(
            f"Reconstructed tree {actual} differs from {expected}:\n"
            + difference.report()
        )


# --- COCO reference hooks -------------------------------------------------


def coco_shared_document(records: EpisodeRecords, role: str) -> bytes:
    """Regenerate a COCO ``instances`` or ``panoptic`` document from samples."""
    categories: dict[int, dict[str, Any]] = {}
    images, annotations = [], []
    for sample in records.samples:
        record = sample.json()
        paths = records.sample_paths[sample.sample_id]
        image_component = next(c for c in paths if c != "panoptic.png")
        image = {
            **record["image"],
            "file_name": PurePosixPath(paths[image_component]).name,
        }
        images.append(image)
        for category in record["categories"]:
            categories.setdefault(category["id"], category)
        if role == "instances":
            annotations.extend(record["annotations"])
        elif role == "panoptic":
            if "panoptic" in record:
                annotations.append(
                    {
                        **record["panoptic"],
                        "file_name": PurePosixPath(paths["panoptic.png"]).name,
                    }
                )
        else:
            raise ReconstructionError(f"Unknown COCO shared role: {role}")
    document = {
        "categories": [categories[k] for k in sorted(categories)],
        "images": images,
        "annotations": annotations,
    }
    return json.dumps(document, allow_nan=False).encode()


def _canonical_coco(document: Mapping[str, Any], ignore_keys: set[str]) -> Any:
    names = {c["id"]: c["name"] for c in document.get("categories", [])}
    files = {i["id"]: i["file_name"] for i in document.get("images", [])}
    result: dict[str, Any] = {
        k: v
        for k, v in document.items()
        if k not in ignore_keys and k not in {"categories", "images", "annotations"}
    }
    result["categories"] = sorted(
        ({k: v for k, v in c.items() if k != "id"} for c in document["categories"]),
        key=lambda c: c["name"],
    )
    result["images"] = sorted(
        ({k: v for k, v in i.items() if k != "id"} for i in document["images"]),
        key=lambda i: i["file_name"],
    )
    annotations = []
    for entry in document["annotations"]:
        entry = {**entry, "image_id": files[entry["image_id"]]}
        if "segments_info" in entry:
            entry["segments_info"] = [
                {**s, "category_id": names[s["category_id"]]}
                for s in entry["segments_info"]
            ]
        else:
            entry.pop("id")
            entry["category_id"] = names[entry["category_id"]]
        annotations.append(entry)
    result["annotations"] = sorted(
        annotations, key=lambda a: json.dumps(a, sort_keys=True)
    )
    return result


def coco_documents_equivalent(
    expected: bytes,
    actual: bytes,
    *,
    ignore_keys: Iterable[str] = ("info", "licenses"),
) -> str | None:
    """Compare COCO documents modulo image, annotation and category renumbering."""
    ignored = set(ignore_keys)
    left = _canonical_coco(json.loads(expected), ignored)
    right = _canonical_coco(json.loads(actual), ignored)
    if left == right:
        return None
    keys = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
    return f"COCO documents differ in {keys}"
