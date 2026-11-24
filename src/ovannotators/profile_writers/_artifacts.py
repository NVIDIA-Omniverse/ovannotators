# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shared metadata artifacts for profile writers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ovannotators.backends import Backend
    from ovannotators.profile_writers._models import (
        ConformanceResult,
        LossReport,
        ProfileMetadata,
        WriterRecord,
    )

PROFILE_ARTIFACT_DIR = "_ovannotators"
_CHECKSUMS_PATH = "_ovannotators/payload_checksums.json"
_SUCCESS_PATH = "_ovannotators/_SUCCESS"
_CONFORMANCE_PATH = "_ovannotators/conformance.json"
_MARKER_MEANING = "serialization_and_structural_validation_completed"
PROFILE_ARTIFACT_PATHS = frozenset(
    {
        "_ovannotators/profile.json",
        "_ovannotators/writer_record.json",
        "_ovannotators/loss_report.json",
        "_ovannotators/conformance.json",
        "_ovannotators/payload_checksums.json",
        "_ovannotators/_SUCCESS",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize profile metadata to stable UTF-8 JSON bytes."""
    return (
        json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def write_matching_or_new(backend: Backend, path: str, data: bytes) -> None:
    """Atomically create a file or verify that an existing file has identical bytes."""
    if not require_atomic_create(backend)(path, data) and backend.read(path) != data:
        raise ValueError(f"destination already contains different bytes: {path}")


def require_atomic_create(backend: Backend):
    """Return the backend's optional atomic create method or raise an explicit error."""
    create = getattr(backend, "write_if_absent", None)
    if not callable(create):
        raise ValueError(
            "native LeRobot publication needs a backend with atomic write_if_absent"
        )
    return create


def publish_native_payload(
    backend: Backend, root: Path, paths: Sequence[Path]
) -> list[str]:
    """Claim the destination for one staged payload and publish its exact bytes."""
    files = {}
    for path in paths:
        with (root / path).open("rb") as source:
            files[path.as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    backend.makedirs(PROFILE_ARTIFACT_DIR)
    write_matching_or_new(backend, _CHECKSUMS_PATH, _checksum_manifest(files))
    for path, digest in files.items():
        data = (root / path).read_bytes()
        if sha256_bytes(data) != digest:
            raise ValueError(f"staged payload changed during publication: {path}")
        write_matching_or_new(backend, path, data)
    return list(files)


def _checksum_manifest(files: dict[str, str]) -> bytes:
    return canonical_json_bytes({"algorithm": "sha256", "files": files})


def validate_completion_marker(backend: Backend, paths: set[str]) -> list[str]:
    """Check the native completion marker and its agreement with conformance evidence."""
    if _SUCCESS_PATH not in paths:
        return [f"missing final {_SUCCESS_PATH} marker"]
    try:
        marker = json.loads(backend.read(_SUCCESS_PATH))
        evidence = json.loads(backend.read(_CONFORMANCE_PATH))
    except Exception as exc:
        return [
            f"invalid _SUCCESS or conformance evidence: {type(exc).__name__}: {exc}"
        ]
    if (
        not isinstance(evidence, dict)
        or evidence.get("serialization") != "passed"
        or evidence.get("structural") != "passed"
        or evidence.get("canonical_loader") not in ("passed", "failed", "not_run")
        or evidence.get("consumer") not in ("passed", "failed", "not_run")
        or marker
        != {
            "meaning": _MARKER_MEANING,
            "canonical_loader_acceptance": evidence["canonical_loader"],
            "consumer_acceptance": evidence["consumer"],
        }
    ):
        return ["_SUCCESS marker does not match completed conformance evidence"]
    return []


def require_disjoint_disk_staging(stage: Path, backend: Backend) -> None:
    """Reject disk destinations that overlap a writer's removable staging tree."""
    from ovannotators.backends import DiskBackend

    if isinstance(backend, DiskBackend):
        stage, root = stage.resolve(), backend._root.resolve()
        if stage.is_relative_to(root) or root.is_relative_to(stage):
            raise ValueError("staging_dir and DiskBackend root_dir must not overlap")


def payload_paths_from_listing(paths: list[str] | set[str]) -> set[str]:
    """Return the dataset files, excluding its profile evidence directory."""
    return {path for path in paths if not path.startswith(f"{PROFILE_ARTIFACT_DIR}/")}


def validate_payload_checksums(backend: Backend, paths: set[str]) -> list[str]:
    """Require complete SHA-256 evidence and report unreadable payloads as failures."""
    manifest_path = _CHECKSUMS_PATH
    if manifest_path not in paths:
        return [f"missing checksum manifest: {manifest_path}"]
    try:
        manifest = json.loads(backend.read(manifest_path))
    except Exception as exc:
        return [f"invalid checksum manifest: {type(exc).__name__}: {exc}"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("algorithm") != "sha256"
        or not isinstance(manifest.get("files"), dict)
    ):
        return [
            "invalid checksum manifest: expected algorithm sha256 and a files mapping"
        ]
    checksums = manifest["files"]
    payload_paths = payload_paths_from_listing(paths)
    messages: list[str] = []
    if set(checksums) != payload_paths:
        messages.append("checksum paths do not exactly match serialized payloads")
    for path, expected in checksums.items():
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            messages.append(f"invalid SHA-256 digest: {path}")
            continue
        if path not in payload_paths:
            messages.append(f"checksummed payload is missing or reserved: {path}")
            continue
        try:
            if sha256_bytes(backend.read(path)) != expected:
                messages.append(f"checksum mismatch: {path}")
        except Exception as exc:
            messages.append(
                f"unreadable checksummed payload {path}: {type(exc).__name__}: {exc}"
            )
    return messages


def write_profile_artifacts(
    backend: Backend,
    *,
    profile: ProfileMetadata,
    writer_record: WriterRecord,
    loss_report: LossReport,
    conformance: ConformanceResult,
    payload_paths: tuple[str, ...],
    retryable: bool = False,
) -> Mapping[str, str]:
    """Write common evidence and, when allowed, write ``_SUCCESS`` last.

    Args:
        backend: Dataset destination.
        profile: Exact output-profile contract.
        writer_record: Reproducibility and implementation versions.
        loss_report: Declared semantic transformations and omissions.
        conformance: Independent acceptance-layer statuses.
        payload_paths: Complete payload paths to checksum. Reserved artifact
            paths are forbidden because their own content contains checksums.
        retryable: Create artifacts atomically or verify that existing bytes match.

    Returns:
        A sorted mapping from payload path to SHA-256 digest.

    Raises:
        ValueError: If a payload path is reserved, missing, or duplicated, or
            if structural completion does not permit a success marker.
    """
    if len(payload_paths) != len(set(payload_paths)):
        raise ValueError("payload_paths must not contain duplicates")
    reserved = sorted(set(payload_paths) & PROFILE_ARTIFACT_PATHS)
    if reserved:
        raise ValueError(f"payload_paths contains reserved paths: {reserved}")

    checksums: dict[str, str] = {}
    for path in sorted(payload_paths):
        if not backend.exists(path):
            raise ValueError(f"cannot checksum missing payload: {path}")
        checksums[path] = sha256_bytes(backend.read(path))

    if not conformance.success_marker_allowed:
        raise ValueError(
            "_SUCCESS requires passed serialization and structural validation"
        )

    backend.makedirs(PROFILE_ARTIFACT_DIR)
    artifacts = {
        "profile.json": profile,
        "writer_record.json": writer_record,
        "loss_report.json": loss_report,
        "conformance.json": conformance,
        "payload_checksums.json": {"algorithm": "sha256", "files": checksums},
        "_SUCCESS": {
            "meaning": _MARKER_MEANING,
            "canonical_loader_acceptance": conformance.canonical_loader,
            "consumer_acceptance": conformance.consumer,
        },
    }
    for name, value in artifacts.items():
        path = f"{PROFILE_ARTIFACT_DIR}/{name}"
        data = canonical_json_bytes(value)
        if retryable:
            write_matching_or_new(backend, path, data)
        else:
            backend.write(path, data)
    return checksums


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value
