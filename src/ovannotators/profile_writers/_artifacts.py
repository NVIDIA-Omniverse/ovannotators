# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shared metadata artifacts for profile writers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePath
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


def write_profile_artifacts(
    backend: Backend,
    *,
    profile: ProfileMetadata,
    writer_record: WriterRecord,
    loss_report: LossReport,
    conformance: ConformanceResult,
    payload_paths: tuple[str, ...],
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
    backend.write("_ovannotators/profile.json", canonical_json_bytes(profile))
    backend.write(
        "_ovannotators/writer_record.json", canonical_json_bytes(writer_record)
    )
    backend.write("_ovannotators/loss_report.json", canonical_json_bytes(loss_report))
    backend.write("_ovannotators/conformance.json", canonical_json_bytes(conformance))
    backend.write(
        "_ovannotators/payload_checksums.json",
        canonical_json_bytes({"algorithm": "sha256", "files": checksums}),
    )
    backend.write(
        "_ovannotators/_SUCCESS",
        canonical_json_bytes(
            {
                "meaning": "serialization_and_structural_validation_completed",
                "canonical_loader_acceptance": conformance.canonical_loader,
                "consumer_acceptance": conformance.consumer,
            }
        ),
    )
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
