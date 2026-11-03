# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos 3.5 multiview candidate registration and materialization."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse

import numpy as np

from ovannotators.config import config_to_dict
from ovannotators.exceptions import ConfigurationError
from ovannotators.profile_writers._artifacts import (
    PROFILE_ARTIFACT_PATHS,
    canonical_json_bytes,
    sha256_bytes,
    write_profile_artifacts,
)
from ovannotators.profile_writers._models import (
    ConformanceResult,
    EpisodeEnvelope,
    LossReport,
    NamedTensor,
    ProfileMetadata,
    SampleEnvelope,
    StreamDescriptor,
    WriterRecord,
)
from ovannotators.profile_writers._registry import ProfileWriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends import Backend

_PROFILE_ID = "cosmos_multiview"
_SCHEMA_VERSION = "1.0"
_ALLOWED_TASKS = frozenset({"t2v", "i2v", "transfer", "action"})
_ID_NAMESPACE = uuid.UUID("892fef45-5084-5ab9-9c92-742f8020e7ce")


@dataclass(frozen=True)
class CosmosQAConfig:
    """Versioned owner-supplied QA policy; values are never silently defaulted."""

    profile_id: str
    version: str
    max_alignment_error_ns: int
    geometry_reprojection_tolerance_px: float
    minimum_stream_coverage: float
    missing_data_policy: Literal["drop_sample", "fail"]
    warning_policy: Literal["forbid", "owner_approval_required"]

    def __post_init__(self) -> None:
        if not self.profile_id or not self.version:
            raise ConfigurationError("QA profile_id and version are required")
        if self.max_alignment_error_ns < 0:
            raise ConfigurationError("max_alignment_error_ns must be non-negative")
        if self.geometry_reprojection_tolerance_px < 0:
            raise ConfigurationError(
                "geometry_reprojection_tolerance_px must be non-negative"
            )
        if self.minimum_stream_coverage != 1.0:
            raise ConfigurationError(
                "minimum_stream_coverage must be 1.0 because every materialized "
                "asset requires complete target-grid coverage"
            )
        if self.missing_data_policy not in {"drop_sample", "fail"}:
            raise ConfigurationError(
                "missing_data_policy must be 'drop_sample' or 'fail'"
            )
        if self.warning_policy not in {"forbid", "owner_approval_required"}:
            raise ConfigurationError(
                "warning_policy must be 'forbid' or 'owner_approval_required'"
            )


@dataclass(frozen=True)
class CosmosMaterializationConfig:
    """Immutable, task-specific timestamp alignment and representation policy."""

    version: str
    duration_ns: int
    stride_ns: int
    reference_stream_id: str
    target_rate_hz: float
    selected_stream_ids: tuple[str, ...]
    alignment_by_stream_type: Mapping[str, str]
    output_encoding_by_stream_type: Mapping[str, str]
    tasks: tuple[str, ...]
    created_by: str
    object_key_policy: Literal["content_addressed"] = "content_addressed"

    def __post_init__(self) -> None:
        if not self.version or not self.reference_stream_id or not self.created_by:
            raise ConfigurationError(
                "materialization version, reference stream, and creator are required"
            )
        if (
            self.duration_ns <= 0
            or self.stride_ns <= 0
            or not math.isfinite(self.target_rate_hz)
            or not 0 < self.target_rate_hz <= 1_000_000_000
        ):
            raise ConfigurationError(
                "duration and stride must be positive, and target rate must be "
                "finite and in (0, 1e9] Hz"
            )
        if not self.selected_stream_ids or len(set(self.selected_stream_ids)) != len(
            self.selected_stream_ids
        ):
            raise ConfigurationError("selected_stream_ids must be non-empty and unique")
        if self.reference_stream_id not in self.selected_stream_ids:
            raise ConfigurationError("reference_stream_id must be a selected stream")
        if (
            not self.tasks
            or len(set(self.tasks)) != len(self.tasks)
            or not set(self.tasks) <= _ALLOWED_TASKS
        ):
            raise ConfigurationError(
                "tasks must be non-empty, unique, and use the Cosmos vocabulary "
                f"{sorted(_ALLOWED_TASKS)}"
            )
        if self.object_key_policy != "content_addressed":
            raise ConfigurationError("object_key_policy must be 'content_addressed'")


@dataclass(frozen=True)
class CosmosMultiviewWriterConfig:
    """Configuration for the exact Cosmos 3.5 two-table profile."""

    output_base_uri: str
    source_version: str
    domain: str
    owner: str
    license_uri: str
    default_reference_frame: str
    supported_uri_schemes: tuple[str, ...]
    materialization: CosmosMaterializationConfig
    qa: CosmosQAConfig
    schema_version: str = _SCHEMA_VERSION
    dataset_info_path: str = "metadata/dataset_info.json"
    sample_config_path: str = "metadata/sample_config.json"

    def __post_init__(self) -> None:
        for name in (
            "output_base_uri",
            "source_version",
            "domain",
            "owner",
            "license_uri",
            "default_reference_frame",
        ):
            if not str(getattr(self, name)).strip():
                raise ConfigurationError(f"{name} is required")
        if not self.supported_uri_schemes:
            raise ConfigurationError("supported_uri_schemes must not be empty")
        if self.schema_version != _SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported Cosmos schema version {self.schema_version!r}; "
                f"this writer supports exactly {_SCHEMA_VERSION}"
            )
        _validate_absolute_uri(self.output_base_uri, self.supported_uri_schemes)
        _validate_absolute_uri(self.license_uri, self.supported_uri_schemes)
        for path in (self.dataset_info_path, self.sample_config_path):
            if not path or path.startswith("/") or ".." in Path(path).parts:
                raise ConfigurationError("metadata paths must be safe relative paths")


@dataclass(frozen=True)
class CosmosValidationFinding:
    """Entity-keyed structural observation for an external QA handoff."""

    status: Literal["info", "fail"]
    check_id: str
    check_version: str
    message: str
    entity: Mapping[str, str]
    measured_value: Any = None
    expected_rule: Any = None


@dataclass(frozen=True)
class CosmosValidationRecord:
    """Minimum interoperable validation-run record from the source contract."""

    validation_run_id: str
    gate: Literal["episode", "sample"]
    status: Literal["unverified"]
    profile_id: str
    input_release: Mapping[str, Any]
    validator: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    started_at: None
    completed_at: None
    summary_metrics: Mapping[str, Any]
    findings: tuple[CosmosValidationFinding, ...]
    artifacts: tuple[Mapping[str, str], ...] = ()
    exclusions: tuple[Mapping[str, str], ...] = ()
    warning_approval: None = None


@dataclass(frozen=True)
class _AlignedSignal:
    tensor: NamedTensor
    policy: str
    target_timestamp_ns: int
    source_timestamps_ns: tuple[int, ...]


@ProfileWriterRegistry.register
class CosmosMultiviewWriter:
    """Write deterministic Cosmos 3.5 candidate episodes, assets, and samples.

    ``register_episodes`` and ``materialize_samples`` are deliberately separate.
    Both stages produce candidates and QA inputs; neither grants QA acceptance.
    """

    name: ClassVar[str] = _PROFILE_ID

    def __init__(self, config: CosmosMultiviewWriterConfig) -> None:
        self.config = config
        self._episodes: list[tuple[EpisodeEnvelope, tuple[SampleEnvelope, ...]]] = []
        self._current_episode: EpisodeEnvelope | None = None
        self._current_samples: list[SampleEnvelope] = []
        self._source_episode_ids: set[tuple[str, str]] = set()
        self._payload_paths: set[str] = set()
        self._episode_rows: list[dict[str, Any]] = []
        self._sample_rows: list[dict[str, Any]] = []
        self._episode_registered = False
        self._samples_materialized = False
        self._finalized = False
        self._dataset_id: str | None = None

    def begin_episode(self, episode: EpisodeEnvelope, backend: Backend) -> None:
        """Start buffering one source episode after validating its inventory."""
        del backend
        if self._finalized or self._episode_registered:
            raise ValueError("cannot add episodes after registration or finalization")
        if self._current_episode is not None:
            raise ValueError("end the active episode before beginning another")
        if self._dataset_id is not None and episode.dataset_id != self._dataset_id:
            raise ValueError("one candidate release may contain only one dataset_id")
        key = (episode.dataset_id, episode.episode_id)
        if key in self._source_episode_ids:
            raise ValueError("source_episode_id must be unique within dataset_id")
        selected = set(self.config.materialization.selected_stream_ids)
        descriptor_by_id = {stream.stream_id: stream for stream in episode.streams}
        declared = set(descriptor_by_id)
        missing = sorted(selected - declared)
        if missing:
            raise ValueError(f"selected streams are not declared: {missing}")
        if self.config.materialization.reference_stream_id not in declared:
            raise ValueError("reference_stream_id is not declared by the episode")
        mismatched_clocks = sorted(
            stream_id
            for stream_id in selected
            if descriptor_by_id[stream_id].source_clock != episode.source_clock
        )
        if mismatched_clocks:
            raise ValueError(
                "selected streams must use the episode source clock: "
                f"{mismatched_clocks}"
            )
        self._validate_descriptors(episode.streams)
        self._dataset_id = episode.dataset_id
        self._source_episode_ids.add(key)
        self._current_episode = episode
        self._current_samples = []

    def write_sample(self, sample: SampleEnvelope, backend: Backend) -> None:
        """Validate one authoritative timestamp and buffer it for both stages."""
        del backend
        episode = self._current_episode
        if episode is None or sample.episode_id != episode.episode_id:
            raise ValueError("sample does not belong to the active episode")
        if (
            self._current_samples
            and sample.timestamp_ns <= self._current_samples[-1].timestamp_ns
        ):
            raise ValueError("sample timestamps must be strictly increasing")
        descriptor_by_id = {stream.stream_id: stream for stream in episode.streams}
        declared = set(descriptor_by_id)
        supplied = set(sample.observations) | set(sample.policy_signals)
        unknown = sorted(supplied - declared)
        if unknown:
            raise ValueError(f"sample contains undeclared streams: {unknown}")
        unknown_missing = sorted(set(sample.missing_data) - declared)
        if unknown_missing:
            raise ValueError(
                f"missing_data names undeclared streams: {unknown_missing}"
            )
        incorrectly_missing = sorted(set(sample.missing_data) & supplied)
        if incorrectly_missing:
            raise ValueError(
                f"missing_data marks supplied streams as absent: {incorrectly_missing}"
            )
        all_signals = {**sample.observations, **sample.policy_signals}
        for stream_id, tensor in all_signals.items():
            self._validate_tensor(descriptor_by_id[stream_id], tensor)
        selected = set(self.config.materialization.selected_stream_ids)
        absent = selected - supplied
        unreported = sorted(absent - set(sample.missing_data))
        if unreported:
            raise ValueError(f"required selected streams are absent: {unreported}")
        if absent and self.config.qa.missing_data_policy == "fail":
            raise ValueError(f"selected streams have missing data: {sorted(absent)}")
        self._current_samples.append(sample)

    def end_episode(self, backend: Backend) -> None:
        """Close the source episode without prematurely materializing it."""
        del backend
        if self._current_episode is None:
            raise ValueError("no active episode")
        if not self._current_samples:
            raise ValueError("an episode must contain at least one sample")
        supplied = {
            stream_id
            for sample in self._current_samples
            for stream_id in self._sample_signal_names(sample)
        }
        missing_required = sorted(
            stream.stream_id
            for stream in self._current_episode.streams
            if stream.required and stream.stream_id not in supplied
        )
        if missing_required:
            raise ValueError(
                f"required streams have no samples in the episode: {missing_required}"
            )
        self._episodes.append((self._current_episode, tuple(self._current_samples)))
        self._current_episode = None
        self._current_samples = []

    def register_episodes(self, backend: Backend) -> tuple[Mapping[str, Any], ...]:
        """Deterministically write source resources and candidate ``episodes.lance``."""
        if self._current_episode is not None:
            raise ValueError("end the active episode before registration")
        if self._episode_registered:
            return tuple(self._episode_rows)
        if not self._episodes:
            raise ValueError("at least one episode is required")

        self._write_dataset_info(backend)
        rows: list[dict[str, Any]] = []
        findings: list[CosmosValidationFinding] = []
        for episode, samples in sorted(
            self._episodes, key=lambda item: item[0].episode_id
        ):
            logical_episode_id = self._episode_id(episode)
            descriptors = []
            for stream in sorted(episode.streams, key=lambda item: item.stream_id):
                descriptor = self._register_stream_resources(
                    backend, logical_episode_id, episode, stream, samples
                )
                descriptors.append(descriptor)
            rows.append(
                {
                    "episode_id": logical_episode_id,
                    "dataset_id": episode.dataset_id,
                    "source_episode_id": episode.episode_id,
                    "stream_types": sorted(
                        {stream.stream_type for stream in episode.streams}
                    ),
                    "streams": descriptors,
                    "dataset_info_uri": self._uri(self.config.dataset_info_path),
                }
            )
            findings.append(
                CosmosValidationFinding(
                    "info",
                    "cosmos.episode.inventory",
                    "1.0",
                    "candidate episode inventory requires independent QA",
                    {
                        "dataset_id": episode.dataset_id,
                        "episode_id": logical_episode_id,
                    },
                    len(descriptors),
                    "all declared streams represented exactly once",
                )
            )
        self._write_lance_dataset(backend, "episodes.lance", self._episode_table(rows))
        self._episode_rows = rows
        self._write_validation_record(backend, "episode", findings, len(rows))
        self._episode_registered = True
        return tuple(rows)

    def materialize_samples(self, backend: Backend) -> tuple[Mapping[str, Any], ...]:
        """Timestamp-align selected streams and write immutable sample assets."""
        if self._samples_materialized:
            return tuple(self._sample_rows)
        if not self._episode_registered:
            raise ValueError("register_episodes must complete before materialization")
        self._write_sample_config(backend)
        row_by_source = {row["source_episode_id"]: row for row in self._episode_rows}
        rows: list[dict[str, Any]] = []
        findings: list[CosmosValidationFinding] = []
        exclusions: list[dict[str, str]] = []
        for episode, samples in sorted(
            self._episodes, key=lambda item: item[0].episode_id
        ):
            episode_row = row_by_source[episode.episode_id]
            descriptor_by_id = {item.stream_id: item for item in episode.streams}
            coverage_end_ns = self._episode_coverage_end_ns(samples)
            for start_ns_abs in self._window_start_timestamps(
                episode.source_origin_ns, coverage_end_ns
            ):
                end_ns_abs = start_ns_abs + self.config.materialization.duration_ns
                if end_ns_abs > coverage_end_ns:
                    exclusions.append(
                        {
                            "episode_id": episode_row["episode_id"],
                            "source_start_offset_ns": str(
                                start_ns_abs - episode.source_origin_ns
                            ),
                            "reason": "incomplete half-open window exceeds source coverage",
                        }
                    )
                    break
                target_timestamps_ns = self._target_timestamps_ns(
                    start_ns_abs, end_ns_abs
                )
                aligned_signals, missing = self._align_window(
                    samples,
                    target_timestamps_ns,
                    descriptor_by_id,
                )
                if missing:
                    reason = f"incomplete aligned coverage: {missing}"
                    if self.config.qa.missing_data_policy == "fail":
                        raise ValueError(reason)
                    exclusions.append(
                        {
                            "episode_id": episode_row["episode_id"],
                            "source_start_offset_ns": str(
                                start_ns_abs - episode.source_origin_ns
                            ),
                            "reason": reason,
                        }
                    )
                    continue
                assets = self._materialize_assets(
                    backend,
                    episode,
                    descriptor_by_id,
                    aligned_signals,
                )
                start_ns = start_ns_abs - episode.source_origin_ns
                end_ns = end_ns_abs - episode.source_origin_ns
                for task in sorted(self.config.materialization.tasks):
                    sample_id = self._sample_id(
                        episode_row["episode_id"], start_ns, end_ns, task
                    )
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "episode_id": episode_row["episode_id"],
                            "task": task,
                            "source_start_offset_s": start_ns / 1_000_000_000.0,
                            "source_end_offset_s": end_ns / 1_000_000_000.0,
                            "assets_json": canonical_json_bytes(
                                {"schema_version": _SCHEMA_VERSION, "assets": assets}
                            )
                            .decode("utf-8")
                            .rstrip("\n"),
                            "caption": None,
                            "sample_config_uri": self._uri(
                                self.config.sample_config_path
                            ),
                            "created_by": self.config.materialization.created_by,
                        }
                    )
        rows.sort(key=lambda row: row["sample_id"])
        if not rows:
            raise ValueError("materialization produced no sample rows")
        episode_ids = {row["episode_id"] for row in self._episode_rows}
        if any(row["episode_id"] not in episode_ids for row in rows):
            raise ValueError("sample contains an invalid episode foreign key")
        self._write_lance_dataset(backend, "samples.lance", self._sample_table(rows))
        findings.append(
            CosmosValidationFinding(
                "info",
                "cosmos.sample.materialization",
                "1.0",
                "candidate assets require independent media, sync, geometry, and dedup QA",
                {"dataset_id": self._dataset_id or ""},
                len(rows),
                "all rows have valid episode foreign keys and immutable asset references",
            )
        )
        self._sample_rows = rows
        self._write_validation_record(
            backend, "sample", findings, len(rows), tuple(exclusions)
        )
        self._samples_materialized = True
        return tuple(rows)

    def validate(self, backend: Backend) -> ConformanceResult:
        """Run structural validation without claiming external Cosmos acceptance."""
        required = {
            self.config.dataset_info_path,
            self.config.sample_config_path,
            "validation/episode.json",
            "validation/sample.json",
        }
        messages: list[str] = []
        missing = sorted(path for path in required if not backend.exists(path))
        if missing:
            messages.append(f"missing required resources: {missing}")
        for prefix in ("episodes.lance", "samples.lance"):
            if not backend.list(prefix):
                messages.append(f"{prefix} is empty or missing")
        try:
            episode_table = self._read_lance_table(backend, "episodes.lance")
            sample_table = self._read_lance_table(backend, "samples.lance")
            if not episode_table.schema.equals(
                self._episode_schema(), check_metadata=True
            ):
                messages.append(
                    "episodes.lance has an incompatible Arrow schema or metadata"
                )
            if not sample_table.schema.equals(
                self._sample_schema(), check_metadata=True
            ):
                messages.append(
                    "samples.lance has an incompatible Arrow schema or metadata"
                )
            persisted_episode_ids = set(episode_table.column("episode_id").to_pylist())
            invalid_foreign_keys = sorted(
                set(sample_table.column("episode_id").to_pylist())
                - persisted_episode_ids
            )
            if invalid_foreign_keys:
                messages.append(
                    f"samples.lance has invalid episode foreign keys: {invalid_foreign_keys}"
                )
            episode_rows = episode_table.to_pylist()
            sample_rows = sample_table.to_pylist()
        except Exception as exc:  # noqa: BLE001 - report malformed persisted output
            messages.append(f"Lance structural read failed: {exc}")
            episode_rows = []
            sample_rows = []
        for row in episode_rows:
            self._validate_persisted_uri(
                backend,
                row["dataset_info_uri"],
                "dataset_info_uri",
                messages,
            )
            stream_types = sorted({stream["stream_type"] for stream in row["streams"]})
            if row["stream_types"] != stream_types:
                messages.append(
                    f"episode {row['episode_id']} stream_types does not match streams"
                )
            stream_ids = [stream["stream_id"] for stream in row["streams"]]
            if len(stream_ids) != len(set(stream_ids)):
                messages.append(f"episode {row['episode_id']} has duplicate stream IDs")
            for stream in row["streams"]:
                for field_name in (
                    "data_uri",
                    "calibration_uri",
                    "pose_uri",
                    "timestamps_uri",
                ):
                    uri = stream[field_name]
                    if uri is None:
                        continue
                    try:
                        relative = self._relative_uri(uri)
                    except ValueError as exc:
                        messages.append(str(exc))
                        continue
                    if not backend.exists(relative):
                        messages.append(f"unresolvable {field_name}: {uri}")
                    elif field_name == "data_uri":
                        try:
                            np.load(
                                io.BytesIO(backend.read(relative)), allow_pickle=False
                            )
                        except Exception as exc:  # noqa: BLE001 - validation boundary
                            messages.append(f"unusable source data {uri}: {exc}")
        for row in sample_rows:
            self._validate_persisted_uri(
                backend,
                row["sample_config_uri"],
                "sample_config_uri",
                messages,
            )
            try:
                manifest = json.loads(row["assets_json"])
            except Exception as exc:  # noqa: BLE001 - validation boundary
                messages.append(f"sample {row['sample_id']} assets_json failed: {exc}")
                continue
            if (
                canonical_json_bytes(manifest).decode().rstrip("\n")
                != row["assets_json"]
            ):
                messages.append(
                    f"sample {row['sample_id']} assets_json is not canonical"
                )
            if row["source_end_offset_s"] <= row["source_start_offset_s"]:
                messages.append(f"sample {row['sample_id']} has an empty window")
            if row["task"] not in _ALLOWED_TASKS:
                messages.append(f"sample {row['sample_id']} has unsupported task")
            assets = manifest.get("assets")
            if not isinstance(assets, list):
                messages.append(
                    f"sample {row['sample_id']} assets_json lacks an assets list"
                )
                continue
            for asset in assets:
                if not isinstance(asset, dict):
                    messages.append(
                        f"sample {row['sample_id']} has an invalid asset record"
                    )
                    continue
                asset_uri = asset.get("uri")
                if not isinstance(asset_uri, str):
                    messages.append(
                        f"sample {row['sample_id']} has an invalid asset URI"
                    )
                    continue
                try:
                    relative = self._relative_uri(asset_uri)
                except ValueError as exc:
                    messages.append(str(exc))
                    continue
                if not backend.exists(relative):
                    messages.append(f"missing asset {asset_uri}")
                else:
                    materialized_bytes = backend.read(relative)
                    if sha256_bytes(materialized_bytes) != asset.get("sha256"):
                        messages.append(f"asset checksum changed: {asset_uri}")
                    else:
                        try:
                            materialized = np.load(
                                io.BytesIO(materialized_bytes), allow_pickle=False
                            )
                            if list(materialized.shape) != asset.get("shape"):
                                messages.append(f"asset shape changed: {asset_uri}")
                        except Exception as exc:  # noqa: BLE001 - validation boundary
                            messages.append(f"asset decode failed {asset_uri}: {exc}")
                timestamps_uri = asset.get("timestamps_uri")
                if not isinstance(timestamps_uri, str):
                    messages.append(f"invalid timestamps URI {timestamps_uri}")
                    continue
                try:
                    timestamps_relative = self._relative_uri(timestamps_uri)
                except ValueError as exc:
                    messages.append(f"invalid timestamps URI {timestamps_uri}: {exc}")
                    continue
                if not backend.exists(timestamps_relative):
                    messages.append(f"missing timestamps asset {timestamps_uri}")
                else:
                    timestamps_bytes = backend.read(timestamps_relative)
                    if sha256_bytes(timestamps_bytes) != asset.get("timestamps_sha256"):
                        messages.append(
                            f"timestamps checksum changed: {timestamps_uri}"
                        )
                    else:
                        try:
                            _, pq = _arrow_modules()
                            timing = pq.read_table(
                                io.BytesIO(timestamps_bytes)
                            ).to_pylist()
                            targets = [row["target_timestamp_ns"] for row in timing]
                            native = [row["native_timestamp_ns"] for row in timing]
                            episode_times = [row["episode_time_ns"] for row in timing]
                            sources = [row["source_timestamps_ns"] for row in timing]
                            if targets != asset.get("target_timestamps_ns"):
                                messages.append(
                                    f"target timestamp evidence changed: {timestamps_uri}"
                                )
                            if native != targets:
                                messages.append(
                                    f"native timestamps do not describe materialized targets: {timestamps_uri}"
                                )
                            episode_origin_ns = asset.get("episode_origin_ns")
                            if not isinstance(
                                episode_origin_ns, int
                            ) or episode_times != [
                                target - episode_origin_ns for target in targets
                            ]:
                                messages.append(
                                    f"episode timestamps do not describe materialized targets: {timestamps_uri}"
                                )
                            if sources != asset.get("source_timestamps_ns"):
                                messages.append(
                                    f"source timestamp evidence changed: {timestamps_uri}"
                                )
                        except Exception as exc:  # noqa: BLE001 - validation boundary
                            messages.append(
                                f"timestamps decode failed {timestamps_uri}: {exc}"
                            )
        if backend.exists("_ovannotators/_SUCCESS"):
            messages.extend(self._completion_messages(backend))
        elif any(
            backend.exists(path)
            for path in PROFILE_ARTIFACT_PATHS
            if path != "_ovannotators/_SUCCESS"
        ):
            messages.append("profile artifacts exist without the final _SUCCESS marker")
        status = "failed" if messages else "passed"
        return ConformanceResult(
            status,
            status,
            semantic="not_run",
            canonical_loader="not_run",
            consumer="not_run",
            messages=tuple(messages),
            evidence={
                "episode_validation": self._uri("validation/episode.json"),
                "sample_validation": self._uri("validation/sample.json"),
            },
        )

    def finalize(self, backend: Backend) -> None:
        """Complete both stages and write common evidence and ``_SUCCESS`` last."""
        if self._finalized:
            return
        if self._current_episode is not None:
            raise ValueError("end the active episode before finalization")
        self.register_episodes(backend)
        self.materialize_samples(backend)
        conformance = self.validate(backend)
        if backend.exists("_ovannotators/_SUCCESS"):
            self._verify_existing_completion(backend, conformance)
            self._finalized = True
            return
        write_profile_artifacts(
            backend,
            profile=ProfileMetadata(
                _PROFILE_ID,
                "cosmos_multiview",
                _SCHEMA_VERSION,
                _SCHEMA_VERSION,
                tuple(self.config.materialization.selected_stream_ids),
                temporal_model="signed_int64_ns_half_open_windows",
                sharding_behavior="immutable_lance_tables_and_content_addressed_assets",
                resume_behavior="completed_immutable_objects_reused_in_new_candidate",
            ),
            writer_record=WriterRecord(
                self._dataset_id or "",
                self.name,
                "1",
                tuple(
                    f"{episode.dataset_id}:{episode.episode_id}:{self.config.source_version}"
                    for episode, _ in self._episodes
                ),
                tuple(sorted(self._payload_paths)),
                library_versions={"pylance": "11.0.0", "pyarrow": "25.0.1"},
                validator_versions={"cosmos_candidate_structural": "1.0"},
                validation_evidence={
                    "episode": self._uri("validation/episode.json"),
                    "sample": self._uri("validation/sample.json"),
                },
            ),
            loss_report=LossReport(
                consumed=(
                    "episode.streams",
                    "episode.source_origin_ns",
                    "sample.timestamp_ns",
                    "sample.observations",
                    "sample.policy_signals",
                    "sample.missing_data",
                ),
                transformed=(
                    "source envelopes to typed Lance rows",
                    "selected tensors to content-addressed NumPy assets",
                    "stride-based complete windows resampled to target-rate timestamps",
                    "per-stream nearest-timestamp alignment or linear interpolation",
                    "nanosecond offsets to exact half-open seconds columns",
                ),
                notes=(
                    "missing samples follow the versioned owner-supplied QA policy",
                    "QA and consumer acceptance remain external and unverified",
                ),
            ),
            conformance=conformance,
            payload_paths=tuple(sorted(self._payload_paths)),
        )
        self._finalized = True

    def _validate_descriptors(self, streams: tuple[StreamDescriptor, ...]) -> None:
        ids = {stream.stream_id for stream in streams}
        selected = set(self.config.materialization.selected_stream_ids)
        for stream in streams:
            if (
                stream.derived_from_stream_id
                and stream.derived_from_stream_id not in ids
            ):
                raise ValueError(
                    f"stream {stream.stream_id} derives from an unknown stream"
                )
            if stream.stream_id not in selected:
                continue
            if (
                stream.stream_type
                not in self.config.materialization.alignment_by_stream_type
            ):
                raise ValueError(
                    f"no alignment policy for stream type {stream.stream_type}"
                )
            policy = self.config.materialization.alignment_by_stream_type[
                stream.stream_type
            ]
            if policy not in {"nearest_timestamp", "linear_interpolation"}:
                raise ConfigurationError(
                    f"unsupported alignment policy {policy!r} for "
                    f"stream type {stream.stream_type}"
                )
            encoding = self.config.materialization.output_encoding_by_stream_type.get(
                stream.stream_type
            )
            if encoding != "npy":
                raise ConfigurationError(
                    f"stream type {stream.stream_type} must explicitly select the "
                    "currently supported 'npy' materialization encoding"
                )

    @staticmethod
    def _validate_tensor(descriptor: StreamDescriptor, tensor: NamedTensor) -> None:
        actual_shape = tuple(np.shape(tensor.data))
        if actual_shape != descriptor.shape:
            raise ValueError(
                f"stream {descriptor.stream_id} changed shape: "
                f"expected {descriptor.shape}, found {actual_shape}"
            )
        actual_dtype = str(np.asarray(tensor.data).dtype)
        if actual_dtype != descriptor.dtype:
            raise ValueError(
                f"stream {descriptor.stream_id} changed dtype: "
                f"expected {descriptor.dtype}, found {actual_dtype}"
            )
        expected_units = descriptor.units
        if isinstance(expected_units, str):
            expected_units = (expected_units,)
        if expected_units is not None and tuple(expected_units) != tensor.units:
            raise ValueError(
                f"stream {descriptor.stream_id} changed units: "
                f"expected {tuple(expected_units)}, found {tensor.units}"
            )
        if not tensor.representation:
            raise ValueError(
                f"stream {descriptor.stream_id} requires an explicit representation"
            )

    def _episode_coverage_end_ns(self, samples: tuple[SampleEnvelope, ...]) -> int:
        stream_ends: list[int] = []
        for stream_id in self.config.materialization.selected_stream_ids:
            timestamps = [
                sample.timestamp_ns
                for sample in samples
                if self._signal(sample, stream_id) is not None
            ]
            if not timestamps:
                raise ValueError(f"selected stream {stream_id} has no source samples")
            stream_ends.append(max(timestamps))
        return min(stream_ends)

    def _window_start_timestamps(
        self, episode_origin_ns: int, coverage_end_ns: int
    ) -> tuple[int, ...]:
        if coverage_end_ns <= episode_origin_ns:
            return ()
        stride_ns = self.config.materialization.stride_ns
        count = (coverage_end_ns - episode_origin_ns) // stride_ns + 1
        return tuple(episode_origin_ns + index * stride_ns for index in range(count))

    def _target_timestamps_ns(self, start_ns: int, end_ns: int) -> tuple[int, ...]:
        rate = Fraction(str(self.config.materialization.target_rate_hz))
        timestamps: list[int] = []
        index = 0
        while True:
            offset = round(
                Fraction(index * 1_000_000_000 * rate.denominator, rate.numerator)
            )
            timestamp = start_ns + offset
            if timestamp >= end_ns:
                break
            timestamps.append(timestamp)
            index += 1
        if not timestamps:
            raise ValueError(
                "target_rate_hz produces no timestamps in the sample window"
            )
        return tuple(timestamps)

    def _register_stream_resources(
        self,
        backend: Backend,
        logical_episode_id: str,
        episode: EpisodeEnvelope,
        stream: StreamDescriptor,
        samples: tuple[SampleEnvelope, ...],
    ) -> dict[str, Any]:
        stem = f"sources/{logical_episode_id}/{_safe_component(stream.stream_id)}"
        present = [
            sample
            for sample in samples
            if stream.stream_id in self._sample_signal_names(sample)
        ]
        tensors = [self._signal(sample, stream.stream_id) for sample in present]
        arrays = [np.asarray(tensor.data) for tensor in tensors if tensor is not None]
        if arrays:
            source_array = np.stack(arrays, axis=0)
        else:
            source_array = np.empty((0, *stream.shape), dtype=np.dtype(stream.dtype))
        data_path = f"{stem}/data.npy"
        data_payload = self._npy_bytes(source_array)
        self._write_immutable(backend, data_path, data_payload)
        timestamp_path = f"{stem}/timestamps.parquet"
        timestamp_payload = self._timestamp_parquet_bytes(episode, present)
        self._write_immutable(backend, timestamp_path, timestamp_payload)
        manifest_path = f"{stem}/stream.json"
        self._write_immutable(
            backend,
            manifest_path,
            canonical_json_bytes(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "source_version": self.config.source_version,
                    "episode_clock": episode.source_clock,
                    "descriptor": dataclasses.asdict(stream),
                    "source_sample_ids": [sample.sample_id for sample in present],
                    "data_uri": self._uri(data_path),
                    "data_sha256": sha256_bytes(data_payload),
                    "timestamps_uri": self._uri(timestamp_path),
                    "timestamps_sha256": sha256_bytes(timestamp_payload),
                }
            ),
        )
        calibration_uri = None
        if stream.geometry:
            calibration_path = f"{stem}/calibration.json"
            self._validate_geometry(stream)
            self._write_immutable(
                backend,
                calibration_path,
                canonical_json_bytes(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "stream_id": stream.stream_id,
                        **stream.geometry,
                    }
                ),
            )
            calibration_uri = self._uri(calibration_path)
        offsets = [sample.timestamp_ns - episode.source_origin_ns for sample in present]
        if not offsets:
            start_ns = end_ns = 0
        else:
            start_ns = min(offsets)
            end_ns = max(offsets) + 1
        return {
            "stream_id": stream.stream_id,
            "stream_type": stream.stream_type,
            "data_uri": self._uri(data_path),
            "calibration_uri": calibration_uri,
            "pose_uri": None,
            "timestamps_uri": self._uri(timestamp_path),
            "framerate": stream.nominal_rate_hz,
            "start_offset_s": start_ns / 1_000_000_000.0,
            "end_offset_s": end_ns / 1_000_000_000.0,
            "derived_from_stream_id": stream.derived_from_stream_id,
        }

    def _materialize_assets(
        self,
        backend: Backend,
        episode: EpisodeEnvelope,
        descriptors: Mapping[str, StreamDescriptor],
        aligned_signals: Mapping[str, tuple[_AlignedSignal, ...]],
    ) -> list[dict[str, Any]]:
        assets = []
        for stream_id in sorted(self.config.materialization.selected_stream_ids):
            aligned_sequence = aligned_signals[stream_id]
            first_tensor = aligned_sequence[0].tensor
            descriptor = descriptors[stream_id]
            materialized = np.stack(
                [np.asarray(aligned.tensor.data) for aligned in aligned_sequence],
                axis=0,
            )
            payload = self._npy_bytes(materialized)
            digest = sha256_bytes(payload)
            asset_path = f"objects/{digest[:2]}/{digest}.npy"
            self._write_immutable(backend, asset_path, payload)
            timestamp_payload = self._aligned_timestamp_parquet_bytes(
                episode, aligned_sequence
            )
            timestamp_digest = sha256_bytes(timestamp_payload)
            timestamp_path = (
                f"objects/{timestamp_digest[:2]}/{timestamp_digest}.timestamps.parquet"
            )
            self._write_immutable(backend, timestamp_path, timestamp_payload)
            asset: dict[str, Any] = {
                "stream_id": stream_id,
                "stream_type": descriptor.stream_type,
                "uri": self._uri(asset_path),
                "timestamps_uri": self._uri(timestamp_path),
                "sha256": digest,
                "timestamps_sha256": timestamp_digest,
                "representation": first_tensor.representation,
                "dtype": str(materialized.dtype),
                "shape": list(materialized.shape),
                "alignment_policy": self.config.materialization.alignment_by_stream_type[
                    descriptor.stream_type
                ],
                "target_timestamps_ns": [
                    aligned.target_timestamp_ns for aligned in aligned_sequence
                ],
                "source_timestamps_ns": [
                    list(aligned.source_timestamps_ns) for aligned in aligned_sequence
                ],
                "episode_origin_ns": episode.source_origin_ns,
            }
            if first_tensor.units:
                asset["units"] = list(first_tensor.units)
            if first_tensor.coordinate_frame:
                asset["coordinate_frame"] = first_tensor.coordinate_frame
            if descriptor.derived_from_stream_id:
                asset["derived_from_stream_id"] = descriptor.derived_from_stream_id
            if descriptor.geometry:
                calibration_path = (
                    f"sources/{self._episode_id(episode)}/"
                    f"{_safe_component(stream_id)}/calibration.json"
                )
                asset["calibration_uri"] = self._uri(calibration_path)
            assets.append(asset)
        return assets

    def _align_window(
        self,
        episode_samples: tuple[SampleEnvelope, ...],
        target_timestamps_ns: tuple[int, ...],
        descriptors: Mapping[str, StreamDescriptor],
    ) -> tuple[dict[str, tuple[_AlignedSignal, ...]], list[str]]:
        """Align a complete target-time grid without using frame or list indexes."""
        aligned: dict[str, tuple[_AlignedSignal, ...]] = {}
        missing: list[str] = []
        for stream_id in self.config.materialization.selected_stream_ids:
            descriptor = descriptors[stream_id]
            policy = self.config.materialization.alignment_by_stream_type[
                descriptor.stream_type
            ]
            candidates = [
                (source.timestamp_ns, signal)
                for source in episode_samples
                if (signal := self._signal(source, stream_id)) is not None
            ]
            stream_values: list[_AlignedSignal] = []
            for target_ns in target_timestamps_ns:
                value = self._align_signal(candidates, target_ns, policy)
                if value is None:
                    missing.append(f"{stream_id}@{target_ns}")
                    break
                stream_values.append(value)
            if len(stream_values) == len(target_timestamps_ns):
                aligned[stream_id] = tuple(stream_values)
        return aligned, sorted(missing)

    def _align_signal(
        self,
        candidates: list[tuple[int, NamedTensor]],
        target_ns: int,
        policy: str,
    ) -> _AlignedSignal | None:
        exact = next(
            (signal for timestamp, signal in candidates if timestamp == target_ns),
            None,
        )
        if exact is not None:
            return _AlignedSignal(exact, "exact_timestamp", target_ns, (target_ns,))
        if policy == "nearest_timestamp" and candidates:
            timestamp, signal = min(
                candidates,
                key=lambda item: (abs(item[0] - target_ns), item[0]),
            )
            if abs(timestamp - target_ns) <= self.config.qa.max_alignment_error_ns:
                return _AlignedSignal(signal, policy, target_ns, (timestamp,))
        elif policy == "linear_interpolation":
            before = [item for item in candidates if item[0] < target_ns]
            after = [item for item in candidates if item[0] > target_ns]
            if before and after:
                left_ns, left = max(before, key=lambda item: item[0])
                right_ns, right = min(after, key=lambda item: item[0])
                max_error = max(target_ns - left_ns, right_ns - target_ns)
                left_array = np.asarray(left.data)
                right_array = np.asarray(right.data)
                if (
                    max_error <= self.config.qa.max_alignment_error_ns
                    and left_array.shape == right_array.shape
                ):
                    weight = (target_ns - left_ns) / (right_ns - left_ns)
                    left_values = left_array.astype(np.float64, copy=False)
                    right_values = right_array.astype(np.float64, copy=False)
                    data = left_values + (right_values - left_values) * weight
                    if np.issubdtype(left_array.dtype, np.integer):
                        data = np.rint(data)
                    data = data.astype(left_array.dtype, copy=False)
                    return _AlignedSignal(
                        dataclasses.replace(left, data=data),
                        policy,
                        target_ns,
                        (left_ns, right_ns),
                    )
        return None

    def _write_dataset_info(self, backend: Backend) -> None:
        self._write_immutable(
            backend,
            self.config.dataset_info_path,
            canonical_json_bytes(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "dataset_id": self._dataset_id,
                    "domain": self.config.domain,
                    "owner": self.config.owner,
                    "license_uri": self.config.license_uri,
                    "default_reference_frame": self.config.default_reference_frame,
                    "episode_clock": {
                        "unit": "seconds",
                        "origin": "provider_defined_episode_start",
                        "timestamp_conversion": "round(seconds * 1e9)",
                    },
                    "source_version": self.config.source_version,
                }
            ),
        )

    def _write_sample_config(self, backend: Backend) -> None:
        data = config_to_dict(self.config.materialization)
        data.update(
            {
                "schema_version": _SCHEMA_VERSION,
                "episodes_uri": self._uri("episodes.lance"),
                "episodes_version": self.config.source_version,
                "qa_profile": config_to_dict(self.config.qa),
                "max_alignment_error_s": self.config.qa.max_alignment_error_ns
                / 1_000_000_000.0,
                "missing_data_policy": self.config.qa.missing_data_policy,
            }
        )
        self._write_immutable(
            backend, self.config.sample_config_path, canonical_json_bytes(data)
        )

    def _write_validation_record(
        self,
        backend: Backend,
        gate: Literal["episode", "sample"],
        findings: list[CosmosValidationFinding],
        count: int,
        exclusions: tuple[Mapping[str, str], ...] = (),
    ) -> None:
        release_hash = self._config_hash()
        record = CosmosValidationRecord(
            validation_run_id=str(
                uuid.uuid5(_ID_NAMESPACE, f"{self._dataset_id}:{gate}:{release_hash}")
            ),
            gate=gate,
            status="unverified",
            profile_id=f"{self.config.qa.profile_id}@{self.config.qa.version}",
            input_release={
                "source_version": self.config.source_version,
                "config_sha256": release_hash,
                "table_uri": self._uri(f"{gate}s.lance"),
            },
            validator={
                "name": "ovannotators.cosmos_candidate_structural",
                "version": "1.0",
                "dependencies": {"pylance": "11.0.0", "pyarrow": "25.0.1"},
                "configuration": config_to_dict(self.config.qa),
            },
            thresholds={
                "id": self.config.qa.profile_id,
                "version": self.config.qa.version,
                **config_to_dict(self.config.qa),
            },
            started_at=None,
            completed_at=None,
            summary_metrics={"candidate_rows": count},
            findings=tuple(findings),
            exclusions=exclusions,
        )
        self._write_immutable(
            backend, f"validation/{gate}.json", canonical_json_bytes(record)
        )

    def _write_lance_dataset(self, backend: Backend, prefix: str, table: Any) -> None:
        existing_paths = backend.list(prefix)
        if existing_paths:
            try:
                existing = self._read_lance_table(backend, prefix)
            except Exception as exc:
                raise ValueError(
                    f"incomplete or unreadable immutable Lance tree at {prefix}"
                ) from exc
            if not existing.equals(table, check_metadata=True):
                raise ValueError(
                    f"immutable {prefix} already exists with different logical rows"
                )
            self._payload_paths.update(existing_paths)
            return
        try:
            import lance
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigurationError(
                "Cosmos Multiview writing requires ovannotators[cosmos-multiview]"
            ) from exc
        _require_dependency_version("pylance", "11.0.0")
        with tempfile.TemporaryDirectory(prefix="ovannotators-cosmos-lance-") as tmp:
            root = Path(tmp) / prefix
            # Lance storage 2.2 is required for the caption map field.
            lance.write_dataset(
                table,
                root.as_posix(),
                mode="create",
                data_storage_version="2.2",
            )
            for source in sorted(path for path in root.rglob("*") if path.is_file()):
                relative = f"{prefix}/{source.relative_to(root).as_posix()}"
                self._write_immutable(backend, relative, source.read_bytes())

    def _read_lance_table(self, backend: Backend, prefix: str) -> Any:
        try:
            import lance
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigurationError(
                "Cosmos Multiview validation requires ovannotators[cosmos-multiview]"
            ) from exc
        _require_dependency_version("pylance", "11.0.0")
        paths = backend.list(prefix)
        if not paths:
            raise ValueError(f"{prefix} has no files")
        with tempfile.TemporaryDirectory(prefix="ovannotators-cosmos-validate-") as tmp:
            root = Path(tmp)
            for path in paths:
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(backend.read(path))
            return lance.dataset((root / prefix).as_posix()).to_table()

    def _episode_table(self, rows: list[dict[str, Any]]) -> Any:
        pa, _ = _arrow_modules()
        return pa.Table.from_pylist(rows, schema=self._episode_schema())

    def _episode_schema(self) -> Any:
        pa, _ = _arrow_modules()
        stream_type = pa.struct(
            [
                ("stream_id", pa.string()),
                ("stream_type", pa.string()),
                ("data_uri", pa.string()),
                ("calibration_uri", pa.string()),
                ("pose_uri", pa.string()),
                ("timestamps_uri", pa.string()),
                ("framerate", pa.float64()),
                ("start_offset_s", pa.float64()),
                ("end_offset_s", pa.float64()),
                ("derived_from_stream_id", pa.string()),
            ]
        )
        return pa.schema(
            [
                ("episode_id", pa.string()),
                ("dataset_id", pa.string()),
                ("source_episode_id", pa.string()),
                ("stream_types", pa.list_(pa.string())),
                ("streams", pa.list_(stream_type)),
                ("dataset_info_uri", pa.string()),
            ],
            metadata={b"cosmos_schema_version": _SCHEMA_VERSION.encode()},
        )

    def _sample_table(self, rows: list[dict[str, Any]]) -> Any:
        pa, _ = _arrow_modules()
        return pa.Table.from_pylist(rows, schema=self._sample_schema())

    def _sample_schema(self) -> Any:
        pa, _ = _arrow_modules()
        caption_type = pa.struct(
            [
                ("text", pa.string()),
                ("language", pa.string()),
                ("per_stream", pa.map_(pa.string(), pa.string())),
            ]
        )
        return pa.schema(
            [
                ("sample_id", pa.string()),
                ("episode_id", pa.string()),
                ("task", pa.string()),
                ("source_start_offset_s", pa.float64()),
                ("source_end_offset_s", pa.float64()),
                ("assets_json", pa.string()),
                ("caption", caption_type),
                ("sample_config_uri", pa.string()),
                ("created_by", pa.string()),
            ],
            metadata={b"cosmos_schema_version": _SCHEMA_VERSION.encode()},
        )

    def _timestamp_parquet_bytes(
        self,
        episode: EpisodeEnvelope,
        samples: tuple[SampleEnvelope, ...] | list[SampleEnvelope],
    ) -> bytes:
        pa, pq = _arrow_modules()
        table = pa.table(
            {
                "sample_index": pa.array(range(len(samples)), type=pa.int64()),
                "native_timestamp_ns": pa.array(
                    [sample.timestamp_ns for sample in samples], type=pa.int64()
                ),
                "episode_time_ns": pa.array(
                    [
                        sample.timestamp_ns - episode.source_origin_ns
                        for sample in samples
                    ],
                    type=pa.int64(),
                ),
            }
        )
        output = pa.BufferOutputStream()
        pq.write_table(table, output, compression="zstd")
        return output.getvalue().to_pybytes()

    def _aligned_timestamp_parquet_bytes(
        self,
        episode: EpisodeEnvelope,
        aligned: tuple[_AlignedSignal, ...],
    ) -> bytes:
        pa, pq = _arrow_modules()
        table = pa.table(
            {
                "sample_index": pa.array(range(len(aligned)), type=pa.int64()),
                "native_timestamp_ns": pa.array(
                    [value.target_timestamp_ns for value in aligned],
                    type=pa.int64(),
                ),
                "episode_time_ns": pa.array(
                    [
                        value.target_timestamp_ns - episode.source_origin_ns
                        for value in aligned
                    ],
                    type=pa.int64(),
                ),
                "target_timestamp_ns": pa.array(
                    [value.target_timestamp_ns for value in aligned], type=pa.int64()
                ),
                "source_timestamps_ns": pa.array(
                    [list(value.source_timestamps_ns) for value in aligned],
                    type=pa.list_(pa.int64()),
                ),
                "alignment_policy": pa.array(
                    [value.policy for value in aligned], type=pa.string()
                ),
            }
        )
        output = pa.BufferOutputStream()
        pq.write_table(table, output, compression="zstd")
        return output.getvalue().to_pybytes()

    @staticmethod
    def _npy_bytes(array: np.ndarray) -> bytes:
        output = io.BytesIO()
        np.save(output, array, allow_pickle=False)
        return output.getvalue()

    def _validate_geometry(self, stream: StreamDescriptor) -> None:
        geometry = stream.geometry
        required = {
            "camera_model",
            "image_size_px",
            "reference_frame_id",
            "T_reference_from_sensor",
        }
        if not required <= set(geometry):
            raise ValueError(
                f"camera stream {stream.stream_id} lacks complete geometry metadata"
            )
        transform = geometry["T_reference_from_sensor"]
        if len(transform) != 16 or not all(np.isfinite(transform)):
            raise ValueError(f"camera stream {stream.stream_id} has invalid transform")
        image_size = geometry["image_size_px"]
        if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
            raise ValueError(f"camera stream {stream.stream_id} has invalid image size")

    def _write_immutable(self, backend: Backend, path: str, data: bytes) -> None:
        if backend.exists(path):
            if backend.read(path) != data:
                raise ValueError(
                    f"immutable output already exists with different bytes: {path}"
                )
        else:
            backend.write(path, data)
        self._payload_paths.add(path)

    def _verify_existing_completion(
        self, backend: Backend, conformance: ConformanceResult
    ) -> None:
        if not conformance.success_marker_allowed:
            raise ValueError(
                "existing candidate no longer passes structural validation"
            )
        stored = json.loads(backend.read("_ovannotators/payload_checksums.json"))
        if stored.get("algorithm") != "sha256":
            raise ValueError(
                "existing completion uses an unsupported checksum algorithm"
            )
        expected_paths = set(stored.get("files", {}))
        if expected_paths != self._payload_paths:
            raise ValueError(
                "existing completion payload set differs from deterministic rerun"
            )
        for path, digest in stored["files"].items():
            if not backend.exists(path) or sha256_bytes(backend.read(path)) != digest:
                raise ValueError(f"existing completed payload changed: {path}")
        profile = json.loads(backend.read("_ovannotators/profile.json"))
        if profile.get("profile_id") != _PROFILE_ID:
            raise ValueError("existing completion belongs to a different profile")

    def _completion_messages(self, backend: Backend) -> list[str]:
        messages: list[str] = []
        missing = sorted(
            path for path in PROFILE_ARTIFACT_PATHS if not backend.exists(path)
        )
        if missing:
            return [f"completed output is missing profile artifacts: {missing}"]
        try:
            success = json.loads(backend.read("_ovannotators/_SUCCESS"))
            profile = json.loads(backend.read("_ovannotators/profile.json"))
            conformance = json.loads(backend.read("_ovannotators/conformance.json"))
            checksums = json.loads(backend.read("_ovannotators/payload_checksums.json"))
        except Exception as exc:  # noqa: BLE001 - validation boundary
            return [f"completed profile artifacts are invalid JSON: {exc}"]
        if (
            success.get("meaning")
            != "serialization_and_structural_validation_completed"
        ):
            messages.append("_SUCCESS has an incompatible meaning")
        if profile.get("profile_id") != _PROFILE_ID:
            messages.append("completed output belongs to a different profile")
        if (
            conformance.get("serialization") != "passed"
            or conformance.get("structural") != "passed"
        ):
            messages.append("completed conformance does not record structural success")
        if checksums.get("algorithm") != "sha256" or not isinstance(
            checksums.get("files"), dict
        ):
            messages.append("payload checksum manifest is invalid")
            return messages
        recorded = checksums["files"]
        actual_payloads = {
            path for path in backend.list() if not path.startswith("_ovannotators/")
        }
        if set(recorded) != actual_payloads:
            messages.append(
                "payload checksum manifest does not cover the exact payload set"
            )
        for path, digest in recorded.items():
            if not backend.exists(path):
                messages.append(f"completed payload is missing: {path}")
            elif sha256_bytes(backend.read(path)) != digest:
                messages.append(f"completed payload checksum changed: {path}")
        return messages

    def _episode_id(self, episode: EpisodeEnvelope) -> str:
        return str(
            uuid.uuid5(
                _ID_NAMESPACE,
                f"episode:{episode.dataset_id}:{episode.episode_id}:{self.config.source_version}",
            )
        )

    def _sample_id(self, episode_id: str, start_ns: int, end_ns: int, task: str) -> str:
        return str(
            uuid.uuid5(
                _ID_NAMESPACE,
                f"sample:{episode_id}:{start_ns}:{end_ns}:{task}:{self._config_hash()}",
            )
        )

    def _config_hash(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(config_to_dict(self.config))
        ).hexdigest()

    def _uri(self, relative: str) -> str:
        return f"{self.config.output_base_uri.rstrip('/')}/{relative}"

    def _relative_uri(self, uri: str) -> str:
        prefix = self.config.output_base_uri.rstrip("/") + "/"
        if not uri.startswith(prefix):
            raise ValueError(f"URI is outside the immutable candidate root: {uri}")
        parsed = urlparse(uri)
        relative = uri[len(prefix) :]
        if (
            not relative
            or parsed.query
            or parsed.fragment
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise ValueError(f"URI contains an unsafe relative path: {uri}")
        return relative

    def _validate_persisted_uri(
        self,
        backend: Backend,
        uri: Any,
        field_name: str,
        messages: list[str],
    ) -> None:
        if not isinstance(uri, str):
            messages.append(f"invalid {field_name}: {uri!r}")
            return
        try:
            relative = self._relative_uri(uri)
        except ValueError as exc:
            messages.append(str(exc))
            return
        if not backend.exists(relative):
            messages.append(f"unresolvable {field_name}: {uri}")

    @staticmethod
    def _sample_signal_names(sample: SampleEnvelope) -> set[str]:
        return set(sample.observations) | set(sample.policy_signals)

    @staticmethod
    def _signal(sample: SampleEnvelope, stream_id: str) -> NamedTensor | None:
        signal = sample.observations.get(stream_id)
        if signal is not None:
            return signal
        return sample.policy_signals.get(stream_id)


def _validate_absolute_uri(uri: str, supported_schemes: tuple[str, ...]) -> None:
    parsed = urlparse(uri)
    if parsed.scheme not in supported_schemes:
        raise ConfigurationError(
            f"URI scheme {parsed.scheme!r} is not in supported_uri_schemes"
        )
    if not parsed.scheme or (
        parsed.scheme == "file" and not parsed.path.startswith("/")
    ):
        raise ConfigurationError(f"URI must be absolute: {uri}")
    if parsed.scheme != "file" and not parsed.netloc:
        raise ConfigurationError(f"URI must include an authority: {uri}")


def _safe_component(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _arrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigurationError(
            "Cosmos Multiview writing requires ovannotators[cosmos-multiview]"
        ) from exc
    _require_dependency_version("pyarrow", "25.0.1")
    return pa, pq


def _require_dependency_version(distribution: str, expected: str) -> None:
    try:
        actual = version(distribution)
    except PackageNotFoundError as exc:  # pragma: no cover - import guards catch this
        raise ConfigurationError(
            f"required dependency {distribution} is not installed"
        ) from exc
    if actual != expected:
        raise ConfigurationError(
            f"{_PROFILE_ID} requires {distribution}=={expected}; found {actual}"
        )


__all__ = [
    "CosmosMaterializationConfig",
    "CosmosMultiviewWriter",
    "CosmosMultiviewWriterConfig",
    "CosmosQAConfig",
    "CosmosValidationFinding",
    "CosmosValidationRecord",
]
