# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Producer-neutral contracts shared by dataset profile writers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

CheckStatus = Literal["passed", "failed", "not_run"]


def _require_text(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class NamedTensor:
    """A tensor with semantic feature and coordinate information.

    Args:
        data: Tensor payload. NumPy conversion is deliberately left to the
            selected profile so GPU-resident producer arrays can stay lazy.
        feature_names: Stable names for the tensor's semantic features.
        units: Units for each feature, or one unit applying to the full tensor.
        representation: Representation contract such as ``joint_position``.
        coordinate_frame: Frame in which spatial values are expressed.
    """

    data: Any
    feature_names: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    representation: str = ""
    coordinate_frame: str | None = None

    def __post_init__(self) -> None:
        if self.units and len(self.units) not in (1, len(self.feature_names)):
            raise ValueError("units must contain one value or one per feature")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the payload shape without forcing a dtype conversion."""
        return tuple(np.shape(self.data))


@dataclass(frozen=True)
class StreamDescriptor:
    """Description of one independently timed observation or policy stream."""

    stream_id: str
    stream_type: str
    source_clock: str
    nominal_rate_hz: float | None
    dtype: str
    shape: tuple[int, ...]
    units: str | tuple[str, ...] | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)
    encoding: str | None = None
    derived_from_stream_id: str | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)
    immutable_uri_metadata: Mapping[str, str] = field(default_factory=dict)
    required: bool = True

    def __post_init__(self) -> None:
        _require_text("stream_id", self.stream_id)
        _require_text("stream_type", self.stream_type)
        _require_text("source_clock", self.source_clock)
        _require_text("dtype", self.dtype)
        if self.nominal_rate_hz is not None and self.nominal_rate_hz <= 0:
            raise ValueError("nominal_rate_hz must be positive when provided")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("shape dimensions must be non-negative")


@dataclass(frozen=True)
class EpisodeEnvelope:
    """Identity, timing, stream, task, and provenance for one episode."""

    dataset_id: str
    sequence_id: str
    episode_id: str
    task: str
    embodiment: str
    source_clock: str
    source_origin_ns: int
    nominal_rate_hz: float | None
    streams: tuple[StreamDescriptor, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "sequence_id",
            "episode_id",
            "task",
            "embodiment",
            "source_clock",
        ):
            _require_text(field_name, getattr(self, field_name))
        if self.nominal_rate_hz is not None and self.nominal_rate_hz <= 0:
            raise ValueError("nominal_rate_hz must be positive when provided")
        stream_ids = [stream.stream_id for stream in self.streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("stream IDs must be unique within an episode")


@dataclass(frozen=True)
class SampleEnvelope:
    """One source-timed sample containing observations and policy signals."""

    episode_id: str
    sample_id: str
    frame_id: int
    timestamp_ns: int
    observations: Mapping[str, NamedTensor]
    policy_signals: Mapping[str, NamedTensor] = field(default_factory=dict)
    missing_data: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("episode_id", self.episode_id)
        _require_text("sample_id", self.sample_id)
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not isinstance(self.timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer number of nanoseconds")
        overlap = set(self.observations) & set(self.policy_signals)
        if overlap:
            raise ValueError(
                f"signal names must be unique across maps: {sorted(overlap)}"
            )


@dataclass(frozen=True)
class ProfileMetadata:
    """Machine-readable identity and behavior contract for a profile."""

    profile_id: str
    profile_family: str
    profile_version: str
    schema_version: str
    required_modalities: tuple[str, ...]
    optional_modalities: tuple[str, ...] = ()
    temporal_model: str = "authoritative_source_timestamps_ns"
    episode_semantics: str = "explicit_begin_end"
    sharding_behavior: str = "profile_defined"
    resume_behavior: str = "reject_incomplete"

    def __post_init__(self) -> None:
        for field_name in (
            "profile_id",
            "profile_family",
            "profile_version",
            "schema_version",
            "temporal_model",
            "episode_semantics",
            "sharding_behavior",
            "resume_behavior",
        ):
            _require_text(field_name, getattr(self, field_name))


@dataclass(frozen=True)
class WriterRecord:
    """Reproducibility record for one completed profile serialization."""

    dataset_id: str
    writer_name: str
    writer_version: str
    input_identifiers: tuple[str, ...]
    output_identifiers: tuple[str, ...]
    codec_versions: Mapping[str, str] = field(default_factory=dict)
    library_versions: Mapping[str, str] = field(default_factory=dict)
    validator_versions: Mapping[str, str] = field(default_factory=dict)
    canonical_loader_versions: Mapping[str, str] = field(default_factory=dict)
    accepted_warning_overrides: tuple[str, ...] = ()
    validation_evidence: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LossReport:
    """Declared semantic handling of producer-envelope fields."""

    consumed: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()
    synthesized: tuple[str, ...] = ()
    quantized: tuple[str, ...] = ()
    transformed: tuple[str, ...] = ()
    unrepresentable: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConformanceResult:
    """Independent status for each profile acceptance layer."""

    serialization: CheckStatus
    structural: CheckStatus
    semantic: CheckStatus = "not_run"
    canonical_loader: CheckStatus = "not_run"
    consumer: CheckStatus = "not_run"
    messages: tuple[str, ...] = ()
    evidence: Mapping[str, str] = field(default_factory=dict)

    @property
    def success_marker_allowed(self) -> bool:
        """Whether serialization and structural checks permit ``_SUCCESS``."""
        return self.serialization == "passed" and self.structural == "passed"

    @property
    def fully_conformant(self) -> bool:
        """Whether all acceptance layers have passed."""
        return all(
            status == "passed"
            for status in (
                self.serialization,
                self.structural,
                self.semantic,
                self.canonical_loader,
                self.consumer,
            )
        )
