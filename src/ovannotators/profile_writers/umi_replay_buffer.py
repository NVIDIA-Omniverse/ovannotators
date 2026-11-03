# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UMI-compatible Zarr v2 ReplayBuffer profile writer."""

from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ovannotators.profile_writers._artifacts import write_profile_artifacts
from ovannotators.profile_writers._models import (
    ConformanceResult,
    EpisodeEnvelope,
    LossReport,
    NamedTensor,
    ProfileMetadata,
    SampleEnvelope,
    WriterRecord,
)
from ovannotators.profile_writers._registry import ProfileWriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends import Backend

UMI_FORMAT_VERSION = "replay_buffer_zarr_v2"


@dataclass(frozen=True)
class UMIFieldSpec:
    """Map one envelope signal to a stable UMI ReplayBuffer data key."""

    stream_id: str
    output_key: str
    source: Literal["observation", "policy_signal"]
    dtype: str
    shape: tuple[int, ...]
    feature_names: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    representation: str = ""
    coordinate_frame: str | None = None
    rgb: bool = False

    def __post_init__(self) -> None:
        if not self.stream_id or not self.output_key:
            raise ValueError("stream_id and output_key must not be empty")
        if "/" in self.output_key:
            raise ValueError("output_key must be a direct child of the UMI data group")
        np.dtype(self.dtype)
        if self.rgb and (
            self.dtype != "uint8" or len(self.shape) != 3 or self.shape[-1] != 3
        ):
            raise ValueError("RGB fields must be HWC uint8 arrays")
        if self.feature_names and len(self.feature_names) != int(np.prod(self.shape)):
            raise ValueError("feature_names must name every flattened tensor element")
        if self.units and len(self.units) not in (1, len(self.feature_names)):
            raise ValueError("units must contain one value or one per feature")


@dataclass(frozen=True)
class UMIReplayBufferWriterConfig:
    """Configuration for an exact UMI-style ReplayBuffer zip."""

    frequency_hz: float
    fields: tuple[UMIFieldSpec, ...]
    output_path: str = "dataset.zarr.zip"
    timestamp_tolerance_ns: int = 100_000
    rgb_compression_level: int = 99
    low_dim_chunk_steps: int = 1024

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if not self.fields:
            raise ValueError("fields must not be empty")
        if len({item.output_key for item in self.fields}) != len(self.fields):
            raise ValueError("UMI output keys must be unique")
        if len({(item.source, item.stream_id) for item in self.fields}) != len(
            self.fields
        ):
            raise ValueError("UMI source mappings must be unique")
        if not self.output_path.endswith(".zarr.zip"):
            raise ValueError("output_path must end with .zarr.zip")
        if self.timestamp_tolerance_ns < 0:
            raise ValueError("timestamp_tolerance_ns must not be negative")
        if not 0 <= self.rgb_compression_level <= 100:
            raise ValueError("rgb_compression_level must be in [0, 100]")
        if self.low_dim_chunk_steps <= 0:
            raise ValueError("low_dim_chunk_steps must be positive")


@ProfileWriterRegistry.register
class UMIReplayBufferWriter:
    """Write aligned episodes directly into UMI's Zarr ReplayBuffer layout."""

    name = "umi_replay_buffer"

    def __init__(self, config: UMIReplayBufferWriterConfig) -> None:
        self.config = config
        self._active: EpisodeEnvelope | None = None
        self._active_values: dict[str, list[np.ndarray]] = {}
        self._episodes: list[dict[str, np.ndarray]] = []
        self._episode_ids: list[str] = []
        self._dataset_id: str | None = None
        self._embodiment: str | None = None
        self._last_timestamp_ns: int | None = None
        self._backend_initialized = False
        self._finalized = False

    def begin_episode(self, episode: EpisodeEnvelope, backend: Backend) -> None:
        """Validate the episode against the configured fields and start buffering it."""
        if self._finalized:
            raise RuntimeError("cannot begin an episode after finalize")
        if self._active is not None:
            raise RuntimeError("end the active episode before beginning another")
        if not self._backend_initialized:
            if backend.list():
                raise ValueError("UMI output must be empty")
            self._backend_initialized = True
        if episode.episode_id in self._episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        if episode.nominal_rate_hz is None or not np.isclose(
            episode.nominal_rate_hz, self.config.frequency_hz
        ):
            raise ValueError("episode nominal rate must match frequency_hz")
        descriptors = {item.stream_id: item for item in episode.streams}
        missing = sorted(
            {item.stream_id for item in self.config.fields} - descriptors.keys()
        )
        if missing:
            raise ValueError(f"episode is missing required streams: {missing}")
        for field in self.config.fields:
            descriptor = descriptors[field.stream_id]
            if descriptor.shape != field.shape or np.dtype(
                descriptor.dtype
            ) != np.dtype(field.dtype):
                raise ValueError(
                    f"descriptor schema does not match {field.stream_id!r}"
                )
        if self._dataset_id is None:
            self._dataset_id, self._embodiment = episode.dataset_id, episode.embodiment
        elif (episode.dataset_id, episode.embodiment) != (
            self._dataset_id,
            self._embodiment,
        ):
            raise ValueError("dataset_id and embodiment must remain constant")
        self._active = episode
        self._active_values = {item.output_key: [] for item in self.config.fields}
        self._active_values["timestamp_ns"] = []
        self._last_timestamp_ns = None

    def write_sample(self, sample: SampleEnvelope, backend: Backend) -> None:
        """Validate one sample on the fixed-rate timeline and buffer it in memory."""
        del backend
        if self._active is None:
            raise RuntimeError("begin_episode must be called before write_sample")
        if sample.episode_id != self._active.episode_id:
            raise ValueError("sample episode_id does not match the active episode")
        index = len(self._active_values["timestamp_ns"])
        expected = round(index * 1_000_000_000 / self.config.frequency_hz)
        error = abs(sample.timestamp_ns - self._active.source_origin_ns - expected)
        if error > self.config.timestamp_tolerance_ns:
            raise ValueError(f"sample timestamp is off the UMI timeline by {error} ns")
        if (
            self._last_timestamp_ns is not None
            and sample.timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("sample timestamps must be strictly increasing")
        mapped: dict[str, np.ndarray] = {}
        for field in self.config.fields:
            source = (
                sample.observations
                if field.source == "observation"
                else sample.policy_signals
            )
            if field.stream_id not in source:
                raise ValueError(
                    f"sample is missing required signal {field.stream_id!r}"
                )
            mapped[field.output_key] = _validate_tensor(field, source[field.stream_id])
        for key, value in mapped.items():
            self._active_values[key].append(value)
        self._active_values["timestamp_ns"].append(
            np.asarray(sample.timestamp_ns, dtype=np.int64)
        )
        self._last_timestamp_ns = sample.timestamp_ns

    def end_episode(self, backend: Backend) -> None:
        """Stack the active episode into arrays held in memory until finalize."""
        del backend
        if self._active is None:
            raise RuntimeError("no active episode")
        if not self._active_values["timestamp_ns"]:
            raise ValueError("an episode must contain at least one sample")
        self._episodes.append(
            {key: np.stack(values) for key, values in self._active_values.items()}
        )
        self._episode_ids.append(self._active.episode_id)
        self._active = None
        self._active_values = {}
        self._last_timestamp_ns = None

    def finalize(self, backend: Backend) -> None:
        """Write all buffered episodes as one Zarr v2 zip plus profile artifacts."""
        if self._finalized:
            return
        if self._active is not None:
            raise RuntimeError("end the active episode before finalize")
        if not self._episodes:
            raise ValueError("cannot finalize an empty UMI dataset")
        payload = self._serialize()
        conformance = _validate_zip_bytes(payload)
        if conformance.structural != "passed":
            raise ValueError(
                "UMI structural validation failed: " + "; ".join(conformance.messages)
            )
        backend.write(self.config.output_path, payload)
        write_profile_artifacts(
            backend,
            profile=ProfileMetadata(
                profile_id=self.name,
                profile_family="umi",
                profile_version=UMI_FORMAT_VERSION,
                schema_version="zarr_v2",
                required_modalities=tuple(
                    item.output_key for item in self.config.fields
                ),
                temporal_model="fixed_rate_authoritative_ns",
                sharding_behavior="single_zip_store",
                resume_behavior="immutable_after_finalize",
            ),
            writer_record=WriterRecord(
                dataset_id=self._dataset_id or "",
                writer_name=self.name,
                writer_version="1",
                input_identifiers=tuple(self._episode_ids),
                output_identifiers=(self.config.output_path,),
                library_versions={"zarr": "2.x"},
                validator_versions={"umi_replay_buffer": "1"},
                validation_evidence={"structural": "_ovannotators/conformance.json"},
            ),
            loss_report=LossReport(
                consumed=(
                    "episode.identity",
                    "episode.nominal_rate_hz",
                    "sample.timestamp_ns",
                    "sample.observations",
                    "sample.policy_signals",
                ),
                ignored=("episode.task", "episode.provenance"),
                transformed=(
                    "episodes concatenated along the shared time dimension",
                    "episode boundaries encoded as cumulative exclusive indices",
                ),
                notes=(
                    "timestamp_ns is retained as an additional ReplayBuffer data array.",
                ),
            ),
            conformance=conformance,
            payload_paths=(self.config.output_path,),
        )
        self._finalized = True

    def validate(self, backend: Backend) -> ConformanceResult:
        """Reopen the persisted Zarr zip and report structural conformance."""
        if not backend.exists(self.config.output_path):
            return ConformanceResult(
                "failed", "failed", messages=("missing UMI Zarr zip",)
            )
        result = _validate_zip_bytes(backend.read(self.config.output_path))
        if not backend.exists("_ovannotators/_SUCCESS"):
            return ConformanceResult(
                result.serialization,
                "failed",
                semantic=result.semantic,
                canonical_loader=result.canonical_loader,
                consumer=result.consumer,
                messages=(*result.messages, "missing _SUCCESS"),
                evidence=result.evidence,
            )
        return result

    def _serialize(self) -> bytes:
        zarr = _require_zarr()
        with tempfile.TemporaryDirectory(prefix="ovannotators-umi-") as directory:
            path = Path(directory, "dataset.zarr.zip")
            store = zarr.ZipStore(path, mode="w")
            try:
                root = zarr.group(store=store)
                data_group = root.require_group("data")
                all_keys = tuple(self._episodes[0])
                for key in all_keys:
                    values = np.concatenate(
                        [episode[key] for episode in self._episodes], axis=0
                    )
                    field = next(
                        (item for item in self.config.fields if item.output_key == key),
                        None,
                    )
                    chunks = (
                        (1, *values.shape[1:])
                        if field and field.rgb
                        else (
                            min(self.config.low_dim_chunk_steps, len(values)),
                            *values.shape[1:],
                        )
                    )
                    compressor = (
                        _jpegxl(self.config.rgb_compression_level)
                        if field and field.rgb
                        else None
                    )
                    data_group.create_dataset(
                        key, data=values, chunks=chunks, compressor=compressor
                    )
                ends = np.cumsum(
                    [len(item["timestamp_ns"]) for item in self._episodes],
                    dtype=np.int64,
                )
                root.require_group("meta").create_dataset(
                    "episode_ends", data=ends, chunks=(len(ends),)
                )
            finally:
                store.close()
            return path.read_bytes()


def _validate_tensor(field: UMIFieldSpec, tensor: NamedTensor) -> np.ndarray:
    value = np.asarray(tensor.data)
    if value.shape != field.shape or value.dtype != np.dtype(field.dtype):
        raise ValueError(
            f"{field.stream_id!r} expected {field.dtype} {field.shape}, got {value.dtype} {value.shape}"
        )
    if field.feature_names and tensor.feature_names != field.feature_names:
        raise ValueError(f"{field.stream_id!r} feature names do not match")
    if field.units and tensor.units != field.units:
        raise ValueError(f"{field.stream_id!r} units do not match")
    if field.representation and tensor.representation != field.representation:
        raise ValueError(f"{field.stream_id!r} representation does not match")
    if field.coordinate_frame != tensor.coordinate_frame:
        raise ValueError(f"{field.stream_id!r} coordinate frame does not match")
    return np.array(value, copy=True)


def _require_zarr() -> Any:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("UMIReplayBufferWriter requires ovannotators[umi]") from exc
    if not str(zarr.__version__).startswith("2."):
        raise RuntimeError(
            f"UMI ReplayBuffer requires Zarr 2.x, got {zarr.__version__}"
        )
    return zarr


def _jpegxl(level: int) -> Any:
    try:
        from imagecodecs.numcodecs import Jpegxl
    except ImportError as exc:
        raise ImportError("UMI RGB compression requires ovannotators[umi]") from exc
    return Jpegxl(level=level, numthreads=1)


def _validate_zip_bytes(payload: bytes) -> ConformanceResult:
    messages: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            if "meta/episode_ends/.zarray" not in names:
                messages.append("missing meta/episode_ends")
            if not any(
                name.startswith("data/") and name.endswith("/.zarray") for name in names
            ):
                messages.append("missing data arrays")
        zarr = _require_zarr()
        with tempfile.TemporaryDirectory(
            prefix="ovannotators-umi-validate-"
        ) as directory:
            path = Path(directory, "dataset.zarr.zip")
            path.write_bytes(payload)
            store = zarr.ZipStore(path, mode="r")
            try:
                root = zarr.group(store=store)
                ends = np.asarray(root["meta/episode_ends"])
                if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(ends) <= 0):
                    messages.append("episode_ends must be strictly increasing")
                lengths = {array.shape[0] for _, array in root["data"].arrays()}
                if len(lengths) != 1 or (ends.size and int(ends[-1]) not in lengths):
                    messages.append(
                        "all data arrays must share the final episode length"
                    )
            finally:
                store.close()
    except Exception as exc:
        messages.append(f"Zarr validation failed: {type(exc).__name__}: {exc}")
    status = "failed" if messages else "passed"
    return ConformanceResult(
        "passed" if payload else "failed",
        status,
        messages=tuple(messages),
        evidence={"validator": "umi_replay_buffer@1"},
    )
