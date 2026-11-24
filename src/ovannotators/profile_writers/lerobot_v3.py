# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native LeRobot Dataset v3.0 profile writer."""

from __future__ import annotations

import importlib.metadata
import io
import json
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import TYPE_CHECKING, Any

import numpy as np

from ovannotators._internal.lerobot_common import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
    DEFAULT_V3_FEATURES,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    _LOADER_RESULT_MARKER,
    VideoEncodeFn,
    aggregate_stats,
    build_v3_data_table,
    build_v3_episodes_table,
    build_v3_info,
    build_v3_video_info,
    camera_feature_key,
    camera_source_stride,
    stream_schema,
    camera_timestamp_key,
    compute_episode_stats,
    encode_video_frames,
    flatten_dict,
    open_native_parquet,
    require_pyarrow,
    resample_camera_zoh,
    run_offline_loader,
    serialize_stats,
    stats_dtype,
    symlinks_outside_root,
    update_chunk_file_indices,
    validate_feature_keys,
    write_tasks_parquet,
)
from ovannotators._internal.mp4_mux import (
    _MAX_INDEXED_SAMPLES,
    concatenate_mp4_tracks,
    mp4_track_signature,
    validate_mp4_clip,
)
from ovannotators._internal.video_config import (
    VideoCodec,
    VideoPreset,
    validate_video_options,
)
from ovannotators.profile_writers._artifacts import (
    payload_paths_from_listing,
    publish_native_payload,
    require_atomic_create,
    require_disjoint_disk_staging,
    validate_completion_marker,
    validate_payload_checksums,
    write_profile_artifacts,
)
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
from ovannotators.profile_writers._staging import NativeStage

if TYPE_CHECKING:
    from ovannotators.backends import Backend

LEROBOT_VERSION = "0.7.0"
LEROBOT_COMMIT = "410c095eacc0011d998a73383eb94abed2715af0"
LEROBOT_SCHEMA_VERSION = "v3.0"
_MAX_TIMESTAMP_TOLERANCE_NS = 100_000


@dataclass(frozen=True)
class LeRobotV3FeatureSpec:
    """Required semantic declaration for a state or action tensor."""

    stream_id: str
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    representation: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    coordinate_frame: str | None = None

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must not be empty")
        if not self.feature_names:
            raise ValueError(f"{self.stream_id} feature_names must not be empty")
        if len(self.shape) != 1 or self.shape[0] <= 0:
            raise ValueError(
                f"{self.stream_id} shape must be one positive dimension, got {self.shape}"
            )
        if len(self.feature_names) != self.shape[0]:
            raise ValueError("feature_names must name every vector element")
        if len(self.units) not in (1, len(self.feature_names)):
            raise ValueError("units must contain one value or one per feature")
        if not self.representation:
            raise ValueError("representation must not be empty")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "biuf" or dtype.itemsize > 8:
            raise ValueError(f"unsupported LeRobot v3 vector dtype: {self.dtype!r}")
        object.__setattr__(self, "dtype", dtype.name if dtype.isnative else dtype.str)


def _parquet_size(table: Any) -> int:
    """Return ``table``'s size as a snappy Parquet file, the way shards are written.

    LeRobot rolls data files over on their on-disk size, so the in-memory Arrow
    size would start new shards long before the files reach the limit.
    """
    _, pq = require_pyarrow("LeRobotV3Writer")
    sink = io.BytesIO()
    pq.write_table(
        table,
        sink,
        compression="snappy",
        use_dictionary=True,
        row_group_size=table.num_rows,
    )
    return sink.tell()


@dataclass(frozen=True)
class LeRobotV3WriterConfig:
    """Configuration for the native LeRobot v3.0 dataset writer.

    Attributes:
        fps: Constant dataset frame rate. Every camera rate must divide it.
        camera_stream_ids: Unique RGB camera stream IDs written as MP4 videos.
        state: Semantic declaration of the state vector.
        action: Semantic declaration of the action vector.
        repo_id: Optional LeRobot ``namespace/dataset`` identifier. It is
            validated when given but not written to the dataset; pass it to
            the canonical loader when opening the tree.
        chunks_size: Maximum files per chunk directory.
        video_files_size_in_mb: Target size limit per sharded MP4 file in MB.
        data_files_size_in_mb: Target size limit per sharded Parquet file in MB.
        video_codec: NVENC video codec: ``"h264"`` (default), ``"hevc"``, or ``"av1"``.
        video_lossless: Enable lossless HEVC 4:4:4 encoding when True.
        video_preset: Optional NVENC preset ``"P1"`` through ``"P7"``.
        video_gop: Keyframe interval in frames. Defaults to 2, LeRobot's own
            default, for fast random access; ``None`` uses the NVENC default.
        timestamp_tolerance_ns: Accepted CFR timestamp error, 0 to 100000 ns.
        staging_dir: Optional local directory for building the dataset tree.
            Its parent must be trusted: writable only by you, or sticky like
            the system temporary directory that the default uses.
        keep_staging: Retain ``staging_dir`` after successful finalization.
            A failed attempt retains staging so publication can be retried.
    """

    fps: int
    camera_stream_ids: tuple[str, ...]
    state: LeRobotV3FeatureSpec
    action: LeRobotV3FeatureSpec
    repo_id: str | None = None
    chunks_size: int = DEFAULT_CHUNK_SIZE
    video_files_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    data_files_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB
    video_codec: VideoCodec = "h264"
    video_lossless: bool = False
    video_preset: VideoPreset | None = None
    video_gop: int | None = 2
    timestamp_tolerance_ns: int = _MAX_TIMESTAMP_TOLERANCE_NS
    staging_dir: str | Path | None = None
    keep_staging: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not self.camera_stream_ids or len(set(self.camera_stream_ids)) != len(
            self.camera_stream_ids
        ):
            raise ValueError("camera_stream_ids must be non-empty and unique")
        if self.repo_id is not None and not re.fullmatch(
            r"[^/\s]+/[^/\s]+", self.repo_id
        ):
            raise ValueError("repo_id must have the form 'namespace/dataset'")
        if self.chunks_size <= 0:
            raise ValueError("chunks_size must be positive")
        for name in ("video_files_size_in_mb", "data_files_size_in_mb"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        validate_video_options(
            self.video_codec,
            self.video_lossless,
            self.video_preset,
            gop=self.video_gop,
        )
        if not 0 <= self.timestamp_tolerance_ns <= _MAX_TIMESTAMP_TOLERANCE_NS:
            raise ValueError("timestamp_tolerance_ns must be in [0, 100000]")
        if self.state.stream_id == self.action.stream_id:
            raise ValueError("state and action stream IDs must differ")
        validate_feature_keys(
            self.camera_stream_ids, self.state.stream_id, self.action.stream_id
        )


@dataclass
class _VideoShardState:
    chunk_index: int = 0
    file_index: int = 0
    clips: list[bytes] = None  # type: ignore[assignment]
    total_bytes: int = 0
    # Integer frame count, so shard timestamps never accumulate float error.
    frame_count: int = 0
    track_signature: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.clips is None:
            self.clips = []


@ProfileWriterRegistry.register
class LeRobotV3Writer:
    """Write LeRobot Dataset v3.0 natively using PyArrow and NVENC video encoding.

    Produces the canonical LeRobot v3.0 directory layout (``meta/info.json``,
    ``meta/stats.json``, ``meta/tasks.parquet``, ``meta/episodes/``,
    ``data/chunk-*/file-*.parquet``, and ``videos/<key>/chunk-*/file-*.mp4``)
    without importing ``lerobot``, ``av``, ``torch``, or ``datasets``.
    """

    name = "lerobot_v3"

    def __init__(
        self,
        config: LeRobotV3WriterConfig,
        *,
        video_encoder: VideoEncodeFn | None = None,
    ) -> None:
        self.config = config
        self._video_encoder: VideoEncodeFn = video_encoder or encode_video_frames
        self._stage = (
            Path(config.staging_dir).resolve()
            if config.staging_dir
            else Path(
                tempfile.gettempdir(), f"ovannotators-lerobot-v3-{uuid.uuid4().hex}"
            )
        )
        self._staging = NativeStage(self._stage)
        self._active: EpisodeEnvelope | None = None
        self._active_rows: list[dict[str, Any]] = []
        self._active_cameras: dict[str, list[np.ndarray]] = {}
        self._active_camera_timestamps: dict[str, list[int]] = {}
        self._last_timestamp_ns: int | None = None
        self._dataset_id: str | None = None
        self._embodiment: str | None = None
        self._stream_schema: tuple[tuple[Any, ...], ...] | None = None
        self._episode_ids: list[str] = []
        self._camera_shapes: dict[str, tuple[int, ...]] = {}
        self._camera_rates: dict[str, float] = {}
        self._tasks: dict[str, int] = {}
        self._episode_records: list[dict[str, Any]] = []
        self._episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
        self._global_index: int = 0

        # Active Parquet data shard state (flushed when size limit is reached)
        self._data_chunk_idx: int = 0
        self._data_file_idx: int = 0
        self._pending_data_tables: list[Any] = []
        self._pending_data_bytes: int = 0

        # Per-camera active MP4 shard state
        self._video_shards: dict[str, _VideoShardState] = {}

        self._payload_paths: list[str] = []
        self._backend_initialized = False
        self._finalization_started = False
        self._finalized = False

    @property
    def staging_root(self) -> Path:
        """Return the local staging directory used to assemble the dataset tree."""
        return self._stage

    def begin_episode(self, episode: EpisodeEnvelope, backend: Backend) -> None:
        """Validate episode descriptors and begin buffering a new episode."""
        if self._finalization_started:
            raise RuntimeError("cannot begin an episode after finalization starts")
        if self._active is not None:
            raise RuntimeError("end the active episode before beginning another")
        require_disjoint_disk_staging(self._stage, backend)
        if not self._backend_initialized:
            require_atomic_create(backend)
            if backend.list():
                raise ValueError("LeRobot v3 output must be empty")
            require_pyarrow("LeRobotV3Writer")
            self._staging.reserve()
            self._backend_initialized = True
        shapes, rates = self._validate_episode(episode)
        if self._dataset_id is None:
            self._dataset_id = episode.dataset_id
            self._embodiment = episode.embodiment
            self._stream_schema = stream_schema(episode)
            self._camera_shapes, self._camera_rates = shapes, rates
            self._video_shards = {
                camera_feature_key(stream_id): _VideoShardState()
                for stream_id in self.config.camera_stream_ids
            }
        else:
            if (
                episode.dataset_id != self._dataset_id
                or episode.embodiment != self._embodiment
            ):
                raise ValueError("dataset_id and embodiment must remain constant")
            if stream_schema(episode) != self._stream_schema:
                raise ValueError("stream schema changed between episodes")
        if episode.episode_id in self._episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        self._active = episode
        self._active_rows = []
        self._active_cameras = {key: [] for key in self.config.camera_stream_ids}
        self._active_camera_timestamps = {
            key: [] for key in self.config.camera_stream_ids
        }
        self._last_timestamp_ns = None

    def write_sample(self, sample: SampleEnvelope, backend: Backend) -> None:
        """Validate and buffer one sample for the active episode."""
        del backend
        if self._active is None:
            raise RuntimeError("begin_episode must be called before write_sample")
        frame_index = len(self._active_rows)
        if (
            self._last_timestamp_ns is not None
            and sample.timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError(
                "sample timestamps must be strictly increasing within an episode"
            )
        expected = round(frame_index * 1_000_000_000 / self.config.fps)
        error = abs(sample.timestamp_ns - self._active.source_origin_ns - expected)
        if error > self.config.timestamp_tolerance_ns:
            raise ValueError(
                f"sample timestamp cannot be represented on the LeRobot CFR timeline: error {error} ns"
            )
        row, cameras = self._map_sample(sample, frame_index)
        self._active_rows.append(row)
        for key, image in cameras.items():
            self._active_cameras[key].append(image)
            self._active_camera_timestamps[key].append(sample.timestamp_ns)
        self._last_timestamp_ns = sample.timestamp_ns

    def end_episode(self, backend: Backend) -> None:
        """Align cameras, encode episode videos, compute stats, and stage shards."""
        del backend
        if self._active is None:
            raise RuntimeError("no active episode")
        self._staging.verify()
        if not self._active_rows:
            raise ValueError("an episode must contain at least one accepted sample")

        episode_index = len(self._episode_ids)
        episode_length = len(self._active_rows)
        dataset_from_index = self._global_index
        dataset_to_index = dataset_from_index + episode_length
        task_index = self._tasks.get(self._active.task, len(self._tasks))

        targets = [int(row["source.timestamp_ns"][0]) for row in self._active_rows]
        aligned_cameras: dict[str, list[np.ndarray]] = {}
        aligned_timestamps: dict[str, list[int]] = {}
        for stream_id in self.config.camera_stream_ids:
            frames, source_times = resample_camera_zoh(
                self._active_cameras[stream_id],
                self._active_camera_timestamps[stream_id],
                targets,
                source_rate_hz=self._camera_rates[stream_id],
                tolerance_ns=self.config.timestamp_tolerance_ns,
                stream_id=stream_id,
            )
            aligned_cameras[camera_feature_key(stream_id)] = frames
            aligned_timestamps[camera_timestamp_key(stream_id)] = source_times

        # Populate canonical LeRobot v3 default columns
        complete_rows: list[dict[str, Any]] = []
        for frame_idx, row in enumerate(self._active_rows):
            complete = {
                **row,
                **{
                    key: np.asarray([timestamps[frame_idx]], dtype=np.int64)
                    for key, timestamps in aligned_timestamps.items()
                },
                "timestamp": np.float32(frame_idx / self.config.fps),
                "frame_index": frame_idx,
                "episode_index": episode_index,
                "index": dataset_from_index + frame_idx,
                "task_index": task_index,
            }
            complete_rows.append(complete)

        # Compute per-episode statistics (tabular + video)
        features = self._full_features()
        vector_data: dict[str, np.ndarray] = {}
        for key, ft in features.items():
            if ft["dtype"] == "video":
                continue
            if key == "timestamp":
                vector_data[key] = np.arange(episode_length) / self.config.fps
            elif tuple(ft["shape"]) == (1,):
                vector_data[key] = np.asarray(
                    [
                        np.asarray(r[key]).reshape(-1)[0].item()
                        if isinstance(r[key], np.ndarray)
                        else r[key]
                        for r in complete_rows
                    ],
                    dtype=stats_dtype(key, ft["dtype"]),
                )
            else:
                vector_data[key] = np.stack(
                    [
                        np.asarray(r[key], dtype=np.dtype(ft["dtype"]))
                        for r in complete_rows
                    ]
                )
        ep_stats = compute_episode_stats(vector_data, aligned_cameras)

        ep_table = build_v3_data_table(complete_rows, features)
        clips: dict[str, tuple[bytes, tuple[Any, ...]]] = {}
        for vkey, frames in aligned_cameras.items():
            clip_bytes = self._video_encoder(
                frames,
                self.config.fps,
                codec=self.config.video_codec,
                lossless=self.config.video_lossless,
                preset=self.config.video_preset,
                gop=self.config.video_gop,
            )
            height, width = frames[0].shape[:2]
            validate_mp4_clip(
                clip_bytes,
                frame_count=episode_length,
                fps=self.config.fps,
                width=width,
                height=height,
                codec=self.config.video_codec,
            )
            clips[vkey] = clip_bytes, mp4_track_signature(clip_bytes)

        data_limit = self.config.data_files_size_in_mb * 1024 * 1024
        ep_data_bytes = _parquet_size(ep_table)
        if (
            self._pending_data_tables
            and self._pending_data_bytes + ep_data_bytes > data_limit
        ):
            self._flush_data_shard()
            self._data_chunk_idx, self._data_file_idx = update_chunk_file_indices(
                self._data_chunk_idx, self._data_file_idx, self.config.chunks_size
            )

        video_limit = self.config.video_files_size_in_mb * 1024 * 1024
        video_meta: dict[str, Any] = {}
        for vkey, (clip_bytes, signature) in clips.items():
            shard = self._video_shards[vkey]
            # A shard must also stay within the sample count MP4 indexing accepts.
            if shard.clips and (
                shard.total_bytes + len(clip_bytes) > video_limit
                or shard.frame_count + episode_length > _MAX_INDEXED_SAMPLES
                or shard.track_signature != signature
            ):
                self._flush_video_shard(vkey, shard)
                shard.chunk_index, shard.file_index = update_chunk_file_indices(
                    shard.chunk_index, shard.file_index, self.config.chunks_size
                )

            video_meta[f"videos/{vkey}/chunk_index"] = shard.chunk_index
            video_meta[f"videos/{vkey}/file_index"] = shard.file_index
            video_meta[f"videos/{vkey}/from_timestamp"] = (
                shard.frame_count / self.config.fps
            )
            video_meta[f"videos/{vkey}/to_timestamp"] = (
                shard.frame_count + episode_length
            ) / self.config.fps

        ep_record: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": [self._active.task],
            "length": episode_length,
            "data/chunk_index": self._data_chunk_idx,
            "data/file_index": self._data_file_idx,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_to_index,
            **video_meta,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            **flatten_dict({"stats": ep_stats}),
        }
        self._pending_data_tables.append(ep_table)
        self._pending_data_bytes += ep_data_bytes
        for vkey, (clip_bytes, signature) in clips.items():
            shard = self._video_shards[vkey]
            shard.clips.append(clip_bytes)
            shard.total_bytes += len(clip_bytes)
            shard.frame_count += episode_length
            shard.track_signature = signature
        self._tasks[self._active.task] = task_index
        self._episode_records.append(ep_record)
        self._episode_stats.append(ep_stats)
        self._episode_ids.append(self._active.episode_id)
        self._global_index = dataset_to_index

        self._active = None
        self._active_rows = []
        self._active_cameras = {}
        self._active_camera_timestamps = {}
        self._last_timestamp_ns = None

    def finalize(self, backend: Backend) -> None:
        """Flush remaining shards, write metadata, validate, and publish."""
        if self._finalized:
            return
        if self._active is not None:
            raise RuntimeError("end the active episode before finalize")
        if not self._episode_ids:
            raise ValueError("cannot finalize an empty LeRobot v3 dataset")
        require_disjoint_disk_staging(self._stage, backend)
        self._staging.verify()
        self._finalization_started = True
        self._finalize_and_publish(backend)
        if not self.config.keep_staging:
            self._staging.cleanup()

    def _finalize_and_publish(self, backend: Backend) -> None:
        pa, pq = require_pyarrow("LeRobotV3Writer")
        self._flush_data_shard()
        for vkey, shard in self._video_shards.items():
            self._flush_video_shard(vkey, shard)

        meta_dir = self._stage / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # meta/tasks.parquet
        ordered_tasks = [
            task for task, _ in sorted(self._tasks.items(), key=lambda item: item[1])
        ]
        tasks_path = self._staging.path(DEFAULT_TASKS_PATH)
        write_tasks_parquet(ordered_tasks, tasks_path)
        self._staging.record_file(tasks_path)

        # meta/episodes/chunk-000/file-000.parquet
        episodes_path = self._staging.path(
            DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
        )
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_table = build_v3_episodes_table(self._episode_records)
        pq.write_table(
            episodes_table,
            str(episodes_path),
            compression="snappy",
            use_dictionary=True,
        )
        self._staging.record_file(episodes_path)

        # meta/stats.json
        global_stats = aggregate_stats(self._episode_stats)
        stats_path = self._staging.path("meta/stats.json")
        stats_path.write_text(
            json.dumps(serialize_stats(global_stats), indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        self._staging.record_file(stats_path)

        # meta/info.json
        info = build_v3_info(
            codebase_version=LEROBOT_SCHEMA_VERSION,
            fps=self.config.fps,
            features=self._full_features(),
            total_episodes=len(self._episode_ids),
            total_frames=self._global_index,
            total_tasks=len(self._tasks),
            robot_type=self._embodiment,
            chunks_size=self.config.chunks_size,
            data_files_size_in_mb=self.config.data_files_size_in_mb,
            video_files_size_in_mb=self.config.video_files_size_in_mb,
            data_path=DEFAULT_DATA_PATH,
            video_path=DEFAULT_VIDEO_PATH,
        )
        info_path = self._staging.path("meta/info.json")
        info_path.write_text(
            json.dumps(info, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        self._staging.record_file(info_path)

        self._payload_paths = publish_native_payload(
            backend, self._stage, self._staging.files()
        )
        if set(self._payload_paths) != payload_paths_from_listing(backend.list()):
            raise ValueError(
                "published paths do not match the staged LeRobot v3 payload"
            )
        structural = _validate_structure(backend, require_success=False)
        if structural.structural != "passed":
            raise ValueError(
                "structural validation failed: " + "; ".join(structural.messages)
            )
        conformance = ConformanceResult(
            "passed",
            "passed",
            canonical_loader="not_run",
            evidence={
                "serializer": "ovannotators-native-lerobot-v3",
                "target_schema": f"lerobot=={LEROBOT_VERSION} ({LEROBOT_SCHEMA_VERSION})",
            },
        )
        profile = ProfileMetadata(
            profile_id=self.name,
            profile_family="huggingface_lerobot",
            profile_version=LEROBOT_VERSION,
            schema_version=LEROBOT_SCHEMA_VERSION,
            required_modalities=(
                *(f"video:{key}" for key in self.config.camera_stream_ids),
                self.config.state.stream_id,
                self.config.action.stream_id,
            ),
            temporal_model="official_cfr_timestamp_plus_authoritative_source_ns",
            sharding_behavior="native_lerobot_v3_size_bounded_parquet_and_mp4",
            resume_behavior="immutable_after_finalize",
        )
        writer_record = WriterRecord(
            dataset_id=self._dataset_id or "",
            writer_name=self.name,
            writer_version="3",
            input_identifiers=tuple(self._episode_ids),
            output_identifiers=tuple(self._payload_paths),
            codec_versions={"video": self.config.video_codec},
            library_versions={
                "pyarrow": pa.__version__,
                "target_lerobot_schema": f"{LEROBOT_VERSION}/{LEROBOT_SCHEMA_VERSION}",
            },
            validator_versions={"lerobot_v3_structural": "3"},
            # finalize() never runs the canonical loader; see
            # validate_lerobot_v3_canonical for the out-of-band check.
            validation_evidence={
                "structural": "_ovannotators/conformance.json",
                "checksums": "_ovannotators/payload_checksums.json",
            },
        )
        loss_report = LossReport(
            consumed=(
                "episode.identity",
                "episode.task",
                "episode.embodiment",
                "episode.streams",
                "sample.frame_id",
                "sample.timestamp_ns",
                "sample.observations",
                "sample.policy_signals",
            ),
            ignored=(
                "episode.provenance",
                "sample.sample_id",
                "state.units",
                "state.representation",
                "state.coordinate_frame",
                "action.units",
                "action.representation",
                "action.coordinate_frame",
            ),
            transformed=(
                "timestamp derived from frame_index and fps",
                "camera IDs map to observation.images.*",
                "mixed-rate cameras use zero-order-hold alignment",
            ),
            notes=(
                "source.timestamp_ns retains the authoritative timestamp.",
                "Each aligned camera retains its source timestamp.",
                "Serialized natively by ovannotators without importing lerobot or PyAV.",
            ),
        )
        write_profile_artifacts(
            backend,
            profile=profile,
            writer_record=writer_record,
            loss_report=loss_report,
            conformance=conformance,
            payload_paths=tuple(self._payload_paths),
            retryable=True,
        )
        self._finalized = True

    def validate(self, backend: Backend) -> ConformanceResult:
        """Validate the published dataset structure and return its result."""
        return _validate_structure(backend, require_success=True)

    def _flush_data_shard(self) -> None:
        # Written with the settings _parquet_size measures, so shard sizes
        # track the configured limit the way LeRobot's on-disk sizes do.
        if not self._pending_data_tables:
            return
        _, pq = require_pyarrow("LeRobotV3Writer")
        out_path = self._staging.path(
            DEFAULT_DATA_PATH.format(
                chunk_index=self._data_chunk_idx, file_index=self._data_file_idx
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        schema = self._pending_data_tables[0].schema
        with pq.ParquetWriter(
            str(out_path),
            schema,
            compression="snappy",
            use_dictionary=True,
        ) as pq_writer:
            for ep_table in self._pending_data_tables:
                # One row group per episode, as LeRobot writes it.
                pq_writer.write_table(ep_table, row_group_size=ep_table.num_rows)
        self._staging.record_file(out_path)
        self._pending_data_tables.clear()
        self._pending_data_bytes = 0

    def _flush_video_shard(self, video_key: str, shard: _VideoShardState) -> None:
        if not shard.clips:
            return
        out_path = self._staging.path(
            DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=shard.chunk_index,
                file_index=shard.file_index,
            )
        )
        if not out_path.resolve().is_relative_to(self._stage.resolve()):
            raise ValueError(
                f"video shard path escapes the staging directory: {video_key!r}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(shard.clips) == 1:
            out_path.write_bytes(shard.clips[0])
        else:
            out_path.write_bytes(concatenate_mp4_tracks(shard.clips))
        self._staging.record_file(out_path)
        shard.clips.clear()
        shard.total_bytes = 0
        shard.frame_count = 0

    def _full_features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            self.config.state.stream_id: _feature(self.config.state),
            self.config.action.stream_id: _feature(self.config.action),
            "source.timestamp_ns": {"dtype": "int64", "shape": [1], "names": None},
            "source.frame_id": {"dtype": "int64", "shape": [1], "names": None},
        }
        for stream_id in self.config.camera_stream_ids:
            shape = self._camera_shapes[stream_id]
            features[camera_feature_key(stream_id)] = {
                "dtype": "video",
                "shape": list(shape),
                "names": ["height", "width", "channels"],
                "info": build_v3_video_info(
                    height=shape[0],
                    width=shape[1],
                    fps=self.config.fps,
                    codec=self.config.video_codec,
                    lossless=self.config.video_lossless,
                    gop=self.config.video_gop,
                    preset=self.config.video_preset,
                ),
            }
            features[camera_timestamp_key(stream_id)] = {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            }
        for key, spec in DEFAULT_V3_FEATURES.items():
            features[key] = dict(spec)
        return features

    def _validate_episode(
        self, episode: EpisodeEnvelope
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
        descriptors = {stream.stream_id: stream for stream in episode.streams}
        required = {
            self.config.state.stream_id,
            self.config.action.stream_id,
            *self.config.camera_stream_ids,
        }
        missing = sorted(required - descriptors.keys())
        if missing:
            raise ValueError(
                f"episode is missing required stream descriptors: {missing}"
            )
        shapes: dict[str, tuple[int, ...]] = {}
        rates: dict[str, float] = {}
        for key in self.config.camera_stream_ids:
            stream = descriptors[key]
            if (
                stream.stream_type != "rgb"
                or np.dtype(stream.dtype) != np.uint8
                or len(stream.shape) != 3
                or stream.shape[-1] != 3
            ):
                raise ValueError(f"camera stream {key!r} must be HWC uint8 RGB")
            if not stream.required or stream.nominal_rate_hz is None:
                raise ValueError(
                    f"camera stream {key!r} requires a required explicit nominal rate"
                )
            if camera_source_stride(self.config.fps, stream.nominal_rate_hz) is None:
                raise ValueError(
                    f"camera stream {key!r} rate is not an integer divisor of dataset fps"
                )
            shapes[key], rates[key] = tuple(stream.shape), stream.nominal_rate_hz
        for spec, kind in (
            (self.config.state, "state"),
            (self.config.action, "action"),
        ):
            stream = descriptors[spec.stream_id]
            if stream.stream_type != kind:
                raise ValueError(f"{spec.stream_id!r} must have stream_type {kind!r}")
            if tuple(stream.shape) != tuple(spec.shape) or np.dtype(
                stream.dtype
            ) != np.dtype(spec.dtype):
                raise ValueError(
                    f"descriptor schema does not match config for {spec.stream_id}"
                )
        if self._camera_shapes and (
            shapes != self._camera_shapes or rates != self._camera_rates
        ):
            raise ValueError("camera stream schema changed between episodes")
        return shapes, rates

    def _map_sample(
        self, sample: SampleEnvelope, frame_index: int
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        assert self._active is not None
        if sample.episode_id != self._active.episode_id:
            raise ValueError("sample episode_id does not match the active episode")
        missing: list[str] = []
        for key in self.config.camera_stream_ids:
            stride = camera_source_stride(self.config.fps, self._camera_rates[key])
            assert stride is not None
            expected, present = frame_index % stride == 0, key in sample.observations
            if expected and not present:
                missing.append(key)
            elif not expected and present:
                raise ValueError(
                    f"camera {key!r} is present outside its declared source cadence"
                )
            elif (
                not expected
                and sample.missing_data.get(key) != "not_sampled_at_source_rate"
            ):
                missing.append(key)
        if self.config.state.stream_id not in sample.observations:
            missing.append(self.config.state.stream_id)
        if self.config.action.stream_id not in sample.policy_signals:
            missing.append(self.config.action.stream_id)
        if missing:
            details = {key: sample.missing_data.get(key, "absent") for key in missing}
            raise ValueError(f"sample is missing required signals: {details}")
        state = _validate_tensor(
            self.config.state, sample.observations[self.config.state.stream_id]
        )
        action = _validate_tensor(
            self.config.action, sample.policy_signals[self.config.action.stream_id]
        )
        cameras = {}
        for key in self.config.camera_stream_ids:
            if key in sample.observations:
                image = np.asarray(sample.observations[key].data)
                if image.shape != self._camera_shapes[key] or image.dtype != np.uint8:
                    raise ValueError(
                        f"camera {key!r} expected uint8 {self._camera_shapes[key]}, got {image.dtype} {image.shape}"
                    )
                cameras[key] = np.array(image, copy=True)
        return {
            self.config.state.stream_id: state,
            self.config.action.stream_id: action,
            "source.timestamp_ns": np.asarray([sample.timestamp_ns], dtype=np.int64),
            "source.frame_id": np.asarray([sample.frame_id], dtype=np.int64),
        }, cameras


def validate_lerobot_v3_canonical(
    root: str | Path, repo_id: str, *, expected_version: str = LEROBOT_VERSION
) -> ConformanceResult:
    """Open a completed tree with the pinned upstream ``LeRobotDataset`` loader.

    The loader runs only after local checks pass. The tree must hold
    ``_SUCCESS`` and a SHA-256 manifest that exactly covers and matches its
    payload. No symlink under ``root`` may resolve outside it. Its
    ``data_path`` and ``video_path`` templates must stay inside ``root``, and
    every data and video shard that ``meta/episodes`` references must exist. It must declare no ``image`` feature, because the official
    reader opens image paths that may name remote URLs; ``LeRobotV3Writer``
    stores cameras as ``video``. A tree that fails these checks reports
    ``structural="failed"`` and ``canonical_loader="not_run"``.

    The loader then runs in a child interpreter with ``HF_HUB_OFFLINE=1`` and
    ``HF_DATASETS_OFFLINE=1``, so a tree it still finds incomplete fails
    instead of being completed from the Hugging Face Hub.

    Requires ``lerobot[dataset]`` at ``expected_version`` installed in an
    isolated Python 3.12+ conformance environment (not shipped as a runtime
    extra of ``ovannotators``).

    Args:
        root (str | Path): Completed ``LeRobotV3Writer`` output directory.
        repo_id (str): The ``namespace/dataset`` name the tree was written with.
        expected_version (str): ``lerobot`` version that must be installed.
            Defaults to the pinned target ``0.7.0``; a format-drift probe passes
            the candidate release instead.

    Returns:
        ConformanceResult: Structural and canonical-loader status. Its evidence
        records the ``lerobot`` version actually exercised.
    """
    root = Path(root)
    messages = _canonical_preflight(root)
    if messages:
        return ConformanceResult(
            "passed",
            "failed",
            canonical_loader="not_run",
            messages=tuple(messages),
        )
    outcome = _run_offline_loader(root, repo_id, expected_version)
    if "error" in outcome:
        return ConformanceResult(
            "passed",
            "passed",
            canonical_loader="failed",
            messages=(f"LeRobotDataset failed: {outcome['error']}",),
        )
    return ConformanceResult(
        "passed",
        "passed",
        canonical_loader="passed",
        evidence={"loader": f"lerobot=={outcome['version']}"},
    )


def _load_official(root: str, repo_id: str, expected_version: str) -> dict[str, str]:
    """Open ``root`` with the official loader; ``{"version"}`` or ``{"error"}``."""
    try:
        dataset_type, installed = _official_dataset_type(expected_version)
        dataset = dataset_type(repo_id=repo_id, root=Path(root))
        if len(dataset) <= 0:
            raise RuntimeError("canonical loader returned an empty dataset")
        dataset[0]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"version": installed}


_OFFLINE_LOADER_SCRIPT = (
    "import json, sys\n"
    "from ovannotators.profile_writers.lerobot_v3 import _load_official\n"
    f"print({_LOADER_RESULT_MARKER!r} + json.dumps(_load_official(*sys.argv[1:4])))\n"
)


def _run_offline_loader(
    root: Path, repo_id: str, expected_version: str
) -> dict[str, str]:
    """Run ``_load_official`` in a child interpreter with Hub access disabled.

    ``huggingface_hub`` and ``datasets`` read their offline flags when they are
    imported, so the flags are only guaranteed in a fresh process. ``-P``
    keeps the working directory off ``sys.path`` in the child.
    """
    return run_offline_loader(
        _OFFLINE_LOADER_SCRIPT, str(root), repo_id, expected_version
    )


def _canonical_preflight(root: Path) -> list[str]:
    """Return reasons the official loader must not open ``root``."""
    if not root.is_dir():
        return [f"not a directory: {root}"]
    return symlinks_outside_root(root) or _verified_payload_problems(root)


def _verified_payload_problems(root: Path) -> list[str]:
    """Return problems with the checksummed payload ``DiskBackend`` lists."""
    from ovannotators.backends import DiskBackend, DiskBackendConfig

    backend = DiskBackend(DiskBackendConfig(root))
    paths = set(backend.list())
    structure = _validate_structure(backend, require_success=True, paths=paths)
    if structure.structural != "passed":
        return list(structure.messages)
    info = json.loads(backend.read("meta/info.json"))
    if info.get("storage_format") not in (None, "lerobot"):
        return ["meta/info.json storage_format must be 'lerobot' or omitted"]
    messages = []
    for key in ("data_path", "video_path"):
        template = info.get(key)
        if template is None:
            continue
        if (
            not isinstance(template, str)
            or not template
            or any(character in template for character in ("://", "\\", "\x00"))
            or re.match(r"^[A-Za-z]:", template)
            or Path(template).is_absolute()
            or ".." in Path(template).parts
        ):
            messages.append(f"meta/info.json {key} must be relative to the dataset")
            continue
        fields = {"chunk_index", "file_index"}
        if key == "video_path":
            fields.add("video_key")
        try:
            for _, field, spec, conversion in Formatter().parse(template):
                if field is None:
                    continue
                if (
                    field not in fields
                    or conversion is not None
                    or (
                        spec
                        and (
                            field == "video_key"
                            or not re.fullmatch(r"0?[1-9]?[0-9]?d", spec)
                        )
                    )
                ):
                    raise ValueError("unsupported format field")
        except ValueError as exc:
            messages.append(f"meta/info.json {key} has an invalid template: {exc}")
    features = info.get("features")
    if not isinstance(features, dict):
        return [*messages, "meta/info.json has no features mapping"]
    images = sorted(
        name
        for name, feature in features.items()
        if not isinstance(feature, dict) or feature.get("dtype") == "image"
    )
    if images:
        messages.append(
            f"image features are not produced by LeRobotV3Writer and are not "
            f"opened: {images}"
        )
    if messages:
        return messages
    return _missing_referenced_shards(root, info, features, paths)


def _missing_referenced_shards(
    root: Path, info: dict[str, Any], features: dict[str, Any], paths: set[str]
) -> list[str]:
    """Return problems with the shards ``meta/episodes`` references.

    The official loader downloads any referenced shard it cannot find locally,
    so each one must exist inside ``root`` before the loader opens the tree.
    """
    video_keys = sorted(
        name for name, feature in features.items() if feature.get("dtype") == "video"
    )
    data_path, video_path = info.get("data_path"), info.get("video_path")
    if data_path is None or (video_keys and video_path is None):
        return [
            "meta/info.json must declare data_path and, for video features, video_path"
        ]
    columns = ["episode_index", "data/chunk_index", "data/file_index"]
    for key in video_keys:
        columns += [f"videos/{key}/chunk_index", f"videos/{key}/file_index"]
    referenced_data: set[str] = set()
    referenced_video: set[str] = set()
    messages: list[str] = []
    resolved_root = root.resolve()
    episodes = sorted((root / "meta/episodes").glob("*/*.parquet"))
    data = sorted((root / "data").glob("*/*.parquet"))
    discovered_data = {path.relative_to(root).as_posix() for path in data}
    if not episodes or not data:
        return ["missing reader-discoverable episode or data Parquet shards"]
    episode_shards = set(episodes)
    for shard in [root / DEFAULT_TASKS_PATH, *data, *episodes]:
        relative = shard.relative_to(root).as_posix()
        if relative not in paths or not shard.resolve().is_relative_to(resolved_root):
            messages.append(
                f"Parquet shard is outside the verified payload: {relative}"
            )
            continue
        try:
            with open_native_parquet(shard) as parquet:
                if shard not in episode_shards:
                    continue
                missing = set(columns) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(
                        f"missing shard reference columns: {sorted(missing)}"
                    )
                rows = parquet.read(columns=columns).to_pylist()
        except Exception as exc:
            messages.append(
                f"{relative} lacks shard references: {type(exc).__name__}: {exc}"
            )
            continue
        for row in rows:
            try:
                referenced_data.add(
                    data_path.format(
                        chunk_index=row["data/chunk_index"],
                        file_index=row["data/file_index"],
                    )
                )
                for key in video_keys:
                    referenced_video.add(
                        video_path.format(
                            video_key=key,
                            chunk_index=row[f"videos/{key}/chunk_index"],
                            file_index=row[f"videos/{key}/file_index"],
                        )
                    )
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                messages.append(
                    f"{relative} episode {row.get('episode_index')} has an "
                    f"unresolvable shard path: {type(exc).__name__}: {exc}"
                )
    for path in sorted(referenced_data | referenced_video):
        try:
            target = (root / path).resolve()
        except (OSError, ValueError, RuntimeError) as exc:
            messages.append(f"invalid referenced shard {path!r}: {exc}")
            continue
        if not target.is_relative_to(resolved_root):
            messages.append(f"referenced shard escapes the dataset: {path}")
        elif not target.is_file():
            messages.append(f"referenced shard is missing: {path}")
        elif (
            path not in paths
            or (path in referenced_data and path not in discovered_data)
            or (path in referenced_video and Path(path).suffix != ".mp4")
        ):
            messages.append(
                f"referenced shard is outside the verified reader payload: {path}"
            )
    return messages


def _official_dataset_type(expected_version: str = LEROBOT_VERSION) -> tuple[type, str]:
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "LeRobot 0.7.0 canonical validation requires Python 3.12 or newer"
        )
    try:
        version = importlib.metadata.version("lerobot")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ImportError(
            f"validate_lerobot_v3_canonical requires lerobot=={expected_version} "
            "installed in an isolated Python 3.12 conformance environment "
            "(see tests/requirements/lerobot-v3-conformance.txt)"
        ) from exc
    if version != expected_version:
        raise RuntimeError(
            f"validate_lerobot_v3_canonical requires lerobot=={expected_version}, got {version}"
        )
    return LeRobotDataset, version


def _validate_structure(
    backend: Backend, *, require_success: bool, paths: set[str] | None = None
) -> ConformanceResult:
    messages: list[str] = []
    if paths is None:
        paths = set(backend.list())
    required = {"meta/info.json", "meta/stats.json", "meta/tasks.parquet"}
    missing = sorted(required - paths)
    if missing:
        messages.append(f"missing LeRobot v3 files: {missing}")
    for prefix, suffix, label in (
        ("data/", ".parquet", "data"),
        ("meta/episodes/", ".parquet", "episode metadata"),
        ("videos/", ".mp4", "video"),
    ):
        if not any(path.startswith(prefix) and path.endswith(suffix) for path in paths):
            messages.append(f"no {label} shards")
    try:
        if (
            json.loads(backend.read("meta/info.json")).get("codebase_version")
            != LEROBOT_SCHEMA_VERSION
        ):
            messages.append("meta/info.json is not codebase_version v3.0")
    except Exception as exc:
        messages.append(f"invalid meta/info.json: {type(exc).__name__}: {exc}")
    if require_success:
        messages.extend(validate_completion_marker(backend, paths))
        messages.extend(validate_payload_checksums(backend, paths))
    status = "failed" if messages else "passed"
    return ConformanceResult(
        "passed" if paths else "failed",
        status,
        messages=tuple(messages),
        evidence={"validator": "lerobot_v3_structural@3"},
    )


def _feature(spec: LeRobotV3FeatureSpec) -> dict[str, Any]:
    return {
        "dtype": np.dtype(spec.dtype).name,
        "shape": list(spec.shape),
        "names": list(spec.feature_names),
    }


def _validate_tensor(spec: LeRobotV3FeatureSpec, tensor: NamedTensor) -> np.ndarray:
    value = np.asarray(tensor.data)
    if value.shape != spec.shape or value.dtype != np.dtype(spec.dtype):
        raise ValueError(
            f"{spec.stream_id!r} expected {spec.dtype} {spec.shape}, got {value.dtype} {value.shape}"
        )
    if tensor.feature_names != spec.feature_names or tensor.units != spec.units:
        raise ValueError(
            f"{spec.stream_id!r} feature names or units do not match config"
        )
    if (
        tensor.representation != spec.representation
        or tensor.coordinate_frame != spec.coordinate_frame
    ):
        raise ValueError(
            f"{spec.stream_id!r} representation or coordinate frame does not match config"
        )
    return np.array(value, copy=True)
