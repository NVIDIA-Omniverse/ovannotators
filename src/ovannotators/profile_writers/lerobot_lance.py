# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native LeRobot-compatible Lance profile writer."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ovannotators._internal.filesystem import private_work_directory

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
    build_v3_episodes_table,
    build_v3_info,
    build_v3_video_info,
    camera_feature_key,
    camera_source_stride,
    camera_timestamp_key,
    compute_episode_stats,
    encode_video_frames,
    flatten_dict,
    open_native_parquet,
    require_pyarrow,
    require_pylance,
    resample_camera_zoh,
    run_offline_loader,
    serialize_stats,
    stats_dtype,
    stream_schema,
    symlinks_outside_root,
    to_lance_column,
    validate_feature_keys,
    write_tasks_parquet,
)
from ovannotators._internal.mp4_mux import parse_mp4_video_index, validate_mp4_clip
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

LEROBOT_LANCEDB_VERSION = "0.3.1"
LEROBOT_LANCEDB_COMMIT = "40bcb659a52df5511cb1c8035b80775a2ff773a7"
LEROBOT_SCHEMA_VERSION = "v3.0"
_EPISODE_VIDEO_DIR = ".episode-videos"


@dataclass(frozen=True)
class LeRobotLanceFeatureSpec:
    """Semantic declaration for one vector feature."""

    stream_id: str
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    representation: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    coordinate_frame: str | None = None

    def __post_init__(self) -> None:
        if not self.stream_id or not self.feature_names or not self.representation:
            raise ValueError(
                "stream_id, feature_names, and representation are required"
            )
        if len(self.shape) != 1 or self.shape[0] <= 0:
            raise ValueError(
                f"{self.stream_id} shape must be one positive dimension, got {self.shape}"
            )
        if len(self.feature_names) != self.shape[0]:
            raise ValueError("feature_names must name every vector element")
        if len(self.units) not in (1, len(self.feature_names)):
            raise ValueError("units must contain one value or one per feature")
        np.dtype(self.dtype)


@dataclass(frozen=True)
class LeRobotLanceWriterConfig:
    """Configuration for the native Lance video-blob layout.

    Attributes:
        fps: Constant dataset frame rate. Every camera rate must divide it.
        camera_stream_ids: Unique RGB camera stream IDs written as video blobs.
        state: Semantic declaration of the state vector.
        action: Semantic declaration of the action vector.
        repo_id: Optional LeRobot ``namespace/dataset`` identifier. It is
            validated when given but not written to the dataset; pass it to
            the canonical loader when opening the tree.
        timestamp_tolerance_ns: Accepted CFR timestamp error, 0 to 100000 ns.
        video_codec: NVENC video codec: ``"h264"`` (default), ``"hevc"``, or ``"av1"``.
        video_lossless: Enable lossless HEVC 4:4:4 encoding when True.
        video_preset: Optional NVENC preset ``"P1"`` through ``"P7"``.
        video_gop: Keyframe interval in frames. Defaults to 2, LeRobot's own
            default, so a Lance range read fetches little more than the
            requested frame; ``None`` uses the NVENC default.
        staging_dir: Local directory for encoded episode videos and the native
            Lance tree. It must not exist when the first episode begins.
            Its parent must be trusted: writable only by you, or sticky like
            the system temporary directory that the default uses.
        canonical_loader_check: ``"skip"`` (default) publishes without importing
            ``lerobot-lancedb`` and records ``canonical_loader="not_run"`` in
            ``conformance.json``. ``"required"`` publishes only after the pinned
            ``LanceDBDataset`` loader (installed in an isolated conformance
            environment) opens the tree and decodes a sample.
        keep_staging: Keep the staging tree after ``finalize`` for inspection.
            By default it is removed on success. A ``finalize`` that fails
            before publication removes the partial tree but keeps the encoded
            episode videos; one that fails during publication keeps the built
            tree. Either can be retried.
    """

    fps: int
    camera_stream_ids: tuple[str, ...]
    state: LeRobotLanceFeatureSpec
    action: LeRobotLanceFeatureSpec
    repo_id: str | None = None
    timestamp_tolerance_ns: int = 100_000
    video_codec: VideoCodec = "h264"
    video_lossless: bool = False
    video_preset: VideoPreset | None = None
    video_gop: int | None = 2
    staging_dir: str | Path | None = None
    canonical_loader_check: str = "skip"
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
            raise ValueError("repo_id must have the form namespace/dataset")
        if not 0 <= self.timestamp_tolerance_ns <= 100_000:
            raise ValueError("timestamp_tolerance_ns must be in [0, 100000]")
        if self.state.stream_id == self.action.stream_id:
            raise ValueError("state and action stream IDs must differ")
        validate_feature_keys(
            self.camera_stream_ids,
            self.state.stream_id,
            self.action.stream_id,
            lance=True,
        )
        if self.canonical_loader_check not in ("required", "skip"):
            raise ValueError("canonical_loader_check must be 'required' or 'skip'")
        validate_video_options(
            self.video_codec,
            self.video_lossless,
            self.video_preset,
            gop=self.video_gop,
        )


@ProfileWriterRegistry.register
class LeRobotLanceWriter:
    """Write LeRobot-Lance tables natively using ``pylance``, ``pyarrow``, and NVENC.

    Produces ``frames.lance/``, ``videos.lance/`` (Lance blob v2 with MP4
    byte-offset keyframe indices), ``meta.lance/``, and ``meta/`` without
    importing ``lerobot``, ``lerobot-lancedb``, ``lancedb``, or ``av``.
    """

    name = "lerobot_lance"

    def __init__(
        self,
        config: LeRobotLanceWriterConfig,
        *,
        video_encoder: VideoEncodeFn | None = None,
    ) -> None:
        self.config = config
        self._video_encoder: VideoEncodeFn = video_encoder or encode_video_frames
        self._stage = (
            Path(config.staging_dir).resolve()
            if config.staging_dir
            else Path(
                tempfile.gettempdir(), f"ovannotators-lerobot-lance-{uuid.uuid4().hex}"
            )
        )
        self._staging = NativeStage(self._stage)
        self._active: EpisodeEnvelope | None = None
        self._active_rows: list[dict[str, Any]] = []
        self._active_cameras: dict[str, list[np.ndarray]] = {}
        self._active_camera_times: dict[str, list[int]] = {}
        self._camera_shapes: dict[str, tuple[int, ...]] = {}
        self._camera_rates: dict[str, float] = {}
        self._episodes: list[dict[str, Any]] = []
        self._episode_ids: list[str] = []
        self._tasks: dict[str, int] = {}
        self._total_frames = 0
        self._dataset_id: str | None = None
        self._stream_schema: tuple[tuple[Any, ...], ...] | None = None
        # Only configured streams are written, so only they must stay constant.
        self._schema_stream_ids = frozenset(
            {
                self.config.state.stream_id,
                self.config.action.stream_id,
                *self.config.camera_stream_ids,
            }
        )
        self._embodiment: str | None = None
        self._last_timestamp_ns: int | None = None
        self._backend_initialized = False
        self._built: tuple[dict[str, str], ConformanceResult] | None = None
        self._finalization_started = False
        self._finalized = False

    @property
    def staging_root(self) -> Path:
        """Return the local staging directory for videos and the Lance tree.

        Each ``end_episode`` writes its encoded MP4 clips here, so raw frames
        are not held in memory across episodes. ``finalize`` builds the Lance
        tree here and removes the directory after success unless
        ``keep_staging`` is set. A ``finalize`` that fails before publication
        keeps only the encoded episode videos; one that fails during
        publication keeps the built tree, so a retry republishes the same
        bytes.
        """
        return self._stage

    def begin_episode(self, episode: EpisodeEnvelope, backend: Backend) -> None:
        """Start an episode after validating its streams against the config.

        Args:
            episode: Episode identity, task, and stream descriptors.
            backend: Output backend; it must be empty before the first episode.

        Raises:
            RuntimeError: If another episode is active or ``finalize`` has
                been called, even if it failed.
            ValueError: If the backend or staging directory is not empty, a
                required stream is missing or incompatible, or the dataset
                identity or stream schema changed between episodes.
        """
        if self._finalization_started:
            # A retried finalize may republish an already-built tree, which
            # would silently omit any episode added after the first attempt.
            raise RuntimeError("cannot begin an episode after finalization starts")
        if self._active is not None:
            raise RuntimeError("end the active episode before beginning another")
        require_disjoint_disk_staging(self._stage, backend)
        if not self._backend_initialized:
            require_atomic_create(backend)
            if backend.list():
                raise ValueError("LeRobot Lance output must be empty")
            self._staging.reserve()
            self._backend_initialized = True
        shapes, rates = self._validate_episode(episode)
        if self._dataset_id is None:
            self._dataset_id, self._embodiment = episode.dataset_id, episode.embodiment
            self._camera_shapes, self._camera_rates = shapes, rates
            self._stream_schema = stream_schema(episode, self._schema_stream_ids)
        elif (episode.dataset_id, episode.embodiment) != (
            self._dataset_id,
            self._embodiment,
        ):
            raise ValueError("dataset_id and embodiment must remain constant")
        elif stream_schema(episode, self._schema_stream_ids) != self._stream_schema:
            raise ValueError("stream schema changed between episodes")
        if episode.episode_id in self._episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        self._active = episode
        self._active_rows = []
        self._active_cameras = {item: [] for item in self.config.camera_stream_ids}
        self._active_camera_times = {item: [] for item in self.config.camera_stream_ids}
        self._last_timestamp_ns = None

    def write_sample(self, sample: SampleEnvelope, backend: Backend) -> None:
        """Buffer one sample of the active episode.

        Cameras slower than ``fps`` must appear only on their cadence slots and
        mark every other slot ``not_sampled_at_source_rate`` in
        ``SampleEnvelope.missing_data``.

        Args:
            sample: Sample with state, action, and camera observations.
            backend: Unused; samples are staged in memory until ``finalize``.

        Raises:
            RuntimeError: If no episode is active.
            ValueError: If the timestamp is not strictly increasing or is off
                the CFR timeline, or a signal or camera frame is missing,
                off-cadence, or incompatible with the config.
        """
        del backend
        if self._active is None:
            raise RuntimeError("begin_episode must be called before write_sample")
        index = len(self._active_rows)
        if (
            self._last_timestamp_ns is not None
            and sample.timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("sample timestamps must be strictly increasing")
        error = abs(
            sample.timestamp_ns
            - self._active.source_origin_ns
            - round(index * 1e9 / self.config.fps)
        )
        if error > self.config.timestamp_tolerance_ns:
            raise ValueError(f"sample timestamp is off the CFR timeline by {error} ns")
        row, cameras = self._map_sample(sample, index)
        self._active_rows.append(row)
        for stream_id, frame in cameras.items():
            self._active_cameras[stream_id].append(frame)
            self._active_camera_times[stream_id].append(sample.timestamp_ns)
        self._last_timestamp_ns = sample.timestamp_ns

    def end_episode(self, backend: Backend) -> None:
        """Align cameras to the CFR timeline, encode the episode, and stage it.

        Camera frames are encoded to MP4 and written under ``staging_root``;
        only rows, statistics, and video byte indices stay in memory. A failed
        call leaves the episode active so it can be retried.

        Args:
            backend: Unused; nothing is published until ``finalize``.

        Raises:
            RuntimeError: If no episode is active.
            ValueError: If the episode is empty, a camera has no frame within
                its cadence for some target timestamp, or an encoded clip does
                not match the episode.
        """
        del backend
        if self._active is None:
            raise RuntimeError("no active episode")
        self._staging.verify()
        if not self._active_rows:
            raise ValueError("an episode must contain at least one sample")
        episode_index = len(self._episodes)
        task_index = self._tasks.get(self._active.task, len(self._tasks))
        targets = [int(row["source.timestamp_ns"]) for row in self._active_rows]
        cameras: dict[str, list[np.ndarray]] = {}
        camera_times: dict[str, list[int]] = {}
        for stream_id in self.config.camera_stream_ids:
            frames, source_times = resample_camera_zoh(
                self._active_cameras[stream_id],
                self._active_camera_times[stream_id],
                targets,
                source_rate_hz=self._camera_rates[stream_id],
                tolerance_ns=self.config.timestamp_tolerance_ns,
                stream_id=stream_id,
            )
            cameras[stream_id] = frames
            camera_times[camera_timestamp_key(stream_id)] = source_times
        rows = [
            {
                **row,
                **{key: times[frame_index] for key, times in camera_times.items()},
                "index": self._total_frames + frame_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": float(np.float32(frame_index / self.config.fps)),
                "task_index": task_index,
            }
            for frame_index, row in enumerate(self._active_rows)
        ]
        features = _features(self.config, self._camera_shapes)
        stats = compute_episode_stats(
            _vector_data(rows, features, self.config.fps),
            {
                camera_feature_key(stream_id): cameras[stream_id]
                for stream_id in self.config.camera_stream_ids
            },
        )
        video_dir = self._staging.path(_EPISODE_VIDEO_DIR, directory=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        videos = []
        for camera_index, stream_id in enumerate(self.config.camera_stream_ids):
            frames = cameras[stream_id]
            mp4_bytes = self._video_encoder(
                frames,
                self.config.fps,
                codec=self.config.video_codec,
                lossless=self.config.video_lossless,
                preset=self.config.video_preset,
                gop=self.config.video_gop,
            )
            height, width = frames[0].shape[:2]
            duration_s = validate_mp4_clip(
                mp4_bytes,
                frame_count=len(rows),
                fps=self.config.fps,
                width=width,
                height=height,
                codec=self.config.video_codec,
            )
            path = self._staging.path(
                Path(_EPISODE_VIDEO_DIR) / f"{episode_index:06d}-{camera_index:03d}.mp4"
            )
            path.write_bytes(mp4_bytes)
            self._staging.record_file(path)
            videos.append(
                {
                    "video_key": camera_feature_key(stream_id),
                    "path": path,
                    "duration_s": duration_s,
                    "byte_index": parse_mp4_video_index(mp4_bytes),
                }
            )
        self._episodes.append(
            {
                "episode": self._active,
                "rows": rows,
                "stats": stats,
                "videos": videos,
            }
        )
        self._tasks[self._active.task] = task_index
        self._total_frames += len(rows)
        self._episode_ids.append(self._active.episode_id)
        self._active = None
        self._active_rows = []
        self._active_cameras = {}
        self._active_camera_times = {}
        self._last_timestamp_ns = None

    def finalize(self, backend: Backend) -> None:
        """Build, check, and publish the Lance tree, then write the artifacts.

        When ``canonical_loader_check`` is ``"required"``, the pinned
        ``LanceDBDataset`` loader must open the staged tree and decode a sample
        before anything is published. By default (``"skip"``), publication runs
        without importing ``lerobot-lancedb`` and records
        ``canonical_loader="not_run"``. The staging directory is removed after
        success unless ``keep_staging`` is set. A failed ``finalize`` can be
        retried. Before publication starts, a failure removes the partial tree
        and keeps the encoded episode videos. Once publication starts, the
        built tree is kept and a retry publishes only missing files, accepting
        files that already hold the same bytes.

        Args:
            backend: Output backend that receives the finished tree.

        Raises:
            RuntimeError: If an episode is still active.
            ValueError: If no episode was written, the canonical loader check
                fails, or the published structure is incomplete.
        """
        if self._finalized:
            return
        if self._active is not None:
            raise RuntimeError("end the active episode before finalize")
        if not self._episodes:
            raise ValueError("cannot finalize an empty LeRobot Lance dataset")
        require_disjoint_disk_staging(self._stage, backend)
        self._staging.verify()
        self._finalization_started = True
        self._build_and_publish(backend)
        if not self.config.keep_staging:
            self._staging.cleanup()

    def _build_and_publish(self, backend: Backend) -> None:
        if self._built is None:
            self._built = self._build()
        versions, canonical = self._built
        payload_paths = publish_native_payload(
            backend, self._stage, self._staging.files(exclude=_EPISODE_VIDEO_DIR)
        )
        if set(payload_paths) != payload_paths_from_listing(backend.list()):
            raise ValueError(
                "published paths do not match the staged LeRobot Lance payload"
            )
        structural = _validate_structure_paths(payload_paths)
        if structural.structural != "passed":
            raise ValueError("LeRobot Lance structure is incomplete")
        conformance = ConformanceResult(
            "passed",
            "passed",
            canonical_loader=canonical.canonical_loader,
            messages=canonical.messages,
            evidence=(
                dict(canonical.evidence)
                if canonical.canonical_loader == "passed"
                else {"serializer": "ovannotators-native-lerobot-lance"}
            ),
        )
        write_profile_artifacts(
            backend,
            profile=ProfileMetadata(
                profile_id=self.name,
                profile_family="huggingface_lerobot_lance",
                profile_version=LEROBOT_LANCEDB_VERSION,
                schema_version="lance_video_blob_v2",
                required_modalities=(
                    *(f"video:{item}" for item in self.config.camera_stream_ids),
                    self.config.state.stream_id,
                    self.config.action.stream_id,
                ),
                temporal_model="official_cfr_timestamp_plus_authoritative_source_ns",
                sharding_behavior="native_lance_frames_meta_and_video_blob_tables",
                resume_behavior="immutable_after_finalize",
            ),
            writer_record=WriterRecord(
                dataset_id=self._dataset_id or "",
                writer_name=self.name,
                writer_version="2",
                input_identifiers=tuple(self._episode_ids),
                output_identifiers=tuple(payload_paths),
                codec_versions={"video": self.config.video_codec},
                library_versions=versions,
                validator_versions={"lerobot_lance_structural": "2"},
                canonical_loader_versions=(
                    {
                        "lerobot-lancedb": (
                            f"{LEROBOT_LANCEDB_VERSION}@{LEROBOT_LANCEDB_COMMIT}"
                        )
                    }
                    if canonical.canonical_loader == "passed"
                    else {}
                ),
            ),
            loss_report=LossReport(
                consumed=(
                    "episode.identity",
                    "episode.task",
                    "episode.embodiment",
                    "sample.timestamp_ns",
                    "sample.observations",
                    "sample.policy_signals",
                ),
                transformed=(
                    "camera streams encoded directly to MP4 blob rows",
                    "dotted LeRobot feature keys mapped to Lance column names",
                    "mixed-rate cameras zero-order-hold aligned to CFR",
                ),
                notes=(
                    "No intermediate LeRobot Parquet/video dataset is produced.",
                    "source timestamps remain explicit frame-table columns.",
                    "Serialized natively with pylance, pyarrow, and NVENC without importing lerobot-lancedb or PyAV.",
                ),
            ),
            conformance=conformance,
            payload_paths=tuple(payload_paths),
            retryable=True,
        )
        self._finalized = True

    def _build(self) -> tuple[dict[str, str], ConformanceResult]:
        with private_work_directory(self._stage.parent) as build:
            versions = _write_native_lance(
                build,
                self.config,
                self._episodes,
                self._embodiment or "",
                camera_shapes=self._camera_shapes,
            )
            if self.config.canonical_loader_check == "skip":
                canonical = ConformanceResult(
                    "passed",
                    "passed",
                    canonical_loader="not_run",
                    messages=("canonical loader check skipped by configuration",),
                )
            else:
                canonical = _validate_canonical(build)
                if canonical.canonical_loader != "passed":
                    raise ValueError(
                        "LanceDBDataset validation failed: "
                        + "; ".join(canonical.messages)
                        + " (set canonical_loader_check='skip' to publish without it)"
                    )
            self._staging.verify()
            children = sorted(build.iterdir())
            unexpected = {child.name for child in children} - {
                "frames.lance",
                "videos.lance",
                "meta.lance",
                "meta",
            }
            if unexpected:
                raise ValueError(
                    f"unexpected native Lance build entries: {sorted(unexpected)}"
                )
            moved = []
            try:
                for child in children:
                    moved.append(child.rename(self._stage / child.name))
            except OSError:
                for child in moved:
                    child.rename(build / child.name)
                raise
            for child in moved:
                self._staging.adopt_tree(child)
        return versions, canonical

    def validate(self, backend: Backend) -> ConformanceResult:
        """Check the persisted tables, completion marker, and payload checksums.

        Args:
            backend: Backend holding a finished output.

        Returns:
            Structural conformance only; canonical-loader status is not rerun.
        """
        paths = set(backend.list())
        result = _validate_structure_paths(paths)
        messages = list(result.messages)
        messages.extend(validate_completion_marker(backend, paths))
        if "_ovannotators/_SUCCESS" in paths:
            messages.extend(validate_payload_checksums(backend, paths))
        return ConformanceResult(
            result.serialization,
            "failed" if messages else "passed",
            messages=tuple(messages),
            evidence=result.evidence,
        )

    def _validate_episode(
        self, episode: EpisodeEnvelope
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
        descriptors = {item.stream_id: item for item in episode.streams}
        required = {
            self.config.state.stream_id,
            self.config.action.stream_id,
            *self.config.camera_stream_ids,
        }
        missing = sorted(required - descriptors.keys())
        if missing:
            raise ValueError(f"episode is missing required streams: {missing}")
        shapes, rates = {}, {}
        for stream_id in self.config.camera_stream_ids:
            stream = descriptors[stream_id]
            if (
                stream.stream_type != "rgb"
                or np.dtype(stream.dtype) != np.uint8
                or len(stream.shape) != 3
                or stream.shape[-1] != 3
            ):
                raise ValueError(f"camera {stream_id!r} must be HWC uint8 RGB")
            if not stream.required or stream.nominal_rate_hz is None:
                raise ValueError(
                    f"camera {stream_id!r} requires a required explicit nominal rate"
                )
            if camera_source_stride(self.config.fps, stream.nominal_rate_hz) is None:
                raise ValueError(f"camera {stream_id!r} rate must divide dataset fps")
            shapes[stream_id] = tuple(stream.shape)
            rates[stream_id] = stream.nominal_rate_hz
        for spec, kind in (
            (self.config.state, "state"),
            (self.config.action, "action"),
        ):
            stream = descriptors[spec.stream_id]
            if (
                stream.stream_type != kind
                or tuple(stream.shape) != tuple(spec.shape)
                or np.dtype(stream.dtype) != np.dtype(spec.dtype)
            ):
                raise ValueError(f"descriptor schema does not match {spec.stream_id!r}")
        return shapes, rates

    def _map_sample(
        self, sample: SampleEnvelope, index: int
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        assert self._active is not None
        if sample.episode_id != self._active.episode_id:
            raise ValueError("sample episode_id does not match active episode")
        cameras = {}
        missing = []
        for stream_id in self.config.camera_stream_ids:
            stride = camera_source_stride(
                self.config.fps, self._camera_rates[stream_id]
            )
            assert stride is not None
            expected, present = index % stride == 0, stream_id in sample.observations
            if expected and not present:
                missing.append(stream_id)
            elif not expected and present:
                raise ValueError(f"camera {stream_id!r} is present outside its cadence")
            elif (
                not expected
                and sample.missing_data.get(stream_id) != "not_sampled_at_source_rate"
            ):
                missing.append(stream_id)
            if present:
                frame = np.asarray(sample.observations[stream_id].data)
                if (
                    frame.shape != self._camera_shapes[stream_id]
                    or frame.dtype != np.uint8
                ):
                    raise ValueError(f"camera {stream_id!r} has an incompatible array")
                cameras[stream_id] = np.array(frame, copy=True)
        if self.config.state.stream_id not in sample.observations:
            missing.append(self.config.state.stream_id)
        if self.config.action.stream_id not in sample.policy_signals:
            missing.append(self.config.action.stream_id)
        if missing:
            raise ValueError(f"sample is missing required signals: {sorted(missing)}")
        return {
            self.config.state.stream_id: _tensor(
                self.config.state, sample.observations[self.config.state.stream_id]
            ),
            self.config.action.stream_id: _tensor(
                self.config.action, sample.policy_signals[self.config.action.stream_id]
            ),
            "source.timestamp_ns": sample.timestamp_ns,
            "source.frame_id": sample.frame_id,
        }, cameras


def _features(
    config: LeRobotLanceWriterConfig, camera_shapes: dict[str, tuple[int, ...]]
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        config.state.stream_id: _feature(config.state),
        config.action.stream_id: _feature(config.action),
        "source.timestamp_ns": {"dtype": "int64", "shape": [1], "names": None},
        "source.frame_id": {"dtype": "int64", "shape": [1], "names": None},
    }
    for stream_id in config.camera_stream_ids:
        shape = camera_shapes[stream_id]
        features[camera_feature_key(stream_id)] = {
            "dtype": "video",
            "shape": list(shape),
            "names": ["height", "width", "channels"],
            "info": build_v3_video_info(
                height=shape[0],
                width=shape[1],
                fps=config.fps,
                codec=config.video_codec,
                lossless=config.video_lossless,
                gop=config.video_gop,
                preset=config.video_preset,
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


def _vector_data(
    rows: list[dict[str, Any]], features: dict[str, dict[str, Any]], fps: int
) -> dict[str, np.ndarray]:
    vector_data: dict[str, np.ndarray] = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        if key == "timestamp":
            vector_data[key] = np.asarray([r["frame_index"] for r in rows]) / fps
        elif tuple(ft["shape"]) == (1,):
            vector_data[key] = np.asarray(
                [r[key] for r in rows], dtype=stats_dtype(key, ft["dtype"])
            )
        else:
            vector_data[key] = np.stack(
                [np.asarray(r[key], dtype=np.dtype(ft["dtype"])) for r in rows]
            )
    return vector_data


def _write_native_lance(
    root: Path,
    config: LeRobotLanceWriterConfig,
    episodes: list[dict[str, Any]],
    embodiment: str,
    *,
    camera_shapes: dict[str, tuple[int, ...]],
) -> dict[str, str]:
    pa, pq = require_pyarrow("LeRobotLanceWriter")
    lance = require_pylance("LeRobotLanceWriter")
    root.mkdir(parents=True, exist_ok=True)
    features = _features(config, camera_shapes)

    tasks: dict[str, int] = {}
    frames_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    global_index = 0
    for episode_index, item in enumerate(episodes):
        episode, rows = item["episode"], item["rows"]
        tasks.setdefault(episode.task, len(tasks))
        dataset_from_index = global_index
        frames_rows.extend(rows)
        global_index += len(rows)
        video_meta: dict[str, Any] = {}
        for video in item["videos"]:
            vkey = video["video_key"]
            video_meta[f"videos/{vkey}/chunk_index"] = (
                episode_index // DEFAULT_CHUNK_SIZE
            )
            video_meta[f"videos/{vkey}/file_index"] = episode_index % DEFAULT_CHUNK_SIZE
            video_meta[f"videos/{vkey}/from_timestamp"] = 0.0
            video_meta[f"videos/{vkey}/to_timestamp"] = float(video["duration_s"])
        episode_records.append(
            {
                "episode_index": episode_index,
                "tasks": [episode.task],
                "length": len(rows),
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": dataset_from_index,
                "dataset_to_index": global_index,
                **video_meta,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                **flatten_dict({"stats": item["stats"]}),
            }
        )
    all_episode_stats = [item["stats"] for item in episodes]

    # Write meta/ directory
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ordered_tasks = [t for t, _ in sorted(tasks.items(), key=lambda pair: pair[1])]
    write_tasks_parquet(ordered_tasks, root / DEFAULT_TASKS_PATH)

    episodes_path = root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        build_v3_episodes_table(episode_records),
        str(episodes_path),
        compression="snappy",
        use_dictionary=True,
    )

    global_stats = aggregate_stats(all_episode_stats)
    (meta_dir / "stats.json").write_text(
        json.dumps(serialize_stats(global_stats), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    info = build_v3_info(
        codebase_version=LEROBOT_SCHEMA_VERSION,
        fps=config.fps,
        features=features,
        total_episodes=len(episodes),
        total_frames=global_index,
        total_tasks=len(tasks),
        robot_type=embodiment,
        chunks_size=DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb=DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb=DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        data_path=DEFAULT_DATA_PATH,
        video_path=DEFAULT_VIDEO_PATH,
        storage_format="lance",
    )
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    # 1. frames.lance
    frame_schema = _frame_schema(pa, features, to_lance_column)
    frames_table = pa.Table.from_pylist(
        [_lance_row(row, features, to_lance_column) for row in frames_rows],
        schema=frame_schema,
    )
    lance.write_dataset(frames_table, str(root / "frames.lance"), mode="create")

    # 2. videos.lance (Lance blob v2 requires data_storage_version="2.2")
    video_schema = pa.schema(
        [
            pa.field("video_key", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("file_index", pa.int64()),
            pa.field("file_size", pa.int64()),
            pa.field("moov_offset", pa.int64()),
            pa.field("moov_size", pa.int64()),
            pa.field("kf_indices", pa.list_(pa.int64())),
            pa.field("kf_positions", pa.list_(pa.int64())),
            lance.blob_field("video_bytes"),
        ]
    )

    def video_batches():
        for episode_index, item in enumerate(episodes):
            for video in item["videos"]:
                byte_index = video["byte_index"]
                yield pa.RecordBatch.from_arrays(
                    [
                        pa.array([video["video_key"]], type=pa.string()),
                        pa.array(
                            [episode_index // DEFAULT_CHUNK_SIZE], type=pa.int64()
                        ),
                        pa.array([episode_index % DEFAULT_CHUNK_SIZE], type=pa.int64()),
                        pa.array([byte_index["file_size"]], type=pa.int64()),
                        pa.array([byte_index["moov_offset"]], type=pa.int64()),
                        pa.array([byte_index["moov_size"]], type=pa.int64()),
                        pa.array([byte_index["kf_indices"]], type=pa.list_(pa.int64())),
                        pa.array(
                            [byte_index["kf_positions"]], type=pa.list_(pa.int64())
                        ),
                        lance.blob_array([Path(video["path"]).read_bytes()]),
                    ],
                    schema=video_schema,
                )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(video_schema, video_batches()),
        str(root / "videos.lance"),
        mode="create",
        data_storage_version="2.2",
    )

    # 3. meta.lance
    meta_schema = pa.schema(
        [pa.field("path", pa.string()), pa.field("data", pa.large_binary())]
    )
    meta_files = sorted(path for path in meta_dir.rglob("*") if path.is_file())
    meta_table = pa.Table.from_pylist(
        [
            {
                "path": path.relative_to(meta_dir).as_posix(),
                "data": path.read_bytes(),
            }
            for path in meta_files
        ],
        schema=meta_schema,
    )
    lance.write_dataset(meta_table, str(root / "meta.lance"), mode="create")

    return {
        "pylance": getattr(lance, "__version__", "unknown"),
        "pyarrow": pa.__version__,
        "target_lerobot_lancedb_schema": LEROBOT_LANCEDB_VERSION,
    }


def _frame_schema(pa: Any, features: dict[str, dict[str, Any]], to_column: Any) -> Any:
    fields = []
    for key, feature in features.items():
        if feature["dtype"] == "video":
            continue
        dtype = pa.from_numpy_dtype(np.dtype(feature["dtype"]))
        shape = tuple(feature["shape"])
        fields.append(
            pa.field(
                to_column(key),
                dtype if shape == (1,) else pa.list_(dtype, int(np.prod(shape))),
            )
        )
    return pa.schema(fields)


def _lance_row(
    row: dict[str, Any], features: dict[str, dict[str, Any]], to_column: Any
) -> dict[str, Any]:
    result = {}
    for key, feature in features.items():
        if feature["dtype"] == "video":
            continue
        value = row[key]
        result[to_column(key)] = (
            np.asarray(value).item()
            if tuple(feature["shape"]) == (1,)
            else np.asarray(value).reshape(-1).tolist()
        )
    return result


def _validate_canonical(
    root: Path, *, expected_version: str = LEROBOT_LANCEDB_VERSION
) -> ConformanceResult:
    """Return ``passed`` only after ``LanceDBDataset`` opens ``root`` and decodes a sample.

    Args:
        root: Staged native Lance tree.
        expected_version: ``lerobot-lancedb`` version that must be installed.
            Defaults to the pinned target; a drift probe passes the candidate.

    Returns:
        The loader result. Its evidence records the version actually exercised.
    """
    messages = _canonical_preflight(root)
    if messages:
        return ConformanceResult(
            "passed", "failed", canonical_loader="not_run", messages=tuple(messages)
        )
    outcome = run_offline_loader(_OFFLINE_LOADER_SCRIPT, str(root), expected_version)
    if "error" in outcome:
        return ConformanceResult(
            "passed",
            "passed",
            canonical_loader="failed",
            messages=(f"LanceDBDataset failed: {outcome['error']}",),
        )
    return ConformanceResult(
        "passed",
        "passed",
        canonical_loader="passed",
        evidence={"loader": f"lerobot-lancedb=={outcome['version']}"},
    )


def _canonical_preflight(root: Path) -> list[str]:
    """Reject missing or unreadable local metadata before it can trigger Hub fallback."""
    messages = symlinks_outside_root(root)
    if messages:
        return messages
    try:
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        if info.get("storage_format") not in (None, "lance"):
            return ["meta/info.json storage_format must be 'lance' or omitted"]
        if info["total_tasks"] > 0:
            with open_native_parquet(root / DEFAULT_TASKS_PATH) as tasks:
                tasks.read()
        if info["total_episodes"] > 0:
            episodes = sorted((root / "meta/episodes").glob("*/*.parquet"))
            if not episodes:
                return ["missing local metadata: meta/episodes/*/*.parquet"]
            for path in episodes:
                with open_native_parquet(path) as parquet:
                    parquet.read()
        stats = root / "meta/stats.json"
        if stats.exists():
            json.loads(stats.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid local metadata: {type(exc).__name__}: {exc}"]
    return []


_OFFLINE_LOADER_SCRIPT = (
    "import json, sys\n"
    "from ovannotators.profile_writers.lerobot_lance import _load_official\n"
    f"print({_LOADER_RESULT_MARKER!r} + json.dumps(_load_official(*sys.argv[1:3])))\n"
)


def _load_official(root: str, expected_version: str) -> dict[str, str]:
    """Open local Lance output and decode a sample, returning version or error."""
    try:
        import importlib.metadata

        installed = importlib.metadata.version("lerobot-lancedb")
        if installed != expected_version:
            raise RuntimeError(
                f"expected lerobot-lancedb=={expected_version}, found {installed}"
            )
        from lerobot_lancedb import LanceDBDataset  # type: ignore[import-not-found]

        dataset = LanceDBDataset(root=Path(root), return_uint8=True)
        if len(dataset) <= 0:
            raise RuntimeError("canonical loader returned an empty dataset")
        dataset[0]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"version": installed}


def _validate_structure_paths(paths: list[str] | set[str]) -> ConformanceResult:
    names = set(paths)
    missing = [name for name in ("meta/info.json",) if name not in names]
    for table in ("frames.lance/", "videos.lance/", "meta.lance/"):
        if not any(name.startswith(table) for name in names):
            missing.append(table)
    status = "failed" if missing else "passed"
    return ConformanceResult(
        "passed" if names else "failed",
        status,
        messages=((f"missing: {missing}",) if missing else ()),
        evidence={"validator": "lerobot_lance_structural@2"},
    )


def _feature(spec: LeRobotLanceFeatureSpec) -> dict[str, Any]:
    return {
        "dtype": spec.dtype,
        "shape": list(spec.shape),
        "names": list(spec.feature_names),
    }


def _tensor(spec: LeRobotLanceFeatureSpec, tensor: NamedTensor) -> np.ndarray:
    value = np.asarray(tensor.data)
    if value.shape != spec.shape or value.dtype != np.dtype(spec.dtype):
        raise ValueError(f"{spec.stream_id!r} tensor schema does not match")
    if (
        tensor.feature_names,
        tensor.units,
        tensor.representation,
        tensor.coordinate_frame,
    ) != (spec.feature_names, spec.units, spec.representation, spec.coordinate_frame):
        raise ValueError(f"{spec.stream_id!r} tensor semantics do not match")
    return np.array(value, copy=True)


# Backwards-compatible helper aliases for unit tests
_camera_key = camera_feature_key
_camera_timestamp_key = camera_timestamp_key
_stride = camera_source_stride
