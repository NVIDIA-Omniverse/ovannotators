# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Isaac-GR00T N1.7-compatible LeRobot v2 profile writer."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

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
    ProfileMetadata,
    SampleEnvelope,
    WriterRecord,
)
from ovannotators.profile_writers._registry import ProfileWriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends import Backend
    from ovannotators.profile_writers import NamedTensor

GROOT_RELEASE_TAG = "n1.7-release"
GROOT_RELEASE_COMMIT = "23ace64f17aa5015259b8609d371eb61a357c776"
PROFILE_VERSION = "isaac-groot-n1.7-lerobot-v2"
SCHEMA_VERSION = "2.1"
_STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
_MAX_TIMESTAMP_TOLERANCE_NS = 100_000


@dataclass(frozen=True)
class GrootVideoModality:
    """Explicit mapping from a GR00T camera key to an envelope stream."""

    key: str
    stream_id: str

    def __post_init__(self) -> None:
        if not self.key or not self.stream_id:
            raise ConfigurationError("video modality key and stream_id are required")


@dataclass(frozen=True)
class GrootStateModality:
    """One explicitly positioned state group in ``observation.state``."""

    key: str
    start: int
    end: int
    feature_names: tuple[str, ...]
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_slice(self.key, self.start, self.end, self.feature_names, self.units)


@dataclass(frozen=True)
class GrootActionModality:
    """One action group and the exact GR00T processor interpretation."""

    key: str
    start: int
    end: int
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    representation: Literal["absolute", "relative"]
    control_type: Literal["eef", "non_eef"]
    action_format: Literal["default", "xyz_rot6d", "xyz_rotvec"]
    state_key: str | None = None

    def __post_init__(self) -> None:
        _validate_slice(self.key, self.start, self.end, self.feature_names, self.units)
        if self.representation not in ("absolute", "relative"):
            raise ConfigurationError(
                "action representation must be absolute or relative"
            )
        if self.control_type not in ("eef", "non_eef"):
            raise ConfigurationError("action control_type must be eef or non_eef")
        if self.action_format not in ("default", "xyz_rot6d", "xyz_rotvec"):
            raise ConfigurationError("unsupported GR00T action format")


@dataclass(frozen=True)
class GrootNormalizationConfig:
    """Versioned normalization behavior expected by the GR00T processor."""

    state: tuple[tuple[str, Literal["min_max", "mean_std", "sin_cos"]], ...]
    action: tuple[tuple[str, Literal["min_max", "mean_std"]], ...]
    use_percentiles: bool
    clip_outliers: bool


@dataclass(frozen=True)
class GrootLeRobotV2WriterConfig:
    """Required modality and processing contract for the GR00T v2 profile."""

    embodiment: str
    video_modalities: tuple[GrootVideoModality, ...]
    state_signal: str
    state_modalities: tuple[GrootStateModality, ...]
    action_signal: str
    action_modalities: tuple[GrootActionModality, ...]
    action_horizon: int
    normalization: GrootNormalizationConfig
    relative_action_policy: Literal["disabled", "groot_processor"]
    fps: int = 30
    episodes_per_chunk: int = 1000

    def __post_init__(self) -> None:
        if any(
            mode not in ("min_max", "mean_std", "sin_cos")
            for _, mode in self.normalization.state
        ):
            raise ConfigurationError("unsupported state normalization mode")
        if any(
            mode not in ("min_max", "mean_std") for _, mode in self.normalization.action
        ):
            raise ConfigurationError("unsupported action normalization mode")
        if not self.embodiment or not self.state_signal or not self.action_signal:
            raise ConfigurationError(
                "embodiment, state_signal, and action_signal are required"
            )
        if (
            not self.video_modalities
            or not self.state_modalities
            or not self.action_modalities
        ):
            raise ConfigurationError(
                "video, state, and action modalities must be explicit"
            )
        if self.action_horizon < 1 or self.fps < 1 or self.episodes_per_chunk < 1:
            raise ConfigurationError(
                "action_horizon, fps, and episodes_per_chunk must be positive"
            )
        if self.relative_action_policy not in ("disabled", "groot_processor"):
            raise ConfigurationError(
                "relative_action_policy must be 'disabled' or 'groot_processor'; "
                f"got {self.relative_action_policy!r}"
            )
        _require_unique(
            "video modality keys", [item.key for item in self.video_modalities]
        )
        _require_unique(
            "video stream IDs", [item.stream_id for item in self.video_modalities]
        )
        _validate_contiguous_slices("state", self.state_modalities)
        _validate_contiguous_slices("action", self.action_modalities)
        state_keys = {item.key for item in self.state_modalities}
        action_keys = {item.key for item in self.action_modalities}
        for item in self.action_modalities:
            if (
                item.representation == "relative"
                and (item.state_key or item.key) not in state_keys
            ):
                raise ConfigurationError(
                    f"relative action '{item.key}' references unknown state key "
                    f"'{item.state_key or item.key}'"
                )
            if item.representation == "relative":
                state_item = next(
                    state
                    for state in self.state_modalities
                    if state.key == (item.state_key or item.key)
                )
                if item.end - item.start != state_item.end - state_item.start:
                    raise ConfigurationError(
                        f"relative action '{item.key}' width must match reference "
                        f"state '{state_item.key}' width"
                    )
                if item.control_type == "eef":
                    raise ConfigurationError(
                        "relative EEF actions are not supported until the pinned "
                        "GR00T pose-composition semantics are implemented"
                    )
        has_relative = any(
            item.representation == "relative" for item in self.action_modalities
        )
        if has_relative != (self.relative_action_policy == "groot_processor"):
            raise ConfigurationError(
                "relative_action_policy must be 'groot_processor' exactly when a relative action is configured"
            )
        if {key for key, _ in self.normalization.state} != state_keys:
            raise ConfigurationError(
                "normalization.state keys must exactly match state modalities"
            )
        if {key for key, _ in self.normalization.action} != action_keys:
            raise ConfigurationError(
                "normalization.action keys must exactly match action modalities"
            )
        _require_unique(
            "state normalization keys", [key for key, _ in self.normalization.state]
        )
        _require_unique(
            "action normalization keys", [key for key, _ in self.normalization.action]
        )


@dataclass
class _OpenEpisode:
    envelope: EpisodeEnvelope
    episode_index: int
    task_index: int
    rows: list[dict[str, Any]]
    videos: dict[str, list[np.ndarray]]
    last_timestamp_ns: int | None = None


@ProfileWriterRegistry.register
class GrootLeRobotV2Writer:
    """Serialize the pinned Isaac-GR00T N1.7 LeRobot v2-derived layout.

    Camera frames must have even height and width (H.264 4:2:0); the writer
    never crops. All raw frames of the active episode stay in memory until
    ``end_episode``, about ``frames * cameras * height * width * 3`` bytes.
    """

    name: ClassVar[str] = "groot_lerobot_v2"

    def __init__(self, config: GrootLeRobotV2WriterConfig) -> None:
        self.config = config
        self._active: _OpenEpisode | None = None
        self._episodes: list[dict[str, Any]] = []
        self._tasks: dict[str, int] = {}
        self._payload_paths: list[str] = []
        self._all_state: list[np.ndarray] = []
        self._all_action: list[np.ndarray] = []
        self._episode_state_action: list[tuple[np.ndarray, np.ndarray]] = []
        self._dataset_id: str | None = None
        self._global_index = 0
        self._finalized = False
        self._saw_reorder = False
        self._numeric_casts: set[str] = set()
        self._dropped_alpha = False
        self._video_shapes: dict[str, tuple[int, ...]] = {}

    def begin_episode(self, episode: EpisodeEnvelope, backend: Backend) -> None:
        """Validate the episode schema and begin bounded episode accumulation."""
        del backend
        if self._finalized:
            raise ConfigurationError("cannot begin an episode after finalize")
        if self._active is not None:
            raise ConfigurationError("end the active episode before beginning another")
        if episode.embodiment != self.config.embodiment:
            raise ConfigurationError(
                f"episode embodiment '{episode.embodiment}' does not match configured "
                f"'{self.config.embodiment}'"
            )
        if self._dataset_id is not None and episode.dataset_id != self._dataset_id:
            raise ConfigurationError("one writer lifecycle cannot mix dataset IDs")
        if any(item["episode_id"] == episode.episode_id for item in self._episodes):
            raise ConfigurationError(f"duplicate episode_id: {episode.episode_id}")
        stream_map = {stream.stream_id: stream for stream in episode.streams}
        required = [item.stream_id for item in self.config.video_modalities]
        required.extend((self.config.state_signal, self.config.action_signal))
        missing = sorted(set(required) - set(stream_map))
        if missing:
            raise ConfigurationError(
                f"episode is missing configured streams: {missing}"
            )
        for video in self.config.video_modalities:
            if stream_map[video.stream_id].stream_type != "rgb":
                raise ConfigurationError(
                    f"video stream '{video.stream_id}' must have stream_type 'rgb'"
                )
            descriptor = stream_map[video.stream_id]
            shape = descriptor.shape
            if (
                descriptor.dtype != "uint8"
                or len(shape) != 3
                or shape[-1] not in (3, 4)
            ):
                raise ConfigurationError(
                    f"video stream '{video.stream_id}' must declare HxWx3/4 uint8"
                )
            if shape[0] % 2 or shape[1] % 2:
                raise ConfigurationError(
                    f"video stream '{video.stream_id}' (key '{video.key}') has odd "
                    f"dimensions {shape[0]}x{shape[1]} (height x width); H.264 4:2:0 "
                    "needs even height and width and this profile does not crop "
                    "frames, so supply even-sized frames"
                )
            shape = (*shape[:2], 3)
            previous = self._video_shapes.get(video.key)
            if previous is not None and shape != previous:
                raise ConfigurationError(
                    f"video modality '{video.key}' changed shape from {previous} to {shape}"
                )
        expected_state_dim = self.config.state_modalities[-1].end
        expected_action_dim = self.config.action_modalities[-1].end
        if stream_map[self.config.state_signal].shape != (expected_state_dim,):
            raise ConfigurationError(
                "declared state stream shape does not match configured slices"
            )
        if stream_map[self.config.action_signal].shape != (expected_action_dim,):
            raise ConfigurationError(
                "declared action stream shape does not match configured slices"
            )
        if self._dataset_id is None:
            self._dataset_id = episode.dataset_id
        for video in self.config.video_modalities:
            shape = stream_map[video.stream_id].shape
            self._video_shapes.setdefault(video.key, (*shape[:2], 3))
        task_index = self._tasks.setdefault(episode.task, len(self._tasks))
        self._active = _OpenEpisode(
            episode,
            len(self._episodes),
            task_index,
            [],
            {item.key: [] for item in self.config.video_modalities},
        )

    def write_sample(self, sample: SampleEnvelope, backend: Backend) -> None:
        """Validate all configured modalities before accepting a sample."""
        del backend
        active = self._require_active()
        if sample.episode_id != active.envelope.episode_id:
            raise ConfigurationError(
                f"sample episode '{sample.episode_id}' does not match active episode "
                f"'{active.envelope.episode_id}'"
            )
        if (
            active.last_timestamp_ns is not None
            and sample.timestamp_ns <= active.last_timestamp_ns
        ):
            raise ConfigurationError("sample timestamps must be strictly increasing")
        frame_index = len(active.rows)
        if sample.frame_id != frame_index:
            raise ConfigurationError(
                f"sample frame_id {sample.frame_id} must equal the per-episode "
                f"frame_index {frame_index}; GR00T video frames are addressed by "
                "contiguous zero-based frame_index"
            )
        expected_ns = round(frame_index * 1_000_000_000 / self.config.fps)
        error_ns = abs(
            sample.timestamp_ns - active.envelope.source_origin_ns - expected_ns
        )
        if error_ns > _MAX_TIMESTAMP_TOLERANCE_NS:
            raise ConfigurationError(
                "sample timestamp is off the constant-frame-rate timeline "
                f"({self.config.fps} fps) by {error_ns} ns; the limit is "
                f"{_MAX_TIMESTAMP_TOLERANCE_NS} ns"
            )
        missing_observations = [
            item.stream_id
            for item in self.config.video_modalities
            if item.stream_id not in sample.observations
        ]
        if self.config.state_signal not in sample.observations:
            missing_observations.append(self.config.state_signal)
        if missing_observations:
            details = {
                key: sample.missing_data.get(key, "not provided")
                for key in missing_observations
            }
            raise ConfigurationError(
                f"sample is missing required observations: {details}"
            )
        if self.config.action_signal not in sample.policy_signals:
            raise ConfigurationError(
                f"sample is missing required policy signal '{self.config.action_signal}'"
            )

        state_tensor = sample.observations[self.config.state_signal]
        action_tensor = sample.policy_signals[self.config.action_signal]
        state, state_reordered, state_cast = self._ordered_vector(
            state_tensor, self.config.state_modalities, "state"
        )
        action, action_reordered, action_cast = self._ordered_vector(
            action_tensor, self.config.action_modalities, "action"
        )
        if action_tensor.representation != "absolute":
            raise ConfigurationError(
                "GR00T datasets must store absolute actions; relative conversion belongs to the pinned processor"
            )
        video_frames: dict[str, np.ndarray] = {}
        dropped_alpha = False
        for video in self.config.video_modalities:
            frame = np.asarray(sample.observations[video.stream_id].data)
            if (
                frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[-1] not in (3, 4)
            ):
                raise ConfigurationError(
                    f"video stream '{video.stream_id}' must be HxWx3/4 uint8"
                )
            if tuple(frame[..., :3].shape) != self._video_shapes[video.key]:
                raise ConfigurationError(
                    f"video stream '{video.stream_id}' shape changed within the dataset"
                )
            if frame.shape[-1] == 4:
                dropped_alpha = True
            video_frames[video.key] = np.ascontiguousarray(frame[..., :3])

        for key, frame in video_frames.items():
            active.videos[key].append(frame)
        # Transformation flags change only after the whole sample is accepted.
        self._saw_reorder |= state_reordered or action_reordered
        self._numeric_casts.update(
            note for note in (state_cast, action_cast) if note is not None
        )
        self._dropped_alpha |= dropped_alpha

        active.rows.append(
            {
                "observation.state": state,
                "action": action,
                "timestamp": (sample.timestamp_ns - active.envelope.source_origin_ns)
                / 1e9,
                "timestamp_ns": sample.timestamp_ns,
                "frame_index": frame_index,
                "episode_index": active.episode_index,
                "index": self._global_index,
                "task_index": active.task_index,
            }
        )
        self._global_index += 1
        active.last_timestamp_ns = sample.timestamp_ns

    def end_episode(self, backend: Backend) -> None:
        """Write one immutable episode parquet file and its configured videos."""
        active = self._require_active()
        if not active.rows:
            raise ConfigurationError("an episode must contain at least one sample")
        pa, pq = _require_pyarrow()
        state_dim = self.config.state_modalities[-1].end
        action_dim = self.config.action_modalities[-1].end
        schema = pa.schema(
            [
                pa.field("observation.state", pa.list_(pa.float32(), state_dim)),
                pa.field("action", pa.list_(pa.float32(), action_dim)),
                pa.field("timestamp", pa.float32()),
                pa.field("timestamp_ns", pa.int64()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
            ]
        )
        table = pa.Table.from_pylist(active.rows, schema=schema)
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        chunk = active.episode_index // self.config.episodes_per_chunk
        data_path = f"data/chunk-{chunk:03d}/episode_{active.episode_index:06d}.parquet"
        backend.write(data_path, buffer.getvalue())
        self._payload_paths.append(data_path)
        for video in self.config.video_modalities:
            video_path = (
                f"videos/chunk-{chunk:03d}/observation.images.{video.key}/"
                f"episode_{active.episode_index:06d}.mp4"
            )
            backend.write(
                video_path,
                _encode_video(active.videos[video.key], self.config.fps),
            )
            self._payload_paths.append(video_path)

        state = np.stack([row["observation.state"] for row in active.rows])
        action = np.stack([row["action"] for row in active.rows])
        self._all_state.extend(state)
        self._all_action.extend(action)
        self._episode_state_action.append((state, action))
        self._episodes.append(
            {
                "episode_index": active.episode_index,
                "length": len(active.rows),
                "tasks": [active.envelope.task],
                "task_indices": [active.task_index],
                "episode_id": active.envelope.episode_id,
            }
        )
        self._active = None

    def finalize(self, backend: Backend) -> None:
        """Write deterministic metadata, validate structurally, then mark success."""
        if self._finalized:
            return
        if self._active is not None:
            raise ConfigurationError("end the active episode before finalizing")
        if not self._episodes:
            raise ConfigurationError("cannot finalize an empty GR00T dataset")
        self._write_metadata(backend)
        result = self.validate(backend)
        if result.structural != "passed":
            raise ConfigurationError(
                "GR00T structural validation failed: " + "; ".join(result.messages)
            )
        transformed = ["normalization configured for the pinned GR00T processor"]
        if self._saw_reorder:
            transformed.append(
                "state/action features reordered to configured modality slices"
            )
        if self.config.normalization.clip_outliers:
            transformed.append("processor clips normalized outliers to [-1, 1]")
        if self.config.relative_action_policy == "groot_processor":
            transformed.append(
                "absolute actions converted to relative chunks by the GR00T processor"
            )
        transformed.extend(sorted(self._numeric_casts))
        if self._dropped_alpha:
            transformed.append(
                "RGBA video converted to RGB by dropping the alpha channel"
            )
        write_profile_artifacts(
            backend,
            profile=ProfileMetadata(
                self.name,
                "nvidia_groot_lerobot",
                PROFILE_VERSION,
                SCHEMA_VERSION,
                ("video", "state", "action", "language"),
                temporal_model="authoritative_source_timestamps_ns_and_v2_seconds",
                sharding_behavior="one_episode_per_file_chunked_by_episode_index",
            ),
            writer_record=WriterRecord(
                self._dataset_id or "",
                self.name,
                "1",
                tuple(item["episode_id"] for item in self._episodes),
                tuple(sorted(self._payload_paths)),
                codec_versions={"video": "h264"},
                library_versions={
                    "isaac_groot_tag": GROOT_RELEASE_TAG,
                    "isaac_groot_commit": GROOT_RELEASE_COMMIT,
                    "profile_schema": SCHEMA_VERSION,
                },
                canonical_loader_versions={
                    "LeRobotEpisodeLoader": f"{GROOT_RELEASE_TAG}@{GROOT_RELEASE_COMMIT}"
                },
            ),
            loss_report=LossReport(
                consumed=(
                    "observations.rgb",
                    "observations.state",
                    "policy_signals.action",
                    "task",
                ),
                transformed=tuple(transformed),
                ignored=("unconfigured optional observations",),
                notes=(
                    "Stored state and action remain absolute float32 values.",
                    f"Action horizon is {self.config.action_horizon}.",
                ),
            ),
            conformance=result,
            payload_paths=tuple(self._payload_paths),
        )
        self._finalized = True

    def validate(self, backend: Backend) -> ConformanceResult:
        """Run the profile-specific structural validator without loader fallback."""
        return validate_groot_lerobot_v2(backend)

    def canonical_modality_config(self) -> dict[str, Any]:
        """Build the exact pinned GR00T loader/processor configuration.

        The optional Isaac-GR00T package is imported only when canonical
        acceptance is run in its separately pinned environment.
        """
        try:
            from gr00t.data.types import (
                ActionConfig,
                ActionFormat,
                ActionRepresentation,
                ActionType,
                ModalityConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "canonical GR00T configuration requires Isaac-GR00T "
                f"{GROOT_RELEASE_TAG}@{GROOT_RELEASE_COMMIT}"
            ) from exc

        format_values = {
            "default": ActionFormat.DEFAULT,
            "xyz_rot6d": ActionFormat.XYZ_ROT6D,
            "xyz_rotvec": ActionFormat.XYZ_ROTVEC,
        }
        state_mean_std = [
            key for key, mode in self.config.normalization.state if mode == "mean_std"
        ]
        state_sin_cos = [
            key for key, mode in self.config.normalization.state if mode == "sin_cos"
        ]
        action_mean_std = [
            key for key, mode in self.config.normalization.action if mode == "mean_std"
        ]
        return {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=[item.key for item in self.config.video_modalities],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[item.key for item in self.config.state_modalities],
                sin_cos_embedding_keys=state_sin_cos or None,
                mean_std_embedding_keys=state_mean_std or None,
            ),
            "action": ModalityConfig(
                delta_indices=list(range(self.config.action_horizon)),
                modality_keys=[item.key for item in self.config.action_modalities],
                mean_std_embedding_keys=action_mean_std or None,
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation(item.representation),
                        type=ActionType(item.control_type),
                        format=format_values[item.action_format],
                        state_key=item.state_key,
                    )
                    for item in self.config.action_modalities
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        }

    def _ordered_vector(
        self,
        tensor: NamedTensor,
        modalities: tuple[GrootStateModality, ...] | tuple[GrootActionModality, ...],
        label: str,
    ) -> tuple[np.ndarray, bool, str | None]:
        value = np.asarray(tensor.data)
        if value.ndim != 1:
            raise ConfigurationError(f"{label} signal must be a rank-one tensor")
        desired_names = tuple(
            name for item in modalities for name in item.feature_names
        )
        desired_units = tuple(unit for item in modalities for unit in item.units)
        actual_units = tuple(tensor.units)
        reordered = False
        if tuple(tensor.feature_names) != desired_names:
            if set(tensor.feature_names) != set(desired_names) or len(
                tensor.feature_names
            ) != len(desired_names):
                raise ConfigurationError(
                    f"{label} feature names must exactly match configured names; "
                    f"got {tensor.feature_names}, expected {desired_names}"
                )
            positions = {name: index for index, name in enumerate(tensor.feature_names)}
            value = value[[positions[name] for name in desired_names]]
            actual_units = tuple(
                actual_units[positions[name]] for name in desired_names
            )
            reordered = True
        if actual_units != desired_units:
            raise ConfigurationError(
                f"{label} units must exactly match configured units; got {actual_units}, expected {desired_units}"
            )
        if value.shape != (len(desired_names),):
            raise ConfigurationError(
                f"{label} shape {value.shape} does not match configured shape {(len(desired_names),)}"
            )
        cast_note = (
            f"{label} values cast from {value.dtype} to float32"
            if value.dtype != np.dtype(np.float32)
            else None
        )
        return np.asarray(value, dtype=np.float32), reordered, cast_note

    def _write_metadata(self, backend: Backend) -> None:
        state_dim = self.config.state_modalities[-1].end
        action_dim = self.config.action_modalities[-1].end
        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [
                    name
                    for item in self.config.state_modalities
                    for name in item.feature_names
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [
                    name
                    for item in self.config.action_modalities
                    for name in item.feature_names
                ],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "timestamp_ns": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        for video in self.config.video_modalities:
            features[f"observation.images.{video.key}"] = {
                "dtype": "video",
                "shape": list(self._video_shapes[video.key]),
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": self.config.fps,
                    "video.height": self._video_shapes[video.key][0],
                    "video.width": self._video_shapes[video.key][1],
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        info = {
            "codebase_version": "v2.1",
            "profile": self.name,
            "profile_version": PROFILE_VERSION,
            "groot_release": {"tag": GROOT_RELEASE_TAG, "commit": GROOT_RELEASE_COMMIT},
            "robot_type": self.config.embodiment,
            "fps": self.config.fps,
            "chunks_size": self.config.episodes_per_chunk,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "total_episodes": len(self._episodes),
            "total_frames": self._global_index,
            "total_tasks": len(self._tasks),
            "total_videos": len(self._episodes) * len(self.config.video_modalities),
            "splits": {"train": f"0:{len(self._episodes)}"},
        }
        modality = {
            "state": {
                item.key: {
                    "start": item.start,
                    "end": item.end,
                    "original_key": "observation.state",
                    "feature_names": list(item.feature_names),
                    "units": list(item.units),
                }
                for item in self.config.state_modalities
            },
            "action": {
                item.key: {
                    "start": item.start,
                    "end": item.end,
                    "original_key": "action",
                    "feature_names": list(item.feature_names),
                    "units": list(item.units),
                    "representation": item.representation.upper(),
                    "type": item.control_type.upper(),
                    "format": item.action_format.upper(),
                    "state_key": item.state_key,
                }
                for item in self.config.action_modalities
            },
            "video": {
                item.key: {
                    "original_key": f"observation.images.{item.key}",
                    "stream_id": item.stream_id,
                }
                for item in self.config.video_modalities
            },
            "annotation": {"human.task_description": {"original_key": "task_index"}},
            "ovannotators_config": {
                "embodiment": self.config.embodiment,
                "action_horizon": self.config.action_horizon,
                "action_delta_indices": list(range(self.config.action_horizon)),
                "relative_action_policy": self.config.relative_action_policy,
                "normalization": {
                    "state": dict(self.config.normalization.state),
                    "action": dict(self.config.normalization.action),
                    "use_percentiles": self.config.normalization.use_percentiles,
                    "clip_outliers": self.config.normalization.clip_outliers,
                },
            },
        }
        stats = {
            "observation.state": _statistics(np.stack(self._all_state)),
            "action": _statistics(np.stack(self._all_action)),
        }
        self._write_json(backend, "meta/info.json", info)
        self._write_json(backend, "meta/modality.json", modality)
        self._write_json(backend, "meta/stats.json", stats)
        self._write_jsonl(backend, "meta/episodes.jsonl", self._episodes)
        tasks = [
            {"task_index": index, "task": task}
            for task, index in sorted(self._tasks.items(), key=lambda pair: pair[1])
        ]
        self._write_jsonl(backend, "meta/tasks.jsonl", tasks)
        if self.config.relative_action_policy == "groot_processor":
            self._write_json(
                backend, "meta/relative_stats.json", self._relative_statistics()
            )

    def _relative_statistics(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        state_by_key = {item.key: item for item in self.config.state_modalities}
        horizon = self.config.action_horizon
        for action_group in self.config.action_modalities:
            if action_group.representation != "relative":
                continue
            state_group = state_by_key[action_group.state_key or action_group.key]
            trajectories = []
            for states, actions in self._episode_state_action:
                for start in range(max(0, len(actions) - horizon + 1)):
                    reference = states[start, state_group.start : state_group.end]
                    chunk = actions[
                        start : start + horizon, action_group.start : action_group.end
                    ]
                    trajectories.append(chunk - reference[None, :])
            if not trajectories:
                raise ConfigurationError(
                    f"no episode is long enough for relative action '{action_group.key}' "
                    f"with horizon {horizon}"
                )
            output[action_group.key] = _statistics(np.stack(trajectories))
        return output

    def _write_json(self, backend: Backend, path: str, value: Any) -> None:
        backend.write(path, canonical_json_bytes(value))
        self._payload_paths.append(path)

    def _write_jsonl(
        self, backend: Backend, path: str, values: list[dict[str, Any]]
    ) -> None:
        backend.write(path, b"".join(canonical_json_bytes(value) for value in values))
        self._payload_paths.append(path)

    def _require_active(self) -> _OpenEpisode:
        if self._active is None:
            raise ConfigurationError("begin_episode must be called first")
        return self._active


def validate_groot_lerobot_v2(backend: Backend) -> ConformanceResult:
    """Validate GR00T v2 structure and explicitly reject LeRobot v3."""
    required = (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/modality.json",
        "meta/stats.json",
    )
    serialization_errors = [
        f"missing {path}" for path in required if not backend.exists(path)
    ]
    errors: list[str] = []
    if serialization_errors:
        return ConformanceResult(
            "failed",
            "failed",
            messages=tuple(serialization_errors),
            evidence={
                "target": f"Isaac-GR00T {GROOT_RELEASE_TAG}",
                "commit": GROOT_RELEASE_COMMIT,
            },
        )
    try:
        info = json.loads(backend.read("meta/info.json"))
        modality = json.loads(backend.read("meta/modality.json"))
        stats = json.loads(backend.read("meta/stats.json"))
        episodes = _read_jsonl(backend.read("meta/episodes.jsonl"))
        tasks = _read_jsonl(backend.read("meta/tasks.jsonl"))
        if info.get("codebase_version") != "v2.1":
            errors.append("codebase_version must be v2.1; LeRobot v3 is incompatible")
        if info.get("profile") != "groot_lerobot_v2":
            errors.append("info.json does not declare the groot_lerobot_v2 profile")
        for key in ("chunks_size", "data_path", "video_path", "features"):
            if key not in info:
                errors.append(f"info.json is missing {key}")
        for feature_name in ("observation.state", "action"):
            feature = info.get("features", {}).get(feature_name, {})
            if feature.get("dtype") != "float32" or not feature.get("shape"):
                errors.append(
                    f"info.json has invalid {feature_name} float32 feature metadata"
                )
            names = feature.get("names")
            expected_name_count = int(np.prod(feature.get("shape", [])))
            if (
                not isinstance(names, list)
                or len(names) != expected_name_count
                or any(not isinstance(name, str) or not name for name in names)
            ):
                errors.append(
                    f"info.json {feature_name} names must be a flat, complete string list"
                )
        if info.get("features", {}).get("timestamp", {}).get("dtype") != "float32":
            errors.append("info.json timestamp feature must declare float32")
        for video_meta in modality.get("video", {}).values():
            feature = info.get("features", {}).get(video_meta.get("original_key"), {})
            video_info = feature.get("info", {})
            if (
                feature.get("dtype") != "video"
                or feature.get("names") != ["height", "width", "channels"]
                or any(
                    key not in video_info
                    for key in ("video.height", "video.width", "video.channels")
                )
            ):
                errors.append(
                    "info.json has incomplete canonical video feature metadata"
                )
        for key in ("state", "action", "video", "annotation", "ovannotators_config"):
            if not modality.get(key):
                errors.append(f"modality.json is missing explicit {key} configuration")
        if len(episodes) != info.get("total_episodes"):
            errors.append("episode count does not match info.json")
        if len(tasks) != info.get("total_tasks"):
            errors.append("task count does not match info.json")
        for feature in ("observation.state", "action"):
            if feature not in stats or any(
                name not in stats[feature] for name in _STAT_NAMES
            ):
                errors.append(f"stats.json is incomplete for {feature}")
        relative_keys = [
            key
            for key, value in modality.get("action", {}).items()
            if value.get("representation") == "RELATIVE"
        ]
        if relative_keys and not backend.exists("meta/relative_stats.json"):
            errors.append(
                "relative action configuration requires meta/relative_stats.json"
            )
        pa, pq = _require_pyarrow()
        seen_indexes: list[int] = []
        for episode in episodes:
            episode_index = episode["episode_index"]
            chunk = episode_index // info["chunks_size"]
            data_path = info["data_path"].format(
                episode_chunk=chunk, episode_index=episode_index
            )
            if not backend.exists(data_path):
                errors.append(f"missing episode parquet {data_path}")
                continue
            table = pq.read_table(io.BytesIO(backend.read(data_path)))
            required_columns = {
                "observation.state",
                "action",
                "timestamp",
                "timestamp_ns",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            }
            if not required_columns.issubset(table.column_names):
                errors.append(f"{data_path} is missing required columns")
                continue
            state_type = table.schema.field("observation.state").type
            action_type = table.schema.field("action").type
            state_dim = info["features"]["observation.state"]["shape"][0]
            action_dim = info["features"]["action"]["shape"][0]
            if not (
                pa.types.is_fixed_size_list(state_type)
                and state_type.list_size == state_dim
                and pa.types.is_float32(state_type.value_type)
            ):
                errors.append(
                    f"{data_path} observation.state must be fixed_size_list<float32>[{state_dim}]"
                )
            if not (
                pa.types.is_fixed_size_list(action_type)
                and action_type.list_size == action_dim
                and pa.types.is_float32(action_type.value_type)
            ):
                errors.append(
                    f"{data_path} action must be fixed_size_list<float32>[{action_dim}]"
                )
            if not pa.types.is_float32(table.schema.field("timestamp").type):
                errors.append(f"{data_path} timestamp must be float32")
            if table.num_rows != episode["length"]:
                errors.append(f"{data_path} length does not match episodes.jsonl")
            timestamps = table["timestamp_ns"].to_pylist()
            if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
                errors.append(f"{data_path} timestamps are not strictly increasing")
            if table["frame_index"].to_pylist() != list(range(table.num_rows)):
                errors.append(
                    f"{data_path} frame_index must be contiguous from zero within the episode"
                )
            fps = info.get("fps")
            if isinstance(fps, int) and not isinstance(fps, bool) and fps > 0:
                if any(
                    abs((stamp - timestamps[0]) - round(position * 1e9 / fps))
                    > 2 * _MAX_TIMESTAMP_TOLERANCE_NS
                    for position, stamp in enumerate(timestamps)
                ):
                    errors.append(
                        f"{data_path} timestamp_ns is off the constant-frame-rate "
                        f"timeline for {fps} fps"
                    )
                seconds = table["timestamp"].to_pylist()
                if any(
                    abs(value - position / fps)
                    > 2 * _MAX_TIMESTAMP_TOLERANCE_NS / 1e9
                    + float(np.spacing(np.float32(position / fps)))
                    for position, value in enumerate(seconds)
                ):
                    errors.append(
                        f"{data_path} timestamp seconds do not match frame_index / fps"
                    )
            else:
                errors.append("info.json fps must be a positive integer")
            seen_indexes.extend(table["index"].to_pylist())
            for video_key, video_meta in modality.get("video", {}).items():
                del video_key
                video_path = info["video_path"].format(
                    episode_chunk=chunk,
                    episode_index=episode_index,
                    video_key=video_meta["original_key"],
                )
                if not backend.exists(video_path):
                    errors.append(f"missing configured video {video_path}")
        if seen_indexes != list(range(len(seen_indexes))):
            errors.append("global frame indexes are not contiguous and deterministic")
        _validate_completion_artifacts(backend, serialization_errors, errors)
    except Exception as exc:
        serialization_errors.append(f"invalid GR00T serialization: {exc}")
    return ConformanceResult(
        "failed" if serialization_errors else "passed",
        "failed" if serialization_errors or errors else "passed",
        semantic="not_run",
        canonical_loader="not_run",
        consumer="not_run",
        messages=tuple((*serialization_errors, *errors)),
        evidence={
            "target": f"Isaac-GR00T {GROOT_RELEASE_TAG}",
            "commit": GROOT_RELEASE_COMMIT,
        },
    )


def _validate_completion_artifacts(
    backend: Backend, serialization_errors: list[str], errors: list[str]
) -> None:
    success_path = "_ovannotators/_SUCCESS"
    checksums_path = "_ovannotators/payload_checksums.json"
    present_artifacts = {
        path for path in PROFILE_ARTIFACT_PATHS if backend.exists(path)
    }
    if not present_artifacts:
        return
    missing_artifacts = sorted(PROFILE_ARTIFACT_PATHS - present_artifacts)
    if missing_artifacts:
        serialization_errors.append(
            "partial completion artifacts are invalid; missing "
            + ", ".join(missing_artifacts)
        )
        return
    success = json.loads(backend.read(success_path))
    if success.get("meaning") != "serialization_and_structural_validation_completed":
        errors.append("_SUCCESS has an invalid completion meaning")
    manifest = json.loads(backend.read(checksums_path))
    if manifest.get("algorithm") != "sha256" or not isinstance(
        manifest.get("files"), dict
    ):
        serialization_errors.append("payload_checksums.json is invalid")
        return
    checksums = manifest["files"]
    payload_paths = {
        path for path in backend.list() if not path.startswith("_ovannotators/")
    }
    if set(checksums) != payload_paths:
        serialization_errors.append(
            "payload checksum paths do not exactly match serialized payloads"
        )
    for path, digest in checksums.items():
        if not backend.exists(path):
            serialization_errors.append(f"checksummed payload is missing: {path}")
        elif sha256_bytes(backend.read(path)) != digest:
            serialization_errors.append(f"payload checksum mismatch: {path}")


def _validate_slice(
    key: str,
    start: int,
    end: int,
    feature_names: tuple[str, ...],
    units: tuple[str, ...],
) -> None:
    if not key or start < 0 or end <= start:
        raise ConfigurationError(f"invalid modality slice '{key}' [{start}, {end})")
    if end - start != len(feature_names) or len(units) != len(feature_names):
        raise ConfigurationError(
            f"modality slice '{key}' width, feature_names, and units must match"
        )
    _require_unique(f"feature names for {key}", list(feature_names))


def _validate_contiguous_slices(label: str, slices: tuple[Any, ...]) -> None:
    expected_start = 0
    for item in slices:
        if item.start != expected_start:
            raise ConfigurationError(
                f"{label} modality slices must be ordered and contiguous"
            )
        expected_start = item.end
    _require_unique(f"{label} modality keys", [item.key for item in slices])
    _require_unique(
        f"{label} feature names",
        [name for item in slices for name in item.feature_names],
    )


def _require_unique(label: str, values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ConfigurationError(f"{label} must be unique")


def _statistics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _read_jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _encode_video(frames: list[np.ndarray], fps: int) -> bytes:
    if not frames:
        raise ConfigurationError("cannot encode an empty video modality")
    from ovannotators._internal.codecs import get_video_encoder

    height, width = frames[0].shape[:2]
    encoder = get_video_encoder("", fps, width, height)
    for frame in frames:
        encoder.write_frame(frame)
    return encoder.close()


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for GrootLeRobotV2Writer; install ovannotators[lerobot]"
        ) from exc
    return pa, pq
