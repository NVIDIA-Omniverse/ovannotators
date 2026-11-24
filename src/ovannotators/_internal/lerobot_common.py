# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for native LeRobot v2, GR00T v2.1, LeRobot v3, and Lance writers."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from bisect import bisect_right
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from ovannotators._internal.video_config import VideoCodec, VideoPreset

DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.10, 0.50, 0.90, 0.99)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
DEFAULT_EPISODES_PATH = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)
DEFAULT_TASKS_PATH = "meta/tasks.parquet"

DEFAULT_V3_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

VideoEncodeFn = Callable[..., bytes]


# ---------------------------------------------------------------------------
# Lazy dependency imports
# ---------------------------------------------------------------------------


def require_pyarrow(writer_name: str = "LeRobot writer") -> tuple[Any, Any]:
    """Return ``(pyarrow, pyarrow.parquet)`` or raise a clear ``ImportError``."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            f"pyarrow is required for {writer_name}; install ovannotators[lerobot]"
        ) from exc
    return pa, pq


def symlinks_outside_root(root: Path) -> list[str]:
    """Report links under the dataset that escape its resolved root."""
    resolved_root = root.resolve()
    messages = []
    for directory, subdirectories, files in os.walk(root):
        for name in sorted(subdirectories + files):
            path = Path(directory, name)
            try:
                if not path.is_symlink() and not getattr(
                    path.lstat(), "st_reparse_tag", 0
                ):
                    continue
                if name in subdirectories:
                    subdirectories.remove(name)
                inside = path.resolve().is_relative_to(resolved_root)
            except (OSError, RuntimeError):
                inside = False
            if not inside:
                messages.append(
                    f"symlink resolves outside the dataset: {path.relative_to(root).as_posix()}"
                )
    return messages


_PASSIVE_HF_TYPES = frozenset(
    {
        "Value",
        "List",
        "LargeList",
        "Sequence",
        "ClassLabel",
        *(f"Array{n}D" for n in range(2, 6)),
    }
)


@contextlib.contextmanager
def open_native_parquet(path: Path) -> Iterator[Any]:
    """Open a self-contained Parquet footer with only passive reader features."""
    _, pq = require_pyarrow()
    with pq.ParquetFile(path) as parquet:
        metadata = parquet.metadata
        for index in range(metadata.num_row_groups):
            group = metadata.row_group(index)
            for column in range(group.num_columns):
                if group.column(column).file_path:
                    raise ValueError("Parquet column chunk references an external file")

        for source in (metadata.metadata, parquet.schema_arrow.metadata):
            if not source or b"huggingface" not in source:
                continue
            hf = json.loads(source[b"huggingface"])
            if not isinstance(hf, dict) or not isinstance(hf.get("info"), dict):
                raise ValueError("invalid Hugging Face Parquet metadata")
            features = hf["info"].get("features", {})
            if not isinstance(features, dict):
                raise ValueError("invalid Hugging Face Parquet features")
            pending = [features]
            while pending:
                feature = pending.pop()
                if isinstance(feature, dict):
                    kind = feature.get("_type")
                    if isinstance(kind, str) and kind not in _PASSIVE_HF_TYPES:
                        raise ValueError(
                            f"unsupported Hugging Face Parquet feature: {kind!r}"
                        )
                    pending.extend(feature.values())
                elif isinstance(feature, list):
                    pending.extend(feature)
        yield parquet


def require_pylance(writer_name: str = "LeRobotLanceWriter") -> Any:
    """Return the ``lance`` module (``pylance``) or raise a clear ``ImportError``."""
    try:
        import lance
    except ImportError as exc:
        raise ImportError(
            f"pylance is required for {writer_name}; "
            "install ovannotators[cosmos-multiview] or pip install pylance==11.0.0"
        ) from exc
    return lance


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------


def encode_video_frames(
    frames: Sequence[Any],
    fps: float,
    *,
    codec: VideoCodec = "h264",
    lossless: bool = False,
    preset: VideoPreset | None = None,
    gop: int | None = None,
) -> bytes:
    """Encode a sequence of ``(H, W, 3)`` uint8 RGB frames to MP4 via NVENC."""
    if not frames:
        return b""
    from ovannotators._internal.codecs import get_video_encoder

    h, w = frames[0].shape[:2]
    encoder = get_video_encoder(
        "", fps, w, h, codec=codec, lossless=lossless, preset=preset, gop=gop
    )
    try:
        for frame in frames:
            encoder.write_frame(frame)
    except BaseException:
        with contextlib.suppress(Exception):
            encoder.close()
        raise
    return encoder.close()


# ---------------------------------------------------------------------------
# Camera naming and mixed-rate alignment
# ---------------------------------------------------------------------------


def camera_feature_key(stream_id: str) -> str:
    """Normalize a camera stream ID to ``observation.images.<name>``."""
    return f"observation.images.{stream_id.removeprefix('camera.').replace('/', '.')}"


def camera_timestamp_key(stream_id: str) -> str:
    """Normalize a camera stream ID to its authoritative source-timestamp column."""
    normalized = stream_id.removeprefix("camera.").replace("/", ".")
    return f"source.camera.{normalized}.timestamp_ns"


def validate_feature_keys(
    camera_stream_ids: Sequence[str],
    state_stream_id: str,
    action_stream_id: str,
    *,
    lance: bool = False,
) -> None:
    """Reject feature names that collide with generated keys or storage paths."""
    video_keys = [camera_feature_key(stream_id) for stream_id in camera_stream_ids]
    tabular_keys = [
        *DEFAULT_V3_FEATURES,
        "task",
        "source.timestamp_ns",
        "source.frame_id",
        *(camera_timestamp_key(stream_id) for stream_id in camera_stream_ids),
        state_stream_id,
        action_stream_id,
    ]
    keys = [*video_keys, *tabular_keys]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "state, action, camera, and generated output keys must be unique"
        )
    if any("/" in key for key in (state_stream_id, action_stream_id)):
        raise ValueError("state and action output keys must not contain '/'")
    # LeRobot treats any statistics key containing "image" as a (3, 1, 1)
    # image feature when it aggregates or resumes a dataset.
    image_cameras = sorted(
        stream_id
        for stream_id in camera_stream_ids
        if "image" in camera_timestamp_key(stream_id)
    )
    if image_cameras:
        raise ValueError(
            f"camera stream IDs must not contain 'image': {image_cameras} would "
            "produce a source-timestamp column that LeRobot reads as an image "
            "feature; rename the camera stream"
        )
    image_like = sorted(key for key in tabular_keys if "image" in key)
    if image_like:
        raise ValueError(f"tabular output keys must not contain 'image': {image_like}")
    if lance and len({to_lance_column(key) for key in tabular_keys}) != len(
        tabular_keys
    ):
        raise ValueError("normalized Lance column names must be unique")
    if len({key.casefold() for key in video_keys}) != len(video_keys):
        raise ValueError(
            "camera video paths must be unique on case-insensitive filesystems"
        )
    for key in video_keys:
        if key.endswith((".", " ")) or re.search(r'[<>:"/\\|?*\x00-\x1f]', key):
            raise ValueError(
                f"camera output key is not a portable path component: {key!r}"
            )


def camera_source_stride(dataset_fps: int, source_rate_hz: float) -> int | None:
    """Return the integer frame stride if ``source_rate_hz`` divides ``dataset_fps``."""
    ratio = Fraction(dataset_fps, 1) / Fraction(str(source_rate_hz))
    return ratio.numerator if ratio.denominator == 1 else None


def resample_camera_zoh(
    frames: Sequence[np.ndarray],
    times: Sequence[int],
    targets: Sequence[int],
    *,
    source_rate_hz: float,
    tolerance_ns: int,
    stream_id: str,
) -> tuple[list[np.ndarray], list[int]]:
    """Zero-order-hold align a camera stream onto target CFR timestamps."""
    if not frames:
        raise ValueError(f"camera {stream_id!r} has no source frames")
    max_hold = round(1_000_000_000 / source_rate_hz) + tolerance_ns
    aligned: list[np.ndarray] = []
    source_times: list[int] = []
    for target in targets:
        index = bisect_right(times, target) - 1
        if index < 0 or target - times[index] > max_hold:
            raise ValueError(
                f"camera {stream_id!r} has no source frame within its declared cadence"
            )
        aligned.append(frames[index])
        source_times.append(times[index])
    return aligned, source_times


def stats_dtype(key: str, dtype: str) -> np.dtype:
    """Return the dtype LeRobot uses when computing statistics for ``key``.

    LeRobot buffers ``timestamp`` as Python floats, so its statistics are
    computed in float64 although the column is stored as float32.
    """
    return np.dtype(np.float64) if key == "timestamp" else np.dtype(dtype)


def update_chunk_file_indices(
    chunk_idx: int, file_idx: int, chunks_size: int
) -> tuple[int, int]:
    """Advance ``(chunk_idx, file_idx)`` within ``chunks_size`` files per chunk."""
    if file_idx == chunks_size - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


# ---------------------------------------------------------------------------
# LeRobot v3.0 statistics (matching lerobot==0.7.0 compute_stats.py)
# ---------------------------------------------------------------------------


class _RunningQuantileStats:
    """Histogram-based running statistics matching ``lerobot==0.7.0``."""

    def __init__(
        self,
        quantile_list: Sequence[float] = DEFAULT_QUANTILES,
        num_quantile_bins: int = 5000,
    ) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = num_quantile_bins
        self._quantile_list = list(quantile_list)
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, batch.shape[-1]).astype(np.float64, copy=False)
        num_elements, vector_length = batch.shape
        batch_mean = np.mean(batch, axis=0)
        batch_sq = np.mean(batch**2, axis=0)
        self._count += num_elements
        if self._mean is None:
            self._mean = batch_mean
            self._mean_of_squares = batch_sq
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            assert self._min is not None and self._max is not None
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = bool(np.any(new_max > self._max))
            min_changed = bool(np.any(new_min < self._min))
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

            assert self._mean_of_squares is not None
            weight = num_elements / self._count
            self._mean += (batch_mean - self._mean) * weight
            self._mean_of_squares += (batch_sq - self._mean_of_squares) * weight
        assert self._histograms is not None and self._bin_edges is not None
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _adjust_histograms(self) -> None:
        assert (
            self._histograms is not None
            and self._bin_edges is not None
            and self._min is not None
            and self._max is not None
        )
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding,
                self._max[i] + padding,
                self._num_quantile_bins + 1,
            )
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)
            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    bin_idx = int(np.searchsorted(new_edges, old_center)) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def get_statistics(self) -> dict[str, np.ndarray]:
        assert (
            self._mean is not None
            and self._mean_of_squares is not None
            and self._min is not None
            and self._max is not None
            and self._histograms is not None
            and self._bin_edges is not None
        )
        variance = self._mean_of_squares - self._mean**2
        stats: dict[str, np.ndarray] = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": np.sqrt(np.maximum(0, variance)),
            "count": np.array([self._count]),
        }
        for q, q_key in zip(self._quantile_list, self._quantile_keys, strict=True):
            target = q * self._count
            values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = int(np.searchsorted(cumsum, target))
                if idx == 0:
                    values.append(float(edges[0]))
                elif idx >= len(cumsum):
                    values.append(float(edges[-1]))
                else:
                    before = cumsum[idx - 1]
                    in_bin = cumsum[idx] - before
                    if in_bin == 0:
                        values.append(float(edges[idx]))
                    else:
                        frac = (target - before) / in_bin
                        values.append(
                            float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))
                        )
            stats[q_key] = np.array(values)
        return stats


def _sample_indices(data_len: int) -> list[int]:
    min_samples = min(100, data_len)
    num_samples = max(min_samples, min(int(data_len**0.75), 10_000))
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def _auto_downsample_chw(
    img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300
) -> np.ndarray:
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::factor, ::factor]


def get_feature_stats(
    array: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    quantile_list: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, np.ndarray]:
    """Compute min, max, mean, std, count, and quantiles matching ``lerobot==0.7.0``."""
    original_shape = array.shape
    if axis == (0, 2, 3):
        batch_size, channels, _, _ = array.shape
        reshaped = array.transpose(0, 2, 3, 1).reshape(-1, channels)
        sample_count = batch_size
    elif axis in (0, (0,)):
        reshaped = array.reshape(-1, 1) if array.ndim == 1 else array
        sample_count = array.shape[0]
    elif axis == (1,):
        reshaped = array.T
        sample_count = array.shape[1]
    elif axis is None:
        reshaped = array.reshape(-1, 1)
        sample_count = array.shape[0] if array.ndim > 0 else 1
    else:
        raise ValueError(f"Unsupported axis configuration: {axis}")

    if reshaped.shape[0] < 2:
        stats: dict[str, np.ndarray] = {
            "min": np.min(reshaped, axis=0),
            "max": np.max(reshaped, axis=0),
            "mean": np.mean(reshaped, axis=0),
            "std": np.std(reshaped, axis=0),
            "count": np.array([sample_count]),
        }
        for q in quantile_list:
            stats[f"q{int(q * 100):02d}"] = stats["mean"].copy()
    else:
        running = _RunningQuantileStats(quantile_list=quantile_list)
        running.update(reshaped)
        stats = running.get_statistics()
        stats["count"] = np.array([sample_count])

    result: dict[str, np.ndarray] = {}
    for key, value in stats.items():
        if key == "count":
            result[key] = value
        elif axis == (0, 2, 3):
            result[key] = (
                value.reshape(1, -1, 1, 1) if (keepdims and value.ndim == 1) else value
            )
        elif axis in (0, (0,)):
            if not keepdims:
                result[key] = value
            elif len(original_shape) == 1 and value.ndim > 0:
                result[key] = value.reshape(1)
            elif len(original_shape) >= 2 and value.ndim == 1:
                result[key] = value.reshape(1, -1)
            else:
                result[key] = value
        elif axis == (1,):
            if not keepdims:
                result[key] = value
            elif value.ndim == 0:
                result[key] = value.reshape(1, 1)
            elif value.ndim == 1:
                result[key] = value.reshape(-1, 1)
            else:
                result[key] = value
        elif axis is None:
            result[key] = (
                value.reshape(tuple(1 for _ in original_shape))
                if keepdims
                else np.atleast_1d(value)
            )
    return result


def compute_episode_stats(
    vector_data: Mapping[str, np.ndarray],
    camera_frames: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute LeRobot v3.0 per-episode statistics for tabular and video features."""
    ep_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, arr in vector_data.items():
        data = np.asarray(arr)
        ep_stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)
    for key, frames in camera_frames.items():
        indices = _sample_indices(len(frames))
        sampled = np.stack(
            [
                _auto_downsample_chw(
                    np.asarray(frames[idx], dtype=np.uint8).transpose(2, 0, 1)
                )
                for idx in indices
            ],
            axis=0,
        )
        raw = get_feature_stats(sampled, axis=(0, 2, 3), keepdims=True)
        raw["count"] = np.array([len(frames)])
        ep_stats[key] = {
            stat_name: val if stat_name == "count" else np.squeeze(val / 255.0, axis=0)
            for stat_name, val in raw.items()
        }
    return ep_stats


def aggregate_stats(
    stats_list: Sequence[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate per-episode stats matching ``lerobot==0.7.0`` ``aggregate_stats``."""
    data_keys = {key for stats in stats_list for key in stats}
    aggregated: dict[str, dict[str, np.ndarray]] = {}
    for key in data_keys:
        ft_list = [stats[key] for stats in stats_list if key in stats]
        means = np.stack([s["mean"] for s in ft_list])
        variances = np.stack([s["std"] ** 2 for s in ft_list])
        counts = np.stack([s["count"] for s in ft_list])
        total_count = counts.sum(axis=0)
        expanded_counts = counts
        while expanded_counts.ndim < means.ndim:
            expanded_counts = np.expand_dims(expanded_counts, axis=-1)
        total_mean = (means * expanded_counts).sum(axis=0) / total_count
        delta_means = means - total_mean
        total_var = ((variances + delta_means**2) * expanded_counts).sum(
            axis=0
        ) / total_count
        out: dict[str, np.ndarray] = {
            "min": np.min(np.stack([s["min"] for s in ft_list]), axis=0),
            "max": np.max(np.stack([s["max"] for s in ft_list]), axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_var),
            "count": total_count,
        }
        q_keys = [k for k in ft_list[0] if k.startswith("q") and k[1:].isdigit()]
        for q_key in q_keys:
            if all(q_key in s for s in ft_list):
                q_vals = np.stack([s[q_key] for s in ft_list])
                out[q_key] = (
                    np.min(q_vals, axis=0)
                    if int(q_key[1:]) <= 50
                    else np.max(q_vals, axis=0)
                )
        aggregated[key] = out
    return aggregated


def flatten_dict(
    d: Mapping[str, Any], parent_key: str = "", sep: str = "/"
) -> dict[str, Any]:
    """Flatten nested mappings using ``sep`` (matches ``lerobot.utils.utils.flatten_dict``)."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def serialize_stats(stats: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    """Convert numpy statistics dictionaries to JSON-serializable nested lists."""
    out: dict[str, Any] = {}
    for feature_key, ft_stats in stats.items():
        out[feature_key] = {
            stat_key: val.tolist() if isinstance(val, np.ndarray) else val
            for stat_key, val in ft_stats.items()
        }
    return out


# ---------------------------------------------------------------------------
# LeRobot v3.0 metadata & Parquet writers (zero lerobot / pandas dependency)
# ---------------------------------------------------------------------------


def build_v3_video_info(
    *,
    height: int,
    width: int,
    fps: int,
    codec: VideoCodec,
    lossless: bool = False,
    gop: int | None = None,
    preset: VideoPreset | None = None,
) -> dict[str, Any]:
    """Build the canonical LeRobot v3.0 ``features[key]['info']`` block."""
    pix_fmt = "gbrp" if lossless else "yuv420p"
    info: dict[str, Any] = {
        "video.height": height,
        "video.width": width,
        "video.codec": codec,
        "video.pix_fmt": pix_fmt,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
        "is_depth_map": False,
    }
    if gop is not None:
        info["video.g"] = gop
    if preset is not None:
        info["video.preset"] = preset
    return info


def build_v3_info(
    *,
    codebase_version: str,
    fps: int,
    features: Mapping[str, Mapping[str, Any]],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    robot_type: str | None = None,
    chunks_size: int = DEFAULT_CHUNK_SIZE,
    data_files_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_files_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    data_path: str = DEFAULT_DATA_PATH,
    video_path: str | None = DEFAULT_VIDEO_PATH,
    storage_format: str | None = None,
) -> dict[str, Any]:
    """Construct the canonical LeRobot v3.0 ``meta/info.json`` mapping."""
    serialized_features: dict[str, Any] = {}
    for key, ft in features.items():
        copy_ft = dict(ft)
        if isinstance(copy_ft.get("shape"), tuple):
            copy_ft["shape"] = list(copy_ft["shape"])
        serialized_features[key] = copy_ft
    info: dict[str, Any] = {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": chunks_size,
        "data_files_size_in_mb": data_files_size_in_mb,
        "video_files_size_in_mb": video_files_size_in_mb,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"} if total_episodes > 0 else {},
        "data_path": data_path,
        "video_path": video_path,
        "features": serialized_features,
    }
    if storage_format is not None:
        info["storage_format"] = storage_format
    return info


def write_tasks_parquet(tasks: Sequence[str], path: Path) -> None:
    """Write ``meta/tasks.parquet`` compatible with ``pandas.read_parquet`` in LeRobot.

    Stores ``task_index`` (int64) indexed by ``task`` (string) with pandas
    Parquet schema metadata so ``lerobot.datasets.io_utils.load_tasks`` recovers
    ``tasks.index.name == 'task'`` without requiring ``pandas`` at write time.
    """
    pa, pq = require_pyarrow("LeRobot tasks writer")
    path.parent.mkdir(parents=True, exist_ok=True)
    pandas_meta = {
        "index_columns": ["task"],
        "column_indexes": [
            {
                "name": None,
                "field_name": None,
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": {"encoding": "UTF-8"},
            }
        ],
        "columns": [
            {
                "name": "task_index",
                "field_name": "task_index",
                "pandas_type": "int64",
                "numpy_type": "int64",
                "metadata": None,
            },
            {
                "name": "task",
                "field_name": "task",
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": None,
            },
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    schema = pa.schema(
        [
            pa.field("task_index", pa.int64()),
            pa.field("task", pa.string()),
        ],
        metadata={b"pandas": json.dumps(pandas_meta).encode("utf-8")},
    )
    table = pa.table(
        {
            "task_index": pa.array(list(range(len(tasks))), type=pa.int64()),
            "task": pa.array(list(tasks), type=pa.string()),
        },
        schema=schema,
    )
    pq.write_table(table, str(path), compression="snappy")


def arrow_type_for_feature(pa: Any, dtype_str: str, shape: Sequence[int]) -> Any:
    """Build the PyArrow type for a tabular LeRobot v3 feature."""
    base = pa.from_numpy_dtype(np.dtype(dtype_str))
    shape_tuple = tuple(shape)
    if shape_tuple == (1,):
        return base
    if len(shape_tuple) == 1:
        # Matches ``datasets`` ``Sequence(length=N)``, which LeRobot declares.
        return pa.list_(base, shape_tuple[0])
    arrow_t = base
    for _ in shape_tuple:
        arrow_t = pa.list_(arrow_t)
    return arrow_t


def stream_schema(
    episode: Any, stream_ids: Collection[str] | None = None
) -> tuple[tuple[Any, ...], ...]:
    """Return the episode's stream descriptors as a comparable, order-free key.

    Profile writers compare this key across episodes so that a dataset never
    mixes stream layouts. ``dtype``, ``shape``, and ``units`` are normalized, so
    equivalent spellings such as ``"u1"`` and ``"uint8"`` or a list and a tuple
    compare equal.

    Args:
        episode: Episode envelope whose ``streams`` are described.
        stream_ids: Streams to include. ``None`` includes every stream.
    """
    return tuple(
        sorted(
            (
                item.stream_id,
                item.stream_type,
                np.dtype(item.dtype).str,
                tuple(item.shape),
                tuple(item.units) if isinstance(item.units, list) else item.units,
                item.nominal_rate_hz,
                item.required,
            )
            for item in episode.streams
            if stream_ids is None or item.stream_id in stream_ids
        )
    )


def build_v3_data_table(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Build a PyArrow Table for one LeRobot v3 episode or shard."""
    pa, _ = require_pyarrow("LeRobot data writer")
    fields = []
    columns: dict[str, Any] = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        shape = tuple(ft["shape"])
        arrow_t = arrow_type_for_feature(pa, ft["dtype"], shape)
        fields.append(pa.field(key, arrow_t))
        if shape == (1,):
            col_vals = [
                np.asarray(row[key]).reshape(-1)[0].item()
                if isinstance(row[key], np.ndarray)
                else row[key]
                for row in rows
            ]
        else:
            col_vals = [
                np.asarray(row[key], dtype=np.dtype(ft["dtype"])).tolist()
                for row in rows
            ]
        columns[key] = pa.array(col_vals, type=arrow_t)
    schema = pa.schema(fields)
    return pa.table(columns, schema=schema)


def build_v3_episodes_table(episode_records: Sequence[Mapping[str, Any]]) -> Any:
    """Build a PyArrow Table for ``meta/episodes/chunk-XXX/file-YYY.parquet``."""
    pa, _ = require_pyarrow("LeRobot episode metadata writer")
    combined: dict[str, list[Any]] = {}
    for record in episode_records:
        for key, value in record.items():
            if key not in combined:
                combined[key] = []
            combined[key].append(
                value.tolist() if isinstance(value, np.ndarray) else value
            )
    return pa.Table.from_pydict(combined)


def to_lance_column(key: str) -> str:
    """Map a dotted LeRobot feature key to its Lance column name."""
    return key.replace(".", "_")


_LOADER_RESULT_MARKER = "OVANNOTATORS_CANONICAL_RESULT "


OFFLINE_LOADER_TIMEOUT_S = 1800.0


def run_offline_loader(
    script: str, *args: str, timeout_s: float = OFFLINE_LOADER_TIMEOUT_S
) -> dict[str, str]:
    """Run a canonical loader script in a child with Hub access disabled.

    The Hub and datasets libraries read offline flags at import time, so a
    fresh interpreter is required. ``-P`` keeps its working directory off
    ``sys.path``. The script returns a JSON outcome on a marked stdout line.
    A child still running after ``timeout_s`` seconds is killed and reported
    as an error outcome, so a hung loader cannot block finalization.
    """
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-P", "-c", script, *args],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"loader process timed out after {timeout_s:g} s"}
    except OSError as exc:
        return {"error": f"could not launch loader process: {str(exc)[:1000]}"}
    stderr = proc.stderr.strip().rsplit("\n", 1)[-1]
    if proc.returncode != 0:
        return {
            "error": f"loader process exited {proc.returncode}: {stderr[-1000:] or 'no output'}"
        }
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_LOADER_RESULT_MARKER):
            try:
                result = json.loads(line[len(_LOADER_RESULT_MARKER) :])
            except (ValueError, RecursionError):
                return {"error": "loader process returned invalid JSON"}
            if isinstance(result, dict) and result.keys() in ({"version"}, {"error"}):
                key, value = next(iter(result.items()))
                if isinstance(value, str) and value.strip():
                    return {key: value[:1000]}
            return {"error": "loader process returned an invalid result"}
    return {
        "error": f"loader process exited 0 without a result: {stderr[-1000:] or 'no output'}"
    }
