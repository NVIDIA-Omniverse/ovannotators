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

"""LeRobotWriter — HuggingFace LeRobot dataset format writer."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ovannotators._internal.video_config import (
    VideoCodec,
    VideoPreset,
    validate_video_options,
    video_frame_rate,
)

from ovannotators.writers._helpers import validate_required_vars
from ovannotators._internal.video_frames import snapshot_rgb
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LeRobotWriterConfig:
    """Configuration for :class:`LeRobotWriter`.

    Attributes:
        fps: Frames per second of the episode video; fractional rates such
            as 29.97 are supported.
        episode_index: Episode index written to every parquet row and used in
            file names.
        task_description: Human-readable task string stored in
            ``meta/tasks.json``.
        camera_keys: Observation camera identifiers.  Each key produces a
            ``videos/…/observation.images.{key}/episode_XXXXXX.mp4`` file and
            a corresponding column in the parquet.
        frames_per_chunk: Number of frames per parquet chunk file.
        video_codec: ``"h264"`` (default) or ``"hevc"``.
        video_lossless: Preserve normalized uint8 RGB with HEVC 4:4:4.
            Requires even, fixed dimensions and a compatible video reader.
        video_preset: Optional NVENC P1 through P7; lossless defaults to P5.
    """

    fps: float = 30
    episode_index: int = 0
    task_description: str = ""
    camera_keys: list[str] = field(default_factory=lambda: ["top"])
    frames_per_chunk: int = 1000
    video_codec: VideoCodec = "h264"
    video_lossless: bool = False
    video_preset: VideoPreset | None = None

    def __post_init__(self) -> None:
        """Validate video encoding settings."""
        validate_video_options(self.video_codec, self.video_lossless, self.video_preset)
        video_frame_rate(self.fps, "fps")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_REQUIRED_VARS: frozenset[str] = frozenset({"LdrColor"})


@WriterRegistry.register
class LeRobotWriter:
    """Writes dataset in HuggingFace LeRobot format.

    Output structure::

        {output_dir}/
        ├── data/
        │   └── chunk-{chunk:03d}/
        │       └── episode_{episode_id:06d}.parquet
        ├── videos/
        │   └── chunk-{chunk:03d}/
        │       └── observation.images.{camera_key}/
        │           └── episode_{episode_id:06d}.mp4
        └── meta/
            ├── info.json
            ├── tasks.json
            ├── episodes.json
            └── stats.json

    Uses ``PyNvVideoCodec`` for GPU-accelerated video encoding.
    Requires ``pyarrow`` for parquet output.

    Per-frame parquet row schema:

    * ``index`` (int64) — global monotonic frame counter
    * ``episode_index`` (int64)
    * ``frame_index`` (int64) — counter within this episode
    * ``timestamp`` (float64) — frame_index / fps
    * ``task_index`` (int64) — always 0 for single-task datasets
    """

    name: ClassVar[str] = "lerobot"
    required_render_vars: ClassVar[frozenset[str]] = _REQUIRED_VARS

    def __init__(self, config: LeRobotWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (LeRobotWriterConfig | None): Writer options; defaults are
                used when ``None``.
        """
        self.config: LeRobotWriterConfig = config or LeRobotWriterConfig()
        # Per-frame RGB snapshots (NumPy or device arrays). Every camera key
        # records the same LdrColor stream, so one list serves them all.
        self._frames: list[Any] = []
        # Parquet rows (one per frame, all non-image scalar columns)
        self._rows: list[dict[str, Any]] = []
        self._global_index: int = 0
        self._frame_index: int = 0
        self._finalized: bool = False
        # Resolved image shape (filled on first frame)
        self._image_shape: tuple[int, int, int] | None = None

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Accumulate one frame; data is flushed to disk in :meth:`finalize`.

        Args:
            frame_id (int): Zero-based frame index within the episode.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars``.
            backend (Backend): Destination backend (used at finalize time).

        Raises:
            MissingRenderVarError: If a required render var is missing.
            ValueError: If lossless video frames have non-positive or odd
                dimensions, or change resolution.
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )

        # Without camera keys no video is written, so skip the RGB snapshot.
        if self.config.camera_keys:
            # Normalise to (H, W, 3) uint8 RGB
            img = _to_rgb(render_vars["LdrColor"])
            if self.config.video_lossless:
                if self._image_shape is None:
                    h0, w0 = img.shape[:2]
                    if h0 <= 0 or w0 <= 0 or h0 % 2 or w0 % 2:
                        raise ValueError(
                            "Lossless HEVC requires positive, even dimensions; "
                            "frames are never cropped"
                        )
                elif tuple(img.shape) != self._image_shape:
                    raise ValueError("Lossless video requires a fixed frame resolution")
            if self._image_shape is None and img.ndim == 3:
                self._image_shape = (img.shape[0], img.shape[1], img.shape[2])
            self._frames.append(img)

        self._rows.append(
            {
                "index": self._global_index,
                "episode_index": self.config.episode_index,
                "frame_index": self._frame_index,
                "timestamp": self._frame_index / self.config.fps,
                "task_index": 0,
            }
        )
        self._global_index += 1
        self._frame_index += 1

    def finalize(self, backend: Backend) -> None:
        """Encode videos and write the parquet chunks and meta JSON files.

        Args:
            backend (Backend): Destination backend for the dataset outputs.
        """
        if self._finalized:
            return

        if not self._rows:
            self._finalized = True
            return

        cfg = self.config
        ep_idx = cfg.episode_index
        total_frames = len(self._rows)
        chunk_size = max(1, cfg.frames_per_chunk)
        n_chunks = (total_frames + chunk_size - 1) // chunk_size
        h, w, c = self._image_shape or (0, 0, 3)

        # --- Videos ---
        # Camera keys share one stream, so encode each chunk once and store the
        # same MP4 under every key. No keys means no videos and no encoding.
        for chunk_id in range(n_chunks if cfg.camera_keys else 0):
            chunk_frames = self._frames[
                chunk_id * chunk_size : (chunk_id + 1) * chunk_size
            ]
            mp4_bytes = _encode_video(
                chunk_frames,
                fps=cfg.fps,
                codec=cfg.video_codec,
                lossless=cfg.video_lossless,
                preset=cfg.video_preset,
            )
            for key in cfg.camera_keys:
                video_dir = f"videos/chunk-{chunk_id:03d}/observation.images.{key}"
                backend.makedirs(video_dir)
                backend.write(f"{video_dir}/episode_{ep_idx:06d}.mp4", mp4_bytes)

        # --- Parquet chunks ---
        pa, pq = _require_pyarrow()
        for chunk_id in range(n_chunks):
            chunk_rows = self._rows[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
            table = pa.Table.from_pylist(chunk_rows)
            buf = io.BytesIO()
            pq.write_table(table, buf)
            data_dir = f"data/chunk-{chunk_id:03d}"
            backend.makedirs(data_dir)
            backend.write(f"{data_dir}/episode_{ep_idx:06d}.parquet", buf.getvalue())

        # --- Meta JSON files ---
        backend.makedirs("meta")

        features: dict[str, Any] = {
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        for key in cfg.camera_keys:
            features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": [h, w, c],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": cfg.fps,
                    "video.codec": cfg.video_codec,
                    "video.pix_fmt": "gbrp" if cfg.video_lossless else "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        info: dict[str, Any] = {
            "codebase_version": "v2.0",
            "fps": cfg.fps,
            "dataset_type": "LeRobotDataset",
            "robot_type": "synthetic",
            "features": features,
            "total_episodes": 1,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": n_chunks * len(cfg.camera_keys),
            "chunks_shape": [n_chunks, chunk_size],
        }
        tasks = [{"task_index": 0, "task": cfg.task_description}]
        episodes = [{"episode_index": ep_idx, "tasks": [0], "length": total_frames}]
        stats: dict[str, Any] = {}

        def _w(name: str, obj: Any) -> None:
            backend.write(f"meta/{name}", json.dumps(obj, indent=2).encode())

        _w("info.json", info)
        _w("tasks.json", tasks)
        _w("episodes.json", episodes)
        _w("stats.json", stats)
        self._frames.clear()
        self._finalized = True


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_rgb(arr: Any) -> Any:
    """Convert an image array (NumPy or Warp) to (H, W, 3) uint8 RGB."""
    return snapshot_rgb(arr)


def _encode_video(
    frames: list[Any],
    fps: float,
    *,
    codec: VideoCodec = "h264",
    lossless: bool = False,
    preset: VideoPreset | None = None,
) -> bytes:
    """Encode a list of (H, W, 3) uint8 RGB frames to MP4 bytes via PyNvVideoCodec."""
    if not frames:
        return b""
    from ovannotators._internal.codecs import get_video_encoder

    h, w = frames[0].shape[:2]
    encoder = get_video_encoder(
        "", fps, w, h, codec=codec, lossless=lossless, preset=preset
    )
    try:
        for frame in frames:
            encoder.write_frame(frame)
    except BaseException:
        # Release the NVENC session now; the original error takes precedence.
        with contextlib.suppress(Exception):
            encoder.close()
        raise
    return encoder.close()


def _require_pyarrow() -> tuple[Any, Any]:
    """Return (pyarrow, pyarrow.parquet) or raise ImportError."""
    try:
        import pyarrow as pa  # type: ignore[import]
        import pyarrow.parquet as pq  # type: ignore[import]

        return pa, pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for LeRobotWriter. "
            "Install it with: pip install pyarrow"
        ) from exc
