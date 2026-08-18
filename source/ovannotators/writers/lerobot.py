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

import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ovannotators.writers._helpers import _to_numpy, validate_required_vars
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
        fps: Frames per second of the episode video.
        episode_index: Episode index written to every parquet row and used in
            file names.
        task_description: Human-readable task string stored in
            ``meta/tasks.json``.
        camera_keys: Observation camera identifiers.  Each key produces a
            ``videos/…/observation.images.{key}/episode_XXXXXX.mp4`` file and
            a corresponding column in the parquet.
        frames_per_chunk: Number of frames per parquet chunk file.
    """

    fps: int = 30
    episode_index: int = 0
    task_description: str = ""
    camera_keys: list[str] = field(default_factory=lambda: ["top"])
    frames_per_chunk: int = 1000


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

    Requires ``PyNvVideoCodec`` for GPU-accelerated video encoding.
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
        # Per-frame raw RGB images, keyed by camera key.
        self._frame_images: dict[str, list[np.ndarray]] = {
            k: [] for k in self.config.camera_keys
        }
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
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )

        ldr = render_vars["LdrColor"]

        # Normalise to (H, W, 3) uint8 RGB
        img = _to_rgb(ldr)
        if self._image_shape is None and img.ndim == 3:
            self._image_shape = (img.shape[0], img.shape[1], img.shape[2])

        for key in self.config.camera_keys:
            self._frame_images[key].append(img)

        self._rows.append(
            {
                "index": self._global_index,
                "episode_index": self.config.episode_index,
                "frame_index": self._frame_index,
                "timestamp": self._frame_index / max(1, self.config.fps),
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
        self._finalized = True

        if not self._rows:
            return

        cfg = self.config
        ep_idx = cfg.episode_index
        total_frames = len(self._rows)
        chunk_size = max(1, cfg.frames_per_chunk)
        n_chunks = (total_frames + chunk_size - 1) // chunk_size
        h, w, c = self._image_shape or (0, 0, 3)

        # --- Videos ---
        for key in cfg.camera_keys:
            frames = self._frame_images[key]
            for chunk_id in range(n_chunks):
                chunk_frames = frames[
                    chunk_id * chunk_size : (chunk_id + 1) * chunk_size
                ]
                video_dir = f"videos/chunk-{chunk_id:03d}/observation.images.{key}"
                video_path = f"{video_dir}/episode_{ep_idx:06d}.mp4"
                backend.makedirs(video_dir)
                mp4_bytes = _encode_video(chunk_frames, fps=cfg.fps)
                backend.write(video_path, mp4_bytes)

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
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert an image array (NumPy or Warp) to (H, W, 3) uint8 RGB."""
    arr = _to_numpy(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _encode_video(
    frames: list[np.ndarray],
    fps: int,
) -> bytes:
    """Encode a list of (H, W, 3) uint8 RGB frames to MP4 bytes via PyNvVideoCodec."""
    if not frames:
        return b""
    from ovannotators._internal.codecs import get_video_encoder

    h, w = frames[0].shape[:2]
    encoder = get_video_encoder("", fps, w, h)
    for frame in frames:
        encoder.write_frame(frame)
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
