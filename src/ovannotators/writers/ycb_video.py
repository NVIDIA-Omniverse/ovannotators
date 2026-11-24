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

"""YCBVideoWriter — YCB-Video dataset writer."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ovannotators._internal.mat5 import encode_mat5
from ovannotators.exceptions import ConfigurationError
from ovannotators.writers._helpers import (
    _as_id,
    _label_for,
    _to_numpy,
    extract_camera_params,
    id_label_map_from_render_vars,
    iter_bboxes_2d,
    pose_in_opencv_camera,
    validate_required_vars,
)
from ovannotators.writers._object_pose import PoseCamera
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


@dataclass
class YCBVideoWriterConfig:
    """Configuration for :class:`YCBVideoWriter`.

    Attributes:
        class_name_to_index_map: Maps each semantic label to its YCB-Video class
            index, an integer from 1 to 255. Labels that are not in the map are
            treated as background: they get label 0 in ``-label.png`` and no
            pose or box.
        factor_depth: Depth scale: ``depth_m = stored / factor_depth``. The
            default, 10000, is the YCB-Video value and limits depth to 6.5535 m.
        video_id: Name of the video directory under ``data/YCB_Video/data/``.
    """

    class_name_to_index_map: dict[str, int] = field(default_factory=dict)
    factor_depth: int = 10000
    video_id: str = "0000"


_REQUIRED_VARS: frozenset[str] = frozenset(
    {
        "LdrColor",
        "DistanceToImagePlaneSD",
        "SemanticSegmentation",
        "SemanticIdMap",
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox3D",
        "CameraParams",
    }
)

_ROOT = "data/YCB_Video"

logger = logging.getLogger(__name__)


@WriterRegistry.register
class YCBVideoWriter:
    """Writes a dataset in the YCB-Video layout used by Isaac Sim's ``YCBVideoWriter``.

    Output structure::

        {output_dir}/data/
        ├── YCB_Video/
        │   ├── data/{video_id}/
        │   │   ├── {frame_id:06d}-color.png
        │   │   ├── {frame_id:06d}-depth.png   16-bit, depth_m * factor_depth
        │   │   ├── {frame_id:06d}-label.png   class index per pixel
        │   │   ├── {frame_id:06d}-box.txt     "class_name x1 y1 x2 y2" per object
        │   │   └── {frame_id:06d}-meta.mat    cls_indexes, factor_depth,
        │   │                                  intrinsic_matrix, poses, center
        │   └── train.txt                      "{video_id}/{frame_id:06d}" per frame
        └── models/{model_name}/points.xyz     from :meth:`write_model_points`

    ``poses`` is a ``3 x 4 x N`` array of object-to-camera ``[R | t]`` in the
    OpenCV camera frame (+Z forward, +Y down) with ``t`` in metres, converted
    from scene units with the camera's ``meters_per_scene_unit``. ``center`` holds
    each object's origin projected to pixels. A frame with no mapped object is
    not written.
    """

    name: ClassVar[str] = "ycb_video"
    required_render_vars: ClassVar[frozenset[str]] = _REQUIRED_VARS

    def __init__(self, config: YCBVideoWriterConfig) -> None:
        """Initialize the writer.

        Args:
            config (YCBVideoWriterConfig): Writer options; the class map is required.

        Raises:
            ConfigurationError: If the class map is empty or has an index outside
                1-255, or ``factor_depth`` is not in 1-65535.
        """
        self.config: YCBVideoWriterConfig = config
        indices = config.class_name_to_index_map
        if not indices:
            raise ConfigurationError("class_name_to_index_map must not be empty")
        bad = {name: i for name, i in indices.items() if not 1 <= i <= 255}
        if bad:
            raise ConfigurationError(f"class indices must be in 1-255, got {bad}")
        if not 1 <= config.factor_depth <= 65535:
            raise ConfigurationError("factor_depth must be in 1-65535")
        self._frames: list[int] = []
        self._finalized: bool = False
        self._warned_depth_clip: bool = False

    @property
    def _video_dir(self) -> str:
        return f"{_ROOT}/data/{self.config.video_id}"

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write one frame's images, boxes and metadata.

        Args:
            frame_id (int): Frame index; files are named by the zero-padded id.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars``.
            backend (Backend): Destination backend for the frame's files.

        Raises:
            MissingRenderVarError: If a required render var is missing.
            ConfigurationError: If the camera is not a pinhole camera.
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )
        PoseCamera.from_render_var(self.name, render_vars["CameraParams"])
        id_to_label = id_label_map_from_render_vars(render_vars)
        cls_indexes, poses, centers = self._objects(render_vars, id_to_label)
        if not cls_indexes:
            return

        stem = f"{self._video_dir}/{frame_id:06d}"
        backend.write_array(
            f"{stem}-color.png", render_vars["LdrColor"], encoding="png"
        )
        backend.write_array(
            f"{stem}-depth.png", self._depth(render_vars), encoding="png"
        )
        backend.write_array(
            f"{stem}-label.png",
            self._label_image(render_vars["SemanticSegmentation"], id_to_label),
            encoding="png",
        )
        backend.write(f"{stem}-box.txt", self._boxes(render_vars, id_to_label).encode())

        camera = extract_camera_params(render_vars["CameraParams"])
        meta = {
            "cls_indexes": np.asarray(cls_indexes, dtype=np.uint8),
            "factor_depth": np.array([self.config.factor_depth], dtype=np.uint16),
            "intrinsic_matrix": camera.K,
            "poses": np.stack(poses, axis=-1),
            "center": np.asarray(centers, dtype=np.float64),
        }
        backend.write(f"{stem}-meta.mat", encode_mat5(meta))
        self._frames.append(frame_id)

    def write_model_points(
        self, model_name: str, points: np.ndarray, backend: Backend
    ) -> None:
        """Write one model's vertices to ``data/models/{model_name}/points.xyz``.

        ovannotators does not read USD, so the caller supplies the points: the
        model's vertices in metres, in the same frame as the object origin the
        poses refer to.

        Args:
            model_name (str): YCB-Video model name, for example
                ``"002_master_chef_can"``. It need not match the scene's name.
            points (np.ndarray): ``(N, 3)`` vertex positions.
            backend (Backend): Destination backend.

        Raises:
            ValueError: If ``points`` is not ``(N, 3)`` or ``model_name`` is not a
                plain name.
        """
        if (
            not model_name
            or "/" in model_name
            or "\\" in model_name
            or model_name in {".", ".."}
        ):
            raise ValueError(f"model_name must be a plain name, got {model_name!r}")
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        text = io.StringIO()
        np.savetxt(text, points, fmt="%.6f", delimiter=" ", newline="\n")
        backend.write(f"data/models/{model_name}/points.xyz", text.getvalue().encode())

    def finalize(self, backend: Backend) -> None:
        """Write ``train.txt`` listing every written frame.

        Args:
            backend (Backend): Destination backend for the output file.
        """
        if self._finalized:
            return
        self._finalized = True
        lines = "".join(
            f"{self.config.video_id}/{frame_id:06d}\n"
            for frame_id in sorted(self._frames)
        )
        backend.write(f"{_ROOT}/train.txt", lines.encode())

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _class_index(self, label: str) -> int | None:
        """Return the YCB class index for ``label``, or ``None`` for background."""
        return self.config.class_name_to_index_map.get(label.split(",")[0])

    def _objects(
        self, render_vars: dict[str, Any], id_to_label: dict[int, str] | None
    ) -> tuple[list[int], list[np.ndarray], list[list[float]]]:
        """Return class indexes, ``3 x 4`` camera-frame poses and pixel centres."""
        raw_camera = _to_numpy(render_vars["CameraParams"]).reshape(-1)[0]
        metres = float(raw_camera["meters_per_scene_unit"]) or 1.0
        camera = extract_camera_params(render_vars["CameraParams"])
        fx, fy = camera.K[0, 0], camera.K[1, 1]
        cx, cy = camera.K[0, 2], camera.K[1, 2]

        cls_indexes: list[int] = []
        poses: list[np.ndarray] = []
        centers: list[list[float]] = []
        for box in _to_numpy(render_vars["SemanticBoundingBox3D"]):
            semantic_id = _as_id(box["semantic_id"])
            index = self._class_index(_label_for(semantic_id, id_to_label))
            if index is None:
                continue
            local_to_world = np.asarray(box["transform"], dtype=np.float64).reshape(
                4, 4
            )
            scales = np.linalg.norm(local_to_world[:3, :3], axis=1)
            if not np.all(scales > 0.0):
                continue
            rotation = (local_to_world[:3, :3] / scales[:, None]).T
            rotation_cam, origin_cam = pose_in_opencv_camera(
                camera, rotation, local_to_world[3, :3]
            )
            if origin_cam[2] <= 0.0:  # behind the camera
                continue
            cls_indexes.append(index)
            poses.append(np.hstack([rotation_cam, (origin_cam * metres)[:, None]]))
            centers.append(
                [
                    fx * origin_cam[0] / origin_cam[2] + cx,
                    fy * origin_cam[1] / origin_cam[2] + cy,
                ]
            )
        return cls_indexes, poses, centers

    def _depth(self, render_vars: dict[str, Any]) -> np.ndarray:
        """Return ``DistanceToImagePlaneSD`` (metres) as 16-bit ``depth * factor_depth``."""
        depth = _to_numpy(render_vars["DistanceToImagePlaneSD"]).astype(np.float64)
        depth = depth.reshape(depth.shape[:2])
        scaled = np.where(np.isfinite(depth), depth * self.config.factor_depth, 0.0)
        if not self._warned_depth_clip and np.any(scaled > 65535):
            self._warned_depth_clip = True
            logger.warning(
                "Depth beyond %.4f m is clipped to 65535 at factor_depth=%d; "
                "lower factor_depth to cover the scene's depth range.",
                65535 / self.config.factor_depth,
                self.config.factor_depth,
            )
        return np.clip(np.rint(scaled), 0, 65535).astype(np.uint16)

    def _label_image(
        self, segmentation: Any, id_to_label: dict[int, str] | None
    ) -> np.ndarray:
        """Map semantic ids to class indexes; unmapped labels become 0."""
        ids = _to_numpy(segmentation)
        ids = ids.reshape(ids.shape[:2]).astype(np.int64)
        lookup = np.zeros(int(ids.max()) + 1 if ids.size else 1, dtype=np.uint8)
        for semantic_id in id_to_label or {}:
            index = self._class_index(_label_for(semantic_id, id_to_label))
            if index is not None and semantic_id < len(lookup):
                lookup[semantic_id] = index
        return lookup[ids]

    def _boxes(
        self, render_vars: dict[str, Any], id_to_label: dict[int, str] | None
    ) -> str:
        """Return ``-box.txt`` content: ``class_name x1 y1 x2 y2`` per mapped object."""
        lines = []
        for box in iter_bboxes_2d(
            render_vars["SemanticBoundingBox2DTight"], id_to_label
        ):
            name = box.label.split(",")[0]
            if self._class_index(name) is None:
                continue
            x_min, y_min = int(box.x_min), int(box.y_min)
            x_max, y_max = int(box.x_max), int(box.y_max)
            lines.append(f"{name} {x_min} {y_min} {x_max} {y_max}\n")
        return "".join(lines)
