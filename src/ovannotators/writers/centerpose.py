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

"""CenterPoseWriter — CenterPose category-level object-pose dataset writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from ovannotators.writers._object_pose import (
    KEYPOINT_ORDER_CENTERPOSE,
    ObjectPose,
    check_pose_config,
    write_pose_frame,
)
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


@dataclass
class CenterPoseWriterConfig:
    """Configuration for :class:`CenterPoseWriter`.

    Attributes:
        visibility_threshold: Objects with ``visibility`` at or below this value
            are skipped. ``visibility`` is ``1 - occlusion_ratio``, so the
            default ``0.0`` drops fully occluded objects.
        skip_empty_frames: When ``True``, a frame with no written objects
            produces no image or annotation file.
        frame_padding: Zero-padding width of the frame number in file names.
        image_format: Encoding of the RGB image, ``"png"`` or ``"jpg"``.
    """

    visibility_threshold: float = 0.0
    skip_empty_frames: bool = True
    frame_padding: int = 6
    image_format: Literal["png", "jpg"] = "png"


_REQUIRED_VARS: frozenset[str] = frozenset(
    {"LdrColor", "SemanticBoundingBox3D", "CameraParams"}
)


@WriterRegistry.register
class CenterPoseWriter:
    """Writes the CenterPose layout used by Isaac Sim's ``PoseWriter``.

    Output structure::

        {output_dir}/
        ├── {frame_id:06d}.png    RGB image (or .jpg)
        └── {frame_id:06d}.json   camera_data, keypoint_order, objects

    Each object has ``label``, ``prim_path``, ``visibility``, ``location`` (the
    object origin in the camera frame), ``quaternion_xyzw`` (object to camera
    rotation, ``w >= 0``), ``scale`` (local box size times the object's world
    scale), ``keypoints_3d`` (camera-frame cuboid keypoints), ``projected_cuboid``
    (nine integer pixel keypoints) and ``truncation_ratio``. Keypoints follow
    ``keypoint_order``, centre first. ``camera_data`` includes the image
    ``width`` and ``height``. The camera frame follows USD (looks down -Z, +Y
    up), as in Isaac Sim.

    ``visibility`` is ``1 - occlusion_ratio`` from ``OcclusionSD``. It needs
    ``OcclusionSD``, ``InstanceIdTokenMapSD``, ``InstanceMap`` and
    ``StableIdSemanticIdMap``; when any is missing, or an object has no valid
    ratio, ``visibility`` is ``1.0``.

    Only pinhole cameras are supported. An object with any keypoint behind the
    camera plane is skipped.
    """

    name: ClassVar[str] = "centerpose"
    required_render_vars: ClassVar[frozenset[str]] = _REQUIRED_VARS

    def __init__(self, config: CenterPoseWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (CenterPoseWriterConfig | None): Writer options; defaults are
                used when ``None``.

        Raises:
            ConfigurationError: If ``image_format`` is not ``"png"`` or ``"jpg"``.
        """
        self.config: CenterPoseWriterConfig = config or CenterPoseWriterConfig()
        check_pose_config(self.config)
        self._finalized: bool = False

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write one frame's RGB image and CenterPose annotation JSON.

        Args:
            frame_id (int): Frame index; files are named by the zero-padded id.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars``.
            backend (Backend): Destination backend for the frame's files.

        Raises:
            MissingRenderVarError: If a required render var is missing.
            ConfigurationError: If the camera is not a pinhole camera.
        """
        write_pose_frame(
            self,
            frame_id,
            render_vars,
            backend,
            keypoint_order=KEYPOINT_ORDER_CENTERPOSE,
            annotate=_annotate,
            with_size=True,
        )

    def finalize(self, backend: Backend) -> None:
        """Finish the dataset; every file is written per frame, so this is a no-op.

        Args:
            backend (Backend): Destination backend (unused).
        """
        self._finalized = True


def _annotate(pose: ObjectPose) -> dict[str, Any]:
    return {
        "label": pose.label,
        "prim_path": pose.prim_path,
        "visibility": round(pose.visibility, 3),
        "location": pose.location,
        "quaternion_xyzw": pose.quaternion_xyzw,
        "scale": pose.scale,
        "keypoints_3d": pose.keypoints_camera,
        "projected_cuboid": [list(p) for p in pose.pixels],
        "truncation_ratio": pose.truncation_ratio,
    }
