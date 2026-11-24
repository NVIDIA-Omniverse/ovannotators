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

"""Shared object-pose geometry and frame loop for the DOPE and CenterPose writers.

Both formats write, per frame, an image plus a JSON of per-object camera-frame
poses and projected 3D cuboid keypoints. They differ only in field names,
keypoint order and a few extra fields, so the geometry, visibility handling and
file layout live here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ovannotators.exceptions import ConfigurationError
from ovannotators.writers._helpers import (
    _as_id,
    _label_for,
    _to_numpy,
    extract_camera_params,
    id_label_map_from_render_vars,
    id_prim_path_map_from_render_vars,
    occlusion_by_stable_id,
    validate_required_vars,
)

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend

# Keypoint orders. In an object's local frame L/R are -X/+X, D/U are -Y/+Y and
# B/F are -Z/+Z of its 3D bounding box.
KEYPOINT_ORDER_DOPE: tuple[str, ...] = (
    "LUF",
    "RUF",
    "RDF",
    "LDF",
    "LUB",
    "RUB",
    "RDB",
    "LDB",
    "Center",
)
KEYPOINT_ORDER_CENTERPOSE: tuple[str, ...] = (
    "Center",
    "LDB",
    "LDF",
    "LUB",
    "LUF",
    "RDB",
    "RDF",
    "RUB",
    "RUF",
)


@dataclass(frozen=True)
class PoseCamera:
    """Row-vector view and projection matrices plus image resolution."""

    view: np.ndarray
    projection: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_render_var(cls, writer_name: str, arr: Any) -> PoseCamera:
        """Read a pinhole camera from a ``CameraParams`` render var.

        Args:
            writer_name: Writer name, used in the error message.
            arr: ``CameraParams`` render var.

        Raises:
            ConfigurationError: If the projection model is not pinhole.
        """
        arr = _to_numpy(arr)
        record = arr.reshape(-1)[0]
        model = bytes(record["projection_model"]).split(b"\0", 1)[0]
        if model not in (b"", b"pinhole"):
            raise ConfigurationError(
                f"{writer_name} supports pinhole cameras only, got {model.decode()!r}"
            )
        params = extract_camera_params(arr)
        return cls(
            view=np.asarray(record["view_transform"], dtype=np.float64).reshape(4, 4),
            projection=np.asarray(record["projection"], dtype=np.float64).reshape(4, 4),
            width=params.resolution.width,
            height=params.resolution.height,
            fx=float(params.K[0, 0]),
            fy=float(params.K[1, 1]),
            cx=float(params.K[0, 2]),
            cy=float(params.K[1, 2]),
        )

    def camera_data(self, *, with_size: bool = False) -> dict[str, Any]:
        """Return the ``camera_data`` block, optionally with ``width``/``height``."""
        data: dict[str, Any] = {
            "intrinsics": {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy},
            "camera_view_matrix": np.round(self.view, 5).tolist(),
            "camera_projection_matrix": np.round(self.projection, 5).tolist(),
        }
        if with_size:
            data["width"] = self.width
            data["height"] = self.height
        return data

    def project(self, point_camera: np.ndarray) -> tuple[int, int] | None:
        """Project a homogeneous camera-frame point to pixels, or ``None`` if behind."""
        clip = point_camera @ self.projection
        if clip[3] <= 1e-10:
            return None
        ndc = clip[:2] / clip[3]
        return (
            round((ndc[0] * 0.5 + 0.5) * self.width),
            round((-ndc[1] * 0.5 + 0.5) * self.height),
        )


@dataclass(frozen=True)
class ObjectPose:
    """One object's pose and projected cuboid, in the USD camera frame.

    Attributes:
        label: Class label.
        prim_path: USD prim path, or ``None``.
        visibility: ``1 - occlusion_ratio``.
        location: Object origin in the camera frame.
        quaternion_xyzw: Object-to-camera rotation, unit length, ``w >= 0``.
        scale: Local box size times the object's world scale.
        keypoints_camera: Camera-frame keypoints in the requested order.
        pixels: Integer pixel keypoints in the requested order.
        truncation_ratio: Fraction of the keypoints' box outside the image.
    """

    label: str
    prim_path: str | None
    visibility: float
    location: list[float]
    quaternion_xyzw: list[float]
    scale: list[float]
    keypoints_camera: list[list[float]]
    pixels: list[tuple[int, int]]
    truncation_ratio: float


def check_pose_config(config: Any) -> None:
    """Validate the options shared by the pose writers' configs.

    Raises:
        ConfigurationError: If ``image_format`` is not ``"png"`` or ``"jpg"``.
    """
    if config.image_format not in ("png", "jpg"):
        raise ConfigurationError(
            f"image_format must be 'png' or 'jpg', got {config.image_format!r}"
        )


def write_pose_frame(
    writer: Any,
    frame_id: int,
    render_vars: dict[str, Any],
    backend: Backend,
    *,
    keypoint_order: tuple[str, ...],
    annotate: Callable[[ObjectPose], dict[str, Any]],
    with_size: bool = False,
) -> None:
    """Write one frame's RGB image and pose-annotation JSON.

    Args:
        writer: Writer with ``name``, ``required_render_vars`` and a ``config``
            holding ``visibility_threshold``, ``skip_empty_frames``,
            ``frame_padding`` and ``image_format``.
        frame_id: Frame index; files are named by the zero-padded id.
        render_vars: Frame render vars.
        backend: Destination backend.
        keypoint_order: Keypoint names, in output order.
        annotate: Turns an :class:`ObjectPose` into the format's object entry.
        with_size: Whether ``camera_data`` includes ``width`` and ``height``.

    Raises:
        MissingRenderVarError: If a required render var is missing.
        ConfigurationError: If the camera is not a pinhole camera.
    """
    config = writer.config
    validate_required_vars(
        writer.name, writer.required_render_vars, render_vars, frame_id
    )
    camera = PoseCamera.from_render_var(writer.name, render_vars["CameraParams"])

    id_to_label = id_label_map_from_render_vars(render_vars)
    id_to_prim_path = id_prim_path_map_from_render_vars(render_vars)
    occlusion = occlusion_by_stable_id(render_vars) or {}

    objects: list[dict[str, Any]] = []
    for box in _to_numpy(render_vars["SemanticBoundingBox3D"]):
        ratio = occlusion.get(tuple(int(w) for w in box["instance_id"]))
        visibility = 1.0 if ratio is None else 1.0 - min(abs(ratio), 1.0)
        if visibility <= config.visibility_threshold:
            continue
        pose = object_pose(
            box,
            camera,
            keypoint_order,
            label=_label_for(_as_id(box["semantic_id"]), id_to_label),
            prim_path=(id_to_prim_path or {}).get(_as_id(box["instance_id"])),
            visibility=visibility,
        )
        if pose is not None:
            objects.append(annotate(pose))

    if config.skip_empty_frames and not objects:
        return

    stem = f"{frame_id:0{config.frame_padding}d}"
    annotation = {
        "camera_data": camera.camera_data(with_size=with_size),
        "keypoint_order": list(keypoint_order),
        "objects": objects,
    }
    backend.write_array(
        f"{stem}.{config.image_format}",
        render_vars["LdrColor"],
        encoding=config.image_format,
    )
    backend.write(f"{stem}.json", json.dumps(annotation, indent=2).encode())


def object_pose(
    box: Any,
    camera: PoseCamera,
    keypoint_order: tuple[str, ...],
    *,
    label: str,
    prim_path: str | None,
    visibility: float,
) -> ObjectPose | None:
    """Compute one object's pose and cuboid, or ``None`` when it cannot be projected."""
    local_to_world = np.asarray(box["transform"], dtype=np.float64).reshape(4, 4)
    # Both matrices multiply row vectors, so the object-to-camera transform is
    # the local-to-world transform followed by the view matrix.
    local_to_camera = local_to_world @ camera.view
    world_scale = np.linalg.norm(local_to_world[:3, :3], axis=1)
    if not np.all(world_scale > 0.0):
        return None
    scales = np.linalg.norm(local_to_camera[:3, :3], axis=1)
    rotation = (local_to_camera[:3, :3] / scales[:, None]).T
    # A mirrored transform (a negative scale on an odd number of axes) has no
    # rotation: its normalized block has determinant -1. Skip the object rather
    # than write a quaternion for an orientation it does not have.
    if np.linalg.det(rotation) < 0.0:
        return None

    lo = np.array([box["x_min"], box["y_min"], box["z_min"]], dtype=np.float64)
    hi = np.array([box["x_max"], box["y_max"], box["z_max"]], dtype=np.float64)
    keypoints_camera: list[list[float]] = []
    pixels: list[tuple[int, int]] = []
    for point in _keypoints_local(lo, hi, keypoint_order):
        point_camera = np.append(point, 1.0) @ local_to_camera
        pixel = camera.project(point_camera)
        if pixel is None:
            return None
        keypoints_camera.append(point_camera[:3].tolist())
        pixels.append(pixel)

    return ObjectPose(
        label=label,
        prim_path=prim_path,
        visibility=visibility,
        location=local_to_camera[3, :3].tolist(),
        quaternion_xyzw=quaternion_xyzw(rotation),
        scale=(np.abs(hi - lo) * world_scale).tolist(),
        keypoints_camera=keypoints_camera,
        pixels=pixels,
        truncation_ratio=truncation_ratio(pixels, camera.width, camera.height),
    )


def _keypoints_local(
    lo: np.ndarray, hi: np.ndarray, order: tuple[str, ...]
) -> list[np.ndarray]:
    """Return the box keypoints in ``order`` (local frame)."""
    corners = {
        "LDB": (lo[0], lo[1], lo[2]),
        "LDF": (lo[0], lo[1], hi[2]),
        "LUB": (lo[0], hi[1], lo[2]),
        "LUF": (lo[0], hi[1], hi[2]),
        "RDB": (hi[0], lo[1], lo[2]),
        "RDF": (hi[0], lo[1], hi[2]),
        "RUB": (hi[0], hi[1], lo[2]),
        "RUF": (hi[0], hi[1], hi[2]),
        "Center": tuple((lo + hi) / 2.0),
    }
    return [np.array(corners[name], dtype=np.float64) for name in order]


def quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a rotation matrix to an ``[x, y, z, w]`` quaternion with ``w >= 0``."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    quat = np.array([x, y, z, w])
    quat /= np.linalg.norm(quat)
    if quat[3] < 0.0:
        quat = -quat
    return quat.tolist()


def truncation_ratio(pixels: list[tuple[int, int]], width: int, height: int) -> float:
    """Return the fraction of the keypoints' bounding box outside the image."""
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if area <= 0:
        return 1.0
    clipped_w = min(max(xs), width) - max(min(xs), 0)
    clipped_h = min(max(ys), height) - max(min(ys), 0)
    clipped = max(clipped_w, 0) * max(clipped_h, 0)
    return 1.0 - clipped / area
