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

"""KITTIWriter — KITTI object-detection dataset writer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ovannotators.types import BBox3D
from ovannotators.writers._helpers import (
    _to_numpy,
    extract_camera_params,
    id_label_map_from_render_vars,
    iter_bboxes_2d,
    iter_bboxes_3d,
    validate_required_vars,
)
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


def _sanitize_kitti_type(label: str) -> str:
    """Sanitize a semantic label for use as the KITTI 'type' token.

    KITTI labels are whitespace-delimited, so the type field must be a single
    token with no spaces.  This strips common prefixes (``class:``, ``type:``),
    trailing punctuation, collapses remaining whitespace to underscores, and
    falls back to ``Unknown`` if the result is empty.
    """
    import re

    # Strip common semantic-schema prefixes: "class: X", "type: X"
    cleaned = re.sub(r"^(class|type)\s*:\s*", "", label, flags=re.IGNORECASE)
    # Strip trailing semicolons and whitespace
    cleaned = cleaned.rstrip("; \t")
    # Replace any remaining whitespace with underscores
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned or "Unknown"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KITTIWriterConfig:
    """Configuration for :class:`KITTIWriter`.

    Attributes:
        classes: If set, only objects whose label is in this list are written.
            ``None`` writes all objects.
        write_depth: Write a 16-bit depth PNG alongside each image.
        depth_scale: Multiplier converting depth in metres to stored uint16
            values (``depth_uint16 = round(depth_m * depth_scale)``).
            KITTI convention: 256.0 (1 LSB ≈ 3.9 mm at 1 m).
    """

    classes: list[str] | None = None
    write_depth: bool = False
    depth_scale: float = 256.0


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_REQUIRED_BASE: frozenset[str] = frozenset(
    {
        "LdrColor",
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox3D",
        "CameraParams",
    }
)


@WriterRegistry.register
class KITTIWriter:
    """Writes dataset in KITTI object-detection format.

    Output structure::

        {output_dir}/
        ├── image_2/   {frame_id:06d}.png
        ├── label_2/   {frame_id:06d}.txt
        ├── calib/     {frame_id:06d}.txt
        └── depth/     {frame_id:06d}.png   (only when write_depth=True)

    ``label_2`` row format::

        type truncated occluded alpha left top right bottom height width length x y z rotation_y

    All dimensions and positions are in the camera coordinate frame.
    ``rotation_y`` is the rotation about the camera Y-axis (radians).

    The ``SemanticBoundingBox3D`` render var must be a numpy structured array
    following :data:`OVRTX_BBOX3D_DTYPE`; class labels are resolved from the
    optional ``SemanticIdMap`` render var.
    """

    name: ClassVar[str] = "kitti"
    required_render_vars: ClassVar[frozenset[str]] = _REQUIRED_BASE

    def __init__(self, config: KITTIWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (KITTIWriterConfig | None): Writer options; defaults are used
                when ``None``.
        """
        self.config: KITTIWriterConfig = config or KITTIWriterConfig()
        if self.config.write_depth:
            self.required_render_vars = _REQUIRED_BASE | frozenset(
                {"DistanceToCameraSD"}
            )  # type: ignore[assignment]
        self._finalized: bool = False

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write image, label, calibration, and optionally depth for one frame.

        Args:
            frame_id (int): Zero-based frame index; KITTI files are named
                ``{frame_id:06d}``.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars``.
            backend (Backend): Destination backend for the frame's files.

        Raises:
            MissingRenderVarError: If a required render var is missing.
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )

        stem = f"{frame_id:06d}"

        # --- Image ---
        backend.makedirs("image_2")
        backend.write_array(
            f"image_2/{stem}.png", render_vars["LdrColor"], encoding="png"
        )

        # --- Calibration ---
        cam = extract_camera_params(render_vars["CameraParams"])
        calib_txt = _build_calib(cam.K)
        backend.makedirs("calib")
        backend.write(f"calib/{stem}.txt", calib_txt.encode())

        # --- Labels ---
        id_to_label = id_label_map_from_render_vars(render_vars)
        bboxes_2d = list(
            iter_bboxes_2d(render_vars["SemanticBoundingBox2DTight"], id_to_label)
        )
        bboxes_3d = list(
            iter_bboxes_3d(render_vars["SemanticBoundingBox3D"], id_to_label)
        )

        # Match 3D boxes to 2D boxes by instance_id
        bbox_3d_by_iid: dict[int, BBox3D] = {b.instance_id: b for b in bboxes_3d}

        lines: list[str] = []
        classes_filter = set(self.config.classes) if self.config.classes else None
        for bb2 in bboxes_2d:
            label = bb2.label
            if classes_filter and label not in classes_filter:
                continue

            iid = bb2.instance_id
            bb3 = bbox_3d_by_iid.get(iid)

            # ovrtx SemanticBoundingBox2D* payloads carry no truncation/occlusion
            # information, so KITTI's truncated/occluded fields default to 0.
            truncated = 0.0
            occluded = 0

            left = bb2.x_min
            top = bb2.y_min
            right = bb2.x_max
            bottom = bb2.y_max

            if bb3 is not None:
                center = bb3.center
                extent = bb3.extent
                rot = bb3.rotation
                h_3d = float(2.0 * extent[1])
                w_3d = float(2.0 * extent[0])
                l_3d = float(2.0 * extent[2])
                # KITTI: location is bottom-centre (y += half-height, Y-down)
                x_3d = float(center[0])
                y_3d = float(center[1] + extent[1])
                z_3d = float(center[2])
                # rotation about Y-axis from rotation matrix
                rotation_y = float(math.atan2(-rot[2, 0], rot[2, 2]))
                alpha = rotation_y - math.atan2(x_3d, z_3d)
                alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            else:
                h_3d = w_3d = l_3d = -1.0
                x_3d = y_3d = z_3d = -1000.0
                rotation_y = alpha = -10.0

            line = (
                f"{_sanitize_kitti_type(label)} {truncated:.2f} {occluded} {alpha:.4f} "
                f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
                f"{h_3d:.4f} {w_3d:.4f} {l_3d:.4f} "
                f"{x_3d:.4f} {y_3d:.4f} {z_3d:.4f} {rotation_y:.4f}"
            )
            lines.append(line)

        backend.makedirs("label_2")
        label_txt = "\n".join(lines) + ("\n" if lines else "")
        backend.write(f"label_2/{stem}.txt", label_txt.encode())

        # --- Depth (optional) ---
        if self.config.write_depth and "DistanceToCameraSD" in render_vars:
            depth_m = _to_numpy(render_vars["DistanceToCameraSD"]).astype(np.float32)
            depth_u16 = (
                (depth_m * self.config.depth_scale).clip(0, 65535).astype(np.uint16)
            )
            backend.makedirs("depth")
            backend.write_array(f"depth/{stem}.png", depth_u16, encoding="png")

    def finalize(self, backend: Backend) -> None:
        """No-op; KITTIWriter writes all data eagerly in :meth:`write_frame`.

        Args:
            backend (Backend): Unused; present for :class:`Writer` compatibility.
        """
        self._finalized = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_calib(K: np.ndarray) -> str:
    """Build a KITTI calibration text file from an intrinsic matrix K."""
    # P2 = K extended to 3×4 with zero translation column
    P = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
    p_str = " ".join(f"{v:.6e}" for v in P.ravel())

    # Identity calibration matrices for synthetic data
    r0 = np.eye(3, dtype=np.float64)
    r0_vals = " ".join(f"{v:.6e}" for v in r0.ravel())
    tr = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    tr_vals = " ".join(f"{v:.6e}" for v in tr.ravel())

    lines = [
        f"P0: {p_str}",
        f"P1: {p_str}",
        f"P2: {p_str}",
        f"P3: {p_str}",
        f"R0_rect: {r0_vals}",
        f"Tr_velo_to_cam: {tr_vals}",
        f"Tr_imu_to_velo: {tr_vals}",
    ]
    return "\n".join(lines) + "\n"
