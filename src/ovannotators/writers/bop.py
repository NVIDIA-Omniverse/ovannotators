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

"""BOPWriter — BOP (Benchmark for 6D Object Pose Estimation) dataset writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ovannotators._internal.codecs import get_image_codec
from ovannotators.writers._helpers import (
    _to_numpy,
    extract_camera_params,
    id_label_map_from_render_vars,
    instance_masks,
    iter_bboxes_2d,
    iter_bboxes_3d,
    pose_in_opencv_camera,
    rotation_and_scale,
    validate_required_vars,
    world_to_opencv_camera,
)
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BOPWriterConfig:
    """Configuration for :class:`BOPWriter`.

    Attributes:
        scene_id: BOP scene identifier; output is written under
            ``{scene_id:06d}/``.
        depth_scale: mm per stored depth unit in the 16-bit PNG.
            ``depth_mm = depth_stored * depth_scale``.  Default 1.0 stores
            depth directly in millimetres (1 mm precision, range ~65.5 m),
            matching the common BOP convention. Increase it for scenes deeper
            than ~65 m (e.g. 10.0 gives 10 mm precision and ~655 m range).
        object_id_map: Maps object label strings to integer BOP object IDs.
            ``None`` uses ``instance_id`` from the render var directly.
    """

    scene_id: int = 0
    depth_scale: float = 1.0
    object_id_map: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

MASK_LAYOUT_FILE = "mask_layout.json"
"""Scene file that records how mask filename suffixes are assigned."""

MASK_LAYOUT: dict[str, str] = {"mask_suffix": "gt_index"}
"""Content of :data:`MASK_LAYOUT_FILE`: suffixes are ``scene_gt`` list indices."""

_REQUIRED_VARS: frozenset[str] = frozenset(
    {
        "LdrColor",
        "DistanceToCameraSD",
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox3D",
        "SemanticInstanceSegmentation",
        "CameraParams",
    }
)


@WriterRegistry.register
class BOPWriter:
    """Writes dataset in BOP (6D object pose) format.

    Output structure::

        {output_dir}/{scene_id:06d}/
        ├── rgb/              {frame_id:06d}.png
        ├── depth/            {frame_id:06d}.png  (16-bit, depth_scale mm/unit)
        ├── mask/             {frame_id:06d}_{gt_index:06d}.png
        ├── mask_visib/       {frame_id:06d}_{gt_index:06d}.png
        ├── scene_camera.json
        ├── scene_gt.json
        ├── scene_gt_info.json
        └── mask_layout.json  {"mask_suffix": "gt_index"}

    ``scene_camera.json``, ``scene_gt.json``, and ``scene_gt_info.json`` are
    accumulated in memory and written during :meth:`finalize`, followed by
    ``mask_layout.json``. Its presence distinguishes ground-truth-index mask
    suffixes from the model-ID suffixes that ovannotators 0.2 wrote.

    The ``SemanticBoundingBox3D`` render var provides object poses:

    * ``cam_R_m2c`` = the 3-D bbox ``rotation`` expressed in the OpenCV camera
      frame (+Z forward, +Y down), model-to-camera.
    * ``cam_t_m2c`` = the 3-D bbox centre in the OpenCV camera frame, in mm.
      The bbox centre stands in for the model origin.
    """

    name: ClassVar[str] = "bop"
    required_render_vars: ClassVar[frozenset[str]] = _REQUIRED_VARS

    def __init__(self, config: BOPWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (BOPWriterConfig | None): Writer options; defaults are used
                when ``None``.
        """
        self.config: BOPWriterConfig = config or BOPWriterConfig()
        self._scene_camera: dict[str, Any] = {}
        self._scene_gt: dict[str, Any] = {}
        self._scene_gt_info: dict[str, Any] = {}
        self._finalized: bool = False
        self._finalization_payloads: dict[str, bytes] | None = None
        self._finalized_paths: set[str] = set()

    @property
    def _scene_dir(self) -> str:
        return f"{self.config.scene_id:06d}"

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write per-frame images/masks and accumulate camera and GT metadata.

        Args:
            frame_id (int): Zero-based frame index; BOP files are named
                ``{frame_id:06d}`` within the scene directory.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars``.
            backend (Backend): Destination backend for the frame's files.

        Raises:
            MissingRenderVarError: If a required render var is missing.
            ValueError: If finalization has started or the scene-unit scale is
                not finite and positive.
        """
        if self._finalization_payloads is not None:
            raise ValueError("cannot write frames after finalization has started")
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )
        raw_camera = _to_numpy(render_vars["CameraParams"]).reshape(-1)[0]
        mm_per_scene_unit = float(raw_camera["meters_per_scene_unit"]) * 1000.0
        if not np.isfinite(mm_per_scene_unit) or mm_per_scene_unit <= 0:
            raise ValueError("meters_per_scene_unit must be finite and positive")

        sd = self._scene_dir
        stem = f"{frame_id:06d}"

        # --- RGB ---
        backend.makedirs(f"{sd}/rgb")
        backend.write_array(
            f"{sd}/rgb/{stem}.png", render_vars["LdrColor"], encoding="png"
        )

        # --- Depth (16-bit PNG, depth_scale mm/unit) ---
        depth_m = _to_numpy(render_vars["DistanceToCameraSD"]).astype(np.float32)
        depth_stored = (
            (depth_m * 1000.0 / self.config.depth_scale)
            .clip(0, 65535)
            .astype(np.uint16)
        )
        backend.makedirs(f"{sd}/depth")
        backend.write_array(f"{sd}/depth/{stem}.png", depth_stored, encoding="png")

        # --- Camera params ---
        cam = extract_camera_params(render_vars["CameraParams"])
        K_flat = cam.K.ravel().tolist()
        r_w2c, t_w2c_scene = world_to_opencv_camera(cam)
        R_w2c = r_w2c.ravel().tolist()
        t_w2c = (t_w2c_scene * mm_per_scene_unit).tolist()
        self._scene_camera[str(frame_id)] = {
            "cam_K": K_flat,
            "cam_R_w2c": R_w2c,
            "cam_t_w2c": t_w2c,
            "depth_scale": float(self.config.depth_scale),
        }

        # --- Per-object masks, scene_gt, scene_gt_info ---
        inst_seg = render_vars["SemanticInstanceSegmentation"]
        id_to_label = id_label_map_from_render_vars(render_vars)
        bboxes_2d = {
            b.instance_id: b
            for b in iter_bboxes_2d(
                render_vars["SemanticBoundingBox2DTight"], id_to_label
            )
        }
        gt_list: list[dict[str, Any]] = []
        gt_info_list: list[dict[str, Any]] = []

        backend.makedirs(f"{sd}/mask")
        backend.makedirs(f"{sd}/mask_visib")

        bboxes_3d = list(
            iter_bboxes_3d(render_vars["SemanticBoundingBox3D"], id_to_label)
        )
        inst_masks = instance_masks(inst_seg, [b.instance_id for b in bboxes_3d])
        for gt_index, bb3 in enumerate(bboxes_3d):
            iid = bb3.instance_id
            label = bb3.label
            obj_id = self._resolve_obj_id(label, iid)

            # BOP requires a rotation matrix; a scaled object's box transform
            # carries the scale, so remove it before the pose reaches the file.
            # A zero-length axis has no direction, so there is no rotation to
            # write: refuse it before any file for this object is written.
            unit_rotation, axis_scale = rotation_and_scale(bb3.rotation)
            if not np.all(axis_scale > 0.0):
                raise ValueError(
                    f"BOPWriter: instance {iid} ({label!r}) has a zero-length "
                    "box axis, so it has no rotation to write as cam_R_m2c"
                )

            # Visible mask from instance segmentation
            mask_visib = inst_masks[iid] * 255
            mask_all = mask_visib  # synthetic: no occlusion model

            obj_stem = f"{stem}_{gt_index:06d}"
            # Without an occlusion model both masks are the same image, so
            # encode it once and store the bytes under both paths.
            mask_png = get_image_codec().encode(mask_all, format="png")
            backend.write(f"{sd}/mask/{obj_stem}.png", mask_png)
            backend.write(f"{sd}/mask_visib/{obj_stem}.png", mask_png)

            # GT pose. BOP stores cam_t_m2c as a flat 3-vector [tx, ty, tz]
            # (mm); loaders reshape it to 3x1 themselves.
            rot_cam, center_cam = pose_in_opencv_camera(cam, unit_rotation, bb3.center)
            R_m2c = rot_cam.ravel().tolist()
            t_m2c = (center_cam * mm_per_scene_unit).reshape(3).tolist()
            gt_list.append({"cam_R_m2c": R_m2c, "cam_t_m2c": t_m2c, "obj_id": obj_id})

            # GT info (bounding boxes, pixel counts, visibility)
            px_visib = int(np.count_nonzero(mask_visib))
            px_all = px_visib

            bb2 = bboxes_2d.get(iid)
            if bb2 is not None:
                bx = int(bb2.x_min)
                by = int(bb2.y_min)
                bw = max(1, int(bb2.x_max - bb2.x_min))
                bh = max(1, int(bb2.y_max - bb2.y_min))
                bbox_obj = [bx, by, bw, bh]
                bbox_visib = bbox_obj
            else:
                ys, xs = np.where(mask_visib > 0)
                if len(ys):
                    bbox_visib = bbox_obj = [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ]
                else:
                    bbox_visib = bbox_obj = [0, 0, 0, 0]

            gt_info_list.append(
                {
                    "bbox_obj": bbox_obj,
                    "bbox_visib": bbox_visib,
                    "px_count_all": px_all,
                    "px_count_nobd": px_all,
                    "px_count_visib": px_visib,
                    "visib_fract": 1.0 if px_all == 0 else px_visib / px_all,
                }
            )

        self._scene_gt[str(frame_id)] = gt_list
        self._scene_gt_info[str(frame_id)] = gt_info_list

    def finalize(self, backend: Backend) -> None:
        """Persist scene JSON, retaining successful writes across retries.

        Args:
            backend (Backend): Destination backend for the output files.
        """
        if self._finalized:
            return
        sd = self._scene_dir
        if self._finalization_payloads is None:
            self._finalization_payloads = {
                f"{sd}/{name}.json": json.dumps(value, indent=2).encode()
                for name, value in (
                    ("scene_camera", self._scene_camera),
                    ("scene_gt", self._scene_gt),
                    ("scene_gt_info", self._scene_gt_info),
                )
            }
            self._finalization_payloads[f"{sd}/{MASK_LAYOUT_FILE}"] = json.dumps(
                MASK_LAYOUT
            ).encode()
        for path, payload in self._finalization_payloads.items():
            if path not in self._finalized_paths:
                backend.write(path, payload)
                self._finalized_paths.add(path)
        self._finalized = True

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _resolve_obj_id(self, label: str, instance_id: int) -> int:
        """Return a BOP object ID for the given label or fall back to instance_id."""
        if self.config.object_id_map is not None:
            return self.config.object_id_map.get(label, instance_id)
        return instance_id
