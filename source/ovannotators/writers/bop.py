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

from ovannotators.writers._helpers import (
    _to_numpy,
    extract_camera_params,
    id_label_map_from_render_vars,
    instance_masks,
    iter_bboxes_2d,
    iter_bboxes_3d,
    validate_required_vars,
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
        ├── mask/             {frame_id:06d}_{obj_id:06d}.png
        ├── mask_visib/       {frame_id:06d}_{obj_id:06d}.png
        ├── scene_camera.json
        ├── scene_gt.json
        └── scene_gt_info.json

    ``scene_camera.json``, ``scene_gt.json``, and ``scene_gt_info.json`` are
    accumulated in memory and written during :meth:`finalize`.

    The ``SemanticBoundingBox3D`` render var provides object poses:

    * ``cam_R_m2c`` = the rotation matrix from the 3-D bbox ``rotation`` field
      (model-to-camera).
    * ``cam_t_m2c`` = ``center * 1000`` (model centre in camera frame, mm).
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
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )

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
        R_w2c = cam.R.ravel().tolist()
        t_w2c = (cam.t * 1000.0).tolist()
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
        for bb3 in bboxes_3d:
            iid = bb3.instance_id
            label = bb3.label
            obj_id = self._resolve_obj_id(label, iid)

            # Visible mask from instance segmentation
            mask_visib = inst_masks[iid] * 255
            mask_all = mask_visib  # synthetic: no occlusion model

            obj_stem = f"{stem}_{obj_id:06d}"
            backend.write_array(f"{sd}/mask/{obj_stem}.png", mask_all, encoding="png")
            backend.write_array(
                f"{sd}/mask_visib/{obj_stem}.png", mask_visib, encoding="png"
            )

            # GT pose. BOP stores cam_t_m2c as a flat 3-vector [tx, ty, tz]
            # (mm); loaders reshape it to 3x1 themselves.
            R_m2c = bb3.rotation.ravel().tolist()
            t_m2c = (bb3.center * 1000.0).reshape(3).tolist()
            gt_list.append({"cam_R_m2c": R_m2c, "cam_t_m2c": t_m2c, "obj_id": obj_id})

            # GT info (bounding boxes, pixel counts, visibility)
            px_visib = int(np.sum(mask_visib > 0))
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
        """Write the accumulated ``scene_gt`` and ``scene_gt_info`` JSON files.

        Args:
            backend (Backend): Destination backend for the output files.
        """
        if self._finalized:
            return
        self._finalized = True
        sd = self._scene_dir

        def _dump(obj: Any) -> bytes:
            return json.dumps(obj, indent=2).encode()

        backend.write(f"{sd}/scene_camera.json", _dump(self._scene_camera))
        backend.write(f"{sd}/scene_gt.json", _dump(self._scene_gt))
        backend.write(f"{sd}/scene_gt_info.json", _dump(self._scene_gt_info))

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _resolve_obj_id(self, label: str, instance_id: int) -> int:
        """Return a BOP object ID for the given label or fall back to instance_id."""
        if self.config.object_id_map is not None:
            return self.config.object_id_map.get(label, instance_id)
        return instance_id
