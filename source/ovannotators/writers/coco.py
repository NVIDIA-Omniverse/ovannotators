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

"""COCOWriter — MS-COCO JSON dataset writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from ovannotators.exceptions import MissingRenderVarError
from ovannotators.writers._helpers import (
    _to_numpy,
    id_label_map_from_render_vars,
    instance_masks,
    iter_bboxes_2d,
    mask_to_rle,
    validate_required_vars,
)
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class COCOWriterConfig:
    """Configuration for :class:`COCOWriter`.

    Attributes:
        split: Dataset split name used in annotation filenames
            (``instances_{split}.json``).
        image_format: Encoding format for image files.
        include_segmentation: Attach per-instance RLE segmentation masks.
        include_panoptic: Emit COCO panoptic annotations and per-image panoptic
            PNGs under ``annotations/panoptic_{split}``.
        bbox_mode: ``"tight"`` uses the raw 2-D bounding-box values;
            ``"loose"`` expands each side by 1 % of the box dimension.
    """

    split: str = "train"
    image_format: Literal["jpg", "png"] = "jpg"
    include_segmentation: bool = False
    include_panoptic: bool = False
    bbox_mode: Literal["tight", "loose"] = "tight"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@WriterRegistry.register
class COCOWriter:
    """Accumulates annotations across frames and writes COCO JSON on finalize.

    Output structure::

        {output_dir}/
        ├── images/
        │   └── {frame_id:06d}.jpg
        └── annotations/
            └── instances_{split}.json

    The ``SemanticBoundingBox2DTight`` render var must be a numpy structured
    array following :data:`OVRTX_BBOX2D_DTYPE`.  Class labels are resolved from
    the optional ``SemanticIdMap`` render var.
    """

    name: ClassVar[str] = "coco"
    required_render_vars: ClassVar[frozenset[str]] = frozenset(
        {"LdrColor", "SemanticBoundingBox2DTight"}
    )

    def __init__(self, config: COCOWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (COCOWriterConfig | None): Writer options; defaults are used
                when ``None``.
        """
        self.config: COCOWriterConfig = config or COCOWriterConfig()
        self.required_render_vars = (
            self.__class__.required_render_vars | {"SemanticInstanceSegmentation"}
            if self.config.include_panoptic
            else self.__class__.required_render_vars
        )
        self._images: list[dict[str, Any]] = []
        self._annotations: list[dict[str, Any]] = []
        self._panoptic_annotations: list[dict[str, Any]] = []
        # label → category_id (1-based)
        self._categories: dict[str, int] = {}
        self._ann_id: int = 1
        self._finalized: bool = False

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write the frame image and accumulate COCO annotation data.

        Args:
            frame_id (int): Zero-based frame index, used in image/file naming.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars`` (and
                ``SemanticInstanceSegmentation`` when panoptic output is on).
            backend (Backend): Destination backend for image bytes.

        Raises:
            MissingRenderVarError: If a required render var is missing.
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )
        if (
            self.config.include_panoptic
            and "SemanticInstanceSegmentation" not in render_vars
        ):
            raise MissingRenderVarError(
                self.name, "SemanticInstanceSegmentation", frame_id=frame_id
            )

        ldr = render_vars["LdrColor"]
        bboxes_arr = render_vars["SemanticBoundingBox2DTight"]
        inst_seg: np.ndarray | None = render_vars.get("SemanticInstanceSegmentation")
        # Instance segmentation is consumed by NumPy-only panoptic/encoding paths.
        if inst_seg is not None:
            inst_seg = _to_numpy(inst_seg)
        if inst_seg is not None and inst_seg.ndim > 2:
            # Render backends may expose instance ids with an extra channel axis.
            inst_seg = inst_seg[..., 0]
        id_to_label = id_label_map_from_render_vars(render_vars)

        h, w = (ldr.shape[0], ldr.shape[1]) if ldr.ndim >= 2 else (0, 0)

        # --- Write image ---
        fmt = self.config.image_format
        img_filename = f"{frame_id:06d}.{fmt}"
        backend.makedirs("images")
        backend.write_array(f"images/{img_filename}", ldr, encoding=fmt)

        self._images.append(
            {
                "id": frame_id + 1,
                "file_name": img_filename,
                "width": w,
                "height": h,
                "license": 0,
            }
        )

        # --- Accumulate annotations ---
        instance_to_category: dict[int, int] = {}
        bboxes = list(iter_bboxes_2d(bboxes_arr, id_to_label))
        inst_masks: dict[int, np.ndarray] = {}
        if self.config.include_segmentation and inst_seg is not None:
            inst_masks = instance_masks(inst_seg, [b.instance_id for b in bboxes])
        for bbox in bboxes:
            x_min = bbox.x_min
            y_min = bbox.y_min
            x_max = bbox.x_max
            y_max = bbox.y_max
            label = bbox.label

            if self.config.bbox_mode == "loose":
                pad_x = max(1.0, (x_max - x_min) * 0.01)
                pad_y = max(1.0, (y_max - y_min) * 0.01)
                x_min -= pad_x
                y_min -= pad_y
                x_max += pad_x
                y_max += pad_y

            if label not in self._categories:
                self._categories[label] = len(self._categories) + 1
            cat_id = self._categories[label]
            instance_to_category.setdefault(int(bbox.instance_id), cat_id)

            bw = float(x_max - x_min)
            bh = float(y_max - y_min)

            ann: dict[str, Any] = {
                "id": self._ann_id,
                "image_id": frame_id + 1,
                "category_id": cat_id,
                "bbox": [float(x_min), float(y_min), bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "segmentation": [],
            }

            if self.config.include_segmentation and inst_seg is not None:
                instance_id = bbox.instance_id
                ann["segmentation"] = mask_to_rle(inst_masks[instance_id])

            self._annotations.append(ann)
            self._ann_id += 1

        if self.config.include_panoptic and inst_seg is not None:
            segments_info, panoptic_ids = self._build_panoptic_segments(
                inst_seg=inst_seg, instance_to_category=instance_to_category
            )
            panoptic_file_name = self._panoptic_file_name_for_image(img_filename)
            panoptic_rel_dir = f"annotations/panoptic_{self.config.split}"
            backend.makedirs(panoptic_rel_dir)
            backend.write_array(
                f"{panoptic_rel_dir}/{panoptic_file_name}",
                self._panoptic_ids_to_rgb(panoptic_ids),
                encoding="png",
            )
            self._panoptic_annotations.append(
                {
                    "image_id": frame_id + 1,
                    "file_name": panoptic_file_name,
                    "segments_info": segments_info,
                }
            )

    def finalize(self, backend: Backend) -> None:
        """Write ``instances_{split}.json`` to the annotations directory.

        Args:
            backend (Backend): Destination backend for the annotations file.
        """
        if self._finalized:
            return
        self._finalized = True

        backend.makedirs("annotations")

        categories = [
            {"id": cat_id, "name": name, "supercategory": ""}
            for name, cat_id in sorted(self._categories.items(), key=lambda kv: kv[1])
        ]

        coco: dict[str, Any] = {
            "info": {
                "description": "ovannotators synthetic dataset",
                "version": "1.0",
                "year": 2024,
            },
            "licenses": [{"id": 0, "name": "Synthetic — no license"}],
            "categories": categories,
            "images": self._images,
            "annotations": self._annotations,
        }

        split = self.config.split
        payload = json.dumps(coco, indent=2).encode()
        backend.write(f"annotations/instances_{split}.json", payload)

        if self.config.include_panoptic:
            panoptic_coco: dict[str, Any] = {
                "info": coco["info"],
                "licenses": coco["licenses"],
                "categories": categories,
                "images": self._images,
                "annotations": self._panoptic_annotations,
            }
            panoptic_payload = json.dumps(panoptic_coco, indent=2).encode()
            backend.write(f"annotations/panoptic_{split}.json", panoptic_payload)

    @staticmethod
    def _panoptic_file_name_for_image(image_file_name: str) -> str:
        stem, _dot, _suffix = image_file_name.rpartition(".")
        return f"{stem or image_file_name}.png"

    @staticmethod
    def _panoptic_ids_to_rgb(panoptic_ids: np.ndarray) -> np.ndarray:
        ids = panoptic_ids.astype(np.uint32, copy=False)
        rgb = np.empty((*ids.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (ids & 255).astype(np.uint8)
        rgb[..., 1] = ((ids >> 8) & 255).astype(np.uint8)
        rgb[..., 2] = ((ids >> 16) & 255).astype(np.uint8)
        return rgb

    @staticmethod
    def _build_panoptic_segments(
        inst_seg: np.ndarray, instance_to_category: dict[int, int]
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        panoptic_ids = np.zeros(inst_seg.shape[:2], dtype=np.uint32)
        segments_info: list[dict[str, Any]] = []
        next_segment_id = 1

        instance_ids = sorted(
            int(value) for value in np.unique(inst_seg) if int(value) > 0
        )
        for instance_id in instance_ids:
            category_id = instance_to_category.get(instance_id)
            if category_id is None:
                continue
            mask = inst_seg == instance_id
            area = int(np.count_nonzero(mask))
            if area <= 0:
                continue

            ys, xs = np.where(mask)
            x_min = int(xs.min())
            y_min = int(ys.min())
            width = int(xs.max() - x_min + 1)
            height = int(ys.max() - y_min + 1)

            segment_id = next_segment_id
            next_segment_id += 1
            panoptic_ids[mask] = segment_id
            segments_info.append(
                {
                    "id": segment_id,
                    "category_id": int(category_id),
                    "area": area,
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0,
                }
            )

        return segments_info, panoptic_ids
