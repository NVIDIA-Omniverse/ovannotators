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

"""ASAM OpenLABEL 1.0 image writer for 2D boxes and instance polygons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from ovannotators.writers._helpers import (
    id_label_map_from_render_vars,
    id_prim_path_map_from_render_vars,
    instance_masks,
    iter_bboxes_2d,
    mask_to_polygon,
    validate_required_vars,
)
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


@dataclass
class OpenLABELWriterConfig:
    """Configure the OpenLABEL image writer.

    Attributes:
        image_format: Image encoding for the ``data/`` directory.
        include_segmentation: Add absolute pixel polygons from instance masks.
    """

    image_format: Literal["png", "jpg"] = "png"
    include_segmentation: bool = False


@WriterRegistry.register
class OpenLABELWriter:
    """Write one OpenLABEL JSON annotation per image.

    ``data/{frame_id:06d}.{image_format}`` and ``labels/{frame_id:06d}.json``
    follow the layout imported by FiftyOne's ``OpenLABELImageDataset``.
    Object ``bbox`` values are pixel-space ``[cx, cy, width, height]``;
    optional ``poly2d`` values use ``MODE_POLY2D_ABSOLUTE`` pixel coordinates.
    """

    name: ClassVar[str] = "openlabel"
    required_render_vars: ClassVar[frozenset[str]] = frozenset(
        {"LdrColor", "SemanticBoundingBox2DTight"}
    )

    def __init__(self, config: OpenLABELWriterConfig | None = None) -> None:
        """Initialize the writer and validate configuration.

        Args:
            config: Output options; defaults are used when absent.
        """
        self.config = config or OpenLABELWriterConfig()
        if self.config.image_format not in ("png", "jpg"):
            raise ValueError("image_format must be 'png' or 'jpg'")
        if self.config.include_segmentation:
            self.required_render_vars = self.__class__.required_render_vars | {
                "SemanticInstanceSegmentation"
            }
        self._finalized = False

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write an image and its matching OpenLABEL annotations.

        Args:
            frame_id: Frame identifier used as the image and label basename.
            render_vars: Extracted renderer variables for the frame.
            backend: Destination for the image and JSON file.
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )
        image = render_vars["LdrColor"]
        height, width = image.shape[:2]
        stem = f"{frame_id:06d}"
        fmt = self.config.image_format
        backend.write_array(f"data/{stem}.{fmt}", image, encoding=fmt)

        boxes = list(
            iter_bboxes_2d(
                render_vars["SemanticBoundingBox2DTight"],
                id_label_map_from_render_vars(render_vars),
                id_prim_path_map_from_render_vars(render_vars),
            )
        )
        masks = (
            instance_masks(
                render_vars["SemanticInstanceSegmentation"],
                [box.instance_id for box in boxes],
            )
            if self.config.include_segmentation
            else {}
        )
        objects = {}
        for box in boxes:
            object_data = {
                "bbox": [
                    {
                        "name": "bbox",
                        "val": [
                            (box.x_min + box.x_max) / 2,
                            (box.y_min + box.y_max) / 2,
                            box.x_max - box.x_min,
                            box.y_max - box.y_min,
                        ],
                    }
                ]
            }
            if self.config.include_segmentation:
                polygon = mask_to_polygon(masks[box.instance_id])
                if polygon:
                    object_data["poly2d"] = [
                        {
                            "name": "segmentation",
                            "mode": "MODE_POLY2D_ABSOLUTE",
                            "closed": True,
                            "val": [
                                coordinate * (width if index % 2 == 0 else height)
                                for index, coordinate in enumerate(polygon)
                            ],
                        }
                    ]
            objects[str(box.instance_id)] = {
                "name": box.prim_path or f"instance_{box.instance_id}",
                "type": box.label,
                "object_data": object_data,
            }

        payload = {
            "openlabel": {
                "metadata": {"schema_version": "1.0.0"},
                "objects": objects,
            }
        }
        backend.write(f"labels/{stem}.json", json.dumps(payload, indent=2).encode())

    def finalize(self, backend: Backend) -> None:
        """Complete the writer; OpenLABEL image output has no global index.

        Args:
            backend: Destination backend, unused because frames are independent.
        """
        if self._finalized:
            return
        self._finalized = True
