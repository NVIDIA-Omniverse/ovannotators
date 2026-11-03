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

"""YOLOWriter — Ultralytics YOLO detection/segmentation dataset writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from ovannotators.writers._helpers import (
    id_label_map_from_render_vars,
    instance_masks,
    iter_bboxes_2d,
    mask_to_polygon,
    validate_required_vars,
)
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class YOLOWriterConfig:
    """Configuration for :class:`YOLOWriter`.

    Attributes:
        task: ``"detect"`` writes normalised ``class cx cy w h`` labels.
            ``"segment"`` appends polygon points for instance masks.
        val_split: Fraction of frames assigned to the ``val`` split.
            Frames are assigned deterministically using
            ``frame_id % round(1 / val_split) == 0``.
        image_format: Image file extension written to ``images/{split}/``.
        classes: Explicit class list defining the ``class_id → name`` mapping.
            ``None`` collects class names in first-seen order from the data.
    """

    task: Literal["detect", "segment"] = "detect"
    val_split: float = 0.1
    image_format: Literal["jpg", "png"] = "jpg"
    classes: list[str] | None = None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_REQUIRED_BASE: frozenset[str] = frozenset({"LdrColor", "SemanticBoundingBox2DTight"})


@WriterRegistry.register
class YOLOWriter:
    """Writes dataset in Ultralytics YOLO format.

    Output structure::

        {output_dir}/
        ├── images/
        │   ├── train/{frame_id:06d}.jpg
        │   └── val/{frame_id:06d}.jpg
        ├── labels/
        │   ├── train/{frame_id:06d}.txt
        │   └── val/{frame_id:06d}.txt
        └── data.yaml

    Detection label row: ``class_id cx cy w h`` (normalised to [0, 1]).

    Segmentation label row: ``class_id x1 y1 x2 y2 … xn yn``
    (polygon points normalised to [0, 1]).
    """

    name: ClassVar[str] = "yolo"
    required_render_vars: ClassVar[frozenset[str]] = _REQUIRED_BASE

    def __init__(self, config: YOLOWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (YOLOWriterConfig | None): Writer options; defaults are used
                when ``None``.
        """
        self.config: YOLOWriterConfig = config or YOLOWriterConfig()
        if self.config.task == "segment":
            self.required_render_vars = _REQUIRED_BASE | frozenset(
                {"SemanticInstanceSegmentation"}
            )  # type: ignore[assignment]
        # Ordered class list (built dynamically when config.classes is None)
        self._class_list: list[str] = list(self.config.classes or [])
        self._class_index: dict[str, int] = {
            c: i for i, c in enumerate(self._class_list)
        }
        self._finalized: bool = False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write one frame's image and label file to the correct split folder.

        Args:
            frame_id (int): Zero-based frame index; used to pick the train/val
                split and name files ``{frame_id:06d}``.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars``.
            backend (Backend): Destination backend for the frame's files.

        Raises:
            MissingRenderVarError: If a required render var is missing.
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )

        split = self._split_for(frame_id)
        stem = f"{frame_id:06d}"
        fmt = self.config.image_format

        # --- Image ---
        backend.makedirs(f"images/{split}")
        backend.write_array(
            f"images/{split}/{stem}.{fmt}", render_vars["LdrColor"], encoding=fmt
        )

        # --- Labels ---
        ldr = render_vars["LdrColor"]
        H, W = (ldr.shape[0], ldr.shape[1]) if ldr.ndim >= 2 else (1, 1)
        inst_seg: np.ndarray | None = render_vars.get("SemanticInstanceSegmentation")
        id_to_label = id_label_map_from_render_vars(render_vars)

        lines: list[str] = []
        bboxes = list(
            iter_bboxes_2d(render_vars["SemanticBoundingBox2DTight"], id_to_label)
        )
        inst_masks: dict[int, np.ndarray] = {}
        if self.config.task == "segment" and inst_seg is not None:
            inst_masks = instance_masks(inst_seg, [b.instance_id for b in bboxes])
        for bbox in bboxes:
            label = bbox.label
            class_id = self._get_or_add_class(label)
            x_min, y_min, x_max, y_max = (
                bbox.x_min,
                bbox.y_min,
                bbox.x_max,
                bbox.y_max,
            )
            cx = (x_min + x_max) / 2.0 / W
            cy = (y_min + y_max) / 2.0 / H
            bw = (x_max - x_min) / W
            bh = (y_max - y_min) / H

            if self.config.task == "segment" and inst_seg is not None:
                mask = inst_masks[bbox.instance_id]
                poly = mask_to_polygon(mask)
                if poly:
                    pts_str = " ".join(f"{v:.6f}" for v in poly)
                    lines.append(f"{class_id} {pts_str}")
                    continue
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        backend.makedirs(f"labels/{split}")
        label_txt = "\n".join(lines) + ("\n" if lines else "")
        backend.write(f"labels/{split}/{stem}.txt", label_txt.encode())

    def finalize(self, backend: Backend) -> None:
        """Write the dataset ``data.yaml`` (class names and split paths).

        Args:
            backend (Backend): Destination backend for ``data.yaml``.
        """
        if self._finalized:
            return
        self._finalized = True
        backend.write("data.yaml", _build_data_yaml(self._class_list).encode())

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _split_for(self, frame_id: int) -> str:
        """Return ``"val"`` for every N-th frame, ``"train"`` otherwise."""
        val_step = max(1, round(1.0 / max(self.config.val_split, 1e-6)))
        return "val" if frame_id % val_step == 0 else "train"

    def _get_or_add_class(self, label: str) -> int:
        """Return the class index, registering the label if not yet seen."""
        if label not in self._class_index:
            if self.config.classes is not None:
                # Fixed class list: unknown labels map to 0
                return 0
            self._class_index[label] = len(self._class_list)
            self._class_list.append(label)
        return self._class_index[label]


def _build_data_yaml(classes: list[str]) -> str:
    """Render the YOLO ``data.yaml`` content."""
    nc = len(classes)
    names_repr = "[" + ", ".join(repr(c) for c in classes) + "]"
    return f"train: images/train\nval: images/val\nnc: {nc}\nnames: {names_repr}\n"
