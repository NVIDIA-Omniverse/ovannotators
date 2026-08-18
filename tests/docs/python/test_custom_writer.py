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

"""Tested snippets for the ``new-writer`` skill.

Every illustrative region is wrapped in ``# [snippet:doc-...]`` markers and is
referenced from ``skills/new-writer/SKILL.md`` via ``> **Source:**`` directives.
These tests are the single source of truth for the skill's code: when the writer
API changes, this file must change with it. Runs CPU-only, so it joins the
default (non-GPU) test suite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# [snippet:doc-writer-config]
from dataclasses import dataclass
from typing import ClassVar

from ovannotators.backends._base import Backend
from ovannotators.writers._helpers import (
    id_label_map_from_render_vars,
    iter_bboxes_2d,
    validate_required_vars,
)
from ovannotators.writers._registry import WriterRegistry


@dataclass
class ManifestWriterConfig:
    """Configuration for :class:`ManifestWriter`.

    Args:
        image_format: Encoding for the per-frame color image (``"png"`` or
            ``"jpg"``).
        manifest_name: File name of the manifest written on finalize.
    """

    image_format: str = "png"
    manifest_name: str = "manifest.json"


# [/snippet:doc-writer-config]


# [snippet:doc-writer-class]
@WriterRegistry.register
class ManifestWriter:
    """Example writer: per-frame RGB image plus an accumulated 2D-bbox manifest.

    Demonstrates the full :class:`~ovannotators.writers._base.Writer` contract:
    per-frame validation, incremental writes through the backend, and a deferred
    ``manifest.json`` flushed once in :meth:`finalize`.
    """

    name: ClassVar[str] = "manifest"
    required_render_vars: ClassVar[frozenset[str]] = frozenset(
        {"LdrColor", "SemanticBoundingBox2DTight"}
    )

    def __init__(self, config: ManifestWriterConfig | None = None) -> None:
        self._cfg = config or ManifestWriterConfig()
        self._records: list[dict] = []  # cross-frame state flushed on finalize
        self._finalized = False

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        # Always validate required vars first — raises MissingRenderVarError.
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )

        # Image arrays (NumPy or zero-copy Warp views) go straight to the codec.
        image_path = f"images/{frame_id:06d}.{self._cfg.image_format}"
        backend.makedirs("images")
        backend.write_array(
            image_path, render_vars["LdrColor"], encoding=self._cfg.image_format
        )

        # Decode the structured bbox payload into per-box records; labels come
        # from the optional SemanticIdMap render var.
        id_to_label = id_label_map_from_render_vars(render_vars)
        for box in iter_bboxes_2d(
            render_vars["SemanticBoundingBox2DTight"], id_to_label
        ):
            self._records.append(
                {
                    "frame_id": frame_id,
                    "image": image_path,
                    "label": box.label,
                    "bbox": [box.x_min, box.y_min, box.x_max, box.y_max],
                }
            )

    def finalize(self, backend: Backend) -> None:
        # Idempotent: a second call must not raise or duplicate output.
        if self._finalized:
            return
        payload = json.dumps({"annotations": self._records}, indent=2).encode()
        backend.write(self._cfg.manifest_name, payload)
        self._finalized = True


# [/snippet:doc-writer-class]


# ---------------------------------------------------------------------------
# Test fixtures: synthetic ovrtx-layout render vars (no GPU / renderer needed)
# ---------------------------------------------------------------------------


def _bbox2d(boxes):
    """Build an OVRTX_BBOX2D_DTYPE array from (instance_id, semantic_id, x0, y0, x1, y1)."""
    from ovannotators.writers._helpers import OVRTX_BBOX2D_DTYPE

    arr = np.zeros(len(boxes), dtype=OVRTX_BBOX2D_DTYPE)
    for i, (iid, sid, x0, y0, x1, y1) in enumerate(boxes):
        arr[i]["instance_id"] = (iid, 0, 0, 0)
        arr[i]["semantic_id"] = (sid, 0, 0, 0)
        arr[i]["x_min"], arr[i]["y_min"], arr[i]["x_max"], arr[i]["y_max"] = (
            x0,
            y0,
            x1,
            y1,
        )
    return arr


def _id_map(labels: dict[int, str]) -> np.ndarray:
    """Pack ``{semantic_id: label}`` into a ``SemanticIdMap`` uint8 buffer."""
    entry_dtype = np.dtype(
        [("semantic_id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    items = list(labels.items())
    header_size = len(items) * entry_dtype.itemsize
    strings = b""
    entries = np.zeros(len(items), dtype=entry_dtype)
    for i, (sid, label) in enumerate(items):
        encoded = label.encode("utf-8")
        entries[i]["semantic_id"] = (sid, 0, 0, 0)
        entries[i]["label_length"] = len(encoded)
        entries[i]["label_offset"] = header_size + len(strings)
        strings += encoded
    raw = entries.tobytes() + strings + int(len(items)).to_bytes(4, "little")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _sample_render_vars():
    """Minimal frame: an RGBA image and one labelled 2D box."""
    return {
        "LdrColor": np.zeros((32, 48, 4), dtype=np.uint8),
        "SemanticBoundingBox2DTight": _bbox2d([(1, 1, 10, 20, 30, 25)]),
        "SemanticIdMap": _id_map({1: "car"}),
    }


def test_manifest_writer_roundtrip(disk_backend):
    """The example writer emits a per-frame image and a manifest on finalize."""
    # [snippet:doc-writer-register]
    # Either retrieve the writer by registry name (config-driven code) ...
    writer = WriterRegistry.create("manifest", ManifestWriterConfig(image_format="png"))
    # ... or instantiate the class directly (static typing / autocompletion):
    #   writer = ManifestWriter(ManifestWriterConfig())
    # [/snippet:doc-writer-register]

    # [snippet:doc-writer-test]
    render_vars = _sample_render_vars()
    writer.write_frame(0, render_vars, disk_backend)
    writer.finalize(disk_backend)

    manifest = json.loads(disk_backend.read("manifest.json"))
    assert manifest["annotations"][0]["label"] == "car"
    assert manifest["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 25.0]
    assert disk_backend.exists("images/000000.png")
    # [/snippet:doc-writer-test]


def test_manifest_writer_missing_required_var_raises(disk_backend):
    """Omitting a required render var raises MissingRenderVarError."""
    from ovannotators.exceptions import MissingRenderVarError

    writer = ManifestWriter()
    with pytest.raises(MissingRenderVarError):
        writer.write_frame(0, {"LdrColor": np.zeros((4, 4, 4), np.uint8)}, disk_backend)


def test_manifest_writer_finalize_idempotent(disk_backend):
    """A second finalize call neither raises nor rewrites the manifest."""
    writer = ManifestWriter()
    writer.write_frame(0, _sample_render_vars(), disk_backend)
    writer.finalize(disk_backend)
    writer.finalize(disk_backend)  # guarded by self._finalized: a no-op
