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

"""Exercise OpenLABEL image output and its FiftyOne import path."""

from __future__ import annotations

import json

import pytest

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.exceptions import MissingRenderVarError
from ovannotators.writers import (
    OpenLABELWriter,
    OpenLABELWriterConfig,
    WriterRegistry,
)


def _write(root, render_vars, config=None):
    writer = OpenLABELWriter(config)
    backend = DiskBackend(DiskBackendConfig(root_dir=root))
    writer.write_frame(7, render_vars, backend)
    writer.finalize(backend)
    writer.finalize(backend)
    return json.loads((root / "labels" / "000007.json").read_text())["openlabel"]


def test_image_and_bbox_layout(sample_render_vars, tmp_path):
    data = _write(tmp_path, sample_render_vars)
    assert WriterRegistry.get("openlabel") is OpenLABELWriter
    assert (tmp_path / "data" / "000007.png").is_file()
    assert data["metadata"]["schema_version"] == "1.0.0"
    assert {obj["type"] for obj in data["objects"].values()} == {"car", "person"}
    assert data["objects"]["1"]["object_data"]["bbox"][0]["val"] == [
        55.0,
        110.0,
        90.0,
        180.0,
    ]


def test_instance_polygon_uses_absolute_pixels(sample_render_vars, tmp_path):
    data = _write(
        tmp_path,
        sample_render_vars,
        OpenLABELWriterConfig(include_segmentation=True),
    )
    polygon = data["objects"]["1"]["object_data"]["poly2d"][0]
    assert polygon["mode"] == "MODE_POLY2D_ABSOLUTE"
    assert polygon["closed"] is True
    assert len(polygon["val"]) >= 6
    assert min(polygon["val"]) >= 0
    assert max(polygon["val"][::2]) == pytest.approx(99.5)
    assert max(polygon["val"][1::2]) == pytest.approx(199.5)


def test_missing_mask_rejected_when_enabled(sample_render_vars, tmp_path):
    render_vars = dict(sample_render_vars)
    del render_vars["SemanticInstanceSegmentation"]
    with pytest.raises(MissingRenderVarError):
        _write(
            tmp_path,
            render_vars,
            OpenLABELWriterConfig(include_segmentation=True),
        )


@pytest.mark.parametrize("include_segmentation", [False, True])
def test_fiftyone_import(sample_render_vars, tmp_path, include_segmentation):
    fo = pytest.importorskip("fiftyone")
    _write(
        tmp_path,
        sample_render_vars,
        OpenLABELWriterConfig(include_segmentation=include_segmentation),
    )
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(tmp_path),
        dataset_type=fo.types.OpenLABELImageDataset,
    )
    try:
        assert len(dataset) == 1
        assert {d.label for d in dataset.first().detections.detections} == {
            "car",
            "person",
        }
        if include_segmentation:
            assert {d.label for d in dataset.first().segmentations.detections} == {
                "car",
                "person",
            }
    finally:
        dataset.delete()
