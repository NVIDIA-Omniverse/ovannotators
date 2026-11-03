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

"""Load emitted detection datasets with FiftyOne's public importers."""

from __future__ import annotations

import pytest

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers import COCOWriter, KITTIWriter, YOLOWriter

fo = pytest.importorskip(
    "fiftyone", reason="install FiftyOne to run interoperability tests"
)


def _write(writer, render_vars, root):
    backend = DiskBackend(DiskBackendConfig(root_dir=root))
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)


def _assert_detections(dataset, field="ground_truth"):
    try:
        assert len(dataset) == 1
        detections = dataset.first()[field].detections
        labels = {d.label for d in detections}
        assert labels == {"car", "person"}
        car = next(d for d in detections if d.label == "car")
        assert car.bounding_box == pytest.approx(
            [10 / 640, 20 / 480, 90 / 640, 180 / 480], abs=1e-5
        )
    finally:
        dataset.delete()


def test_coco_import(sample_render_vars, tmp_path):
    _write(COCOWriter(), sample_render_vars, tmp_path)
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(tmp_path / "images"),
        labels_path=str(tmp_path / "annotations" / "instances_train.json"),
        label_field="ground_truth",
    )
    _assert_detections(dataset, "ground_truth_detections")


def test_kitti_import(sample_render_vars, tmp_path):
    _write(KITTIWriter(), sample_render_vars, tmp_path)
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.KITTIDetectionDataset,
        data_path=str(tmp_path / "image_2"),
        labels_path=str(tmp_path / "label_2"),
        label_field="ground_truth",
    )
    _assert_detections(dataset)


def test_yolo_import(sample_render_vars, tmp_path):
    _write(YOLOWriter(), sample_render_vars, tmp_path)
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(tmp_path),
        dataset_type=fo.types.YOLOv5Dataset,
        yaml_path="data.yaml",
        split="val",  # frame 0 goes to validation by the writer's split rule
        label_field="ground_truth",
    )
    _assert_detections(dataset)
