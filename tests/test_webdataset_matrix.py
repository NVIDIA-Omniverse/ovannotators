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

"""Apply the shared WebDataset regression matrix to the Basic and COCO adapters."""

from __future__ import annotations

import numpy as np
import pytest

from ovannotators import (
    BasicWriter,
    BasicWriterConfig,
    COCOWriter,
    COCOWriterConfig,
    DiskBackend,
    DiskBackendConfig,
    SampleFiles,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
)
from ovannotators.exceptions import PackagingError
from ovannotators.writers._helpers import OVRTX_BBOX2D_DTYPE
from tests.webdataset_matrix import AdapterCase, WebDatasetAdapterMatrix

pytestmark = [pytest.mark.integration, pytest.mark.cpu_only]


def _image(seed: int, height: int = 16, width: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _basic_frame(variant: int, frame: int, size: tuple[int, int] = (16, 24)):
    rng = np.random.default_rng(1000 * variant + frame)
    return {
        "LdrColor": _image(1000 * variant + frame, *size),
        "DistanceToCameraSD": rng.uniform(0.1, 50.0, size).astype(np.float32),
    }


def _check_basic(sample, values):
    assert np.array_equal(sample["ldr_color.png"], values["LdrColor"])
    assert np.array_equal(
        sample["distance_to_camera.npy"], values["DistanceToCameraSD"]
    )


def _boxes(labels: list[int]) -> np.ndarray:
    boxes = np.zeros(len(labels), dtype=OVRTX_BBOX2D_DTYPE)
    for index, label in enumerate(labels):
        boxes[index]["instance_id"] = (index + 1, 0, 0, 0)
        boxes[index]["semantic_id"] = (label, 0, 0, 0)
        boxes[index]["x_min"] = index
        boxes[index]["y_min"] = index
        boxes[index]["x_max"] = 8 + index
        boxes[index]["y_max"] = 6 + index
    return boxes


def _coco_frame(variant: int, frame: int, size: tuple[int, int] = (16, 24)):
    return {
        "LdrColor": _image(1000 * variant + frame, *size),
        "SemanticBoundingBox2DTight": _boxes([variant + 1]),
    }


def _check_coco(sample, values):
    assert np.array_equal(sample["png"], values["LdrColor"])
    record = sample["json"]
    assert record["image"]["file_name"] == sample["__key__"] + ".png"
    height, width = values["LdrColor"].shape[:2]
    assert (record["image"]["height"], record["image"]["width"]) == (height, width)
    boxes = values["SemanticBoundingBox2DTight"]
    assert len(record["annotations"]) == len(boxes)
    for annotation, box in zip(record["annotations"], boxes):
        assert annotation["image_id"] == record["image"]["id"]
        assert annotation["bbox"] == [
            float(box["x_min"]),
            float(box["y_min"]),
            float(box["x_max"] - box["x_min"]),
            float(box["y_max"] - box["y_min"]),
        ]
    names = {c["id"]: c["name"] for c in record["categories"]}
    assert [names[a["category_id"]] for a in record["annotations"]] == [
        str(int(box["semantic_id"][0])) for box in boxes
    ]


def _check_coco_negative(sample):
    assert sample["json"]["annotations"] == []
    assert sample["png"].shape == (16, 24, 3)


def _check_coco_manifest(manifest):
    categories = manifest["metadata"]["categories"]
    assert [(c["id"], c["name"]) for c in categories] == [(1, "1"), (2, "2"), (3, "3")]


class TestBasicFrameMatrix(WebDatasetAdapterMatrix):
    @pytest.fixture
    def case(self):
        return AdapterCase(
            name="basic",
            make_writer=lambda: BasicWriter(
                BasicWriterConfig(depth_format="npy", colorize=False)
            ),
            frame=_basic_frame,
            large_frame=lambda frame: _basic_frame(9, frame, (64, 64)),
            check_decoded=_check_basic,
            negative_frame={},
            negative_is_error=True,
            sample_schema="basic-frame-v1",
        )


class TestCOCOFrameMatrix(WebDatasetAdapterMatrix):
    @pytest.fixture
    def case(self):
        return AdapterCase(
            name="coco",
            make_writer=lambda: COCOWriter(COCOWriterConfig(image_format="png")),
            frame=_coco_frame,
            large_frame=lambda frame: _coco_frame(0, frame, (64, 64)),
            check_decoded=_check_coco,
            negative_frame={
                "LdrColor": _image(7),
                "SemanticBoundingBox2DTight": _boxes([]),
            },
            check_negative=_check_coco_negative,
            check_manifest=_check_coco_manifest,
            sample_schema="coco-instance-v1",
        )


@pytest.mark.parametrize(
    "portable, variant",
    [
        ("rgb.png", "RGB.png"),
        ("label.txt", "label.TXT"),
        ("k.npy", "\u212a.npy"),  # Kelvin sign folds to "k"
        ("fi.png", "\ufb01.png"),  # ligature normalizes to "fi"
        ("a.png", "\uff41.png"),  # full-width letter normalizes to "a"
        ("depth.npy", "depth\u200b.npy"),  # zero-width space
    ],
)
def test_packager_rejects_component_names_that_collide_after_folding(
    tmp_path, portable, variant
):
    source = tmp_path / "payload"
    source.write_bytes(b"x")
    packager = WebDatasetPackager(
        DiskBackend(DiskBackendConfig(root_dir=tmp_path / "out")),
        WebDatasetPackagerConfig(dataset_id="fold"),
    )
    with pytest.raises(PackagingError, match="Invalid or reserved component suffix"):
        packager.pack_episode(
            "a", [SampleFiles(0, {portable: source, variant: source})]
        )
    assert packager.is_open
    packager.pack_episode("a", [SampleFiles(0, {portable: source})])
    packager.finalize()
    assert (tmp_path / "out/fold/manifest.json").is_file()
