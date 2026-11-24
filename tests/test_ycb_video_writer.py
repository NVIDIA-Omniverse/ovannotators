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

"""Tests for :class:`ovannotators.writers.ycb_video.YCBVideoWriter`."""

from __future__ import annotations

import numpy as np
import pytest

from ovannotators.exceptions import ConfigurationError, MissingRenderVarError
from ovannotators.writers import WriterRegistry
from ovannotators.writers._helpers import OVRTX_BBOX2D_DTYPE
from ovannotators.writers.ycb_video import YCBVideoWriter, YCBVideoWriterConfig
from tests.conftest import _make_semantic_id_map
from tests.test_dope_writer import HEIGHT, WIDTH, _box, _camera
from tests.test_mat5 import _read

VIDEO = "data/YCB_Video/data/0000"
CLASSES = {"can": 2, "box": 5}
# Camera 20 units from the origin with a 90 degree field of view on a 200x100
# image: fx = 100, principal point (100, 50). Scene units are centimetres.
METRES_PER_UNIT = 0.01


class _Backend:
    def __init__(self) -> None:
        self.written: dict[str, bytes] = {}
        self.arrays: dict[str, np.ndarray] = {}

    def write(self, path: str, data: bytes) -> None:
        self.written[path] = data

    def write_array(self, path: str, array, encoding: str = "png") -> None:
        self.arrays[path] = np.asarray(array)

    def makedirs(self, path: str) -> None:
        pass


def _boxes() -> np.ndarray:
    """Three 3D boxes: a can at the origin, a box at (5, 2, 0), and a table."""
    boxes = np.concatenate([_box(), _box(), _box()])
    for i, (instance, semantic, shift) in enumerate(
        [(1, 7, (0, 0, 0)), (2, 8, (5, 2, 0)), (3, 9, (0, 0, 0))]
    ):
        boxes[i]["instance_id"] = (instance, 0, 0, 0)
        boxes[i]["semantic_id"] = (semantic, 0, 0, 0)
        transform = np.eye(4)
        transform[3, :3] = shift
        boxes[i]["transform"] = transform
    return boxes


def _boxes_2d() -> np.ndarray:
    arr = np.zeros(3, dtype=OVRTX_BBOX2D_DTYPE)
    for i, (semantic, box) in enumerate(
        [(7, (95, 45, 105, 55)), (8, (120, 36, 130, 44)), (9, (0, 0, 199, 99))]
    ):
        arr[i]["instance_id"] = (i + 1, 0, 0, 0)
        arr[i]["semantic_id"] = (semantic, 0, 0, 0)
        arr[i]["x_min"], arr[i]["y_min"], arr[i]["x_max"], arr[i]["y_max"] = box
    return arr


def _frame(**overrides) -> dict:
    depth = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)
    depth[0, 0] = 7.0  # beyond the 6.5535 m range at factor_depth 10000
    depth[0, 1] = np.inf  # sky
    segmentation = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    segmentation[45:55, 95:105] = 7  # can
    segmentation[36:44, 120:130] = 8  # box
    segmentation[80:90, 10:20] = 9  # table: unmapped label
    camera = _camera()
    camera[0]["meters_per_scene_unit"] = METRES_PER_UNIT
    frame = {
        "LdrColor": np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8),
        "DistanceToImagePlaneSD": depth,
        "SemanticSegmentation": segmentation,
        "SemanticIdMap": _make_semantic_id_map({7: "can", 8: "box", 9: "table"}),
        "SemanticBoundingBox2DTight": _boxes_2d(),
        "SemanticBoundingBox3D": _boxes(),
        "CameraParams": camera,
    }
    frame.update(overrides)
    return frame


def _writer(**config) -> YCBVideoWriter:
    return YCBVideoWriter(
        YCBVideoWriterConfig(class_name_to_index_map=CLASSES, **config)
    )


def _write(frame: dict | None = None, frame_id: int = 3) -> _Backend:
    backend = _Backend()
    _writer().write_frame(frame_id, _frame() if frame is None else frame, backend)
    return backend


def test_registered_under_ycb_video() -> None:
    assert YCBVideoWriter.name == "ycb_video"
    assert WriterRegistry.get("ycb_video") is YCBVideoWriter
    assert YCBVideoWriter.required_render_vars == {
        "LdrColor",
        "DistanceToImagePlaneSD",
        "SemanticSegmentation",
        "SemanticIdMap",
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox3D",
        "CameraParams",
    }


@pytest.mark.parametrize(
    "config",
    [
        {"class_name_to_index_map": {}},
        {"class_name_to_index_map": {"can": 0}},
        {"class_name_to_index_map": {"can": 256}},
        {"class_name_to_index_map": {"can": 1}, "factor_depth": 0},
        {"class_name_to_index_map": {"can": 1}, "factor_depth": 70000},
    ],
)
def test_rejects_invalid_config(config) -> None:
    with pytest.raises(ConfigurationError):
        YCBVideoWriter(YCBVideoWriterConfig(**config))


def test_missing_required_var_raises() -> None:
    frame = _frame()
    del frame["DistanceToImagePlaneSD"]
    with pytest.raises(MissingRenderVarError):
        _writer().write_frame(0, frame, _Backend())


def test_rejects_non_pinhole_camera() -> None:
    with pytest.raises(ConfigurationError, match="pinhole"):
        _write(_frame(CameraParams=_camera(model=b"fisheyePolynomial")))


def test_writes_the_ycb_video_file_layout() -> None:
    backend = _write()
    assert sorted(backend.arrays) == [
        f"{VIDEO}/000003-color.png",
        f"{VIDEO}/000003-depth.png",
        f"{VIDEO}/000003-label.png",
    ]
    assert sorted(backend.written) == [
        f"{VIDEO}/000003-box.txt",
        f"{VIDEO}/000003-meta.mat",
    ]


def test_meta_poses_and_centers_are_in_the_opencv_camera_frame_in_metres() -> None:
    meta = _read(_write().written[f"{VIDEO}/000003-meta.mat"])
    assert meta["cls_indexes"].tolist() == [[2, 5]]  # the table is background
    assert meta["factor_depth"].tolist() == [[10000]]
    np.testing.assert_allclose(
        meta["intrinsic_matrix"], [[100, 0, 100], [0, 100, 50], [0, 0, 1]], atol=1e-6
    )
    assert meta["poses"].shape == (3, 4, 2)
    flip = np.diag([1.0, -1.0, -1.0])  # USD camera axes -> OpenCV
    np.testing.assert_allclose(meta["poses"][:, :3, 0], flip, atol=1e-6)
    # Origin is 20 scene units (0.2 m) ahead; the second object is 5 right, 2 up
    # in world space, which is 2 *up* = -2 in the OpenCV frame (+Y down).
    np.testing.assert_allclose(meta["poses"][:, 3, 0], [0, 0, 0.2], atol=1e-6)
    np.testing.assert_allclose(meta["poses"][:, 3, 1], [0.05, -0.02, 0.2], atol=1e-6)
    np.testing.assert_allclose(meta["center"], [[100, 50], [125, 40]], atol=1e-6)


def test_depth_is_16_bit_physical_depth() -> None:
    depth = _write().arrays[f"{VIDEO}/000003-depth.png"]
    assert depth.dtype == np.uint16
    assert depth[10, 10] == 50000  # 5 m * 10000
    assert depth[0, 0] == 65535  # clipped
    assert depth[0, 1] == 0  # no depth


def test_warns_once_when_depth_exceeds_the_factor_depth_range(caplog) -> None:
    writer = _writer()
    backend = _Backend()
    with caplog.at_level("WARNING", logger="ovannotators.writers.ycb_video"):
        writer.write_frame(0, _frame(), backend)  # frame has a 7 m pixel
        writer.write_frame(1, _frame(), backend)
    warnings = [r for r in caplog.records if "clipped" in r.getMessage()]
    assert len(warnings) == 1
    assert "6.5535" in warnings[0].getMessage()


def test_no_warning_when_depth_fits(caplog) -> None:
    frame = _frame()
    frame["DistanceToImagePlaneSD"] = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)
    with caplog.at_level("WARNING", logger="ovannotators.writers.ycb_video"):
        _write(frame)
    assert not caplog.records


def test_label_image_holds_class_indexes_and_zero_for_background() -> None:
    label = _write().arrays[f"{VIDEO}/000003-label.png"]
    assert label.dtype == np.uint8
    assert label[50, 100] == 2  # can
    assert label[40, 125] == 5  # box
    assert label[85, 15] == 0  # table is unmapped
    assert label[0, 0] == 0


def test_box_file_lists_mapped_objects_with_corner_coordinates() -> None:
    text = _write().written[f"{VIDEO}/000003-box.txt"].decode()
    assert text == "can 95 45 105 55\nbox 120 36 130 44\n"


def test_object_behind_the_camera_is_skipped() -> None:
    behind = _camera(translation=(0.0, 0.0, 20.0))
    behind[0]["meters_per_scene_unit"] = METRES_PER_UNIT
    assert _write(_frame(CameraParams=behind)).arrays == {}


def test_frame_without_a_mapped_object_is_not_written_or_listed() -> None:
    writer = _writer()
    backend = _Backend()
    only_table = _frame()
    only_table["SemanticBoundingBox3D"] = _boxes()[2:]
    writer.write_frame(1, only_table, backend)
    assert backend.arrays == {} and backend.written == {}
    writer.write_frame(4, _frame(), backend)
    writer.write_frame(2, _frame(), backend)
    writer.finalize(backend)
    writer.finalize(backend)
    assert backend.written["data/YCB_Video/train.txt"] == b"0000/000002\n0000/000004\n"


def test_video_id_names_the_directory_and_train_list() -> None:
    writer = _writer(video_id="0007")
    backend = _Backend()
    writer.write_frame(0, _frame(), backend)
    writer.finalize(backend)
    assert "data/YCB_Video/data/0007/000000-meta.mat" in backend.written
    assert backend.written["data/YCB_Video/train.txt"] == b"0007/000000\n"


def test_write_model_points_writes_points_xyz() -> None:
    backend = _Backend()
    _writer().write_model_points(
        "002_master_chef_can", np.array([[0.0, 0.5, -1.0], [1.25, 2.0, 3.0]]), backend
    )
    assert backend.written["data/models/002_master_chef_can/points.xyz"] == (
        b"0.000000 0.500000 -1.000000\n1.250000 2.000000 3.000000\n"
    )


@pytest.mark.parametrize("name", ["", "../x", "a/b", "a\\b", ".."])
def test_write_model_points_rejects_unsafe_names(name) -> None:
    with pytest.raises(ValueError, match="model_name"):
        _writer().write_model_points(name, np.zeros((1, 3)), _Backend())


def test_write_model_points_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        _writer().write_model_points("a", np.zeros((4, 2)), _Backend())
