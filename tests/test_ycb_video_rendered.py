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

"""YCBVideoWriter against a real ovrtx render.

The scene is authored in centimetres, so these tests check the metre conversion
of poses against the metre-valued depth AOV. Skipped when ovrtx / a GPU is
unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ovannotators._internal.optional_deps import ovrtx_available
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers.ycb_video import YCBVideoWriter, YCBVideoWriterConfig
from tests.test_dope_rendered import render_vars  # noqa: F401 - pytest fixture
from tests.test_mat5 import _read

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not ovrtx_available(),
        reason="ovrtx not installed — skipping rendered-scene writer tests",
    ),
]

CLASSES = {"sphere": 1, "cone": 2, "cube": 3}  # the plane is background
FACTOR_DEPTH = 5000


@pytest.fixture(scope="module")
def dataset(render_vars, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("ycb")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(root), overwrite=True))
    # The camera is 8.66 m from the origin, beyond the default factor_depth's
    # 6.5535 m range, so use a smaller factor (range 13.1 m).
    writer = YCBVideoWriter(
        YCBVideoWriterConfig(class_name_to_index_map=CLASSES, factor_depth=FACTOR_DEPTH)
    )
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)
    video = root / "data/YCB_Video/data/0000"
    meta = _read((video / "000000-meta.mat").read_bytes())
    return video, meta, root


def test_writes_every_file_and_the_train_list(dataset) -> None:
    video, _, root = dataset
    for suffix in ("color.png", "depth.png", "label.png", "box.txt", "meta.mat"):
        assert (video / f"000000-{suffix}").exists()
    assert (root / "data/YCB_Video/train.txt").read_text() == "0000/000000\n"


def test_objects_are_the_three_labelled_shapes(dataset) -> None:
    _, meta, _ = dataset
    assert sorted(meta["cls_indexes"].ravel().tolist()) == [1, 2, 3, 3]
    assert meta["poses"].shape == (3, 4, 4)
    assert meta["center"].shape == (4, 2)


def test_pose_is_in_metres_and_agrees_with_the_depth_image(dataset) -> None:
    video, meta, _ = dataset
    # The cone's origin is at the world origin: 500*sqrt(3) cm = 8.66 m from the
    # camera, straight ahead (+Z in the OpenCV frame).
    cone = meta["cls_indexes"].ravel().tolist().index(2)
    np.testing.assert_allclose(meta["poses"][:, 3, cone], [0, 0, 8.660], atol=0.01)
    np.testing.assert_allclose(meta["center"][cone], [640, 360], atol=2.0)
    assert meta["factor_depth"].tolist() == [[FACTOR_DEPTH]]
    # Depth is stored as metres * factor_depth. The cone's near surface is a
    # little closer than its origin.
    depth = np.array(Image.open(video / "000000-depth.png"))
    assert depth.dtype == np.uint16
    surface = depth[360, 640] / FACTOR_DEPTH
    assert 0.0 < meta["poses"][2, 3, cone] - surface < 1.0


def test_label_image_marks_each_object_with_its_class_index(dataset) -> None:
    video, meta, _ = dataset
    label = np.array(Image.open(video / "000000-label.png"))
    cone = meta["cls_indexes"].ravel().tolist().index(2)
    cx, cy = (int(round(v)) for v in meta["center"][cone])
    assert label[cy, cx] == 2
    assert set(np.unique(label).tolist()) <= {0, 1, 2, 3}
    assert label[0, 0] == 0  # sky / background
