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

"""CenterPoseWriter against a real ovrtx render.

Skipped when ovrtx / a GPU is unavailable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ovannotators._internal.optional_deps import ovrtx_available
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers.centerpose import CenterPoseWriter
from tests.test_dope_rendered import render_vars  # noqa: F401 - pytest fixture

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not ovrtx_available(),
        reason="ovrtx not installed — skipping rendered-scene writer tests",
    ),
]


@pytest.fixture(scope="module")
def frame(render_vars, tmp_path_factory) -> dict:  # noqa: F811
    root = tmp_path_factory.mktemp("centerpose")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(root), overwrite=True))
    writer = CenterPoseWriter()
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)
    assert (root / "000000.png").exists()
    return json.loads((root / "000000.json").read_text())


def test_scale_matches_the_scene_objects(frame) -> None:
    # Every object in test_scene.usda is a 100-unit sphere, cone or cube.
    for obj in frame["objects"]:
        np.testing.assert_allclose(obj["scale"], [100.0, 100.0, 100.0], atol=0.1)


def test_camera_frame_keypoints_surround_the_object_origin(frame) -> None:
    assert frame["camera_data"]["width"] == 1280
    assert frame["camera_data"]["height"] == 720
    for obj in frame["objects"]:
        keypoints = np.asarray(obj["keypoints_3d"])
        assert keypoints.shape == (9, 3)
        # The cuboid is 100 units across, so the corners are within ~87 of the
        # centre keypoint, and all lie ahead of the camera (negative USD Z).
        centre, corners = keypoints[0], keypoints[1:]
        assert np.all(keypoints[:, 2] < 0)
        assert np.all(np.linalg.norm(corners - centre, axis=1) < 90.0)


def test_visibility_reflects_real_occlusion(frame) -> None:
    visibilities = [obj["visibility"] for obj in frame["objects"]]
    assert all(0.0 < v <= 1.0 for v in visibilities)
    assert min(visibilities) < 1.0
