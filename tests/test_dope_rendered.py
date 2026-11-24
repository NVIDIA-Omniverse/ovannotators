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

"""DOPEWriter against a real ovrtx render, including the OcclusionSD join.

The ``OcclusionSD`` ratios are keyed by renderer instance id while the semantic
boxes carry stable ids, so only a real render can show the join lands on the
right objects. Skipped when ovrtx / a GPU is unavailable.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ovannotators._internal.optional_deps import ovrtx_available
from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators.writers._helpers import occlusion_by_stable_id
from ovannotators.writers.dope import DOPEWriter
from tests.conftest import TEST_SCENE_USDA, render_scene_vars

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not ovrtx_available(),
        reason="ovrtx not installed — skipping rendered-scene writer tests",
    ),
]

_VARS = "</Render/RenderProduct_001/Vars/"
_OCCLUSION_VARS = (
    "OcclusionSD",
    "InstanceIdTokenMapSD",
    "InstanceMap",
    "StableIdSemanticIdMap",
)


def _scene_with_occlusion_vars() -> str:
    """Add the render vars DOPE visibility needs to the AllAnnotators product."""
    usda = TEST_SCENE_USDA.read_text()
    ordered = f"{_VARS}CameraParams>, \n"
    declare = '            def RenderVar "CameraParams"'
    for name in _OCCLUSION_VARS:
        if f"{_VARS}{name}>" not in usda:
            usda = usda.replace(ordered, f"{ordered}            {_VARS}{name}>, \n", 1)
        if f'def RenderVar "{name}"' not in usda:
            usda = usda.replace(
                declare,
                f'            def RenderVar "{name}"\n            {{\n'
                f'                uniform string sourceName = "{name}"\n'
                '                uniform token sourceType = "raw"\n'
                f"            }}\n\n{declare}",
                1,
            )
    return usda


@pytest.fixture(scope="module")
def render_vars() -> dict:
    return render_scene_vars(_scene_with_occlusion_vars(), "dope-occlusion")


@pytest.fixture(scope="module")
def dope_frame(render_vars, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("dope")
    backend = DiskBackend(DiskBackendConfig(root_dir=str(root), overwrite=True))
    writer = DOPEWriter()
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)
    assert (root / "000000.png").exists()
    return json.loads((root / "000000.json").read_text())


def test_occlusion_resolves_to_every_semantic_box(render_vars) -> None:
    ratios = occlusion_by_stable_id(render_vars)
    assert ratios is not None
    box_ids = {
        tuple(int(w) for w in box["instance_id"])
        for box in render_vars["SemanticBoundingBox3D"]
    }
    assert box_ids <= set(ratios)
    assert all(0.0 <= ratio <= 1.0 for ratio in ratios.values())


def test_visibility_reflects_real_occlusion(dope_frame, render_vars) -> None:
    objects = dope_frame["objects"]
    assert len(objects) == len(render_vars["SemanticBoundingBox3D"])
    visibilities = [obj["visibility"] for obj in objects]
    assert all(0.0 < v <= 1.0 for v in visibilities)
    # The scene has partially occluded objects; a failed join would leave all 1.0.
    assert min(visibilities) < 1.0


def test_poses_are_in_the_camera_frame(dope_frame) -> None:
    # The camera is 500*sqrt(3) from the origin and the objects are near it, so
    # every object origin is hundreds of units ahead, along -Z in the USD frame.
    for obj in dope_frame["objects"]:
        x, y, z = obj["location"]
        assert z < 0
        assert 700.0 < math.sqrt(x * x + y * y + z * z) < 1000.0
        assert math.isclose(
            sum(q * q for q in obj["quaternion_xyzw"]), 1.0, abs_tol=1e-6
        )


def test_cuboid_keypoints_are_projected_into_the_image(dope_frame) -> None:
    width, height = 1280, 720
    for obj in dope_frame["objects"]:
        cuboid = np.asarray(obj["projected_cuboid"])
        assert cuboid.shape == (9, 2)
        centre_x, centre_y = cuboid[8]
        assert 0 <= centre_x < width and 0 <= centre_y < height
