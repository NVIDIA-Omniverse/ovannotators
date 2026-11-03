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

"""Tests for ``examples/requirements_capture.py``."""

import json
import re
import runpy
import sys
import types
from pathlib import Path

import pytest

from ovannotators import describe_requirements
from ovannotators._internal.optional_deps import ovrtx_available

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = runpy.run_path(
    str(REPO_ROOT / "examples" / "requirements_capture.py"), run_name="example"
)


def _ordered_var_names(layer: str) -> list[str]:
    block = re.search(r"rel orderedVars = \[(.*?)\]", layer, flags=re.DOTALL)
    return re.findall(r"/Vars/(\w+)>", block.group(1))


@pytest.mark.unit
@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("writer_name", "profile"),
    [("coco", "detection"), ("coco", "instance_segmentation"), ("kitti", "default")],
)
def test_layer_requests_exactly_the_described_render_vars(writer_name, profile):
    requirements = describe_requirements(writer_name, profile=profile)
    requested = requirements.render_vars | requirements.recommended_render_vars

    layer = EXAMPLE["render_product_layer"](requested)

    assert _ordered_var_names(layer) == sorted(requested)
    for name in requested:
        assert f'def RenderVar "{name}"' in layer
        assert f'uniform string sourceName = "{name}"' in layer
    assert f"rel camera = <{EXAMPLE['CAMERA']}>" in layer
    assert 'over "RenderProduct_001" (\n        active = false' in layer
    assert "timeCodesPerSecond = 60" in layer
    scene = (REPO_ROOT / "tests" / "data" / "test_scene.usda").as_posix()
    assert f"subLayers = [@{scene}@]" in layer


@pytest.mark.unit
@pytest.mark.cpu_only
def test_scene_metadata_matches_the_bundled_scene():
    scene = REPO_ROOT / "tests" / "data" / "test_scene.usda"
    # Git LFS leaves a small text pointer in place of the asset when a clone
    # does not fetch LFS objects, as on CI.
    if scene.read_bytes().startswith(b"version https://git-lfs"):
        pytest.skip(f"{scene.name} is an unfetched Git-LFS pointer")
    header = scene.read_text().split(")", 1)[0]
    for line in EXAMPLE["SCENE_METADATA"].splitlines():
        assert line.strip() in header


@pytest.mark.unit
@pytest.mark.cpu_only
def test_profiles_needing_configuration_are_rejected(capsys):
    with pytest.raises(SystemExit, match="needs configuration: output_base_uri"):
        EXAMPLE["resolve_requirements"]("cosmos_multiview", None)
    assert "cosmos_multiview profiles: default" in capsys.readouterr().out


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.skipif(not ovrtx_available(), reason="ovrtx not installed")
def test_coco_instance_segmentation_capture(monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
    out_root = REPO_ROOT / "_out" / "requirements_capture"

    EXAMPLE["main"](
        [
            "--writer",
            "coco",
            "--profile",
            "instance_segmentation",
            "--out",
            str(out_root),
        ]
    )

    out_dir = out_root / "coco-instance_segmentation"
    assert f"to {out_dir}" in capsys.readouterr().out
    assert len(sorted(out_dir.glob("images/*.jpg"))) == 3
    coco = json.loads((out_dir / "annotations" / "instances_train.json").read_text())
    assert coco["annotations"]
    assert all(item["segmentation"] for item in coco["annotations"])
    assert {item["name"] for item in coco["categories"]} <= {
        "cone",
        "cube",
        "sphere",
        "plane",
    }


@pytest.mark.parametrize("failure", ["attach", "capture"])
def test_renderer_and_stage_are_torn_down_when_capture_fails(
    monkeypatch, tmp_path, failure
):
    events = []

    class FakeStage:
        def __init__(self, name):
            events.append("stage")

        def advance_write_floor(self, ordinal):
            return types.SimpleNamespace(wait=lambda: None)

        def destroy(self):
            events.append("destroy-stage")

    class FakeRenderer:
        def attach_ovstage(self, stage):
            if failure == "attach":
                raise RuntimeError("injected attach failure")
            events.append("attach")

        def detach_ovstage(self):
            events.append("detach")

        def destroy(self):
            events.append("destroy-renderer")

    def open_usd_from_string(*args, **kwargs):
        raise RuntimeError("injected capture failure")

    ovrtx = types.ModuleType("ovrtx")
    ovrtx.register_schema_paths = lambda: None
    ovrtx.Renderer = FakeRenderer
    ovstage = types.ModuleType("ovstage")
    ovstage.Stage = FakeStage
    ovstage.population = types.SimpleNamespace(
        open_usd_from_string=open_usd_from_string
    )
    monkeypatch.setitem(sys.modules, "ovrtx", ovrtx)
    monkeypatch.setitem(sys.modules, "ovstage", ovstage)

    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        EXAMPLE["main"](["--out", str(tmp_path)])

    expected = {
        "attach": ["stage", "destroy-stage", "destroy-renderer"],
        "capture": ["stage", "attach", "detach", "destroy-stage", "destroy-renderer"],
    }
    assert events == expected[failure]
