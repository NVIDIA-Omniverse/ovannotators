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

"""Requirements-driven capture: discover a profile's inputs, then render them.

This advanced example connects requirements discovery to the renderer:

1. ``list_profiles`` lists the tasks a writer supports.
2. ``describe_requirements`` returns the render variables, camera count, scene
   prerequisites, and resolved writer config for the chosen profile.
3. The example authors a USD layer over the bundled scene. The layer adds one
   RenderProduct for the scene camera with a RenderVar per requested output,
   and deactivates the scene's all-outputs product.
4. ovrtx renders the layer through an attached ovstage stage, and the writer
   built from ``requirements.config`` writes the dataset.

It needs an NVIDIA RTX-capable GPU, CUDA, ``ovrtx``, and ``ovstage``. Run it
from the repository root:

    python examples/requirements_capture.py --writer coco --profile instance_segmentation

Run ``python examples/requirements_capture.py --help`` for the other options.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

from ovannotators import (
    DiskBackend,
    DiskBackendConfig,
    OvrtxOutput,
    WriterRegistry,
    describe_requirements,
    list_profiles,
)

SCENE_PATH = Path("tests/data/test_scene.usda")
# USD reads stage metadata from the root layer only, so the generated layer
# repeats the scene's units and timeline.
SCENE_METADATA = """\
    metersPerUnit = 0.01
    upAxis = "Y"
    startTimeCode = 0
    endTimeCode = 100
    timeCodesPerSecond = 60"""
SCENE_PRODUCT = "/Render/RenderProduct_001"
CAMERA = "/Replicator/Camera_Xform/Camera"
RENDER_PRODUCT = "/Render/RequirementsProduct"


def render_product_layer(
    render_vars, scene_path=SCENE_PATH, camera=CAMERA, resolution=(1280, 720)
):
    """Return USDA text that renders only ``render_vars`` from ``camera``.

    Args:
        render_vars: ovrtx ``sourceName`` values to request, one RenderVar each.
        scene_path: Scene layer to compose under the generated layer.
        camera: Camera prim path that the RenderProduct renders from.
        resolution: Output ``(width, height)`` in pixels.

    Returns:
        A USDA layer for ``ovstage.population.open_usd_from_string``. Its
        sublayer path is absolute because a layer opened from a string has no
        directory to resolve a relative path against.
    """
    names = sorted(render_vars)
    product = RENDER_PRODUCT.rsplit("/", 1)[1]
    ordered = ",\n".join(
        f"            <{RENDER_PRODUCT}/Vars/{name}>" for name in names
    )
    variables = "\n".join(
        f'            def RenderVar "{name}"\n'
        "            {\n"
        f'                uniform string sourceName = "{name}"\n'
        '                uniform token sourceType = "raw"\n'
        "            }\n"
        for name in names
    )
    width, height = resolution
    return f"""#usda 1.0
(
{SCENE_METADATA}
    subLayers = [@{Path(scene_path).resolve().as_posix()}@]
)

over "Render"
{{
    over "Settings"
    {{
        rel products = [<{RENDER_PRODUCT}>]
    }}

    over "{SCENE_PRODUCT.rsplit("/", 1)[1]}" (
        active = false
    )
    {{
    }}

    def RenderProduct "{product}"
    {{
        rel camera = <{camera}>
        uniform int2 resolution = ({width}, {height})
        token omni:rtx:rendermode = "RealTimePathTracing"
        rel orderedVars = [
{ordered}
        ]

        def Scope "Vars"
        {{
{variables}        }}
    }}
}}
"""


def resolve_requirements(writer_name, profile):
    """Describe ``profile`` and reject inputs this single-camera example lacks.

    Args:
        writer_name: Frame-writer registry name, such as ``"coco"``.
        profile: Profile name from ``list_profiles``, or ``None`` for the
            recommended default.

    Returns:
        The ``DatasetRequirements`` for the selected profile.

    Raises:
        SystemExit: If the profile needs configuration, non-render inputs, or
            more than one camera.
    """
    profiles = list_profiles(writer_name)
    print(f"{writer_name} profiles: {', '.join(profiles)}")
    requirements = describe_requirements(writer_name, profile=profile)
    print(f"Selected profile: {requirements.profile}")
    print(f"Required render vars: {', '.join(sorted(requirements.render_vars))}")
    if requirements.recommended_render_vars:
        recommended = ", ".join(sorted(requirements.recommended_render_vars))
        print(f"Recommended render vars: {recommended}")
    for prerequisite in requirements.scene_prerequisites:
        print(f"Scene prerequisite: {prerequisite}")

    if requirements.config is None or requirements.required_options:
        raise SystemExit(
            f"{writer_name}/{requirements.profile} needs configuration: "
            f"{', '.join(requirements.required_options)}"
        )
    if requirements.non_render_inputs:
        raise SystemExit(
            f"{writer_name}/{requirements.profile} needs non-render inputs: "
            f"{'; '.join(requirements.non_render_inputs)}"
        )
    if requirements.minimum_camera_views != 1:
        raise SystemExit(
            f"{writer_name}/{requirements.profile} needs "
            f"{requirements.minimum_camera_views} camera views; this example "
            "renders one"
        )
    return requirements


def capture(renderer, stage, ordinal, writer, backend, num_frames):
    """Render ``num_frames`` frames of the populated stage into ``writer``.

    Step results are local to this function, so they are released before the
    caller releases the renderer.

    Args:
        renderer: ovrtx ``Renderer`` attached to ``stage``.
        stage: ovstage ``Stage`` already populated at ``ordinal``.
        ordinal: Last ordinal published to ``stage``.
        writer: Frame writer built from the requirements config.
        backend: Destination storage backend.
        num_frames: Frames to render across the first 5 seconds of USD time.
    """
    import ovstage

    extractor = OvrtxOutput()
    frame_id = 0
    start_tc, end_tc = 0.0, 5.0
    for i in range(num_frames):
        usd_time = start_tc + (end_tc - start_tc) * i / max(num_frames - 1, 1)
        ordinal += 1
        ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
        stage.advance_write_floor(ordinal).wait()
        # delta_time must be non-zero, or consecutive frames repeat the same state.
        outputs = renderer.step({RENDER_PRODUCT}, delta_time=0.1, ordinal=ordinal)
        for frame in outputs[RENDER_PRODUCT].frames:
            writer.write_frame(frame_id, extractor.extract_all(frame), backend)
            frame_id += 1


def main(argv=None):
    """Discover requirements, render the requested outputs, and write a dataset.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--writer", default="coco", help="frame writer name")
    parser.add_argument(
        "--profile", help="profile from list_profiles (default: recommended)"
    )
    parser.add_argument("--frames", type=int, default=3, help="frames to render")
    parser.add_argument(
        "--out", type=Path, default=Path("_out/requirements_capture"), help="root"
    )
    args = parser.parse_args(argv)

    requirements = resolve_requirements(args.writer, args.profile)
    requested = requirements.render_vars | requirements.recommended_render_vars

    import ovrtx
    import ovstage

    out_dir = args.out / f"{args.writer}-{requirements.profile}"
    backend = DiskBackend(DiskBackendConfig(root_dir=out_dir, overwrite=True))
    writer = WriterRegistry.create(args.writer, requirements.config)

    # Register schemas before any USD access; the Renderer must exist before
    # the Stage it borrows. ExitStack unwinds in reverse order (detach, stage,
    # renderer) and still runs the remaining steps if one of them raises.
    ovrtx.register_schema_paths()
    renderer = ovrtx.Renderer()
    with ExitStack() as cleanup:
        cleanup.callback(renderer.destroy)
        stage = ovstage.Stage("requirements-capture")
        cleanup.callback(stage.destroy)
        renderer.attach_ovstage(stage)
        cleanup.callback(renderer.detach_ovstage)

        ordinal = 1
        ovstage.population.open_usd_from_string(
            stage, render_product_layer(requested), ordinal=ordinal, time_code=0.0
        )
        stage.advance_write_floor(ordinal).wait()
        capture(renderer, stage, ordinal, writer, backend, args.frames)
        writer.finalize(backend)
    print(f"Wrote {len(backend.list())} files to {out_dir}")


if __name__ == "__main__":
    main()
