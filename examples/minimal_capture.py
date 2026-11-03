# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal capture example: render 3 frames with ovrtx and write them with ``BasicWriter``.

This example needs an NVIDIA RTX-capable GPU, CUDA, ``ovrtx``, and ``ovstage``.
The application owns an ``ovstage.Stage``; the ovrtx renderer borrows it.

    pip install 'ovrtx>=0.5,<0.6' 'ovstage>=0.2,<0.3' \
      --extra-index-url https://pypi.nvidia.com/
    python examples/minimal_capture.py

ovannotators itself does not require a renderer: ``examples/quickstart_cpu.py``
writes NumPy arrays without a GPU.
"""

from contextlib import ExitStack

import ovrtx
import ovstage
from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.renderer_outputs import OvrtxOutput
from ovannotators.writers import BasicWriter, BasicWriterConfig

scene_path = "tests/data/minimal_capture.usda"
out_dir = "_out/basic_minimal"
backend = DiskBackend(DiskBackendConfig(root_dir=out_dir, overwrite=True))
writer = BasicWriter(BasicWriterConfig(image_format="png"))
extractor = OvrtxOutput()
render_product = "/Render/RenderProduct_001"

# Register schemas before any USD access; the Renderer must exist before the
# Stage it borrows. ExitStack unwinds in reverse order (detach, stage, renderer)
# and still runs the remaining steps if one of them raises.
ovrtx.register_schema_paths()
renderer = ovrtx.Renderer()
with ExitStack() as cleanup:
    cleanup.callback(renderer.destroy)
    stage = ovstage.Stage("minimal-capture")
    cleanup.callback(stage.destroy)
    renderer.attach_ovstage(stage)
    cleanup.callback(renderer.detach_ovstage)

    outputs = frame = None
    try:
        ordinal = 1
        ovstage.population.open_usd(stage, scene_path, ordinal=ordinal, time_code=0.0)
        stage.advance_write_floor(ordinal).wait()

        frame_id = 0
        num_frames = 3
        # USD time selects scene samples; a positive delta_time advances the sensor
        # simulation between frames.
        start_tc, end_tc = 0.0, 5.0
        for i in range(num_frames):
            usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
            ordinal += 1
            ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
            stage.advance_write_floor(ordinal).wait()
            outputs = renderer.step({render_product}, delta_time=0.1, ordinal=ordinal)
            for frame in outputs[render_product].frames:
                writer.write_frame(frame_id, extractor.extract_all(frame), backend)
                frame_id += 1

        outputs = frame = None
        writer.finalize(backend)
    finally:
        # Release frame references before the renderer is torn down.
        outputs = frame = None
print(f"Wrote {len(backend.list())} files to {out_dir}")
