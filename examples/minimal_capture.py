# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal capture example: write 3 frames with ``BasicWriter``.

This example renders with ovrtx (https://github.com/NVIDIA-Omniverse/ovrtx)
and needs an NVIDIA RTX-capable GPU. Add the renderer to your ovannotators
environment, then run it:

    pip install ovrtx
    python examples/minimal_capture.py

ovannotators itself does not require a renderer — see the "Without ovrtx"
tutorial for a NumPy-only path that needs no GPU.
"""

import ovrtx

from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.renderer_outputs import OvrtxOutput
from ovannotators.writers import BasicWriter, BasicWriterConfig

scene_path = "tests/data/test_scene.usda"
out_dir = "_out/basic_minimal"
backend = DiskBackend(DiskBackendConfig(root_dir=out_dir, overwrite=True))
writer = BasicWriter(BasicWriterConfig(image_format="png"))
extractor = OvrtxOutput()
render_product = "/Render/RenderProduct_001"

renderer = ovrtx.Renderer()
renderer.open_usd(scene_path)

frame_id = 0
num_frames = 3
# The scene animates over USD timecodes 0..100 (see start/endTimeCode). Sample
# the timeline explicitly with update_from_usd_time — step()'s delta_time drives
# simulation time, not the USD animation timeline, so time-sampled transforms
# (e.g. the moving sphere) only advance when you set the USD time.
start_tc, end_tc = 0.0, 5.0
for i in range(num_frames):
    usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
    renderer.update_from_usd_time(usd_time)
    products = renderer.step(render_products={render_product}, delta_time=0.0)
    product = products[render_product]
    for frame in product.frames:
        # Retrieve and write all AOVs generated
        writer.write_frame(frame_id, extractor.extract_all(frame), backend)
        frame_id += 1

writer.finalize(backend)
print(f"Wrote {len(backend.list())} files to {out_dir}")
