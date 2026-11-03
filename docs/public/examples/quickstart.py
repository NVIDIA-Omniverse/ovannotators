# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quick Start: render a few frames and write a COCO dataset.

This is the canonical capture flow used throughout the documentation: drive the
USD animation timeline with ``ovstage.population.update_from_usd_time``, advance
the sensor simulation with a positive ``step(delta_time=...)``, and hand each
frame to a writer via ``OvrtxOutput.extract_all``. The test suite executes it,
with only the output path and frame count adjusted, so it cannot drift from the
public API.

The renderer does not open USD itself. It borrows an externally owned
:class:`ovstage.Stage`, and ``ovstage.population`` drives what that stage
contains.
"""

import ovrtx
import ovstage

from ovannotators import (
    DiskBackend,
    DiskBackendConfig,
    OvrtxOutput,
    WriterRegistry,
)

# Set up the backend (where bytes are persisted).
backend = DiskBackend(DiskBackendConfig(root_dir="./output/coco", overwrite=True))

# Create a writer by registry name and a render-variable extractor.
writer = WriterRegistry.create("coco")
extractor = OvrtxOutput()

# Register schemas before any USD access; the Renderer must exist before the
# Stage it borrows.
ovrtx.register_schema_paths()
renderer = ovrtx.Renderer()
stage = ovstage.Stage("quickstart")
renderer.attach_ovstage(stage)
render_product = "/Render/RenderProduct_001"

# Use the first animation sample rather than USD Default time.
ordinal = 1
ovstage.population.open_usd(
    stage, "tests/data/test_scene.usda", ordinal=ordinal, time_code=0.0
)
stage.advance_write_floor(ordinal).wait()

# USD time selects scene samples; a positive delta_time advances the sensor
# simulation between frames.
frame_id = 0
num_frames = 3
start_tc, end_tc = 0.0, 5.0
for i in range(num_frames):
    usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
    ordinal += 1
    ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
    # Publish the stage update before rendering at this ordinal.
    stage.advance_write_floor(ordinal).wait()
    outputs = renderer.step({render_product}, delta_time=0.1, ordinal=ordinal)
    product = outputs[render_product]
    for frame in product.frames:
        writer.write_frame(frame_id, extractor.extract_all(frame), backend)
        frame_id += 1

# Flush deferred data (COCO JSON is written here).
writer.finalize(backend)
