# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quick Start: render a few frames and write a COCO dataset.

This is the canonical capture flow used throughout the documentation: drive the
USD animation timeline with ``update_from_usd_time``, advance the sensor
simulation with a non-zero ``step(delta_time=...)``, and hand each frame to a
writer via ``OvrtxOutput.extract_all``. It is executed verbatim by the test
suite so it cannot drift from the public API.
"""

import ovrtx

from ovannotators import (
    DiskBackend,
    DiskBackendConfig,
    OvrtxOutput,
    WriterRegistry,
)

# 1. Set up the backend (where bytes are persisted).
backend = DiskBackend(DiskBackendConfig(root_dir="./output/coco", overwrite=True))

# 2. Get a writer by name from the registry and an AOV extractor.
writer = WriterRegistry.get("coco")()
extractor = OvrtxOutput()

# 3. Open the scene and render. You control the loop:
#    - update_from_usd_time advances the USD animation timeline so time-sampled
#      transforms (moving objects, cameras) change between frames.
#    - step()'s delta_time advances the sensor simulation and must be non-zero,
#      otherwise consecutive frames sample identical stage state.
renderer = ovrtx.Renderer()
renderer.open_usd("tests/data/test_scene.usda")
render_product = "/Render/RenderProduct_001"

frame_id = 0
num_frames = 3
start_tc, end_tc = 0.0, 5.0
for i in range(num_frames):
    usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
    renderer.update_from_usd_time(usd_time)
    outputs = renderer.step({render_product}, delta_time=0.1)
    product = outputs[render_product]
    for frame in product.frames:
        writer.write_frame(frame_id, extractor.extract_all(frame), backend)
        frame_id += 1

# 4. Flush deferred data (COCO JSON is written here).
writer.finalize(backend)
