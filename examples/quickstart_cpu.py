# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU quick start: write renderer-style arrays as a dataset, without a GPU.

Writers take a mapping of render-variable names to arrays, so any producer
works, not only a renderer. This example builds three frames of color, depth,
and normals with NumPy and writes them with ``BasicWriter``:

    python examples/quickstart_cpu.py
"""

import numpy as np
from ovannotators import DiskBackend, DiskBackendConfig, WriterRegistry

out_dir = "_out/quickstart_cpu"
backend = DiskBackend(DiskBackendConfig(root_dir=out_dir, overwrite=True))
writer = WriterRegistry.create("basic")

rng = np.random.default_rng(0)
for frame_id in range(3):
    render_vars = {
        "LdrColor": rng.integers(0, 256, (48, 64, 4), dtype=np.uint8),
        "DistanceToCameraSD": rng.uniform(0.5, 20.0, (48, 64)).astype(np.float32),
        "NormalSD": rng.uniform(-1.0, 1.0, (48, 64, 4)).astype(np.float32),
    }
    writer.write_frame(frame_id, render_vars, backend)

writer.finalize(backend)
print(f"Wrote {len(backend.list())} files to {out_dir}")
