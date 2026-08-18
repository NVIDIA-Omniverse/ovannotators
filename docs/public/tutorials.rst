.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Tutorials
=========

Step-by-step guides for common ``ovannotators`` workflows.

.. contents:: Contents
   :local:
   :depth: 1


Tutorial 1: Basic COCO Dataset from ovrtx
------------------------------------------

Extract 2D detection data from a rendered scene and produce a COCO-format dataset.

**Prerequisites:** the ``coco`` writer needs no extra dependencies — the core
install covers it. ``ovrtx`` is only needed because this example renders the
scene (it requires an NVIDIA GPU + CUDA).

.. code-block:: bash

   pip install ovannotators   # core: numpy, Pillow, openexr, warp-lang
   pip install ovrtx          # renderer used in this example (GPU + CUDA)

**Code:**

.. code-block:: python

   import ovrtx
   from ovannotators import (
       DiskBackend, DiskBackendConfig, WriterRegistry, OvrtxOutput
   )

   # Set up backend
   backend = DiskBackend(DiskBackendConfig(
       root_dir="./output/coco_dataset",
       overwrite=True,
   ))

   # Get writer
   COCOWriter = WriterRegistry.get("coco")
   writer = COCOWriter()  # uses COCOWriterConfig defaults (bbox_mode="tight")

   # Set up renderer + AOV extractor
   renderer = ovrtx.Renderer()
   renderer.open_usd("tests/data/test_scene.usda")
   render_products = {"/Render/RenderProduct_001"}
   extractor = OvrtxOutput()

   # Render loop. update_from_usd_time sets the USD animation time so
   # time-sampled transforms (moving objects, cameras) advance between frames.
   # step()'s delta_time advances the sensor simulation and MUST be non-zero:
   # a zero delta makes the step span an empty time interval, so the updated
   # stage state is never sampled and consecutive frames come out identical.
   # Sweep the scene's authored start/endTimeCode.
   frame_id = 0
   num_frames = 3
   start_tc, end_tc = 0.0, 5.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       renderer.update_from_usd_time(usd_time)
       outputs = renderer.step(render_products, delta_time=0.1)
       product = outputs["/Render/RenderProduct_001"]
       for frame in product.frames:
           writer.write_frame(frame_id, extractor.extract_all(frame), backend)
           frame_id += 1

   # Flush accumulated annotations JSON
   writer.finalize(backend)

**Output:**

::

   ./output/coco_dataset/
   ├── images/
   │   ├── 000000.jpg
   │   ├── 000001.jpg
   │   └── ...
   └── annotations/
       └── instances_train.json


Tutorial 2: Multiple Writers, One Render Loop
----------------------------------------------

Write COCO + KITTI + YOLO simultaneously from the same render pass.

**Prerequisites:** ``coco``, ``kitti``, and ``yolo`` are all core writers — no
extra dependencies beyond the core install. ``ovrtx`` (GPU + CUDA) supplies the
renderer, as in Tutorial 1.

.. code-block:: python

   import ovrtx
   from ovannotators import (
       DiskBackend, DiskBackendConfig, WriterRegistry, OvrtxOutput
   )

   # Create one backend per format
   backends = {
       "coco": DiskBackend(DiskBackendConfig(root_dir="./out/coco", overwrite=True)),
       "kitti": DiskBackend(DiskBackendConfig(root_dir="./out/kitti", overwrite=True)),
       "yolo": DiskBackend(DiskBackendConfig(root_dir="./out/yolo", overwrite=True)),
   }

   # Instantiate writers
   writers = {
       "coco": WriterRegistry.get("coco")(),
       "kitti": WriterRegistry.get("kitti")(),
       "yolo": WriterRegistry.get("yolo")(),
   }

   # Render. update_from_usd_time drives the USD animation timeline; step()'s
   # delta_time advances the sensor simulation and must be non-zero so the
   # updated stage state is actually sampled into each frame.
   renderer = ovrtx.Renderer()
   renderer.open_usd("tests/data/test_scene.usda")
   extractor = OvrtxOutput()

   frame_id = 0
   num_frames = 3
   start_tc, end_tc = 0.0, 5.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       renderer.update_from_usd_time(usd_time)
       outputs = renderer.step({"/Render/RenderProduct_001"}, delta_time=0.1)
       product = outputs["/Render/RenderProduct_001"]
       for frame in product.frames:
           render_vars = extractor.extract_all(frame)
           for fmt, writer in writers.items():
               writer.write_frame(frame_id, render_vars, backends[fmt])
           frame_id += 1

   # Finalize all
   for fmt, writer in writers.items():
       writer.finalize(backends[fmt])


Tutorial 3: Writing to S3
--------------------------

Store datasets directly in an S3 bucket.

**Prerequisites:** the writer itself (``coco`` here) needs no extras; the
``S3Backend`` requires ``boto3``, which the ``[s3]`` extra installs.

.. code-block:: bash

   pip install "ovannotators[s3]"   # adds boto3 for the S3 backend

.. code-block:: python

   from ovannotators import S3Backend, S3BackendConfig, WriterRegistry

   backend = S3Backend(S3BackendConfig(
       bucket="my-ml-datasets",
       prefix="synthetic/coco/run-042/",
       region="us-west-2",
   ))

   writer = WriterRegistry.get("coco")()

   # ... render loop (same pattern as Tutorial 1) ...

   writer.finalize(backend)
   # Data is now at s3://my-ml-datasets/synthetic/coco/run-042/


Tutorial 4: LeRobot Dataset for Robot Learning
-----------------------------------------------

Write episodes in HuggingFace LeRobot format.

**Prerequisites:** the ``lerobot`` writer requires ``pyarrow`` (Parquet output)
and ``PyNvVideoCodec`` (NVENC video encoding, needs an NVIDIA GPU + CUDA),
provided by the ``[lerobot]`` and ``[video]`` extras. ``ovrtx`` supplies the
renderer.

.. code-block:: bash

   pip install "ovannotators[lerobot,video]"  # pyarrow + PyNvVideoCodec (NVENC, needs CUDA)
   pip install ovrtx                           # renderer used in this example

.. code-block:: python

   import ovrtx
   from ovannotators import (
       DiskBackend, DiskBackendConfig, WriterRegistry, OvrtxOutput
   )
   from ovannotators.writers.lerobot import LeRobotWriterConfig

   backend = DiskBackend(DiskBackendConfig(root_dir="./lerobot_data", overwrite=True))

   writer = WriterRegistry.get("lerobot")(LeRobotWriterConfig(
       fps=30,
       episode_index=0,
       task_description="Catch the glowing orb",
       camera_keys=["static_cam"],
   ))

   # Render the episode. The LeRobot writer accumulates each frame's LdrColor
   # and, on finalize(), encodes one mp4 per camera key plus the parquet rows.
   # As in Tutorial 1, update_from_usd_time drives the USD animation timeline;
   # step()'s delta_time advances the sensor simulation and must be non-zero so
   # the updated stage state is sampled into each frame.
   renderer = ovrtx.Renderer()
   renderer.open_usd("tests/data/test_scene.usda")
   extractor = OvrtxOutput()

   frame_id = 0
   num_frames = 30
   start_tc, end_tc = 0.0, 5.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       renderer.update_from_usd_time(usd_time)
       outputs = renderer.step({"/Render/RenderProduct_001"}, delta_time=0.1)
       product = outputs["/Render/RenderProduct_001"]
       for frame in product.frames:
           writer.write_frame(frame_id, extractor.extract_all(frame), backend)
           frame_id += 1

   # Encode videos + parquet + meta JSON
   writer.finalize(backend)

**Output:**

::

   ./lerobot_data/
   ├── data/chunk-000/episode_000000.parquet
   ├── videos/chunk-000/observation.images.static_cam/episode_000000.mp4
   └── meta/{info,tasks,episodes,stats}.json


Tutorial 5: Cosmos Transfer Training Data
------------------------------------------

Generate conditioning video data for NVIDIA Cosmos Transfer. The
``cosmos_transfer`` writer accumulates frames and, on ``finalize()``, encodes an
RGB video plus one conditioning video per requested signal (Canny edges,
colorized depth, colorized segmentation, normal-shaded segmentation) and a
``manifest.json``.

The writer emits the **Cosmos Transfer 2.5** training format (the manifest is
tagged ``cosmos-transfer-2.5``). The same conditioning videos are also valid
control signals for **Cosmos 3** ``video2video`` transfer — see
:ref:`Driving Cosmos 3 transfer <cosmos3-transfer>` below, which converts a clip
straight into a Cosmos 3 inference run. So one capture serves both: use the
manifest directly for Cosmos Transfer 2.5 fine-tuning, or bridge any clip into
Cosmos 3.

**Install:**

.. code-block:: bash

   pip install "ovannotators[video]"  # PyNvVideoCodec for NVENC (required)
   pip install "ovannotators[gpu]"    # optional: nvImageCodec for faster image I/O
   pip install ovrtx                  # renderer used in the capture example below
   # Warp (GPU Canny, with NumPy fallback) ships in the core install.

The ``cosmos_transfer`` writer **requires** ``PyNvVideoCodec`` (``[video]``) for
NVENC encoding and therefore an NVIDIA GPU + CUDA; the ``[gpu]`` image codec is
optional.

**Required render vars:** ``LdrColor``, ``DistanceToCameraSD``, and
``SemanticInstanceSegmentation``. ``NormalSD`` is additionally required when
``conditioning_types`` includes ``"shaded_segmentation"`` or when ``canny_source``
is ``"normals"`` / ``"shaded_segmentation"``.

**Configuration:**

.. code-block:: python

   from ovannotators.writers.cosmos_transfer import CosmosTransferWriterConfig

   config = CosmosTransferWriterConfig(
       fps=24,
       # First clip's id; names output files 000000_* and advances per clip.
       clip_index=0,
       # Which conditioning videos to emit (default: canny, depth, segmentation).
       conditioning_types=["canny", "depth", "segmentation", "shaded_segmentation"],
       # Render var the Canny edges are derived from: "rgb" (default),
       # "normals", "segmentation", or "shaded_segmentation".
       canny_source="shaded_segmentation",
   )

The output resolution is taken from the incoming frames (the render settings
decide it), and the generation prompt is supplied later at transfer time — not
on the writer config.

**Capture (one or more clips):**

Accumulate frames with ``write_frame()``, then call ``cut(backend)`` to close
the buffered frames into a clip and start a new one; the ``clip_index`` advances
automatically. Use it to slice a long capture into sequential clips, or to emit
one clip per scene variation (modify the scene, render more frames, then
``cut`` again). ``cut()`` is a no-op when no frames are buffered and returns the
written ``clip_id`` (or ``None``). ``finalize()`` flushes the last buffered clip
and writes the manifest. For a single clip, just skip ``cut()`` and call
``finalize()`` once.

.. code-block:: python

   import ovrtx
   from ovannotators import (
       DiskBackend, DiskBackendConfig, WriterRegistry, OvrtxOutput
   )

   backend = DiskBackend(DiskBackendConfig(root_dir="./cosmos_data", overwrite=True))
   writer = WriterRegistry.get("cosmos_transfer")(config)

   renderer = ovrtx.Renderer()
   renderer.open_usd("tests/data/test_scene.usda")
   extractor = OvrtxOutput()

   num_frames = 24                           # frames per clip (≈1s at fps=24)
   start_tc, end_tc = 0.0, 5.0
   for variation in scene_variations:        # one clip per scene variation
       variation.apply(renderer)             # e.g. swap materials / lighting / poses
       frame_id = 0
       for i in range(num_frames):
           # Sweep the USD animation timeline explicitly; step()'s delta_time
           # advances the sensor simulation and must be non-zero so the updated
           # stage state is sampled into each frame.
           usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
           renderer.update_from_usd_time(usd_time)
           outputs = renderer.step({"/Render/RenderProduct_001"}, delta_time=0.1)
           product = outputs["/Render/RenderProduct_001"]
           for frame in product.frames:
               writer.write_frame(frame_id, extractor.extract_all(frame), backend)
               frame_id += 1
       writer.cut(backend)                   # close clip 000000, 000001, ...

   writer.finalize(backend)                  # flush the final buffered clip, if any

**Output:**

::

   ./cosmos_data/
   ├── videos/
   │   └── 000000_rgb.mp4
   ├── conditioning/
   │   ├── 000000_canny.mp4
   │   ├── 000000_depth.mp4
   │   ├── 000000_segmentation.mp4
   │   └── 000000_shaded_segmentation.mp4
   ├── thumbnails/
   │   └── 000000.jpg
   └── manifest.json          # one entry per clip

Each additional clip (one per ``cut()`` / scene variation) adds a parallel set
of files with the next ``clip_index``: ``000001_rgb.mp4``, ``000001_*`` in
``conditioning/``, ``000001.jpg``, and another ``manifest.json`` entry.

The writer emits the RGB clip plus one conditioning video per requested signal.
These are the inputs you feed to Cosmos transfer:

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; gap: 12px; margin: 1em 0;">
     <figure style="flex: 1 1 200px; max-width: 300px; margin: 0;">
       <video autoplay loop muted playsinline style="width: 100%; height: auto; border-radius: 6px;">
         <source src="_static/media/cosmos3_input_rgb.mp4" type="video/mp4">
       </video>
       <figcaption style="text-align: center; font-size: 0.85em;">RGB (<code>000000_rgb.mp4</code>)</figcaption>
     </figure>
     <figure style="flex: 1 1 200px; max-width: 300px; margin: 0;">
       <video autoplay loop muted playsinline style="width: 100%; height: auto; border-radius: 6px;">
         <source src="_static/media/cosmos3_input_canny.mp4" type="video/mp4">
       </video>
       <figcaption style="text-align: center; font-size: 0.85em;">Canny (<code>000000_canny.mp4</code>)</figcaption>
     </figure>
     <figure style="flex: 1 1 200px; max-width: 300px; margin: 0;">
       <video autoplay loop muted playsinline style="width: 100%; height: auto; border-radius: 6px;">
         <source src="_static/media/cosmos3_input_depth.mp4" type="video/mp4">
       </video>
       <figcaption style="text-align: center; font-size: 0.85em;">Depth (<code>000000_depth.mp4</code>)</figcaption>
     </figure>
     <figure style="flex: 1 1 200px; max-width: 300px; margin: 0;">
       <video autoplay loop muted playsinline style="width: 100%; height: auto; border-radius: 6px;">
         <source src="_static/media/cosmos3_input_segmentation.mp4" type="video/mp4">
       </video>
       <figcaption style="text-align: center; font-size: 0.85em;">Segmentation (<code>000000_segmentation.mp4</code>)</figcaption>
     </figure>
     <figure style="flex: 1 1 200px; max-width: 300px; margin: 0;">
       <video autoplay loop muted playsinline style="width: 100%; height: auto; border-radius: 6px;">
         <source src="_static/media/cosmos3_input_shaded_segmentation.mp4" type="video/mp4">
       </video>
       <figcaption style="text-align: center; font-size: 0.85em;">Shaded segmentation (<code>000000_shaded_segmentation.mp4</code>)</figcaption>
     </figure>
   </div>

.. _cosmos3-transfer:

**Driving Cosmos 3 transfer:**

The conditioning videos are control signals for Cosmos 3 ``video2video``
transfer. The text prompt is provided here, at transfer time. The bridge reads
the manifest and produces a spec + inference command (running it needs the
Cosmos Framework, a Cosmos 3 checkpoint such as ``Cosmos3-Nano``, and a GPU).

.. code-block:: python

   from ovannotators.integrations.cosmos3_transfer import (
       prepare_transfer, build_inference_command,
   )

   spec_path, spec = prepare_transfer(
       "./cosmos_data",                  # writer output dir (has manifest.json)
       "./transfer_specs",               # where the spec JSON is written
       conditioning="depth",             # which conditioning video to control on
       clip_index=0,                     # which manifest clip to transfer
       prompt=(
           "A warmly glowing paper lantern rotates around some beautiful mossy "
           "rocks in a wavy ocean. The scene is warm and beautifully lit by the "
           "lantern in the late evening and stars can be seen reflected on the "
           "water's surface"
       ),
       control_guidance=1.0,             # strength of the control signal
   )
   cmd = build_inference_command(spec_path, "./transfer_out", checkpoint="Cosmos3-Nano")
   # subprocess.run(cmd, check=True)  # requires Cosmos Framework + GPU

Running the command above with the depth conditioning and the lantern prompt
produces a transferred clip like this (depth control, ``control_guidance=1.0``):

.. raw:: html

   <video controls autoplay loop muted playsinline width="640"
          poster="_static/media/cosmos3_transfer_depth_poster.jpg"
          style="max-width: 100%; height: auto; border-radius: 6px;">
     <source src="_static/media/cosmos3_transfer_depth.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>


Tutorial 6: Custom Writer
--------------------------

Create a writer for your own dataset format.

**Prerequisites:** none beyond the core install — this example uses only the
standard library and ``numpy`` (a core dependency).

.. code-block:: python

   import json
   import numpy as np
   from ovannotators import WriterRegistry
   from ovannotators.backends._base import Backend


   @WriterRegistry.register
   class DepthStatsWriter:
       """Writes per-frame depth statistics as JSON lines."""

       name = "depth_stats"
       required_render_vars = frozenset({"depth"})

       def __init__(self):
           self._records = []
           self._finalized = False

       def write_frame(self, frame_id, render_vars, backend):
           depth = render_vars["depth"]
           valid = depth[depth > 0]

           self._records.append({
               "frame_id": frame_id,
               "min_depth": float(valid.min()) if valid.size > 0 else None,
               "max_depth": float(valid.max()) if valid.size > 0 else None,
               "mean_depth": float(valid.mean()) if valid.size > 0 else None,
               "coverage": float((depth > 0).sum() / depth.size),
           })

       def finalize(self, backend):
           if self._finalized:
               return
           lines = "\n".join(json.dumps(r) for r in self._records)
           backend.write("depth_stats.jsonl", lines.encode())
           self._finalized = True


Tutorial 7: Using OvrtxOutput Helper
--------------------------------------

``OvrtxOutput`` is the recommended way to turn an ovrtx ``FrameOutput`` into the
``render_vars`` dict writers expect — it handles the DLPack mapping and returns
zero-copy views (Warp arrays, or decoded structured arrays for annotator
buffers) for every AOV, so you don't map render vars by hand.
``extract_all(frame)`` returns the full dict; ``extract(frame, name)`` pulls a
single AOV when you only need a subset. Views stay on the GPU by default
(``device="cuda"``), where ovrtx rendered them; pass ``device="cpu"`` to copy
them to host memory instead.

**Prerequisites:** the ``basic`` writer is core (no extras). ``ovrtx`` (GPU +
CUDA) is required here because ``OvrtxOutput`` consumes ovrtx frame output.

.. code-block:: python

   import ovrtx
   from ovannotators import OvrtxOutput, DiskBackend, DiskBackendConfig, WriterRegistry

   extractor = OvrtxOutput()
   backend = DiskBackend(DiskBackendConfig(root_dir="./output", overwrite=True))
   writer = WriterRegistry.get("basic")()

   renderer = ovrtx.Renderer()
   renderer.open_usd("tests/data/test_scene.usda")

   # Advance the USD animation timeline with update_from_usd_time; step()'s
   # delta_time advances the sensor simulation and must be non-zero so the
   # updated stage state is sampled into each frame.
   frame_id = 0
   num_frames = 50
   start_tc, end_tc = 0.0, 100.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       renderer.update_from_usd_time(usd_time)
       outputs = renderer.step({"/Render/RenderProduct_001"}, delta_time=0.1)
       product = outputs["/Render/RenderProduct_001"]
       for frame in product.frames:
           # All AOVs at once (recommended):
           writer.write_frame(frame_id, extractor.extract_all(frame), backend)
           frame_id += 1

           # ...or pull only the ones you need:
           # render_vars = {n: extractor.extract(frame, n) for n in ("LdrColor",)}
           # writer.write_frame(frame_id, render_vars, backend)

   writer.finalize(backend)


Tutorial 8: Without ovrtx (Numpy Arrays Directly)
---------------------------------------------------

``ovannotators`` works with any numpy arrays — no renderer required.

**Prerequisites:** none beyond the core install — no ``ovrtx`` and no extras,
since the ``basic`` writer and ``DiskBackend`` are both core.

.. code-block:: python

   import numpy as np
   from ovannotators import DiskBackend, DiskBackendConfig, WriterRegistry

   backend = DiskBackend(DiskBackendConfig(root_dir="./output", overwrite=True))
   writer = WriterRegistry.get("basic")()

   # Simulate render output with random data
   rng = np.random.default_rng(42)

   for frame_id in range(10):
       render_vars = {
           "LdrColor": rng.integers(0, 256, (480, 640, 4), dtype=np.uint8),
           "depth": rng.uniform(0.5, 20.0, (480, 640)).astype(np.float32),
           "normals": rng.uniform(-1, 1, (480, 640, 3)).astype(np.float32),
       }
       writer.write_frame(frame_id, render_vars, backend)

   writer.finalize(backend)

This is useful for testing, converting existing datasets, or integrating with non-ovrtx renderers.


Tutorial 9: Faster Capture with ovrtx (Async Pipelining)
---------------------------------------------------------

The render loops above are *serial*: each iteration renders a frame, then
extracts and writes it while the GPU sits idle. Because rendering happens on the
GPU and writing happens on the CPU, you can overlap them and let the GPU render
the next frame while the CPU writes the current one.

ovrtx ``step()`` is shorthand for ``step_async().wait().fetch()`` — it blocks
immediately. ``step_async()`` instead returns right away after only *queuing* the
GPU work; you collect the result later with ``.wait().fetch()``. The trick is to
queue the **next** render before collecting (and writing) the **previous** one,
so the two run at the same time::

   serial:    [render 0][write 0][render 1][write 1]   (GPU idle during writes)
   pipelined: [render 0][render 1][render 2]...        (GPU never idle)
                        [write 0] [write 1] ...        (CPU overlaps render)

Keep a small, fixed number of renders "in flight" (a FIFO queue of pending
results). ``depth=2`` is the sweet spot — enough to hide a write behind the next
render; higher rarely helps once the loop is GPU-bound, and keeping *every* frame
in flight would pin all their GPU buffers at once (risking out-of-memory).

**Prerequisites:** the ``basic`` writer is core (no extras). ``ovrtx`` (GPU +
CUDA) supplies the renderer.

.. code-block:: python

   from collections import deque

   import ovrtx
   from ovannotators import (
       DiskBackend, DiskBackendConfig, OvrtxOutput, WriterRegistry
   )

   extractor = OvrtxOutput()
   backend = DiskBackend(DiskBackendConfig(root_dir="./output", overwrite=True))
   writer = WriterRegistry.get("basic")()

   renderer = ovrtx.Renderer()
   renderer.open_usd("tests/data/test_scene.usda")
   render_product = "/Render/RenderProduct_001"

   # Keep `depth` renders in flight so the GPU works ahead of the writer.
   num_frames = 500
   depth = 2

   inflight = deque()           # FIFO of (frame_id, pending render op)
   issued = done = 0
   while done < num_frames:
       # Queue renders (non-blocking) until the pipeline is full. step_async()
       # returns immediately — it only enqueues GPU work.
       while issued < num_frames and len(inflight) < depth:
           inflight.append((issued, renderer.step_async({render_product}, delta_time=0.1)))
           issued += 1

       # Collect the oldest in-flight frame. Its render overlapped the previous
       # iteration's write, so .wait() usually returns right away.
       frame_id, op = inflight.popleft()
       outputs = op.wait().fetch()
       # The camera render product yields one frame per step.
       for frame in outputs[render_product].frames:
           writer.write_frame(frame_id, extractor.extract_all(frame), backend)
       done += 1

   writer.finalize(backend)

This writes the same per-frame output as Tutorial 7 — only the scheduling
changes. Nothing in ``ovannotators`` needs to change to use the async path:
``OvrtxOutput`` and the writers consume one frame at a time regardless of how the
renderer produced it.

.. note::

   Always drain every queued operation with ``.wait().fetch()`` (the loop above
   empties ``inflight``). A ``step_async`` result that is garbage-collected
   without being fetched makes ovrtx block to clean it up and emit a
   ``ResourceWarning``.

To advance an animated timeline as in the other tutorials, call
``update_from_usd_time`` before each ``step_async`` in the fill loop.


Tutorial 10: Saving and Reloading Configs
------------------------------------------

Every ``*BackendConfig`` and ``*WriterConfig`` is a plain dataclass, which makes
a run reproducible: snapshot the configuration to a file next to the dataset,
then rebuild the exact backend and writer later (or on another machine). Use
``config_to_dict`` to serialize and ``config_from_dict`` to load — unlike a bare
``dataclasses.asdict`` + ``Config(**data)``, they also handle ``pathlib.Path``
fields and nested config dataclasses.

**Prerequisites:** none beyond the core install — no ``ovrtx`` and no extras
(``coco`` and ``DiskBackend`` are both core).

Describe the run as data — each role pairs a registry name with its config — then
write the snapshot through the backend so it lands beside the dataset:

.. code-block:: python

   import json

   from ovannotators import (
       BackendRegistry, COCOWriterConfig, DiskBackendConfig,
       WriterRegistry, config_to_dict,
   )

   backend_config = DiskBackendConfig(root_dir="./output", overwrite=True)
   writer_config = COCOWriterConfig()

   backend = BackendRegistry.create("disk", backend_config)
   writer = WriterRegistry.create("coco", writer_config)

   # ... render loop + writer.finalize(backend), as in Tutorial 1 ...

   # Snapshot the exact configuration alongside the dataset. config_to_dict()
   # converts each dataclass into JSON-friendly values (pathlib.Path -> str,
   # nested configs -> dicts), so the whole structure serializes cleanly.
   run_config = {
       "backend": {"name": "disk", "config": config_to_dict(backend_config)},
       "writer": {"name": "coco", "config": config_to_dict(writer_config)},
   }
   backend.write("run_config.json", json.dumps(run_config, indent=2).encode())

To reproduce the run later, point a bootstrap backend at the existing dataset,
read the snapshot, and rebuild the configs. Mapping each registry name to its
config class lets the file fully describe the pipeline:

.. code-block:: python

   import json

   from ovannotators import (
       BackendRegistry, COCOWriterConfig, DiskBackendConfig,
       WriterRegistry, config_from_dict,
   )

   # Map each registry name to its config dataclass.
   BACKEND_CONFIGS = {"disk": DiskBackendConfig}
   WRITER_CONFIGS = {"coco": COCOWriterConfig}

   # Read the snapshot back through a bootstrap backend at the same root.
   bootstrap = BackendRegistry.create("disk", DiskBackendConfig(root_dir="./output"))
   run_config = json.loads(bootstrap.read("run_config.json"))

   b = run_config["backend"]
   backend = BackendRegistry.create(
       b["name"], config_from_dict(BACKEND_CONFIGS[b["name"]], b["config"])
   )

   w = run_config["writer"]
   writer = WriterRegistry.create(
       w["name"], config_from_dict(WRITER_CONFIGS[w["name"]], w["config"])
   )

   # `backend` and `writer` are now configured exactly as the original run.

.. note::

   A field typed as an ambiguous union such as ``DiskBackendConfig.root_dir``
   (``Union[str, Path]``) keeps the serialized value as-is on load; the backend
   normalizes it (``DiskBackend`` wraps ``root_dir`` in ``Path``).


.. _example-scene:

Appendix: Example Scene
-----------------------

The tutorials above run against ``tests/data/test_scene.usda``. It defines the
``/Render/RenderProduct_001`` render product (requesting every ovrtx AOV),
labelled objects with ``SemanticsAPI``, a camera, and an animated timeline
(``startTimeCode``/``endTimeCode``) so ``update_from_usd_time`` has motion to
sample.

.. literalinclude:: ../../tests/data/test_scene.usda
   :language: text
   :caption: tests/data/test_scene.usda
