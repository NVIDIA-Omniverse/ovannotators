.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Tutorials
=========

Step-by-step guides for common ``ovannotators`` workflows.

Run renderer-backed examples from the repository root to resolve the bundled
scene paths. In these examples, ovrtx borrows an application-owned ovstage
stage. Register schemas before creating the renderer, then create and attach
the stage. Publish each stage update at an increasing ``ordinal``, wait for
``advance_write_floor(ordinal)``, and pass that ordinal to the render call.
USD time selects animation samples; a positive render ``delta_time`` advances
the sensor simulation between frames.

.. contents:: Contents
   :local:
   :depth: 1


External Formats and Dependencies
---------------------------------

The tutorials use the following externally maintained formats and libraries:
`MS-COCO
<https://cocodataset.org/#format-data>`_, `KITTI
<https://www.cvlibs.net/datasets/kitti/eval_object.php>`_, `Ultralytics YOLO
<https://docs.ultralytics.com/datasets/detect/>`_, `BOP
<https://bop.felk.cvut.cz/>`_, `Hugging Face LeRobot
<https://huggingface.co/docs/lerobot/index>`_, `NVIDIA Cosmos
<https://docs.nvidia.com/cosmos/latest/>`_, `NumPy <https://numpy.org/>`_,
`Warp <https://nvidia.github.io/warp/>`_, `nvImageCodec
<https://docs.nvidia.com/cuda/nvimagecodec/>`_, and `PyNvVideoCodec
<https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html>`_.
Renderer-backed examples additionally use `ovrtx
<https://github.com/NVIDIA-Omniverse/ovrtx>`_, `ovstage
<https://github.com/NVIDIA-Omniverse/ovstage>`_, and
`OpenUSD <https://openusd.org/release/index.html>`_.


.. _writer-tutorial-coco:

Tutorial 1: Basic COCO Dataset from ovrtx
------------------------------------------

Extract 2D detection data from a rendered scene and produce a COCO-format dataset.

**Prerequisites:** The ``coco`` writer is included in the core installation and
needs no extras. Rendering the scene requires ``ovrtx``, ``ovstage``, an NVIDIA
GPU, and CUDA.

.. code-block:: bash

   pip install ovannotators   # core: numpy, Pillow, openexr, warp-lang
   pip install "ovrtx>=0.5,<0.6" "ovstage>=0.2,<0.3" \
       --extra-index-url https://pypi.nvidia.com/  # renderer used in this example (GPU + CUDA)

**Code:**

.. code-block:: python

   import ovrtx
   import ovstage
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

   # Set up renderer + render-variable extractor. Register schemas before any
   # USD access; the Renderer must exist before the Stage it borrows.
   ovrtx.register_schema_paths()
   renderer = ovrtx.Renderer()
   stage = ovstage.Stage("tutorial")
   renderer.attach_ovstage(stage)

   # Use the first animation sample rather than USD Default time.
   ordinal = 1
   ovstage.population.open_usd(
       stage, "tests/data/test_scene.usda", ordinal=ordinal, time_code=0.0
   )
   stage.advance_write_floor(ordinal).wait()
   render_products = {"/Render/RenderProduct_001"}
   extractor = OvrtxOutput()

   # Render loop. update_from_usd_time sets the USD animation time so
   # time-sampled transforms (moving objects, cameras) advance between frames.
   # A positive delta_time advances the sensor simulation between frames.
   # Sweep time codes 0 through 5 of the scene's authored 0-100 range.
   frame_id = 0
   num_frames = 3
   start_tc, end_tc = 0.0, 5.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       ordinal += 1
       ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
       stage.advance_write_floor(ordinal).wait()
       outputs = renderer.step(render_products, delta_time=0.1, ordinal=ordinal)
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

**Prerequisites:** The ``coco``, ``kitti``, and ``yolo`` writers are included in
the core installation and need no extras. Rendering requires ``ovrtx``,
``ovstage``, an NVIDIA GPU, and CUDA, as in Tutorial 1.

.. code-block:: python

   import ovrtx
   import ovstage
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

   # Renderer setup and render loop as in Tutorial 1.
   ovrtx.register_schema_paths()
   renderer = ovrtx.Renderer()
   stage = ovstage.Stage("tutorial")
   renderer.attach_ovstage(stage)

   # Use the first animation sample rather than USD Default time.
   ordinal = 1
   ovstage.population.open_usd(
       stage, "tests/data/test_scene.usda", ordinal=ordinal, time_code=0.0
   )
   stage.advance_write_floor(ordinal).wait()
   extractor = OvrtxOutput()

   frame_id = 0
   num_frames = 3
   start_tc, end_tc = 0.0, 5.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       ordinal += 1
       ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
       stage.advance_write_floor(ordinal).wait()
       outputs = renderer.step(
           {"/Render/RenderProduct_001"}, delta_time=0.1, ordinal=ordinal
       )
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

**Prerequisites:** The ``coco`` writer needs no extras. The ``S3Backend`` requires
``boto3``, which the ``[s3]`` extra installs. To use the render loop from
Tutorial 1, install its renderer dependencies as well.

.. code-block:: bash

   pip install "ovannotators[s3]"   # adds boto3 for the S3 backend

Supply ``frames`` as an iterable of extracted COCO frame mappings, each with
``LdrColor`` and ``SemanticBoundingBox2DTight``. Replace the bucket, prefix,
and region with your destination:

.. code-block:: python

   from ovannotators import S3Backend, S3BackendConfig, WriterRegistry

   backend = S3Backend(S3BackendConfig(
       bucket="my-ml-datasets",
       prefix="synthetic/coco/run-042/",
       region="us-west-2",
   ))

   writer = WriterRegistry.get("coco")()

   for frame_id, render_vars in enumerate(frames):
       writer.write_frame(frame_id, render_vars, backend)

   writer.finalize(backend)
   # Data is now at s3://my-ml-datasets/synthetic/coco/run-042/


.. _writer-tutorial-lerobot:

Tutorial 4: Legacy LeRobot v2-Style Dataset
-----------------------------------------------

Write one episode in the legacy LeRobot v2-style format. This tutorial does not
produce the independently configured ``groot_lerobot_v2`` profile and does not
claim GR00T compatibility.

**Prerequisites:** The ``lerobot`` writer requires ``pyarrow`` for Parquet output
and ``PyNvVideoCodec`` for NVIDIA hardware video encoding (NVENC). Install the
``[lerobot]`` and ``[video]`` extras, and use an NVIDIA GPU with CUDA. Rendering
also requires ``ovrtx`` and ``ovstage``.

.. code-block:: bash

   pip install "ovannotators[lerobot,video]"  # pyarrow + PyNvVideoCodec (NVENC, needs CUDA)
   pip install "ovrtx>=0.5,<0.6" "ovstage>=0.2,<0.3" \
       --extra-index-url https://pypi.nvidia.com/  # renderer used in this example

.. code-block:: python

   import ovrtx
   import ovstage
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
   # Renderer setup and render loop as in Tutorial 1.
   ovrtx.register_schema_paths()
   renderer = ovrtx.Renderer()
   stage = ovstage.Stage("tutorial")
   renderer.attach_ovstage(stage)

   # Use the first animation sample rather than USD Default time.
   ordinal = 1
   ovstage.population.open_usd(
       stage, "tests/data/test_scene.usda", ordinal=ordinal, time_code=0.0
   )
   stage.advance_write_floor(ordinal).wait()
   extractor = OvrtxOutput()

   frame_id = 0
   num_frames = 30
   start_tc, end_tc = 0.0, 5.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       ordinal += 1
       ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
       stage.advance_write_floor(ordinal).wait()
       outputs = renderer.step(
           {"/Render/RenderProduct_001"}, delta_time=0.1, ordinal=ordinal
       )
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


Lossless Video
--------------

Basic, Cosmos Transfer, and LeRobot support opt-in lossless HEVC:

.. literalinclude:: ../../tests/docs/python/test_lossless_video.py
   :language: python
   :start-after: # [snippet:doc-lossless-video]
   :end-before: # [/snippet:doc-lossless-video]
   :dedent: 4

Use even, fixed dimensions and an HEVC Range Extensions/GBR 4:4:4 reader.
Losslessness covers normalized uint8 RGB, not alpha or original float precision.
CUDA input pixels remain on-device during packing and NVENC encoding; only
compressed packets return to the CPU for MP4 packaging and storage.
See :ref:`shared video options <video-encoding-options>` for requirements.

.. _writer-tutorial-cosmos:

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

**Prerequisites:** The ``cosmos_transfer`` writer requires ``PyNvVideoCodec``
from the ``[video]`` extra for NVIDIA hardware video encoding (NVENC), an NVIDIA
GPU, and CUDA. The ``[gpu]`` image codec is optional. The capture example also
requires ``ovrtx`` and ``ovstage``.

**Install:**

.. code-block:: bash

   pip install "ovannotators[video]"  # PyNvVideoCodec for NVENC (required)
   pip install "ovannotators[gpu]"    # optional: nvImageCodec for faster image I/O
   pip install "ovrtx>=0.5,<0.6" "ovstage>=0.2,<0.3" \
       --extra-index-url https://pypi.nvidia.com/  # renderer used in the capture example below
   # Warp (Canny kernels on CUDA or Warp's CPU device) ships in the core install.

**Required Render Variables:** The required inputs depend on the configuration.
The color image (``LdrColor``) is always required.
``DistanceToCameraSD`` is required for ``"depth"`` conditioning.
``SemanticInstanceSegmentation`` is required for ``"segmentation"`` or
``"shaded_segmentation"`` conditioning, or Canny derived from either source.
``NormalSD`` is required for ``"shaded_segmentation"`` conditioning, or Canny
derived from ``"normals"`` or ``"shaded_segmentation"``. RGB-only Canny requires
only ``LdrColor``.

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

**Capture (One or More Clips):**

Accumulate frames with ``write_frame()``, then call ``cut(backend)`` to close
the buffered frames into a clip and start a new one; the ``clip_index`` advances
automatically. Use it to slice a long capture into sequential clips, or to emit
one clip per scene variation (modify the scene, render more frames, then
``cut`` again). ``cut()`` is a no-op when no frames are buffered and returns the
written ``clip_id`` (or ``None``). ``finalize()`` flushes the last buffered clip
and writes the manifest. For a single clip, just skip ``cut()`` and call
``finalize()`` once.

This fragment uses ``config`` from the preceding block. Supply
``scene_variations`` as your application's iterable of scene variations.
``variation.apply(stage, ordinal)`` is an application-defined hook, not an
ovannotators API: it publishes its scene changes to the stage at one or more
new ordinals, advances the write floor, and returns the last ordinal it used.

.. code-block:: python

   import ovrtx
   import ovstage
   from ovannotators import (
       DiskBackend, DiskBackendConfig, WriterRegistry, OvrtxOutput
   )

   backend = DiskBackend(DiskBackendConfig(root_dir="./cosmos_data", overwrite=True))
   writer = WriterRegistry.get("cosmos_transfer")(config)

   # Renderer setup as in Tutorial 1.
   ovrtx.register_schema_paths()
   renderer = ovrtx.Renderer()
   stage = ovstage.Stage("tutorial")
   renderer.attach_ovstage(stage)

   ordinal = 1
   ovstage.population.open_usd(
       stage, "tests/data/test_scene.usda", ordinal=ordinal, time_code=0.0
   )
   stage.advance_write_floor(ordinal).wait()
   extractor = OvrtxOutput()

   num_frames = 24                           # frames per clip (≈1s at fps=24)
   start_tc, end_tc = 0.0, 5.0
   for variation in scene_variations:        # one clip per scene variation
       ordinal = variation.apply(stage, ordinal)  # e.g. swap materials / lighting / poses
       frame_id = 0                          # frame IDs restart in each clip
       for i in range(num_frames):
           usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
           ordinal += 1
           ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
           stage.advance_write_floor(ordinal).wait()
           outputs = renderer.step(
               {"/Render/RenderProduct_001"}, delta_time=0.1, ordinal=ordinal
           )
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

**Driving Cosmos 3 Transfer:**

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
   print(spec_path)
   print(spec["model_mode"])
   print(cmd)

The example prints the spec path, model mode, and command argument list.
Run the argument list with ``subprocess.run(cmd, check=True)`` in an environment
with the Cosmos Framework, checkpoint, and GPU installed.

Running the generated command with depth conditioning and the lantern prompt
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

**Prerequisites:** The core installation is sufficient. This example uses only
the Python standard library and NumPy, which is a core dependency.

The writer reads ``DistanceToCameraSD`` on the host. With ovrtx, pass
``extractor.extract_all(frame, device="cpu")`` so the depth view is copied off
the GPU first; ``np.asarray`` then accepts either a NumPy array or a CPU Warp
array.

.. code-block:: python

   import json

   import numpy as np
   from ovannotators import MissingRenderVarError, WriterRegistry


   @WriterRegistry.register
   class DepthStatsWriter:
       """Writes per-frame depth statistics as JSON lines."""

       name = "depth_stats"
       required_render_vars = frozenset({"DistanceToCameraSD"})

       def __init__(self):
           self._records = []
           self._finalized = False

       def write_frame(self, frame_id, render_vars, backend):
           if "DistanceToCameraSD" not in render_vars:
               raise MissingRenderVarError(self.name, "DistanceToCameraSD", frame_id)
           depth = np.asarray(render_vars["DistanceToCameraSD"])
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
single AOV when you only need a subset. Both methods return GPU views by
default (``device="cuda"``), where ovrtx rendered them; pass ``device="cpu"`` to
either method, for example ``extractor.extract(frame, "LdrColor", device="cpu")``,
to copy to host memory instead.

**Prerequisites:** The ``basic`` writer is included in the core installation and
needs no extras. This example uses ``OvrtxOutput`` to read renderer frames, so
it also requires ``ovrtx``, ``ovstage``, an NVIDIA GPU, and CUDA.

.. code-block:: python

   import ovrtx
   import ovstage
   from ovannotators import OvrtxOutput, DiskBackend, DiskBackendConfig, WriterRegistry

   extractor = OvrtxOutput()
   backend = DiskBackend(DiskBackendConfig(root_dir="./output", overwrite=True))
   writer = WriterRegistry.get("basic")()

   # Renderer setup and render loop as in Tutorial 1, sweeping the full 0-100 range.
   ovrtx.register_schema_paths()
   renderer = ovrtx.Renderer()
   stage = ovstage.Stage("tutorial")
   renderer.attach_ovstage(stage)

   ordinal = 1
   ovstage.population.open_usd(
       stage, "tests/data/test_scene.usda", ordinal=ordinal, time_code=0.0
   )
   stage.advance_write_floor(ordinal).wait()

   frame_id = 0
   num_frames = 50
   start_tc, end_tc = 0.0, 100.0
   for i in range(num_frames):
       usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
       ordinal += 1
       ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
       stage.advance_write_floor(ordinal).wait()
       outputs = renderer.step(
           {"/Render/RenderProduct_001"}, delta_time=0.1, ordinal=ordinal
       )
       product = outputs["/Render/RenderProduct_001"]
       for frame in product.frames:
           writer.write_frame(frame_id, extractor.extract_all(frame), backend)
           frame_id += 1

   writer.finalize(backend)


.. _writer-tutorial-numpy:

Tutorial 8: Without ovrtx (NumPy Arrays Directly)
---------------------------------------------------

The ``ovannotators`` library works with NumPy arrays without a renderer.

**Prerequisites:** The core installation is sufficient because it includes the
``basic`` writer and ``DiskBackend``. No renderer or extras are required.

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
           "DistanceToCameraSD": rng.uniform(0.5, 20.0, (480, 640)).astype(np.float32),
           "NormalSD": rng.uniform(-1, 1, (480, 640, 4)).astype(np.float32),
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

The ``ovrtx`` method ``step()`` is shorthand for ``step_async().wait().fetch()`` — it blocks
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

**Prerequisites:** The ``basic`` writer is included in the core installation and
needs no extras. Rendering requires ``ovrtx``, ``ovstage``, an NVIDIA GPU, and
CUDA.

.. code-block:: python

   from collections import deque

   import ovrtx
   import ovstage
   from ovannotators import (
       DiskBackend, DiskBackendConfig, OvrtxOutput, WriterRegistry
   )

   extractor = OvrtxOutput()
   backend = DiskBackend(DiskBackendConfig(root_dir="./output", overwrite=True))
   writer = WriterRegistry.get("basic")()

   # Renderer setup as in Tutorial 1. This loop renders a static scene.
   ovrtx.register_schema_paths()
   renderer = ovrtx.Renderer()
   stage = ovstage.Stage("tutorial")
   renderer.attach_ovstage(stage)

   ordinal = 1
   ovstage.population.open_usd(
       stage, "tests/data/test_scene.usda", ordinal=ordinal, time_code=0.0
   )
   stage.advance_write_floor(ordinal).wait()
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
           op = renderer.step_async({render_product}, delta_time=0.1, ordinal=ordinal)
           inflight.append((issued, op))
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

This writes the same kind of per-frame output as Tutorial 7 — only the
scheduling changes. Nothing in ``ovannotators`` needs to change to use the async path:
``OvrtxOutput`` and the writers consume one frame at a time regardless of how the
renderer produced it.

.. note::

   Always drain every queued operation with ``.wait().fetch()`` (the loop above
   empties ``inflight``). A ``step_async`` result that is garbage-collected
   without being fetched makes ovrtx block to clean it up and emit a
   ``ResourceWarning``.

To advance an animated timeline as in the other tutorials, do the following
before each ``step_async`` in the fill loop: increment ``ordinal``, call
``update_from_usd_time`` at that ordinal, call
``stage.advance_write_floor(ordinal).wait()``, and pass the same ordinal to
``step_async``.


Tutorial 10: Saving and Reloading Configs
------------------------------------------

Every ``*BackendConfig`` and ``*WriterConfig`` is a plain dataclass, which makes
a run reproducible: snapshot the configuration to a file next to the dataset,
then rebuild the exact backend and writer later (or on another machine). Use
``config_to_dict`` to serialize and ``config_from_dict`` to load — unlike a bare
``dataclasses.asdict`` + ``Config(**data)``, they also handle ``pathlib.Path``
fields and nested config dataclasses.

**Prerequisites:** The core installation is sufficient because it includes the
``coco`` writer and ``DiskBackend``. Saving and reloading configurations does
not require a renderer or extras.

Describe the run as data — each role pairs a registry name with its config — then
write the snapshot through the backend so it lands beside the dataset:

.. code-block:: python

   import json

   from ovannotators import (
       BackendRegistry, COCOWriterConfig, DiskBackendConfig, config_to_dict,
   )

   backend_config = DiskBackendConfig(root_dir="./output", overwrite=True)
   writer_config = COCOWriterConfig()

   backend = BackendRegistry.create("disk", backend_config)
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


.. _webdataset-packaging:

WebDataset Capture and Existing-File Packaging
----------------------------------------------

.. _webdataset-choosing-path:

Choosing a Packaging Path
~~~~~~~~~~~~~~~~~~~~~~~~~

All three entry points write the same shard layout and completion manifest
through a ``WebDatasetPackager``. See :ref:`webdataset-api` for the API
reference and the custom ``CaptureAdapter`` contract.

.. Rows are in a fixed order: managed capture, existing files, finalized output.
   Each row is one "* -" block; add or edit a whole block.

.. list-table:: Packaging entry points
   :header-rows: 1
   :widths: 20 25 20 15 20

   * - Entry point
     - Use when
     - Input
     - Sample unit
     - Completion
   * - ``DatasetCapture``
     - You capture frames now and a writer adapter defines the samples.
     - Renderer mappings passed to ``episode.write_frame`` inside episode
       contexts.
     - The adapter's ``sample_unit``: ``frame``, ``episode`` or ``clip``.
       See :ref:`webdataset-episode-clip-samples`.
     - Leaving the dataset context publishes the final shard, then the
       manifest. An uncaught error aborts.
   * - ``WebDatasetPackager.pack_episode`` with ``SampleFiles``
     - You already have encoded files and know which files form each sample.
     - Local encoded files grouped into samples; one call per episode.
     - Whatever you group: frame, multicamera capture, clip or episode.
     - Leaving the packager context, or ``finalize()``, publishes the manifest.
       ``abort()`` or an error discards pending output.
   * - ``export_finalized_dataset``
     - A writer already finalized native output in one local directory per
       episode.
     - A writer configuration template and an episode-ID-to-directory mapping.
     - The adapter's ``sample_unit``: ``frame``, ``episode`` or ``clip``.
       See :ref:`webdataset-episode-clip-samples`.
     - Returns after publishing the manifest; any failure aborts. Source
       directories are not modified.

``DatasetCapture`` and ``export_finalized_dataset`` select a built-in adapter
by exact writer type. Pass ``adapter=`` for a subclass of a built-in writer,
for a custom writer, or for an adapter that is never selected automatically,
such as ``BasicVideoEpisodeAdapter``.

.. Rows are sorted by writer class name; for the same writer, the
   automatically selected adapter comes first. Each row is one "* -" block;
   insert a new adapter as a single block in sorted position.

.. list-table:: Built-in adapters
   :header-rows: 1
   :widths: 18 18 16 10 38

   * - Writer
     - Adapter
     - Sample schema
     - Sample unit
     - Notes
   * - ``BasicWriter``
     - ``BasicFrameAdapter``
     - ``basic-frame-v1``
     - ``frame``
     - Original per-frame file names become component suffixes, such as
       ``ldr_color.png``. Rejects ``video=True``; use
       ``BasicVideoEpisodeAdapter``.
   * - ``BasicWriter`` with ``video=True``
     - ``BasicVideoEpisodeAdapter``, passed as ``adapter=``
     - ``basic-video-episode-v1``
     - ``episode``
     - Never selected automatically. Every episode MP4 with unchanged encoded
       bytes, per-frame files selected by ``frame_files`` (default
       ``("*.json",)``) and an ``episode.json`` with per-stream timestamps and
       source frame indices.
   * - ``BOPWriter``
     - ``BOPFrameAdapter``
     - ``bop-frame-v1``
     - ``frame``
     - One scene frame: ``rgb.png``, 16-bit ``depth.png``, every object mask
       and a ``json`` record with the frame's ``scene_camera``, ``scene_gt``
       and ``scene_gt_info`` entries. IDs, units and ``depth_scale`` are native.
   * - ``COCOWriter``
     - ``COCOFrameAdapter``
     - ``coco-instance-v1``
     - ``frame``
     - ``jpg`` or ``png`` image plus a per-image COCO-style ``json`` record.
       Image and annotation IDs are unique across the dataset; the manifest
       holds the complete category table.
   * - ``KITTIWriter``
     - ``KITTIFrameAdapter``
     - ``kitti-frame-v1``
     - ``frame``
     - ``image_2.png``, ``label_2.txt``, ``calib.txt`` and, with
       ``write_depth=True``, ``depth.png`` components with unchanged native
       bytes. An empty ``label_2.txt`` is a negative example.
   * - ``YOLOWriter``
     - ``YOLOFrameAdapter``
     - ``yolo-frame-v1``
     - ``frame``
     - Native image, ``txt`` label and a ``json`` record with the frame's
       original ``split``. Class IDs map to one dataset-wide table in the
       manifest; an empty ``txt`` is a negative example.

Formats Not Packaged as WebDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No built-in adapter exists for the profile writers below, and they are not
packaged as WebDataset. Use their native output:

- **Legacy LeRobot** (``LeRobotWriter``): writes the older ``v2.0`` layout
  (``codebase_version``). A LeRobot v3.0 profile is coming soon; see
  :doc:`writers/lerobot_v3`.
- **Cosmos multiview** (``CosmosMultiviewWriter``): its ``samples.lance`` rows
  reference assets by absolute URIs under the configured ``output_base_uri``.
  Packaging files into tar shards does not change those URIs, so keep the
  native output where those URIs resolve.

A custom ``CaptureAdapter`` can package another writer; see
:ref:`webdataset-api`.

Capture and Packaging Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a WebDataset from COCO-style frame samples. ``episodes`` supplies episode
IDs and extracted renderer mappings; each frame requires ``LdrColor`` and
``SemanticBoundingBox2DTight``. ``SemanticIdMap`` optionally supplies class labels.
The contexts handle finalization and preserve episode boundaries by default.
In these examples, ``output_dir`` is a ``pathlib.Path`` for the destination
directory, and ``episodes`` is an iterable of ``(episode_id, frames)`` pairs.

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-capture]
   :end-before: # [/snippet:doc-webdataset-capture]
   :dedent: 4

The result is ``training-run/manifest.json`` plus numbered tar shards. Each shard
contains keys such as ``training-run/episode-episode001/sample-0.jpg``,
``sample-0.json`` and ``sample-0.metadata.json``. All sample components are adjacent.
See :doc:`api` for the annotation schema, count/byte limits and oversized policy.

Package existing encoded files with explicit sample membership. This preserves
the original bytes and does not re-encode images. A sample can be a frame,
synchronized multicamera capture, clip or complete episode, depending on the
components you supply.

Supply ``image_path`` and ``annotation_path`` as paths to an existing encoded
image and its JSON annotation file:

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-pack-files]
   :end-before: # [/snippet:doc-webdataset-pack-files]
   :dedent: 4

For an already finalized BasicWriter directory, use the writer's adapter to
discover complete samples without listing files yourself. Pass one native
directory per episode. The source directory remains unchanged; the completion
manifest references a checksummed native-path map.
Here, ``native`` is that directory's path. The snippet assumes it was written
with the default ``BasicWriterConfig()``; the configuration must match the one
that produced the directory.

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-export-finalized]
   :end-before: # [/snippet:doc-webdataset-export-finalized]
   :dedent: 4

KITTI Frames
^^^^^^^^^^^^

A finalized KITTI directory is packaged the same way. The writer configuration
must match the directory, including ``write_depth``. Each sample carries the
original ``image_2``, ``label_2``, ``calib`` and optional ``depth`` files as
separate components, so label rows, calibration matrices and stored depth
values are read exactly as the native writer produced them. The episode
directory must contain only the KITTI directories; a stray entry such as
``.DS_Store`` is rejected. See :ref:`webdataset-sample-schemas` for the KITTI
sample conventions.

Supply ``kitti_dir`` as the path to the finalized native dataset and
``output_dir`` as the destination directory:

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-kitti]
   :end-before: # [/snippet:doc-webdataset-kitti]
   :dedent: 4

BOP Scene Frames
^^^^^^^^^^^^^^^^

``DatasetCapture`` selects ``BOPFrameAdapter`` for an exact ``BOPWriter``. Each
episode is one scene with the configured ``scene_id``; each sample is one frame.
Frame indices may restart in the next episode because keys include the episode.
Supply ``episodes`` and ``output_dir`` as in the COCO capture example above,
with each frame containing the :doc:`BOP inputs <writers/bop>`.

.. literalinclude:: ../../tests/docs/python/test_webdataset_bop.py
   :language: python
   :start-after: # [snippet:doc-webdataset-bop-capture]
   :end-before: # [/snippet:doc-webdataset-bop-capture]
   :dedent: 4

A sample's ``json`` record holds that frame's original ``scene_camera``,
``scene_gt`` and ``scene_gt_info`` entries. ``masks`` lists, per ``scene_gt``
entry, the component names of its mask files, or ``null`` when a file is absent.
See :ref:`webdataset-sample-schemas` for the full BOP record. Here,
``shard_urls`` is the ordered list of shard file URLs from the manifest, as
constructed in :ref:`Reading the Output <webdataset-reading-output>`. Decode
16-bit depth without an 8-bit image conversion:

.. literalinclude:: ../../tests/docs/python/test_webdataset_bop.py
   :language: python
   :start-after: # [snippet:doc-webdataset-bop-read]
   :end-before: # [/snippet:doc-webdataset-bop-read]
   :dedent: 4

YOLO Frames and Splits
^^^^^^^^^^^^^^^^^^^^^^

``DatasetCapture`` selects ``YOLOFrameAdapter`` for an exact ``YOLOWriter``.
Each frame keeps the writer's original ``train`` or ``val`` assignment in its
``json`` record and ``metadata.json``; episodes are not repartitioned, so one
shard can hold both splits. Select a split in the reader. The manifest's
``metadata.split_counts`` gives the number of samples per split, and
``metadata.classes`` holds the dataset-wide class table used by every ``txt``
label. See :ref:`webdataset-sample-schemas` for the YOLO sample record.
Supply ``episodes`` and ``output_dir`` as in the COCO capture example above.

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-yolo-split-filter]
   :end-before: # [/snippet:doc-webdataset-yolo-split-filter]
   :dedent: 4

.. _webdataset-episode-clip-samples:

Episode and Clip Samples
~~~~~~~~~~~~~~~~~~~~~~~~

The adapter's ``sample_unit`` decides what one WebDataset sample holds. Episode
and clip units are experimental.

- ``frame`` (default): one sample per written frame. Use it for per-image
  training data, such as Basic or COCO frames.
- ``episode``: one sample per finalized episode. Use it when the writer's
  output only makes sense as a whole, such as episode-wide video.
- ``clip``: one sample per clip. Use it when an episode holds several
  independent segments, such as one clip per scene variation.

With a clip adapter, call ``episode.end_clip()`` between clips. Frames written
after the last boundary form the final clip when the episode exits
successfully. ``end_clip()`` raises ``PackagingError`` when no frame was
written since the previous boundary or the adapter is not a clip adapter.
This example includes a text writer, its clip adapter, and the producer loop.
Supply ``output_dir`` as the destination path:

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-clip-adapter]
   :end-before: # [/snippet:doc-webdataset-clip-adapter]
   :dedent: 4

Shard limits apply to whole samples:

- ``max_samples_per_shard`` and the manifest's ``samples`` count episodes or
  clips, not frames.
- A sample is never split across shards. An episode sample larger than
  ``max_shard_bytes`` raises ``PackagingError`` unless the packager config sets
  ``oversized_episode_policy="dedicated_shard"``.
  ``allow_episode_spanning=True`` does not help, because it only rotates
  shards between samples.
- The clips of one episode stay in one shard unless spanning is enabled.

The manifest records the unit as ``metadata.sample_unit``. See
:ref:`webdataset-api` for the full adapter contract, including
``discover_clips`` for finalized export.

Basic Video Episodes
^^^^^^^^^^^^^^^^^^^^

``BasicWriter(video=True)`` writes episode-wide MP4 files, which cannot be
split into frame samples, so the default Basic adapter rejects it. Pass
``adapter=BasicVideoEpisodeAdapter()`` to package each episode as one sample.
The sample holds every MP4 with its encoded bytes unchanged, the per-frame
files selected by ``frame_files`` and an ``episode.json`` record. For each
stream, the record lists the presentation timestamps and the episode frame
index of every video frame:

Supply ``output_dir`` as the destination path and ``frames`` as an iterable of
frame mappings containing ``LdrColor`` and ``CameraParams``. Encoding requires
the ``[video]`` extra and an NVENC-capable NVIDIA GPU; reading requires the
separate ``webdataset`` package.

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-basic-video]
   :end-before: # [/snippet:doc-webdataset-basic-video]
   :dedent: 4

``frame_files`` defaults to ``("*.json",)``, which keeps the per-frame JSON
that the videos do not carry. Use ``("*",)`` for every frame file or ``()``
for videos only. An episode larger than ``max_shard_bytes`` needs
``oversized_episode_policy="dedicated_shard"``. See :ref:`webdataset-api` for
the ``episode.json`` fields.

.. _webdataset-reading-output:

Reading the Output
~~~~~~~~~~~~~~~~~~

A packaged dataset is one ``<dataset_id>/`` prefix on the storage backend:

.. code-block:: text

   training-run/
     shard-000000.tar
     shard-000001.tar
     auxiliary/source-paths.json
     manifest.json

The completion manifest is published only after every shard has been written.
Do not consume a dataset, or list its shards from storage, before
``manifest.json`` exists: an aborted run can leave shards without a manifest.
Take the ordered shard list from the manifest's ``shards`` entries and compare
each shard's SHA-256 with its ``sha256`` field before training.
``auxiliary/source-paths.json``, written by managed capture and finalized
export, maps each sample back to its native writer file names. It is listed with
a checksum under the manifest's ``auxiliary`` field and is not needed for
training.

Inside a shard, every member name is ``<dataset_id>/episode-<episode_id>/sample-<sample_id>.<suffix>``.
The part before the first dot is the sample key; the remainder is the component
suffix, such as ``jpg``, ``json`` or ``ldr_color.png``. All components of a sample
are adjacent. Each sample also contains a generated ``metadata.json`` component
with ``schema_version``, ``dataset_id``, ``episode_id``, ``sample_id`` and the
application ``metadata`` (for the built-in frame adapters, the writer name and
``frame_index``).

Read the shards with the standard-library ``tarfile`` module or with the
separate ``webdataset`` reader package, which yields one dictionary per sample
keyed by component suffix. ``ovannotators`` does not depend on the reader: install
it with ``pip install webdataset`` (the repository's ``test`` extra pins
``webdataset==1.0.2``). The reader opens URLs, so pass local shards as file URLs
(``path.resolve().as_uri()``); a Windows drive path such as ``C:\`` is read as a URL scheme.
Here, ``output_dir`` is the destination directory of a COCO capture with the
``training-run`` dataset ID.

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-read]
   :end-before: # [/snippet:doc-webdataset-read]
   :dedent: 4

Reader shuffling, node/worker splitting, batching and augmentation are
configured by the training application. Packaging does not configure these.
For a complete CPU-only capture and read-back, run
``python examples/webdataset_capture.py`` from the repository root.
