.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

ovannotators
============

.. contents:: Contents
   :local:
   :depth: 2


What Is ovannotators?
---------------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- DOCS:description:start -->
   :end-before: <!-- DOCS:description:end -->


Capabilities
------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- DOCS:capabilities:start -->
   :end-before: <!-- DOCS:capabilities:end -->


Architecture
------------

::

   Renderer (ovrtx)
       │
       ▼ DLPack / warp/numpy arrays
   ┌──────────────────────────────┐
   │       ovannotators            │
   │                              │
   │  RendererOutput → Writer → Backend  │
   │                              │
   └──────────────────────────────┘
       │
       ▼
   Disk / S3 / ovstorage

Three layers:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Layer
     - Responsibility
   * - **RendererOutput**
     - Extracts arrays from renderer-specific frame objects
   * - **Writer**
     - Converts arrays into a target dataset format (COCO, KITTI, YOLO, etc.)
   * - **Backend**
     - Persists bytes to a storage destination (disk, S3, ovstorage)

You control the render loop yourself — call ``writer.write_frame()`` for each frame.


Installation
------------

``ovannotators`` is a standard PyPI package — install it with whatever tool your
project uses (``pip``, ``uv``, ``poetry``, …). Examples below use ``pip``; the
``uv`` equivalent is ``uv add "ovannotators[...]"``.

.. code-block:: bash

   # Core (numpy, Pillow, openexr, warp-lang — enough for the image and
   # annotation writers on CPU)
   pip install ovannotators

   # GPU image codec (adds nvImageCodec; Warp ships in the core install).
   # Recommended on any CUDA host: much faster PNG/JPEG I/O, zero-copy for
   # GPU-resident frames.
   pip install "ovannotators[gpu]"

   # Full CPU stack with all backends
   pip install "ovannotators[all-cpu]"

   # Full GPU stack
   pip install "ovannotators[all-gpu]"


Install Extras
~~~~~~~~~~~~~~

Add an extra with ``pip install "ovannotators[<name>]"`` (or ``uv add "ovannotators[<name>]"``):

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- DOCS:extras:start -->
   :end-before: <!-- DOCS:extras:end -->

Core dependencies (always installed): ``numpy``, ``Pillow``, ``openexr``, ``warp-lang``.

.. tip::

   On a machine with an NVIDIA GPU, install the ``[gpu]`` extra
   (``pip install "ovannotators[gpu]"``). Image encoding then runs on the GPU
   via nvImageCodec — several times faster than the Pillow CPU fallback, and
   zero-copy when frames are GPU-resident (e.g. ``OvrtxOutput().extract_all(frame)``, whose default ``device="cuda"``
   keeps frames on the GPU).
   The codec is selected automatically; without the extra ovannotators falls
   back to Pillow on the CPU.


Quick Start
-----------

Run both examples from the repository root. Start without a GPU; the second
example adds the renderer.

Without a GPU (CPU only)
~~~~~~~~~~~~~~~~~~~~~~~~

Writers accept NumPy arrays from any producer, so no renderer is needed. This
example writes three frames of color, depth, and normals under
``_out/quickstart_cpu``:

.. literalinclude:: ../../examples/quickstart_cpu.py
   :language: python
   :start-at: import numpy

Install with ``pip install ovannotators`` and run
``python examples/quickstart_cpu.py``. Each frame directory holds
``ldr_color.png``, ``distance_to_camera.exr``, ``normal.exr``, and colorized
PNG previews of depth and normals.

With an NVIDIA GPU (ovrtx and ovstage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example renders the bundled scene with ``ovrtx`` and writes COCO images
and annotations under ``output/coco``. It needs an NVIDIA RTX-capable GPU,
CUDA 12, ``ovrtx``, and ``ovstage``. ``ovstage`` installs from PyPI; the
``ovrtx`` renderer wheels are published on NVIDIA's public package index:

.. code-block:: bash

   pip install ovannotators "ovrtx>=0.5,<0.6" "ovstage>=0.2,<0.3" \
       --extra-index-url https://pypi.nvidia.com/

The application owns the stage; each population update is published before the
renderer consumes it.

.. literalinclude:: examples/quickstart.py
   :language: python
   :start-at: import ovrtx

The test suite executes the CPU example and executes the GPU example when
``ovrtx`` is available. It adjusts only the GPU example's output path and frame
count, so each executed example matches the current API. See :doc:`tutorials` for full walkthroughs and copy-paste snippets.


Built-in Writers
----------------

See :doc:`writers/index` for a comparison and a dedicated page for each
documented built-in writer. The ``groot_lerobot_v2`` and ``umi_replay_buffer``
profiles are described in :doc:`api`. Each page explains purpose, required inputs,
output fields and data types, configuration, usage, and format limitations, with
direct upstream references where available.

FiftyOne Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~

The COCO, KITTI, and YOLO outputs above are importable through FiftyOne's
``COCODetectionDataset``, ``KITTIDetectionDataset``, and ``YOLOv5Dataset``
types. The layouts need explicit import arguments: pass ``data_path`` and
``labels_path`` for COCO and KITTI; pass ``yaml_path="data.yaml"`` and the
desired ``split`` for YOLO. The import calls and label assertions are exercised
by ``tests/test_fiftyone_interop.py`` in the separate FiftyOne CI job.

FiftyOne's ``FiftyOneDataset`` type is a serialized snapshot of a FiftyOne
dataset. After importing a standard output, FiftyOne can export that snapshot
itself. The writers keep their format-specific output usable by training tools
without requiring FiftyOne at capture time.


Built-in Backends
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 20

   * - Backend
     - Destination
     - Extra required
   * - ``disk``
     - Local filesystem
     - —
   * - ``s3``
     - Amazon S3 / compatible
     - ``[s3]``
   * - ``ovstorage``
     - ovstorage Stack routes
     - ``[ovstorage]``


Render Variable Keys
--------------------

Writers look up data by string key in the ``render_vars`` dictionary you pass
to ``write_frame()``. The keys below are the names each writer expects.
You are responsible for mapping renderer output names to these keys.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Type
     - Description
   * - ``LdrColor``
     - uint8 (H, W, 4)
     - RGBA low dynamic range image
   * - ``HdrColor``
     - float16 (H, W, 4)
     - RGBA high dynamic range image (written as EXR)
   * - ``DiffuseAlbedoSD``
     - uint8 (H, W, 4)
     - Diffuse albedo (written as PNG)
   * - ``DistanceToCameraSD``
     - float32 (H, W)
     - Distance from camera origin to surface
   * - ``DistanceToImagePlaneSD``
     - float32 (H, W)
     - Distance from image plane to surface (Z-depth)
   * - ``NormalSD``
     - float32 (H, W, 4)
     - Surface normals (camera space)
   * - ``Camera3dPositionSD``
     - float32 (H, W, 4)
     - 3D world-space position per pixel
   * - ``SemanticInstanceSegmentation``
     - uint32 (H, W)
     - Instance ID per pixel
   * - ``SemanticSegmentation``
     - uint32 (H, W)
     - Semantic class ID per pixel
   * - ``SemanticBoundingBox2DTight``
     - structured array
     - 2D tight bounding boxes
   * - ``SemanticBoundingBox2DLoose``
     - structured array
     - 2D loose bounding boxes
   * - ``SemanticBoundingBox3D``
     - structured array
     - 3D oriented bounding boxes
   * - ``CameraParams``
     - structured array
     - Intrinsics + extrinsics
   * - ``Motion2d``
     - float32 (H, W, 4)
     - 2D motion vectors
   * - ``PointCloud``
     - varies
     - Composite lidar/radar output (written as NPZ, one member per tensor)

.. note::

   Render variable keys use CamelCase (e.g. ``LdrColor``, ``DistanceToCameraSD``)
   matching the ovrtx ``sourceName`` convention. Use
   ``OvrtxOutput.extract(frame, "LdrColor")`` to resolve these names from an
   ovrtx frame. Starting with ovrtx 0.5, direct access through
   ``frame.render_vars`` requires the full authored RenderVar prim path. Output
   **filenames** are converted to lowercase snake_case by the BasicWriter (e.g.
   ``ldr_color.png``, ``distance_to_camera.exr``).


Requirements
------------

- Python 3.11+
- NumPy, Pillow, openexr, warp-lang (installed automatically)
- No NVIDIA runtime required for core functionality (Warp runs on CPU when no GPU is present)
- GPU extras require CUDA 12.x + compatible driver


License
-------

Apache 2.0 — see ``LICENSE``.


.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   self
   writers/index
   render_outputs
   api
   tutorials
   troubleshooting
   agent-skills
