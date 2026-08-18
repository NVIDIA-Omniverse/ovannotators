.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

ovannotators
============

.. contents:: Contents
   :local:
   :depth: 2


What is ovannotators?
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
   Disk / S3 / Storage API

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
     - Persists bytes to a storage destination (disk, S3, HTTP API)

You control the render loop yourself — call ``writer.write_frame()`` for each frame.


Installation
------------

``ovannotators`` is a standard PyPI package — install it with whatever tool your
project uses (``pip``, ``uv``, ``poetry``, …). Examples below use ``pip``; the
``uv`` equivalent is ``uv add "ovannotators[...]"``.

.. code-block:: bash

   # Core (numpy, Pillow, openexr, warp-lang — enough for all writers on CPU)
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
   zero-copy when frames are GPU-resident (e.g. ``OvrtxOutput(device="cuda")``).
   The codec is selected automatically; without the extra ovannotators falls
   back to Pillow on the CPU.


Quick Start
-----------

.. literalinclude:: examples/quickstart.py
   :language: python
   :start-at: import ovrtx

This snippet is executed verbatim by the test suite, so it always matches the
current API. See :doc:`tutorials` for full walkthroughs and copy-paste snippets.


Built-in Writers
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 10 55

   * - Writer
     - Format
     - GPU?
     - Use case
   * - ``basic``
     - Flat per-annotator layout
     - No
     - Quick inspection, debugging
   * - ``coco``
     - MS-COCO JSON + images
     - No
     - 2D detection, segmentation
   * - ``kitti``
     - KITTI txt + images
     - No
     - Autonomous driving detection
   * - ``yolo``
     - YOLO txt + images
     - No
     - Real-time detection training
   * - ``bop``
     - BOP challenge format
     - No
     - 6DoF pose estimation
   * - ``lerobot``
     - LeRobot Parquet + video
     - Yes¹
     - Robot learning, imitation learning
   * - ``cosmos_transfer``
     - Cosmos Transfer 2.5 video
     - Yes¹
     - World model training

¹ Video writers require GPU with PyNvVideoCodec (NVENC). GPU Canny (Warp) is optional for Cosmos Transfer.


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
   * - ``storage_api``
     - HTTP storage API
     - ``[storage-api]``


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
   * - ``GenericModelOutput``
     - varies
     - Custom model output (written as NPZ)

.. note::

   Render variable keys use CamelCase (e.g. ``LdrColor``, ``DistanceToCameraSD``)
   matching the ovrtx render variable naming convention. These are accessed via
   ``frame.render_vars["LdrColor"]`` etc. Output **filenames** are converted
   to lowercase snake_case by the BasicWriter (e.g. ``ldr_color.png``,
   ``distance_to_camera.exr``).


Requirements
------------

- Python 3.10+
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
   render_outputs
   api
   tutorials
   agent-skills
