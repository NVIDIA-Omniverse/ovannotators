.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

ovrtx render outputs
====================

``OvrtxOutput`` extracts RenderVars by their ovrtx ``sourceName``. Its default
``decode=True`` path maps image tensors through DLPack and decodes packed
annotator buffers before passing them to a writer. The complete, runtime-tested
camera catalog is maintained in the `ovrtx camera output documentation
<https://nvidia-omniverse.github.io/ovrtx/sensors/cameras/outputs.html>`_.

The ovrtx 0.4.0 inventory was tested on one semantic robot-lineup scene. Do not
infer support from a renderer catalog entry alone: several legacy RT/PathTracing
names appear in a frame but are zero-size or fail ovrtx readback.

Image and tensor outputs
------------------------

``BasicWriter`` has explicit filenames and encodings for every passing image or
tensor output below. Unknown names still fall back to ``.npy``.

.. list-table::
   :header-rows: 1
   :widths: 34 24 42

   * - ``sourceName``
     - BasicWriter encoding
     - Notes
   * - ``LdrColor``
     - PNG/JPEG
     - Display-ready uint8 RGBA.
   * - ``HdrColor``
     - EXR
     - Linear float16 RGBA.
   * - ``NormalSD``
     - EXR plus colorized PNG
     - Working RT2 normal alternative to ``SmoothNormal``.
   * - ``DepthSD``
     - EXR or NPY
     - Python readback returned non-zero float32 depth.
   * - ``DistanceToCameraSD``, ``DistanceToImagePlaneSD``
     - EXR/NPY plus colorized PNG
     - Metric distance outputs.
   * - ``DiffuseAlbedoSD``
     - PNG/JPEG
     - Working RT2 alternative to ``DiffuseAlbedo``.
   * - ``Camera3dPositionSD``
     - EXR
     - Camera-space position.
   * - ``TargetMotionSD``
     - EXR
     - Working SDG alternative to ``Motion2d``.
   * - ``SemanticSegmentation``
     - NPY plus colorized PNG
     - Scalar semantic IDs.
   * - ``SemanticInstanceSegmentation``
     - NPY plus colorized PNG
     - 128-bit IDs as four uint32 channels.
   * - ``NonStableInstanceSegmentation``
     - NPY
     - Frame-local scalar instance IDs.
   * - ``StableIdSegmentation``
     - NPY
     - 128-bit stable IDs as four uint32 channels.

Decoded buffers
---------------

The adapter decodes these buffers when ``decode=True``. ``BasicWriter`` writes
the decoded representation as JSON.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``sourceName``
     - Decoded representation
   * - ``SemanticBoundingBox2DTight``, ``SemanticBoundingBox2DLoose``
     - Structured records with 128-bit instance/semantic IDs and pixel bounds.
   * - ``SemanticBoundingBox3D``
     - Structured records with IDs, 3-D bounds, and a 4x4 transform.
   * - ``SemanticIdMap``
     - ``dict[int, str]`` mapping semantic ID to label.
   * - ``StableIdMap``
     - ``dict[int, str]`` mapping stable ID to USD prim path.
   * - ``StableIdSemanticIdMap``
     - Structured 32-byte stable-ID/semantic-ID pair records.
   * - ``InstanceMap``
     - Structured instance index to stable-semantic-table index records; ``0xffffffff`` is invalid.
   * - ``CameraParams``
     - Structured camera matrices, resolution, clipping, and lens fields.

Static-scene delta buffers
--------------------------

``SemanticIdMapDeltas`` and ``StableIdMapDeltas`` mapped as four-byte zero-count
payloads after static warm-up. They are not decoded automatically because a
non-empty delta layout has not yet been runtime-verified.

Unsupported legacy names
------------------------

The tested ovrtx runtime could not export the legacy ``SmoothNormal``,
``BumpNormal``, ``AmbientOcclusion``, ``Motion2d``, ``DiffuseAlbedo``,
``SpecularAlbedo``, ``Roughness``, direct/indirect lighting, reflections,
``DepthLinearized``, ``EmissionAndForegroundMask``, or any ``Pt*`` debug AOV.
See the linked ovrtx page and its CSV for the per-name ``EMPTY_BUFFER`` versus
``EXPORT_NOT_WIRED`` evidence and supported alternatives.

Sensor-only outputs
-------------------

``GenericModelOutput``, ``RtxSensorCpu``, and ``RtxSensorGpu`` are not camera
outputs. Route them through the corresponding lidar/radar workflow instead.
