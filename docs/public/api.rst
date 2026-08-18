.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

API Reference
=============

Complete reference for the ``ovannotators`` public API.

.. contents:: Contents
   :local:
   :depth: 2


Module: ``ovannotators``
------------------------

Top-level exports:

.. code-block:: python

   from ovannotators import (
       # Version
       __version__,
       # Writers
       Writer, WriterRegistry,
       # Backends
       Backend, BackendRegistry,
       DiskBackend, DiskBackendConfig,
       S3Backend, S3BackendConfig,
       StorageAPIBackend, StorageAPIBackendConfig,
       # Renderer outputs
       RendererOutput, RendererOutputRegistry, OvrtxOutput,
       # Config serialization
       config_to_dict, config_from_dict,
       # Types
       Resolution, BBox2D, BBox3D, CameraParams, PointCloudResult,
       # Exceptions
       OVAnnotatorsError, ConfigurationError, ExtractionError,
       ValidationError, MissingRenderVarError,
       BackendWriteError, BackendReadError,
   )


Writers
-------

Writer Protocol
~~~~~~~~~~~~~~~

All writers implement this protocol:

.. code-block:: python

   class Writer(Protocol):
       name: ClassVar[str]                            # Registry key (e.g. "coco")
       required_render_vars: ClassVar[frozenset[str]] # AOVs this writer needs

       def write_frame(
           self,
           frame_id: int,
           render_vars: dict[str, np.ndarray],
           backend: Backend,
       ) -> None: ...

       def finalize(self, backend: Backend) -> None: ...

**Contracts:**

- ``write_frame()`` is called once per frame, sequentially.
- ``render_vars`` contains at least the vars in ``required_render_vars``; writers should tolerate extra keys.
- ``finalize()`` is called exactly once after the last frame. Must be idempotent.
- Writers must not assume frame IDs are contiguous or start at zero.


WriterRegistry
~~~~~~~~~~~~~~

.. code-block:: python

   WriterRegistry.get(name: str) -> type[Writer]      # Get writer class by name
   WriterRegistry.create(name, *args, **kwargs)        # Instantiate by name
   WriterRegistry.list_names() -> list[str]            # List registered names
   WriterRegistry.register(cls) -> cls                 # Decorator to register

There are two equivalent ways to obtain a plugin (writer, backend, or renderer
output), and the same pattern applies to ``BackendRegistry`` and
``RendererOutputRegistry``:

* **Registry by name** — best for dynamic/config-driven code and for any
  third-party plugin: ``WriterRegistry.create("coco")`` (or
  ``WriterRegistry.get("coco")(...)``). This works without importing the
  plugin's module.
* **Direct import** — best for static typing and editor autocompletion: every
  built-in writer and its config are exported from the top-level package, e.g.
  ``from ovannotators import COCOWriter, COCOWriterConfig``.


BasicWriter
~~~~~~~~~~~

:Registry name: ``basic``
:GPU required: No
:Required render vars: None (writes whatever is provided)
:Extra dependencies: None

Writes all present render variables in a flat per-frame directory layout. Useful for debugging and quick inspection.

**Config:** ``BasicWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ``render_vars``
     - ``list[str] | None``
     - ``None``
     - Which vars to write. ``None`` = all present
   * - ``depth_format``
     - ``"npy" | "exr"``
     - ``"exr"``
     - Storage format for depth arrays
   * - ``image_format``
     - ``"png" | "jpg"``
     - ``"png"``
     - Encoding for ``LdrColor`` and similar image AOVs
   * - ``frame_id_digits``
     - ``int``
     - ``4``
     - Zero-padding width for the frame directory name
   * - ``colorize``
     - ``bool``
     - ``True``
     - Write colorized PNG visualizations alongside raw data for depth,
       normal, and segmentation vars
   * - ``depth_near_bright``
     - ``bool``
     - ``True``
     - In colorized depth, render near surfaces bright

**Output structure** (with default ``depth_format="exr"``, ``image_format="png"``):

::

   {root}/
   └── {frame_id:04d}/
       ├── ldr_color.png
       ├── hdr_color.exr
       ├── diffuse_albedo.png
       ├── distance_to_camera.exr
       ├── distance_to_camera_colorized.png
       ├── distance_to_image_plane.exr
       ├── distance_to_image_plane_colorized.png
       ├── normal.exr
       ├── normal_colorized.png
       ├── camera_3d_position.exr
       ├── instance_segmentation.npy
       ├── instance_segmentation_colorized.png
       ├── semantic_segmentation.npy
       ├── semantic_segmentation_colorized.png
       ├── bbox_2d_tight.json
       ├── bbox_2d_loose.json
       ├── bbox_3d.json
       ├── camera_params.json
       ├── motion_2d.exr
       └── generic_model_output.npz

Filenames follow lowercase snake_case convention, matching ``omni.replicator.core``
BasicWriter output naming. Only render variables present in the input are written.
Depth vars (``distance_to_camera``, ``distance_to_image_plane``) follow
``depth_format``; ``normal``, ``camera_3d_position``, and ``motion_2d`` are always
written as EXR. Colorized PNGs are emitted for depth, normal, and segmentation
vars when ``colorize=True``.


COCOWriter
~~~~~~~~~~

:Registry name: ``coco``
:GPU required: No
:Required render vars: ``LdrColor``, ``SemanticBoundingBox2DTight`` (plus ``SemanticInstanceSegmentation`` when ``include_panoptic=True``)
:Extra dependencies: None

Accumulates annotations across frames and writes MS-COCO format JSON on finalize.

**Config:** ``COCOWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 22 18 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ``split``
     - ``str``
     - ``"train"``
     - Split name used in the annotation filename
   * - ``image_format``
     - ``"jpg" | "png"``
     - ``"jpg"``
     - Image encoding format
   * - ``include_segmentation``
     - ``bool``
     - ``False``
     - Attach per-instance RLE segmentation masks
   * - ``include_panoptic``
     - ``bool``
     - ``False``
     - Emit COCO panoptic annotations and per-image panoptic PNGs
   * - ``bbox_mode``
     - ``"tight" | "loose"``
     - ``"tight"``
     - ``"loose"`` expands each side by 1 % of the box dimension

**Output structure:**

::

   {root}/
   ├── images/
   │   └── {frame_id:06d}.jpg
   └── annotations/
       ├── instances_{split}.json
       └── panoptic_{split}/              (only when include_panoptic=True)
           └── {frame_id:06d}.png

**Input format for** ``SemanticBoundingBox2DTight``:

Structured numpy array (:data:`OVRTX_BBOX2D_DTYPE`) with fields ``instance_id``
and ``semantic_id`` (each a 128-bit id, 4× ``uint32``) plus ``x_min``, ``y_min``,
``x_max``, ``y_max`` (``int32``). Class label strings are resolved from the
optional ``SemanticIdMap`` render var, not stored in the box array.


KITTIWriter
~~~~~~~~~~~

:Registry name: ``kitti``
:GPU required: No
:Required render vars: ``LdrColor``, ``SemanticBoundingBox2DTight``, ``SemanticBoundingBox3D``, ``CameraParams``
:Optional render vars: ``DistanceToCameraSD`` (required when ``write_depth=True``)
:Extra dependencies: None

Writes dataset in KITTI object-detection format.

**Config:** ``KITTIWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 20 25 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``classes``
     - ``list[str] | None``
     - ``None``
     - Filter to these class names only
   * - ``write_depth``
     - ``bool``
     - ``False``
     - Write 16-bit depth PNGs
   * - ``depth_scale``
     - ``float``
     - ``256.0``
     - uint16 units per metre (``depth_uint16 = round(depth_m * depth_scale)``)

**Output structure:**

::

   {root}/
   ├── image_2/   {frame_id:06d}.png
   ├── label_2/   {frame_id:06d}.txt
   ├── calib/     {frame_id:06d}.txt
   └── depth/     {frame_id:06d}.png   (only when write_depth=True)

**Label format (per row):**

::

   type truncated occluded alpha left top right bottom height width length x y z rotation_y


YOLOWriter
~~~~~~~~~~

:Registry name: ``yolo``
:GPU required: No
:Required render vars: ``LdrColor``, ``SemanticBoundingBox2DTight``
:Optional render vars: ``SemanticInstanceSegmentation`` (required when ``task="segment"``)
:Extra dependencies: None

Writes dataset in Ultralytics YOLO format.

**Config:** ``YOLOWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ``task``
     - ``"detect" | "segment"``
     - ``"detect"``
     - ``"segment"`` appends polygon points for instance masks
   * - ``val_split``
     - ``float``
     - ``0.1``
     - Fraction of frames assigned to ``val`` (deterministic by frame id)
   * - ``image_format``
     - ``"jpg" | "png"``
     - ``"jpg"``
     - Image encoding format
   * - ``classes``
     - ``list[str] | None``
     - ``None``
     - Class list; ``None`` = collect in first-seen order

**Output structure:**

::

   {root}/
   ├── images/
   │   ├── train/{frame_id:06d}.jpg
   │   └── val/{frame_id:06d}.jpg
   ├── labels/
   │   ├── train/{frame_id:06d}.txt
   │   └── val/{frame_id:06d}.txt
   └── data.yaml

**Label format:**

- Detection: ``class_id cx cy w h`` (normalized to [0, 1])
- Segmentation: ``class_id x1 y1 x2 y2 … xn yn`` (polygon, normalized). Rows with
  no extractable polygon fall back to the detection format.


BOPWriter
~~~~~~~~~

:Registry name: ``bop``
:GPU required: No
:Required render vars: ``LdrColor``, ``DistanceToCameraSD``, ``SemanticBoundingBox2DTight``, ``SemanticBoundingBox3D``, ``SemanticInstanceSegmentation``, ``CameraParams``
:Extra dependencies: None

Writes dataset in BOP (Benchmark for 6D Object Pose Estimation) format.

**Config:** ``BOPWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 20 22 15 43

   * - Parameter
     - Type
     - Default
     - Description
   * - ``scene_id``
     - ``int``
     - ``0``
     - BOP scene identifier (output under ``{scene_id:06d}/``)
   * - ``depth_scale``
     - ``float``
     - ``1.0``
     - mm per stored depth unit (``depth_mm = depth_stored * depth_scale``)
   * - ``object_id_map``
     - ``dict[str, int] | None``
     - ``None``
     - Maps label strings to BOP object IDs; ``None`` uses the render var's
       ``instance_id`` directly

**Output structure:**

::

   {root}/{scene_id:06d}/
   ├── rgb/              {frame_id:06d}.png
   ├── depth/            {frame_id:06d}.png  (16-bit)
   ├── mask/             {frame_id:06d}_{obj_id:06d}.png
   ├── mask_visib/       {frame_id:06d}_{obj_id:06d}.png
   ├── scene_camera.json
   ├── scene_gt.json
   └── scene_gt_info.json


LeRobotWriter
~~~~~~~~~~~~~

:Registry name: ``lerobot``
:GPU required: Yes (NVENC via PyNvVideoCodec)
:Required render vars: ``LdrColor``
:Extra dependencies: ``[lerobot]`` (pyarrow, Parquet output), ``[video]`` (PyNvVideoCodec, NVENC video encoding)

Writes dataset in HuggingFace LeRobot format for robot learning.

**Config:** ``LeRobotWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``fps``
     - ``int``
     - ``30``
     - Frames per second of episode video
   * - ``episode_index``
     - ``int``
     - ``0``
     - Episode index
   * - ``task_description``
     - ``str``
     - ``""``
     - Task description string
   * - ``camera_keys``
     - ``list[str]``
     - ``["top"]``
     - Camera names in output paths
   * - ``frames_per_chunk``
     - ``int``
     - ``1000``
     - Frames per chunk directory

**Output structure:**

::

   {root}/
   ├── data/
   │   └── chunk-{chunk:03d}/
   │       └── episode_{episode_id:06d}.parquet
   ├── videos/
   │   └── chunk-{chunk:03d}/
   │       └── observation.images.{camera_key}/
   │           └── episode_{episode_id:06d}.mp4
   └── meta/
       ├── info.json
       ├── tasks.json
       ├── episodes.json
       └── stats.json

**Parquet schema (per frame):**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Column
     - Type
     - Description
   * - ``index``
     - int64
     - Global monotonic frame counter
   * - ``episode_index``
     - int64
     - Episode ID
   * - ``frame_index``
     - int64
     - Frame within episode
   * - ``timestamp``
     - float64
     - frame_index / fps
   * - ``task_index``
     - int64
     - Always 0 (single-task)


CosmosTransferWriter
~~~~~~~~~~~~~~~~~~~~

:Registry name: ``cosmos_transfer``
:GPU required: Yes (NVENC via PyNvVideoCodec for video encoding)
:Required render vars: ``LdrColor``, ``DistanceToCameraSD``, ``SemanticInstanceSegmentation`` (plus ``NormalSD`` when ``"shaded_segmentation"`` conditioning or a normal-based ``canny_source`` is used)
:Extra dependencies: ``[video]`` for PyNvVideoCodec

Writes training data for NVIDIA Cosmos Transfer 2.5 models — RGB video plus conditioning signal videos.

**Config:** ``CosmosTransferWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 22 22 18 38

   * - Parameter
     - Type
     - Default
     - Description
   * - ``fps``
     - ``int``
     - ``24``
     - Output video frame rate
   * - ``clip_index``
     - ``int``
     - ``0``
     - Index used to name this clip's output files, zero-padded to six digits
       (e.g. ``0`` → ``videos/000000_rgb.mp4``). Give each clip a distinct value
       when generating a multi-clip dataset.
   * - ``conditioning_types``
     - ``list[str]``
     - ``["canny", "depth", "segmentation"]``
     - Conditioning signal videos to produce. Valid values: ``"canny"``,
       ``"depth"``, ``"segmentation"``, ``"shaded_segmentation"``.
   * - ``canny_source``
     - ``str``
     - ``"rgb"``
     - Render var the Canny edges are derived from (``"rgb"``, ``"normals"``,
       ``"segmentation"``, ``"shaded_segmentation"``)
   * - ``canny_sigma``
     - ``float``
     - ``1.0``
     - Gaussian sigma for Canny
   * - ``canny_low_threshold``
     - ``float``
     - ``0.1``
     - Low hysteresis threshold
   * - ``canny_high_threshold``
     - ``float``
     - ``0.2``
     - High hysteresis threshold
   * - ``shaded_seg_light_dir``
     - ``tuple[float, float, float]``
     - ``(0.3, 0.5, 0.81)``
     - Light direction used to shade segmentation for ``"shaded_segmentation"``
       (normalized internally)

**Output structure:**

::

   {root}/
   ├── videos/
   │   └── {clip_id:06d}_rgb.mp4
   ├── conditioning/
   │   ├── {clip_id:06d}_canny.mp4
   │   ├── {clip_id:06d}_depth.mp4
   │   └── {clip_id:06d}_segmentation.mp4
   ├── thumbnails/
   │   └── {clip_id:06d}.jpg
   └── manifest.json

**Multiple clips:**

A single writer can emit several clips that share one ``manifest.json``.
Accumulate frames with ``write_frame()``, then call ``cut(backend)`` to encode
the buffered frames into a clip and start a new one (the ``clip_index``
advances automatically); ``finalize(backend)`` flushes the last clip. Use this
to slice a long capture into sequential clips, or to emit one clip per scene
variation (modify the scene, render more frames, then ``cut`` again).
``cut()`` is a no-op when no frames are buffered and returns the written
``clip_id`` (or ``None``).

.. code-block:: python

   for frame_id in range(num_frames_a):
       writer.write_frame(frame_id, render_vars_a, backend)
   writer.cut(backend)              # clip 000000

   # ... modify the scene / continue the capture ...
   for frame_id in range(num_frames_b):
       writer.write_frame(frame_id, render_vars_b, backend)
   writer.finalize(backend)         # clip 000001

**GPU behavior:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - GPU available
     - CPU only
   * - Canny edge detection
     - Warp kernels on CUDA (~1ms @ 720p)
     - Same Warp kernels on Warp's CPU device
   * - Image encoding (PNG/JPEG)
     - nvImageCodec via the ``[gpu]`` extra (zero-copy for on-device frames)
     - Pillow (CPU) — automatic fallback
   * - Video encoding
     - PyNvVideoCodec NVENC (H.264)
     - N/A (GPU required)

Thumbnails and frame resizing use Pillow (CPU). Canny runs the same Warp
kernels (``ovannotators.writers.canny``) on whatever device the data lives on —
CUDA when present, otherwise Warp's CPU device — with no separate code path and
no ``prefer_gpu`` switch.


Backends
--------

Backend Protocol
~~~~~~~~~~~~~~~~

.. code-block:: python

   class Backend(Protocol):
       def write(self, path: str, data: bytes) -> None: ...
       def write_array(self, path: str, array: np.ndarray, encoding: str = "npy") -> None: ...
       def read(self, path: str) -> bytes: ...
       def list(self, prefix: str = "") -> list[str]: ...
       def exists(self, path: str) -> bool: ...
       def makedirs(self, path: str) -> None: ...

All paths are relative to the backend's configured root.


DiskBackend
~~~~~~~~~~~

Writes to local filesystem.

.. code-block:: python

   from ovannotators import DiskBackend, DiskBackendConfig

   backend = DiskBackend(DiskBackendConfig(
       root_dir="/data/output",
       overwrite=True,
   ))

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Config
     - Type
     - Default
     - Description
   * - ``root_dir``
     - ``str``
     - (required)
     - Output directory
   * - ``overwrite``
     - ``bool``
     - ``False``
     - Allow overwriting existing files

Image encoding always prefers the GPU automatically when one is available (see
`Image Codecs`_); there is no per-backend codec switch.


S3Backend
~~~~~~~~~

Writes to Amazon S3 or compatible object stores.

.. code-block:: python

   from ovannotators import S3Backend, S3BackendConfig

   backend = S3Backend(S3BackendConfig(
       bucket="my-datasets",
       prefix="coco/run-001/",
       region="us-east-1",
   ))

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Config
     - Type
     - Default
     - Description
   * - ``bucket``
     - ``str``
     - (required)
     - S3 bucket name
   * - ``prefix``
     - ``str``
     - ``""``
     - Key prefix
   * - ``region``
     - ``str | None``
     - ``None``
     - AWS region
   * - ``endpoint_url``
     - ``str | None``
     - ``None``
     - Custom endpoint (MinIO, etc.)
   * - ``multipart_threshold_mb``
     - ``int``
     - ``8``
     - Object size (MB) above which multipart upload is used

**Requires:** ``pip install "ovannotators[s3]"``


StorageAPIBackend
~~~~~~~~~~~~~~~~~

Writes to an HTTP storage endpoint.

.. code-block:: python

   from ovannotators import StorageAPIBackend, StorageAPIBackendConfig

   backend = StorageAPIBackend(StorageAPIBackendConfig(
       base_url="https://storage.example.com/api/v1",
       api_key="my-auth-token",
   ))

.. list-table::
   :header-rows: 1
   :widths: 18 20 18 44

   * - Config
     - Type
     - Default
     - Description
   * - ``base_url``
     - ``str``
     - (required)
     - Storage API base URL
   * - ``api_key``
     - ``str | None``
     - ``None``
     - Bearer auth token; falls back to the ``STORAGE_API_KEY`` env var
   * - ``prefix``
     - ``str``
     - ``""``
     - Path prefix prepended to every key
   * - ``timeout_s``
     - ``float``
     - ``30.0``
     - Per-request timeout in seconds
   * - ``verify_ssl``
     - ``bool``
     - ``True``
     - Verify TLS certificates

**Requires:** ``pip install "ovannotators[storage-api]"``


Renderer Outputs
----------------

OvrtxOutput
~~~~~~~~~~~

Extracts render vars from ovrtx ``FrameOutput`` objects as zero-copy Warp arrays
via DLPack (annotator byte buffers such as bounding boxes and camera params are
decoded into structured NumPy arrays).

.. code-block:: python

   from ovannotators import OvrtxOutput

   extractor = OvrtxOutput()

   # CUDA view by default (params remain on the CPU); keeps the data on the
   # device where ovrtx rendered it, ready for zero-copy GPU encode.
   rgb_gpu = extractor.extract(frame_output, "LdrColor")

   # Host (CPU) view (forces a device-to-host copy)
   rgb = extractor.extract(frame_output, "LdrColor", device="cpu")

.. list-table::
   :header-rows: 1
   :widths: 44 26 30

   * - Method
     - Returns
     - Description
   * - ``extract(frame_output, var_name, device="cuda", decode=True)``
     - ``wp.array`` (or decoded ``np.ndarray`` / dict)
     - Zero-copy DLPack view on ``"cuda"`` (default) or ``"cpu"``; annotator
       buffers are decoded to structured NumPy arrays when ``decode=True``


Types
-----

.. code-block:: python

   from ovannotators.types import Resolution, BBox2D, BBox3D, CameraParams, PointCloudResult

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Description
   * - ``Resolution``
     - Frozen dataclass ``(width: int, height: int)``
   * - ``BBox2D``
     - 2D bounding box (x_min, y_min, x_max, y_max, semantic_id, instance_id, label, occlusion_ratio=0.0)
   * - ``BBox3D``
     - 3D oriented bbox (center, extent, rotation, semantic_id, instance_id, label, coordinate_system)
   * - ``CameraParams``
     - Camera intrinsics (K) + extrinsics (R, t) + resolution, near, far, fov_x, fov_y
   * - ``PointCloudResult``
     - Dense point cloud (points, normals, colors, semantic_ids)


Exceptions
----------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Exception
     - Parent
     - When raised
   * - ``OVAnnotatorsError``
     - ``Exception``
     - Base for all ovannotators errors
   * - ``ConfigurationError``
     - ``OVAnnotatorsError``
     - Invalid config values
   * - ``ExtractionError``
     - ``OVAnnotatorsError``
     - RendererOutput failed to extract/decode a var from a FrameOutput
   * - ``ValidationError``
     - ``OVAnnotatorsError``
     - A frame is missing an expected/required render var
   * - ``MissingRenderVarError``
     - ``ValidationError``
     - Writer requires a render var that was not provided
   * - ``BackendWriteError``
     - ``OVAnnotatorsError``
     - Backend write operation failed
   * - ``BackendReadError``
     - ``OVAnnotatorsError``
     - Backend read operation failed


Configuration Serialization
---------------------------

Every ``*WriterConfig`` and ``*BackendConfig`` is a plain
:func:`dataclasses.dataclass`, which makes configs easy to persist and reload.
Two helpers handle the round-trip, including ``pathlib.Path`` fields and nested
config dataclasses that a bare ``dataclasses.asdict`` + ``Config(**data)`` would
not:

.. code-block:: python

   config_to_dict(config) -> Any          # config dataclass -> JSON-friendly dict
   config_from_dict(config_cls, data)     # dict -> config dataclass instance

- ``config_to_dict`` recurses through nested config dataclasses, converts
  ``Path`` values to ``str``, and turns ``tuple``/``set``/``frozenset`` into
  ``list`` so the result dumps cleanly via ``json``/``yaml``/``toml``.
- ``config_from_dict`` is the inverse: it reconstructs nested configs, coerces
  ``Path`` fields from strings, falls back to dataclass defaults for missing
  keys, and raises ``TypeError`` for unknown keys.

.. code-block:: python

   import json
   from ovannotators import DiskBackendConfig, config_from_dict, config_to_dict

   config = DiskBackendConfig(root_dir="./output/coco", overwrite=True)
   with open("backend.json", "w") as f:
       json.dump(config_to_dict(config), f, indent=2)

   with open("backend.json") as f:
       loaded = config_from_dict(DiskBackendConfig, json.load(f))

Pair a registry name with its config payload in one file to drive an entire
pipeline from data (resolve the config class per registry name, then dispatch via
``BackendRegistry`` / ``WriterRegistry``).

.. note::

   Fields typed as an ambiguous union such as ``Union[str, Path]`` (e.g.
   ``DiskBackendConfig.root_dir``) keep the serialized value as-is on load; the
   backend/writer normalizes it (``DiskBackend`` wraps ``root_dir`` in ``Path``).


Custom Writers
--------------

Register a custom writer via ``@WriterRegistry.register`` or entry points:

.. code-block:: python

   from ovannotators.writers import WriterRegistry
   from ovannotators.backends._base import Backend

   @WriterRegistry.register
   class MyWriter:
       name = "my_format"
       required_render_vars = frozenset({"LdrColor", "depth"})

       def write_frame(self, frame_id, render_vars, backend):
           backend.write_array(
               f"{frame_id:06d}/color.png",
               render_vars["LdrColor"],
               encoding="png",
           )

       def finalize(self, backend):
           pass

**Entry point registration** (auto-discovered on install):

.. code-block:: toml

   # pyproject.toml
   [project.entry-points."ovannotators.writers"]
   my_format = "mypackage.writers:MyWriter"


Custom Backends
---------------

Implement the ``Backend`` protocol:

.. code-block:: python

   class MyBackend:
       def write(self, path: str, data: bytes) -> None: ...
       def write_array(self, path: str, array: np.ndarray, encoding: str = "npy") -> None: ...
       def read(self, path: str) -> bytes: ...
       def list(self, prefix: str = "") -> list[str]: ...
       def exists(self, path: str) -> bool: ...
       def makedirs(self, path: str) -> None: ...

The ``write_array`` encoding parameter accepts: ``"npy"``, ``"png"``, ``"jpg"``, ``"exr"``.


Image Codecs
~~~~~~~~~~~~

``ovannotators`` uses a tiered codec system for image I/O:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Format
     - Primary
     - Fallback
   * - PNG / JPEG
     - ``nvidia-nvimgcodec-cu12`` (GPU; ~5–6× faster for PNG, and zero-copy for GPU-resident frames)
     - Pillow (CPU)
   * - EXR
     - ``OpenEXR`` (lossless float16/32)
     - N/A
   * - NPY / NPZ
     - NumPy
     - N/A

The codec backend is selected automatically by ``get_image_codec()``: it returns
the GPU ``NvImageCodecBackend`` whenever a CUDA device and ``nvidia-nvimgcodec``
are both available, and falls back to ``PillowCodecBackend`` otherwise. There is
no ``prefer_gpu`` argument — the GPU is always preferred when present:

.. tip::

   **Install the GPU codec when a GPU is available.** On a CUDA host, add the
   ``[gpu]`` extra (``pip install "ovannotators[gpu]"``) so image encoding uses
   nvImageCodec instead of Pillow — several times faster for PNG/JPEG. When
   frames are already GPU-resident (e.g. from ``OvrtxOutput(device="cuda")``),
   nvImageCodec encodes them **directly on the device with no host copy**.
   Without the extra (or without a CUDA device), ovannotators transparently
   falls back to Pillow on the CPU — correct, just slower; no code changes
   needed either way.

.. code-block:: python

   from ovannotators._internal.codecs import get_image_codec

   codec = get_image_codec()         # NvImageCodecBackend if a GPU is available, else Pillow
   codec.write("output.png", array)  # extension selects the format
   data = codec.encode(array, format="jpg", quality=85)  # -> bytes
   array = codec.decode(data)                     # bytes -> np.ndarray

For EXR files (HDR data like ``HdrColor``), ``read_exr`` is the public reader:

.. code-block:: python

   from ovannotators._internal.codecs import read_exr

   data = read_exr("hdr_color.exr")  # returns a float16 ndarray

EXR writing happens through ``Backend.write_array(..., encoding="exr")`` rather
than a public codec helper.
