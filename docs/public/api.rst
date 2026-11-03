.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

API Reference
=============

Complete reference for the ``ovannotators`` public API.

.. contents:: Contents
   :local:
   :depth: 2


Choose a component
------------------

``ovannotators`` has three composable roles: a renderer-output extractor turns
a renderer frame into arrays, a writer turns those arrays into a dataset
layout, and a backend persists the resulting files. Most integrations begin by
choosing a writer, then a backend.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - If you need to…
     - Start with
   * - Export a standard dataset or direct per-frame files
     - :ref:`Built-in writers <api-writers>`
   * - Write to a local directory, `Amazon S3 <https://aws.amazon.com/s3/>`_,
       or an ovstorage Stack
     - :ref:`Built-in backends <api-backends>`
   * - Extract data from an `ovrtx <https://github.com/NVIDIA-Omniverse/ovrtx>`_
       frame
     - :ref:`Renderer outputs <api-renderer-outputs>`
   * - Recreate a writer/backend setup from stored configuration
     - :ref:`Configuration serialization <api-configuration-serialization>`



.. _api-dataset-requirements:

Dataset profiles and capture requirements
-----------------------------------------

Start with :func:`ovannotators.list_profiles` and
:func:`ovannotators.describe_requirements` to discover what a dataset writer
needs before rendering. The first listed profile is the recommended default.
The same API accepts exact identifiers from ``WriterRegistry`` and
``ProfileWriterRegistry``; it never constructs writers or opens a stage.

.. list-table:: Built-in profiles
   :header-rows: 1
   :widths: 30 35 35

   * - Writer identifier
     - Profiles (recommended first)
     - Capture guidance
   * - ``coco``
     - ``detection``, ``instance_segmentation``, ``panoptic``
     - Color and tight 2D boxes; mask profiles also require instance segmentation.
   * - ``yolo``
     - ``detection``, ``instance_segmentation``
     - Color and tight 2D boxes; segmentation requires instance masks.
   * - ``basic``
     - ``default``
     - Recommends color; override ``render_vars`` for specific camera/sensor outputs.
   * - ``kitti``
     - ``default``
     - Color, 2D/3D boxes, and camera parameters; optional depth.
   * - ``bop``
     - ``default``
     - Color, depth, 2D/3D boxes, masks, and camera parameters.
   * - ``lerobot``
     - ``default``
     - Legacy image-only layout; camera keys duplicate one incoming image.
   * - ``cosmos_transfer``
     - ``default``
     - Color plus selected conditioning inputs; depth is 8-bit inverse-depth video.
   * - ``cosmos_multiview``
     - ``default``
     - Caller-defined stream schema, alignment, representations, and QA policies.
   * - ``groot_lerobot_v2``
     - ``default``
     - Explicit camera, state/action slice, embodiment, and relative-action mapping on a fixed-rate timeline; RGB uint8 with even dimensions, NVENC video.
   * - ``umi_replay_buffer``
     - ``default``
     - Caller-defined observation and policy-signal mappings on a fixed-rate timeline; RGB as HWC uint8.

COCO discovery to dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

``render_vars`` names are writer input keys (ovrtx ``sourceName`` values),
not USD prim paths. Request those outputs on a camera's RenderProduct and
include ``recommended_render_vars`` for readable semantic category names.
One view is sufficient for the frame writer; this is not a claim about
coverage sufficient for a training task. The producer owns cameras, scene
labels, RenderProducts, and renderer configuration.

For a renderer-backed version that authors that RenderProduct from the
requirements and renders it with ``ovrtx``, run
``python examples/requirements_capture.py --writer coco --profile instance_segmentation``
from the repository root. It needs an NVIDIA RTX-capable GPU, CUDA, ``ovrtx``,
and ``ovstage``.

Given a ``render_vars`` mapping containing the COCO instance-segmentation
inputs and an ``output_dir`` path, this CPU-only example lists the COCO
profiles, selects the requested arrays for ``instance_segmentation``, and
writes a dataset:

.. literalinclude:: ../../tests/docs/python/test_requirements.py
   :language: python
   :start-after: # [snippet:doc-dataset-requirements]
   :end-before: # [/snippet:doc-dataset-requirements]
   :dedent: 4

The returned ``requirements.config`` is passed unchanged to the writer.
Explicit keyword options override defaults. A config dataclass or
partial config mapping is also accepted through ``config=``. If ``profile`` is
omitted, explicit task options select the matching profile; otherwise the
recommended default is used. Explicit profile constraints cannot be overridden:
choose a different profile to change its task.

COCO ``panoptic`` describes the existing instance-derived export; it does not
add a stuff-label authoring API. No unsupported COCO tasks are advertised.

Representations and unresolved choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``options`` lists all writer config fields, resolved defaults, and Literal
choices. ``required_options`` names fields the caller must supply before
``config`` can be constructed. Defaults never invent robot semantics, stage
units, geometry, or an owner-supplied QA policy. ``minimum_camera_views=None``
means the count depends on the unresolved schema. ``camera_stream_ids`` names
explicit independent views; ``sensor_requirements`` provides guidance for
sensor-only outputs. Robot/simulation signals are in ``non_render_inputs``.

Capture depth values and output depth encodings are separate. The Cosmos
Transfer writer emits normalized 8-bit inverse-depth conditioning, which is not
physical depth; its requirements report that encoding under
``representations["depth"]``. Use ``BasicWriter`` for raw depth arrays. LeRobot
v3 currently exposes RGB plus state/action, without claiming depth support.

.. literalinclude:: ../../tests/docs/python/test_requirements.py
   :language: python
   :start-after: # [snippet:doc-depth-requirements]
   :end-before: # [/snippet:doc-depth-requirements]
   :dedent: 4

Extension writers
~~~~~~~~~~~~~~~~~

Use :func:`ovannotators.register_dataset_profiles` to register a config type,
named profiles, and a side-effect-free requirements resolver for a writer
class. Both frame and episode writers are supported. The resolver receives
resolved values, including partial values when required fields remain missing.
A writer without metadata gets only its class-level required inputs and an
explicit incomplete-discovery note; options are not inferred.

.. automodule:: ovannotators.requirements
   :members:


Module: ``ovannotators``
------------------------

Selected top-level exports (see ``ovannotators.__all__`` for the complete list):

.. code-block:: python

   from ovannotators import (
       # Version
       __version__,
       # Writers
       Writer, WriterRegistry,
       BasicWriter, BasicWriterConfig,
       COCOWriter, COCOWriterConfig,
       KITTIWriter, KITTIWriterConfig,
       YOLOWriter, YOLOWriterConfig,
       BOPWriter, BOPWriterConfig,
       LeRobotWriter, LeRobotWriterConfig,
       CosmosTransferWriter, CosmosTransferWriterConfig,
       # Episode-aware profile writers
       ProfileWriter, ProfileWriterRegistry,
       EpisodeEnvelope, SampleEnvelope, StreamDescriptor, NamedTensor,
       ProfileMetadata, WriterRecord, LossReport, ConformanceResult,
       CosmosMultiviewWriter, CosmosMultiviewWriterConfig,
       CosmosMaterializationConfig, CosmosQAConfig,
       CosmosValidationFinding, CosmosValidationRecord,
       UMIReplayBufferWriter, UMIReplayBufferWriterConfig, UMIFieldSpec,
       GrootLeRobotV2Writer, GrootLeRobotV2WriterConfig,
       GrootVideoModality, GrootStateModality, GrootActionModality,
       GrootNormalizationConfig, validate_groot_lerobot_v2,
       # Backends
       Backend, BackendRegistry,
       DiskBackend, DiskBackendConfig,
       S3Backend, S3BackendConfig,
       OvstorageBackend, OvstorageBackendConfig,
       # Renderer outputs
       RendererOutput, RendererOutputRegistry, OvrtxOutput,
       # Config serialization
       config_to_dict, config_from_dict,
       # Types
       Resolution, BBox2D, BBox3D, CameraParams, PointCloudResult,
       # Exceptions
       OVAnnotatorsError, ConfigurationError, ExtractionError,
       ValidationError, MissingRenderVarError,
       BackendWriteError, BackendReadError, PackagingError,
   )


Episode-aware profile writers
-----------------------------

Exact training-dataset profiles use a separate additive lifecycle. Existing
frame writers and ``WriterRegistry`` are unchanged.

.. code-block:: python

   from typing import ClassVar, Protocol

   from ovannotators import (
       Backend, ConformanceResult, EpisodeEnvelope, SampleEnvelope,
   )


   class ProfileWriter(Protocol):
       name: ClassVar[str]

       def begin_episode(self, episode: EpisodeEnvelope, backend: Backend) -> None: ...
       def write_sample(self, sample: SampleEnvelope, backend: Backend) -> None: ...
       def end_episode(self, backend: Backend) -> None: ...
       def finalize(self, backend: Backend) -> None: ...
       def validate(self, backend: Backend) -> ConformanceResult: ...

``EpisodeEnvelope`` declares dataset, sequence, episode, task, embodiment,
source clock/origin, nominal rate, independent stream descriptors, and
provenance. ``SampleEnvelope`` carries sample/frame identity, its authoritative
signed nanosecond timestamp, named observations, policy signals, and explicit
missing-data metadata. ``NamedTensor`` retains feature names, units,
representation, and coordinate frame.

Completed profiles reserve ``_ovannotators/`` for ``profile.json``,
``writer_record.json``, ``loss_report.json``, ``conformance.json``, payload
checksums, and ``_SUCCESS``. The success marker certifies only completed
serialization and structural validation. Canonical-loader, semantic, and named
consumer acceptance are recorded separately and are not implied by the marker.

Third-party profile writers register with ``@ProfileWriterRegistry.register``
or the ``ovannotators.profile_writers`` entry-point group.

LeRobot v3 and LeRobot Lance Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lerobot_v3`` and ``lerobot_lance`` profile writers are coming soon. They
are not part of 0.2.0 because they depend on ``lerobot`` 0.6.1, which requires
Pillow earlier than 13, and 0.2.0 requires Pillow 13. See
:doc:`writers/lerobot_v3`.

Cosmos 3.5 Multiview profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Registry name: ``cosmos_multiview``
:GPU required: No for registration and NumPy asset materialization
:Extra: ``cosmos-multiview`` (``pylance==11.0.0``, ``pyarrow==25.0.1``)

``CosmosMultiviewWriter`` is separate from the frame-oriented
``cosmos_transfer`` writer. It implements two explicit candidate-producing
stages: ``register_episodes(backend)`` writes a deterministic
``episodes.lance`` whose ``streams`` column is native Arrow ``list<struct>``,
then ``materialize_samples(backend)`` aligns by authoritative signed nanosecond
timestamps and writes immutable content-addressed assets plus ``samples.lance``.
Window starts advance from the provider-defined episode origin by ``stride_ns``.
Only complete half-open windows are materialized; each asset contains the full
sequence at ``target_rate_hz`` and retains its target and contributing source
timestamps. Linear interpolation requires source samples on both sides of the
target; it never silently degrades to nearest neighbor. The ``assets_json``
column is canonical JSON.

``CosmosMultiviewWriterConfig`` requires an absolute immutable output URI,
source identity and ownership, supported URI schemes, a
``CosmosMaterializationConfig``, and a versioned ``CosmosQAConfig``. Numeric QA
thresholds and warning policy have no defaults. Sample windows are half-open;
seconds columns are derived from integer nanoseconds, never from frame indexes.
The current materializer requires ``minimum_stream_coverage=1.0`` because every
accepted asset covers its complete target grid.

Output layout:

.. code-block:: text

   episodes.lance/                       # one row per source episode
   samples.lance/                        # one row per task/window
   metadata/dataset_info.json
   metadata/sample_config.json
   sources/<episode>/<stream>/
     data.npy                             # immutable raw source sequence
     stream.json
     timestamps.parquet
     calibration.json                    # when geometry is declared
   objects/<sha-prefix>/<sha>.npy
   objects/<sha-prefix>/<sha>.timestamps.parquet
   validation/episode.json               # status remains unverified
   validation/sample.json                # status remains unverified
   _ovannotators/...

Lance is always built in a temporary local staging directory and then uploaded
as a complete immutable tree through the selected backend. ``_SUCCESS`` is
written last and certifies serialization and structural validation only. The
writer never self-assigns Cosmos QA acceptance, canonical-loader acceptance, or
consumer usefulness. ``validate(backend)`` reads only persisted resources, so a
fresh writer instance can validate a completed output and verify its
``_SUCCESS`` marker and complete payload-checksum manifest. Validation also
enforces exact Arrow types and schema metadata and resolves every
``dataset_info_uri`` and ``sample_config_uri`` through the backend.

UMI ReplayBuffer Profile
~~~~~~~~~~~~~~~~~~~~~~~~

:Registry name: ``umi_replay_buffer``
:Target: UMI/Diffusion Policy Zarr v2 ReplayBuffer zip
:GPU required: No
:Extra dependencies: ``[umi]`` for Zarr v2 and JPEG-XL codecs (``zarr>=2.16,<3``; Zarr 3 is rejected at runtime)

``UMIReplayBufferWriter`` writes one zip (default ``dataset.zarr.zip``)
containing equal-length arrays under ``data/`` and cumulative exclusive episode
boundaries under ``meta/episode_ends``. ``UMIFieldSpec`` makes every observation
or policy-signal mapping explicit. RGB arrays use one-frame chunks and JPEG-XL
compression; low-dimensional arrays use bounded time chunks and no compression.
Authoritative nanosecond timestamps are retained in ``data/timestamp_ns``.

The name deliberately identifies the ReplayBuffer contract rather than
claiming every dataset described as UMI-compatible. Samples must lie on the
configured fixed-rate timeline and all data arrays share the same leading time
dimension.

``UMIFieldSpec`` fields:

- ``stream_id``: envelope stream to read. Required.
- ``output_key``: array name under ``data/``; a direct child, so it cannot
  contain ``/``. Required.
- ``source``: ``"observation"`` or ``"policy_signal"``, selecting which
  ``SampleEnvelope`` mapping supplies the tensor. Required.
- ``dtype`` and ``shape``: exact per-sample tensor dtype and shape. Samples and
  episode stream descriptors that differ are rejected. Required.
- ``feature_names``: optional names for every flattened tensor element. When
  set, each sample's ``NamedTensor.feature_names`` must match.
- ``units``: optional units, either one value or one per feature name. When
  set, each tensor's ``units`` must match.
- ``representation`` and ``coordinate_frame``: optional semantics checked
  against the tensor; an empty ``representation`` is not checked, and
  ``coordinate_frame`` must equal the tensor's value (``None`` by default).
- ``rgb``: marks an HWC ``uint8`` image field that is JPEG-XL compressed
  (default ``False``).

``UMIReplayBufferWriterConfig`` fields:

- ``frequency_hz``: fixed sample rate of the timeline; must be positive and
  must match each episode's ``nominal_rate_hz``. Required.
- ``fields``: non-empty tuple of ``UMIFieldSpec``. Output keys and
  ``(source, stream_id)`` pairs must be unique. Required.
- ``output_path``: backend path of the zip; must end with ``.zarr.zip``
  (default ``dataset.zarr.zip``).
- ``timestamp_tolerance_ns``: accepted error between a sample timestamp and its
  slot on the fixed-rate timeline (default 100,000 ns, that is 100 us).
- ``rgb_compression_level``: JPEG-XL level for RGB fields, 0 to 100
  (default 99).
- ``low_dim_chunk_steps``: time steps per chunk for non-RGB arrays
  (default 1024).

.. note::

   The writer buffers every episode in memory until ``finalize``. It then
   concatenates the arrays, builds the zip in a temporary directory, and reads
   the zip back into memory before ``backend.write``, so peak memory exceeds
   the raw dataset size: each array is held once per episode and again once
   concatenated, plus the compressed zip. Size the dataset to the available
   RAM, or write several smaller datasets.


GR00T LeRobot v2 Profile
~~~~~~~~~~~~~~~~~~~~~~~~

:Registry name: ``groot_lerobot_v2``
:Pinned consumer: Isaac-GR00T ``n1.7-release`` at commit ``23ace64f17aa5015259b8609d371eb61a357c776``
:GPU required: Yes for H.264/NVENC video serialization and canonical loader acceptance
:Extra dependencies: ``[lerobot]`` (pyarrow), ``[video]`` (PyNvVideoCodec)

This independently named profile emits the GR00T LeRobot ``v2.1``-derived layout.
It does not claim that ``lerobot`` or a LeRobot v3 dataset is GR00T-compatible.
Its contract follows the pinned `Isaac-GR00T data configuration
<https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/getting_started/data_config.md>`_
and `N1.7 loader
<https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/data/dataset/lerobot_episode_loader.py>`_.
The constructor requires explicit mappings for every video stream and every
state/action slice, plus embodiment, action horizon, representation,
normalization, clipping, and relative-action policy. The writer stores absolute
state/action values and, when configured, generates horizon-shaped
``relative_stats.json`` for GR00T's relative-action processor.
Relative non-EEF groups require a reference state slice of the same width.
Relative EEF groups are rejected until the pinned pose-composition algorithm
is implemented; they are never approximated with element-wise subtraction.
``canonical_modality_config()`` builds the pinned loader and processor config
directly from the writer's explicit contract in a GR00T-enabled environment.

**Config:** ``GrootLeRobotV2WriterConfig`` with nested
``GrootVideoModality``, ``GrootStateModality``, ``GrootActionModality``, and
``GrootNormalizationConfig`` values. No modality configuration is inferred
from arbitrary column names.

.. list-table:: ``GrootLeRobotV2WriterConfig``
   :header-rows: 1
   :widths: 24 22 12 42

   * - Parameter
     - Type
     - Default
     - Description
   * - ``embodiment``
     - ``str``
     - required
     - Embodiment tag. Every episode's ``embodiment`` must match it; written to
       ``robot_type`` in ``meta/info.json``
   * - ``video_modalities``
     - ``tuple[GrootVideoModality, ...]``
     - required
     - One entry per camera, mapping a GR00T video key to an RGB stream ID.
       Keys and stream IDs must be unique
   * - ``state_signal``
     - ``str``
     - required
     - Stream ID of the observation tensor stored as ``observation.state``
   * - ``state_modalities``
     - ``tuple[GrootStateModality, ...]``
     - required
     - Ordered, contiguous slices of ``observation.state``
   * - ``action_signal``
     - ``str``
     - required
     - Stream ID of the policy signal stored as ``action``
   * - ``action_modalities``
     - ``tuple[GrootActionModality, ...]``
     - required
     - Ordered, contiguous slices of ``action``
   * - ``action_horizon``
     - ``int``
     - required
     - Number of action steps the GR00T processor reads per sample (at least 1)
   * - ``normalization``
     - ``GrootNormalizationConfig``
     - required
     - Normalization behavior expected by the GR00T processor
   * - ``relative_action_policy``
     - ``"disabled"`` or ``"groot_processor"``
     - required
     - Must be ``"groot_processor"`` exactly when an action group is relative
   * - ``fps``
     - ``int``
     - ``30``
     - Video frame rate and the constant-frame-rate timeline the writer enforces
   * - ``episodes_per_chunk``
     - ``int``
     - ``1000``
     - Episodes per ``chunk-NNN`` directory; written to ``chunks_size``

.. list-table:: Nested modality configs
   :header-rows: 1
   :widths: 28 72

   * - Type and fields
     - Description
   * - ``GrootVideoModality(key, stream_id)``
     - Camera key (``observation.images.{key}``) and the envelope stream that
       supplies its ``uint8`` HxWx3 or HxWx4 frames. Alpha is dropped and
       recorded in the loss report
   * - ``GrootStateModality(key, start, end, feature_names, units)``
     - One state group occupying ``[start, end)`` of ``observation.state``.
       ``feature_names`` and ``units`` must each have ``end - start`` entries
   * - ``GrootActionModality(key, start, end, feature_names, units, representation, control_type, action_format, state_key=None)``
     - One action group occupying ``[start, end)`` of ``action``.
       ``representation`` is ``"absolute"`` or ``"relative"``; ``control_type``
       is ``"eef"`` or ``"non_eef"``; ``action_format`` is ``"default"``,
       ``"xyz_rot6d"``, or ``"xyz_rotvec"``. ``state_key`` names the reference
       state group for relative actions and defaults to the action key.
       Stored values are always absolute
   * - ``GrootNormalizationConfig(state, action, use_percentiles, clip_outliers)``
     - ``state`` and ``action`` are tuples of ``(key, mode)`` pairs covering
       every state and action group exactly once. State modes are
       ``"min_max"``, ``"mean_std"``, or ``"sin_cos"``; action modes are
       ``"min_max"`` or ``"mean_std"``. ``use_percentiles`` and
       ``clip_outliers`` are recorded in ``meta/modality.json``;
       ``clip_outliers`` is also recorded in the loss report. ``meta/stats.json``
       always carries ``q01`` and ``q99`` next to ``min`` and ``max``

Invalid combinations raise ``ConfigurationError`` when the config is
constructed. ``GROOT_RELEASE_TAG`` and ``GROOT_RELEASE_COMMIT`` in
``ovannotators.profile_writers`` expose the pinned Isaac-GR00T release.

**Output structure:**

::

   {root}/
   ├── data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
   ├── videos/chunk-{episode_chunk:03d}/
   │   └── observation.images.{camera}/episode_{episode_index:06d}.mp4
   ├── meta/
   │   ├── info.json
   │   ├── episodes.jsonl
   │   ├── tasks.jsonl
   │   ├── modality.json
   │   ├── stats.json
   │   └── relative_stats.json  (when relative actions are configured)
   └── _ovannotators/
       ├── profile.json
       ├── writer_record.json
       ├── loss_report.json
       ├── conformance.json
       ├── payload_checksums.json
       └── _SUCCESS

``_SUCCESS`` records serialization and structural validation only. A pinned
GR00T loader/training result and consumer acceptance remain independent fields
in ``conformance.json`` and cannot be replaced by raw Parquet parsing.

**Video input requirements:** each configured camera stream must declare and
supply ``uint8`` HxWx3 or HxWx4 frames of a constant size. H.264 4:2:0 needs an
even height and width, and this exact-conformance profile does not crop or pad
frames, so ``begin_episode`` raises ``ConfigurationError`` for an odd height or
width before any file is written.

**Memory:** the writer holds every raw frame of the active episode in memory
until ``end_episode`` encodes it, so peak memory grows as
``frames x cameras x height x width x 3`` bytes. For example, a 30 s episode at
30 fps (900 frames) with two 640x480 cameras holds about
``900 x 2 x 640 x 480 x 3`` = 1.66 GB. Size episodes, or split long recordings
into several episodes, to fit available RAM.

**Timeline contract:** ``write_sample`` requires each sample's ``frame_id`` to
equal its zero-based position within the episode, and its ``timestamp_ns``
(relative to the episode's ``source_origin_ns``) to sit on the ``fps`` grid
within 100 microseconds. A sample that
violates either rule raises ``ConfigurationError`` and is not buffered, so the
encoded video and the Parquet rows cannot silently desynchronize. A repeated
``episode_id`` is also rejected.

**Validation:** ``validate_groot_lerobot_v2(backend)`` (exported from
``ovannotators`` and ``ovannotators.profile_writers``) checks a finished
dataset through any backend and returns a ``ConformanceResult``. It checks the
required ``meta/`` files, ``codebase_version == "v2.1"`` (a LeRobot v3 dataset
is rejected), feature metadata and Arrow column types, episode and task counts,
statistics, per-episode ``frame_index`` continuity and the constant-frame-rate
timeline, one video per configured camera, and, when ``_ovannotators/`` is
present, the ``_SUCCESS`` marker and payload checksums. It reports ``semantic``,
``canonical_loader``, and ``consumer`` as ``not_run``; they are never inferred
from structural success. ``GrootLeRobotV2Writer.validate(backend)`` calls it.


.. _api-writers:

Built-in writers
----------------

Writer Protocol
~~~~~~~~~~~~~~~

All writers implement this protocol:

.. code-block:: python

   from typing import ClassVar, Protocol

   import numpy as np
   from ovannotators import Backend


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
- Frame IDs are zero-based and monotonically increasing within one clip, but
  writers must not assume they are contiguous. After ``cut()`` on a clip
  writer such as ``cosmos_transfer``, frame IDs restart at 0.


WriterRegistry
~~~~~~~~~~~~~~

.. code-block:: text

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

Registering a name that another class already holds raises ``ValueError`` on
``WriterRegistry``, ``BackendRegistry``, ``ProfileWriterRegistry``, and
``RendererOutputRegistry``. Re-registering the same class is a no-op. Pick a
unique ``name`` (or ``renderer_name``) for a third-party plugin.

When a third-party writer entry point (``ovannotators.writers``) fails to
import, ``import ovannotators`` still succeeds and logs a warning,
``Skipping writer entry point '<name>'``, with the import traceback. The
writer is then missing from ``WriterRegistry``.


For dataset purpose, output schemas, usage, and upstream format references,
see :doc:`writers/index`. That guide also covers the episode-aware
``cosmos_multiview`` profile.


BasicWriter
~~~~~~~~~~~

:Registry name: ``basic``
:GPU required: No (yes for ``video=True``: `NVENC <https://developer.nvidia.com/video-codec-sdk>`_ via `PyNvVideoCodec <https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html>`_)
:Required render vars: None (writes whatever is provided)
:Extra dependencies: None (``[video]`` for PyNvVideoCodec when ``video=True``)

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
   * - ``exr_compression``
     - ``str | None``
     - ``None``
     - EXR compression scheme; ``None`` uses OpenEXR's default ZIP
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
   * - ``video``
     - ``bool``
     - ``False``
     - Also write one MP4 per video-capable var, spanning every frame
       (H.264 by default, HEVC with ``video_codec``; requires ``PyNvVideoCodec``)
   * - ``video_fps``
     - ``float``
     - ``24``
     - Frame rate of the written MP4 files; fractional rates such as 29.97
       are exact
   * - ``video_vars``
     - ``list[str] | None``
     - ``None``
     - Which vars to encode as video. ``None`` = all video-capable vars present
   * - ``video_dir``
     - ``str``
     - ``"videos"``
     - Directory the MP4 files are written to (``""`` = backend root)
   * - ``segmentation_palette_accessibility``
     - ``str``
     - ``"none"``
     - Optionally optimize distinct colors for ``"protanopia"``,
       ``"deuteranopia"``, or ``"tritanopia"``
   * - ``semantic_color_key``
     - ``str``
     - ``"semantic_id"``
     - Color semantic segmentation by ``"semantic_id"`` or
       ``"semantic_label"``
   * - ``segmentation_palette_path``
     - ``str | Path | None``
     - ``None``
     - Optional versioned JSON file for cross-session color assignments

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
       ├── stable_id_segmentation.npy
       ├── stable_id_segmentation_colorized.png
       ├── semantic_segmentation.npy
       ├── semantic_segmentation_colorized.png
       ├── bbox_2d_tight.json
       ├── bbox_2d_loose.json
       ├── bbox_3d.json
       ├── camera_params.json
       ├── motion_2d.exr
       └── point_cloud.npz

Filenames follow lowercase snake_case convention, matching ``omni.replicator.core``
BasicWriter output naming. Only render variables present in the input are written.
Depth vars (``distance_to_camera``, ``distance_to_image_plane``) follow
``depth_format``; ``hdr_color``, ``normal``, ``camera_3d_position``, ``motion_2d``,
and the 55 named float image AOVs (for example ``pt_world_pos``) are always
written as EXR, which keeps the float16, float32, or uint32 dtype. Values are exact
with the default ZIP scheme and the other lossless schemes; lossy
``exr_compression`` schemes (``pxr24``, ``b44``, ``b44a``, ``dwaa``, ``dwab``) can
change them.
:doc:`render_outputs` lists every named AOV and its filename. Colorized PNGs are
emitted for depth, normal, and segmentation
vars when ``colorize=True``.

Segmentation visualizations use the same deterministic, perceptually separated
palette implementation as ``CosmosTransferWriter``. Instance, stable-ID, and
semantic segmentation assignments are namespaced by key type in one writer
session. Set ``semantic_color_key="semantic_label"`` to make semantic IDs with
the same resolved ``SemanticIdMap`` label share a color. Set
``segmentation_palette_path`` to atomically persist every typed assignment on
``finalize()`` and restore exact colors in a later writer session. Concurrent
writers must not share one palette file: the last writer to save replaces the
file and drops the other writers' new assignments. A state file is bound to the
accessibility mode that wrote it, and a file written by ``CosmosTransferWriter``
is also bound to that writer's single ``segmentation_color_key``. A writer
rejects a file whose recorded modes do not match its configuration.

**Video output**

With ``video=True`` the per-frame files above are still written, and each
video-capable render var is additionally encoded into a single MP4 covering
every frame the writer saw. H.264 is the default; set ``video_codec="hevc"``
for HEVC::

   {root}/
   └── videos/
       ├── ldr_color.mp4
       ├── diffuse_albedo.mp4
       ├── distance_to_camera_colorized.mp4
       ├── normal_colorized.mp4
       ├── instance_segmentation_colorized.mp4
       ├── stable_id_segmentation_colorized.mp4
       └── semantic_segmentation_colorized.mp4

``LdrColor`` and ``DiffuseAlbedoSD`` are encoded directly; depth, normal, and
segmentation vars are encoded from their colorized visualization (which is
computed for the video even when ``colorize=False``). Use ``video_vars`` to
restrict the set, ``video_fps`` for the frame rate, and ``video_dir`` for the
destination directory. Frames are streamed into the encoder as they arrive, so
only compressed output accumulates in memory; the files are written on
``finalize()``. In the default lossy mode, odd-sized frames are cropped by one
row/column so both dimensions
are even (a 4:2:0 requirement), and frames whose resolution differs from
the first frame of their stream are skipped with a warning.

CUDA RGB/RGBA frames stay on their device through video
normalization, cropping, channel packing, and NVENC input. The encoder shares
the input device's CUDA context and synchronizes producer work before encoding;
only the compressed bitstream is returned to the host. NumPy inputs remain
supported. Normal visualization also stays on the input device; depth and
segmentation visualization retain their existing NumPy preprocessing, but their
GPU-generated colorized frames can go directly to NVENC.

The NVENC session is initialized on the first frame to select its owning CUDA
context and stream. An NVENC flush failure propagates instead of writing a
truncated MP4.

Video output requires ``PyNvVideoCodec`` (``pip install PyNvVideoCodec``) and an
NVENC-capable GPU. To package the episode videos as WebDataset samples, pass
``adapter=BasicVideoEpisodeAdapter()`` to ``DatasetCapture``; see
:ref:`webdataset-api`.


.. _video-encoding-options:

**Shared video encoding options**

``BasicWriterConfig``, ``CosmosTransferWriterConfig``, and
``LeRobotWriterConfig`` accept these additional fields:

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ``video_codec``
     - ``"h264" | "hevc"``
     - ``"h264"``
     - Codec used for MP4 output
   * - ``video_lossless``
     - ``bool``
     - ``False``
     - Exact normalized RGB preservation; requires ``video_codec="hevc"``
   * - ``video_preset``
     - ``None | "P1" ... "P7"``
     - ``None``
     - NVENC speed/compression preset; None retains the existing lossy default
       or selects P5 for lossless HEVC

Lossless HEVC stores unmodified 8-bit GBR planes with full-range identity
color signaling (``gbrp``). Decoding through a compatible reader reproduces
the normalized RGB pixels exactly, including saturated colors and fine detail.
Lossy HEVC and the default H.264 path use 4:2:0 color sampling.
Lossless video requires positive, even, fixed frame dimensions: odd dimensions
and resolution changes raise an error instead of cropping, resizing or skipping
RGB frames. Codec initialization errors identify the requested mode; no lossy
fallback is used.

CUDA RGB inputs are packed to GBR on-device and passed directly to NVENC.
Only compressed packets return to the CPU for MP4 packaging and storage.
All video modes require NVENC and the ``[video]`` extra.

The lossless boundary is the writer's normalized RGB image: float RGB is
scaled/clipped to uint8, grayscale is replicated, and alpha is discarded.
Depth, normals, and segmentation videos preserve their generated visualizations,
including any visualization resizing, rather than their original numeric arrays.
Keep the separate raw outputs when exact depth values or label IDs are needed.

HEVC lossless requires NVENC HEVC 4:4:4/lossless support and a compatible
driver/PyNvVideoCodec installation. Output uses the HEVC Range Extensions
profile, which has more limited browser and hardware-player support than
ordinary H.264. FFmpeg software decoding supports it; consumers of LeRobot
and Cosmos datasets must use a reader supporting this profile. This setting
does not configure the downstream reader. Files may be substantially larger
than lossy video. Lower-numbered presets favor speed; higher ones favor
compression without changing the lossless guarantee.

COCOWriter
~~~~~~~~~~

:Registry name: ``coco``
:GPU required: No
:Required render vars: ``LdrColor``, ``SemanticBoundingBox2DTight`` (plus ``SemanticInstanceSegmentation`` when ``include_segmentation=True`` or ``include_panoptic=True``)
:Extra dependencies: None

Accumulates annotations across frames and writes `MS-COCO
<https://cocodataset.org/#format-data>`_ format JSON on finalize.

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

Structured `NumPy <https://numpy.org/>`_ array (:data:`OVRTX_BBOX2D_DTYPE`) with fields ``instance_id``
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

Writes dataset in `KITTI <https://www.cvlibs.net/datasets/kitti/eval_object.php>`_
object-detection format.

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
     - uint16 units per metre (``depth_m * depth_scale``, clipped to 0-65535 and truncated)

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

Writes dataset in `Ultralytics YOLO
<https://docs.ultralytics.com/datasets/detect/>`_ format.

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


OpenLABELWriter
~~~~~~~~~~~~~~~

:Registry name: ``openlabel``
:GPU required: No
:Required render vars: ``LdrColor``, ``SemanticBoundingBox2DTight``
:Optional render vars: ``SemanticInstanceSegmentation`` (required when ``include_segmentation=True``), ``SemanticIdMap``, ``StableIdMap``
:Extra dependencies: None

Writes one `ASAM OpenLABEL 1.0 <https://www.asam.net/standards/detail/openlabel/>`_
JSON file per image in the layout accepted by FiftyOne's
``OpenLABELImageDataset`` importer. The writer currently emits 2D boxes and,
optionally, instance polygons. It does not emit temporal tracks or 3D cuboids.

**Config:** ``OpenLABELWriterConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``image_format``
     - ``"png" | "jpg"``
     - ``"png"``
     - Image encoding
   * - ``include_segmentation``
     - ``bool``
     - ``False``
     - Add absolute-pixel ``poly2d`` polygons from instance masks

**Output structure:**

::

   {root}/
   ├── data/{frame_id:06d}.png
   └── labels/{frame_id:06d}.json

Each JSON file includes ``openlabel.metadata.schema_version=1.0.0`` and an
``objects`` mapping keyed by instance ID. Each object's ``bbox`` is
``[center_x, center_y, width, height]`` in image pixels. With segmentation
enabled, ``poly2d`` vertices are also in image pixels.


BOPWriter
~~~~~~~~~

:Registry name: ``bop``
:GPU required: No
:Required render vars: ``LdrColor``, ``DistanceToCameraSD``, ``SemanticBoundingBox2DTight``, ``SemanticBoundingBox3D``, ``SemanticInstanceSegmentation``, ``CameraParams``
:Extra dependencies: None

Writes dataset in `BOP <https://bop.felk.cvut.cz/>`_ (Benchmark for 6D Object
Pose Estimation) format.

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

This is the legacy frame-oriented writer and is not the exact LeRobot v3
profile. The LeRobot v3 profile is coming soon; see :doc:`writers/lerobot_v3`.

:Registry name: ``lerobot``
:GPU required: Yes (NVENC via PyNvVideoCodec)
:Required render vars: ``LdrColor``
:Extra dependencies: ``[lerobot]`` (`PyArrow <https://arrow.apache.org/docs/python/>`_, `Apache Parquet <https://parquet.apache.org/>`_ output), ``[video]`` (PyNvVideoCodec, NVENC video encoding)

CUDA image inputs are snapshotted into owned device memory when received and
kept there until video encoding in ``finalize()``. Renderer buffers can be reused
after ``write_frame()`` returns. The snapshots consume GPU memory proportional
to the buffered episode and are released after successful finalization.

Writes dataset in `Hugging Face LeRobot
<https://huggingface.co/docs/lerobot/index>`_ format for robot learning. It
writes the repository's legacy single-episode ``v2.0``-style layout and carries
no GR00T compatibility claim; use ``groot_lerobot_v2`` for the pinned
Isaac-GR00T contract.

**Config:** ``LeRobotWriterConfig``

Also accepts the :ref:`shared video encoding options <video-encoding-options>`.
Video metadata reflects the selected codec and pixel format.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``fps``
     - ``float``
     - ``30``
     - Frames per second of episode video; fractional rates such as 29.97
       are exact
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
:Required render vars: Config-dependent; ``LdrColor`` is always required. The segmentation render var follows ``segmentation_color_key``.
:Extra dependencies: ``[video]`` for PyNvVideoCodec

Writes training data for `NVIDIA Cosmos Transfer 2.5
<https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html>`_ models — RGB
video plus conditioning signal videos.

Additional inputs depend on ``conditioning_types`` and, when ``"canny"`` is
enabled, ``canny_source``:

- ``DistanceToCameraSD`` is required for ``"depth"`` conditioning.
- The segmentation render var selected by ``segmentation_color_key``
  (``SemanticInstanceSegmentation`` by default) is required for
  ``"segmentation"`` or ``"shaded_segmentation"`` conditioning, or Canny
  derived from either source.
- ``NormalSD`` is required for ``"shaded_segmentation"`` conditioning, or
  Canny derived from ``"normals"`` or ``"shaded_segmentation"``.

RGB-only Canny requires only ``LdrColor``. Use the configured writer's
``required_render_vars`` or :func:`ovannotators.describe_requirements` to
resolve the selected inputs before capture.

**Config:** ``CosmosTransferWriterConfig``

Also accepts the :ref:`shared video encoding options <video-encoding-options>`,
applied to both RGB and conditioning videos.
Each manifest clip has a ``video_encoding`` object containing ``codec``,
``lossless``, and ``pixel_format``.

.. list-table::
   :header-rows: 1
   :widths: 22 22 18 38

   * - Parameter
     - Type
     - Default
     - Description
   * - ``fps``
     - ``float``
     - ``24``
     - Output video frame rate; fractional rates such as 29.97 are exact
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
   * - ``segmentation_palette_accessibility``
     - ``str``
     - ``"none"``
     - Optionally optimize distinct colors for ``"protanopia"``,
       ``"deuteranopia"``, or ``"tritanopia"``.
   * - ``segmentation_color_key``
     - ``str``
     - ``"instance_id"``
     - Identity used to assign colors. Select ``"instance_id"``,
       ``"stable_id"``, ``"semantic_id"``, or ``"semantic_label"``.
   * - ``segmentation_palette_path``
     - ``str | Path | None``
     - ``None``
     - Optional JSON file used to load and atomically persist palette colors
       and typed key assignments across writer sessions.

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

**Segmentation colors:**

Segmentation colors remain fixed for each instance for the lifetime of the
writer, including across ``cut()`` boundaries and when instances leave and
re-enter view. Every channel of a multi-channel instance ID participates in
the mapping, and the all-zero ID stays black. Plain and shaded segmentation
share the mapping.

The palette lazily extends deterministically as new IDs appear. It uses
farthest-point selection in the perceptually uniform
Oklab color space over a compact candidate grid, avoiding a full 24-bit color
lookup table. The candidate grid contains more than 10,000 unique colors and
is intended for datasets with hundreds to thousands of instances. Select a
``segmentation_palette_accessibility`` mode to include the selected vision
simulation in the distance score.

``segmentation_color_key`` selects the identity buffer used for colorization:
``SemanticInstanceSegmentation`` for ``"instance_id"``,
``StableIdSegmentation`` for ``"stable_id"``, and ``SemanticSegmentation`` for
the semantic modes. ``"semantic_label"`` additionally resolves IDs through
``SemanticIdMap``; IDs with the same label share a color. Missing identity or
label data raises an error rather than falling back to another key type.

Set ``segmentation_palette_path`` to retain exact assignments across writer
sessions. The versioned JSON file records the palette algorithm, accessibility
mode, key mode, RGB colors, and key-to-slot assignments. It is loaded when the
writer is constructed and atomically updated after each successful ``cut()``
or ``finalize()`` flush. A file whose recorded modes do not match the writer
configuration is rejected. The file is bound to one accessibility mode and to
this writer's one ``segmentation_color_key``, and concurrent writers must not
share it: the last writer to save replaces the file. A crash in the middle of a
clip loses that clip's new assignments, because the file is written only at
``cut()`` and ``finalize()``.

**Multiple clips:**

A single writer can emit several clips that share one ``manifest.json``.
Accumulate frames with ``write_frame()``, then call ``cut(backend)`` to encode
the buffered frames into a clip and start a new one (the ``clip_index``
advances automatically); ``finalize(backend)`` flushes the last clip. Use this
to slice a long capture into sequential clips, or to emit one clip per scene
variation (modify the scene, render more frames, then ``cut`` again).
``cut()`` is a no-op when no frames are buffered and returns the written
``clip_id`` (or ``None``).

With a configured Cosmos Transfer ``writer`` and ``backend``, supply two
iterables of frame mappings, ``frames_a`` and ``frames_b``:

.. code-block:: python

   for frame_id, render_vars in enumerate(frames_a):
       writer.write_frame(frame_id, render_vars, backend)
   writer.cut(backend)              # clip 000000

   for frame_id, render_vars in enumerate(frames_b):
       writer.write_frame(frame_id, render_vars, backend)
   writer.finalize(backend)         # clip 000001

**GPU behavior:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - GPU available
     - CPU only
   * - Canny edge detection
     - `Warp <https://nvidia.github.io/warp/>`_ kernels on `CUDA
       <https://developer.nvidia.com/cuda-toolkit>`_ (~1ms @ 720p)
     - Same Warp kernels on Warp's CPU device
   * - Image encoding (PNG/JPEG)
     - `nvImageCodec <https://docs.nvidia.com/cuda/nvimagecodec/>`_ via the
       ``[gpu]`` extra (zero-copy for on-device frames)
     - `Pillow <https://python-pillow.github.io/>`_ (CPU) — automatic fallback
   * - Video encoding
     - PyNvVideoCodec NVENC (H.264 by default, or HEVC)
     - N/A (GPU required)

RGB frames are snapshotted on their source device until ``cut()`` or
``finalize()``; buffered CUDA clips therefore consume GPU memory. RGB-derived
Canny keeps grayscale and edge images on the GPU (only reduction/convergence
scalars are read by the CPU). Depth, segmentation and shaded-segmentation
conditioning retain their existing NumPy preprocessing. Thumbnails use the
selected image codec, and frame resizing uses Pillow (CPU). Canny runs the same Warp
kernels (``ovannotators.writers.canny``) on whatever device the data lives on —
CUDA when present, otherwise Warp's CPU device — with no separate code path and
no ``prefer_gpu`` switch.


.. _api-backends:

Built-in backends
-----------------

Backend Protocol
~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Protocol

   import numpy as np


   class Backend(Protocol):
       def write(self, path: str, data: bytes) -> None: ...
       def write_array(
           self,
           path: str,
           array: np.ndarray,
           encoding: str = "npy",
           *,
           exr_compression: str | None = None,
       ) -> None: ...
       def read(self, path: str) -> bytes: ...
       def list(self, prefix: str = "") -> list[str]: ...
       def exists(self, path: str) -> bool: ...
       def makedirs(self, path: str) -> None: ...

Backends resolve relative paths against their configured root. For
``DiskBackend``, absolute paths are also accepted when they resolve inside
``root_dir``.


DiskBackend
~~~~~~~~~~~

Writes to local filesystem. ``root_dir`` may be absolute. Caller paths may be
relative to ``root_dir`` or absolute, but must resolve inside ``root_dir``.
Paths that resolve outside it, including ``..`` traversal, are rejected.

* Symlinks are followed before the check. A symlink (to a file or a directory)
  that points outside ``root_dir`` is rejected; a symlink that points to another
  location inside ``root_dir`` is allowed. A path that cannot be resolved, such
  as a symlink loop, is rejected the same way.
* ``list()`` omits symlinks whose targets are outside ``root_dir`` (the same ones
  ``read()`` rejects) and broken or looping symlinks. Regular files and symlinks
  that stay inside ``root_dir`` are listed.
* A relative ``root_dir`` is resolved against the current working directory once,
  when the backend is constructed. Later changes to the working directory do not
  move the root.
* ``read()`` and ``list()`` raise ``BackendReadError`` for a rejected path.
  ``write()``, ``write_array()`` and ``makedirs()`` raise ``BackendWriteError``.
  ``exists()`` returns ``False``.
* The check runs at call time. It is not a sandbox against another process
  changing the filesystem concurrently.

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
     - ``str | Path``
     - (required)
     - Output directory. A relative path is resolved once, at backend construction.
   * - ``overwrite``
     - ``bool``
     - ``False``
     - Allow overwriting existing files

Image encoding always prefers the GPU automatically when one is available (see
`Image Codecs`_); there is no per-backend codec switch.


S3Backend
~~~~~~~~~

Writes to Amazon S3 or compatible object stores. Paths stay below the
configured ``prefix``: a path that is absolute or contains a ``..`` or empty
segment makes ``write`` raise ``BackendWriteError``, ``read`` and ``list``
raise ``BackendReadError``, and ``exists`` return ``False``, without a request
to S3. ``.`` segments are dropped.

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
     - Encoded object size (MiB) at or above which ``write`` uses a multipart
       upload. Must be positive; values above ``5120`` (the 5 GiB
       single-request limit) are capped to ``5120``. At the default, larger
       encoded arrays (for example float32 depth at 4K, or large EXR frames)
       upload as multipart (create, parts, complete) rather than one
       ``PutObject``. Part size follows the threshold, bounded to 5-64 MiB, so
       memory use stays bounded. The credentials need multipart permissions such as
       ``s3:AbortMultipartUpload``. Raise the value to keep single-request
       uploads for objects of that size

**Requires:** ``pip install "ovannotators[s3]"``


.. _ovstoragebackend:

OvstorageBackend
~~~~~~~~~~~~~~~~

Writes through a caller-owned, already built `ovstorage Stack
<https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/docs/public/library-python/README.md>`_.
Registered as ``"ovstorage"``. Configure connections, plugins, authentication,
routing, and caching on the Stack; the backend uses its address root and does
not own the Stack lifetime. Call ``backend.close()`` after writing.

.. code-block:: python

   import asyncio
   from pathlib import Path
   import ovstorage
   from ovannotators import OvstorageBackend, OvstorageBackendConfig

   root = Path("_out/dataset").resolve()
   root.mkdir(parents=True, exist_ok=True)
   request = ovstorage.ConnectionRequest("file")
   request.add_config("root", ovstorage.ConfigValue.string(str(root)))
   stack = asyncio.run((
       ovstorage.Stack(root="files")
       .backend(ovstorage.file.FileBackend("files"))
       .connection("files", request)
   ).build())
   backend = OvstorageBackend(OvstorageBackendConfig(root_uri=root.as_uri()), stack)
   try:
       backend.write("sample.bin", b"data")
   finally:
       backend.close()

For cloud or cached storage, pass a Stack configured with the appropriate
``ovstorage`` plugins or layers and use a matching ``root_uri`` (for example,
``s3://bucket/dataset/``). The ``ovstorage`` extra is optional; disk and S3
backends can be installed independently.

Build one Stack per process in the application and pass that same object to
every ``OvstorageBackend``; do not build a Stack per writer or per backend.
ovstorage keeps one authentication store per process, and its `layered
architecture <https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/docs/rfcs/0066-layered-architecture.md>`_
expects every component in a process to share the application's Stack.

For a remote connection, follow the upstream `Python tutorial
<https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/ovstorage-core/examples/python/README.md>`_
and `Stack configuration guide
<https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/docs/public/configuration.md>`_.
The `backend authentication guide
<https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/skills/ovstorage-user-authenticate-to-backend/SKILL.md>`_
describes credentials and interactive sign-in for each backend. For an
Omniverse Storage Service connection, see its `plugin guide
<https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/docs/public/plugin-storage/plugin-services-client.md>`_
and the Python binding's `interactive authentication example
<https://github.com/NVIDIA-Omniverse/ovstorage/blob/main/docs/public/library-python/README.md#interactive-authentication>`_.
The `ovstorage skills catalog
<https://github.com/NVIDIA-Omniverse/ovstorage/tree/main/skills>`_ links all
its user, operator, and contributor workflows.
Authenticate the connection used by this Stack on the host that writes the
dataset; a browser or another client's sign-in is not shared with it.

After authentication, load the Stack from its credential-free configuration
file with its plugin registry, then pass it to ``OvstorageBackend``. The
``tests/test_ovstorage_remote.py`` smoke test exercises this exact path.
Construction checks the route before any frame is written: it raises
``ConfigurationError`` when the Stack reports no route or configuration for
``root_uri``, or when the connection still needs sign-in. A root that does not
exist yet is valid. Writes create missing parent directories, so ``makedirs``
is a no-op. The caller retains ownership of the Stack and its credential
lifecycle.

The test suite exercises this adapter against a real local-file Stack and
checks its SDK calls with a mock Stack. To verify a remote route, supply an
``ovstorage`` TOML configuration with an authenticated connection that covers
the target root, then run the opt-in smoke test::

   OVANNOTATORS_OVSTORAGE_SMOKE_CONFIG=/path/to/ovstorage.toml \
   OVANNOTATORS_OVSTORAGE_SMOKE_ROOT=https://example.blob.core.windows.net/container/Playground/ \
   pytest -q tests/test_ovstorage_remote.py

The smoke test writes a unique object below ``ovannotators-smoke/`` and checks
read, existence, and listing. It leaves the object in place and prints its
address. The test skips when either environment variable is absent. Use a root
within your assigned writable folder. The adapter walks paginated directory
listings so it also works with routes that do not support recursive listing.
``overwrite=False`` requires atomic fail-if-exists writes. Construction raises
``ConfigurationError`` when the advertised root covering ``root_uri`` reports no
such support; the RC Storage API route is one example.

.. list-table::
   :header-rows: 1
   :widths: 18 20 18 44

   * - Config
     - Type
     - Default
     - Description
   * - ``root_uri``
     - ``str``
     - (required)
     - Dataset directory address within the caller's Stack
   * - ``overwrite``
     - ``bool``
     - ``True``
     - Allow replacing existing objects; ``False`` uses an atomic fail-if-exists write

**Requires:** ``pip install "ovannotators[ovstorage]"``


.. _api-renderer-outputs:

Renderer outputs
----------------

OvrtxOutput
~~~~~~~~~~~

Extracts render vars from ovrtx ``FrameOutput`` objects as zero-copy Warp arrays
via DLPack (annotator byte buffers such as bounding boxes and camera params are
decoded into structured NumPy arrays).

Given a live renderer frame ``frame_output``, extract the color array on the
rendering device or on the CPU:

.. code-block:: python

   from ovannotators import OvrtxOutput

   extractor = OvrtxOutput()

   # CUDA view by default; keeps the data on the device where ovrtx rendered
   # it, ready for zero-copy GPU encode. Annotator buffers such as CameraParams
   # are always decoded on the host.
   rgb_gpu = extractor.extract(frame_output, "LdrColor")

   # Host (CPU) view (forces a device-to-host copy)
   rgb = extractor.extract(frame_output, "LdrColor", device="cpu")
   print(rgb_gpu.device, rgb.device)  # cuda:0 cpu

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


.. _api-types:

Types
-----

.. code-block:: python

   from ovannotators.types import (
       Resolution, BBox2D, BBox3D, CameraParams, PointCloudResult,
   )

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
     - Point-cloud data (points, normals, colors, semantic_ids) in Warp arrays
       on CPU or CUDA. Optional ``count`` holds a device-side valid count for
       capacity-sized CUDA buffers; without it, every row is valid.

Pointcloud results
~~~~~~~~~~~~~~~~~~

``point_cloud_from_depth`` and its batched variant ``point_clouds_from_depth``
return ``PointCloudResult`` with compact Warp CPU arrays by default.
Use ``output_device="cuda"`` for GPU-resident arrays; only the first
``count[0]`` rows are valid. Use ``device="cpu"`` for CPU-only execution
(with ``output_device="cpu"``). Attributes require Warp-supported numeric dtypes.
Returned arrays are ready to use and remain valid across later calls.


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
   * - ``PackagingError``
     - ``OVAnnotatorsError``
     - Samples cannot be packaged into a dataset archive


.. _api-configuration-serialization:

Configuration serialization
---------------------------

Every ``*WriterConfig`` and ``*BackendConfig`` is a plain
:func:`dataclasses.dataclass`, which makes configs easy to persist and reload.
Two helpers handle the round-trip, including ``pathlib.Path`` fields and nested
config dataclasses that a bare ``dataclasses.asdict`` + ``Config(**data)`` would
not:

.. code-block:: text

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
   print(loaded.root_dir, loaded.overwrite)

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

   @WriterRegistry.register
   class MyWriter:
       name = "my_format"
       required_render_vars = frozenset({"LdrColor"})

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

   import numpy as np


   class MyBackend:
       def write(self, path: str, data: bytes) -> None: ...
       def write_array(
           self,
           path: str,
           array: np.ndarray,
           encoding: str = "npy",
           *,
           exr_compression: str | None = None,
       ) -> None: ...
       def read(self, path: str) -> bytes: ...
       def list(self, prefix: str = "") -> list[str]: ...
       def exists(self, path: str) -> bool: ...
       def makedirs(self, path: str) -> None: ...

The ``write_array`` encoding parameter accepts: ``"npy"``, ``"png"``, ``"jpg"``, ``"exr"``.

``exr_compression`` is a keyword-only parameter that selects the EXR compression
scheme (see :ref:`api-exr-compression`). ``BasicWriter`` passes it only when
``BasicWriterConfig.exr_compression`` is set, so a backend that does not accept
the keyword keeps working until a user selects a scheme. A backend that supports
EXR should forward the scheme to its EXR encoder; a backend that does not should
raise ``ValueError`` when ``exr_compression`` is not ``None``. The built-in
backends raise ``ValueError`` when it is combined with an encoding other than
``"exr"``.


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
     - `OpenEXR <https://openexr.com/>`_ (float16/32/uint32; exact with lossless schemes, see :ref:`api-exr-compression`)
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
   frames are already GPU-resident (e.g. from ``OvrtxOutput().extract_all(frame)``, which defaults to ``device="cuda"``),
   nvImageCodec encodes them **directly on the device with no host copy**.
   Without the extra (or without a CUDA device), ovannotators transparently
   falls back to Pillow on the CPU — correct, just slower; no code changes
   needed either way.

.. code-block:: python

   import numpy as np
   from ovannotators._internal.codecs import get_image_codec

   array = np.zeros((48, 64, 3), dtype=np.uint8)
   codec = get_image_codec()         # NvImageCodecBackend if a GPU is available, else Pillow
   codec.write("output.png", array)  # extension selects the format
   data = codec.encode(array, format="jpg", quality=85)  # -> bytes
   decoded = codec.decode(data)
   print(decoded.shape, decoded.dtype)

For EXR files (HDR data like ``HdrColor``), ``read_exr`` is the public reader:

.. code-block:: python

   from ovannotators._internal.codecs import read_exr

   data = read_exr("hdr_color.exr")  # preserves stored float16, float32, or uint32 channels
   print(data.shape, data.dtype)

EXR writing happens through ``Backend.write_array(..., encoding="exr")`` rather
than a public codec helper.

.. _api-exr-compression:

EXR compression
^^^^^^^^^^^^^^^

EXR output uses lossless ZIP compression by default. Choose a scheme for one
array with ``exr_compression``::

   backend.write_array("depth.exr", depth, encoding="exr", exr_compression="piz")

Set ``BasicWriterConfig(exr_compression="piz")`` to apply the scheme to every
EXR written by ``BasicWriter``, including depth when ``depth_format="exr"``.
The setting does not affect PNG, JPEG, NPY, or NPZ output. The built-in disk,
S3, and ovstorage backends accept it; a third-party backend must also accept
the keyword when a non-default scheme is requested.

Supported names are ``none`` (uncompressed), ``rle``, ``zips``, ``zip``,
``piz``, ``htj2k256``, and ``htj2k32`` (lossless); and ``pxr24``, ``b44``,
``b44a``, ``dwaa``, and ``dwab`` (potentially lossy). The dtype (float16,
float32, or uint32) is always preserved, but values are exact only with
``none`` and the lossless schemes. ``pxr24`` reduces float32 precision, and
``b44``, ``b44a``, ``dwaa``, and ``dwab`` can change float values; avoid lossy
schemes for exact SDG ground truth.
Availability of newer schemes such as HTJ2K depends on the installed OpenEXR
build. An unsupported scheme raises ``ValueError``.

.. _webdataset-api:

WebDataset packaging
--------------------

``WebDatasetPackager`` wraps any storage backend with
sample-aware packaging. It accepts explicit episodes rather than arbitrary file
writes, so it is not a ``Backend`` or a ``BackendRegistry`` destination.
Use ``DatasetCapture`` with a writer and its adapter,
``SampleFiles`` and ``pack_episode`` for already encoded files, or
``export_finalized_dataset`` for finalized native writer output. The adapter
mechanism supports additional formats through adapters that you pass with
``adapter=``. No additional
runtime dependency or GPU is required for packaging. ``DatasetCapture`` and
``CaptureEpisode`` are experimental and may change before they are declared
stable. Packaging failures raise ``PackagingError``.
See :ref:`webdataset-choosing-path` for when to use each entry point and the
built-in writer adapters.

.. automodule:: ovannotators.backends.webdataset
   :members: WebDatasetPackagerConfig, SampleFiles, WebDatasetPackager

.. automodule:: ovannotators.capture
   :members: DatasetCapture, CaptureEpisode, export_finalized_dataset

.. automodule:: ovannotators.capture_adapters
   :members:
   :exclude-members: resolve_adapter, source_paths

``DatasetCapture`` and ``export_finalized_dataset`` select a built-in adapter
by exact writer type: each ``*FrameAdapter`` class above serves the writer
named in its description. Pass the matching adapter, for example
``adapter=BasicFrameAdapter()``, for a subclass of a built-in writer. Any other
writer needs a ``CaptureAdapter`` subclass that defines ``sample_schema``,
``create_writer`` and ``collect_samples``.
Optional hooks validate the writer, check each frame's output, describe the
writer for the manifest, discover frames for finalized export and add
dataset-wide metadata. Adapters keep dataset-wide state in
``AdapterContext.state``. ``export_finalized_dataset`` uses the same adapter on
existing local episode directories and leaves source files unchanged. The COCO
adapter requires the finalized images to be listed in increasing frame order,
as ``DatasetCapture`` does. The
``episodes`` mapping of ``export_finalized_dataset`` supplies unique IDs and native directory roots. Custom
adapters are selected per call; there is no public adapter registry. The
adapter API is experimental and may change before it is declared stable.

Sample units
~~~~~~~~~~~~

``CaptureAdapter.sample_unit`` is ``"frame"`` (default), ``"episode"`` or
``"clip"``; another value fails with ``ConfigurationError`` before capture.
The manifest records it as ``metadata.sample_unit``. Basic and COCO adapters
use frames; ``BasicVideoEpisodeAdapter`` uses episodes.

- ``frame``: one sample per written frame.
- ``episode``: ``collect_samples`` returns exactly one sample for the whole
  finalized episode, such as an episode video bundle. Shard sample limits then
  count episodes. The sample is never split: ``allow_episode_spanning=True``
  only rotates between samples, so an episode larger than a shard raises unless
  ``oversized_episode_policy="dedicated_shard"`` is set.
- ``clip``: one sample per clip. The adapter must override ``cut_clip``, which
  closes the writer's current clip through a public writer method (such as
  ``CosmosTransferWriter.cut``) and returns its native clip index. The producer
  calls ``CaptureEpisode.end_clip()`` between clips; frames written after the
  last boundary form a final clip on successful episode exit. Each closed clip
  is recorded as a ``ClipSpan`` (``clip_index``, ``frame_ids``) in
  ``AdapterContext.clips``, and ``collect_samples`` returns one sample per span
  in order. ``end_clip`` raises ``PackagingError`` without changing the episode
  when the adapter is not a clip adapter or no frame was written since the
  previous boundary. A failure inside ``cut_clip``, or a repeated or invalid
  clip index, discards the episode. The clips of one episode stay in one
  shard unless spanning is enabled.

Episode and clip sample counts are checked after collection. Finalized export
calls ``discover_frames`` for frame and episode adapters and
``discover_clips`` for clip adapters. A writer that rewrites a native file
during an episode, such as the per-cut ``manifest.json`` of
``CosmosTransferWriter``, needs ``allow_native_overwrite = True`` on its
adapter; otherwise the episode disk backend rejects the overwrite. The flag
applies to the whole episode backend, including frame writes, so it removes the
backend's guard against a writer that reuses a file name for two frames.

The following example includes a small text writer and its clip adapter.
Supply ``output_dir`` as the destination path.

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-clip-adapter]
   :end-before: # [/snippet:doc-webdataset-clip-adapter]
   :dedent: 4

Custom writers and manifest metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The writer that ``CaptureAdapter.create_writer`` returns must provide
``write_frame(frame_id, render_vars, backend)`` and ``finalize(backend)``.
``DatasetCapture`` calls ``write_frame`` with a temporary disk backend for each
frame and ``finalize`` once at episode exit, then calls ``collect_samples`` on
the finalized output. ``export_finalized_dataset`` skips both calls and passes
the existing directory to ``discover_frames`` and ``collect_samples``.

The adapter, not the writer, supplies manifest metadata, and construction
validates it so a mistake fails before capture starts:

- ``sample_schema`` and ``sample_unit`` (default ``"frame"``) must be non-empty
  strings. A missing or empty value raises ``ConfigurationError``.
- ``describe_writer`` records ``writer`` (the writer's ``name`` attribute, or
  its class name when it has none) and ``writer_config``. A dataclass ``config``
  is converted with ``config_to_dict``; any other ``config`` is recorded as
  given. The result must be finite JSON, or construction raises
  ``ConfigurationError``. Override ``describe_writer`` for configuration that
  is not JSON-serializable.
- ``validate`` may reject unsupported writer configurations, and
  ``dataset_metadata`` may add dataset-wide fields such as a category table.

Each sample's component files must lie under the native episode root or
``AdapterContext.generated_root``; other paths raise ``PackagingError``.

Configuration and limits
~~~~~~~~~~~~~~~~~~~~~~~~

``max_shard_bytes`` defaults to ``"512 MiB"`` and accepts integer bytes or explicit
``B``, ``KB``, ``MB``, ``GB``, ``TB``, ``KiB``, ``MiB``, ``GiB``, ``TiB`` units.
Decimal values must resolve to whole bytes. ``GB`` is decimal; ``GiB`` is binary.
The minimum is 10240 bytes. Construction normalizes the value to integer bytes.
Optional ``max_samples_per_shard`` and ``max_episodes_per_shard`` must be positive
integers. A new shard starts before adding a unit that would exceed any limit.

Episodes stay together by default. Set ``allow_episode_spanning=True`` to allow
rotation between complete samples. A sample never spans shards. An episode
that exceeds an empty shard's byte or sample limit raises by default. Explicit
``oversized_episode_policy="dedicated_shard"`` permits a dedicated oversized shard,
recorded as ``oversized`` in the manifest. With spanning enabled, this policy
applies to individual samples instead. It may exceed both byte and sample limits.

Sizing uses actual encoded files and includes USTAR headers, 512-byte member
padding, closing blocks and 10240-byte record padding. Archives are uncompressed.
Episodes and pending shard contents are staged on temporary disk; destination
publication currently materializes one full shard in memory. Dedicated oversized
shards can therefore require more memory than the configured byte limit. A
single component cannot exceed the 8 GiB USTAR member limit. No median/average
estimate or archive rollback is used.

Lifecycle and identities
~~~~~~~~~~~~~~~~~~~~~~~~

Use the dataset and episode context managers. Successful episode exit finalizes
its fresh format writer and stages complete samples; successful dataset exit
publishes the final shard and then the completion manifest. Native disk/S3/HTTP
backends gain no new lifecycle requirement.

An episode rejected by validation, shard limits or its writer is discarded and
the capture or packager stays open: catch the ``PackagingError`` (or writer error)
to skip or retry that episode ID. An episode without frames or samples is
rejected. An uncaught error, a storage failure or completing an episode after a
failed frame write aborts the dataset: temporary files are cleaned and pending
output is discarded. Already published shards remain, but no completion manifest
is written. Do not consume a dataset until its manifest exists. To reuse an
aborted ``dataset_id``, delete its prefix from storage first. Instances are single-use and single-threaded; concurrent producers,
resume and overwrite are not supported.

Identifiers contain 1-40 lowercase ASCII letters, digits, underscores or hyphens,
starting with a letter or digit; Windows device names are rejected. Integer
identifiers are converted to strings. Episode IDs are unique across a dataset;
sample IDs are unique within an episode. Duplicate IDs are rejected rather than
renamed or overwritten. Generated member keys contain dataset, episode and sample
identity, making extraction of all shards into one directory collision-free.
Independent datasets must use distinct ``dataset_id`` namespaces (generated by
default). Component suffixes are lowercase portable names of up to 40 characters;
``metadata.json`` is reserved for the generated identity/application metadata.

.. _webdataset-sample-schemas:

Writer sample schemas
~~~~~~~~~~~~~~~~~~~~~

``DatasetCapture`` treats the supplied writer as a configuration template and
creates a fresh writer per episode. Frames must have nonnegative increasing
indices; gaps are allowed and indices restart per episode. Renderer arrays and
codec dispatch use the existing writer unchanged. Basic per-frame output is
packaged using original filenames as component suffixes, such as
``ldr_color.png`` and ``distance_to_camera.npy``. The default Basic adapter
rejects ``video=True``; pass ``adapter=BasicVideoEpisodeAdapter()`` to package
episode video instead. It is never selected automatically, because it changes
what a sample and a shard sample count mean.
A Basic frame that emits no files, or a file name that is not a portable
component suffix, raises ``PackagingError`` during ``write_frame`` and discards
the episode.

COCO samples contain ``jpg`` or ``png``, a ``json`` component with ``schema_version``,
``image``, ``annotations`` and ``categories``, plus generated ``metadata.json``.
Optional RLE segmentation and ``panoptic.png``/``panoptic`` records are retained.
Bounding boxes remain COCO pixel ``[x, y, width, height]``. Image and annotation
IDs are renumbered to be unique across the dataset, and category IDs are mapped
consistently across episodes by category name. Each sample's ``categories``
lists its episode's categories; the manifest holds the complete table. Image and panoptic file references identify extracted members.
These are WebDataset samples with COCO-style annotations, not an unchanged COCO
directory dataset. An ordinary COCO loader needs a sample adapter.

Basic video episode samples (``basic-video-episode-v1``) hold one complete
episode: every MP4 that ``BasicWriter`` wrote, as ``<stream>.mp4`` with its
encoded bytes unchanged, and the per-frame files selected by ``frame_files``,
as ``<frame directory>.<file name>`` components such as
``0003.camera_params.json``. ``frame_files`` holds case-sensitive ``fnmatch``
patterns; the default ``("*.json",)`` keeps per-frame JSON annotations and
metadata that the videos do not carry. Use ``("*",)`` for every frame file or
``()`` for videos only; the manifest records the selection as
``frame_file_patterns``. A selected name longer than the 40-character
component limit raises ``PackagingError`` when the frame is written.

The generated ``episode.json`` component records ``schema_version``,
``episode_id``, ``frame_ids``, ``video_fps``, ``frame_file_patterns``,
``frames`` (each frame's selected file names mapped to components) and
``streams``. Each stream entry gives ``component``, ``codec``, ``width``,
``height``, ``timescale``, ``frame_count``, ``duration`` and, per video frame,
``pts`` (presentation time in ``timescale`` units, read from the MP4 without
decoding, on the media timeline: MP4 edit lists are not applied) and ``frame_ids`` (the episode frame index). Video frame ``i`` was
encoded from frame ``frame_ids[i]``; the encoder timeline advances one frame
period per encoded frame, so frame-index gaps are not reflected in ``pts``.
Colorized streams pair with the frames that hold their source render var. An
episode without a video, or a video whose frame count differs from those
frames, for example after the writer skipped a frame whose resolution changed,
raises ``PackagingError``. One episode is one sample and is never split, so an
episode larger than ``max_shard_bytes`` requires
``oversized_episode_policy="dedicated_shard"``.

In the example below, ``output_dir`` is a destination ``pathlib.Path``, and
``frames`` is an iterable of mappings containing ``LdrColor`` and ``CameraParams``.
Encoding requires the ``[video]`` extra and an NVENC-capable NVIDIA GPU;
reading requires the separate ``webdataset`` package.

.. literalinclude:: ../../tests/docs/python/test_webdataset.py
   :language: python
   :start-after: # [snippet:doc-webdataset-basic-video]
   :end-before: # [/snippet:doc-webdataset-basic-video]
   :dedent: 4



KITTI samples (``kitti-frame-v1``) contain the native files as distinct
components with unchanged bytes: ``image_2.png``, ``label_2.txt``, ``calib.txt``
and, with ``write_depth=True``, ``depth.png``. Label rows keep KITTI pixel and
camera-frame conventions; an empty ``label_2.txt`` is a negative example.
The writer stored uint16 depth as metres times ``depth_scale`` clipped to
0-65535 and truncated to an integer, so fractions are lost and saturated values
cannot be recovered by dividing by ``depth_scale``. ``depth_scale`` is recorded
in ``writer_config`` and in each sample's ``metadata.json``. ``KITTIWriter`` is
selected by exact type. Finalized export requires the same frame set in every
directory and a ``depth`` directory exactly when ``write_depth`` is set.

BOP samples (``bop-frame-v1``) contain one frame of the configured
``{scene_id:06d}`` scene: ``rgb.png``, 16-bit ``depth.png``, every
``mask.<obj>.png`` and ``mask_visib.<obj>.png`` file named after the frame, a
generated ``json`` record and ``metadata.json``. Encoded bytes are unchanged.
The record has ``schema_version``, ``scene_id``, ``frame_id``, the frame's
unchanged ``scene_camera``, ``scene_gt`` and ``scene_gt_info`` entries (``null``
when the scene document is absent) and ``masks``: per ``scene_gt`` entry, its
``obj_id`` and the ``mask``/``mask_visib`` component names, or ``null`` for an
absent file. Entries that share an ``obj_id`` reference the same native mask
file. Object IDs, intrinsics, millimetre poses and ``depth_scale``
(``depth_mm = depth * depth_scale``) keep native values; the manifest adds the
sorted dataset ``object_ids``. Finalized export reads frames from
``scene_camera.json``, requires each frame's RGB and depth, ``cam_K`` and an
entry in every present scene document, checks ``depth_scale`` against the
writer, and rejects non-finite values, negative or non-integer ``obj_id``
values and files that belong to no frame or scene document. ``source-paths.json`` lists the three scene
documents as ``shared`` entries.

YOLO samples (``yolo-frame-v1``, ``YOLOFrameAdapter``, experimental) contain
the native ``jpg`` or ``png`` image, a ``txt`` label and a ``json`` record with
``schema_version``, ``task``, ``split``, ``frame_index``, ``objects`` and
``classes``. ``txt`` rows keep the writer's normalized detection or polygon
text; only the leading class index is mapped, by class name, to the dataset-wide
table in manifest ``metadata.classes``. Explicit ``YOLOWriterConfig.classes``
define that table; otherwise it grows in first-seen order across episodes. Each
record's ``classes`` lists its episode's classes with dataset ``id`` and native
``source_id``. When the episode's native indices differ, ``native.txt`` holds
the unchanged native label. A negative example has an empty ``txt``. ``split``
(also in ``metadata.json``) is the writer's original per-frame assignment, so
an episode can hold both splits; episodes are not repartitioned and shards are
not split-homogeneous. Filter by split in the reader; manifest
``metadata.split_counts`` gives the per-split sample counts. Finalized export
requires the writer's ``data.yaml``, one image and label per frame and, with
explicit classes, matching ``data.yaml`` names.

Completion manifest
~~~~~~~~~~~~~~~~~~~

``<dataset_id>/manifest.json`` is published after every shard. Its
``schema_version`` is ``1``; readers must ignore unknown fields so later minor
additions remain compatible. Fields:

- ``format`` (``"webdataset"``), ``dataset_id`` and ``generator``
  (``name``/``version`` of the producing library).
- The configured limits, ``allow_episode_spanning`` and
  ``oversized_episode_policy``.
- ``episodes``: sorted episode IDs; ``samples``: total sample count.
- ``shards``: in order, each with ``path``, ``bytes``, ``sha256``, ``samples``,
  ``episodes``, ``episode_samples`` (sample count per episode in that shard)
  and ``oversized``.
- ``metadata``: caller or capture metadata. Managed capture and finalized
  export record ``writer``, ``writer_config``, ``sample_schema``,
  ``sample_unit``, ``provenance`` (``capture`` or ``finalized-export``) and
  adapter metadata such as the final COCO ``categories`` table.
- ``auxiliary``: optional list of ``path``, ``bytes`` and ``sha256`` for
  metadata files published before the manifest.

Managed capture and finalized export publish ``auxiliary/source-paths.json``.
It is informational, not a reconstruction contract: for each episode it lists
``shared`` episode-wide documents (COCO annotation files, BOP scene JSON files,
YOLO ``data.yaml``) and, per sample key, each native component's path relative to the episode's
native output root. Clip adapters also list ``clips`` with each ``clip_index``,
its ``sample`` key and ``frame_ids``. Symlinks keep their own names. Generated
components, such as per-image COCO JSON, are omitted. For managed capture the
native root was a temporary directory that no longer exists.

Unsupported writers fail before capture; the generic packer preserves supplied
encoded bytes and metadata but does not infer relationships in arbitrary datasets.
