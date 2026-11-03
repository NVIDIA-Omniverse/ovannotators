# Changelog

## [0.2.0] - 2026-10-16

### Added
- `PointCloudResult` holds Warp arrays on CPU or CUDA through one result type,
  with an optional CUDA valid count for capacity-sized GPU output. Pointcloud
  helpers return Warp arrays and use Warp's transfer/interoperability APIs;
  there is no separate CUDA result class or NumPy conversion method.

- Reuse temporary CUDA pointcloud output allocations through bounded, view-safe leases (returned CUDA outputs opt in with `reuse_gpu_outputs=True`) and reduce stable compaction scan storage with four-pixel groups. Point ordering, numeric results, and default CPU delivery are preserved; images above `2**31-4` pixels are rejected before allocation to prevent group-index overflow.

- CUDA pointcloud batching with bounded reusable scratch/pinned staging, and optional owned GPU-resident outputs with device-side valid counts (`output_device="cuda"`). Compact Warp CPU output remains the default.

- Add a true `ovstorage` SDK backend (registry name `ovstorage`, optional
  `[ovstorage]` extra pinned to `ovstorage>=0.3,<0.4` and included in
  `all-cpu`) that writes through a caller-owned, built Stack, including its
  cloud connections and caching layers. Construction raises
  `ConfigurationError` when the Stack has no route or signed-in connection for
  `root_uri`, or when `overwrite=False` targets a route without fail-if-exists
  writes. `makedirs` is a no-op because writes create parent directories. The
  ovstorage Linux wheels require glibc 2.34 or later. It reuses the names of
  the removed HTTP backend with a different constructor,
  `OvstorageBackend(OvstorageBackendConfig(root_uri=...), stack)`.

- Basic, Cosmos Transfer, and LeRobot video writers support
  `video_codec="hevc"`, `video_lossless=True`, and optional NVENC
  `video_preset="P1"` through `"P7"`. Lossless HEVC preserves normalized
  uint8 RGB exactly using full-range GBR 4:4:4, with GPU-resident packing for
  CUDA inputs. It requires compatible NVENC hardware, even fixed dimensions,
  and a reader supporting HEVC Range Extensions; it never falls back to lossy
  encoding. H.264 defaults remain unchanged. Cosmos and LeRobot metadata
  report the selected codec and pixel format; every Cosmos manifest clip,
  including default H.264 clips, carries a `video_encoding` object.
- Add the `openlabel` image writer for ASAM OpenLABEL 1.0 with 2D boxes and
  optional instance polygons in a layout readable by FiftyOne.
- Add the episode-aware `umi_replay_buffer` profile writer for UMI-compatible
  Zarr v2 zip stores with explicit signal mappings and episode boundaries.
  Install the `[umi]` extra (`zarr>=2.16,<3`, `numcodecs`,
  `imagecodecs-numcodecs`); the writer supports Zarr 2 only and raises on Zarr 3.
  The extra is part of `all-cpu`.
- Allow callers to choose EXR compression through
  `Backend.write_array(..., exr_compression=...)` and
  `BasicWriterConfig.exr_compression`. The default remains lossless ZIP. Accepted
  names are `none`, `rle`, `zips`, `zip`, `piz`, `pxr24`, `b44`, `b44a`, `dwaa`,
  `dwab`, `htj2k256`, and `htj2k32`; an unknown name raises `ValueError` when
  the `BasicWriterConfig` is created, and a non-EXR encoding raises `ValueError`
  in the backend.
- `list_profiles`, `describe_requirements`, and `register_dataset_profiles`
  expose recommended dataset tasks, config options, capture inputs, camera
  guidance, timing, non-render inputs, and depth representations across frame
  and episode writers without constructing writers or rendering.
- Add `WebDatasetPackager`, `WebDatasetPackagerConfig` and `SampleFiles` for
  packaging existing encoded files, plus experimental managed
  `DatasetCapture`/`CaptureEpisode` APIs for COCO and Basic frame samples.
  Packaging supports readable byte limits, sample and episode limits, default
  episode preservation, optional spanning, explicit oversized shards, portable
  extraction identities, and versioned completion manifests. A rejected episode
  is discarded while capture stays open; empty frames, empty episodes and
  non-portable component names are rejected. Packaging failures raise the new
  `PackagingError`. No new runtime dependency is required; the test extra adds
  the official WebDataset reader for compatibility verification.
- Add experimental writer adapters (`CaptureAdapter`, `AdapterContext`,
  `BasicFrameAdapter`, `COCOFrameAdapter`) for Basic/COCO capture and custom
  writers or built-in writer subclasses, plus `export_finalized_dataset` for
  existing local episode output. `DatasetCapture` gains an `adapter=` keyword
  to select an adapter for custom writers and built-in writer subclasses; exact
  `BasicWriter` and `COCOWriter` types still select a built-in adapter. The
  export checks COCO split and panoptic settings against the files. Adapters
  must define non-empty `sample_schema` and `sample_unit` strings; construction
  rejects an adapter without them. `WebDatasetPackager.finalize` gains an
  `auxiliary=` keyword for checksummed metadata files published before the
  manifest, and the manifest gains an `auxiliary` list plus `sample_unit` and
  `provenance` metadata. Managed datasets publish an informational
  `auxiliary/source-paths.json` native-path map.
- Add experimental episode and clip sample units to `CaptureAdapter`
  (`sample_unit`, validated before capture). Episode adapters package one
  complete finalized episode as one sample. Clip adapters close clips through
  the new `CaptureEpisode.end_clip()` and `CaptureAdapter.cut_clip` hook,
  record `ClipSpan` clip index and frame IDs in `AdapterContext.clips`, and
  package one sample per clip; empty clips and clip boundaries for frame or
  episode adapters raise `PackagingError`. `export_finalized_dataset` supports
  both units through `discover_frames` and the new `discover_clips` hook, and
  `source-paths.json` lists clip spans. `allow_native_overwrite` lets an adapter
  accept writers that rewrite a native file during an episode. Basic and COCO
  frame adapters are unchanged.
- Add experimental `BasicVideoEpisodeAdapter` (`basic-video-episode-v1`), which
  packages `BasicWriter(video=True)` output as one episode sample: every episode
  MP4 with unchanged encoded bytes, original per-frame files selected by
  `frame_files` (default `("*.json",)`), and an `episode.json` record of each
  stream's codec, size, timescale, frame count, presentation timestamps and
  source frame indices. It is opt-in through `adapter=`; the default Basic
  adapter still rejects video mode and now names this adapter in its error.
- Add experimental `KITTIFrameAdapter` (`kitti-frame-v1`), selected for exact
  `KITTIWriter` capture and finalized export. Samples keep the native image,
  label, calibration and optional depth files as separate, byte-identical
  components.
- Add experimental `BOPFrameAdapter` (`bop-frame-v1`), selected for exact
  `BOPWriter` capture and finalized export. Each WebDataset sample holds one
  scene frame's RGB, depth, every object mask and its `scene_camera`,
  `scene_gt` and `scene_gt_info` entries with native IDs, units and
  `depth_scale`; absent optional masks and scene documents are recorded as
  `null`.
- Add experimental `YOLOFrameAdapter` (`yolo-frame-v1` samples), selected for
  exact `YOLOWriter` capture and usable with `export_finalized_dataset`. Samples
  keep native images, normalized detection/polygon label text, empty labels for
  negative examples and each frame's original `train`/`val` split as metadata.
  Label class indices are mapped by name to one dataset-wide table recorded in
  the manifest with per-split sample counts; episodes are not repartitioned.
- Name the 98 ovrtx 0.5 robot-scene AOVs explicitly in `BasicWriter`: write six
  display-ready uint8 outputs as PNG/JPEG, 55 float image passes (including
  path-tracing lighting decomposition) as EXR, and 37 ID, packed-record, and
  auxiliary buffers as NPY. `docs/public/render_outputs.rst` lists every name
  and filename.
- `BasicWriterConfig` and `CosmosTransferWriterConfig` add
  `segmentation_palette_accessibility` (`none`, `protanopia`, `deuteranopia`,
  `tritanopia`) and `segmentation_palette_path`. Segmentation colors come from
  one lazily generated perceptual palette, and the optional JSON state file
  keeps color assignments across writer sessions. A state file belongs to one
  writer at a time and one key mode and accessibility mode.
- `CosmosTransferWriterConfig` adds `segmentation_color_key` (`instance_id`,
  `stable_id`, `semantic_id`, `semantic_label`) to choose what identifies a
  segmentation color.
- `BasicWriter` colorizes `StableIdSegmentation`, writing
  `stable_id_segmentation_colorized.png` per frame and, with `video=True`,
  `stable_id_segmentation_colorized.mp4`. `BasicWriterConfig.semantic_color_key`
  optionally assigns semantic colors by resolved label instead of numeric ID.

### Added
- Add the exact `cosmos_multiview` profile writer for Cosmos 3.5 with deterministic
  `episodes.lance` registration, stride-based half-open windows, target-rate
  timestamp-aligned immutable sample assets with bracketed interpolation,
  exact Arrow/URI validation, `samples.lance`, and
  machine-readable external-QA handoff records. Install
  the pinned `cosmos-multiview` extra; the existing `cosmos_transfer` writer is
  unchanged.
- Add the independently named `groot_lerobot_v2` profile writer for the
  pinned Isaac-GR00T `n1.7-release` LeRobot v2-derived dataset contract. It
  requires explicit video, state, action, embodiment, horizon, representation,
  normalization, and relative-action configuration (`GrootLeRobotV2WriterConfig`
  with `GrootVideoModality`, `GrootStateModality`, `GrootActionModality`, and
  `GrootNormalizationConfig`) and records all configured transformations in the
  profile loss report. The output follows the pinned `v2.1` physical schema;
  unsupported relative EEF pose conversion is rejected rather than approximated.
  `validate_groot_lerobot_v2` checks a finished dataset. Install the `lerobot`
  extra (`pyarrow`) and the `video` extra (`PyNvVideoCodec`); video encoding is
  NVENC, so it needs an NVIDIA GPU. Frame timestamps must follow a constant
  frame rate grid, `frame_index` must be contiguous within each episode, and
  camera frames must have even height and width.
- Add an episode-aware `ProfileWriter` lifecycle, separate
  `ProfileWriterRegistry`, producer-neutral episode/sample/stream/tensor
  envelopes, and common `_ovannotators/` conformance artifacts for exact
  training-dataset profiles. Existing frame writers and registry names are
  unchanged.
- `BasicWriter` can also write video. Set `video=True` on `BasicWriterConfig` to
  get one H.264 MP4 per video-capable render var, spanning every frame, next to
  the per-frame files it already writes. Tune it with `video_fps` (default 24),
  `video_vars` (default: every video-capable var present) and `video_dir`
  (default `videos`). Encoding is NVENC via `PyNvVideoCodec`, so it needs a GPU
  and the `[video]` extra; leaving `video=False` keeps `BasicWriter` CPU-only.

### Changed
- ovannotators requires Python 3.11 or newer and Pillow 13 (`>=13.0.0,<14`).
  Python 3.10 reached end of life in October 2026, and Pillow 13.0.0 is the
  first release that fixes several high-severity Pillow CVEs.
- `PointCloudResult` stores Warp arrays on both CPU and CUDA. CPU construction
  and delivery return compact Warp arrays; attributes must use Warp-supported
  numeric dtypes. CUDA-resident output uses the same result type with an optional
  device-side valid count.
- `BasicWriterConfig.video_fps`, `CosmosTransferWriterConfig.fps` and
  `LeRobotWriterConfig.fps` accept fractional frame rates. NTSC rates such as
  23.976, 29.97 and 59.94 are stored exactly as `N*1000/1001` in the MP4
  timing. A non-positive or non-finite rate raises `ValueError` when the config
  is created (for `BasicWriterConfig`, when `video=True`).
- CUDA video frames stay on-device through normalization, cropping, channel
  packing and NVENC encoding. `BasicWriter`, `LeRobotWriter` and
  `CosmosTransferWriter` avoid downloading RGB frames solely to upload them
  again for video encoding. Normal visualization and RGB-derived Cosmos Canny
  also stay on-device; only Canny convergence/reduction scalars reach the CPU.
- Buffered `LeRobotWriter` and `CosmosTransferWriter` RGB frames are snapshotted
  on their source device, so renderer buffer reuse cannot overwrite earlier
  frames. CUDA clips/episodes hold GPU memory until `cut()` or `finalize()`;
  LeRobot releases the snapshots after successful finalization.
- NVENC sessions initialize on the first input frame to select its CUDA context
  and stream. NVENC remains required for video encoding.
- `point_cloud_from_depth` accepts a CUDA construction `device` for stable GPU
  pixel compaction and backprojection, with synchronous CPU output delivery.
  CUDA is the default; use `device="cpu"` explicitly for CPU-only construction.
  CUDA inputs avoid a host round trip before construction. CUDA is required
  unless CPU construction is explicitly selected.
- Use Warp for CPU point-cloud filtering, compaction, backprojection and
  attribute gathering at every image size. Remove the NumPy fallback and
  pixel-coordinate cache; inputs remain unmodified and outputs preserve raster
  order. CPU intrinsics must be finite and representable as float32, matching
  the CUDA path.
- `CosmosTransferWriter` keeps instance-to-color assignments for the writer's
  full dataset session, including across `cut()` clip boundaries, and supports
  thousands of unique colors before reuse. The fixed 255-color palette is gone,
  so segmentation clips no longer reuse colors after 255 instances.
- `BasicWriter` default segmentation colors change: `SemanticInstanceSegmentation`
  and `SemanticSegmentation` colorized outputs now use the shared perceptual
  palette instead of per-ID random RGB colors, and each ID keeps its color
  across frames.
- The `Backend` protocol's `write_array` gains a keyword-only
  `exr_compression: str | None = None` parameter. The built-in disk, S3, and
  ovstorage backends accept it. `BasicWriter` passes it only when
  `BasicWriterConfig.exr_compression` is set, so existing custom backends keep
  working by default; a custom backend must accept the keyword before users
  select a scheme, and static type checkers report a backend without it as not
  matching the protocol.
- `BasicWriter` output filenames change for the 98 newly named ovrtx 0.5 AOVs.
  They were written as the lowercased `sourceName` with a `.npy` extension; they
  are now written as the snake_case name with the extension for their format.
  Update any code that reads these paths. Examples: `ptworldpos.npy` is now
  `pt_world_pos.exr`; `ptzdepth.npy` is now `pt_z_depth.exr`;
  `ldrcolorsd.npy` is now `ldr_color_sd.png` (or `.jpg` with
  `image_format="jpg"`); `stableid.npy` is now `stable_id.npy`;
  `boundingbox2dtightsd.npy` is now `bounding_box_2d_tight_sd.npy`. `Depth` is
  now `depth_native.exr` (it was `depth.npy`), and `DiffuseAlbedo` is now
  `diffuse_albedo_native.png`. Files for names that already had an explicit
  mapping are unchanged. The full list of names and filenames is in
  `docs/public/render_outputs.rst`.
- `BasicWriter` and the EXR codec now write float32 arrays as 32-bit float EXR
  instead of narrowing them to float16, so `hdr_color`, `normal`,
  `camera_3d_position`, `motion_2d`, and `depth_format="exr"` depth files are
  roughly twice as large as before.
- EXR writes now raise `TypeError` for float64, signed integer, `bool`, and
  `uint64` arrays, which were previously cast to float16. Cast the array to
  float16 or float32 first, or write it as `.npy`. A `TypeError` raised for one
  render var aborts `BasicWriter.write_frame` for that frame.
- `read_exr` returns the stored channel type (float16, float32, or uint32)
  instead of always float16, and raises `ValueError` when the channels have
  mixed types.
- COCO instance-segmentation output now requires
  `SemanticInstanceSegmentation` rather than silently writing empty masks.
- Cosmos Transfer validates the RenderVars required by its selected conditioning
  options, including normals for shaded outputs; color-only Canny no longer
  requires unused depth and segmentation.
- `CosmosTransferWriter.__init__` now raises `ValueError` for unknown
  `conditioning_types` entries or an unsupported `canny_source`, instead of
  accepting them at construction.
- Start from the root README's task-first navigation, then use the documentation
  map, example index, troubleshooting guide, or complete agent-skill catalog for
  the workflow you need.
- `examples/minimal_capture.py` now consumes the untouched OVRTX frame carried
  by `usd_randomizer_omniverse.FrameOutput`. Scene population, ordinals, and
  renderer attachment belong to that OVStage-native frontend; `OvrtxOutput`
  remains the decoder between renderer output and dataset writers. The example
  calls `Scene.update_from_usd_time()` before each render so time-sampled scene
  state advances independently from renderer simulation time.
- Target ovrtx 0.5 and ovstage 0.2, from the same release train. The `[dev]`
  extra now asks for `ovrtx>=0.5,<0.6` and `ovstage>=0.2,<0.3`.
- Call `ovrtx.register_schema_paths()` before creating or populating an
  `ovstage.Stage`; ovrtx 0.5 no longer registers its USD schemas on import or
  on `Renderer()`, and sensor products fail to map without it.
- Render vars stay keyed by ovrtx `sourceName`, although ovrtx 0.5 re-keyed
  `FrameOutput.render_vars` by RenderVar prim path. `OvrtxOutput` absorbs the
  change, so writers and the dicts you pass to `write_frame` are unchanged.
- `OvrtxOutput.extract()` also accepts a full RenderVar prim path, which picks
  between two prims in one frame that share a `sourceName`.
- `BasicWriter` writes the composite `PointCloud` output as `point_cloud.npz`
  (one member per tensor, `param_`-prefixed params), replacing
  `generic_model_output.npz`: ovrtx 0.5 removed `GenericModelOutput`.
- Author `omni:sensor:modelVersion` as `"latest"` or a positive integer string
  such as `"1"`; ovrtx 0.5 rejects empty and dotted values.
- The `[dev-cpu]` extra pins `ruff==0.15.16` to match CI's lint job.
- Target ovrtx 0.4: stage population moves to `ovstage.population` driving an
  externally owned `ovstage.Stage` that the renderer borrows via
  `Renderer.attach_ovstage()`. The renderer's own USD entry points
  (`open_usd`, `open_usd_from_string`, `update_from_usd_time`) are deprecated.
- `BasicWriter` decodes raw ovrtx buffers for the JSON-encoded ID map render
  vars (`SemanticIdMap`, `StableIdMap`, `StableIdSemanticIdMap`, `InstanceMap`)
  instead of dumping them as bytes.

- `RendererOutputRegistry.register()` raises `ValueError` when a different
  class claims an already-registered `renderer_name`, matching the writer,
  backend, and profile-writer registries. Re-registering the same class is
  still a no-op.
- A third-party writer entry point that fails to import is logged as a warning
  naming the entry point instead of being dropped silently.
- `BOPWriter` writes mask PNGs through `backend.write(path, bytes)` after
  encoding them with the shared image codec, instead of
  `backend.write_array(..., encoding="png")`. A custom backend that special-cases
  PNG in `write_array` no longer sees BOP masks; handle them in `write`.
- Faster hot paths with identical output: `COCOWriter` panoptic annotations
  label each frame in one pass, `BasicWriter` reads a colorized GPU array back
  once, `LeRobotWriter` and `BOPWriter` encode each image once, and the MP4 muxer
  finds H.264 start codes with `bytes.find`.
- With the default `multipart_threshold_mb=8`, `S3Backend` now sends encoded
  payloads of 8 MiB or more, such as float32 depth at 4K or large EXR frames,
  as multipart uploads (create, parts, complete) instead of one `PutObject`.
  The credentials need multipart permissions such as
  `s3:AbortMultipartUpload`; raise `multipart_threshold_mb` to keep
  single-request uploads. A non-positive `multipart_threshold_mb` raises
  `ConfigurationError`.

### Deprecated
- `DepthSD` is deprecated by ovrtx 0.5 and still works; prefer
  `DistanceToImagePlaneSD` or `DistanceToCameraSD`.

### Removed
- **Breaking:** Remove the HTTP storage backend, `OvstorageBackend` and
  `OvstorageBackendConfig` (registry name `ovstorage`), and the
  `[storage-api]` extra. It used a custom `PUT`/`GET`/`HEAD` protocol that no
  supported service implements. For another HTTP endpoint, register a custom
  backend.

### Fixed
- CPU delivery of CUDA pointclouds avoids a redundant device-wide fence when
  recycling internal output buffers, so it does not wait for unrelated CUDA
  streams. Caller-exposed pooled CUDA outputs retain their disposal fence.
- Cosmos Transfer rejects lossless RGB resolution changes while accepting
  frames, before altering clip buffers or writing partial output.
- Lossless video rejects empty selected BasicWriter frames and LeRobot
  resolution changes across chunk boundaries instead of silently dropping
  frames or writing videos inconsistent with dataset metadata.
- Cosmos Transfer and LeRobot allow retrying finalization after an encoding
  failure; retrying partial output requires a backend that permits overwrites.
- `BasicWriter.finalize()` writes every video stream and saves the
  segmentation palette even when one stream fails, then raises `RuntimeError`
  naming the failed streams.
- Lossless Cosmos Transfer and LeRobot reject odd or empty first-frame
  dimensions when the frame is written, before any output is produced.
- A video that starts with host frames and later receives CUDA frames keeps
  one NVENC session instead of starting a second one whose parameter sets the
  MP4 muxer could reject.
- A CUDA-array-interface-only frame on a multi-GPU host is imported on the
  GPU that owns its memory rather than the current Warp device.
- `BasicWriter` colorizes CUDA `NormalSD` given as 2D `vec3`/`vec4` arrays
  and rejects other shapes with a clear `ValueError`.
- Cosmos Transfer RGB Canny uses the same grayscale scaling for host and
  CUDA frames: integer RGB is always divided by 255, so dark frames no longer
  produce different edges depending on where they live.
- CUDA-array-interface-only video inputs do not need sequence methods, and
  their producer stays alive while Warp aliases the input buffer.
- `BasicWriter` imports DLPack-only CUDA frames before EXR serialization, so
  raw normals and their video visualization can be written together.
- Preserve CPU Warp normal visualization when video output is disabled.
- NVENC flush errors propagate and discard buffered output instead of writing
  a truncated MP4.
- `S3Backend` keeps paths below the configured prefix, like `DiskBackend`
  keeps them below `root_dir`. A path that is absolute or contains a `..` or
  empty segment now raises `BackendWriteError` or `BackendReadError` (or makes
  `exists` return `False`) without a request to S3, instead of being joined
  into a key outside the prefix. `list` now strips a configured prefix that
  starts with `/`, so its results can be passed back to `read`.
- Preserve float32 EXR channels on write and read instead of narrowing them to
  float16, which overflowed large camera-position values and depth sentinels.
  Preserve uint32 EXR channels as integers and reject types that cannot be
  encoded without losing values.
- `CosmosTransferWriter` segmentation and shaded-segmentation videos keep one
  color per instance for the whole clip, including when instances leave and
  re-enter view, and every channel of a multi-channel instance ID takes part
  in the mapping. Colors previously shifted from frame to frame. Color
  assignments persist across `cut()` clip boundaries for the writer's whole
  session, and across sessions when `segmentation_palette_path` is set.
- `S3Backend` honors `S3BackendConfig.multipart_threshold_mb`: payloads at or
  above the threshold use a managed multipart upload instead of a single
  `PutObject`, so large videos and arrays no longer hit the 5 GiB
  single-request limit. Part size follows the threshold, bounded to 5-64 MiB.
  The setting was previously accepted but ignored. Values above 5120 MiB are
  capped to that limit.
- Reject profile artifact emission before writing metadata when serialization or
  structural validation has failed.
- `BasicWriter`'s `video=True` output works with PyNvVideoCodec 2.2, which
  returns encoded packets as a list of mappings rather than raw bytes. Videos
  written against 2.2 were previously failing with a `TypeError`; 2.1 keeps
  working.
- `DiskBackend` rejects paths that resolve outside `root_dir`, including `..`
  traversal and absolute paths outside the root. `read()` and `list()` raise
  `BackendReadError`, `write()`, `write_array()` and `makedirs()` raise
  `BackendWriteError`, and `exists()` returns `False`. Absolute paths inside
  `root_dir` remain valid.
- `DiskBackend` rejects symlinks, to files or directories, that point outside
  `root_dir`, and `list()` no longer returns symlinks whose targets are outside
  `root_dir` or are broken or looping. Symlinks that point inside `root_dir`
  still work.
- `DiskBackend` resolves a relative `root_dir` once, when the backend is
  constructed, instead of on every call.

## [0.1.0] - 2026-06-15

### Added
- Initial release
