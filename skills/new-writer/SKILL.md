---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: new-writer
description: >
  Create a new ovannotators output-format writer or episode-aware profile writer
  from a spec (a website/URL, a local file, or a prose prompt). Use when the user
  asks to add, implement, or debug a writer that persists rendered frames,
  episodes, or aligned streams in a dataset format such as COCO, KITTI, YOLO,
  BOP, LeRobot, Cosmos, or any custom format -- e.g. "add a new writer",
  "implement VOC/Pascal format", "create a writer for X dataset", "export
  annotations as Y".
license: Apache-2.0
version: "0.1.1"
author: NVIDIA ovannotators
tags:
  - ovannotators
  - writers
  - datasets
tools:
  - Read
  - Grep
---

# Creating a New Writer

## When to Use

Use this skill when the user wants to turn rendered frames into a specific
dataset layout: adding a new writer, implementing a published format (COCO,
KITTI, YOLO, BOP, LeRobot, Pascal VOC, ...), or debugging an existing writer.
The format spec may arrive as a **website/URL**, a **local file**, or a **prose
prompt**.

## Inputs

Resolve the format spec in this order, then restate the requirements before
coding:

- **URL** — fetch it and extract the file layout, required fields, and
  coordinate/units conventions.
- **Local file** — read it (a format README, JSON schema, sample annotation).
- **Prompt** — use the description directly; ask only if a load-bearing detail
  (coordinate origin, normalization, split policy) is missing.

Also gather, in this order (repository files and the snippets below first, then
the user request, then broader context):

- The per-frame files and global files the target format expects.
- Which render vars (AOVs) the format needs -> the writer's
  `required_render_vars` (see Key Types / Functions).
- Naming/units conventions (e.g. normalized vs pixel bboxes, depth scale).

Treat the referenced `> **Source:**` snippets as the API source of truth.

## Prerequisites

- Work in an ovannotators checkout where `tests/docs/python/test_custom_writer.py`
  exists; read the referenced snippet before writing code.
- Know the writer protocol contract: `src/ovannotators/writers/_base.py`.
- Study the closest built-in writer in `src/ovannotators/writers/`
  (`basic.py`, `coco.py`, `kitti.py`, `yolo.py`, `bop.py`, `lerobot.py`,
  `cosmos_transfer.py`) before inventing a new pattern.

## Choose the writer contract

Use the frame-oriented ``Writer`` only when every input is an independent
renderer frame. Use ``ProfileWriter`` when the format requires explicit
episodes, authoritative timestamps, independently named streams, policy
signals, or separate canonical-loader/consumer acceptance. Profile writers use
``ProfileWriterRegistry`` and the ``ovannotators.profile_writers`` entry-point
group; they do not change ``Writer.write_frame`` or ``WriterRegistry``.

> **Source:** `tests/docs/python/test_profile_writer.py` snippet `doc-profile-writer-class`

Follow the complete lifecycle and keep ``finalize`` idempotent. A common
``_SUCCESS`` marker means serialization and structural validation completed;
canonical-loader and consumer acceptance remain independent conformance fields.

> **Source:** `tests/docs/python/test_profile_writer.py` snippet `doc-profile-writer-lifecycle`

## Instructions

1. Resolve the spec (see Inputs) and list the exact output files and the AOVs
   each needs.
2. Define the inputs required by the selected writer contract:
   - For a frame-oriented `Writer`, map output AOVs to ovrtx render-var names
     and set `required_render_vars` (see Key Types / Functions). Declare only
     strictly required vars; probe optional ones with `render_vars.get(...)`
     at runtime.
   - For a `ProfileWriter`, define the required episode identity and metadata,
     stream descriptors, `NamedTensor` fields, authoritative sample timestamps,
     and observation and policy-signal fields in the producer-neutral
     `EpisodeEnvelope` and `SampleEnvelope` contracts.
3. For a frame-oriented `Writer`, create
   `src/ovannotators/writers/<format>.py`; for a `ProfileWriter`, create
   `src/ovannotators/profile_writers/<format>.py`. Include the NVIDIA
   Apache-2.0 SPDX header and a Google-style docstring.
4. Add a `@dataclass <Format>WriterConfig` for tunables.
   > **Source:** `tests/docs/python/test_custom_writer.py` snippet `doc-writer-config`
5. Implement the writer class and register a frame-oriented writer with
   `@WriterRegistry.register`, or a profile writer with
   `@ProfileWriterRegistry.register`. Validate first, write incrementally, and
   flush deferred state idempotently in `finalize`.
   > **Source:** `tests/docs/python/test_custom_writer.py` snippet `doc-writer-class`
6. Export the writer + config from its `writers/` or `profile_writers/`
   `__init__.py` (and the top-level `src/ovannotators/__init__.py` `__all__`)
   for the direct-import path. Add installable discovery through the
   `ovannotators.writers` or `ovannotators.profile_writers` entry-point group,
   respectively (see Output Format).
7. Add a CPU-only test next to the existing snippet tests, then run it. For a
   profile writer, also run the profile-writer snippet tests.
   > **Source:** `tests/docs/python/test_custom_writer.py` snippet `doc-writer-test`
8. Document the new writer in `docs/public/api.rst` (registry name, GPU need,
   required vars, config table, output tree).

## Discovering capture requirements

Use `list_profiles` and `describe_requirements` before capturing frames. The
returned config and requirements refer to the same selected task. Register
extension discovery with `register_dataset_profiles`; keep its resolver free of
writer construction, rendering, and optional-codec imports. Specify non-render
inputs and unresolved caller choices rather than inventing robot semantics or
scene units.

> **Source:** `tests/docs/python/test_requirements.py` snippet `doc-dataset-requirements`

For physical 16-bit depth, supply source stage units and a range covering the
clip; capture values and storage precision are separate contracts.

> **Source:** `tests/docs/python/test_requirements.py` snippet `doc-depth-requirements`

## Retrieving the writer

A registered writer is reachable two equivalent ways — by registry name (for
config-driven code, no import of the writer module needed) or by direct import
(for static typing/autocompletion):

> **Source:** `tests/docs/python/test_custom_writer.py` snippet `doc-writer-register`

## Serializing configs

`*WriterConfig` classes are plain dataclasses so they can be persisted to a file
and reloaded. Use `config_to_dict` / `config_from_dict` (they handle
`pathlib.Path` and nested config dataclasses, unlike a bare
`dataclasses.asdict` + `Config(**data)`):

> **Source:** `tests/docs/python/test_config_serialization.py` snippet `doc-config-save-load`

To make the whole pipeline data-driven, pair a registry name with its config
payload in one file and dispatch through the registry:

> **Source:** `tests/docs/python/test_config_serialization.py` snippet `doc-config-registry`

## Output Format

- For explanations: cite the relevant API names, the source snippet(s), and the
  resulting output-file tree.
- For code changes: summarize the module added, the exports/entry-point wired,
  the docs section, and the test run.

Entry-point registration (small inline fragment) makes the writer auto-register
on install without an explicit import:

```toml
# pyproject.toml
[project.entry-points."ovannotators.writers"]
my_format = "ovannotators.writers.my_format:MyFormatWriter"
```

## Scripts

This skill has no scripts.

## Key Types / Functions

| Symbol | Purpose |
|--------|---------|
| `Writer` (Protocol, `writers/_base.py`) | `write_frame(frame_id, render_vars, backend)` + `finalize(backend)`; class vars `name`, `required_render_vars` |
| `WriterRegistry` (`writers/_registry.py`) | `register` (decorator), `get(name)`, `create(name, *a, **kw)`, `list_names()` |
| `validate_required_vars(name, required, render_vars, frame_id)` | Per-frame guard; raises `MissingRenderVarError`. Call it first in `write_frame` |
| `iter_bboxes_2d` / `iter_bboxes_3d` (`writers/_helpers.py`) | Decode structured bbox payloads into `BBox2D`/`BBox3D` dataclasses |
| `id_label_map_from_render_vars` / `id_prim_path_map_from_render_vars` | Resolve labels/prim paths from `SemanticIdMap` / `StableIdMap` |
| `_to_numpy` (`writers/_helpers.py`) | Materialize a NumPy view from a Warp array for host-side work |
| `Backend` (`backends/_base.py`) | `write`, `write_array(path, arr, encoding="npy"\|"png"\|"jpg"\|"exr")`, `read`, `list`, `exists`, `makedirs` |

Common render-var (AOV) keys passed in `render_vars`:

| Key | Shape / type | Notes |
|-----|--------------|-------|
| `LdrColor` | (H, W, 4) uint8 | RGBA sRGB |
| `DistanceToCameraSD` | (H, W) float32 | meters; `inf` = sky |
| `NormalSD` | (H, W, 3\|4) float32 | world-space |
| `SemanticInstanceSegmentation` | (H, W) uint32 | per-instance id |
| `SemanticSegmentation` | (H, W) uint32 | class id |
| `SemanticBoundingBox2DTight` | structured array | decode with `iter_bboxes_2d` |
| `SemanticBoundingBox3D` | structured array | decode with `iter_bboxes_3d` |
| `CameraParams` | structured array | `K`/`R`/`t` are derived (see `extract_camera_params`) |

## Limitations

- The referenced snippets remain the source of truth; add or update a tested
  snippet before documenting new API usage.
- Snippet tests are CPU-only — full ovrtx render-driven validation runs under
  the GPU-marked suite, not here.

## Troubleshooting

- Call `validate_required_vars(...)` at the **top** of `write_frame`; do not
  hand-roll the missing-var check.
- `write_frame` runs once per frame, sequentially, single-threaded; no locking.
- `finalize` must be idempotent (guard with a `self._finalized` flag).
- Frame ids may be non-contiguous and need not start at 0 — use `frame_id`
  as-is; never assume order.
- Tolerate extra keys in `render_vars` beyond `required_render_vars`.
- `render_vars` values may be zero-copy **Warp** arrays; images/EXR/NPY pass
  straight to `backend.write_array`, but convert with `_to_numpy` before JSON,
  structured-array indexing, or `.tolist()`.
- For image vars, pass the array to `backend.write_array(..., encoding="png"|"jpg")`
  **unchanged** — do not force it to host NumPy yourself. The backend encodes via
  `get_image_codec()`, which uses GPU nvImageCodec (the `[gpu]` extra) when a CUDA
  device is present — several× faster, and zero-copy when the frame is already
  GPU-resident — and falls back to Pillow otherwise. Forcing a host copy defeats
  the zero-copy path. Tell users to install `[gpu]` on GPU hosts for faster image
  I/O.
- Backend writes are synchronous; `BackendWriteError` propagates to the caller.

## References

- Writer protocol contract: `src/ovannotators/writers/_base.py`
- Backend protocol: `src/ovannotators/backends/_base.py`
- Built-in writers: `src/ovannotators/writers/`
- Companion skill for storage destinations: `skills/new-backend/SKILL.md`
- Keep skills, snippets, and `docs/public/api.rst` synchronized when the API
  changes.

## WebDataset packaging composition

WebDatasetPackager wraps an ordinary storage backend and accepts explicitly
complete samples and episodes. It is not a Backend or an entry in
BackendRegistry: arbitrary file writes cannot identify
sample boundaries or deferred annotations. Use DatasetCapture, which packages
each writer's output through a CaptureAdapter: built-in adapters are selected
by exact writer type, and `adapter=` selects others. Existing writer and
backend contracts are unchanged.

> **Source:** `tests/docs/python/test_webdataset.py` snippet `doc-webdataset-capture`

For a custom writer, subclass the experimental `CaptureAdapter` and pass it to
`DatasetCapture` with `adapter=`. Define `sample_schema`, `create_writer` and
`collect_samples`, which runs after writer finalization; override
`discover_frames` to package existing finalized native directories. The adapter
is selected per call and is not registered globally.

> **Source:** `tests/docs/python/test_webdataset.py` snippet `doc-webdataset-custom-adapter`

Set `sample_unit = "episode"` when one finalized episode is one sample, or
`sample_unit = "clip"` for clip writers. A clip adapter overrides `cut_clip`
to call the writer's public cut method; producers close clips with
`episode.end_clip()`, and `collect_samples` returns one sample per
`context.clips` span. Override `discover_clips` for finalized export, and set
`allow_native_overwrite = True` only if the writer rewrites a native file,
such as a per-cut manifest; it lifts overwrite protection for the whole
episode backend.

> **Source:** `tests/docs/python/test_webdataset.py` snippet `doc-webdataset-clip-adapter`
