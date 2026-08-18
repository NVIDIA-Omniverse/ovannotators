---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: new-backend
description: >
  Create a new ovannotators storage backend from a spec (a website/URL, a local
  file, or a prose prompt). Use when the user asks to add, implement, or debug a
  backend that persists writer output to a destination such as Google Cloud
  Storage, Azure Blob, an HTTP storage API, a database, or any custom store --
  e.g. "add a GCS backend", "write datasets to Azure", "implement a backend for
  X store".
license: Apache-2.0
version: "0.1.0"
author: NVIDIA ovannotators
tags:
  - ovannotators
  - backends
  - storage
tools:
  - Read
  - Grep
---

# Creating a New Backend

## When to Use

Use this skill when the user wants to send writer output to a new destination:
adding a backend for an object store (GCS, Azure Blob, MinIO), an HTTP storage
API, a database, or any custom store; or debugging an existing backend. The
destination spec may arrive as a **website/URL** (SDK/API docs), a **local
file** (config/sample), or a **prose prompt**.

## Inputs

Resolve the destination spec in this order, then restate the requirements before
coding:

- **URL** — fetch the SDK/API docs and extract the auth model, the put/get/list
  calls, and the path/key conventions.
- **Local file** — read it (client config, sample request, schema).
- **Prompt** — use the description directly; ask only if auth or the
  put/get/list surface is ambiguous.

Also gather, in this order (repository files and the snippets below first, then
the user request, then broader context):

- The client library/SDK and its auth (credentials, env vars, tokens).
- How the store maps the six `Backend` methods (see Key Types / Functions),
  especially whether it has real directories or is a flat key space.
- Optional dependency: which extra package the backend needs.

Treat the referenced `> **Source:**` snippets as the API source of truth.

## Prerequisites

- Work in an ovannotators checkout where `tests/docs/python/test_custom_backend.py`
  exists; read the referenced snippet before writing code.
- Know the backend protocol: `source/ovannotators/backends/_base.py`.
- Study the closest built-in backend in `source/ovannotators/backends/`
  (`disk.py` for the `write_array` encoding dispatch, `s3.py` for the lazy
  optional-import pattern, `storage_api.py` for HTTP) before inventing a new
  pattern.

## Instructions

1. Resolve the spec (see Inputs) and map the store's put/get/list operations
   onto the six `Backend` methods.
2. Create one module `source/ovannotators/backends/<store>.py` with the NVIDIA
   Apache-2.0 SPDX header and Google-style docstrings.
3. Add a `@dataclass <Store>BackendConfig` (endpoint, prefix, auth, options).
   > **Source:** `tests/docs/python/test_custom_backend.py` snippet `doc-backend-config`
4. Implement the backend class with a class-level `name` and all six methods.
   Reuse `write_array`'s encoding dispatch (`npy`/`png`/`jpg`/`exr`) from
   `disk.py`, honor an `overwrite` guard, and raise `BackendWriteError` /
   `BackendReadError` on failure. For third-party SDKs, import lazily and set a
   `_..._AVAILABLE` flag (mirror `s3.py`).
   > **Source:** `tests/docs/python/test_custom_backend.py` snippet `doc-backend-class`
5. Register it: add the class to `_register_builtins()` in
   `source/ovannotators/backends/_registry.py` (or use `@BackendRegistry.register`),
   export it + its config from `source/ovannotators/backends/__init__.py` and the
   top-level `source/ovannotators/__init__.py` `__all__`, and add an entry-point
   line (see Output Format).
6. If the backend needs a new dependency, add an extra under
   `[project.optional-dependencies]` in `pyproject.toml` and fold it into the
   `all-*` extras.
7. Add a CPU-only test next to the existing snippet tests (mock the SDK/HTTP
   client like the S3/storage-api tests do), then run it.
   > **Source:** `tests/docs/python/test_custom_backend.py` snippet `doc-backend-test`
8. Document the new backend in `docs/public/api.rst` (config table + install
   extra).

## Retrieving the backend

A registered backend is reachable two equivalent ways — by registry name (for
config-driven code) or by direct import (for static typing/autocompletion):

> **Source:** `tests/docs/python/test_custom_backend.py` snippet `doc-backend-register`

## Serializing configs

`*BackendConfig` classes are plain dataclasses so they can be persisted to a
file and reloaded. Use `config_to_dict` / `config_from_dict` (they handle
`pathlib.Path` and nested config dataclasses, unlike a bare
`dataclasses.asdict` + `Config(**data)`):

> **Source:** `tests/docs/python/test_config_serialization.py` snippet `doc-config-save-load`

To make the whole pipeline data-driven, pair a registry name with its config
payload in one file and dispatch through `BackendRegistry`:

> **Source:** `tests/docs/python/test_config_serialization.py` snippet `doc-config-registry`

## Output Format

- For explanations: cite the relevant API names, the source snippet(s), and the
  config/auth model.
- For code changes: summarize the module added, the registration + exports +
  optional-dependency wired, the docs section, and the test run.

Entry-point registration (small inline fragment) makes the backend
auto-register on install:

```toml
# pyproject.toml
[project.entry-points."ovannotators.backends"]
my_store = "ovannotators.backends.my_store:MyStoreBackend"
```

## Scripts

This skill has no scripts.

## Key Types / Functions

| Symbol | Purpose |
|--------|---------|
| `Backend` (Protocol, `backends/_base.py`) | The six methods below; all paths are relative to the backend root |
| `BackendRegistry` (`backends/_registry.py`) | `register` (decorator), `get(name)`, `create(name, *a, **kw)`, `list_names()`; `_register_builtins()` wires shipped backends |
| `BackendWriteError` / `BackendReadError` (`exceptions.py`) | Raise on write/read failure; constructed with `path=` and optional `message=` |
| `get_image_codec()` (`_internal.codecs`) | GPU/Pillow image encoder used by `write_array` for `png`/`jpg` |

Backend protocol methods to implement:

| Method | Contract |
|--------|----------|
| `write(path, data)` | Persist raw bytes; honor the overwrite guard |
| `write_array(path, array, encoding="npy")` | Encode (`npy`/`png`/`jpg`/`exr`) then `write`; reject unknown encodings |
| `read(path)` | Return bytes; raise `BackendReadError` if absent |
| `list(prefix="")` | Sorted relative paths under `prefix` |
| `exists(path)` | `bool` |
| `makedirs(path)` | Ensure a directory exists; a no-op for flat key/object stores |

## Limitations

- The referenced snippets remain the source of truth; add or update a tested
  snippet before documenting new API usage.
- Snippet tests are CPU-only and mock external SDKs; live network/store calls
  are out of scope for this suite.

## Troubleshooting

- Keep all paths relative to the configured root/prefix; join the prefix
  internally, never leak absolute keys to callers.
- Object stores have no directories — make `makedirs` a no-op (see `s3.py`).
- Import optional SDKs lazily inside `__init__`/methods and gate on a
  `_..._AVAILABLE` flag so the package imports without the extra installed.
- Wrap SDK/HTTP errors in `BackendWriteError` / `BackendReadError`; don't let
  raw client exceptions escape.
- `write_array` may receive Warp arrays. For `npy`/`exr`, copy to host NumPy
  before serializing (as `disk.py` does via `_to_numpy`). For `png`/`jpg`, do
  **not** copy: hand the array straight to `get_image_codec().encode(array, ...)`.
- Encode `png`/`jpg` only via `get_image_codec()` — never hardcode Pillow. It
  auto-selects GPU nvImageCodec when the `[gpu]` extra and a CUDA device are
  present (several× faster, and zero-copy for GPU-resident Warp arrays) and
  falls back to Pillow otherwise. Passing the array through unchanged keeps the
  on-device fast path; recommend users install `[gpu]` on GPU hosts.

## References

- Backend protocol: `source/ovannotators/backends/_base.py`
- Built-in backends: `source/ovannotators/backends/` (`disk.py`, `s3.py`,
  `storage_api.py`)
- Registry + built-in registration: `source/ovannotators/backends/_registry.py`
- Companion skill for output formats: `skills/new-writer/SKILL.md`
- Keep skills, snippets, and `docs/public/api.rst` synchronized when the API
  changes.
