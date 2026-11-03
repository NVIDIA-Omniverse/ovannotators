<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Documentation

Use this page to choose the shortest path to the information you need.

These pages are built with Sphinx (MyST). On GitHub, cross-reference roles such
as `{ref}` and the `{toctree}` block in the writer guides appear as plain text;
the built documentation renders them as links.

## Start using ovannotators

- [Getting started](public/index.rst#installation) explains installation,
  extras, and the first renderer-backed workflow.
- [Overview and architecture](public/index.rst#what-is-ovannotators) explains
  the renderer-output → writer → backend model.
- [Example index](../examples/README.md) provides runnable commands and expected
  results.

## Build with ovannotators

- [Writer formats](public/writers/index.md) explain each documented writer's purpose, input
  requirements, output schema, configuration, and upstream references.
- [Python API reference](public/api.rst) is the public programming contract.
- [Renderer output reference](public/render_outputs.rst) lists supported render
  variables and extraction behavior.
- [Tutorials](public/tutorials.rst) cover common writers, storage backends,
  custom extensions, and CPU-only input.
- [Agent skills](../skills/README.md) provide task-specific implementation and
  validation workflows for coding agents.

## Solve a problem

- [Troubleshooting](public/troubleshooting.rst) covers installation, optional
  dependencies, GPU runtime setup, render-variable mismatches, and output
  verification.
- [Support](../README.md#support) explains where to ask questions and report
  security issues.
