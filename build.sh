#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# build.sh - Build the ovannotators sdist + wheel locally with uv.
#
# Syncs the CPU dev environment (build tooling + CPU feature set, no GPU
# required) and produces dist/ artifacts. Extra args are forwarded to
# `uv build` (e.g. `./build.sh --sdist`).
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> uv sync (dev-cpu: build + lint + test tooling, no GPU)"
uv sync --extra dev-cpu

echo "==> uv build"
uv build "$@"

echo "==> Artifacts:"
ls -la dist/
