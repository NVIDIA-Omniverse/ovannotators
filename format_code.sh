#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# format_code.sh - Format + auto-fix lint with ruff via uv.
#
# Syncs the CPU dev environment (ruff lives in the `dev-cpu` extra), then
# applies `ruff format` and `ruff check --fix` over src/ and tests/.
# Pass `--check` to only verify without writing changes (CI-equivalent).
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> uv sync (dev-cpu: includes ruff)"
uv sync --extra dev-cpu

if [ "${1:-}" = "--check" ]; then
  echo "==> uv run ruff format --check src/ tests/"
  uv run ruff format --check src/ tests/
  echo "==> uv run ruff check src/ tests/"
  uv run ruff check src/ tests/
else
  echo "==> uv run ruff format src/ tests/"
  uv run ruff format src/ tests/
  echo "==> uv run ruff check --fix src/ tests/"
  uv run ruff check --fix src/ tests/
fi
