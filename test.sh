#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# test.sh - Run the FULL test suite locally with uv, including the GPU /
# ovrtx full-pipeline tests (the ones skipped on CI via `-m 'not gpu'`).
#
# Requires an NVIDIA GPU + CUDA at runtime. Syncs the full dev environment
# (`dev` extra = CPU dev tooling + GPU codecs + video + ovrtx + test deps).
#
# By default this overrides the project's default `-m 'not gpu'` marker so
# that GPU tests ARE collected and run. Pass your own args to narrow it, e.g.
#   ./test.sh tests/test_full_pipeline.py
#   ./test.sh -k kitti
#   ./test.sh -m 'not gpu'        # CI-equivalent (skip GPU tests)
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> uv sync (dev: GPU runtime + ovrtx + video + test tooling)"
uv sync --extra dev

# PyNvVideoCodec ships shared libs inside its site-packages dir; the LeRobot
# video tests need them on the loader path. Resolve the active uv venv site dir.
VENV_SITE="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NVVC_DIR="$VENV_SITE/pynvvideocodec"
if [ -d "$NVVC_DIR" ]; then
  export LD_LIBRARY_PATH="$NVVC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Run the full suite. If the caller passes args, use them verbatim; otherwise
# override the default `-m 'not gpu'` so GPU/ovrtx full-pipeline tests run too.
if [ "$#" -gt 0 ]; then
  echo "==> uv run pytest $*"
  uv run pytest "$@"
else
  echo "==> uv run pytest -m '' (full suite incl. GPU/ovrtx)"
  uv run pytest -m ''
fi
