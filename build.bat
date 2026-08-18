@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
REM build.bat - Build the ovannotators sdist + wheel locally with uv.
REM Syncs the CPU dev environment (build tooling + CPU feature set) and
REM produces dist\ artifacts. Extra args are forwarded to `uv build`.
setlocal
cd /d "%~dp0"

echo ==^> uv sync (dev-cpu: build + lint + test tooling, no GPU)
uv sync --extra dev-cpu
if errorlevel 1 exit /b 1

echo ==^> uv build
uv build %*
if errorlevel 1 exit /b 1

echo ==^> Artifacts:
dir dist
endlocal
