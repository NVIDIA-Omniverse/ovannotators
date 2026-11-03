@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
REM format_code.bat - Format + auto-fix lint with ruff via uv.
REM Syncs the CPU dev environment (ruff lives in `dev-cpu`), then applies
REM `ruff format` and `ruff check --fix` over src\ and tests\.
REM Pass `--check` to only verify without writing changes (CI-equivalent).
setlocal
cd /d "%~dp0"

echo ==^> uv sync (dev-cpu: includes ruff)
uv sync --extra dev-cpu
if errorlevel 1 exit /b 1

if "%~1"=="--check" (
  echo ==^> uv run ruff format --check src/ tests/
  uv run ruff format --check src/ tests/
  if errorlevel 1 exit /b 1
  echo ==^> uv run ruff check src/ tests/
  uv run ruff check src/ tests/
  if errorlevel 1 exit /b 1
) else (
  echo ==^> uv run ruff format src/ tests/
  uv run ruff format src/ tests/
  if errorlevel 1 exit /b 1
  echo ==^> uv run ruff check --fix src/ tests/
  uv run ruff check --fix src/ tests/
  if errorlevel 1 exit /b 1
)
endlocal
