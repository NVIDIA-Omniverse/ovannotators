@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
REM test.bat - Run the FULL test suite locally with uv, including GPU / ovrtx
REM full-pipeline tests (skipped on CI via `-m "not gpu"`).
REM Requires an NVIDIA GPU + CUDA at runtime. Syncs the `dev` extra
REM (CPU dev tooling + GPU codecs + video + ovrtx + test deps).
REM
REM Usage:
REM   test.bat                       full suite incl. GPU/ovrtx
REM   test.bat tests\test_kitti.py   narrow to a path
REM   test.bat -m "not gpu"          CI-equivalent (skip GPU tests)
setlocal
cd /d "%~dp0"

echo ==^> uv sync (dev: GPU runtime + ovrtx + video + test tooling)
uv sync --extra dev
if errorlevel 1 exit /b 1

REM Make PyNvVideoCodec shared libs discoverable for the LeRobot video tests.
for /f "delims=" %%i in ('uv run python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"') do set "VENV_SITE=%%i"
if exist "%VENV_SITE%\pynvvideocodec" set "PATH=%VENV_SITE%\pynvvideocodec;%PATH%"

if "%~1"=="" (
  echo ==^> uv run pytest -m "" ^(full suite incl. GPU/ovrtx^)
  uv run pytest -m ""
) else (
  echo ==^> uv run pytest %*
  uv run pytest %*
)
if errorlevel 1 exit /b 1
endlocal
