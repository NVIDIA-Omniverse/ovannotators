# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the renderer-call watchdog in ``tests/_ovstage_session.py``.

The watchdog ends the process, so each case runs in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._ovstage_session import (
    RENDERER_TIMEOUT_ENV,
    renderer_deadline,
    renderer_timeout_s,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_under_deadline(body: str, timeout_s: str) -> subprocess.CompletedProcess:
    script = (
        "import time\n"
        "from tests._ovstage_session import renderer_deadline\n"
        "with renderer_deadline():\n"
        f"    {body}\n"
        "time.sleep(0.5)\n"
        "print('finished')\n"
    )
    env = {**os.environ, RENDERER_TIMEOUT_ENV: timeout_s}
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_stalled_call_exits_with_traceback() -> None:
    result = _run_under_deadline("time.sleep(30)", "0.2")

    assert result.returncode == 1
    assert "Timeout" in result.stderr
    assert "<string>" in result.stderr  # the stalled frame is in the dump
    assert "finished" not in result.stdout


def test_deadline_is_cancelled_when_the_call_returns() -> None:
    # The block returns well inside the deadline; the sleep after it outlasts
    # the deadline, so a watchdog left armed would end the process.
    result = _run_under_deadline("pass", "0.2")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "finished"


@pytest.mark.parametrize("raw", ["soon", "0", "-5", "nan", "inf", "-inf", "1e999"])
def test_invalid_timeout_names_the_variable(monkeypatch, raw) -> None:
    monkeypatch.setenv(RENDERER_TIMEOUT_ENV, raw)
    with pytest.raises(ValueError, match=RENDERER_TIMEOUT_ENV):
        renderer_timeout_s()


def test_nested_deadline_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(RENDERER_TIMEOUT_ENV, "60")
    with (
        renderer_deadline(),
        pytest.raises(RuntimeError, match="nest"),
        renderer_deadline(),
    ):
        pass
    with renderer_deadline():  # the outer block released the guard
        pass
