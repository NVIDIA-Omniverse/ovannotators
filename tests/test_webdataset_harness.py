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

"""Check that the WebDataset benchmark harness fails fast when a phase dies."""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from tests.benchmarks import webdataset_harness


def _crash(queue: Any, name: str, kwargs: dict[str, Any]) -> None:
    os._exit(3)  # die without reporting, as an out-of-memory kill would


def _hang(queue: Any, name: str, kwargs: dict[str, Any]) -> None:
    time.sleep(600)


def _report(queue: Any, name: str, kwargs: dict[str, Any]) -> None:
    queue.put({"samples": 1})


@pytest.mark.integration
@pytest.mark.cpu_only
class TestRunIsolated:
    def test_crashed_phase_raises_instead_of_hanging(self, monkeypatch):
        monkeypatch.setattr(webdataset_harness, "_phase", _crash)
        with pytest.raises(RuntimeError, match="exited with code 3"):
            webdataset_harness.run_isolated("read_loose", timeout_s=60)

    def test_phase_exception_raises(self):
        # An unknown phase name raises inside the child before any result.
        with pytest.raises(RuntimeError, match="before reporting a result"):
            webdataset_harness.run_isolated("package", timeout_s=60)

    def test_silent_phase_times_out(self, monkeypatch):
        monkeypatch.setattr(webdataset_harness, "_phase", _hang)
        with pytest.raises(RuntimeError, match="did not report within"):
            webdataset_harness.run_isolated("read_loose", timeout_s=1)

    def test_reported_result_is_returned(self, monkeypatch):
        monkeypatch.setattr(webdataset_harness, "_phase", _report)
        assert webdataset_harness.run_isolated("read_loose") == {"samples": 1}
