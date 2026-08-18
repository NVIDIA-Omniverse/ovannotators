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

"""Tests for ovannotators/_internal/optional_deps.py availability probes.

These exercise the *success* returns of the probes on a GPU-less host by
injecting fake modules into ``sys.modules``. ``nvimgcodec_available`` resolves
through the normal import machinery (so a ``sys.modules`` entry is enough),
while ``ovrtx_available`` is additionally shielded from the test harness's
import shim (which hard-blocks ``import ovrtx``) by routing that one name
through a fake ``__import__`` — mirroring the real-CI condition where an
installed ``ovrtx`` would import successfully.
"""

from __future__ import annotations

import builtins
import sys
import types
from unittest.mock import patch

from ovannotators._internal import optional_deps


def test_ovrtx_available_true_when_renderer_present():
    """A real ``ovrtx`` exposing ``Renderer`` reports available (success return)."""
    fake_ovrtx = types.ModuleType("ovrtx")
    fake_ovrtx.Renderer = object

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ovrtx" or name.startswith("ovrtx."):
            return sys.modules["ovrtx"]
        return real_import(name, *args, **kwargs)

    optional_deps.ovrtx_available.cache_clear()
    try:
        with (
            patch.dict(sys.modules, {"ovrtx": fake_ovrtx}),
            patch("builtins.__import__", fake_import),
        ):
            assert optional_deps.ovrtx_available() is True
    finally:
        optional_deps.ovrtx_available.cache_clear()


def test_ovrtx_available_false_without_renderer():
    """An importable stub lacking ``Renderer`` is treated as unavailable."""
    fake_ovrtx = types.ModuleType("ovrtx")  # no Renderer attribute

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ovrtx" or name.startswith("ovrtx."):
            return sys.modules["ovrtx"]
        return real_import(name, *args, **kwargs)

    optional_deps.ovrtx_available.cache_clear()
    try:
        with (
            patch.dict(sys.modules, {"ovrtx": fake_ovrtx}),
            patch("builtins.__import__", fake_import),
        ):
            assert optional_deps.ovrtx_available() is False
    finally:
        optional_deps.ovrtx_available.cache_clear()


def test_nvimgcodec_available_true_when_importable():
    """An importable ``nvidia.nvimgcodec`` reports available (success return)."""
    fake_nvimgcodec = types.ModuleType("nvidia.nvimgcodec")
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.nvimgcodec = fake_nvimgcodec

    optional_deps.nvimgcodec_available.cache_clear()
    try:
        with patch.dict(
            sys.modules,
            {"nvidia": fake_nvidia, "nvidia.nvimgcodec": fake_nvimgcodec},
        ):
            assert optional_deps.nvimgcodec_available() is True
    finally:
        optional_deps.nvimgcodec_available.cache_clear()
