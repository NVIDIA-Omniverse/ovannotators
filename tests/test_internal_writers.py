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

"""Discovery of writers that ship in the development repository only.

The mechanism, not the writers: these build a throwaway ``.internal`` directory
so the assertions hold in a checkout that has none. That is the property worth
pinning down — a public checkout must import and behave identically to one that
carries internal writers.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from ovannotators.writers import _internal_loader
from ovannotators.writers._internal_loader import (
    INTERNAL_PACKAGE,
    load_internal_writers,
)
from ovannotators.writers._registry import WriterRegistry

_WRITER_MODULE = """
from ovannotators.writers._registry import WriterRegistry


@WriterRegistry.register
class {cls}:
    name = "{name}"
    required_render_vars = frozenset()

    def write_frame(self, frame_id, render_vars, backend):
        pass

    def finalize(self, backend):
        pass
"""


@pytest.fixture
def internal_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the loader at an empty throwaway internal directory.

    Also unbinds the alias package and restores the writer registry, so a test
    that registers a writer cannot leak it into the rest of the session.
    """
    registered = dict(WriterRegistry._registry)
    modules = dict(sys.modules)

    directory = tmp_path / ".internal"
    directory.mkdir()
    monkeypatch.setattr(_internal_loader, "internal_dir", lambda: directory)
    sys.modules.pop(INTERNAL_PACKAGE, None)

    yield directory

    WriterRegistry._registry.clear()
    WriterRegistry._registry.update(registered)
    for name in set(sys.modules) - set(modules):
        del sys.modules[name]
    sys.modules.pop(INTERNAL_PACKAGE, None)


def _write_module(directory: Path, stem: str, body: str) -> None:
    (directory / f"{stem}.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_absent_directory_loads_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """The public tree has no internal directory, and must not notice."""
    monkeypatch.setattr(_internal_loader, "internal_dir", lambda: tmp_path / "nope")
    monkeypatch.delitem(sys.modules, INTERNAL_PACKAGE, raising=False)

    assert load_internal_writers() == []
    assert INTERNAL_PACKAGE not in sys.modules


def test_empty_directory_loads_nothing(internal_dir: Path):
    assert load_internal_writers() == []


def test_module_is_imported_and_registers_its_writer(internal_dir: Path):
    _write_module(
        internal_dir, "demo", _WRITER_MODULE.format(cls="DemoWriter", name="demo")
    )

    assert load_internal_writers() == [f"{INTERNAL_PACKAGE}.demo"]
    assert WriterRegistry.get("demo").name == "demo"


def test_writer_is_reachable_by_import_as_well_as_by_name(internal_dir: Path):
    """Both retrieval paths work, so internal writers are not registry-only."""
    _write_module(
        internal_dir, "demo", _WRITER_MODULE.format(cls="DemoWriter", name="demo")
    )
    load_internal_writers()

    import importlib

    module = importlib.import_module(f"{INTERNAL_PACKAGE}.demo")

    assert module.DemoWriter is WriterRegistry.get("demo")


def test_internal_modules_can_import_each_other(internal_dir: Path):
    """The alias package is what makes a shared internal helper possible."""
    _write_module(internal_dir, "_shared", "VALUE = 7\n")
    _write_module(
        internal_dir,
        "demo",
        f"""
        from {INTERNAL_PACKAGE}._shared import VALUE

        RESULT = VALUE
        """,
    )
    load_internal_writers()

    assert sys.modules[f"{INTERNAL_PACKAGE}.demo"].RESULT == 7


def test_underscore_modules_are_treated_as_helpers(internal_dir: Path):
    _write_module(
        internal_dir,
        "_helper",
        _WRITER_MODULE.format(cls="HelperWriter", name="helper"),
    )

    assert load_internal_writers() == []
    assert "helper" not in WriterRegistry.list_names()


def test_a_broken_module_is_skipped_rather_than_fatal(
    internal_dir: Path, caplog: pytest.LogCaptureFixture
):
    """An internal writer missing an optional dependency must not break imports."""
    _write_module(internal_dir, "broken", "raise ImportError('no ovphysx here')\n")
    _write_module(
        internal_dir, "demo", _WRITER_MODULE.format(cls="DemoWriter", name="demo")
    )

    with caplog.at_level("WARNING"):
        loaded = load_internal_writers()

    assert loaded == [f"{INTERNAL_PACKAGE}.demo"]
    assert "broken.py" in caplog.text


def test_loading_twice_is_idempotent(internal_dir: Path):
    """``import ovannotators`` may run more than once per interpreter."""
    _write_module(
        internal_dir, "demo", _WRITER_MODULE.format(cls="DemoWriter", name="demo")
    )

    first = load_internal_writers()

    assert load_internal_writers() == first
