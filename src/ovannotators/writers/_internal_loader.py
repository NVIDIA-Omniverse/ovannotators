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

"""Discovery for writers that ship only in the development repository.

Writers under the dot-prefixed internal directory are ordinary writer modules
that must not reach the public mirror. The leading dot is what enforces that:
the public-source export drops any path with an ``.internal`` component at any
depth, and the same dot keeps the directory out of the wheel build. The cost is
that ``.internal`` is not a Python identifier, so the directory cannot be an
ordinary subpackage and is instead bound to the importable alias
``ovannotators.writers.internal`` by :func:`load_internal_writers`.

In a public checkout the directory is simply absent and discovery is a no-op,
so nothing here is conditional on which repository the code came from.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

logger = logging.getLogger(__name__)

#: Directory holding internal writer modules, relative to this package.
INTERNAL_DIRNAME = ".internal"

#: Importable alias bound to :data:`INTERNAL_DIRNAME`. Internal writers are
#: reachable both by registry name and as
#: ``ovannotators.writers.internal.<module>``.
INTERNAL_PACKAGE = "ovannotators.writers.internal"


def internal_dir() -> Path:
    """Return the path of the internal writer directory.

    Returns:
        Path: ``<writers package>/.internal``. The directory need not exist; a
        public checkout has none.
    """
    return Path(__file__).resolve().parent / INTERNAL_DIRNAME


def _bind_alias_package(directory: Path) -> None:
    """Bind ``directory`` to the :data:`INTERNAL_PACKAGE` module name.

    Registers a synthetic namespace package whose search path is ``directory``,
    so the ordinary import machinery resolves ``INTERNAL_PACKAGE.<module>``.
    Without this, a dot-prefixed directory would only be reachable through
    per-file :func:`importlib.util.spec_from_file_location` calls, and its
    modules could not import one another.

    Args:
        directory (Path): The ``.internal`` directory to expose.
    """
    if INTERNAL_PACKAGE in sys.modules:
        return
    spec = ModuleSpec(INTERNAL_PACKAGE, None, is_package=True)
    spec.submodule_search_locations = [str(directory)]  # type: ignore[assignment]
    module = importlib.util.module_from_spec(spec)
    module.__doc__ = "Writers that ship only in the development repository."
    sys.modules[INTERNAL_PACKAGE] = module

    parent_name, _, attr = INTERNAL_PACKAGE.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attr, module)


def load_internal_writers() -> list[str]:
    """Import every internal writer module so it self-registers.

    Each ``*.py`` in the internal directory is imported for its
    ``@WriterRegistry.register`` side effect. Modules whose names start with an
    underscore are treated as private helpers and skipped, matching the
    convention in the public writers package.

    A module that raises on import is logged and skipped rather than being
    allowed to break ``import ovannotators``: an internal writer with a missing
    optional dependency must not take the library down with it.

    Returns:
        list[str]: Fully qualified names of the modules imported successfully,
        sorted. Empty when the directory is absent, which is the normal state of
        a public checkout.
    """
    directory = internal_dir()
    if not directory.is_dir():
        return []

    _bind_alias_package(directory)

    loaded: list[str] = []
    for path in sorted(directory.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module_name = f"{INTERNAL_PACKAGE}.{path.stem}"
        try:
            importlib.import_module(module_name)
        except Exception:
            logger.warning("Skipping internal writer %s", path.name, exc_info=True)
            continue
        loaded.append(module_name)
    return loaded
