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

"""Tested snippets for the ``new-backend`` skill.

Every illustrative region is wrapped in ``# [snippet:doc-...]`` markers and is
referenced from ``skills/new-backend/SKILL.md`` via ``> **Source:**`` directives.
These tests are the single source of truth for the skill's code. Runs CPU-only,
so it joins the default (non-GPU) test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

# [snippet:doc-backend-config]
import io
from dataclasses import dataclass, field
from typing import ClassVar

from ovannotators.backends._registry import BackendRegistry
from ovannotators.exceptions import BackendReadError, BackendWriteError


@dataclass
class MemoryBackendConfig:
    """Configuration for :class:`MemoryBackend`.

    Args:
        overwrite: When ``False`` (default), writing an existing path raises
            :class:`~ovannotators.exceptions.BackendWriteError`.
        store: Optional shared in-memory store; a fresh dict is used when omitted.
    """

    overwrite: bool = False
    store: dict[str, bytes] = field(default_factory=dict)


# [/snippet:doc-backend-config]


# [snippet:doc-backend-class]
@BackendRegistry.register
class MemoryBackend:
    """Example backend: an in-memory key/value store implementing ``Backend``.

    Mirrors :class:`~ovannotators.backends.disk.DiskBackend` semantics (relative
    paths, overwrite guard, ``write_array`` encoding dispatch) without touching
    the filesystem — useful as a template for object stores or HTTP APIs.
    """

    name: ClassVar[str] = "memory"

    def __init__(self, config: MemoryBackendConfig | None = None) -> None:
        self._cfg = config or MemoryBackendConfig()
        self._store = self._cfg.store

    def write(self, path: str, data: bytes) -> None:
        if path in self._store and not self._cfg.overwrite:
            raise BackendWriteError(
                path=path, message=f"Path exists and overwrite=False: {path}"
            )
        self._store[path] = bytes(data)

    def write_array(self, path: str, array: np.ndarray, encoding: str = "npy") -> None:
        if encoding == "npy":
            buf = io.BytesIO()
            np.save(buf, np.asarray(array))
            self.write(path, buf.getvalue())
        elif encoding in ("png", "jpg"):
            from ovannotators._internal.codecs import get_image_codec

            self.write(path, get_image_codec().encode(array, format=encoding))
        else:
            raise BackendWriteError(
                path=path,
                message=f"Unsupported encoding '{encoding}'. Use 'npy', 'png', or 'jpg'.",
            )

    def read(self, path: str) -> bytes:
        try:
            return self._store[path]
        except KeyError as exc:
            raise BackendReadError(path=path, message="No such path") from exc

    def list(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self._store if p.startswith(prefix))

    def exists(self, path: str) -> bool:
        return path in self._store

    def makedirs(self, path: str) -> None:
        # No directory concept in a flat key/value store (as for object stores).
        pass


# [/snippet:doc-backend-class]


def test_memory_backend_roundtrip():
    """Bytes and arrays round-trip; list/exists behave as documented."""
    # [snippet:doc-backend-register]
    # Registry by name (config-driven), or construct the class directly.
    backend = BackendRegistry.create("memory", MemoryBackendConfig(overwrite=True))
    # [/snippet:doc-backend-register]

    # [snippet:doc-backend-test]
    backend.write("meta/info.json", b'{"v": 1}')
    assert backend.read("meta/info.json") == b'{"v": 1}'

    backend.write_array("frames/0000.npy", np.arange(6).reshape(2, 3))
    restored = np.load(io.BytesIO(backend.read("frames/0000.npy")))
    assert restored.tolist() == [[0, 1, 2], [3, 4, 5]]

    assert backend.exists("meta/info.json")
    assert backend.list("frames/") == ["frames/0000.npy"]
    # [/snippet:doc-backend-test]


def test_memory_backend_overwrite_guard():
    """Re-writing an existing path raises unless overwrite=True."""
    backend = MemoryBackend(MemoryBackendConfig(overwrite=False))
    backend.write("a.bin", b"first")
    with pytest.raises(BackendWriteError):
        backend.write("a.bin", b"second")


def test_memory_backend_read_missing_raises():
    """Reading an absent path raises BackendReadError."""
    backend = MemoryBackend()
    with pytest.raises(BackendReadError):
        backend.read("nope.bin")


def test_memory_backend_rejects_unknown_encoding():
    """An unsupported write_array encoding raises BackendWriteError."""
    backend = MemoryBackend(MemoryBackendConfig(overwrite=True))
    with pytest.raises(BackendWriteError):
        backend.write_array("x.tiff", np.zeros((2, 2), np.uint8), encoding="tiff")
