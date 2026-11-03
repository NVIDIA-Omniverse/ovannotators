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

"""Backend Protocol — abstracts a write/read destination."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Backend(Protocol):
    """Abstracts a write/read destination (disk, S3, ovstorage, etc.)."""

    def write(self, path: str, data: bytes) -> None:
        """Write raw bytes to a destination path.

        Args:
            path (str): Destination path, relative to the backend root.
            data (bytes): Bytes to write.
        """
        ...

    def write_array(
        self,
        path: str,
        array: np.ndarray,
        encoding: str = "npy",
        *,
        exr_compression: str | None = None,
    ) -> None:
        """Serialize an array and write the encoded bytes.

        Args:
            path (str): Destination path, relative to the backend root.
            array (np.ndarray): Image/data array to encode (NumPy or Warp).
            encoding (str): Serialization format, one of ``"npy"``, ``"png"``,
                ``"jpg"``, or ``"exr"``.
            exr_compression (str | None): EXR compression scheme; ``None`` uses
                OpenEXR's default (ZIP). Only valid with ``encoding="exr"``.
        """
        ...  # pragma: no cover

    def read(self, path: str) -> bytes:
        """Read raw bytes from a source path.

        Args:
            path (str): Source path, relative to the backend root.

        Returns:
            bytes: The stored bytes.
        """
        ...

    def list(self, prefix: str = "") -> list[str]:
        """List stored paths under a prefix.

        Args:
            prefix (str): Path prefix to filter by; empty lists everything.

        Returns:
            list[str]: Matching paths, relative to the backend root.
        """
        ...

    def exists(self, path: str) -> bool:
        """Report whether a path exists.

        Args:
            path (str): Path to check, relative to the backend root.

        Returns:
            bool: ``True`` if the path exists.
        """
        ...

    def makedirs(self, path: str) -> None:
        """Ensure a directory exists (no-op for object stores).

        Args:
            path (str): Directory path, relative to the backend root.
        """
        ...
