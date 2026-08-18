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

"""DiskBackend — writes and reads files on the local filesystem."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Union

import numpy as np

from ovannotators.exceptions import BackendReadError, BackendWriteError


@dataclass
class DiskBackendConfig:
    """Configuration for :class:`DiskBackend`.

    Attributes:
        root_dir (Union[str, Path]): Directory that all relative paths resolve
            against; created on demand when writing.
        overwrite (bool): If ``False`` (default), writing to an existing path
            raises :class:`~ovannotators.exceptions.BackendWriteError` instead
            of overwriting it.
    """

    root_dir: Union[str, Path]
    overwrite: bool = False


class DiskBackend:
    """Filesystem backend. All paths are relative to root_dir.

    Parent directories are created automatically on write.
    """

    name: ClassVar[str] = "disk"

    def __init__(self, config: DiskBackendConfig) -> None:
        """Initialize the backend from its configuration.

        Args:
            config (DiskBackendConfig): Root directory and write options.
        """
        self._root = Path(config.root_dir)
        self._overwrite = config.overwrite

    def _abs(self, path: str) -> Path:
        return self._root / path

    def write(self, path: str, data: bytes) -> None:
        """Write raw bytes to ``path`` under root_dir.

        Parent directories are created as needed.

        Args:
            path (str): Destination path, relative to root_dir.
            data (bytes): Bytes to write.

        Raises:
            BackendWriteError: If the path exists and ``overwrite`` is ``False``,
                or the write fails.
        """
        dest = self._abs(path)
        if dest.exists() and not self._overwrite:
            raise BackendWriteError(
                path=path,
                message=f"Path already exists and overwrite=False: {dest}",
            )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc

    def write_array(self, path: str, array: np.ndarray, encoding: str = "npy") -> None:
        """Serialize an array and write it to ``path`` under root_dir.

        PNG/JPEG encoding uses the shared image codec
        (:func:`ovannotators._internal.codecs.get_image_codec`), which prefers
        nvImageCodec on the GPU (keeping GPU-resident Warp arrays on-device for
        zero-copy encoding) and falls back to Pillow. Other encodings copy to
        host NumPy.

        Args:
            path (str): Destination path, relative to root_dir.
            array (np.ndarray): Image/data array to encode (NumPy or Warp).
            encoding (str): One of ``"npy"``, ``"png"``, ``"jpg"``, or ``"exr"``.

        Raises:
            BackendWriteError: If the encoding is unsupported or the write fails.
        """
        from ovannotators._internal.codecs import _to_numpy

        buf = io.BytesIO()
        if encoding == "npy":
            np.save(buf, _to_numpy(array))
            self.write(path, buf.getvalue())
        elif encoding in ("png", "jpg"):
            from ovannotators._internal.codecs import get_image_codec

            buf.write(get_image_codec().encode(array, format=encoding))
            self.write(path, buf.getvalue())
        elif encoding == "exr":
            try:
                from ovannotators._internal.codecs import _encode_exr_bytes

                self.write(path, _encode_exr_bytes(array))
            except ImportError:
                raise BackendWriteError(
                    path=path,
                    message="Cannot encode 'exr': install OpenEXR",
                ) from None
        else:
            raise BackendWriteError(
                path=path,
                message=f"Unsupported encoding '{encoding}'. Use 'npy', 'png', 'jpg', or 'exr'.",
            )

    def read(self, path: str) -> bytes:
        """Read raw bytes from ``path`` under root_dir.

        Args:
            path (str): Source path, relative to root_dir.

        Returns:
            bytes: The file contents.

        Raises:
            BackendReadError: If the file cannot be read.
        """
        src = self._abs(path)
        try:
            return src.read_bytes()
        except OSError as exc:
            raise BackendReadError(path=path, message=str(exc)) from exc

    def list(self, prefix: str = "") -> list[str]:
        """List file paths under ``prefix``, relative to root_dir.

        Args:
            prefix (str): Subdirectory prefix to search; empty walks root_dir.

        Returns:
            list[str]: Sorted file paths relative to root_dir; empty if the
            prefix does not exist.
        """
        base = self._abs(prefix) if prefix else self._root
        if not base.exists():
            return []
        results: list[str] = []
        for item in base.rglob("*"):
            if item.is_file():
                results.append(item.relative_to(self._root).as_posix())
        return sorted(results)

    def exists(self, path: str) -> bool:
        """Report whether ``path`` exists under root_dir.

        Args:
            path (str): Path to check, relative to root_dir.

        Returns:
            bool: ``True`` if the path exists.
        """
        return self._abs(path).exists()

    def makedirs(self, path: str) -> None:
        """Create a directory (and parents) under root_dir.

        Args:
            path (str): Directory path, relative to root_dir.

        Raises:
            BackendWriteError: If the directory cannot be created.
        """
        try:
            self._abs(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc
