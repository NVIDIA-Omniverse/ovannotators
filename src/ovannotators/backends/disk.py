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

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Union

import numpy as np

from ovannotators.backends._encoding import encode_array
from ovannotators.exceptions import BackendReadError, BackendWriteError


@dataclass
class DiskBackendConfig:
    """Configuration for :class:`DiskBackend`.

    Attributes:
        root_dir (Union[str, Path]): Directory that relative paths resolve
            against; may be absolute and is created on demand when writing.
            A relative ``root_dir`` is resolved against the current working
            directory once, when :class:`DiskBackend` is constructed; later
            changes to the working directory do not move the root.
        overwrite (bool): If ``False`` (default), writing to an existing path
            raises :class:`~ovannotators.exceptions.BackendWriteError` instead
            of overwriting it.
    """

    root_dir: Union[str, Path]
    overwrite: bool = False


class DiskBackend:
    """Filesystem backend. Caller paths must resolve within ``root_dir``.

    Caller paths may be relative or absolute. Parent directories are created
    automatically on write.

    ``root_dir`` is resolved once at construction. Every caller path is then
    resolved, with symlinks followed, and must stay inside ``root_dir``:
    ``..`` traversal, absolute paths outside the root, and symlinks (file or
    directory) that point outside the root are rejected, and ``list()`` omits
    them. Symlinks that point to another location inside the root are allowed.
    Paths that cannot be resolved, such as a symlink loop, are rejected too.
    The check runs at call time; it does not protect against another process
    changing the filesystem concurrently.
    """

    name: ClassVar[str] = "disk"

    def __init__(self, config: DiskBackendConfig) -> None:
        """Initialize the backend from its configuration.

        Args:
            config (DiskBackendConfig): Root directory and write options.
        """
        self._root = Path(config.root_dir).resolve()
        self._overwrite = config.overwrite

    def _abs(self, path: str) -> Path:
        """Resolve a caller path and require it to remain beneath ``_root``."""
        try:
            resolved = (self._root / path).resolve()
        except (OSError, RuntimeError) as exc:
            # Symlink loops raise RuntimeError on older Pythons and OSError on
            # newer ones; either way the path cannot be resolved safely.
            raise ValueError(
                f"Path cannot be resolved within backend root_dir: {path!r}"
            ) from exc
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(
                f"Path must remain within backend root_dir: {path!r}"
            ) from exc
        return resolved

    def write(self, path: str, data: bytes) -> None:
        """Write raw bytes to ``path`` under root_dir.

        Parent directories are created as needed.

        Args:
            path (str): Destination path inside root_dir; may be absolute.
            data (bytes): Bytes to write.

        Raises:
            BackendWriteError: If the path resolves outside root_dir (including
                through a symlink), the path exists and ``overwrite`` is
                ``False``, or the write fails.
        """
        try:
            dest = self._abs(path)
        except ValueError as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc
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

    def write_array(
        self,
        path: str,
        array: np.ndarray,
        encoding: str = "npy",
        *,
        exr_compression: str | None = None,
    ) -> None:
        """Serialize an array and write it to ``path`` under root_dir.

        PNG/JPEG encoding uses the shared image codec
        (:func:`ovannotators._internal.codecs.get_image_codec`), which prefers
        nvImageCodec on the GPU (keeping GPU-resident Warp arrays on-device for
        zero-copy encoding) and falls back to Pillow. Other encodings copy to
        host NumPy.

        Args:
            path (str): Destination path inside root_dir; may be absolute.
            array (np.ndarray): Image/data array to encode (NumPy or Warp).
            encoding (str): One of ``"npy"``, ``"png"``, ``"jpg"``, or ``"exr"``.
            exr_compression (str | None): EXR compression scheme, or ``None``
                for OpenEXR's default ZIP. Only valid for EXR.

        Raises:
            BackendWriteError: If the path resolves outside root_dir (including
                through a symlink), the encoding is unsupported, or the write
                fails.
            ValueError: If ``exr_compression`` is set and ``encoding`` is not
                ``"exr"``, or if the EXR compression scheme is not supported.
        """
        self.write(
            path, encode_array(path, array, encoding, exr_compression=exr_compression)
        )

    def read(self, path: str) -> bytes:
        """Read raw bytes from ``path`` under root_dir.

        Args:
            path (str): Source path inside root_dir; may be absolute.

        Returns:
            bytes: The file contents.

        Raises:
            BackendReadError: If the path resolves outside root_dir (including
                through a symlink) or the file cannot be read.
        """
        try:
            src = self._abs(path)
        except ValueError as exc:
            raise BackendReadError(path=path, message=str(exc)) from exc
        try:
            return src.read_bytes()
        except OSError as exc:
            raise BackendReadError(path=path, message=str(exc)) from exc

    def list(self, prefix: str = "") -> list[str]:
        """List file paths under ``prefix`` inside root_dir.

        Args:
            prefix (str): Subdirectory to search, relative or absolute within
                root_dir; empty walks root_dir.

        Returns:
            list[str]: Sorted file paths relative to root_dir; empty if the
            prefix does not exist. Symlinks whose targets are outside root_dir,
            and broken or looping symlinks, are omitted.

        Raises:
            BackendReadError: If ``prefix`` resolves outside root_dir (including
                through a symlink) or cannot be resolved.
        """
        try:
            base = self._abs(prefix) if prefix else self._root
        except ValueError as exc:
            raise BackendReadError(path=prefix, message=str(exc)) from exc
        if not base.exists():
            return []
        results: list[str] = []
        for item in base.rglob("*"):
            try:
                if not item.is_file():
                    continue
                # Match read(): skip links whose target is outside root_dir.
                item.resolve().relative_to(self._root)
            except (OSError, RuntimeError, ValueError):
                continue
            results.append(item.relative_to(self._root).as_posix())
        return sorted(results)

    def exists(self, path: str) -> bool:
        """Report whether ``path`` exists under root_dir.

        Args:
            path (str): Path to check inside root_dir; may be absolute.

        Returns:
            bool: ``True`` if the path exists inside root_dir; ``False`` if it
            does not exist or resolves outside root_dir.
        """
        try:
            return self._abs(path).exists()
        except ValueError:
            return False

    def makedirs(self, path: str) -> None:
        """Create a directory (and parents) under root_dir.

        Args:
            path (str): Directory path inside root_dir; may be absolute.

        Raises:
            BackendWriteError: If the path resolves outside root_dir (including
                through a symlink) or the directory cannot be created.
        """
        try:
            self._abs(path).mkdir(parents=True, exist_ok=True)
        except ValueError as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc
        except OSError as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc
