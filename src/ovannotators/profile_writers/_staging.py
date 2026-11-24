# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private staging ownership for the native LeRobot writers."""

from __future__ import annotations

import os
import logging
import shutil
import stat
from itertools import chain
from pathlib import Path
from uuid import uuid4

from ovannotators._internal.filesystem import make_private_directory

_OWNER_FILE = ".ovannotators-stage-owner"
logger = logging.getLogger(__name__)


class NativeStage:
    """Reserve a private staging root and check its writer-owned entries."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._token = uuid4().hex.encode()
        self._identity: os.stat_result | None = None
        self._entries: dict[Path, bool] = {Path(_OWNER_FILE): False}
        self._written: dict[Path, os.stat_result] = {}

    def reserve(self) -> None:
        """Create the root exclusively and reject staging below another writer."""
        for parent in self.root.parents:
            if os.path.lexists(parent / _OWNER_FILE):
                raise ValueError("staging_dir is nested inside another writer's stage")
        try:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            make_private_directory(self.root)
        except FileExistsError as exc:
            raise ValueError("staging_dir must not already exist") from exc
        self._identity = self.root.lstat()
        with (self.root / _OWNER_FILE).open("xb") as token:
            token.write(self._token)
        self.verify()

    def path(self, relative: str | Path, *, directory: bool = False) -> Path:
        """Record a generated file or directory and return its local path."""
        relative = Path(relative)
        self._written.pop(relative, None)
        self._entries[relative] = directory
        for parent in relative.parents:
            if parent != Path("."):
                self._entries[parent] = True
        return self.root / relative

    def record_file(self, path: Path) -> None:
        """Bind a generated payload path to the file the writer created there."""
        self._written[path.relative_to(self.root)] = path.lstat()

    def adopt_tree(self, root: Path) -> None:
        """Record an output tree assembled in a private build directory."""
        for entry in chain((root,), root.rglob("*")):
            is_dir = entry.is_dir()
            self.path(entry.relative_to(self.root), directory=is_dir)
            if not is_dir:
                self.record_file(entry)

    def verify(self, *, complete: bool = False) -> None:
        """Reject foreign or aliased staging paths before writing or cleaning up."""
        info = self.root.lstat()
        if (
            self._identity is None
            or not stat.S_ISDIR(info.st_mode)
            or not os.path.samestat(info, self._identity)
        ):
            raise ValueError("writer staging directory was replaced")
        actual: set[Path] = set()
        for entry in self.root.rglob("*"):
            relative = entry.relative_to(self.root)
            details = entry.lstat()
            if (
                relative not in self._entries
                or getattr(details, "st_reparse_tag", 0)
                or not (
                    stat.S_ISDIR(details.st_mode)
                    if self._entries[relative]
                    else stat.S_ISREG(details.st_mode) and details.st_nlink == 1
                )
            ):
                raise ValueError(
                    f"foreign entry in writer staging directory: {relative}"
                )
            actual.add(relative)
            if match := self._written.get(relative):
                if (
                    not os.path.samestat(match, details)
                    or match.st_mtime_ns != details.st_mtime_ns
                ):
                    raise ValueError(
                        f"writer staging file was replaced or changed: {relative}"
                    )
        if (self.root / _OWNER_FILE).read_bytes() != self._token:
            raise ValueError("writer staging directory ownership changed")
        if complete and (missing := self._entries.keys() - actual):
            raise ValueError(
                f"writer staging files are missing: {sorted(map(str, missing))}"
            )

    def files(self, *, exclude: str | None = None) -> list[Path]:
        """Return the complete owned payload, optionally excluding one subtree."""
        self.verify(complete=True)
        return [
            relative
            for relative, is_dir in sorted(self._entries.items())
            if not is_dir
            and relative.name != _OWNER_FILE
            and relative.parts[0] != exclude
        ]

    def cleanup(self) -> None:
        """Remove the owned stage after confirming it still contains only writer entries."""
        try:
            self.verify()
            shutil.rmtree(self.root)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not remove writer staging directory %s: %s", self.root, exc
            )
