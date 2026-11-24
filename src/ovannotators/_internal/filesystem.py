# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem security operations that Python does not expose uniformly."""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def make_private_directory(path: Path) -> None:
    """Exclusively create a directory accessible to its owner and the OS."""
    if os.name != "nt":
        path.mkdir(mode=0o700)
        try:
            path.chmod(0o700)
        except OSError:
            path.rmdir()
            raise
        return

    from ctypes import wintypes

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("descriptor", ctypes.c_void_p),
            ("inherit_handle", wintypes.BOOL),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    create = kernel.CreateDirectoryW
    create.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(_SecurityAttributes)]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    if not convert(
        "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)", 1, ctypes.byref(descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes = _SecurityAttributes(
            ctypes.sizeof(_SecurityAttributes), descriptor, False
        )
        if not create(str(path), ctypes.byref(attributes)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


@contextlib.contextmanager
def private_work_directory(parent: Path) -> Iterator[Path]:
    """Create a fresh private work directory on the same filesystem.

    Removal on exit is best-effort: a cleanup failure is logged rather than
    raised, so it cannot turn a completed build into a failure that a retry
    cannot recover from.
    """
    path = parent / f".ovannotators-work-{uuid4().hex}"
    make_private_directory(path)
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning("Could not remove work directory %s: %s", path, exc)
