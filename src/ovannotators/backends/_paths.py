# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path confinement shared by remote backends."""

from __future__ import annotations

from pathlib import PureWindowsPath


def confine_relative(path: str, *, allow_empty: bool = False) -> str:
    """Normalize a caller path that must stay below a backend root.

    Drops ``.`` segments and keeps a single trailing ``/``. Remote stores do
    not resolve paths the way a filesystem does, so anything that could name a
    key outside the root is rejected rather than resolved.

    Args:
        path (str): Caller path, relative to the backend root.
        allow_empty (bool): Whether an empty path names the root itself.

    Returns:
        str: The normalized relative path.

    Raises:
        ValueError: If ``path`` is absolute (POSIX, drive-rooted, or UNC), contains a ``..`` or empty
            segment, or is empty when ``allow_empty`` is ``False``.
    """
    if path.startswith("/") or PureWindowsPath(path).is_absolute():
        raise ValueError(f"Path must be relative to the backend root: {path!r}")
    trailing = "/" if path.endswith("/") else ""
    body = path[:-1] if trailing else path
    segments = [] if not body else body.split("/")
    if ".." in segments:
        raise ValueError(f"Path must remain within the backend root: {path!r}")
    if "" in segments:
        raise ValueError(f"Path must not contain empty segments: {path!r}")
    kept = [segment for segment in segments if segment != "."]
    if not kept:
        if allow_empty:
            return ""
        raise ValueError(f"Path must name an object below the backend root: {path!r}")
    return "/".join(kept) + trailing
