# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for version consistency."""

from __future__ import annotations

import importlib
import re

import pytest


class TestVersion:
    """Verify single-source version convention (PEP recommendation)."""

    def test_version_matches_metadata(self):
        """__version__ should match importlib.metadata.version('ovannotators')."""
        from importlib.metadata import version, PackageNotFoundError

        import ovannotators

        try:
            dist_version = version("ovannotators")
        except PackageNotFoundError:
            pytest.skip("ovannotators not installed as a distribution package")

        assert ovannotators.__version__ == dist_version

    def test_version_is_valid_semver(self):
        """Version string should be a valid semver-ish format."""
        import ovannotators

        # Matches: 0.1.0, 1.2.3, 0.1.0.dev0, 1.0.0rc1, etc.
        pattern = r"^\d+\.\d+\.\d+(\.\w+\d*)?$"
        assert re.match(pattern, ovannotators.__version__), (
            f"Invalid version format: {ovannotators.__version__}"
        )


def test_version_fallback_when_package_not_found(monkeypatch):
    """The sentinel fallback fires when the package isn't installed."""
    import importlib.metadata as ilm

    def _raise(_name):
        raise ilm.PackageNotFoundError("ovannotators")

    monkeypatch.setattr(ilm, "version", _raise)
    mod = importlib.reload(importlib.import_module("ovannotators"))
    try:
        assert mod.__version__ == "0.0.0+unknown"
    finally:
        # Restore the normally-resolved version for the rest of the session.
        monkeypatch.undo()
        importlib.reload(mod)
