# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in write/read smoke test against a configured remote ovstorage Stack."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from ovannotators.backends.ovstorage import OvstorageBackend, OvstorageBackendConfig

ovstorage = pytest.importorskip("ovstorage")


@pytest.mark.network
@pytest.mark.integration
def test_remote_write_read_list():
    """Write under an explicitly supplied remote root and verify the bytes."""
    config_path = os.environ.get("OVANNOTATORS_OVSTORAGE_SMOKE_CONFIG")
    root_uri = os.environ.get("OVANNOTATORS_OVSTORAGE_SMOKE_ROOT")
    if not config_path or not root_uri:
        pytest.skip("Set OVANNOTATORS_OVSTORAGE_SMOKE_CONFIG and _ROOT to opt in")

    stack = asyncio.run(
        ovstorage.Stack.from_config(path=config_path)
        .with_registry(ovstorage.PluginRegistry([str(ovstorage.bundled_plugins_dir())]))
        .build()
    )
    backend = OvstorageBackend(OvstorageBackendConfig(root_uri=root_uri), stack)
    unique_path = f"ovannotators-smoke/{uuid4().hex}/probe.bin"
    payload = f"ovannotators:{uuid4().hex}".encode()
    try:
        backend.makedirs(unique_path.rsplit("/", 1)[0])
        backend.write(unique_path, payload)
        assert backend.read(unique_path) == payload
        assert backend.exists(unique_path)
        assert unique_path in backend.list("ovannotators-smoke")
        print(f"Verified remote object: {backend._address(unique_path)}")
    finally:
        backend.close()
