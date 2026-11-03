# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the ovstorage adapter against the real local-file Stack."""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest

from ovannotators.backends.ovstorage import OvstorageBackend, OvstorageBackendConfig
from ovannotators.exceptions import BackendReadError, BackendWriteError

ovstorage = pytest.importorskip("ovstorage")


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    request = ovstorage.ConnectionRequest("file")
    request.add_config("root", ovstorage.ConfigValue.string(str(root)))
    stack = asyncio.run(
        (
            ovstorage.Stack(root="files")
            .backend(ovstorage.file.FileBackend("files"))
            .connection("files", request)
        ).build()
    )
    result = OvstorageBackend(OvstorageBackendConfig(root_uri=root.as_uri()), stack)
    try:
        yield result
    finally:
        result.close()


def test_round_trip_and_listing(backend):
    backend.makedirs("sub/deep")
    backend.makedirs("sub/deep")
    backend.write("sub/hello world.txt", b"hello")
    backend.write("sub/second.bin", b"second")
    assert backend.read("sub/hello world.txt") == b"hello"
    assert backend.exists("sub/hello world.txt")
    assert not backend.exists("sub/missing.txt")
    assert backend.list() == ["sub/hello world.txt", "sub/second.bin"]
    assert backend.list("sub") == ["sub/hello world.txt", "sub/second.bin"]
    assert backend.list("other") == []


def test_fail_if_exists_and_invalid_path(backend):
    backend.write("sample.bin", b"original")
    backend._config.overwrite = False
    with pytest.raises(BackendWriteError):
        backend.write("sample.bin", b"replacement")
    assert backend.read("sample.bin") == b"original"
    with pytest.raises(BackendWriteError):
        backend.write("../outside.bin", b"escape")


def test_read_missing_and_running_event_loop(backend):
    with pytest.raises(BackendReadError):
        backend.read("missing.bin")

    async def write_inside_loop():
        backend.write("async.bin", b"ok")
        return backend.read("async.bin")

    assert asyncio.run(write_inside_loop()) == b"ok"


def test_array_encoding(backend):
    array = np.arange(6).reshape(2, 3)
    backend.write_array("values.npy", array)
    np.testing.assert_array_equal(
        np.load(io.BytesIO(backend.read("values.npy"))), array
    )


def test_close_releases_loop(backend):
    backend.close()
    with pytest.raises(BackendWriteError, match="closed"):
        backend.write("closed.bin", b"x")


def test_no_overwrite_root_created_by_writes(backend):
    strict = OvstorageBackend(
        OvstorageBackendConfig(
            root_uri=backend._config.root_uri + "/run-1", overwrite=False
        ),
        backend._stack,
    )
    try:
        strict.write("frames/0001/rgb.bin", b"one")
        assert strict.read("frames/0001/rgb.bin") == b"one"
        with pytest.raises(BackendWriteError):
            strict.write("frames/0001/rgb.bin", b"two")
    finally:
        strict.close()
    assert backend.list("run-1") == ["run-1/frames/0001/rgb.bin"]
