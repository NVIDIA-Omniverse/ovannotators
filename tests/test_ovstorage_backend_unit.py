# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the adapter's calls to a built ovstorage Stack in isolation."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ovannotators.backends.ovstorage import OvstorageBackend, OvstorageBackendConfig
from ovannotators.exceptions import (
    BackendReadError,
    BackendWriteError,
    ConfigurationError,
)

ovstorage = pytest.importorskip("ovstorage")


def _stack(**overrides):
    calls = dict(
        write=AsyncMock(),
        read_bytes=AsyncMock(return_value=(b"payload", None)),
        list=AsyncMock(),
        stat=AsyncMock(),
        create_directory=AsyncMock(),
        list_address_roots=AsyncMock(return_value=[]),
    )
    calls.update(overrides)
    return SimpleNamespace(**calls)


def _root(address, *, no_overwrite):
    return SimpleNamespace(
        address=address,
        capabilities=SimpleNamespace(supports_no_overwrite_write=no_overwrite),
    )


def _entry(address, kind):
    return SimpleNamespace(address=address, kind=kind)


@pytest.fixture
def backend():
    stack = _stack()
    result = OvstorageBackend(
        OvstorageBackendConfig(root_uri="s3://bucket/datasets/", overwrite=False),
        stack,
    )
    try:
        yield result, stack
    finally:
        result.close()


def test_write_uses_full_address_and_atomic_no_overwrite(backend):
    adapter, stack = backend
    adapter.write("frame 1/data.bin", b"payload")
    stack.write.assert_awaited_once_with(
        "s3://bucket/datasets/frame%201/data.bin",
        b"payload",
        if_dest_exists="fail",
    )

    stack.write.side_effect = RuntimeError("upload failed")
    with pytest.raises(BackendWriteError, match="upload failed"):
        adapter.write("failed.bin", b"payload")


def test_list_follows_pages_and_returns_relative_file_paths(backend):
    adapter, stack = backend
    stack.list.side_effect = [
        SimpleNamespace(
            items=[
                SimpleNamespace(
                    address="s3://bucket/datasets/sub/nested/", kind="directory"
                ),
                SimpleNamespace(
                    address="s3://bucket/datasets/sub/z%20one.bin", kind="file"
                ),
            ],
            next_page_token="page-2",
        ),
        SimpleNamespace(
            items=[
                SimpleNamespace(address="s3://bucket/datasets/sub/a.bin", kind="file")
            ],
            next_page_token=None,
        ),
        SimpleNamespace(
            items=[
                SimpleNamespace(
                    address="s3://bucket/datasets/sub/nested/deep.bin", kind="file"
                )
            ],
            next_page_token=None,
        ),
    ]

    assert adapter.list("sub") == [
        "sub/a.bin",
        "sub/nested/deep.bin",
        "sub/z one.bin",
    ]
    assert stack.list.await_args_list[0].kwargs == {
        "recursive": False,
        "page_token": None,
    }
    assert stack.list.await_args_list[1].kwargs == {
        "recursive": False,
        "page_token": "page-2",
    }
    assert stack.list.await_args_list[2].args[0] == "s3://bucket/datasets/sub/nested/"


def test_read_and_exists_distinguish_missing_from_other_failures(backend):
    adapter, stack = backend
    assert adapter.read("file.bin") == b"payload"
    assert adapter.exists("file.bin")

    stack.stat.side_effect = ovstorage.NotFoundError("missing")
    assert not adapter.exists("missing.bin")
    stack.stat.side_effect = RuntimeError("permission denied")
    with pytest.raises(BackendReadError, match="permission denied"):
        adapter.exists("private.bin")


def test_list_descends_flat_store_directory_kinds(backend):
    adapter, stack = backend
    stack.list.side_effect = [
        SimpleNamespace(
            items=[
                _entry("s3://bucket/datasets/0001/", "directory_inferred"),
                _entry("s3://bucket/datasets/0002/", "directory_marker"),
                _entry("s3://bucket/datasets/placeholder/", "file"),
                _entry("s3://bucket/datasets/top.json", "file"),
            ],
            next_page_token=None,
        ),
        SimpleNamespace(
            items=[
                _entry("s3://bucket/datasets/0002/", "directory_marker"),
                _entry("s3://bucket/datasets/0002/depth.npy", "file"),
            ],
            next_page_token=None,
        ),
        SimpleNamespace(
            items=[_entry("s3://bucket/datasets/0001/rgb.png", "file")],
            next_page_token=None,
        ),
    ]

    assert adapter.list() == ["0001/rgb.png", "0002/depth.npy", "top.json"]
    assert stack.list.await_count == 3


def test_makedirs_leaves_directory_creation_to_writes(backend):
    adapter, stack = backend
    adapter.makedirs("0001/rgb")
    stack.create_directory.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        ovstorage.NoRouteError("no route"),
        ovstorage.NotConfiguredError("not configured"),
        ovstorage.AuthRequiredError("sign in"),
    ],
)
def test_construction_rejects_unusable_route(error):
    stack = _stack(stat=AsyncMock(side_effect=error))
    with pytest.raises(ConfigurationError, match="s3://bucket/datasets/"):
        OvstorageBackend(
            OvstorageBackendConfig(root_uri="s3://bucket/datasets/"), stack
        )


def test_construction_accepts_missing_root():
    stack = _stack(stat=AsyncMock(side_effect=ovstorage.NotFoundError("missing")))
    OvstorageBackend(
        OvstorageBackendConfig(root_uri="s3://bucket/datasets/"), stack
    ).close()
    stack.list_address_roots.assert_not_awaited()


def test_no_overwrite_requires_route_support():
    roots = [
        _root("s3://bucket/", no_overwrite=True),
        _root("s3://bucket/datasets/", no_overwrite=False),
        _root("s3://other/", no_overwrite=True),
    ]
    stack = _stack(list_address_roots=AsyncMock(return_value=roots))
    with pytest.raises(ConfigurationError, match="fail-if-exists"):
        OvstorageBackend(
            OvstorageBackendConfig(
                root_uri="s3://bucket/datasets/run/", overwrite=False
            ),
            stack,
        )

    stack = _stack(
        list_address_roots=AsyncMock(
            return_value=[_root("s3://bucket/", no_overwrite=True)]
        )
    )
    OvstorageBackend(
        OvstorageBackendConfig(root_uri="s3://bucket/datasets/", overwrite=False),
        stack,
    ).close()


def test_list_skips_child_removed_during_walk(backend):
    adapter, stack = backend
    stack.list.side_effect = [
        SimpleNamespace(
            items=[
                _entry("s3://bucket/datasets/gone/", "directory"),
                _entry("s3://bucket/datasets/kept.bin", "file"),
            ],
            next_page_token=None,
        ),
        ovstorage.NotFoundError("gone"),
    ]

    assert adapter.list() == ["kept.bin"]


def test_list_of_missing_prefix_is_empty(backend):
    adapter, stack = backend
    stack.list.side_effect = ovstorage.NotFoundError("missing")

    assert adapter.list("absent") == []


def test_close_waits_for_in_flight_operation():
    started = threading.Event()
    release = threading.Event()

    async def slow_write(*_args, **_kwargs):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)

    adapter = OvstorageBackend(
        OvstorageBackendConfig(root_uri="s3://bucket/datasets/"),
        _stack(write=slow_write),
    )
    errors = []

    def write():
        try:
            adapter.write("a.bin", b"x")
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    writer = threading.Thread(target=write)
    writer.start()
    assert started.wait(5)
    closer = threading.Thread(target=adapter.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()

    release.set()
    closer.join(5)
    writer.join(5)
    assert not closer.is_alive()
    assert not writer.is_alive()
    assert errors == []
    with pytest.raises(BackendWriteError, match="closed"):
        adapter.write("b.bin", b"x")
