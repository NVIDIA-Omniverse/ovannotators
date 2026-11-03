# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend adapter for a built ovstorage Stack."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import unquote

import numpy as np

from ovannotators.backends._encoding import encode_array
from ovannotators.exceptions import (
    BackendReadError,
    BackendWriteError,
    ConfigurationError,
)

try:
    import ovstorage

    _OVSTORAGE_AVAILABLE = True
except ImportError:
    ovstorage = None  # type: ignore[assignment]
    _OVSTORAGE_AVAILABLE = False


@dataclass
class OvstorageBackendConfig:
    """Configuration for a dataset rooted at an ovstorage address.

    Attributes:
        root_uri (str): Directory address below which writer paths are stored.
        overwrite (bool): Whether writes may replace existing objects.
    """

    root_uri: str
    overwrite: bool = True


class OvstorageBackend:
    """Adapt a caller-owned, built ovstorage Stack to the synchronous Backend API.

    The caller configures connections, plugins, auth, caching, and routing on
    the Stack. This adapter owns only an event loop for synchronous dispatch;
    it does not close the caller's Stack. Call :meth:`close` when finished.
    """

    name: ClassVar[str] = "ovstorage"

    def __init__(self, config: OvstorageBackendConfig, stack: object) -> None:
        """Bind the backend to a built Stack and a directory address.

        Args:
            config (OvstorageBackendConfig): Dataset root and overwrite policy.
            stack (object): Result of awaiting ``ovstorage.Stack.build()``.

        Raises:
            ImportError: If the optional ovstorage wheel is unavailable.
            ValueError: If the root is not a valid directory address.
            ConfigurationError: If the Stack has no route or signed-in
                connection for the root, or ``overwrite=False`` targets a root
                that reports no fail-if-exists write support.
        """
        if not _OVSTORAGE_AVAILABLE:
            raise ImportError(
                "OvstorageBackend requires ovstorage. "
                "Install it with: pip install 'ovannotators[ovstorage]'"
            )
        self._config = config
        self._stack = stack
        self._root = ovstorage.address.to_directory(config.root_uri)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._closed = False
        self._in_flight: set[concurrent.futures.Future] = set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()
        try:
            self._check_route()
        except BaseException:
            self.close()
            raise

    def _check_route(self) -> None:
        """Reject a Stack that cannot serve the root before any frame is written.

        Errors other than a missing route or sign-in are left to the first
        operation. The ``overwrite=False`` check uses the longest advertised
        root covering the dataset root; a ``False`` capability is authoritative,
        and roots reached only through alias layers are not checked.
        """
        try:
            self._await(self._stack.stat(self._root))
        except (ovstorage.NoRouteError, ovstorage.NotConfiguredError) as exc:
            raise ConfigurationError(
                f"ovstorage Stack has no route for {self._root}: {exc}"
            ) from exc
        except (ovstorage.AuthRequiredError, ovstorage.CredentialExpiredError) as exc:
            raise ConfigurationError(
                f"ovstorage connection for {self._root} needs sign-in: {exc}"
            ) from exc
        except ovstorage.Error:
            pass  # A root that does not exist yet is valid.
        if self._config.overwrite:
            return
        covering = [
            root
            for root in self._await(self._stack.list_address_roots())
            if ovstorage.address.is_prefix_of(
                ovstorage.address.to_directory(root.address), self._root
            )
        ]
        if not covering:
            return
        route = max(covering, key=lambda root: len(root.address))
        if not route.capabilities.supports_no_overwrite_write:
            raise ConfigurationError(
                f"overwrite=False needs fail-if-exists writes, which the "
                f"ovstorage route {route.address} does not support"
            )

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def _await(self, coroutine: object) -> object:
        with self._lock:
            if self._closed or not self._thread.is_alive():
                coroutine.close()  # type: ignore[attr-defined]
                raise RuntimeError("OvstorageBackend is closed")
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            self._in_flight.add(future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._in_flight.discard(future)

    def _address(self, path: str) -> str:
        if not path:
            return self._root
        return ovstorage.address.join_relative(self._root, path)

    def write(self, path: str, data: bytes) -> None:
        """Write bytes through the caller's ovstorage Stack.

        Args:
            path (str): Path relative to the dataset root.
            data (bytes): Payload to store.

        Raises:
            BackendWriteError: If the write fails.
        """
        try:
            self._await(
                self._stack.write(
                    self._address(path),
                    data,
                    if_dest_exists="overwrite" if self._config.overwrite else "fail",
                )
            )
        except Exception as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc

    def write_array(
        self,
        path: str,
        array: np.ndarray,
        encoding: str = "npy",
        *,
        exr_compression: str | None = None,
    ) -> None:
        """Encode an array and write it through the Stack.

        Args:
            path (str): Path relative to the dataset root.
            array (np.ndarray): Array to encode.
            encoding (str): One of ``npy``, ``png``, ``jpg``, or ``exr``.
            exr_compression (str | None): Optional EXR compression scheme.
        """
        self.write(
            path, encode_array(path, array, encoding, exr_compression=exr_compression)
        )

    def read(self, path: str) -> bytes:
        """Read bytes from a path relative to the dataset root.

        Args:
            path (str): Relative object path.

        Returns:
            bytes: Object contents.

        Raises:
            BackendReadError: If the read fails.
        """
        try:
            data, _info = self._await(self._stack.read_bytes(self._address(path)))
            return data
        except Exception as exc:
            raise BackendReadError(path=path, message=str(exc)) from exc

    def list(self, prefix: str = "") -> list[str]:
        """List all objects below a relative prefix, across SDK pages.

        Args:
            prefix (str): Relative object prefix.

        Returns:
            list[str]: Relative object paths.

        Raises:
            BackendReadError: If listing fails.
        """
        try:
            address = (
                ovstorage.address.to_directory(self._address(prefix))
                if prefix
                else self._root
            )
            paths: set[str] = set()
            pending = [address]
            visited: set[str] = set()
            while pending:
                directory = pending.pop()
                if directory in visited:
                    continue
                visited.add(directory)
                page_token = None
                while True:
                    try:
                        page = self._await(
                            self._stack.list(
                                directory, recursive=False, page_token=page_token
                            )
                        )
                    except ovstorage.NotFoundError:
                        if directory == address and page_token is None:
                            return []
                        break  # A child removed during the walk; keep the rest.
                    for item in page.items:
                        if (
                            ovstorage.address.strip_prefix(item.address, directory)
                            is None
                        ):
                            continue
                        relative = ovstorage.address.strip_prefix(
                            item.address, self._root
                        )
                        if relative is None:
                            continue
                        if item.kind != "file":
                            # Flat stores report subprefixes as
                            # directory_inferred and markers as directory_marker.
                            pending.append(ovstorage.address.to_directory(item.address))
                        elif not ovstorage.address.is_directory(item.address):
                            path = unquote(relative)
                            if not prefix or path.startswith(prefix.rstrip("/") + "/"):
                                paths.add(path)
                    page_token = page.next_page_token
                    if page_token is None:
                        break
            return sorted(paths)
        except Exception as exc:
            raise BackendReadError(path=prefix, message=str(exc)) from exc

    def exists(self, path: str) -> bool:
        """Return whether an object exists at a relative path.

        Args:
            path (str): Relative object path.

        Returns:
            bool: Whether the Stack can stat the object.

        Raises:
            BackendReadError: If stat fails for a reason other than not found.
        """
        try:
            self._await(self._stack.stat(self._address(path)))
            return True
        except ovstorage.NotFoundError:
            return False
        except Exception as exc:
            raise BackendReadError(path=path, message=str(exc)) from exc

    def makedirs(self, path: str) -> None:
        """No-op; ovstorage writes create their own parent directories.

        Calling ``create_directory`` here would cost a round trip per path
        level and, on flat object stores, write a marker object.

        Args:
            path (str): Ignored; present for :class:`Backend` compatibility.
        """

    def close(self) -> None:
        """Stop the adapter's event loop without closing the caller's Stack.

        Operations already in flight on other threads finish first; later
        calls raise the backend's read or write error.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            in_flight = list(self._in_flight)
        concurrent.futures.wait(in_flight)
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
