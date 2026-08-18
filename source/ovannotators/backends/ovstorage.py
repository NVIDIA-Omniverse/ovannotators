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

"""OvstorageBackend — reads and writes via an HTTP storage API service."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ovannotators.exceptions import BackendReadError, BackendWriteError

try:
    import httpx  # type: ignore[import-untyped]

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]


@dataclass
class OvstorageBackendConfig:
    """Configuration for :class:`OvstorageBackend`.

    Attributes:
        base_url (str): Base URL of the storage API service.
        api_key (str | None): Bearer token; when ``None`` it is read from the
            ``STORAGE_API_KEY`` environment variable at construction time.
        prefix (str): Path prefix prepended to every relative path.
        timeout_s (float): Per-request timeout in seconds.
        verify_ssl (bool): Whether to verify TLS certificates.
    """

    base_url: str
    api_key: str | None = None
    prefix: str = ""
    timeout_s: float = 30.0
    verify_ssl: bool = True


class OvstorageBackend:
    """HTTP storage API backend.

    Credentials are resolved from env var STORAGE_API_KEY when api_key is None
    in the config. The api_key value is consumed at __init__ time and not stored
    beyond construction.

    Requires httpx: `pip install httpx`.
    makedirs() is a no-op (object store).
    """

    name: ClassVar[str] = "ovstorage"

    def __init__(self, config: OvstorageBackendConfig) -> None:
        """Initialize the backend and its HTTP client.

        The API key is consumed to build the auth header and is not retained on
        the instance.

        Args:
            config (OvstorageBackendConfig): Service URL, auth, and client
                options.

        Raises:
            ImportError: If ``httpx`` is not installed.
        """
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "OvstorageBackend requires httpx. Install it with: pip install httpx"
            )
        self._config = config
        api_key = config.api_key or os.environ.get("STORAGE_API_KEY")
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_s,
            verify=config.verify_ssl,
        )
        # Do not retain api_key beyond construction.

    def _url(self, path: str) -> str:
        prefix = self._config.prefix.rstrip("/")
        rel = f"{prefix}/{path}".lstrip("/") if prefix else path
        return f"/{rel}"

    def write(self, path: str, data: bytes) -> None:
        """Upload raw bytes via HTTP ``PUT``.

        Args:
            path (str): Destination path, relative to the configured prefix.
            data (bytes): Bytes to upload.

        Raises:
            BackendWriteError: If the request fails or returns an error status.
        """
        url = self._url(path)
        try:
            response = self._client.put(url, content=data)
            response.raise_for_status()
        except Exception as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc

    def write_array(self, path: str, array: np.ndarray, encoding: str = "npy") -> None:
        """Serialize an array and upload the encoded bytes via HTTP PUT.

        PNG/JPEG encoding uses the shared image codec
        (:func:`ovannotators._internal.codecs.get_image_codec`), which prefers
        nvImageCodec on the GPU and falls back to Pillow. Warp arrays are
        materialized to host bytes as needed (HTTP uploads require host bytes
        regardless).

        Args:
            path (str): Destination path, relative to the configured prefix.
            array (np.ndarray): Image/data array to encode (NumPy or Warp).
            encoding (str): One of ``"npy"``, ``"png"``, ``"jpg"``, or ``"exr"``.

        Raises:
            BackendWriteError: If the encoding is unsupported or the upload fails.
        """
        from ovannotators._internal.codecs import _to_numpy

        buf = io.BytesIO()
        if encoding == "npy":
            np.save(buf, _to_numpy(array))
        elif encoding in ("png", "jpg"):
            from ovannotators._internal.codecs import get_image_codec

            buf.write(get_image_codec().encode(array, format=encoding))
        elif encoding == "exr":
            try:
                from ovannotators._internal.codecs import _encode_exr_bytes

                buf.write(_encode_exr_bytes(array))
            except ImportError:
                raise BackendWriteError(
                    path=path,
                    message="Cannot encode 'exr': install OpenEXR",
                ) from None
        else:
            raise BackendWriteError(
                path=path,
                message=f"Unsupported encoding '{encoding}'.",
            )
        self.write(path, buf.getvalue())

    def read(self, path: str) -> bytes:
        """Download raw bytes via HTTP ``GET``.

        Args:
            path (str): Source path, relative to the configured prefix.

        Returns:
            bytes: The response body.

        Raises:
            BackendReadError: If the request fails or returns an error status.
        """
        url = self._url(path)
        try:
            response = self._client.get(url)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            raise BackendReadError(path=path, message=str(exc)) from exc

    def list(self, prefix: str = "") -> list[str]:
        """List paths under a prefix via HTTP ``GET``.

        Args:
            prefix (str): Path prefix to filter by, relative to the configured
                prefix; empty lists everything.

        Returns:
            list[str]: Paths returned by the service.

        Raises:
            BackendReadError: If the request fails or returns an error status.
        """
        url = self._url(prefix)
        try:
            response = self._client.get(url, params={"list": "true"})
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise BackendReadError(path=prefix, message=str(exc)) from exc

    def exists(self, path: str) -> bool:
        """Report whether a path exists via HTTP ``HEAD``.

        Args:
            path (str): Path to check, relative to the configured prefix.

        Returns:
            bool: ``True`` if the service returns a ``200`` status.
        """
        url = self._url(path)
        try:
            response = self._client.head(url)
            return response.status_code == 200
        except Exception:
            return False

    def makedirs(self, path: str) -> None:
        """No-op; the object store has no real directories.

        Args:
            path (str): Ignored; present for :class:`Backend` compatibility.
        """

    def __del__(self) -> None:
        """Close the underlying HTTP client on garbage collection."""
        try:
            self._client.close()
        except Exception:
            pass
