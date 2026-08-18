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

"""S3Backend — reads and writes to Amazon S3 or S3-compatible stores."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ovannotators.exceptions import BackendReadError, BackendWriteError

try:
    import boto3  # type: ignore[import-untyped]
    import botocore.exceptions  # type: ignore[import-untyped]

    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore[assignment]
    botocore = None  # type: ignore[assignment]


@dataclass
class S3BackendConfig:
    """Configuration for :class:`S3Backend`.

    Attributes:
        bucket (str): Target S3 bucket name.
        prefix (str): Key prefix prepended to every relative path.
        region (str | None): AWS region; ``None`` uses the boto3 default.
        endpoint_url (str | None): Custom endpoint for non-AWS, S3-compatible
            stores (e.g. MinIO); ``None`` uses AWS.
        multipart_threshold_mb (int): Object size (in MiB) above which multipart
            transfer is used.
    """

    bucket: str
    prefix: str = ""
    region: str | None = None
    endpoint_url: str | None = None
    multipart_threshold_mb: int = 8


class S3Backend:
    """S3 backend. Credentials resolved via standard boto3 chain.

    Requires boto3: `pip install boto3`.
    makedirs() is a no-op (S3 has no real directories).
    """

    name: ClassVar[str] = "s3"

    def __init__(self, config: S3BackendConfig) -> None:
        """Initialize the backend and its boto3 client.

        Args:
            config (S3BackendConfig): Bucket, prefix, and client options.

        Raises:
            ImportError: If ``boto3`` is not installed.
        """
        if not _BOTO3_AVAILABLE:
            raise ImportError(
                "S3Backend requires boto3. Install it with: pip install boto3"
            )
        self._config = config
        kwargs: dict[str, Any] = {}
        if config.region:
            kwargs["region_name"] = config.region
        if config.endpoint_url:
            kwargs["endpoint_url"] = config.endpoint_url
        self._client = boto3.client("s3", **kwargs)
        threshold = config.multipart_threshold_mb * 1024 * 1024
        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=threshold,
            multipart_chunksize=threshold,
        )
        self._transfer_config = transfer_config

    def _key(self, path: str) -> str:
        prefix = self._config.prefix.rstrip("/")
        return f"{prefix}/{path}".lstrip("/") if prefix else path

    def write(self, path: str, data: bytes) -> None:
        """Upload raw bytes to the resolved S3 key.

        Args:
            path (str): Destination path, relative to the configured prefix.
            data (bytes): Bytes to upload.

        Raises:
            BackendWriteError: If the upload fails.
        """
        key = self._key(path)
        try:
            self._client.put_object(
                Bucket=self._config.bucket,
                Key=key,
                Body=data,
            )
        except Exception as exc:
            raise BackendWriteError(path=path, message=str(exc)) from exc

    def write_array(self, path: str, array: np.ndarray, encoding: str = "npy") -> None:
        """Serialize an array and upload the encoded bytes to S3.

        PNG/JPEG encoding uses the shared image codec
        (:func:`ovannotators._internal.codecs.get_image_codec`), which prefers
        nvImageCodec on the GPU and falls back to Pillow. Warp arrays are
        materialized to host bytes as needed (object storage uploads require
        host bytes regardless).

        Args:
            path (str): Destination key, relative to the configured prefix.
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
        """Download raw bytes from the resolved S3 key.

        Args:
            path (str): Source path, relative to the configured prefix.

        Returns:
            bytes: The object contents.

        Raises:
            BackendReadError: If the object cannot be read.
        """
        key = self._key(path)
        try:
            response = self._client.get_object(Bucket=self._config.bucket, Key=key)
            return response["Body"].read()
        except Exception as exc:
            raise BackendReadError(path=path, message=str(exc)) from exc

    def list(self, prefix: str = "") -> list[str]:
        """List object keys under a prefix.

        Args:
            prefix (str): Path prefix to filter by, relative to the configured
                prefix; empty lists the whole prefix.

        Returns:
            list[str]: Object paths relative to the configured prefix.

        Raises:
            BackendReadError: If the listing fails.
        """
        key_prefix = self._key(prefix)
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            results: list[str] = []
            for page in paginator.paginate(
                Bucket=self._config.bucket, Prefix=key_prefix
            ):
                for obj in page.get("Contents", []):
                    # Strip the backend prefix to return relative paths.
                    rel = obj["Key"]
                    root = self._config.prefix.rstrip("/")
                    if root and rel.startswith(root + "/"):
                        rel = rel[len(root) + 1 :]
                    results.append(rel)
            return results
        except Exception as exc:
            raise BackendReadError(path=prefix, message=str(exc)) from exc

    def exists(self, path: str) -> bool:
        """Report whether an S3 key exists.

        Args:
            path (str): Path to check, relative to the configured prefix.

        Returns:
            bool: ``True`` if the object exists.
        """
        key = self._key(path)
        try:
            self._client.head_object(Bucket=self._config.bucket, Key=key)
            return True
        except Exception:
            return False

    def makedirs(self, path: str) -> None:
        """No-op; S3 has no real directories.

        Args:
            path (str): Ignored; present for :class:`Backend` compatibility.
        """
