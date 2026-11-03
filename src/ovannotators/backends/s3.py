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

from ovannotators.backends._encoding import encode_array
from ovannotators.backends._paths import confine_relative
from ovannotators.exceptions import (
    BackendReadError,
    BackendWriteError,
    ConfigurationError,
)

try:
    import boto3  # type: ignore[import-untyped]
    import boto3.s3.transfer  # type: ignore[import-untyped]
    import botocore.exceptions  # type: ignore[import-untyped]

    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore[assignment]
    botocore = None  # type: ignore[assignment]


# S3 rejects a single PutObject larger than 5 GiB, so larger payloads must use
# multipart upload whatever threshold is configured.
_MAX_SINGLE_PUT_BYTES = 5 * 1024**3

# S3's minimum multipart part size, and an upper bound on part size. s3transfer
# buffers in-flight parts in memory, so parts stay small even when a large
# threshold only routes big payloads to multipart. S3 allows at most 10,000
# parts: 64 MiB parts cover 10,000 x 64 MiB = 640,000 MiB (about 625 GiB), well
# beyond any payload held in memory; s3transfer enlarges parts automatically if
# a payload would still exceed 10,000.
_MIN_PART_BYTES = 5 * 1024**2
_MAX_PART_BYTES = 64 * 1024**2


@dataclass
class S3BackendConfig:
    """Configuration for :class:`S3Backend`.

    Attributes:
        bucket (str): Target S3 bucket name.
        prefix (str): Key prefix prepended to every relative path.
        region (str | None): AWS region; ``None`` uses the boto3 default.
        endpoint_url (str | None): Custom endpoint for non-AWS, S3-compatible
            stores (e.g. MinIO); ``None`` uses AWS.
        multipart_threshold_mb (int): Object size (in MiB) at or above which
            :meth:`S3Backend.write` switches from a single ``PutObject`` to a
            managed multipart upload. Parts are as large as the threshold,
            bounded to 5-64 MiB. Values above 5120 (S3's 5 GiB
            single-request limit) are capped to it for routing.
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
    Paths stay below the configured prefix: an absolute path or one with a
    ``..`` or empty segment is rejected before any request is made.
    """

    name: ClassVar[str] = "s3"

    def __init__(self, config: S3BackendConfig) -> None:
        """Initialize the backend and its boto3 client.

        Args:
            config (S3BackendConfig): Bucket, prefix, and client options.

        Raises:
            ImportError: If ``boto3`` is not installed.
            ConfigurationError: If ``multipart_threshold_mb`` is not positive.
        """
        if not _BOTO3_AVAILABLE:
            raise ImportError(
                "S3Backend requires boto3. Install it with: pip install boto3"
            )
        if config.multipart_threshold_mb <= 0:
            raise ConfigurationError(
                "multipart_threshold_mb must be positive, got "
                f"{config.multipart_threshold_mb}"
            )
        self._config = config
        self._prefix = config.prefix.strip("/")
        kwargs: dict[str, Any] = {}
        if config.region:
            kwargs["region_name"] = config.region
        if config.endpoint_url:
            kwargs["endpoint_url"] = config.endpoint_url
        self._client = boto3.client("s3", **kwargs)
        self._multipart_threshold = min(
            config.multipart_threshold_mb * 1024 * 1024, _MAX_SINGLE_PUT_BYTES
        )
        self._transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=self._multipart_threshold,
            multipart_chunksize=max(
                _MIN_PART_BYTES, min(self._multipart_threshold, _MAX_PART_BYTES)
            ),
        )

    def _key(self, path: str, *, allow_empty: bool = False) -> str:
        relative = confine_relative(path, allow_empty=allow_empty)
        return f"{self._prefix}/{relative}" if self._prefix else relative

    def write(self, path: str, data: bytes) -> None:
        """Upload raw bytes to the resolved S3 key.

        Payloads smaller than ``multipart_threshold_mb`` go up in one
        ``PutObject`` request; larger ones (videos, big EXR/NPY arrays) use a
        managed multipart upload, which parallelizes parts and stays under the
        5 GiB single-request limit.

        Args:
            path (str): Destination path, relative to the configured prefix.
            data (bytes): Bytes to upload.

        Raises:
            BackendWriteError: If the path leaves the prefix or the upload fails.
        """
        try:
            key = self._key(path)
            if len(data) >= self._multipart_threshold:
                self._client.upload_fileobj(
                    io.BytesIO(data),
                    self._config.bucket,
                    key,
                    Config=self._transfer_config,
                )
            else:
                self._client.put_object(
                    Bucket=self._config.bucket,
                    Key=key,
                    Body=data,
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
            exr_compression (str | None): EXR compression scheme, or ``None``
                for OpenEXR's default ZIP. Only valid for EXR.

        Raises:
            BackendWriteError: If the encoding is unsupported or the upload fails.
            ValueError: If ``exr_compression`` is set and ``encoding`` is not
                ``"exr"``, or if the EXR compression scheme is not supported.
        """
        self.write(
            path, encode_array(path, array, encoding, exr_compression=exr_compression)
        )

    def read(self, path: str) -> bytes:
        """Download raw bytes from the resolved S3 key.

        Args:
            path (str): Source path, relative to the configured prefix.

        Returns:
            bytes: The object contents.

        Raises:
            BackendReadError: If the path leaves the prefix or the object
                cannot be read.
        """
        try:
            key = self._key(path)
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
            BackendReadError: If the prefix leaves the configured prefix or the
                listing fails.
        """
        try:
            key_prefix = self._key(prefix, allow_empty=True)
            paginator = self._client.get_paginator("list_objects_v2")
            results: list[str] = []
            for page in paginator.paginate(
                Bucket=self._config.bucket, Prefix=key_prefix
            ):
                for obj in page.get("Contents", []):
                    # Strip the backend prefix to return relative paths.
                    rel = obj["Key"]
                    root = self._prefix
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
            bool: ``True`` if the object exists; ``False`` if it does not or
            the path leaves the prefix.
        """
        try:
            key = self._key(path)
            self._client.head_object(Bucket=self._config.bucket, Key=key)
            return True
        except Exception:
            return False

    def makedirs(self, path: str) -> None:
        """No-op; S3 has no real directories.

        Args:
            path (str): Ignored; present for :class:`Backend` compatibility.
        """
