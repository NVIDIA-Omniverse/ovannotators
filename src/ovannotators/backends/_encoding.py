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

"""Array serialization shared by the built-in backends' ``write_array``."""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from ovannotators.exceptions import BackendWriteError


def encode_array(
    path: str,
    array: Any,
    encoding: str,
    *,
    exr_compression: str | None = None,
) -> bytes:
    """Serialize an array to the bytes a backend should store at ``path``.

    PNG/JPEG encoding uses the shared image codec
    (:func:`ovannotators._internal.codecs.get_image_codec`), which prefers
    nvImageCodec on the GPU (encoding GPU-resident Warp arrays without a host
    round trip) and falls back to Pillow. Other encodings copy to host NumPy.

    Args:
        path (str): Destination path, used only in error messages.
        array (Any): NumPy or Warp array to encode.
        encoding (str): One of ``"npy"``, ``"png"``, ``"jpg"``, or ``"exr"``.
        exr_compression (str | None): EXR compression scheme, or ``None`` for
            OpenEXR's default ZIP. Only valid for EXR.

    Returns:
        bytes: The encoded payload.

    Raises:
        BackendWriteError: If the encoding is unsupported or OpenEXR is missing.
        ValueError: If ``exr_compression`` is set and ``encoding`` is not
            ``"exr"``, or if the EXR compression scheme is not supported.
    """
    if exr_compression is not None and encoding != "exr":
        raise ValueError("exr_compression requires encoding='exr'")
    if encoding == "npy":
        from ovannotators._internal.codecs import _to_numpy

        buf = io.BytesIO()
        np.save(buf, _to_numpy(array))
        return buf.getvalue()
    if encoding in ("png", "jpg"):
        from ovannotators._internal.codecs import get_image_codec

        return get_image_codec().encode(array, format=encoding)
    if encoding == "exr":
        try:
            from ovannotators._internal.codecs import _encode_exr_bytes

            return _encode_exr_bytes(array, compression=exr_compression)
        except ImportError:
            raise BackendWriteError(
                path=path,
                message="Cannot encode 'exr': install OpenEXR",
            ) from None
    raise BackendWriteError(
        path=path,
        message=f"Unsupported encoding '{encoding}'. Use 'npy', 'png', 'jpg', or 'exr'.",
    )
