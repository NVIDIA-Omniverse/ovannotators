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

"""Minimal MATLAB Level 5 MAT-file writer.

Writes uncompressed little-endian MAT v5 files holding numeric arrays, the
subset dataset formats such as YCB-Video need, without a scipy dependency. A
1-D array is stored as a row vector, matching ``scipy.io.savemat``.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping

import numpy as np

# MAT-file data types (miXXX) and array classes (mxXXX_CLASS).
_MI_INT8, _MI_UINT8, _MI_INT32, _MI_UINT16 = 1, 2, 5, 4
_MI_UINT32, _MI_DOUBLE, _MI_MATRIX = 6, 9, 14
_MX_DOUBLE_CLASS, _MX_UINT8_CLASS, _MX_UINT16_CLASS = 6, 9, 11

# numpy dtype -> (miXXX data type, mxXXX_CLASS array class)
_SUPPORTED: dict[np.dtype, tuple[int, int]] = {
    np.dtype("<f8"): (_MI_DOUBLE, _MX_DOUBLE_CLASS),
    np.dtype("u1"): (_MI_UINT8, _MX_UINT8_CLASS),
    np.dtype("<u2"): (_MI_UINT16, _MX_UINT16_CLASS),
}

_MAX_NAME_LENGTH = 63


def _element(data_type: int, payload: bytes) -> bytes:
    """Return one data element: tag plus payload, padded as the format requires.

    Payloads of at most four bytes use the compact "small element" tag, the
    rest the regular 8-byte tag with the payload padded to 8 bytes.
    """
    size = len(payload)
    if size <= 4:
        return struct.pack("<I", (size << 16) | data_type) + payload.ljust(4, b"\0")
    return struct.pack("<II", data_type, size) + payload.ljust(
        (size + 7) // 8 * 8, b"\0"
    )


def _matrix(name: str, array: np.ndarray) -> bytes:
    """Return the ``miMATRIX`` element for one named numeric array."""
    if not name or len(name) > _MAX_NAME_LENGTH or not name.isascii():
        raise ValueError(
            f"MAT variable name must be 1-{_MAX_NAME_LENGTH} ASCII characters"
        )
    array = np.asarray(array)
    dtype = array.dtype.newbyteorder("<")
    if dtype not in _SUPPORTED:
        raise TypeError(
            f"MAT variable {name!r} has dtype {array.dtype}; supported: float64, uint8, uint16"
        )
    data_type, array_class = _SUPPORTED[dtype]
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    # MATLAB stores arrays column-major.
    values = np.ascontiguousarray(array.astype(dtype, copy=False).ravel(order="F"))
    body = b"".join(
        (
            _element(_MI_UINT32, struct.pack("<II", array_class, 0)),
            _element(_MI_INT32, struct.pack(f"<{array.ndim}i", *array.shape)),
            _element(_MI_INT8, name.encode("ascii")),
            _element(data_type, values.tobytes()),
        )
    )
    return struct.pack("<II", _MI_MATRIX, len(body)) + body


def encode_mat5(variables: Mapping[str, np.ndarray]) -> bytes:
    """Encode named numeric arrays as an uncompressed MAT v5 file.

    Args:
        variables: Mapping of variable name to array. Supported dtypes are
            ``float64``, ``uint8`` and ``uint16``. Scalars and 1-D arrays are
            stored as ``1 x N`` row vectors.

    Returns:
        The file contents. They are deterministic: no timestamp is written.

    Raises:
        TypeError: If an array has an unsupported dtype.
        ValueError: If a name is empty, longer than 63 characters, or not ASCII.
    """
    text = b"MATLAB 5.0 MAT-file, written by ovannotators"
    header = text.ljust(116, b" ") + bytes(8) + struct.pack("<H", 0x0100) + b"IM"
    return header + b"".join(_matrix(name, array) for name, array in variables.items())
