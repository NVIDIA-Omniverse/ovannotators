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

"""Tests for the minimal MAT v5 writer in :mod:`ovannotators._internal.mat5`.

``_read`` parses files straight from the MAT-file Level 5 layout, independently of
the writer, so the always-on tests need no extra dependency. The scipy test
cross-checks against ``scipy.io.loadmat`` when scipy is installed.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from ovannotators._internal.mat5 import encode_mat5

_DTYPES = {1: "i1", 2: "u1", 4: "<u2", 5: "<i4", 6: "<u4", 9: "<f8"}
_CLASSES = {6: "<f8", 9: "u1", 11: "<u2"}


def _elements(buf: bytes):
    """Yield ``(data_type, payload)`` for each data element in ``buf``."""
    offset = 0
    while offset < len(buf):
        (word,) = struct.unpack_from("<I", buf, offset)
        if word >> 16:  # small data element: size in the upper half of the tag
            size, data_type = word >> 16, word & 0xFFFF
            yield data_type, buf[offset + 4 : offset + 4 + size]
            offset += 8
        else:
            size = struct.unpack_from("<I", buf, offset + 4)[0]
            yield word, buf[offset + 8 : offset + 8 + size]
            offset += 8 + (size + 7) // 8 * 8


def _read(data: bytes) -> dict[str, np.ndarray]:
    assert data[126:128] == b"IM"  # little-endian marker
    assert struct.unpack_from("<H", data, 124)[0] == 0x0100  # version
    assert (len(data) - 128) % 8 == 0  # elements are 8-byte aligned
    variables = {}
    for data_type, payload in _elements(data[128:]):
        assert data_type == 14  # miMATRIX
        flags, dims, name, real = (p for _, p in _elements(payload))
        array_class = struct.unpack_from("<I", flags)[0] & 0xFF
        shape = struct.unpack(f"<{len(dims) // 4}i", dims)
        values = np.frombuffer(real, dtype=_CLASSES[array_class])
        variables[name.decode()] = values.reshape(shape, order="F")
    return variables


def test_round_trips_every_supported_dtype_and_shape() -> None:
    poses = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
    variables = {
        "poses": poses,
        "intrinsic_matrix": np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]]),
        "cls_indexes": np.array([2, 5, 21], dtype=np.uint8),
        "factor_depth": np.array([10000], dtype=np.uint16),
        "center": np.array([[1.5, 2.5], [3.5, 4.5]]),
    }
    decoded = _read(encode_mat5(variables))
    assert list(decoded) == list(variables)
    np.testing.assert_array_equal(decoded["poses"], poses)
    np.testing.assert_array_equal(
        decoded["intrinsic_matrix"], variables["intrinsic_matrix"]
    )
    np.testing.assert_array_equal(decoded["center"], variables["center"])
    # 1-D arrays are row vectors, as with scipy.io.savemat.
    assert decoded["cls_indexes"].shape == (1, 3)
    assert decoded["cls_indexes"].tolist() == [[2, 5, 21]]
    assert decoded["factor_depth"].shape == (1, 1)
    assert decoded["factor_depth"].dtype == np.dtype("<u2")


def test_arrays_are_stored_column_major() -> None:
    decoded = _read(encode_mat5({"m": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}))
    np.testing.assert_array_equal(decoded["m"], [[1, 2, 3], [4, 5, 6]])
    raw = encode_mat5({"m": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})[-48:]
    assert np.frombuffer(raw, "<f8").tolist() == [1, 4, 2, 5, 3, 6]


def test_header_is_fixed_size_text_and_deterministic() -> None:
    data = encode_mat5({"a": np.zeros(2)})
    assert data[:20] == b"MATLAB 5.0 MAT-file,"
    assert data[116:124] == bytes(8)  # no subsystem data
    assert data == encode_mat5({"a": np.zeros(2)})


def test_long_and_short_names_use_the_right_element_layout() -> None:
    names = ["a", "abcd", "abcde", "x" * 63]
    decoded = _read(encode_mat5({name: np.arange(3.0) for name in names}))
    assert list(decoded) == names


def test_rejects_unsupported_dtype_and_bad_names() -> None:
    with pytest.raises(TypeError, match="float32"):
        encode_mat5({"a": np.zeros(2, dtype=np.float32)})
    with pytest.raises(ValueError, match="name"):
        encode_mat5({"": np.zeros(2)})
    with pytest.raises(ValueError, match="name"):
        encode_mat5({"x" * 64: np.zeros(2)})


def test_scipy_loadmat_reads_the_file() -> None:
    scipy_io = pytest.importorskip("scipy.io")
    poses = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
    variables = {
        "poses": poses,
        "cls_indexes": np.array([2, 5], dtype=np.uint8),
        "factor_depth": np.array([10000], dtype=np.uint16),
        "intrinsic_matrix": np.eye(3),
    }
    loaded = scipy_io.loadmat(io.BytesIO(encode_mat5(variables)))
    np.testing.assert_array_equal(loaded["poses"], poses)
    assert (
        loaded["cls_indexes"].shape == (1, 2)
        and loaded["cls_indexes"].dtype == np.uint8
    )
    assert loaded["factor_depth"].tolist() == [[10000]]
    assert loaded["factor_depth"].dtype == np.uint16
    # Same layout scipy itself writes for these variables.
    reference = io.BytesIO()
    scipy_io.savemat(reference, variables)
    reloaded = scipy_io.loadmat(io.BytesIO(reference.getvalue()))
    for name in variables:
        np.testing.assert_array_equal(loaded[name], reloaded[name])
        assert loaded[name].dtype == reloaded[name].dtype
        assert loaded[name].shape == reloaded[name].shape
