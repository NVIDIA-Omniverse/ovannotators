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

"""Tests for the pure-Python H.264 Annex-B → MP4 muxer."""

from __future__ import annotations

import struct

import pytest

from ovannotators._internal.mp4_mux import mux_h264_annexb

_SC = b"\x00\x00\x00\x01"
_SC3 = b"\x00\x00\x01"
_SPS = b"\x67\x42\x00\x1e\xda\x01\x40"  # type 7, profile 0x42, level 0x1e
_PPS = b"\x68\xce\x3c\x80"  # type 8
_IDR = b"\x65\x88\x84\x00\x21\x00"  # type 5 (IDR slice)
_NON_IDR = b"\x41\x9a\x00\x10"  # type 1 (non-IDR slice)


def _iter_boxes(data: bytes):
    """Yield top-level ``(type, size)`` boxes from an ISO-BMFF byte string."""
    i = 0
    n = len(data)
    while i + 8 <= n:
        size = struct.unpack(">I", data[i : i + 4])[0]
        box_type = data[i + 4 : i + 8]
        yield box_type, size
        if size == 0:
            break
        i += size


def _find_box(data: bytes, box_type: bytes) -> bool:
    return any(t == box_type for t, _ in _iter_boxes(data))


def test_mux_produces_valid_top_level_boxes():
    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR
    mp4 = mux_h264_annexb(stream, fps=30, width=64, height=48)

    boxes = list(_iter_boxes(mp4))
    types = [t for t, _ in boxes]
    assert types[0] == b"ftyp"
    assert b"moov" in types
    assert b"mdat" in types
    # Box sizes should exactly tile the file (no gaps / overruns).
    assert sum(size for _, size in boxes) == len(mp4)


def test_mux_sample_count_matches_access_units():
    # 3 frames: 1 IDR + 2 non-IDR pictures.
    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR + _SC + _NON_IDR + _SC + _NON_IDR
    mp4 = mux_h264_annexb(stream, fps=24, width=32, height=32)

    # stsz carries: version/flags(4) sample_size(4) sample_count(4) sizes...
    idx = mp4.find(b"stsz")
    assert idx != -1
    sample_count = struct.unpack(">I", mp4[idx + 12 : idx + 16])[0]
    assert sample_count == 3


def test_mux_timescale_encodes_fps():
    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR
    fps = 25
    mp4 = mux_h264_annexb(stream, fps=fps, width=16, height=16)

    # mdhd layout (v0): version/flags(4) ctime(4) mtime(4) timescale(4) ...
    idx = mp4.find(b"mdhd")
    assert idx != -1
    timescale = struct.unpack(">I", mp4[idx + 16 : idx + 20])[0]
    sample_delta_rate = timescale  # delta is 1000 → rate = timescale/1000
    assert sample_delta_rate // 1000 == fps


def test_mux_accepts_3byte_start_codes():
    stream = _SC3 + _SPS + _SC3 + _PPS + _SC3 + _IDR
    mp4 = mux_h264_annexb(stream, fps=30, width=64, height=48)
    assert mp4[4:8] == b"ftyp"
    assert _find_box(mp4, b"mdat")


def test_mux_empty_stream_raises():
    with pytest.raises(ValueError, match="empty"):
        mux_h264_annexb(b"", fps=30, width=64, height=48)


def test_mux_stream_without_nal_units_raises():
    # Non-empty bytes but no Annex-B start codes -> no NAL units parsed.
    with pytest.raises(ValueError, match="No NAL units"):
        mux_h264_annexb(b"\xff\xff\xff\xff", fps=24, width=16, height=16)


def test_mux_missing_parameter_sets_raises():
    # Only a slice, no SPS/PPS.
    stream = _SC + _IDR
    with pytest.raises(ValueError, match="SPS/PPS"):
        mux_h264_annexb(stream, fps=30, width=64, height=48)


def test_mux_no_coded_frames_raises():
    # SPS/PPS present but no VCL slice.
    stream = _SC + _SPS + _SC + _PPS
    with pytest.raises(ValueError, match="frames"):
        mux_h264_annexb(stream, fps=30, width=64, height=48)
