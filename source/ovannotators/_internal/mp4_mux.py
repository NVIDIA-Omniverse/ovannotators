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

"""Zero-dependency MP4 muxer for H.264 Annex-B elementary streams.

NVENC (via ``PyNvVideoCodec``) emits a raw H.264 Annex-B elementary stream,
which is not a playable container.

Assumptions (satisfied by the project's encoder configuration):

* The stream contains **no B-frames** (``bf=0``), so decode order equals
  display order and no composition-time (``ctts``) table is required.
* Each access unit holds a single coded picture (one VCL NAL per frame).

These keep the muxer small while producing files that standard players
(VLC, ffmpeg, browsers) accept.
"""

from __future__ import annotations

import struct

__all__ = ["mux_h264_annexb"]

# H.264 NAL unit types.
_NAL_SLICE_NON_IDR = 1
_NAL_SLICE_IDR = 5
_NAL_SEI = 6
_NAL_SPS = 7
_NAL_PPS = 8
_NAL_AUD = 9


def _iter_nal_units(stream: bytes):
    """Yield raw NAL units (without start codes) from an Annex-B byte stream.

    Handles both 3-byte (``00 00 01``) and 4-byte (``00 00 00 01``) start
    codes.

    Args:
        stream: H.264 Annex-B elementary stream.

    Yields:
        bytes: Each NAL unit payload, excluding its start code.
    """
    n = len(stream)
    i = 0
    # Locate the first start code.
    starts: list[int] = []
    while i + 3 <= n:
        if stream[i] == 0 and stream[i + 1] == 0:
            if stream[i + 2] == 1:
                starts.append((i, 3))
                i += 3
                continue
            if i + 4 <= n and stream[i + 2] == 0 and stream[i + 3] == 1:
                starts.append((i, 4))
                i += 4
                continue
        i += 1

    for idx, (pos, sc_len) in enumerate(starts):
        nal_start = pos + sc_len
        nal_end = starts[idx + 1][0] if idx + 1 < len(starts) else n
        nal = stream[nal_start:nal_end]
        if nal:
            yield nal


def _group_access_units(nals: list[bytes]) -> list[list[bytes]]:
    """Group NAL units into access units (one coded picture each).

    A new access unit begins at the first VCL NAL of a picture, and any
    leading parameter-set / SEI / AUD NALs are attached to the picture that
    follows them.

    Args:
        nals: Ordered NAL units from the elementary stream.

    Returns:
        list[list[bytes]]: Access units, each a list of NAL units.
    """
    access_units: list[list[bytes]] = []
    current: list[bytes] = []
    seen_vcl = False

    for nal in nals:
        nal_type = nal[0] & 0x1F
        is_vcl = _NAL_SLICE_NON_IDR <= nal_type <= _NAL_SLICE_IDR
        boundary = nal_type in (_NAL_AUD, _NAL_SEI, _NAL_SPS, _NAL_PPS) or is_vcl
        if seen_vcl and boundary:
            access_units.append(current)
            current = []
            seen_vcl = False
        current.append(nal)
        if is_vcl:
            seen_vcl = True

    if current:
        access_units.append(current)
    return access_units


def _box(box_type: bytes, payload: bytes) -> bytes:
    """Build an ISO-BMFF box: ``size(4) + type(4) + payload``."""
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def _full_box(box_type: bytes, version: int, flags: int, payload: bytes) -> bytes:
    """Build a FullBox (version + 24-bit flags prefix)."""
    return _box(box_type, struct.pack(">I", (version << 24) | flags) + payload)


def _avcc(sps: bytes, pps: bytes) -> bytes:
    """Build an ``avcC`` AVCDecoderConfigurationRecord from SPS/PPS."""
    # profile_idc, profile_compatibility, level_idc come from the SPS payload.
    profile_idc = sps[1]
    profile_compat = sps[2]
    level_idc = sps[3]
    payload = bytes(
        [
            1,  # configurationVersion
            profile_idc,
            profile_compat,
            level_idc,
            0xFF,  # 6 bits reserved (1) + lengthSizeMinusOne = 3 (4-byte NAL len)
            0xE1,  # 3 bits reserved (1) + numOfSequenceParameterSets = 1
        ]
    )
    payload += struct.pack(">H", len(sps)) + sps
    payload += bytes([1])  # numOfPictureParameterSets
    payload += struct.pack(">H", len(pps)) + pps
    return _box(b"avcC", payload)


def _avc1(width: int, height: int, sps: bytes, pps: bytes) -> bytes:
    """Build the ``avc1`` sample entry (video sample description)."""
    payload = bytes(6)  # reserved
    payload += struct.pack(">H", 1)  # data_reference_index
    payload += bytes(16)  # pre_defined + reserved
    payload += struct.pack(">HH", width, height)
    payload += struct.pack(">II", 0x00480000, 0x00480000)  # 72 dpi h/v res
    payload += struct.pack(">I", 0)  # reserved
    payload += struct.pack(">H", 1)  # frame_count
    payload += bytes(32)  # compressorname (empty)
    payload += struct.pack(">H", 0x0018)  # depth
    payload += struct.pack(">h", -1)  # pre_defined
    payload += _avcc(sps, pps)
    return _box(b"avc1", payload)


def _stbl(
    width: int,
    height: int,
    sps: bytes,
    pps: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    """Build the ``stbl`` sample table."""
    stsd = _full_box(
        b"stsd", 0, 0, struct.pack(">I", 1) + _avc1(width, height, sps, pps)
    )

    n = len(sample_sizes)
    stts = _full_box(
        b"stts", 0, 0, struct.pack(">I", 1) + struct.pack(">II", n, sample_delta)
    )

    stsc = _full_box(b"stsc", 0, 0, struct.pack(">I", 1) + struct.pack(">III", 1, n, 1))

    stsz_payload = struct.pack(">II", 0, n) + b"".join(
        struct.pack(">I", s) for s in sample_sizes
    )
    stsz = _full_box(b"stsz", 0, 0, stsz_payload)

    stco = _full_box(
        b"stco", 0, 0, struct.pack(">I", 1) + struct.pack(">I", chunk_offset)
    )

    children = stsd + stts
    if sync_samples and len(sync_samples) != n:
        stss = _full_box(
            b"stss",
            0,
            0,
            struct.pack(">I", len(sync_samples))
            + b"".join(struct.pack(">I", s) for s in sync_samples),
        )
        children += stss
    children += stsc + stsz + stco
    return _box(b"stbl", children)


def _minf(
    width: int,
    height: int,
    sps: bytes,
    pps: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    vmhd = _full_box(b"vmhd", 0, 1, struct.pack(">HHHH", 0, 0, 0, 0))
    url = _full_box(b"url ", 0, 1, b"")  # self-contained
    dref = _full_box(b"dref", 0, 0, struct.pack(">I", 1) + url)
    dinf = _box(b"dinf", dref)
    stbl = _stbl(
        width,
        height,
        sps,
        pps,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    return _box(b"minf", vmhd + dinf + stbl)


def _mdia(
    timescale: int,
    duration: int,
    width: int,
    height: int,
    sps: bytes,
    pps: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    mdhd = _full_box(
        b"mdhd",
        0,
        0,
        struct.pack(">IIII", 0, 0, timescale, duration)
        + struct.pack(">HH", 0x55C4, 0),  # language 'und', pre_defined
    )
    hdlr = _full_box(
        b"hdlr",
        0,
        0,
        struct.pack(">I", 0) + b"vide" + bytes(12) + b"VideoHandler\x00",
    )
    minf = _minf(
        width,
        height,
        sps,
        pps,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    return _box(b"mdia", mdhd + hdlr + minf)


def _trak(
    timescale: int,
    duration: int,
    width: int,
    height: int,
    sps: bytes,
    pps: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    tkhd = _full_box(
        b"tkhd",
        0,
        0x7,  # track enabled | in movie | in preview
        struct.pack(">IIIII", 0, 0, 1, 0, duration)
        + bytes(8)  # reserved
        + struct.pack(">hhhh", 0, 0, 0, 0)  # layer, alt_group, volume, reserved
        + _identity_matrix()
        + struct.pack(">II", width << 16, height << 16),
    )
    mdia = _mdia(
        timescale,
        duration,
        width,
        height,
        sps,
        pps,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    return _box(b"trak", tkhd + mdia)


def _identity_matrix() -> bytes:
    """ISO-BMFF unity transformation matrix."""
    return struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)


def _moov(
    timescale: int,
    duration: int,
    width: int,
    height: int,
    sps: bytes,
    pps: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    mvhd = _full_box(
        b"mvhd",
        0,
        0,
        struct.pack(">IIII", 0, 0, timescale, duration)
        + struct.pack(">I", 0x00010000)  # rate 1.0
        + struct.pack(">H", 0x0100)  # volume 1.0
        + bytes(10)  # reserved
        + _identity_matrix()
        + bytes(24)  # pre_defined
        + struct.pack(">I", 2),  # next_track_ID
    )
    trak = _trak(
        timescale,
        duration,
        width,
        height,
        sps,
        pps,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    return _box(b"moov", mvhd + trak)


def mux_h264_annexb(stream: bytes, fps: int, width: int, height: int) -> bytes:
    """Wrap an H.264 Annex-B elementary stream in an MP4 container.

    Args:
        stream: Raw H.264 Annex-B bitstream (start-code delimited NAL units),
            with no B-frames so decode order equals display order.
        fps: Frame rate; used to derive sample durations.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        bytes: A self-contained, playable MP4 file.

    Raises:
        ValueError: If the stream is empty or contains no SPS/PPS or no coded
            frames.
    """
    if not stream:
        raise ValueError("Cannot mux an empty H.264 stream")

    nals = list(_iter_nal_units(stream))
    if not nals:
        raise ValueError("No NAL units found in H.264 stream")

    sps = next((n for n in nals if (n[0] & 0x1F) == _NAL_SPS), None)
    pps = next((n for n in nals if (n[0] & 0x1F) == _NAL_PPS), None)
    if sps is None or pps is None:
        raise ValueError("H.264 stream is missing SPS/PPS parameter sets")

    access_units = _group_access_units(nals)

    # Build per-sample byte payloads (length-prefixed NAL units) and metadata.
    sample_payloads: list[bytes] = []
    sample_sizes: list[int] = []
    sync_samples: list[int] = []
    for idx, au in enumerate(access_units):
        # Only count access units that actually carry a coded picture.
        if not any(_NAL_SLICE_NON_IDR <= (n[0] & 0x1F) <= _NAL_SLICE_IDR for n in au):
            continue
        payload = b"".join(struct.pack(">I", len(n)) + n for n in au)
        sample_payloads.append(payload)
        sample_sizes.append(len(payload))
        if any((n[0] & 0x1F) == _NAL_SLICE_IDR for n in au):
            sync_samples.append(len(sample_payloads))  # 1-based index

    if not sample_payloads:
        raise ValueError("No coded video frames found in H.264 stream")

    # Use a 1000x timescale so integer per-frame durations express fps exactly.
    timescale = int(fps) * 1000
    sample_delta = 1000
    duration = sample_delta * len(sample_sizes)

    mdat_payload = b"".join(sample_payloads)
    ftyp = _box(
        b"ftyp",
        b"isom" + struct.pack(">I", 0x200) + b"isom" + b"iso2" + b"avc1" + b"mp41",
    )

    # The chunk offset (into mdat data) depends on the moov size, which is
    # independent of the offset *value*, so build once to size it, then rebuild
    # with the correct offset.
    placeholder = _moov(
        timescale,
        duration,
        width,
        height,
        sps,
        pps,
        sample_sizes,
        sync_samples,
        sample_delta,
        0,
    )
    chunk_offset = len(ftyp) + len(placeholder) + 8  # +8 for the mdat header
    moov = _moov(
        timescale,
        duration,
        width,
        height,
        sps,
        pps,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    mdat = _box(b"mdat", mdat_payload)
    return ftyp + moov + mdat
