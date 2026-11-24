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

"""Zero-dependency MP4 muxer and container utilities for H.264, HEVC, and AV1.

NVENC (via ``PyNvVideoCodec``) emits a raw Annex-B elementary stream (H.264 /
HEVC) or an AV1 OBU / IVF bitstream, which is not a playable container.

Assumptions (satisfied by the project's encoder configuration):

* The stream contains **no B-frames** (``bf=0``), so decode order equals
  display order and no composition-time (``ctts``) table is required.
* H.264 uses one VCL NAL per picture. HEVC also supports multiple slices.
* AV1 uses low-overhead OBU framing (or IVF-wrapped OBUs) with no B-frames.

These keep the muxer small. HEVC 4:4:4 requires a compatible decoder.
"""

from __future__ import annotations

import struct
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any

__all__ = [
    "concatenate_mp4_tracks",
    "mp4_duration_s",
    "mux_av1_obu",
    "mux_h264_annexb",
    "mux_hevc_annexb",
    "parse_mp4_video_index",
]

# H.264 NAL unit types.
_NAL_SLICE_NON_IDR = 1
_NAL_SLICE_IDR = 5
_NAL_SEI = 6
_NAL_SPS = 7
_NAL_PPS = 8
_NAL_AUD = 9

# ISO/IEC 14496-12 VisualSampleEntry: an 8-byte box header, then 78 bytes of
# fixed fields before any codec configuration box.
_VISUAL_SAMPLE_ENTRY_SIZE = 8 + 78
_MAX_INDEXED_SAMPLES = 1_000_000


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
    starts: list[tuple[int, int]] = []
    # bytes.find scans in C; a byte-at-a-time Python loop cost ~35 ms per MB
    # of encoded video at finalize. Every start code ends in ``00 00 01``; it
    # is a 4-byte code when the byte before it (not yet consumed) is also 0.
    while True:
        pos = stream.find(b"\x00\x00\x01", i)
        if pos < 0:
            break
        if pos > i and stream[pos - 1] == 0:
            starts.append((pos - 1, 4))
        else:
            starts.append((pos, 3))
        i = pos + 3

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


_MAX_32BIT_BOX_SIZE = 0xFFFFFFFF
_MAX_32BIT_DURATION = 0xFFFFFFFF


def _box(box_type: bytes, payload: bytes) -> bytes:
    """Build an ISO-BMFF box: ``size(4) + type(4) + payload``."""
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def _full_box(
    box_type: bytes, version: int, flags: int, payload: bytes, suffix: bytes = b""
) -> bytes:
    """Build a FullBox (version + 24-bit flags prefix)."""
    return _box(box_type, struct.pack(">I", (version << 24) | flags) + payload + suffix)


def _versioned_times(timescale: int, duration: int) -> tuple[int, int, bytes]:
    """Return ``(version, flags, times)`` for ``mvhd``/``mdhd``.

    Version 1 carries 64-bit creation, modification, and duration fields and
    is used only when ``duration`` does not fit in 32 bits.
    """
    if duration > _MAX_32BIT_DURATION:
        return 1, 0, struct.pack(">QQIQ", 0, 0, timescale, duration)
    return 0, 0, struct.pack(">IIII", 0, 0, timescale, duration)


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
    return _visual_entry(b"avc1", width, height, _avcc(sps, pps))


def _visual_entry(kind: bytes, width: int, height: int, config: bytes) -> bytes:
    """Build a visual sample entry with codec-specific configuration."""
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
    return _box(kind, payload + config)


def _stbl(
    sample_entry: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    """Build the ``stbl`` sample table.

    Raises:
        ValueError: If there are samples but no sync samples. An empty ``stss``
            marks every frame unseekable, and omitting it marks every frame
            seekable, so neither describes a stream whose keyframes were not
            detected.
    """
    if sample_sizes and not sync_samples:
        raise ValueError("video stream has no detected keyframes (sync samples)")
    stsd = _full_box(b"stsd", 0, 0, struct.pack(">I", 1) + sample_entry)

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
    if len(sync_samples) != n:
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
    sample_entry: bytes,
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
        sample_entry,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    return _box(b"minf", vmhd + dinf + stbl)


def _mdia(
    timescale: int,
    duration: int,
    sample_entry: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    mdhd = _full_box(
        b"mdhd",
        *_versioned_times(timescale, duration),
        struct.pack(">HH", 0x55C4, 0),  # language 'und', pre_defined
    )
    hdlr = _full_box(
        b"hdlr",
        0,
        0,
        struct.pack(">I", 0) + b"vide" + bytes(12) + b"VideoHandler\x00",
    )
    minf = _minf(
        sample_entry,
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
    sample_entry: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    long_duration = duration > _MAX_32BIT_DURATION
    tkhd = _full_box(
        b"tkhd",
        1 if long_duration else 0,
        0x7,  # track enabled | in movie | in preview
        (
            struct.pack(">QQIIQ", 0, 0, 1, 0, duration)
            if long_duration
            else struct.pack(">IIIII", 0, 0, 1, 0, duration)
        )
        + bytes(8)  # reserved
        + struct.pack(">hhhh", 0, 0, 0, 0)  # layer, alt_group, volume, reserved
        + _identity_matrix()
        + struct.pack(">II", width << 16, height << 16),
    )
    mdia = _mdia(
        timescale,
        duration,
        sample_entry,
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
    sample_entry: bytes,
    sample_sizes: list[int],
    sync_samples: list[int],
    sample_delta: int,
    chunk_offset: int,
) -> bytes:
    mvhd = _full_box(
        b"mvhd",
        *_versioned_times(timescale, duration),
        struct.pack(">I", 0x00010000)  # rate 1.0
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
        sample_entry,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    return _box(b"moov", mvhd + trak)


def mux_h264_annexb(stream: bytes, fps: float, width: int, height: int) -> bytes:
    """Wrap an H.264 Annex-B elementary stream in an MP4 container.

    Args:
        stream: Raw H.264 Annex-B bitstream (start-code delimited NAL units),
            with no B-frames so decode order equals display order.
        fps: Frame rate; used to derive sample durations. Fractional rates
            such as 29.97 are exact.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        bytes: A self-contained, playable MP4 file.

    Raises:
        ValueError: If the stream is empty or contains no SPS/PPS, no coded
            frames, or no IDR frame.
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
    sync_samples: list[int] = []
    for au in access_units:
        # Only count access units that actually carry a coded picture.
        if not any(_NAL_SLICE_NON_IDR <= (n[0] & 0x1F) <= _NAL_SLICE_IDR for n in au):
            continue
        payload = b"".join(struct.pack(">I", len(n)) + n for n in au)
        sample_payloads.append(payload)
        if any((n[0] & 0x1F) == _NAL_SLICE_IDR for n in au):
            sync_samples.append(len(sample_payloads))  # 1-based index

    if not sample_payloads:
        raise ValueError("No coded video frames found in H.264 stream")

    return _mux_samples(
        sample_payloads,
        sync_samples,
        _sample_timing(fps),
        width,
        height,
        _avc1(width, height, sps, pps),
        b"avc1",
    )


def _sample_timing(fps: float) -> tuple[int, int]:
    from ovannotators._internal.video_config import video_frame_rate

    rate = video_frame_rate(fps)
    if rate.denominator == 1:
        return rate.numerator * 1000, 1000
    return rate.numerator, rate.denominator


def _mux_samples(
    samples: list[bytes],
    sync_samples: list[int],
    timing: tuple[int, int],
    width: int,
    height: int,
    sample_entry: bytes,
    brand: bytes,
) -> bytes:
    """Mux ordered coded pictures using a codec-specific sample entry."""
    if not samples:
        raise ValueError("No coded video frames found")
    sample_sizes = [len(sample) for sample in samples]
    timescale, sample_delta = timing
    duration = sample_delta * len(sample_sizes)

    mdat_payload = b"".join(samples)
    ftyp = _box(
        b"ftyp",
        b"isom" + struct.pack(">I", 0x200) + b"isom" + b"iso2" + brand + b"mp41",
    )

    # The chunk offset (into mdat data) depends on the moov size, which is
    # independent of the offset *value*, so build once to size it, then rebuild
    # with the correct offset.
    placeholder = _moov(
        timescale,
        duration,
        width,
        height,
        sample_entry,
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
        sample_entry,
        sample_sizes,
        sync_samples,
        sample_delta,
        chunk_offset,
    )
    if 8 + len(mdat_payload) > _MAX_32BIT_BOX_SIZE:
        # ISO-BMFF largesize header; the single chunk still starts below 4 GiB
        # because ``moov`` precedes ``mdat``, so ``stco`` stays 32-bit.
        moov = _moov(
            timescale,
            duration,
            width,
            height,
            sample_entry,
            sample_sizes,
            sync_samples,
            sample_delta,
            chunk_offset + 8,
        )
        mdat = struct.pack(">I4sQ", 1, b"mdat", 16 + len(mdat_payload)) + mdat_payload
    else:
        mdat = _box(b"mdat", mdat_payload)
    return ftyp + moov + mdat


class _Bits:
    """Bounded HEVC RBSP reader, including unsigned Exp-Golomb values."""

    def __init__(self, nal: bytes):
        rbsp = bytearray()
        zeros = 0
        for value in nal[2:]:
            if zeros == 2 and value == 3:
                zeros = 0
                continue
            rbsp.append(value)
            zeros = zeros + 1 if value == 0 else 0
        self.data = bytes(rbsp)
        self.pos = 0

    def read(self, count: int) -> int:
        if self.pos + count > len(self.data) * 8:
            raise ValueError("Truncated HEVC parameter set")
        value = 0
        for _ in range(count):
            value = value * 2 + ((self.data[self.pos // 8] >> (7 - self.pos % 8)) & 1)
            self.pos += 1
        return value

    def ue(self) -> int:
        zeros = 0
        while self.read(1) == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("Invalid HEVC Exp-Golomb value")
        return (1 << zeros) - 1 + self.read(zeros)


def _hvcc(parameter_sets: dict[int, bytes]) -> bytes:
    """Build hvcC from the SPS profile, chroma/bit depth and VPS/SPS/PPS."""
    bits = _Bits(parameter_sets[33])
    bits.read(4)  # sps_video_parameter_set_id
    layers = bits.read(3)
    nested = bits.read(1)
    profile = bits.read(96).to_bytes(12, "big")
    sublayers = [(bits.read(1), bits.read(1)) for _ in range(layers)]
    if layers:
        bits.read(2 * (8 - layers))
    for has_profile, has_level in sublayers:
        if has_profile:
            bits.read(88)
        if has_level:
            bits.read(8)
    bits.ue()  # sps_seq_parameter_set_id
    chroma = bits.ue()
    if chroma > 3:
        raise ValueError("Invalid HEVC chroma format")
    if chroma == 3:
        bits.read(1)  # separate_colour_plane_flag
    bits.ue()  # pic_width_in_luma_samples
    bits.ue()  # pic_height_in_luma_samples
    if bits.read(1):
        for _ in range(4):
            bits.ue()  # conformance window
    luma_depth, chroma_depth = bits.ue(), bits.ue()
    if max(luma_depth, chroma_depth) > 7:
        raise ValueError("Unsupported HEVC bit depth")
    payload = (
        b"\x01"
        + profile
        + struct.pack(
            ">HBBBBHB",
            0xF000,
            0xFC,
            0xFC | chroma,
            0xF8 | luma_depth,
            0xF8 | chroma_depth,
            0,
            ((layers + 1) << 3) | (nested << 2) | 3,
        )
        + b"\x03"
    )
    for kind, nal in parameter_sets.items():
        payload += struct.pack(">BHH", 0x80 | kind, 1, len(nal)) + nal
    return _box(b"hvcC", payload)


def _signal_rgb(sps: bytes) -> bytes:
    """Signal identity-matrix, full-range GBR in an NVENC HEVC SPS.

    PyNvVideoCodec does not expose these VUI fields. Only the simple SPS
    emitted by our encoder is accepted; unexpected syntax fails closed.
    """
    bits = _Bits(sps)
    bits.read(4)
    if bits.read(3):
        raise ValueError("RGB HEVC requires a single temporal layer")
    bits.read(1 + 96)
    bits.ue()
    if bits.ue() != 3 or bits.read(1):
        raise ValueError("RGB HEVC requires non-separate 4:4:4 planes")
    bits.ue()
    bits.ue()
    if bits.read(1):
        for _ in range(4):
            bits.ue()
    if bits.ue() or bits.ue():
        raise ValueError("RGB HEVC requires 8-bit planes")
    log2_poc = bits.ue() + 4
    bits.read(1)
    for _ in range(3 + 6):
        bits.ue()  # sublayer ordering, coding/transform block sizes
    if bits.read(1):
        raise ValueError("RGB HEVC does not support SPS scaling lists")
    bits.read(2)  # AMP, SAO
    if bits.read(1):
        raise ValueError("RGB HEVC does not support PCM")
    delta_pocs = 0
    for index in range(bits.ue()):
        if index and bits.read(1):
            bits.read(1)  # delta_rps_sign
            bits.ue()  # abs_delta_rps_minus1
            count = 0
            for _ in range(delta_pocs + 1):
                used = bits.read(1)
                count += bool(used or bits.read(1))
            delta_pocs = count
        else:
            negative, positive = bits.ue(), bits.ue()
            delta_pocs = negative + positive
            for _ in range(delta_pocs):
                bits.ue()
                bits.read(1)
    if bits.read(1):
        for _ in range(bits.ue()):
            bits.read(log2_poc + 1)
    bits.read(2)  # temporal MVP, strong intra smoothing
    if not bits.read(1):
        raise ValueError("RGB HEVC requires VUI parameters")
    if bits.read(1):  # aspect ratio
        if bits.read(8) == 255:
            bits.read(32)
    if bits.read(1):  # overscan
        bits.read(1)
    start = bits.pos
    if bits.read(1):  # existing video signal type
        bits.read(4)
        if bits.read(1):
            bits.read(24)
    binary = "".join(f"{value:08b}" for value in bits.data).rstrip("0")
    # present, unspecified video format, full range, colour description;
    # BT.709 primaries, sRGB transfer, identity matrix.
    signal = "110111" + f"{1:08b}{13:08b}{0:08b}"
    binary = binary[:start] + signal + binary[bits.pos :]
    binary += "0" * (-len(binary) % 8)
    rbsp = int(binary, 2).to_bytes(len(binary) // 8, "big")
    escaped = bytearray(sps[:2])
    zeros = 0
    for value in rbsp:
        if zeros == 2 and value <= 3:
            escaped.append(3)
            zeros = 0
        escaped.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return bytes(escaped)


def mux_hevc_annexb(
    stream: bytes, fps: float, width: int, height: int, *, rgb: bool = False
) -> bytes:
    """Mux single-layer HEVC without B-frames into an hvc1 MP4.

    Args:
        stream: Annex-B encoded pictures with VPS, SPS and PPS.
        fps: Positive frame rate; fractional rates such as 29.97 are exact.
        width: Display width.
        height: Display height.
        rgb: Signal full-range identity-matrix GBR in the SPS and container.

    Returns:
        bytes: MP4 with parameter sets stored exclusively in hvcC.

    Raises:
        ValueError: On missing, truncated or changing parameter sets.
    """
    if min(fps, width, height) <= 0:
        raise ValueError("Frame rate and dimensions must be positive")
    parameters: dict[int, bytes] = {}
    pictures: list[list[bytes]] = []
    current: list[bytes] = []
    seen_vcl = False
    for nal in _iter_nal_units(stream):
        if len(nal) < 3 or nal[0] & 0x80 or (nal[1] & 7) == 0:
            raise ValueError("Invalid HEVC NAL header")
        if (nal[0] & 1) or (nal[1] >> 3):
            raise ValueError("Multilayer HEVC is unsupported")
        kind = (nal[0] >> 1) & 63
        vcl = kind < 32
        first_slice = vcl and bool(nal[2] & 0x80)
        if seen_vcl and (first_slice or kind in (32, 33, 34, 35, 39)):
            pictures.append(current)
            current, seen_vcl = [], False
        if kind in (32, 33, 34):
            if kind in parameters and parameters[kind] != nal:
                raise ValueError("Changing HEVC parameter sets are unsupported")
            parameters[kind] = nal
            continue
        if vcl and not seen_vcl and not first_slice:
            raise ValueError("HEVC picture is missing its first slice")
        current.append(nal)
        seen_vcl |= vcl
    if seen_vcl:
        pictures.append(current)
    if set(parameters) != {32, 33, 34}:
        raise ValueError("HEVC stream is missing VPS/SPS/PPS parameter sets")
    if rgb:
        parameters[33] = _signal_rgb(parameters[33])
    config = _hvcc({kind: parameters[kind] for kind in (32, 33, 34)})
    if rgb:
        # BT.709 primaries, sRGB transfer, identity matrix, full range.
        config += _box(b"colr", b"nclx" + struct.pack(">HHHB", 1, 13, 0, 0x80))
    samples = [b"".join(struct.pack(">I", len(n)) + n for n in au) for au in pictures]
    sync = [
        i
        for i, au in enumerate(pictures, 1)
        if any(16 <= ((n[0] >> 1) & 63) <= 21 for n in au)
    ]
    return _mux_samples(
        samples,
        sync,
        _sample_timing(fps),
        width,
        height,
        _visual_entry(b"hvc1", width, height, config),
        b"hvc1",
    )


# ---------------------------------------------------------------------------
# AV1 OBU -> MP4 muxing
# ---------------------------------------------------------------------------

_OBU_SEQUENCE_HEADER = 1
_OBU_TEMPORAL_DELIMITER = 2
_OBU_FRAME_HEADER = 3
_OBU_TILE_GROUP = 4
_OBU_FRAME = 6


class _RawBits:
    """Bounded MSB-first bitstream reader without emulation-prevention removal."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, count: int) -> int:
        if self.pos + count > len(self.data) * 8:
            raise ValueError("Truncated AV1 sequence header")
        value = 0
        for _ in range(count):
            value = (value << 1) | (
                (self.data[self.pos // 8] >> (7 - self.pos % 8)) & 1
            )
            self.pos += 1
        return value

    def uvlc(self) -> int:
        zeros = 0
        while True:
            done = self.read(1)
            if done:
                break
            zeros += 1
            if zeros >= 32:
                return (1 << 32) - 1
        return self.read(zeros) + (1 << zeros) - 1


def _read_leb128(data: bytes, offset: int) -> tuple[int, int]:
    """Decode an unsigned LEB128 integer; return ``(value, next_offset)``."""
    value = 0
    for i in range(8):
        if offset + i >= len(data):
            raise ValueError("Truncated LEB128 OBU size")
        byte = data[offset + i]
        value |= (byte & 0x7F) << (7 * i)
        if not (byte & 0x80):
            return value, offset + i + 1
    raise ValueError("Invalid LEB128 OBU size (> 8 bytes)")


def _encode_leb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _strip_ivf_wrapper(stream: bytes) -> list[bytes] | None:
    """If ``stream`` is an IVF container, return per-frame OBU byte chunks."""
    if len(stream) < 32 or stream[:4] != b"DKIF":
        return None
    header_size = struct.unpack_from("<H", stream, 6)[0]
    if header_size < 32 or header_size > len(stream):
        raise ValueError("Invalid IVF header size")
    frames: list[bytes] = []
    pos = header_size
    while pos < len(stream):
        if pos + 12 > len(stream):
            raise ValueError("Truncated IVF frame header")
        frame_size = struct.unpack_from("<I", stream, pos)[0]
        pos += 12
        if pos + frame_size > len(stream):
            raise ValueError("Truncated IVF frame payload")
        frames.append(stream[pos : pos + frame_size])
        pos += frame_size
    return frames


def _iter_obus(data: bytes) -> list[tuple[int, bytes, bytes]]:
    """Parse OBUs from a low-overhead AV1 OBU byte stream.

    Returns a list of ``(obu_type, raw_obu_with_size, payload)`` tuples. If an
    input OBU omits ``obu_has_size_field`` (trailing OBU in a unit), the
    returned ``raw_obu_with_size`` re-encodes the header with
    ``obu_has_size_field=1`` as required by ISO-BMFF AV1 samples.
    """
    obus: list[tuple[int, bytes, bytes]] = []
    pos = 0
    n = len(data)
    while pos < n:
        header = data[pos]
        if header & 0x80:
            raise ValueError("Invalid AV1 OBU header (forbidden bit set)")
        obu_type = (header >> 3) & 0x0F
        has_extension = bool((header >> 2) & 0x01)
        has_size = bool((header >> 1) & 0x01)
        start = pos
        pos += 1
        ext_byte = b""
        if has_extension:
            if pos >= n:
                raise ValueError("Truncated AV1 OBU extension header")
            ext_byte = data[pos : pos + 1]
            pos += 1
        if has_size:
            size, pos = _read_leb128(data, pos)
            if pos + size > n:
                raise ValueError("Truncated AV1 OBU payload")
            payload = data[pos : pos + size]
            raw = data[start : pos + size]
            pos += size
        else:
            payload = data[pos:n]
            raw = (
                bytes([header | 0x02])
                + ext_byte
                + _encode_leb128(len(payload))
                + payload
            )
            pos = n
        obus.append((obu_type, raw, payload))
    return obus


def _parse_av1_seq_header(payload: bytes) -> dict[str, int]:
    """Parse fields required for ``av1C`` from an ``OBU_SEQUENCE_HEADER``."""
    bits = _RawBits(payload)
    seq_profile = bits.read(3)
    bits.read(1)  # still_picture
    reduced = bits.read(1)
    if reduced:
        seq_level_idx_0 = bits.read(5)
        seq_tier_0 = 0
    else:
        timing_info = bits.read(1)
        decoder_model_info = 0
        buffer_delay_len = 0
        if timing_info:
            bits.read(32)  # num_units_in_display_tick
            bits.read(32)  # time_scale
            if bits.read(1):  # equal_picture_interval
                bits.uvlc()
            decoder_model_info = bits.read(1)
            if decoder_model_info:
                buffer_delay_len = bits.read(5) + 1
                bits.read(32)  # num_units_in_decoding_tick
                bits.read(5)  # buffer_removal_time_length_minus_1
                bits.read(5)  # frame_presentation_time_length_minus_1
        initial_display_delay = bits.read(1)
        op_cnt = bits.read(5) + 1
        seq_level_idx_0 = 0
        seq_tier_0 = 0
        for idx in range(op_cnt):
            bits.read(12)  # operating_point_idc
            level = bits.read(5)
            tier = bits.read(1) if level > 7 else 0
            if idx == 0:
                seq_level_idx_0, seq_tier_0 = level, tier
            if decoder_model_info and bits.read(1):
                bits.read(buffer_delay_len)
                bits.read(buffer_delay_len)
                bits.read(1)
            if initial_display_delay and bits.read(1):
                bits.read(4)

    w_bits = bits.read(4) + 1
    h_bits = bits.read(4) + 1
    bits.read(w_bits)  # max_frame_width_minus_1
    bits.read(h_bits)  # max_frame_height_minus_1
    if not reduced:
        if bits.read(1):  # frame_id_numbers_present_flag
            bits.read(4)  # delta_frame_id_length_minus_2
            bits.read(3)  # additional_frame_id_length_minus_1
    bits.read(
        3
    )  # use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    if not reduced:
        bits.read(4)  # interintra, masked, warped, dual_filter
        enable_order_hint = bits.read(1)
        if enable_order_hint:
            bits.read(2)  # jnt_comp, ref_frame_mvs
        force_screen = 2 if bits.read(1) else bits.read(1)
        if force_screen > 0 and not bits.read(1):
            bits.read(1)
        if enable_order_hint:
            bits.read(3)  # order_hint_bits_minus_1
    bits.read(3)  # superres, cdef, restoration

    # color_config()
    high_bitdepth = bits.read(1)
    twelve_bit = bits.read(1) if (seq_profile == 2 and high_bitdepth) else 0
    bit_depth = 12 if twelve_bit else (10 if high_bitdepth else 8)
    monochrome = 0 if seq_profile == 1 else bits.read(1)
    if bits.read(1):  # color_description_present_flag
        cp = bits.read(8)
        tc = bits.read(8)
        mc = bits.read(8)
    else:
        cp = tc = mc = 2
    if monochrome:
        bits.read(1)  # color_range
        sub_x, sub_y, chroma_pos = 1, 1, 0
    elif cp == 1 and tc == 13 and mc == 0:
        sub_x, sub_y, chroma_pos = 0, 0, 0
    else:
        bits.read(1)  # color_range
        if seq_profile == 0:
            sub_x, sub_y = 1, 1
        elif seq_profile == 1:
            sub_x, sub_y = 0, 0
        elif bit_depth == 12:
            sub_x = bits.read(1)
            sub_y = bits.read(1) if sub_x else 0
        else:
            sub_x, sub_y = 1, 0
        chroma_pos = bits.read(2) if (sub_x and sub_y) else 0

    return {
        "reduced": reduced,
        "seq_profile": seq_profile,
        "seq_level_idx_0": seq_level_idx_0,
        "seq_tier_0": seq_tier_0,
        "high_bitdepth": high_bitdepth,
        "twelve_bit": twelve_bit,
        "monochrome": monochrome,
        "subsampling_x": sub_x,
        "subsampling_y": sub_y,
        "chroma_sample_position": chroma_pos,
    }


def _av1c(seq_obu_raw: bytes, seq_info: dict[str, int]) -> bytes:
    """Build an ISO-BMFF ``av1C`` AV1CodecConfigurationRecord box."""
    byte0 = 0x81  # marker=1, version=1
    byte1 = ((seq_info["seq_profile"] & 0x07) << 5) | (
        seq_info["seq_level_idx_0"] & 0x1F
    )
    byte2 = (
        ((seq_info["seq_tier_0"] & 1) << 7)
        | ((seq_info["high_bitdepth"] & 1) << 6)
        | ((seq_info["twelve_bit"] & 1) << 5)
        | ((seq_info["monochrome"] & 1) << 4)
        | ((seq_info["subsampling_x"] & 1) << 3)
        | ((seq_info["subsampling_y"] & 1) << 2)
        | (seq_info["chroma_sample_position"] & 0x03)
    )
    byte3 = 0x00  # reserved(3)=0, initial_presentation_delay_present=0
    return _box(b"av1C", bytes([byte0, byte1, byte2, byte3]) + seq_obu_raw)


def _is_av1_keyframe(frame_payload: bytes, *, reduced: int) -> bool:
    """Return True if an ``OBU_FRAME`` or ``OBU_FRAME_HEADER`` starts a key frame."""
    if reduced:
        return True
    if not frame_payload:
        return False
    first = frame_payload[0]
    show_existing = (first >> 7) & 1
    if show_existing:
        return False
    frame_type = (first >> 5) & 0x03
    return frame_type == 0 and bool(first & 0x10)


def mux_av1_obu(stream: bytes, fps: float, width: int, height: int) -> bytes:
    """Wrap an AV1 OBU (or IVF-framed AV1) stream in an ``av01`` MP4 container.

    Args:
        stream: Raw AV1 low-overhead OBU stream or IVF-wrapped AV1 stream, with
            no B-frames so decode order equals display order.
        fps: Positive frame rate; fractional rates such as 29.97 are exact.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        bytes: A self-contained, playable MP4 file with ``av01`` + ``av1C``.

    Raises:
        ValueError: If the stream is empty, has no sequence header, or has no
            coded frames.
    """
    if not stream:
        raise ValueError("Cannot mux an empty AV1 stream")
    if min(fps, width, height) <= 0:
        raise ValueError("Frame rate and dimensions must be positive")

    ivf_frames = _strip_ivf_wrapper(stream)
    temporal_units: list[list[tuple[int, bytes, bytes]]] = []
    if ivf_frames is not None:
        for chunk in ivf_frames:
            obus = _iter_obus(chunk)
            if obus:
                temporal_units.append(obus)
    else:
        obus = _iter_obus(stream)
        if not obus:
            raise ValueError("No OBUs found in AV1 stream")
        current: list[tuple[int, bytes, bytes]] = []
        seen_frame = False
        for obu in obus:
            obu_type = obu[0]
            is_frame = obu_type in (_OBU_FRAME_HEADER, _OBU_FRAME)
            if obu_type == _OBU_TEMPORAL_DELIMITER:
                if current and seen_frame:
                    temporal_units.append(current)
                    current = []
                    seen_frame = False
                continue
            if seen_frame and (obu_type == _OBU_SEQUENCE_HEADER or is_frame):
                temporal_units.append(current)
                current = []
                seen_frame = False
            current.append(obu)
            if is_frame:
                seen_frame = True
        if current and seen_frame:
            temporal_units.append(current)

    samples: list[bytes] = []
    sync_samples: list[int] = []
    active_header: bytes | None = None
    config: bytes | None = None
    reduced = False
    for unit in temporal_units:
        # Strip temporal delimiters from MP4 samples per AV1-ISOBMFF §2.4.
        kept = [obu for obu in unit if obu[0] != _OBU_TEMPORAL_DELIMITER]
        first_frame = next(
            (
                i
                for i, obu in enumerate(kept)
                if obu[0] in (_OBU_FRAME_HEADER, _OBU_FRAME)
            ),
            None,
        )
        prefix = kept if first_frame is None else kept[:first_frame]
        headers = [
            (raw, payload)
            for kind, raw, payload in prefix
            if kind == _OBU_SEQUENCE_HEADER
        ]
        for raw_header, payload in headers:
            seq_info = _parse_av1_seq_header(payload)
            if config is None:
                config = _av1c(raw_header, seq_info)
            active_header = raw_header
            reduced = bool(seq_info["reduced"])
        if first_frame is None:
            continue
        if any(kind == _OBU_SEQUENCE_HEADER for kind, _, _ in kept[first_frame:]):
            raise ValueError(
                "AV1 sequence header follows the first frame in its sample"
            )
        if active_header is None:
            raise ValueError("AV1 stream has no sequence header before its first frame")
        is_sync = _is_av1_keyframe(kept[first_frame][2], reduced=reduced)
        payload = b"".join(raw for _, raw, _ in kept)
        if is_sync:
            if not headers:
                payload = active_header + payload
            sync_samples.append(len(samples) + 1)
        samples.append(payload)

    if not samples:
        raise ValueError("No coded video frames found in AV1 stream")

    assert config is not None
    return _mux_samples(
        samples,
        sync_samples,
        _sample_timing(fps),
        width,
        height,
        _visual_entry(b"av01", width, height, config),
        b"av01",
    )


# ---------------------------------------------------------------------------
# Pure-Python MP4 inspection and concatenation (zero FFmpeg/PyAV)
# ---------------------------------------------------------------------------


def _iter_iso_boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yield ``(box_type, offset, size, payload_offset)`` for boxes in ``data[start:end]``."""
    limit = len(data) if end is None else end
    pos = start
    while pos + 8 <= limit:
        size, box_type = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > limit:
                raise ValueError("Truncated 64-bit ISO-BMFF box header")
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = limit - pos
        if size < header or pos + size > limit:
            raise ValueError(f"Invalid ISO-BMFF box size {size} for {box_type!r}")
        yield box_type, pos, size, pos + header
        pos += size


def _find_child_box(
    data: bytes, start: int, end: int, target: bytes
) -> tuple[int, int, int] | None:
    matches = [
        (offset, size, payload)
        for box_type, offset, size, payload in _iter_iso_boxes(data, start, end)
        if box_type == target
    ]
    if len(matches) > 1:
        raise ValueError(f"MP4 contains more than one {target.decode()} box")
    return matches[0] if matches else None


_CODEC_ENTRIES = {
    b"avc1": ("h264", b"avcC"),
    b"avc3": ("h264", b"avcC"),
    b"hvc1": ("hevc", b"hvcC"),
    b"hev1": ("hevc", b"hvcC"),
    b"av01": ("av1", b"av1C"),
}


def _parameter_sets(config: bytes, pos: int, count: int, kind: int, hevc: bool) -> int:
    for _ in range(count):
        size = struct.unpack_from(">H", config, pos)[0]
        start = pos + 2
        pos = start + size
        if size < (2 if hevc else 1) or pos > len(config):
            raise ValueError("invalid parameter set size")
        nal_type = (config[start] >> 1) & 63 if hevc else config[start] & 31
        if nal_type != kind:
            raise ValueError("invalid parameter set type")
    return pos


def _validate_decoder_configuration(codec: str, config: bytes) -> None:
    try:
        if codec == "h264":
            if config[0] != 1 or config[4] & 3 == 2:
                raise ValueError("invalid AVC header")
            sps_count = config[5] & 31
            pos = _parameter_sets(config, 6, sps_count, 7, False)
            pps_count = config[pos]
            _parameter_sets(config, pos + 1, pps_count, 8, False)
            if not sps_count or not pps_count:
                raise ValueError("AVC requires SPS and PPS")
        elif codec == "hevc":
            if config[0] != 1 or config[21] & 3 == 2:
                raise ValueError("invalid HEVC header")
            pos = 23
            present = set()
            for _ in range(config[22]):
                kind = config[pos] & 63
                count = struct.unpack_from(">H", config, pos + 1)[0]
                pos = _parameter_sets(config, pos + 3, count, kind, True)
                if count:
                    present.add(kind)
            if pos != len(config) or not {32, 33, 34} <= present:
                raise ValueError("HEVC requires VPS, SPS, and PPS")
        else:
            if config[0] != 0x81:
                raise ValueError("invalid AV1 header")
            sequence_headers = [
                payload for kind, _, payload in _iter_obus(config[4:]) if kind == 1
            ]
            if len(sequence_headers) != 1:
                raise ValueError("AV1 requires one sequence header")
            _parse_av1_seq_header(sequence_headers[0])
    except (IndexError, struct.error, ValueError) as exc:
        raise ValueError(f"invalid MP4 {codec} decoder configuration: {exc}") from exc


def _parse_single_video_track(mp4_bytes: bytes) -> dict[str, Any]:
    """Parse and validate MP4 track metadata and indexed media ranges."""
    try:
        return _read_single_video_track(mp4_bytes)
    except (IndexError, struct.error) as exc:
        raise ValueError("truncated MP4 sample table") from exc


def _read_single_video_track(mp4_bytes: bytes) -> dict[str, Any]:
    data_ranges = [
        (payload, offset + size)
        for kind, offset, size, payload in _iter_iso_boxes(mp4_bytes)
        if kind == b"mdat"
    ]
    moov = _find_child_box(mp4_bytes, 0, len(mp4_bytes), b"moov")
    if moov is None:
        raise ValueError("MP4 is missing moov box")
    moov_offset, moov_size, moov_payload = moov
    moov_end = moov_offset + moov_size

    ftyp = _find_child_box(mp4_bytes, 0, len(mp4_bytes), b"ftyp")
    brand = b"avc1"
    if ftyp is not None:
        _, ftyp_size, ftyp_payload = ftyp
        ftyp_body = mp4_bytes[ftyp_payload : ftyp[0] + ftyp_size]
        # Compatible brands start at offset 8 inside ftyp payload.
        for candidate in (b"av01", b"hvc1", b"avc1"):
            if candidate in ftyp_body[8:]:
                brand = candidate
                break

    trak = _find_child_box(mp4_bytes, moov_payload, moov_end, b"trak")
    if trak is None:
        raise ValueError("MP4 moov is missing trak box")
    trak_end = trak[0] + trak[1]

    has_edit_list = _find_child_box(mp4_bytes, trak[2], trak_end, b"edts") is not None
    tkhd = _find_child_box(mp4_bytes, trak[2], trak_end, b"tkhd")
    if tkhd is None:
        raise ValueError("MP4 trak is missing tkhd box")
    track_header = memoryview(mp4_bytes)[tkhd[2] : tkhd[0] + tkhd[1]]
    tkhd_ver = track_header[0]
    if tkhd_ver not in (0, 1):
        raise ValueError("unsupported MP4 tkhd version")
    dims_off = 88 if tkhd_ver == 1 else 76
    encoded_width, encoded_height = struct.unpack_from(">II", track_header, dims_off)
    width, height = encoded_width >> 16, encoded_height >> 16

    mdia = _find_child_box(mp4_bytes, trak[2], trak_end, b"mdia")
    if mdia is None:
        raise ValueError("MP4 trak is missing mdia box")
    mdia_end = mdia[0] + mdia[1]
    handler = _find_child_box(mp4_bytes, mdia[2], mdia_end, b"hdlr")
    if handler is None or mp4_bytes[handler[2] + 8 : handler[2] + 12] != b"vide":
        raise ValueError("MP4 requires a video track")

    mdhd = _find_child_box(mp4_bytes, mdia[2], mdia_end, b"mdhd")
    if mdhd is None:
        raise ValueError("MP4 mdia is missing mdhd box")
    media_header = memoryview(mp4_bytes)[mdhd[2] : mdhd[0] + mdhd[1]]
    mdhd_ver = media_header[0]
    if mdhd_ver == 1:
        timescale, duration = struct.unpack_from(">IQ", media_header, 20)
    elif mdhd_ver == 0:
        timescale, duration = struct.unpack_from(">II", media_header, 12)
    else:
        raise ValueError("unsupported MP4 mdhd version")

    minf = _find_child_box(mp4_bytes, mdia[2], mdia_end, b"minf")
    if minf is None:
        raise ValueError("MP4 mdia is missing minf box")
    dinf = _find_child_box(mp4_bytes, minf[2], minf[0] + minf[1], b"dinf")
    dref = (
        _find_child_box(mp4_bytes, dinf[2], dinf[0] + dinf[1], b"dref")
        if dinf is not None
        else None
    )
    if dref is None or struct.unpack_from(">I", mp4_bytes, dref[2] + 4)[0] != 1:
        raise ValueError("MP4 requires one local media data reference")
    references = list(_iter_iso_boxes(mp4_bytes, dref[2] + 8, dref[0] + dref[1]))
    if (
        len(references) != 1
        or references[0][0] != b"url "
        or references[0][1] + references[0][2] != references[0][3] + 4
        or mp4_bytes[references[0][3] : references[0][3] + 4] != b"\x00\x00\x00\x01"
    ):
        raise ValueError("MP4 requires one local media data reference")
    stbl = _find_child_box(mp4_bytes, minf[2], minf[0] + minf[1], b"stbl")
    if stbl is None:
        raise ValueError("MP4 minf is missing stbl box")
    stbl_start, stbl_end = stbl[2], stbl[0] + stbl[1]

    def table_payload(box_type: bytes) -> bytes | None:
        box = _find_child_box(mp4_bytes, stbl_start, stbl_end, box_type)
        return None if box is None else mp4_bytes[box[2] : box[0] + box[1]]

    def required_table(box_type: bytes) -> bytes:
        payload = table_payload(box_type)
        if payload is None:
            raise ValueError(f"MP4 stbl is missing {box_type.decode()} box")
        return payload

    def table_entries(
        payload: bytes, prefix: int, entry_format: str, count: int
    ) -> tuple:
        if prefix + count * struct.calcsize(entry_format) != len(payload):
            raise ValueError("invalid MP4 sample table length")
        return tuple(struct.iter_unpack(entry_format, memoryview(payload)[prefix:]))

    stsd = required_table(b"stsd")
    descriptions = struct.unpack_from(">I", stsd, 4)[0]
    sample_entry = stsd[8:]
    entries = list(_iter_iso_boxes(sample_entry))
    if descriptions != 1 or len(entries) != 1 or entries[0][2] != len(sample_entry):
        raise ValueError("MP4 requires exactly one sample description")
    if len(sample_entry) < _VISUAL_SAMPLE_ENTRY_SIZE:
        raise ValueError("MP4 sample description is truncated")
    kind, _, _, entry_payload = entries[0]
    if kind not in _CODEC_ENTRIES:
        raise ValueError(f"unsupported MP4 video sample entry: {kind!r}")
    config_offset = entry_payload + 78
    if config_offset > len(sample_entry):
        raise ValueError("MP4 sample description is truncated")
    reference = struct.unpack_from(">H", sample_entry, entry_payload + 6)[0]
    dimensions = struct.unpack_from(">HH", sample_entry, entry_payload + 24)
    if reference != 1 or width <= 0 or height <= 0 or dimensions != (width, height):
        raise ValueError(
            "MP4 sample description has invalid dimensions or data reference"
        )
    codec, config_kind = _CODEC_ENTRIES[kind]
    config = _find_child_box(
        sample_entry, config_offset, len(sample_entry), config_kind
    )
    if config is None:
        raise ValueError(f"missing MP4 {codec} decoder configuration")
    _validate_decoder_configuration(
        codec, sample_entry[config[2] : config[0] + config[1]]
    )

    stsz = required_table(b"stsz")
    fixed_size, sample_count = struct.unpack_from(">II", stsz, 4)
    media_bytes = sum(end - start for start, end in data_ranges)
    if not 0 < sample_count <= min(_MAX_INDEXED_SAMPLES, media_bytes):
        raise ValueError("invalid MP4 indexed sample count")
    if fixed_size:
        if len(stsz) != 12 or sample_count > media_bytes // fixed_size:
            raise ValueError("invalid MP4 fixed-size sample count")
        sample_sizes = [fixed_size] * sample_count
    else:
        sample_sizes = [size for (size,) in table_entries(stsz, 12, ">I", sample_count)]

    has_composition_offsets = table_payload(b"ctts") is not None
    stts = required_table(b"stts")
    stts_count = struct.unpack_from(">I", stts, 4)[0]
    if not 0 < stts_count <= sample_count:
        raise ValueError("invalid MP4 stts entry count or sample table length")
    timing = table_entries(stts, 8, ">II", stts_count)
    timed_samples, sample_delta = timing[0]
    if (
        timescale <= 0
        or any(count <= 0 or delta <= 0 for count, delta in timing)
        or sum(count for count, _ in timing) != sample_count
        or sum(count * delta for count, delta in timing) != duration
    ):
        raise ValueError("MP4 timing is invalid or not constant frame rate")

    stsc = required_table(b"stsc")
    stsc_entries_count = struct.unpack_from(">I", stsc, 4)[0]
    if not 0 < stsc_entries_count <= sample_count:
        raise ValueError("invalid MP4 stsc entry count")
    stsc_entries = table_entries(stsc, 8, ">III", stsc_entries_count)
    if any(description != 1 for _, _, description in stsc_entries):
        raise ValueError("MP4 samples must reference sample description 1")
    first_chunks = [entry[0] for entry in stsc_entries]
    if first_chunks and (
        first_chunks[0] != 1
        or any(b <= a for a, b in zip(first_chunks, first_chunks[1:], strict=False))
    ):
        raise ValueError("MP4 stsc first_chunk indices must start at 1 and increase")

    stco, co64 = table_payload(b"stco"), table_payload(b"co64")
    if (stco is None) == (co64 is None):
        raise ValueError("MP4 requires exactly one chunk offset table")
    chunks, offset_format = (co64, ">Q") if stco is None else (stco, ">I")
    chunk_count = struct.unpack_from(">I", chunks, 4)[0]
    if not 0 < chunk_count <= sample_count:
        raise ValueError("invalid MP4 chunk count")
    chunk_offsets = [
        offset for (offset,) in table_entries(chunks, 8, offset_format, chunk_count)
    ]

    chunk_runs = [
        (
            first_chunk,
            stsc_entries[entry_idx + 1][0]
            if entry_idx + 1 < len(stsc_entries)
            else len(chunk_offsets) + 1,
            samples_per_chunk,
        )
        for entry_idx, (first_chunk, samples_per_chunk, _) in enumerate(stsc_entries)
    ]
    if (
        any(
            not 1 <= start < end <= chunk_count + 1 or count <= 0
            for start, end, count in chunk_runs
        )
        or sum((end - start) * count for start, end, count in chunk_runs)
        != sample_count
    ):
        raise ValueError("MP4 stsc sample count does not match stsz")

    # Compute exact file byte offset for every 0-based sample.
    sample_offsets: list[int] = []
    sample_idx = 0
    for first_chunk, next_first, samples_per_chunk in chunk_runs:
        for chunk_1based in range(first_chunk, next_first):
            offset = chunk_offsets[chunk_1based - 1]
            for _ in range(samples_per_chunk):
                sample_offsets.append(offset)
                offset += sample_sizes[sample_idx]
                sample_idx += 1

    stss = table_payload(b"stss")
    if stss is not None:
        sync_count = struct.unpack_from(">I", stss, 4)[0]
        if not 0 < sync_count <= sample_count:
            raise ValueError("invalid MP4 stss sync sample count")
        sync_samples_1based = [
            index for (index,) in table_entries(stss, 8, ">I", sync_count)
        ]
    else:
        sync_samples_1based = list(range(1, sample_count + 1))
    if any(not 1 <= idx <= sample_count for idx in sync_samples_1based) or any(
        b <= a
        for a, b in zip(sync_samples_1based, sync_samples_1based[1:], strict=False)
    ):
        raise ValueError("MP4 stss sync samples must be increasing 1-based indices")
    if sync_samples_1based[0] != 1:
        raise ValueError("MP4 must have a keyframe at the start of the episode")

    media_starts = [start for start, _ in data_ranges]
    previous_end = 0
    for offset, size in sorted(zip(sample_offsets, sample_sizes, strict=True)):
        media_index = bisect_right(media_starts, offset) - 1
        if media_index < 0 or size <= 0 or offset + size > data_ranges[media_index][1]:
            raise ValueError("MP4 sample points outside the media data")
        if offset < previous_end:
            raise ValueError("MP4 sample ranges overlap")
        previous_end = offset + size

    return {
        "file_size": len(mp4_bytes),
        "moov_offset": moov_offset,
        "moov_size": moov_size,
        "brand": brand,
        "codec": codec,
        "width": width,
        "height": height,
        "timescale": timescale,
        "duration": duration,
        "sample_delta": sample_delta,
        "timed_samples": timed_samples,
        "timing_runs": stts_count,
        "has_composition_offsets": has_composition_offsets,
        "has_edit_list": has_edit_list,
        "sample_entry": sample_entry,
        "sample_sizes": sample_sizes,
        "sample_offsets": sample_offsets,
        "sync_samples_1based": sync_samples_1based,
    }


def parse_mp4_video_index(mp4_bytes: bytes) -> dict[str, Any]:
    """Compute LeRobot-Lance byte-index metadata from an MP4 without FFmpeg/PyAV.

    Returns the exact five columns required by ``lerobot-lancedb``'s ``videos``
    table: ``file_size``, ``moov_offset``, ``moov_size``, ``kf_indices`` (0-based
    keyframe frame indices), and ``kf_positions`` (file byte offsets of each
    keyframe sample).
    """
    info = _parse_single_video_track(mp4_bytes)
    kf_indices = [idx - 1 for idx in info["sync_samples_1based"]]
    kf_positions = [info["sample_offsets"][idx] for idx in kf_indices]
    return {
        "file_size": info["file_size"],
        "moov_offset": info["moov_offset"],
        "moov_size": info["moov_size"],
        "kf_indices": kf_indices,
        "kf_positions": kf_positions,
    }


def mp4_duration_s(mp4_bytes: bytes) -> float:
    """Return the media track duration of an MP4 in seconds."""
    info = _parse_single_video_track(mp4_bytes)
    return info["duration"] / info["timescale"]


def _require_decode_order_timing(info: dict[str, Any]) -> None:
    """Reject timing that sample-table concatenation cannot preserve.

    Composition offsets (``ctts``, from B-frames) and edit lists (``edts``)
    shift presentation time away from decode order. Concatenation rebuilds
    the sample table from decode order alone, so it would drop or misplace
    frames from such clips.
    """
    if info["has_composition_offsets"]:
        raise ValueError(
            "MP4 clip has composition offsets (ctts, e.g. B-frames); "
            "encode without B-frames so decode order equals display order"
        )
    if info["has_edit_list"]:
        raise ValueError("MP4 clip has an edit list (edts), which cannot be preserved")


def validate_mp4_clip(
    mp4_bytes: bytes,
    *,
    frame_count: int,
    fps: int,
    width: int,
    height: int,
    codec: str,
) -> float:
    """Validate a camera clip's sample table and return its duration in seconds."""
    info = _parse_single_video_track(mp4_bytes)
    if len(info["sample_sizes"]) != frame_count:
        raise ValueError(f"MP4 must contain {frame_count} indexed camera frames")
    _require_decode_order_timing(info)
    if (
        info["timing_runs"] != 1
        or info["timed_samples"] != frame_count
        or info["sample_delta"] <= 0
        or info["timescale"] != fps * info["sample_delta"]
        or info["duration"] != frame_count * info["sample_delta"]
    ):
        raise ValueError("MP4 timing does not match the episode frame rate and length")
    if (info["width"], info["height"]) != (width, height):
        raise ValueError("MP4 dimensions do not match the camera")
    if info["codec"] != codec:
        raise ValueError(f"MP4 video track does not match the configured {codec} codec")
    return info["duration"] / info["timescale"]


_TRACK_COMPATIBILITY_FIELDS = (
    "brand",
    "width",
    "height",
    "timescale",
    "sample_delta",
    "sample_entry",
)


def mp4_track_signature(mp4_bytes: bytes) -> tuple[Any, ...]:
    """Return the track fields that must match to concatenate validated clips."""
    info = _parse_single_video_track(mp4_bytes)
    return tuple(info[field] for field in _TRACK_COMPATIBILITY_FIELDS)


def concatenate_mp4_tracks(mp4_streams: Sequence[bytes]) -> bytes:
    """Concatenate compatible CFR single-track MP4 clips without re-encoding.

    All clips must share identical codec configuration (``stsd``), dimensions,
    timescale, and sample duration. The output is faststart (``moov`` before
    ``mdat``) and preserves exact keyframe indices.
    """
    if not mp4_streams:
        raise ValueError("Cannot concatenate an empty list of MP4 streams")
    parsed = [_parse_single_video_track(blob) for blob in mp4_streams]
    for idx, item in enumerate(parsed):
        _require_decode_order_timing(item)
        if (
            item["timing_runs"] != 1
            or item["sample_delta"] <= 0
            or item["timed_samples"] != len(item["sample_sizes"])
        ):
            raise ValueError(
                f"MP4 clip {idx} is not constant frame rate and cannot be concatenated"
            )
    if len(mp4_streams) == 1:
        return mp4_streams[0]
    ref = parsed[0]
    for idx, item in enumerate(parsed[1:], start=1):
        for field in _TRACK_COMPATIBILITY_FIELDS:
            if item[field] != ref[field]:
                raise ValueError(
                    f"MP4 clip {idx} is incompatible on {field!r} and cannot be concatenated"
                )

    samples: list[bytes] = []
    sync_samples: list[int] = []
    frame_offset = 0
    for blob, item in zip(mp4_streams, parsed, strict=True):
        for off, size in zip(item["sample_offsets"], item["sample_sizes"], strict=True):
            samples.append(blob[off : off + size])
        for sync_1based in item["sync_samples_1based"]:
            sync_samples.append(frame_offset + sync_1based)
        frame_offset += len(item["sample_sizes"])

    return _mux_samples(
        samples,
        sync_samples,
        (ref["timescale"], ref["sample_delta"]),
        ref["width"],
        ref["height"],
        ref["sample_entry"],
        ref["brand"],
    )
