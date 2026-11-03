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

"""Zero-dependency MP4 muxer for H.264 and HEVC Annex-B streams.

NVENC (via ``PyNvVideoCodec``) emits a raw Annex-B elementary stream,
which is not a playable container.

Assumptions (satisfied by the project's encoder configuration):

* The stream contains **no B-frames** (``bf=0``), so decode order equals
  display order and no composition-time (``ctts``) table is required.
* H.264 uses one VCL NAL per picture. HEVC also supports multiple slices.

These keep the muxer small. HEVC 4:4:4 requires a compatible decoder.
"""

from __future__ import annotations

import struct

__all__ = ["mux_h264_annexb", "mux_hevc_annexb"]

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
    """Build the ``stbl`` sample table."""
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
        fps,
        width,
        height,
        _avc1(width, height, sps, pps),
        b"avc1",
    )


def _mux_samples(
    samples: list[bytes],
    sync_samples: list[int],
    fps: float,
    width: int,
    height: int,
    sample_entry: bytes,
    brand: bytes,
) -> bytes:
    """Mux ordered coded pictures using a codec-specific sample entry."""
    if not samples:
        raise ValueError("No coded video frames found")
    sample_sizes = [len(sample) for sample in samples]
    from ovannotators._internal.video_config import video_frame_rate

    # Integer rates keep a 1000x timescale; fractional rates such as
    # 30000/1001 use the rate's numerator so each frame lasts exactly
    # ``denominator`` ticks.
    rate = video_frame_rate(fps)
    if rate.denominator == 1:
        timescale, sample_delta = rate.numerator * 1000, 1000
    else:
        timescale, sample_delta = rate.numerator, rate.denominator
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
        fps,
        width,
        height,
        _visual_entry(b"hvc1", width, height, config),
        b"hvc1",
    )
