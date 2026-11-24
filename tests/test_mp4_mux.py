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

import random
import struct

import pytest

from ovannotators._internal.mp4_mux import (
    _iter_nal_units,
    _parse_single_video_track,
    concatenate_mp4_tracks,
    mp4_duration_s,
    mux_av1_obu,
    mux_h264_annexb,
    parse_mp4_video_index,
    validate_mp4_clip,
)

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


def _reference_nal_units(stream: bytes):
    """Byte-at-a-time Annex-B scan: the behavior ``_iter_nal_units`` must keep."""
    n = len(stream)
    i = 0
    starts: list[tuple[int, int]] = []
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
        nal_end = starts[idx + 1][0] if idx + 1 < len(starts) else n
        nal = stream[pos + sc_len : nal_end]
        if nal:
            yield nal


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        pytest.param(_SC + _SPS + _SC + _PPS, [_SPS, _PPS], id="four-byte"),
        pytest.param(_SC3 + _SPS + _SC3 + _PPS, [_SPS, _PPS], id="three-byte"),
        pytest.param(
            _SC + _SPS + _SC3 + _PPS + _SC + _IDR, [_SPS, _PPS, _IDR], id="mixed"
        ),
        pytest.param(b"\x00" * 5 + _SC3 + _SPS, [_SPS], id="leading-zero-run"),
        pytest.param(_SC + _SC3 + _SPS, [_SPS], id="empty-nal-skipped"),
        pytest.param(b"\x00\x00\x00", [], id="zeros-only"),
        pytest.param(b"", [], id="empty"),
    ],
)
def test_iter_nal_units_start_code_forms(stream, expected):
    assert list(_iter_nal_units(stream)) == expected


def test_iter_nal_units_extra_zero_byte_stays_in_previous_nal():
    # A zero run longer than a 4-byte start code (``00 00 00 00 01``): the last
    # four zeros-and-one bytes are the start code; the extra zero stays in the
    # preceding NAL, as the original byte-at-a-time scan did.
    stream = _SC + _SPS + b"\x00" + _SC + _PPS
    assert list(_iter_nal_units(stream)) == [_SPS + b"\x00", _PPS]


def test_iter_nal_units_matches_byte_scan_on_random_streams():
    rng = random.Random(1234)
    for _ in range(2000):
        stream = bytes(
            rng.choice((0, 0, 0, 1, 1, 2, 0x65)) for _ in range(rng.randint(0, 64))
        )
        assert list(_iter_nal_units(stream)) == list(_reference_nal_units(stream)), (
            stream
        )


def _make_obu(obu_type: int, payload: bytes) -> bytes:
    header = ((obu_type & 0x0F) << 3) | 0x02  # obu_has_size_field = 1
    size = len(payload)
    leb = bytearray()
    while True:
        byte = size & 0x7F
        size >>= 7
        if size:
            leb.append(byte | 0x80)
        else:
            leb.append(byte)
            break
    return bytes([header]) + bytes(leb) + payload


# Minimal AV1 Sequence Header (profile 0, level 0, 8-bit 4:2:0) + key/inter frames.
_AV1_SEQ_HDR = _make_obu(1, b"\x00\x00\x00\x00\x00\x00\x00\x00")
_AV1_TD = _make_obu(2, b"")
# OBU_FRAME (type 6): show_existing_frame=0 (bit 7), frame_type=00 (KEY_FRAME, bits 6..5)
_AV1_KEY_FRAME = _make_obu(6, b"\x10\x11\x22\x33")
# OBU_FRAME (type 6): show_existing_frame=0 (bit 7), frame_type=01 (INTER_FRAME, bits 6..5 -> 0x20)
_AV1_INTER_FRAME = _make_obu(6, b"\x30\x44\x55\x66")


def test_mux_av1_obu_raw_and_ivf():
    raw_stream = _AV1_TD + _AV1_SEQ_HDR + _AV1_KEY_FRAME + _AV1_TD + _AV1_INTER_FRAME
    mp4 = mux_av1_obu(raw_stream, fps=30, width=64, height=48)
    boxes = list(_iter_boxes(mp4))
    assert [t for t, _ in boxes] == [b"ftyp", b"moov", b"mdat"]
    assert sum(sz for _, sz in boxes) == len(mp4)
    assert b"av01" in mp4
    assert b"av1C" in mp4
    idx = parse_mp4_video_index(mp4)
    assert idx["file_size"] == len(mp4)
    assert idx["kf_indices"] == [0]
    assert len(idx["kf_positions"]) == 1
    assert mp4_duration_s(mp4) == pytest.approx(2 / 30)

    # Same frames wrapped in an IVF container.
    ivf_hdr = (
        b"DKIF"
        + struct.pack("<HH", 0, 32)
        + b"AV01"
        + struct.pack("<HHIIQ", 64, 48, 30, 1, 2)
    )
    frame0 = _AV1_SEQ_HDR + _AV1_KEY_FRAME
    frame1 = _AV1_INTER_FRAME
    ivf = (
        ivf_hdr
        + struct.pack("<IQ", len(frame0), 0)
        + frame0
        + struct.pack("<IQ", len(frame1), 1)
        + frame1
    )
    mp4_from_ivf = mux_av1_obu(ivf, fps=30, width=64, height=48)
    assert b"av01" in mp4_from_ivf
    assert b"av1C" in mp4_from_ivf
    assert parse_mp4_video_index(mp4_from_ivf)["kf_indices"] == [0]


@pytest.mark.parametrize("corruption", ["missing", "header", "sequence", "timing"])
def test_av1_index_requires_a_valid_decoder_configuration(corruption):
    clip = mux_av1_obu(_AV1_SEQ_HDR + _AV1_KEY_FRAME, fps=30, width=64, height=48)
    bad = bytearray(clip)
    config = bad.index(b"av1C")
    if corruption == "missing":
        bad[config : config + 4] = b"free"
    elif corruption == "header":
        bad[config + 4] = 0
    elif corruption == "sequence":
        bad[config + 8] = (15 << 3) | 2
    else:
        # A non-reduced sequence header that sets timing_info_present_flag but
        # ends inside the 64 timing bits. Only the OBU payload, not the OBU
        # header byte, may be parsed as the sequence header.
        assert bad[config + 8 : config + 10] == b"\x0a\x08"
        bad[config + 10] = 0x04
    with pytest.raises(ValueError, match="decoder configuration"):
        parse_mp4_video_index(bytes(bad))


def test_parse_mp4_video_index_and_concatenate_tracks():
    stream_a = _SC + _SPS + _SC + _PPS + _SC + _IDR + _SC + _NON_IDR
    stream_b = _SC + _SPS + _SC + _PPS + _SC + _IDR + _SC + _NON_IDR + _SC + _NON_IDR
    mp4_a = mux_h264_annexb(stream_a, fps=10, width=32, height=32)
    mp4_b = mux_h264_annexb(stream_b, fps=10, width=32, height=32)

    idx_a = parse_mp4_video_index(mp4_a)
    assert idx_a["file_size"] == len(mp4_a)
    assert idx_a["moov_offset"] > 0
    assert idx_a["moov_size"] > 0
    assert idx_a["kf_indices"] == [0]
    assert len(idx_a["kf_positions"]) == 1
    assert mp4_duration_s(mp4_a) == pytest.approx(0.2)

    merged = concatenate_mp4_tracks([mp4_a, mp4_b])
    boxes = list(_iter_boxes(merged))
    assert [t for t, _ in boxes] == [b"ftyp", b"moov", b"mdat"]
    assert sum(sz for _, sz in boxes) == len(merged)
    merged_idx = parse_mp4_video_index(merged)
    assert merged_idx["kf_indices"] == [0, 2]
    assert len(merged_idx["kf_positions"]) == 2
    assert merged_idx["kf_positions"][1] > merged_idx["kf_positions"][0]
    assert mp4_duration_s(merged) == pytest.approx(0.5)


def test_mux_uses_largesize_mdat_beyond_32_bit_box_size(monkeypatch):
    from ovannotators._internal import mp4_mux

    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR + _SC + _NON_IDR
    small = mux_h264_annexb(stream, fps=10, width=32, height=32)
    monkeypatch.setattr(mp4_mux, "_MAX_32BIT_BOX_SIZE", 8)
    large = mux_h264_annexb(stream, fps=10, width=32, height=32)

    mdat = large.index(b"mdat") - 4
    assert struct.unpack_from(">I", large, mdat)[0] == 1
    assert struct.unpack_from(">Q", large, mdat + 8)[0] == len(large) - mdat
    small_index = parse_mp4_video_index(small)
    large_index = parse_mp4_video_index(large)
    assert large_index["kf_positions"][0] == small_index["kf_positions"][0] + 8
    first = large_index["kf_positions"][0]
    assert large[first : first + 4] == struct.pack(">I", len(_SPS))


def test_mux_uses_version_1_times_beyond_32_bit_duration(monkeypatch):
    from ovannotators._internal import mp4_mux

    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR + _SC + _NON_IDR + _SC + _NON_IDR
    monkeypatch.setattr(mp4_mux, "_MAX_32BIT_DURATION", 10)
    mp4 = mux_h264_annexb(stream, fps=10, width=32, height=24)

    for box in (b"mvhd", b"tkhd", b"mdhd"):
        assert mp4[mp4.index(box) + 4] == 1
    assert mp4_duration_s(mp4) == pytest.approx(0.3)
    assert mp4_mux.validate_mp4_clip(
        mp4, frame_count=3, fps=10, width=32, height=24, codec="h264"
    ) == pytest.approx(0.3)
    merged = concatenate_mp4_tracks([mp4, mp4])
    assert mp4_duration_s(merged) == pytest.approx(0.6)


def _insert_box(mp4: bytes, parent: bytes, child: bytes, payload: bytes) -> bytes:
    """Append a ``child`` box at the end of ``parent`` and fix enclosing sizes."""
    from ovannotators._internal.mp4_mux import _box

    new_box = _box(child, payload)
    path = {
        b"stbl": (b"moov", b"trak", b"mdia", b"minf", b"stbl"),
        b"trak": (b"moov", b"trak"),
    }[parent]
    out = bytearray(mp4)
    pos, end, starts = 0, len(out), []
    for name in path:
        while struct.unpack_from(">4s", out, pos + 4)[0] != name:
            pos += struct.unpack_from(">I", out, pos)[0]
        starts.append(pos)
        end = pos + struct.unpack_from(">I", out, pos)[0]
        pos += 8
    out[end:end] = new_box
    for start in starts:
        size = struct.unpack_from(">I", out, start)[0]
        struct.pack_into(">I", out, start, size + len(new_box))
    # moov precedes mdat, so the chunk offset moves by the inserted size.
    stco = out.index(b"stco") - 4
    offset = struct.unpack_from(">I", out, stco + 16)[0]
    struct.pack_into(">I", out, stco + 16, offset + len(new_box))
    return bytes(out)


@pytest.mark.parametrize(
    ("parent", "child", "error"),
    [
        (b"stbl", b"ctts", "composition offsets"),
        (b"trak", b"edts", "edit list"),
    ],
)
def test_clips_with_presentation_timing_are_rejected(parent, child, error):
    from ovannotators._internal.mp4_mux import validate_mp4_clip

    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR + _SC + _NON_IDR
    clip = mux_h264_annexb(stream, fps=10, width=32, height=32)
    assert validate_mp4_clip(
        clip, frame_count=2, fps=10, width=32, height=32, codec="h264"
    ) == pytest.approx(0.2)
    timed = _insert_box(clip, parent, child, bytes(8))
    assert parse_mp4_video_index(timed)["kf_indices"] == [0]  # still parseable
    with pytest.raises(ValueError, match=error):
        validate_mp4_clip(
            timed, frame_count=2, fps=10, width=32, height=32, codec="h264"
        )
    with pytest.raises(ValueError, match=error):
        concatenate_mp4_tracks([clip, timed])


def _patch_u32(mp4: bytes, box: bytes, offset: int, value: int) -> bytes:
    """Overwrite one big-endian u32 at ``offset`` bytes past ``box``'s size field."""
    out = bytearray(mp4)
    struct.pack_into(">I", out, out.index(box) - 4 + offset, value)
    return bytes(out)


_TWO_FRAME_H264 = _SC + _SPS + _SC + _PPS + _SC + _IDR + _SC + _NON_IDR


@pytest.mark.parametrize("defect", ["config", "offset", "duplicate offsets"])
def test_mp4_entry_points_validate_the_same_media_ranges_and_configuration(defect):
    clip = mux_h264_annexb(_TWO_FRAME_H264, fps=10, width=32, height=32)
    if defect == "config":
        broken = clip.replace(b"avcC", b"free", 1)
        error = "decoder configuration"
    elif defect == "offset":
        broken = _patch_u32(clip, b"stco", 16, 0)
        error = "outside the media data"
    else:
        broken = _insert_box(clip, b"stbl", b"co64", bytes(8))
        error = "one chunk offset table"
    consumers = (
        parse_mp4_video_index,
        lambda data: concatenate_mp4_tracks([data]),
        lambda data: concatenate_mp4_tracks([clip, data]),
        lambda data: validate_mp4_clip(
            data, frame_count=2, fps=10, width=32, height=32, codec="h264"
        ),
    )
    for consume in consumers:
        with pytest.raises(ValueError, match=error):
            consume(broken)


@pytest.mark.parametrize(
    ("box", "offset", "value", "error"),
    [
        (b"stsd", 12, 2, "exactly one sample description"),
        (b"stsc", 24, 2, "sample description 1"),
        (b"stsz", 16, 3, "sample table length"),
        (b"stsc", 16, 0, "stsc first_chunk"),
        (b"stsc", 20, 3, "stsc sample count"),
        (b"stsc", 20, 1, "stsc sample count"),
        (b"stss", 16, 0, "stss sync samples"),
        (b"stss", 16, 3, "stss sync samples"),
    ],
)
def test_malformed_sample_table_indices_are_rejected(box, offset, value, error):
    clip = mux_h264_annexb(_TWO_FRAME_H264, fps=10, width=32, height=32)
    with pytest.raises(ValueError, match=error):
        parse_mp4_video_index(_patch_u32(clip, box, offset, value))


@pytest.mark.parametrize(
    ("offset", "value", "error"),
    [
        (12, 2, "sample table length"),
        (16, 1, "not constant frame rate"),
        (20, 0, "not constant frame rate"),
    ],
)
def test_concatenate_rejects_non_cfr_clips(offset, value, error):
    clip = mux_h264_annexb(_TWO_FRAME_H264, fps=10, width=32, height=32)
    broken = _patch_u32(clip, b"stts", offset, value)
    with pytest.raises(ValueError, match=error):
        concatenate_mp4_tracks([clip, broken])


def test_mux_rejects_a_stream_without_keyframes():
    stream = _SC + _SPS + _SC + _PPS + _SC + _NON_IDR + _SC + _NON_IDR
    with pytest.raises(ValueError, match="no detected keyframes"):
        mux_h264_annexb(stream, fps=30, width=16, height=16)


def test_av1_sync_samples_carry_a_sequence_header_and_show_the_keyframe():
    hidden_key = _make_obu(6, b"\x00")
    stream = (
        _AV1_SEQ_HDR + _AV1_KEY_FRAME + _AV1_TD + hidden_key + _AV1_TD + _AV1_KEY_FRAME
    )
    clip = mux_av1_obu(stream, fps=30, width=64, height=48)

    info = _parse_single_video_track(clip)
    assert info["sync_samples_1based"] == [1, 3]
    offset, size = info["sample_offsets"][2], info["sample_sizes"][2]
    assert clip[offset : offset + size] == _AV1_SEQ_HDR + _AV1_KEY_FRAME


def test_concatenate_preserves_exact_media_timing():
    clip = mux_h264_annexb(_TWO_FRAME_H264, fps=30, width=32, height=32)
    clip = _patch_u32(clip, b"mdhd", 20, 30001)
    clip = _patch_u32(clip, b"mdhd", 24, 2002)
    clip = _patch_u32(clip, b"stts", 20, 1001)

    info = _parse_single_video_track(concatenate_mp4_tracks([clip, clip]))

    assert (info["timescale"], info["sample_delta"], info["duration"]) == (
        30001,
        1001,
        4004,
    )


def test_camera_clip_rejects_overlapping_sample_ranges():
    clip = mux_h264_annexb(_TWO_FRAME_H264, fps=30, width=32, height=32)
    info = _parse_single_video_track(clip)
    stco_offset = clip.index(b"stco") - 4
    stsc_offset = clip.index(b"stsc") - 4
    modified = bytearray(clip)
    struct.pack_into(">I", modified, stsc_offset + 20, 1)
    struct.pack_into(">II", modified, stco_offset, 24, int.from_bytes(b"stco", "big"))
    struct.pack_into(">I", modified, stco_offset + 12, 2)
    first_offset = info["sample_offsets"][0] + 4
    struct.pack_into(">I", modified, stco_offset + 16, first_offset)
    modified[stco_offset + 20 : stco_offset + 20] = struct.pack(">I", first_offset)
    for box in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
        offset = modified.index(box) - 4
        (size,) = struct.unpack_from(">I", modified, offset)
        struct.pack_into(">I", modified, offset, size + 4)

    with pytest.raises(ValueError, match="sample ranges overlap"):
        validate_mp4_clip(
            bytes(modified), frame_count=2, fps=30, width=32, height=32, codec="h264"
        )


def test_camera_clip_rejects_stsc_that_disagrees_with_stsz():
    clip = bytearray(mux_h264_annexb(_TWO_FRAME_H264, fps=30, width=32, height=32))
    struct.pack_into(">I", clip, clip.index(b"stsc") - 4 + 20, 3)

    with pytest.raises(ValueError, match="stsc sample count"):
        validate_mp4_clip(
            bytes(clip), frame_count=2, fps=30, width=32, height=32, codec="h264"
        )


def _truncate_sample_entry(mp4: bytes, entry: bytes) -> bytes:
    """Drop every byte after ``entry``'s 8-byte box header and fix the sizes."""
    out = bytearray(mp4)
    start = out.index(entry, out.index(b"stsd")) - 4  # ftyp also lists avc1
    (size,) = struct.unpack_from(">I", out, start)
    removed = size - 8
    del out[start + 8 : start + size]
    for box in (b"stsd", b"stbl", b"minf", b"mdia", b"trak", b"moov"):
        offset = out.index(box) - 4
        (old,) = struct.unpack_from(">I", out, offset)
        struct.pack_into(">I", out, offset, old - removed)
    struct.pack_into(">I", out, start, 8)
    # moov precedes mdat, so the chunk offset moves by the removed size.
    stco = out.index(b"stco") - 4
    (chunk,) = struct.unpack_from(">I", out, stco + 16)
    struct.pack_into(">I", out, stco + 16, chunk - removed)
    return bytes(out)


def test_camera_clip_rejects_a_truncated_sample_entry():
    clip = mux_h264_annexb(_TWO_FRAME_H264, fps=30, width=32, height=32)
    assert validate_mp4_clip(
        clip, frame_count=2, fps=30, width=32, height=32, codec="h264"
    )
    truncated = _truncate_sample_entry(clip, b"avc1")

    with pytest.raises(ValueError, match="sample description is truncated"):
        validate_mp4_clip(
            truncated, frame_count=2, fps=30, width=32, height=32, codec="h264"
        )


def test_mux_rejects_a_frame_rate_that_rounds_to_zero():
    with pytest.raises(ValueError, match="too small"):
        mux_h264_annexb(_TWO_FRAME_H264, fps=0.0001, width=32, height=32)
