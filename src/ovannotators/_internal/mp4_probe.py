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

"""Read video track timing from an MP4 container without decoding it.

Only the ``moov`` box is loaded; media data is never read, so probing a large
episode video is cheap. Timestamps are on the track's media timeline (edit
lists are not applied), which is what ``mp4_mux.mux_h264_annexb`` writes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

__all__ = ["Mp4VideoTrack", "probe_mp4_video"]

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


@dataclass(frozen=True)
class Mp4VideoTrack:
    """Timing and format of the first video track of an MP4 file.

    Attributes:
        codec: Sample-entry four-character code, such as ``avc1``.
        width: Coded width in pixels from the sample entry.
        height: Coded height in pixels from the sample entry.
        timescale: Media time units per second.
        frame_count: Number of encoded samples (frames).
        timestamps: Presentation time of each frame in timescale units, in
            presentation order, on the media timeline. Edit lists (``edts``)
            are ignored, so a stream with a non-zero edit offset, such as some
            H.264 streams with B-frames from other muxers, reports times
            shifted from true presentation time.
        duration: Sum of sample durations in timescale units.
    """

    codec: str
    width: int
    height: int
    timescale: int
    frame_count: int
    timestamps: tuple[int, ...]
    duration: int


def _boxes(data: bytes, start: int = 0, end: int | None = None) -> Iterator:
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                raise ValueError("Truncated MP4 box header")
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f"Malformed MP4 box {kind!r}")
        yield kind, pos + header, pos + size
        pos += size


def _read_moov(stream: BinaryIO) -> bytes:
    """Seek over top-level boxes and return the ``moov`` payload.

    Raises:
        ValueError: If a header is truncated, a box size is malformed or runs
            past the end of the file, or there is no ``moov`` box.
    """
    stream.seek(0, 2)
    total = stream.tell()
    stream.seek(0)
    while True:
        header = stream.read(8)
        if len(header) < 8:
            raise ValueError("MP4 file has no moov box")
        size, kind = struct.unpack(">I4s", header)
        consumed = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) < 8:
                raise ValueError("Truncated MP4 box header")
            size = struct.unpack(">Q", extended)[0]
            consumed = 16
        if kind == b"moov" and size == 0:
            return stream.read()
        if size == 0:
            raise ValueError("MP4 file has no moov box")
        if size < consumed:
            raise ValueError(f"Malformed MP4 box {kind!r}")
        if size - consumed > total - stream.tell():
            raise ValueError(f"Truncated MP4 box {kind!r}")
        if kind == b"moov":
            return stream.read(size - consumed)
        stream.seek(size - consumed, 1)


def _children(data: bytes, start: int, end: int) -> dict[bytes, tuple[int, int]]:
    found: dict[bytes, tuple[int, int]] = {}
    for kind, body, stop in _boxes(data, start, end):
        found.setdefault(kind, (body, stop))
    return found


def _video_track(moov: bytes) -> Mp4VideoTrack:
    for kind, body, stop in _boxes(moov):
        if kind != b"trak":
            continue
        mdia = _children(moov, body, stop).get(b"mdia")
        if mdia is None:
            continue
        parts = _children(moov, *mdia)
        hdlr = parts.get(b"hdlr")
        if hdlr is None or moov[hdlr[0] + 8 : hdlr[0] + 12] != b"vide":
            continue
        return _parse_media(moov, parts)
    raise ValueError("MP4 file has no video track")


def _parse_media(data: bytes, mdia: dict[bytes, tuple[int, int]]) -> Mp4VideoTrack:
    try:
        mdhd = mdia[b"mdhd"][0]
        minf = _children(data, *mdia[b"minf"])
        stbl = _children(data, *minf[b"stbl"])
        stsd, stts, stsz = (stbl[k][0] for k in (b"stsd", b"stts", b"stsz"))
    except KeyError as exc:
        raise ValueError(f"MP4 video track lacks a {exc.args[0]!r} box") from exc
    version = data[mdhd]
    offset = mdhd + (20 if version == 1 else 12)
    timescale = struct.unpack_from(">I", data, offset)[0]
    if timescale == 0:
        raise ValueError("MP4 video track has a zero timescale")
    entry = stsd + 8  # version/flags + entry_count
    codec = data[entry + 4 : entry + 8].decode("ascii", "replace")
    width, height = struct.unpack_from(">HH", data, entry + 8 + 24)
    frame_count = struct.unpack_from(">I", data, stsz + 8)[0]

    decode_times = []
    clock = 0
    (entries,) = struct.unpack_from(">I", data, stts + 4)
    for index in range(entries):
        count, delta = struct.unpack_from(">II", data, stts + 8 + 8 * index)
        for _ in range(count):
            decode_times.append(clock)
            clock += delta
    if len(decode_times) != frame_count:
        raise ValueError(
            f"MP4 stts covers {len(decode_times)} samples but stsz lists {frame_count}"
        )
    times = decode_times
    if b"ctts" in stbl:
        ctts = stbl[b"ctts"][0]
        signed = data[ctts] == 1
        (entries,) = struct.unpack_from(">I", data, ctts + 4)
        offsets: list[int] = []
        for index in range(entries):
            count, value = struct.unpack_from(
                ">Ii" if signed else ">II", data, ctts + 8 + 8 * index
            )
            offsets.extend([value] * count)
        if len(offsets) != frame_count:
            raise ValueError("MP4 ctts does not cover every sample")
        times = sorted(t + o for t, o in zip(decode_times, offsets))
    return Mp4VideoTrack(
        codec=codec,
        width=width,
        height=height,
        timescale=timescale,
        frame_count=frame_count,
        timestamps=tuple(times),
        duration=clock,
    )


def probe_mp4_video(path: str | Path) -> Mp4VideoTrack:
    """Return timing and format of the first video track in an MP4 file.

    Args:
        path: Local MP4 file.

    Returns:
        Track codec, coded size, timescale, frame count and per-frame
        presentation timestamps. Timestamps come from the ``stts`` and
        ``ctts`` boxes on the media timeline; edit lists are not applied.

    Raises:
        ValueError: If the file is not a well-formed MP4 with a video track.
    """
    with open(path, "rb") as stream:
        moov = _read_moov(stream)
    try:
        return _video_track(moov)
    except struct.error as exc:
        raise ValueError(f"Truncated MP4 video track: {exc}") from exc
