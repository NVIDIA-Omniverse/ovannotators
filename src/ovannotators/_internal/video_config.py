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

"""Shared validation for optional video encoding settings."""

import math
from fractions import Fraction
from numbers import Real
from typing import Literal

VideoCodec = Literal["h264", "hevc", "av1"]
VideoPreset = Literal["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def validate_video_options(
    codec: VideoCodec,
    lossless: bool,
    preset: VideoPreset | None,
    *,
    gop: int | None = None,
) -> None:
    """Validate codec options without importing GPU dependencies.

    Args:
        codec: H.264, HEVC, or AV1.
        lossless: Preserve normalized RGB pixels exactly (HEVC only).
        preset: NVENC preset, or None to retain the codec's default.
        gop: Optional keyframe interval (positive integer), or None for encoder
            default.

    Raises:
        ValueError: If any option or combination is unsupported.
    """
    if codec not in ("h264", "hevc", "av1"):
        raise ValueError("video_codec must be 'h264', 'hevc', or 'av1'")
    if not isinstance(lossless, bool):
        raise ValueError("video_lossless must be a bool")
    if lossless and codec != "hevc":
        raise ValueError("video_lossless requires video_codec='hevc'")
    if preset is not None and preset not in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        raise ValueError("video_preset must be None or 'P1' through 'P7'")
    if gop is not None and (
        isinstance(gop, bool) or not isinstance(gop, int) or gop <= 0
    ):
        raise ValueError("video_gop must be a positive integer or None")


def video_frame_rate(fps: float, name: str = "Frame rate") -> Fraction:
    """Return ``fps`` as an exact rational frame rate.

    NTSC-family rates such as 23.976, 29.97 and 59.94 snap to their exact
    ``N * 1000 / 1001`` values; other rates use the nearest fraction with a
    denominator no larger than 1001.

    Args:
        fps: Positive, finite frames per second.
        name: Setting name used in the error message.

    Returns:
        Fraction: The frame rate as ``numerator / denominator``.

    Raises:
        ValueError: If ``fps`` is not a positive, finite real number, or is so
            small that it rounds to a zero rational rate.
    """
    if (
        isinstance(fps, bool)
        or not isinstance(fps, Real)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(f"{name} must be a positive, finite number, got {fps!r}")
    ntsc = round(fps * 1.001)
    if ntsc and not float(fps).is_integer() and abs(fps - ntsc / 1.001) < 0.005:
        return Fraction(ntsc * 1000, 1001)
    rate = Fraction(fps).limit_denominator(1001)
    if rate <= 0:
        raise ValueError(f"{name} {fps!r} is too small to represent as a frame rate")
    return rate
