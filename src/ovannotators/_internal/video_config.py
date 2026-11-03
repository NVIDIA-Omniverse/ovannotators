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

VideoCodec = Literal["h264", "hevc"]
VideoPreset = Literal["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def validate_video_options(
    codec: VideoCodec, lossless: bool, preset: VideoPreset | None
) -> None:
    """Validate codec options without importing GPU dependencies.

    Args:
        codec: H.264 or HEVC.
        lossless: Preserve normalized RGB pixels exactly (HEVC only).
        preset: NVENC preset, or None to retain the codec's default.

    Raises:
        ValueError: If any option or combination is unsupported.
    """
    if codec not in ("h264", "hevc"):
        raise ValueError("video_codec must be 'h264' or 'hevc'")
    if not isinstance(lossless, bool):
        raise ValueError("video_lossless must be a bool")
    if lossless and codec != "hevc":
        raise ValueError("video_lossless requires video_codec='hevc'")
    if preset is not None and preset not in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        raise ValueError("video_preset must be None or 'P1' through 'P7'")


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
        ValueError: If ``fps`` is not a positive, finite real number.
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
    return Fraction(fps).limit_denominator(1001)
