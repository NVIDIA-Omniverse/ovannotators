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

"""CPU-testable configuration example for lossless video capture."""


def test_lossless_video_configuration():
    # [snippet:doc-lossless-video]
    from ovannotators.writers import BasicWriter, BasicWriterConfig

    writer = BasicWriter(
        BasicWriterConfig(
            video=True,
            video_codec="hevc",
            video_lossless=True,
            video_preset="P5",
            video_vars=["LdrColor"],
        )
    )
    # Feed frames as usual, then call writer.finalize(backend) to write the MP4.
    # [/snippet:doc-lossless-video]
    assert writer.config.video and writer.config.video_lossless
    assert writer.config.video_codec == "hevc"
    assert writer.config.video_vars == ["LdrColor"]
