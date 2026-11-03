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

"""integrations — bridges from ovannotators outputs to external model runtimes."""

from __future__ import annotations

from ovannotators.integrations.cosmos3_transfer import (
    CONDITIONING_TO_HINT,
    build_inference_command,
    build_transfer_spec,
    load_manifest,
    output_video_path,
    prepare_transfer,
    write_transfer_spec,
)

__all__ = [
    "CONDITIONING_TO_HINT",
    "build_inference_command",
    "build_transfer_spec",
    "load_manifest",
    "output_video_path",
    "prepare_transfer",
    "write_transfer_spec",
]
