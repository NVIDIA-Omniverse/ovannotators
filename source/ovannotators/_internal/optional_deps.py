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

"""Availability probes for optional, environment-specific dependencies.

These helpers centralize "is this optional backend usable?" checks so that
both library code and the test suite agree on a single definition of
availability. The checks are deliberately conservative: importability alone is
not enough when a package can be present as a content-less stub or when the
bindings import on a host that has no usable GPU/driver.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def ovrtx_available() -> bool:
    """Report whether a usable ``ovrtx`` renderer backend is present.

    Guards on the real renderer API rather than bare importability: a
    content-less stub ``ovrtx`` (for example the placeholder published to public
    indexes) can import successfully without exposing a ``Renderer`` class, and
    rendering would then fail only later at call time. Treating that stub as
    "available" would cause GPU render tests to run and crash on CPU-only hosts
    and CI runners instead of skipping cleanly.

    Returns:
        bool: ``True`` if ``ovrtx`` imports and exposes a ``Renderer``
        attribute, otherwise ``False``.
    """
    try:
        import ovrtx
    except Exception:
        return False
    return hasattr(ovrtx, "Renderer")


@functools.lru_cache(maxsize=1)
def nvimgcodec_available() -> bool:
    """Report whether the ``nvidia.nvimgcodec`` bindings can be imported.

    This checks only that the Python bindings are importable. Callers that need
    a working GPU encode/decode path are responsible for separately confirming
    that a usable CUDA device is present (see
    :func:`ovannotators._internal.codecs._gpu_image_codec_available`), since the
    bindings can import on a host with no usable driver or device.

    Returns:
        bool: ``True`` if ``nvidia.nvimgcodec`` imports, otherwise ``False``.
    """
    try:
        import nvidia.nvimgcodec  # noqa: F401
    except Exception:
        return False
    return True
