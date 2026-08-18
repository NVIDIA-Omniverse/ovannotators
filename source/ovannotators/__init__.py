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

"""ovannotators — ground-truth annotation library for Isaac Sim / OmniVerse renderers."""

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

try:
    # Single source of truth: the version declared in pyproject.toml, read from
    # the installed distribution metadata.
    __version__ = version("ovannotators")
except PackageNotFoundError:
    # Running from a source tree that was never installed; no real version is
    # available, so use a sentinel rather than duplicating the project version.
    __version__ = "0.0.0+unknown"

# Re-export primary public surface.
from ovannotators.backends import (
    Backend,
    BackendRegistry,
    DiskBackend,
    DiskBackendConfig,
    S3Backend,
    S3BackendConfig,
    OvstorageBackend,
    OvstorageBackendConfig,
)
from ovannotators.config import config_from_dict, config_to_dict
from ovannotators.exceptions import (
    BackendReadError,
    BackendWriteError,
    ConfigurationError,
    ExtractionError,
    MissingRenderVarError,
    OVAnnotatorsError,
    ValidationError,
)
from ovannotators.renderer_outputs import (
    OvrtxOutput,
    RendererOutput,
    RendererOutputRegistry,
)
from ovannotators.writers import (
    BasicWriter,
    BasicWriterConfig,
    BOPWriter,
    BOPWriterConfig,
    COCOWriter,
    COCOWriterConfig,
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
    KITTIWriter,
    KITTIWriterConfig,
    LeRobotWriter,
    LeRobotWriterConfig,
    Writer,
    WriterRegistry,
    YOLOWriter,
    YOLOWriterConfig,
)
from ovannotators.types import (
    BBox2D,
    BBox3D,
    CameraParams,
    PointCloudResult,
    Resolution,
)

__all__ = [
    "__version__",
    # config serialization
    "config_to_dict",
    "config_from_dict",
    # types
    "Resolution",
    "BBox2D",
    "BBox3D",
    "CameraParams",
    "PointCloudResult",
    # exceptions
    "OVAnnotatorsError",
    "ConfigurationError",
    "ExtractionError",
    "ValidationError",
    "MissingRenderVarError",
    "BackendWriteError",
    "BackendReadError",
    # renderer outputs
    "RendererOutput",
    "OvrtxOutput",
    "RendererOutputRegistry",
    # writers
    "Writer",
    "WriterRegistry",
    "BasicWriter",
    "BasicWriterConfig",
    "COCOWriter",
    "COCOWriterConfig",
    "KITTIWriter",
    "KITTIWriterConfig",
    "YOLOWriter",
    "YOLOWriterConfig",
    "BOPWriter",
    "BOPWriterConfig",
    "LeRobotWriter",
    "LeRobotWriterConfig",
    "CosmosTransferWriter",
    "CosmosTransferWriterConfig",
    # backends
    "Backend",
    "DiskBackend",
    "DiskBackendConfig",
    "S3Backend",
    "S3BackendConfig",
    "OvstorageBackend",
    "OvstorageBackendConfig",
    "BackendRegistry",
]
