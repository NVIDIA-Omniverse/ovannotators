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

from importlib.metadata import PackageNotFoundError, version

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
    OvstorageBackend,
    OvstorageBackendConfig,
    S3Backend,
    S3BackendConfig,
)
from ovannotators.backends.webdataset import (
    SampleFiles,
    WebDatasetPackager,
    WebDatasetPackagerConfig,
)
from ovannotators.capture import (
    CaptureEpisode,
    DatasetCapture,
    export_finalized_dataset,
)
from ovannotators.capture_adapters import (
    AdapterContext,
    BasicFrameAdapter,
    BasicVideoEpisodeAdapter,
    BOPFrameAdapter,
    CaptureAdapter,
    ClipSpan,
    COCOFrameAdapter,
    KITTIFrameAdapter,
    YOLOFrameAdapter,
)
from ovannotators.config import config_from_dict, config_to_dict
from ovannotators.exceptions import (
    BackendReadError,
    BackendWriteError,
    ConfigurationError,
    ExtractionError,
    MissingRenderVarError,
    OVAnnotatorsError,
    PackagingError,
    ValidationError,
)
from ovannotators.profile_writers import (
    ConformanceResult,
    CosmosMaterializationConfig,
    CosmosMultiviewWriter,
    CosmosMultiviewWriterConfig,
    CosmosQAConfig,
    CosmosValidationFinding,
    CosmosValidationRecord,
    EpisodeEnvelope,
    GrootActionModality,
    GrootLeRobotV2Writer,
    GrootLeRobotV2WriterConfig,
    GrootNormalizationConfig,
    GrootStateModality,
    GrootVideoModality,
    LeRobotLanceFeatureSpec,
    LeRobotLanceWriter,
    LeRobotLanceWriterConfig,
    LeRobotV3FeatureSpec,
    LeRobotV3Writer,
    LeRobotV3WriterConfig,
    LossReport,
    NamedTensor,
    ProfileMetadata,
    ProfileWriter,
    ProfileWriterRegistry,
    SampleEnvelope,
    StreamDescriptor,
    UMIFieldSpec,
    UMIReplayBufferWriter,
    UMIReplayBufferWriterConfig,
    WriterRecord,
    validate_groot_lerobot_v2,
    validate_lerobot_v3_canonical,
)
from ovannotators.renderer_outputs import (
    OvrtxOutput,
    RendererOutput,
    RendererOutputRegistry,
)
from ovannotators.requirements import (
    DatasetOption,
    DatasetProfile,
    DatasetRequirements,
    describe_requirements,
    list_profiles,
    register_dataset_profiles,
)
from ovannotators.types import (
    BBox2D,
    BBox3D,
    CameraParams,
    PointCloudResult,
    Resolution,
)
from ovannotators.writers import (
    BasicWriter,
    BasicWriterConfig,
    BOPWriter,
    BOPWriterConfig,
    CenterPoseWriter,
    CenterPoseWriterConfig,
    COCOWriter,
    COCOWriterConfig,
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
    DOPEWriter,
    DOPEWriterConfig,
    KITTIWriter,
    KITTIWriterConfig,
    LeRobotWriter,
    LeRobotWriterConfig,
    OpenLABELWriter,
    OpenLABELWriterConfig,
    Writer,
    WriterRegistry,
    YCBVideoWriter,
    YCBVideoWriterConfig,
    YOLOWriter,
    YOLOWriterConfig,
)

__all__ = [
    "CaptureEpisode",
    "DatasetCapture",
    "AdapterContext",
    "BasicFrameAdapter",
    "BOPFrameAdapter",
    "BasicVideoEpisodeAdapter",
    "CaptureAdapter",
    "ClipSpan",
    "COCOFrameAdapter",
    "KITTIFrameAdapter",
    "YOLOFrameAdapter",
    "export_finalized_dataset",
    "SampleFiles",
    "WebDatasetPackager",
    "WebDatasetPackagerConfig",
    "DatasetOption",
    "DatasetProfile",
    "DatasetRequirements",
    "describe_requirements",
    "list_profiles",
    "register_dataset_profiles",
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
    "EpisodeEnvelope",
    "SampleEnvelope",
    "StreamDescriptor",
    "NamedTensor",
    "ProfileMetadata",
    "WriterRecord",
    "LossReport",
    "ConformanceResult",
    "CosmosMultiviewWriter",
    "CosmosMultiviewWriterConfig",
    "CosmosMaterializationConfig",
    "CosmosQAConfig",
    "CosmosValidationFinding",
    "CosmosValidationRecord",
    "LeRobotLanceFeatureSpec",
    "LeRobotLanceWriter",
    "LeRobotLanceWriterConfig",
    "LeRobotV3FeatureSpec",
    "LeRobotV3Writer",
    "LeRobotV3WriterConfig",
    "validate_lerobot_v3_canonical",
    "GrootActionModality",
    "GrootLeRobotV2Writer",
    "GrootLeRobotV2WriterConfig",
    "GrootNormalizationConfig",
    "GrootStateModality",
    "GrootVideoModality",
    "validate_groot_lerobot_v2",
    "UMIFieldSpec",
    "UMIReplayBufferWriter",
    "UMIReplayBufferWriterConfig",
    # exceptions
    "OVAnnotatorsError",
    "ConfigurationError",
    "ExtractionError",
    "ValidationError",
    "MissingRenderVarError",
    "PackagingError",
    "BackendWriteError",
    "BackendReadError",
    # renderer outputs
    "RendererOutput",
    "OvrtxOutput",
    "RendererOutputRegistry",
    # writers
    "Writer",
    "WriterRegistry",
    "ProfileWriter",
    "ProfileWriterRegistry",
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
    "DOPEWriter",
    "DOPEWriterConfig",
    "CenterPoseWriter",
    "CenterPoseWriterConfig",
    "YCBVideoWriter",
    "YCBVideoWriterConfig",
    "LeRobotWriter",
    "LeRobotWriterConfig",
    "OpenLABELWriter",
    "OpenLABELWriterConfig",
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
