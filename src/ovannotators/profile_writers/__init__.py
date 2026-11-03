# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode-aware output profile contracts and registry."""

from __future__ import annotations

import importlib.metadata as _imeta
import logging

from ovannotators.profile_writers._artifacts import (
    PROFILE_ARTIFACT_DIR,
    PROFILE_ARTIFACT_PATHS,
    canonical_json_bytes,
    sha256_bytes,
    write_profile_artifacts,
)
from ovannotators.profile_writers._base import ProfileWriter
from ovannotators.profile_writers._models import (
    ConformanceResult,
    EpisodeEnvelope,
    LossReport,
    NamedTensor,
    ProfileMetadata,
    SampleEnvelope,
    StreamDescriptor,
    WriterRecord,
)
from ovannotators.profile_writers._registry import ProfileWriterRegistry
from ovannotators.profile_writers.cosmos_multiview import (
    CosmosMaterializationConfig,
    CosmosMultiviewWriter,
    CosmosMultiviewWriterConfig,
    CosmosQAConfig,
    CosmosValidationFinding,
    CosmosValidationRecord,
)
from ovannotators.profile_writers.groot_lerobot_v2 import (
    GROOT_RELEASE_COMMIT,
    GROOT_RELEASE_TAG,
    GrootActionModality,
    GrootLeRobotV2Writer,
    GrootLeRobotV2WriterConfig,
    GrootNormalizationConfig,
    GrootStateModality,
    GrootVideoModality,
    validate_groot_lerobot_v2,
)
from ovannotators.profile_writers.umi_replay_buffer import (
    UMIFieldSpec,
    UMIReplayBufferWriter,
    UMIReplayBufferWriterConfig,
)

__all__ = [
    "PROFILE_ARTIFACT_DIR",
    "PROFILE_ARTIFACT_PATHS",
    "ConformanceResult",
    "CosmosMaterializationConfig",
    "CosmosMultiviewWriter",
    "CosmosMultiviewWriterConfig",
    "CosmosQAConfig",
    "CosmosValidationFinding",
    "CosmosValidationRecord",
    "EpisodeEnvelope",
    "GROOT_RELEASE_COMMIT",
    "GROOT_RELEASE_TAG",
    "GrootActionModality",
    "GrootLeRobotV2Writer",
    "GrootLeRobotV2WriterConfig",
    "GrootNormalizationConfig",
    "GrootStateModality",
    "GrootVideoModality",
    "LossReport",
    "NamedTensor",
    "ProfileMetadata",
    "ProfileWriter",
    "ProfileWriterRegistry",
    "SampleEnvelope",
    "StreamDescriptor",
    "WriterRecord",
    "canonical_json_bytes",
    "sha256_bytes",
    "write_profile_artifacts",
    "validate_groot_lerobot_v2",
    "UMIFieldSpec",
    "UMIReplayBufferWriter",
    "UMIReplayBufferWriterConfig",
]

logger = logging.getLogger(__name__)


def _load_entry_points() -> None:
    try:
        entry_points = _imeta.entry_points(group="ovannotators.profile_writers")
    except Exception:
        logger.warning("Unable to enumerate profile-writer entry points", exc_info=True)
        return
    for entry_point in entry_points:
        try:
            entry_point.load()
        except Exception:
            logger.warning(
                "Unable to load profile-writer entry point %s",
                entry_point.name,
                exc_info=True,
            )


_load_entry_points()
