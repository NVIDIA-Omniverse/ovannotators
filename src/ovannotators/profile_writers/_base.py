# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle protocol for episode-aware dataset profile writers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ovannotators.backends import Backend
    from ovannotators.profile_writers._models import (
        ConformanceResult,
        EpisodeEnvelope,
        SampleEnvelope,
    )


@runtime_checkable
class ProfileWriter(Protocol):
    """Additive writer contract for episode-aware training-data profiles.

    This protocol intentionally does not extend the frame-oriented
    :class:`ovannotators.writers.Writer` contract. A profile writer receives
    explicit episodes, independently named streams, authoritative source time,
    and policy signals.
    """

    name: ClassVar[str]

    def begin_episode(self, episode: EpisodeEnvelope, backend: Backend) -> None:
        """Start an episode after validating its declared stream schema."""
        ...

    def write_sample(self, sample: SampleEnvelope, backend: Backend) -> None:
        """Validate and persist one sample in the active episode."""
        ...

    def end_episode(self, backend: Backend) -> None:
        """Close the active episode and flush its bounded state."""
        ...

    def finalize(self, backend: Backend) -> None:
        """Finalize the dataset idempotently and write ``_SUCCESS`` last."""
        ...

    def validate(self, backend: Backend) -> ConformanceResult:
        """Validate persisted output without claiming external acceptance."""
        ...
