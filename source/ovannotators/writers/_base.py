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

"""Writer protocol definition for ovannotators."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


@runtime_checkable
class Writer(Protocol):
    """Protocol that all writer implementations must satisfy.

    Writers receive rendered frames as numpy arrays and persist them in a
    dataset-specific layout.  Per-frame data is written incrementally via
    :meth:`write_frame`; any deferred work (COCO JSON, YOLO ``data.yaml``, …)
    is flushed in :meth:`finalize`.

    Validation contract
    -------------------
    ``required_render_vars`` is validated *per frame*, not at construction time:
    a writer cannot know which vars a future frame will contain until it sees
    it. At the start of :meth:`write_frame`, implementations check the frame's
    ``render_vars`` against ``required_render_vars`` (use the shared
    :func:`ovannotators.writers._helpers.validate_required_vars` helper) and
    raise :class:`~ovannotators.exceptions.MissingRenderVarError` (a
    :class:`~ovannotators.exceptions.ValidationError`) naming the writer, the
    missing var, and the offending ``frame_id`` if any required var is absent.

    Class attributes
    ----------------
    name:
        Unique registry identifier, e.g. ``"coco"`` or ``"yolo"``.
    required_render_vars:
        AOV/render-var names every frame must provide. May be empty (e.g.
        :class:`~ovannotators.writers.basic.BasicWriter`), in which case any
        subset of vars is accepted.
    """

    name: ClassVar[str]
    required_render_vars: ClassVar[frozenset[str]]

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Persist a single frame.

        Implementations validate ``render_vars`` against
        ``required_render_vars`` first (see the class-level validation
        contract) before writing anything.

        Args:
            frame_id: Zero-based monotonically increasing frame counter.
            render_vars: Mapping of render-var name to numpy array. Extra keys
                beyond ``required_render_vars`` are tolerated and ignored.
            backend: Destination backend (satisfies :class:`Backend`).

        Raises:
            MissingRenderVarError: If a var in ``required_render_vars`` is
                absent from ``render_vars``.
        """
        ...  # pragma: no cover

    def finalize(self, backend: Backend) -> None:
        """Flush accumulated state and release resources.

        Called exactly once after the last :meth:`write_frame` call.  Must be
        idempotent — a second call must not raise or produce duplicate output.

        Args:
            backend: Same backend instance passed to :meth:`write_frame`.
        """
        ...  # pragma: no cover
