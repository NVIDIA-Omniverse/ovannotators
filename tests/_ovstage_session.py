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

"""ovstage-backed renderer session used by the GPU test suite.

ovrtx 0.4 deprecated the renderer's own USD entry points (``open_usd``,
``open_usd_from_string``, ``update_from_usd_time``, ...) in favour of
``ovstage.population`` driving an externally owned :class:`ovstage.Stage` that
the renderer borrows via ``Renderer.attach_ovstage``.

Two details make the attached flow easy to get wrong, so they are centralized
here rather than repeated at every call site:

* Every population call publishes at an *ordinal*, and the publication is only
  visible to the renderer once the stage's write floor has advanced past it.
  Stepping against an uncommitted ordinal fails with ``ordinal N exceeds
  ovstage write floor``.
* ``Renderer.step`` requires ``ordinal`` in attached mode and rejects it in
  standalone mode, so attaching a stage forces every step call to change too.
* ovrtx's USD schema families must be registered before ovstage's first
  population, and from ovrtx 0.5 importing ``ovrtx`` no longer does that.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable
from typing import Any, Self


def render_vars_by_source_name(frame: Any) -> dict[str, Any]:
    """Re-key a frame's render vars by ovrtx ``sourceName``.

    ovrtx 0.5 keys ``FrameOutput.render_vars`` by full RenderVar prim path, so
    ``frame.render_vars["LdrColor"]`` raises ``KeyError`` and
    ``"LdrColor" in frame.render_vars`` is ``False``. Tests that map render vars
    directly (rather than going through
    :class:`~ovannotators.renderer_outputs.ovrtx.OvrtxOutput`) re-key through
    here so the rule has one definition. Reserved ovrtx outputs carry no
    ``source_name`` and keep their ``render_vars`` key.

    Args:
        frame: An ovrtx ``FrameOutput``.

    Returns:
        Mapping of ``sourceName`` to its ovrtx ``RenderVarOutput``. When two
        prims share a ``sourceName`` the first in ``render_vars`` order wins.
    """
    by_name: dict[str, Any] = {}
    for path, raw_var in frame.render_vars.items():
        by_name.setdefault(getattr(raw_var, "source_name", "") or path, raw_var)
    return by_name


def render_var(frame: Any, source_name: str) -> Any:
    """Look up one render var in a frame by its ovrtx ``sourceName``.

    Args:
        frame: An ovrtx ``FrameOutput``.
        source_name: The AOV kind to find, e.g. ``"LdrColor"``.

    Returns:
        The matching ovrtx ``RenderVarOutput``.

    Raises:
        KeyError: If the frame carries no render var with that ``sourceName``.
    """
    by_name = render_vars_by_source_name(frame)
    try:
        return by_name[source_name]
    except KeyError:
        raise KeyError(
            f"No render var with sourceName {source_name!r} in this frame; "
            f"available: {sorted(by_name)}"
        ) from None


class OvstageSession:
    """An ``ovstage.Stage`` attached to an ``ovrtx.Renderer``.

    Wraps population + ordinal bookkeeping so tests read roughly the same as
    they did against the deprecated ``Renderer.open_usd`` API.

    Attributes:
        stage: The owned :class:`ovstage.Stage`.
        renderer: The :class:`ovrtx.Renderer` the stage is attached to.
        ordinal: The most recently committed publication ordinal.
    """

    def __init__(self, name: str = "ovannotators-test", config: Any = None) -> None:
        """Create a stage, a renderer, and attach the two.

        Args:
            name: Stage name, surfaced in ovstage diagnostics.
            config: Optional ``ovrtx.RendererConfig``. When ``None`` the
                renderer is built with its defaults.
        """
        import ovrtx
        import ovstage

        self._ovstage = ovstage
        # Register ovrtx's USD schema families before any stage exists. Without
        # them OpenUSD has no RenderProduct/RenderVar definitions and sensor
        # products fail to map with an empty ``render_product=''``. From ovrtx
        # 0.5 this registration is also what makes those families visible to an
        # attached ovstage when it populates, and plain ``import ovrtx`` does
        # not perform it (only OVRTX_PXR_SCHEMA_AUTO_REGISTER=1 does), so call
        # it explicitly rather than relying on a side effect of constructing
        # the Renderer.
        ovrtx.register_schema_paths()
        # Order still matters: construct the Renderer *before* the Stage, as
        # ovrtx's own ovstage examples do.
        self.renderer = ovrtx.Renderer() if config is None else ovrtx.Renderer(config)
        self.stage = ovstage.Stage(name)
        self.renderer.attach_ovstage(self.stage)
        self.ordinal = 0

    def _publish(self, populate: Any) -> None:
        """Run a population call at the next ordinal and commit it.

        Args:
            populate: Callable taking the ordinal to publish at.
        """
        next_ordinal = self.ordinal + 1
        populate(next_ordinal)
        # The renderer only observes a publication once the write floor covers
        # it; advance_write_floor clamps, so this is safe to call repeatedly.
        self.stage.advance_write_floor(next_ordinal).wait()
        self.ordinal = next_ordinal

    def open_usd(self, usd_path: str, time_code: float = 0.0) -> None:
        """Populate the stage from a USD file.

        Args:
            usd_path: Path to the USD file to open.
            time_code: Stage time in seconds to evaluate at. Defaults to ``0.0``
                to match the deprecated ``Renderer.open_usd``; see the note on
                :meth:`open_usd_from_string`.
        """
        self._publish(
            lambda n: self._ovstage.population.open_usd(
                self.stage, str(usd_path), ordinal=n, time_code=time_code
            )
        )

    def open_usd_from_string(self, usda: str, time_code: float = 0.0) -> None:
        """Populate the stage from inline USDA content.

        ``ovstage.population`` defaults ``time_code`` to ``math.nan``, which
        evaluates at USD's *Default* time code — not time 0, which is what the
        deprecated ``Renderer.open_usd_from_string`` used. For a scene with
        time-sampled transforms the two resolve to different stage state, so
        this defaults to ``0.0`` to keep renders identical across the migration.

        Args:
            usda: USDA source text.
            time_code: Stage time in seconds to evaluate at.
        """
        self._publish(
            lambda n: self._ovstage.population.open_usd_from_string(
                self.stage, usda, ordinal=n, time_code=time_code
            )
        )

    def update_from_usd_time(self, time_code: float) -> None:
        """Re-evaluate time-sampled attributes at ``time_code``.

        Args:
            time_code: Stage time in **seconds** (not USD timecodes).
        """
        self._publish(
            lambda n: self._ovstage.population.update_from_usd_time(
                self.stage, n, time_code
            )
        )

    def step(self, render_products: Iterable[str], delta_time: float) -> Any:
        """Render one step against the latest committed publication.

        Args:
            render_products: Render product paths to step.
            delta_time: Simulation time delta for this step.

        Returns:
            The ovrtx ``RenderProductSetOutputs`` for this step.
        """
        return self.renderer.step(
            render_products=set(render_products),
            delta_time=delta_time,
            ordinal=self.ordinal,
        )

    def step_async(self, render_products: Iterable[str], delta_time: float) -> Any:
        """Enqueue a step against the latest committed publication.

        Args:
            render_products: Render product paths to step.
            delta_time: Simulation time delta for this step.

        Returns:
            The pending ovrtx operation handle.
        """
        return self.renderer.step_async(
            render_products=set(render_products),
            delta_time=delta_time,
            ordinal=self.ordinal,
        )

    def close(self) -> None:
        """Detach and tear down deterministically.

        Mirrors the existing teardown discipline: drop the renderer while the
        interpreter is fully alive rather than letting shutdown GC race ovrtx's
        atexit teardown. Detaching first keeps the borrowed stage valid for the
        renderer's whole lifetime.
        """
        # Explicit detach -> stage.destroy() -> renderer.destroy(), matching the
        # teardown ovrtx's own ovstage examples use. Leaving this to GC lets the
        # Stage outlive or predecease the renderer nondeterministically, which
        # can deadlock at interpreter shutdown.
        for step in (
            self.renderer.detach_ovstage,
            self.stage.destroy,
            self.renderer.destroy,
        ):
            try:
                step()
            except RuntimeError as exc:  # pragma: no cover - already torn down
                # A renderer that never attached, or was already destroyed,
                # raises here. Surface it rather than swallowing it silently:
                # a real teardown failure otherwise shows up much later as a
                # leaked-mapping warning or a hang at interpreter exit.
                warnings.warn(
                    f"ovstage teardown step {step.__name__} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        del self.renderer
        del self.stage
        gc.collect()

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            This session.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Tear the session down on context exit."""
        self.close()
