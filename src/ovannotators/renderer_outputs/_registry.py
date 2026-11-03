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

"""RendererOutputRegistry — maps renderer names to RendererOutput implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ovannotators.exceptions import ConfigurationError

if TYPE_CHECKING:
    from ovannotators.renderer_outputs._base import RendererOutput


class RendererOutputRegistry:
    """Registry mapping renderer name strings to RendererOutput classes.

    Renderer outputs self-identify with a class-level ``renderer_name``
    attribute and register via :meth:`register` (usable as a decorator).
    """

    _registry: dict[str, type[RendererOutput]] = {}

    @classmethod
    def register(cls, renderer_cls: type[RendererOutput]) -> type[RendererOutput]:
        """Register a renderer-output class by its ``renderer_name``.

        Args:
            renderer_cls (type[RendererOutput]): Class with a class-level
                ``renderer_name``. Usable as a decorator.

        Returns:
            type[RendererOutput]: The same class, to support decorator use.

        Raises:
            ValueError: If ``renderer_name`` is already registered by a
                different class.
        """
        existing = cls._registry.get(renderer_cls.renderer_name)
        if existing is not None and existing is not renderer_cls:
            raise ValueError(
                f"Renderer output name '{renderer_cls.renderer_name}' is already "
                f"registered by {existing.__qualname__}"
            )
        cls._registry[renderer_cls.renderer_name] = renderer_cls
        return renderer_cls

    @classmethod
    def get(cls, name: str) -> type[RendererOutput]:
        """Return the renderer-output class registered under ``name``.

        Args:
            name (str): Registry key (e.g. ``"ovrtx"``).

        Returns:
            type[RendererOutput]: The registered class.

        Raises:
            ConfigurationError: If no renderer output is registered under
                ``name``.
        """
        try:
            return cls._registry[name]
        except KeyError:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ConfigurationError(
                f"Unknown renderer output '{name}'. Available: {available}"
            ) from None

    @classmethod
    def create(cls, name: str, *args: object, **kwargs: object) -> RendererOutput:
        """Instantiate the renderer output registered under ``name``.

        A convenience over ``RendererOutputRegistry.get(name)(*args, **kwargs)``
        that works uniformly for built-in and third-party renderer outputs.

        Args:
            name (str): Registry key (e.g. ``"ovrtx"``).
            *args: Positional arguments forwarded to the constructor.
            **kwargs: Keyword arguments forwarded to the constructor.

        Returns:
            RendererOutput: A new renderer-output instance.
        """
        return cls.get(name)(*args, **kwargs)

    @classmethod
    def list_names(cls) -> list[str]:
        """Return the sorted list of registered renderer names.

        Returns:
            list[str]: All registered renderer names, sorted.
        """
        return sorted(cls._registry)


def _register_builtins() -> None:
    """Register the renderer outputs shipped with ovannotators."""
    from ovannotators.renderer_outputs.ovrtx import OvrtxOutput

    RendererOutputRegistry.register(OvrtxOutput)


_register_builtins()
