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

"""Central registry for Writer implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ovannotators.exceptions import ConfigurationError

if TYPE_CHECKING:
    from ovannotators.writers._base import Writer


class WriterRegistry:
    """Registry mapping name strings to Writer classes.

    Writers self-register at import time via the :meth:`register` decorator.
    Third-party writers can register via the ``ovannotators.writers``
    entry-point group.

    Example::

        @WriterRegistry.register
        class MyWriter:
            name = "my_writer"
            required_render_vars = frozenset({"LdrColor"})
            ...
    """

    _registry: dict[str, type[Writer]] = {}

    @classmethod
    def register(cls, writer_cls: type[Writer]) -> type[Writer]:
        """Register a writer class by its ``name`` attribute.

        Args:
            writer_cls (type[Writer]): Writer class with a class-level ``name``.
                Usable as a decorator.

        Returns:
            type[Writer]: The same class, to support decorator use.

        Raises:
            ValueError: If ``name`` is already registered by another class.
        """
        existing = cls._registry.get(writer_cls.name)
        if existing is not None and existing is not writer_cls:
            raise ValueError(
                f"Writer name '{writer_cls.name}' is already registered by "
                f"{existing.__qualname__}"
            )
        cls._registry[writer_cls.name] = writer_cls
        return writer_cls

    @classmethod
    def get(cls, name: str) -> type[Writer]:
        """Return the writer class registered under ``name``.

        Args:
            name (str): Registry key (e.g. ``"coco"``).

        Returns:
            type[Writer]: The registered writer class.

        Raises:
            ConfigurationError: If no writer is registered under ``name``.
        """
        try:
            return cls._registry[name]
        except KeyError:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ConfigurationError(
                f"No writer registered as '{name}'. Available: {available}"
            ) from None

    @classmethod
    def create(cls, name: str, *args: object, **kwargs: object) -> Writer:
        """Instantiate the writer registered under ``name``.

        A convenience over ``WriterRegistry.get(name)(*args, **kwargs)`` that
        works uniformly for built-in and third-party writers, so callers can
        construct any writer by name without importing its config's module.

        Args:
            name (str): Registry key (e.g. ``"coco"``).
            *args: Positional arguments forwarded to the writer constructor.
            **kwargs: Keyword arguments forwarded to the writer constructor.

        Returns:
            Writer: A new writer instance.
        """
        return cls.get(name)(*args, **kwargs)

    @classmethod
    def list_names(cls) -> list[str]:
        """Return the sorted list of registered writer names.

        Returns:
            list[str]: All registered writer names, sorted.
        """
        return sorted(cls._registry)
