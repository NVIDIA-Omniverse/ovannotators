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

"""BackendRegistry — maps backend name strings to Backend classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ovannotators.exceptions import ConfigurationError

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend


class BackendRegistry:
    """Registry mapping backend name strings to Backend classes.

    Backends self-identify with a class-level ``name`` attribute (mirroring
    :class:`~ovannotators.writers.WriterRegistry` and
    :class:`~ovannotators.renderer_outputs.RendererOutputRegistry`). Use
    :meth:`register` as a decorator or call it directly.

    Example::

        @BackendRegistry.register
        class MyBackend:
            name = "my_backend"
            ...
    """

    _registry: dict[str, type[Backend]] = {}

    @classmethod
    def register(cls, backend_cls: type[Backend]) -> type[Backend]:
        """Register a backend class by its ``name`` attribute.

        Args:
            backend_cls (type[Backend]): Backend class with a non-empty
                class-level ``name``. Usable as a decorator.

        Returns:
            type[Backend]: The same class, to support decorator use.

        Raises:
            ValueError: If ``backend_cls`` has no ``name``, or the name is
                already registered by a different class.
        """
        name = getattr(backend_cls, "name", None)
        if not name:
            raise ValueError(
                f"Backend {backend_cls.__qualname__} must define a non-empty "
                "class-level 'name' attribute to be registered"
            )
        existing = cls._registry.get(name)
        if existing is not None and existing is not backend_cls:
            raise ValueError(
                f"Backend name '{name}' is already registered by "
                f"{existing.__qualname__}"
            )
        cls._registry[name] = backend_cls
        return backend_cls

    @classmethod
    def get(cls, name: str) -> type[Backend]:
        """Return the backend class registered under ``name``.

        Args:
            name (str): Registry key (e.g. ``"disk"``).

        Returns:
            type[Backend]: The registered backend class.

        Raises:
            ConfigurationError: If no backend is registered under ``name``.
        """
        try:
            return cls._registry[name]
        except KeyError:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ConfigurationError(
                f"Unknown backend '{name}'. Available: {available}"
            ) from None

    @classmethod
    def create(cls, name: str, *args: object, **kwargs: object) -> Backend:
        """Instantiate the backend registered under ``name``.

        A convenience over ``BackendRegistry.get(name)(*args, **kwargs)`` that
        works uniformly for built-in and third-party backends.

        Args:
            name (str): Registry key (e.g. ``"disk"``).
            *args: Positional arguments forwarded to the backend constructor.
            **kwargs: Keyword arguments forwarded to the backend constructor.

        Returns:
            Backend: A new backend instance.
        """
        return cls.get(name)(*args, **kwargs)

    @classmethod
    def list_names(cls) -> list[str]:
        """Return the sorted list of registered backend names.

        Returns:
            list[str]: All registered backend names, sorted.
        """
        return sorted(cls._registry)


def _register_builtins() -> None:
    """Register the backends shipped with ovannotators."""
    from ovannotators.backends.disk import DiskBackend
    from ovannotators.backends.s3 import S3Backend
    from ovannotators.backends.ovstorage import OvstorageBackend

    for backend_cls in (DiskBackend, S3Backend, OvstorageBackend):
        BackendRegistry.register(backend_cls)


_register_builtins()
