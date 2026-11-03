# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry for episode-aware profile writers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ovannotators.exceptions import ConfigurationError

if TYPE_CHECKING:
    from ovannotators.profile_writers._base import ProfileWriter


class ProfileWriterRegistry:
    """Map exact public profile identifiers to profile writer classes."""

    _registry: ClassVar[dict[str, type[ProfileWriter]]] = {}

    @classmethod
    def register(cls, writer_cls: type[ProfileWriter]) -> type[ProfileWriter]:
        """Register a profile writer by its class-level ``name``."""
        existing = cls._registry.get(writer_cls.name)
        if existing is not None and existing is not writer_cls:
            raise ValueError(
                f"Profile writer name '{writer_cls.name}' is already registered by "
                f"{existing.__qualname__}"
            )
        cls._registry[writer_cls.name] = writer_cls
        return writer_cls

    @classmethod
    def get(cls, name: str) -> type[ProfileWriter]:
        """Return the profile writer class registered under ``name``."""
        try:
            return cls._registry[name]
        except KeyError:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ConfigurationError(
                f"No profile writer registered as '{name}'. Available: {available}"
            ) from None

    @classmethod
    def create(cls, name: str, *args: object, **kwargs: object) -> ProfileWriter:
        """Instantiate a registered profile writer."""
        return cls.get(name)(*args, **kwargs)

    @classmethod
    def list_names(cls) -> list[str]:
        """Return registered profile identifiers in sorted order."""
        return sorted(cls._registry)
