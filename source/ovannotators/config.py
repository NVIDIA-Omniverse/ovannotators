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

"""Serialization helpers for the ``*Config`` dataclasses.

Every writer/backend config in ovannotators is a plain :func:`dataclasses.dataclass`,
which makes them easy to persist: convert one to a plain dict with
:func:`config_to_dict`, dump it with ``json``/``yaml``/``toml``, then rebuild it
later with :func:`config_from_dict`. Compared with calling
:func:`dataclasses.asdict` and ``Config(**data)`` directly, these helpers also
coerce :class:`pathlib.Path` fields and reconstruct nested config dataclasses, so
the round-trip survives JSON.
"""

from __future__ import annotations

import dataclasses
import types
from pathlib import PurePath
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

__all__ = ["config_to_dict", "config_from_dict"]

_T = TypeVar("_T")

# Union origins for both ``typing.Union[...]`` and PEP 604 ``X | Y`` annotations.
_UNION_ORIGINS = (Union, types.UnionType)


def config_to_dict(config: Any) -> Any:
    """Convert a config dataclass into a JSON-serializable structure.

    Recurses through nested config dataclasses, converts :class:`pathlib.Path`
    values to strings, and turns ``tuple``/``set``/``frozenset`` into ``list`` so
    the result round-trips cleanly through ``json``/``yaml``/``toml``. Values that
    are already JSON-friendly pass through unchanged.

    Args:
        config (Any): A config dataclass instance (e.g.
            :class:`~ovannotators.DiskBackendConfig`), or any value reachable
            from one.

    Returns:
        Any: A structure of plain ``dict``/``list``/scalar values. Pair it with
        :func:`config_from_dict` to reconstruct the original config.
    """
    return _to_serializable(config)


def config_from_dict(config_cls: type[_T], data: dict[str, Any]) -> _T:
    """Reconstruct a config dataclass from a plain mapping.

    Inverse of :func:`config_to_dict`. Reconstructs nested config dataclasses,
    coerces :class:`pathlib.Path` fields from strings, and recurses into
    ``list``/``tuple``/``set``/``dict`` fields whose element type is itself a
    config dataclass. Keys absent from ``data`` fall back to the dataclass
    defaults.

    Args:
        config_cls (type): The target dataclass type (e.g.
            :class:`~ovannotators.DiskBackendConfig`).
        data (dict): A mapping produced by :func:`config_to_dict` or loaded from
            a serialized config file.

    Returns:
        The reconstructed ``config_cls`` instance.

    Raises:
        TypeError: If ``config_cls`` is not a dataclass type, or ``data``
            contains a key that is not a field of ``config_cls``.
    """
    if not (dataclasses.is_dataclass(config_cls) and isinstance(config_cls, type)):
        raise TypeError(f"{config_cls!r} is not a config dataclass type")
    # ``get_type_hints`` resolves the string annotations produced by
    # ``from __future__ import annotations`` in each config module.
    hints = get_type_hints(config_cls)
    kwargs = {key: _coerce(hints.get(key), value) for key, value in data.items()}
    # Unknown keys fall through to the constructor, which raises a clear TypeError.
    return config_cls(**kwargs)


def _to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_serializable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_serializable(v) for v in value]
    return value


def _coerce(hint: Any, value: Any) -> Any:
    if hint is None or value is None:
        return value

    origin = get_origin(hint)

    if origin in _UNION_ORIGINS:
        return _coerce_union(get_args(hint), value)

    if (
        dataclasses.is_dataclass(hint)
        and isinstance(hint, type)
        and isinstance(value, dict)
    ):
        return config_from_dict(hint, value)

    if isinstance(hint, type) and issubclass(hint, PurePath):
        return hint(value)

    if origin in (list, tuple, set, frozenset) and isinstance(value, (list, tuple)):
        args = get_args(hint)
        elem_hint = args[0] if args else None
        coerced = [_coerce(elem_hint, item) for item in value]
        if origin is list:
            return coerced
        return origin(coerced)

    if origin is dict and isinstance(value, dict):
        args = get_args(hint)
        val_hint = args[1] if len(args) == 2 else None
        return {key: _coerce(val_hint, item) for key, item in value.items()}

    return value


def _coerce_union(args: tuple[Any, ...], value: Any) -> Any:
    members = [arg for arg in args if arg is not type(None)]
    # A dict most likely targets a nested config member (e.g. Optional[SubConfig]).
    if isinstance(value, dict):
        for member in members:
            if dataclasses.is_dataclass(member) and isinstance(member, type):
                return config_from_dict(member, value)
    # Optional[X]: a single real member, so coerce against it (handles Optional[Path]).
    if len(members) == 1:
        return _coerce(members[0], value)
    # Ambiguous unions (e.g. Union[str, Path]) keep the serialized value as-is;
    # the consuming config/backend constructor normalizes it.
    return value
