# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover dataset profiles and their producer-neutral capture requirements.

Discovery never constructs a writer, opens a stage, or initializes a renderer.
The returned configuration is the configuration to pass to the selected writer.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import PurePath
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from ovannotators.config import config_from_dict
from ovannotators.exceptions import ConfigurationError


@dataclass(frozen=True)
class DatasetProfile:
    """A supported dataset task or representation with recommended defaults.

    Attributes:
        name: Stable profile identifier within a writer.
        description: Task or representation supported by this profile.
        defaults: Configuration values applied before explicit overrides.
        constraints: Configuration values that identify this profile and
            cannot be overridden; choose a different profile to change them.
    """

    name: str
    description: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetOption:
    """A writer configuration option exposed by discovery.

    Attributes:
        name: Configuration field name.
        required: Whether the caller must supply this field.
        default: Resolved value, or ``None`` for an unresolved required field.
        choices: Allowed values for a Literal-typed option, if present.
        fixed_by_profile: Whether changing this value requires another profile.
    """

    name: str
    required: bool
    default: Any = None
    choices: tuple[Any, ...] = ()
    fixed_by_profile: bool = False


@dataclass(frozen=True)
class DatasetRequirements:
    """Capture inputs and serialization settings for a configured dataset.

    Render-var names are writer input keys (ovrtx ``sourceName`` values), not
    USD prim paths. A scene integration must bind views and author products.
    A minimum view count describes format input, not sufficient ML coverage.

    Attributes:
        writer_name: Exact writer registry identifier.
        profile: Selected named profile.
        render_vars: Required outputs per camera view.
        recommended_render_vars: Additional useful outputs, such as labels.
        minimum_camera_views: Minimum independent camera streams; ``None``
            means the caller's unresolved schema determines the count.
        camera_stream_ids: Explicitly configured independent camera streams.
        non_render_inputs: Simulation, policy, or other inputs outside rendering.
        scene_prerequisites: Scene authoring requirements for useful annotations.
        temporal_requirements: Episode, synchronization, or timing constraints.
        representations: Capture/output encoding, units, and scale descriptions.
        notes: Limits of the writer or discovery contract.
        config: Resolved writer config; ``None`` until required fields are supplied.
        config_type: Config dataclass type, if known.
        options: Supported configuration fields and their resolved defaults.
        required_options: Configuration fields still requiring caller input.
        sensor_requirements: Sensor input guidance for non-camera outputs.
    """

    writer_name: str
    profile: str
    render_vars: frozenset[str] = frozenset()
    recommended_render_vars: frozenset[str] = frozenset()
    minimum_camera_views: int | None = 1
    camera_stream_ids: tuple[str, ...] = ()
    non_render_inputs: tuple[str, ...] = ()
    scene_prerequisites: tuple[str, ...] = ()
    temporal_requirements: tuple[str, ...] = ()
    representations: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    config: Any = None
    config_type: type | None = None
    options: tuple[DatasetOption, ...] = ()
    required_options: tuple[str, ...] = ()
    sensor_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Definition:
    config_type: type
    profiles: tuple[DatasetProfile, ...]
    resolver: Callable[[str, Mapping[str, Any]], DatasetRequirements]


_definitions: dict[type, _Definition] = {}


def register_dataset_profiles(
    writer_class: type,
    config_type: type,
    profiles: tuple[DatasetProfile, ...],
    resolver: Callable[[str, Mapping[str, Any]], DatasetRequirements],
) -> None:
    """Register discovery for a frame or episode writer without changing its API.

    The first profile is the recommended default. Resolvers receive the profile
    name and resolved config values, including partial values when required
    options remain unresolved. They must perform no renderer or writer setup.
    Registration is keyed by class so replacing a writer cannot inherit another
    implementation's requirements accidentally.

    Args:
        writer_class: Class registered in either writer registry.
        config_type: Writer config dataclass type.
        profiles: Non-empty tuple with unique profile names, default first.
        resolver: Function returning the input and representation requirements.

    Raises:
        ConfigurationError: If metadata is invalid or the class is registered twice.
    """
    if not isinstance(config_type, type) or not is_dataclass(config_type):
        raise ConfigurationError("config_type must be a dataclass type")
    names = [profile.name for profile in profiles]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ConfigurationError("profile names must be non-empty and unique")
    keys = {item.name for item in fields(config_type) if item.init}
    for profile in profiles:
        unknown = (set(profile.defaults) | set(profile.constraints)) - keys
        if unknown:
            raise ConfigurationError(f"unknown profile options: {sorted(unknown)}")
        for key, value in profile.constraints.items():
            if key in profile.defaults and profile.defaults[key] != value:
                raise ConfigurationError(f"profile defaults conflict with {key!r}")
    if writer_class in _definitions:
        raise ConfigurationError(
            f"discovery already registered for {writer_class.__name__}"
        )
    _definitions[writer_class] = _Definition(config_type, deepcopy(profiles), resolver)


def _writer_class(writer_name: str) -> type:
    from ovannotators.profile_writers import ProfileWriterRegistry
    from ovannotators.writers import WriterRegistry
    from ovannotators._requirements_catalog import register_builtins

    register_builtins()
    frame = writer_name in WriterRegistry.list_names()
    episode = writer_name in ProfileWriterRegistry.list_names()
    if frame and episode:
        raise ConfigurationError(f"ambiguous writer identifier {writer_name!r}")
    if frame:
        return WriterRegistry.get(writer_name)
    if episode:
        return ProfileWriterRegistry.get(writer_name)
    raise ConfigurationError(f"unknown dataset writer {writer_name!r}")


def list_profiles(writer_name: str) -> list[str]:
    """List supported profiles, with the recommended default first.

    Args:
        writer_name: Exact identifier in either writer registry.

    Returns:
        Profile names, or ``['default']`` for a writer without discovery metadata.

    Raises:
        ConfigurationError: If the writer name is unknown or ambiguous.
    """
    definition = _definitions.get(_writer_class(writer_name))
    return [item.name for item in definition.profiles] if definition else ["default"]


def _valid_value(hint: Any, value: Any) -> bool:
    origin = get_origin(hint)
    args = get_args(hint)
    if hint is Any:
        return True
    if origin in (Union, types.UnionType):
        return any(_valid_value(member, value) for member in args)
    if origin is Literal:
        return any(type(value) is type(choice) and value == choice for choice in args)
    if hint is type(None):
        return value is None
    if hint is float:
        return type(value) in (float, int)
    if hint in (bool, int, str):
        return type(value) is hint
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            return False
        if not args:
            return True
        if origin is list or (len(args) == 2 and args[1] is Ellipsis):
            return all(_valid_value(args[0], item) for item in value)
        return len(args) == len(value) and all(
            _valid_value(member, item) for member, item in zip(args, value)
        )
    if origin in (dict, Mapping):
        return isinstance(value, Mapping) and (
            not args
            or all(
                _valid_value(args[0], key) and _valid_value(args[1], item)
                for key, item in value.items()
            )
        )
    if isinstance(hint, type):
        if issubclass(hint, PurePath):
            return isinstance(value, (str, PurePath))
        if is_dataclass(hint):
            return isinstance(value, (hint, Mapping))
        return isinstance(value, hint)
    return True


def describe_requirements(
    writer_name: str,
    profile: str | None = None,
    *,
    config: Any = None,
    **overrides: Any,
) -> DatasetRequirements:
    """Resolve a dataset's capture requirements and matching writer options.

    Config may be the matching config dataclass or a partial mapping. Named
    profile defaults apply first, explicit config second, keyword overrides
    last. Profile-defining constraints cannot be overridden. Discovery returns
    unresolved required fields instead of inventing scene units or robot policy
    semantics. No writer is instantiated and no optional codec is imported.

    Args:
        writer_name: Exact identifier in either writer registry.
        profile: Supported profile name; omitted selects a profile matching explicit options,
            or the recommended default when no task options are supplied.
        config: Matching dataclass instance or partial configuration mapping.
        **overrides: Writer config fields overriding the recommended values.

    Returns:
        Capture requirements, available options, and a config ready for the writer
        when ``required_options`` is empty.

    Raises:
        ConfigurationError: For unknown profiles, invalid options, conflicting
            profile constraints, or config types.
    """
    writer_class = _writer_class(writer_name)
    definition = _definitions.get(writer_class)
    if definition is None:
        if profile not in (None, "default") or config is not None or overrides:
            raise ConfigurationError("this writer has no registered option discovery")
        return DatasetRequirements(
            writer_name,
            "default",
            render_vars=frozenset(getattr(writer_class, "required_render_vars", ())),
            minimum_camera_views=None,
            notes=(
                "Only class-level inputs are known; register_dataset_profiles for complete discovery.",
            ),
        )
    explicit: dict[str, Any] = {}
    if config is not None:
        if isinstance(config, definition.config_type):
            explicit = {
                item.name: deepcopy(getattr(config, item.name))
                for item in fields(definition.config_type)
                if item.init
            }
        elif isinstance(config, Mapping):
            explicit = deepcopy(dict(config))
        else:
            raise ConfigurationError(
                f"config must be {definition.config_type.__name__} or a mapping"
            )
    explicit.update(deepcopy(overrides))
    if profile is None:
        candidates = [
            item
            for item in definition.profiles
            if all(
                key not in explicit or explicit[key] == value
                for key, value in item.constraints.items()
            )
        ]
        if not candidates:
            raise ConfigurationError("options do not match any supported profile")
        profile = candidates[0].name
    selected = next(
        (item for item in definition.profiles if item.name == profile),
        None,
    )
    if selected is None:
        raise ConfigurationError(
            f"unknown profile {profile!r}; available: {list_profiles(writer_name)}"
        )
    values: dict[str, Any] = {}
    config_fields = [item for item in fields(definition.config_type) if item.init]
    for item in config_fields:
        if item.default is not MISSING:
            values[item.name] = deepcopy(item.default)
        elif item.default_factory is not MISSING:
            values[item.name] = item.default_factory()
    values.update(deepcopy(dict(selected.defaults)))
    values.update(deepcopy(dict(selected.constraints)))
    values.update(explicit)
    unknown = set(values) - {item.name for item in config_fields}
    if unknown:
        raise ConfigurationError(
            f"unknown options for {writer_name}: {sorted(unknown)}"
        )
    for key, value in selected.constraints.items():
        if values[key] != value:
            raise ConfigurationError(
                f"profile {selected.name!r} requires {key}={value!r}"
            )
    hints = get_type_hints(definition.config_type)
    options = []
    for item in config_fields:
        hint = hints.get(item.name)
        choices = get_args(hint) if get_origin(hint) is Literal else ()
        if item.name in values and not _valid_value(hint, values[item.name]):
            raise ConfigurationError(
                f"invalid type or value for {item.name}: expected {hint}"
            )
        options.append(
            DatasetOption(
                item.name,
                item.name not in values,
                deepcopy(values.get(item.name)),
                choices,
                item.name in selected.constraints,
            )
        )
    missing = tuple(item.name for item in options if item.required)
    for key, value in list(values.items()):
        hint = hints.get(key)
        if isinstance(hint, type) and is_dataclass(hint) and isinstance(value, Mapping):
            try:
                values[key] = config_from_dict(hint, dict(value))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"invalid {key}: {exc}") from exc
    resolved = None
    if not missing:
        try:
            resolved = config_from_dict(definition.config_type, values)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid {writer_name} options: {exc}") from exc
        values = {item.name: getattr(resolved, item.name) for item in config_fields}
    try:
        result = definition.resolver(selected.name, values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid {writer_name} options: {exc}") from exc

    return replace(
        result,
        writer_name=writer_name,
        profile=selected.name,
        config=resolved,
        config_type=definition.config_type,
        options=tuple(options),
        required_options=missing,
    )
