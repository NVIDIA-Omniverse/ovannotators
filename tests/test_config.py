# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the config serialization helpers in ``ovannotators.config``.

Exercises ``config_to_dict`` / ``config_from_dict`` round-trips across the
container and union shapes the coercion logic special-cases: nested dataclasses,
``Path`` fields, ``list``/``tuple``/``set`` of nested configs, typed ``dict``
fields, and ``Optional[...]`` unions. The local ``@dataclass`` fixtures stand in
for real ``*Config`` types — the helpers are generic over any config dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest

from ovannotators.config import config_from_dict, config_to_dict


@dataclass
class _Leaf:
    out_dir: Path = Path("/tmp/leaf")
    name: str = "leaf"


@dataclass
class _Parent:
    leaf: _Leaf = field(default_factory=_Leaf)
    leaves: list[_Leaf] = field(default_factory=list)
    leaf_tuple: tuple[_Leaf, ...] = ()
    leaf_set: frozenset[str] = frozenset()
    by_name: dict[str, _Leaf] = field(default_factory=dict)
    meta: dict[str, int] = field(default_factory=dict)
    optional_leaf: _Leaf | None = None
    optional_path: Path | None = None
    ambiguous: str | int = "x"


def test_config_to_dict_serializes_nested_path_and_dict_fields():
    """A nested dataclass with Path and dict fields becomes a pure dict/list/scalar
    tree (covers the dict branch of _to_serializable)."""
    parent = _Parent(
        leaf=_Leaf(out_dir=Path("/data/a"), name="a"),
        by_name={"b": _Leaf(out_dir=Path("/data/b"), name="b")},
        meta={"count": 3},
    )

    result = config_to_dict(parent)

    assert result["leaf"] == {"out_dir": "/data/a", "name": "a"}, result["leaf"]
    assert result["by_name"]["b"] == {"out_dir": "/data/b", "name": "b"}, result[
        "by_name"
    ]
    assert result["meta"] == {"count": 3}, result["meta"]
    assert isinstance(result["leaf"]["out_dir"], str), "Path must serialize to str"


def test_config_to_dict_turns_tuple_and_set_into_lists():
    """tuple/set/frozenset collapse to list so the result round-trips through JSON."""
    parent = _Parent(leaf_tuple=(_Leaf(name="x"),), leaf_set=frozenset({"a", "b"}))

    result = config_to_dict(parent)

    assert isinstance(result["leaf_tuple"], list), result["leaf_tuple"]
    assert isinstance(result["leaf_set"], list), result["leaf_set"]
    assert sorted(result["leaf_set"]) == ["a", "b"], result["leaf_set"]


def test_config_from_dict_rebuilds_list_of_nested_configs():
    """A list field of nested configs is reconstructed element-wise (origin is list)."""
    data = {
        "leaves": [
            {"out_dir": "/data/0", "name": "n0"},
            {"out_dir": "/data/1", "name": "n1"},
        ]
    }

    parent = config_from_dict(_Parent, data)

    assert [leaf.name for leaf in parent.leaves] == ["n0", "n1"], parent.leaves
    assert parent.leaves[0].out_dir == Path("/data/0"), parent.leaves[0].out_dir
    assert isinstance(parent.leaves[0], _Leaf), type(parent.leaves[0])


def test_config_from_dict_rebuilds_tuple_field_preserving_type():
    """A tuple field of nested configs reconstructs back to a tuple, not a list
    (covers the non-list origin return path)."""
    data = {"leaf_tuple": [{"out_dir": "/data/t", "name": "t"}]}

    parent = config_from_dict(_Parent, data)

    assert isinstance(parent.leaf_tuple, tuple), type(parent.leaf_tuple)
    assert parent.leaf_tuple[0].name == "t", parent.leaf_tuple


def test_config_from_dict_coerces_typed_dict_values():
    """A typed dict field (dict[str, _Leaf]) reconstructs each value as the config type."""
    data = {"by_name": {"k": {"out_dir": "/data/k", "name": "k"}}}

    parent = config_from_dict(_Parent, data)

    assert isinstance(parent.by_name["k"], _Leaf), type(parent.by_name["k"])
    assert parent.by_name["k"].out_dir == Path("/data/k"), parent.by_name["k"].out_dir


def test_config_from_dict_coerces_optional_nested_config_from_dict():
    """An Optional[_Leaf] union given a dict reconstructs the nested config member."""
    parent = config_from_dict(
        _Parent, {"optional_leaf": {"out_dir": "/data/o", "name": "o"}}
    )

    assert isinstance(parent.optional_leaf, _Leaf), type(parent.optional_leaf)
    assert parent.optional_leaf.name == "o", parent.optional_leaf


def test_config_from_dict_coerces_optional_path_from_string():
    """An Optional[Path] union with a single real member coerces a string to Path."""
    parent = config_from_dict(_Parent, {"optional_path": "/data/p"})

    assert parent.optional_path == Path("/data/p"), parent.optional_path


def test_config_round_trip_preserves_full_nested_structure():
    """to_dict then from_dict reconstructs an equal config — the core guarantee."""
    original = _Parent(
        leaf=_Leaf(out_dir=Path("/data/root"), name="root"),
        leaves=[_Leaf(name="a"), _Leaf(name="b")],
        leaf_tuple=(_Leaf(name="t"),),
        by_name={"k": _Leaf(name="k")},
        meta={"n": 1},
        optional_leaf=_Leaf(name="opt"),
        optional_path=Path("/data/opt"),
    )

    rebuilt = config_from_dict(_Parent, config_to_dict(original))

    assert rebuilt == original, (
        f"round-trip changed the config: {rebuilt} != {original}"
    )


def test_config_from_dict_raises_type_error_on_non_dataclass_type():
    """config_from_dict rejects a non-dataclass target with a clear TypeError."""
    with pytest.raises(TypeError, match="not a config dataclass"):
        config_from_dict(dict, {"a": 1})


def test_config_from_dict_raises_type_error_on_unknown_field():
    """An unknown key falls through to the constructor and raises TypeError."""
    with pytest.raises(TypeError):
        config_from_dict(_Leaf, {"not_a_field": 1})


def test_config_from_dict_keeps_ambiguous_union_value_as_is():
    """A union with multiple real members (e.g. str | int) and a non-dict value can't be
    disambiguated, so the serialized value passes through for the constructor to normalize."""
    parent = config_from_dict(_Parent, {"ambiguous": "kept"})

    assert parent.ambiguous == "kept", parent.ambiguous


def test_config_to_dict_passes_through_plain_path_subclasses():
    """Any PurePath subclass (not just pathlib.Path) serializes to its string form."""
    assert config_to_dict(PurePosixPath("/x/y")) == "/x/y"


def test_config_to_dict_leaves_scalars_unchanged():
    """JSON-friendly scalars pass through untouched."""
    assert config_to_dict(42) == 42
    assert config_to_dict("s") == "s"
    assert config_to_dict(None) is None
