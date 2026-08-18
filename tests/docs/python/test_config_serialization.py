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

"""Tested snippets for serializing ``*Config`` dataclasses to/from files.

Every illustrative region is wrapped in ``# [snippet:doc-...]`` markers and is
referenced from ``skills/new-writer/SKILL.md`` and ``skills/new-backend/SKILL.md``
via ``> **Source:**`` directives. These tests are the single source of truth for
the skills' code. Runs CPU-only, so it joins the default (non-GPU) test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ovannotators import (
    BackendRegistry,
    COCOWriterConfig,
    DiskBackend,
    DiskBackendConfig,
    config_from_dict,
    config_to_dict,
)


def test_config_save_and_load(tmp_path):
    """A backend config round-trips through a JSON file and rebuilds the backend."""
    config_path = tmp_path / "backend.json"

    # [snippet:doc-config-save-load]
    import json

    from ovannotators import DiskBackendConfig, config_from_dict, config_to_dict

    # Save: config_to_dict() yields JSON-friendly values (pathlib.Path -> str,
    # nested configs -> dicts), so it dumps cleanly.
    config = DiskBackendConfig(root_dir="./output/coco", overwrite=True)
    config_path.write_text(json.dumps(config_to_dict(config), indent=2))

    # Load: config_from_dict() rebuilds the dataclass, restoring Path fields and
    # any nested configs.
    loaded = config_from_dict(DiskBackendConfig, json.loads(config_path.read_text()))
    assert loaded == config
    # [/snippet:doc-config-save-load]


def test_config_registry_dispatch(tmp_path):
    """A name + config payload selects the backend via the registry and builds it."""
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        '{"name": "disk", "config": {"root_dir": "out", "overwrite": true}}'
    )

    # [snippet:doc-config-registry]
    import json

    # A config file can pair a registry name with its config payload, so the
    # whole pipeline is data-driven. Map each registry name to its config class.
    CONFIG_CLASSES = {"disk": DiskBackendConfig}

    spec = json.loads(config_path.read_text())
    config = config_from_dict(CONFIG_CLASSES[spec["name"]], spec["config"])
    backend = BackendRegistry.create(spec["name"], config)
    # [/snippet:doc-config-registry]

    assert isinstance(backend, DiskBackend)
    assert backend.exists is not None


def test_config_to_dict_is_json_friendly():
    """Path values become strings and the dict matches the field layout."""
    config = DiskBackendConfig(root_dir=Path("/data/run-1"), overwrite=True)
    assert config_to_dict(config) == {"root_dir": "/data/run-1", "overwrite": True}


def test_config_from_dict_uses_defaults_for_missing_keys():
    """Keys absent from the mapping fall back to the dataclass defaults."""
    loaded = config_from_dict(DiskBackendConfig, {"root_dir": "out"})
    assert loaded == DiskBackendConfig(root_dir="out", overwrite=False)


def test_config_from_dict_rejects_unknown_keys():
    """Unknown keys surface as a clear constructor TypeError."""
    import pytest

    with pytest.raises(TypeError):
        config_from_dict(DiskBackendConfig, {"root_dir": "out", "bogus": 1})


def test_real_writer_config_roundtrips():
    """A built-in writer config round-trips unchanged through the helpers."""
    config = COCOWriterConfig()
    assert config_from_dict(COCOWriterConfig, config_to_dict(config)) == config


@dataclass
class _Retry:
    attempts: int = 3
    backoff: float = 0.5


@dataclass
class _ClientConfig:
    """A config with a nested dataclass and a Path field, for coverage."""

    endpoint: str
    cache_dir: Optional[Path] = None
    retry: _Retry = field(default_factory=_Retry)
    tags: tuple[str, ...] = ()


def test_nested_and_path_fields_roundtrip():
    """Nested configs, Optional[Path], and tuple fields survive the round-trip."""
    config = _ClientConfig(
        endpoint="https://example.test",
        cache_dir=Path("/tmp/cache"),
        retry=_Retry(attempts=5, backoff=1.0),
        tags=("a", "b"),
    )

    as_dict = config_to_dict(config)
    assert as_dict == {
        "endpoint": "https://example.test",
        "cache_dir": "/tmp/cache",
        "retry": {"attempts": 5, "backoff": 1.0},
        "tags": ["a", "b"],
    }

    restored = config_from_dict(_ClientConfig, as_dict)
    assert restored == config
    assert isinstance(restored.cache_dir, Path)
    assert isinstance(restored.retry, _Retry)
    assert restored.tags == ("a", "b")
