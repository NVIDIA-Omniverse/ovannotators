# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime SBOM and third-party notices match the declared dependencies."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Developer-only extras: test runners, docs tooling, and the NVIDIA renderer
# packages used by examples and GPU tests. They are not runtime dependencies.
DEVELOPER_EXTRAS = {"test", "dev-cpu", "dev"}


def _name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def _sbom():
    return json.loads(
        (ROOT / ".osrb/runtime-dependencies.json").read_text(encoding="utf-8")
    )


def _declared() -> dict[str, str]:
    project = _project()
    scopes = {_name(r): "required" for r in project["dependencies"]}
    for extra, requirements in project["optional-dependencies"].items():
        if extra in DEVELOPER_EXTRAS:
            continue
        for requirement in requirements:
            name = _name(requirement)
            if name != "ovannotators":
                scopes.setdefault(name, "optional")
    return scopes


def test_sbom_lists_every_runtime_dependency_with_its_scope():
    components = {_name(c["name"]): c["scope"] for c in _sbom()["components"]}
    assert components == _declared()


def test_sbom_describes_this_release():
    component = _sbom()["metadata"]["component"]
    version = _project()["version"]
    assert component["version"] == version
    assert component["bom-ref"] == f"pkg:pypi/ovannotators@{version}"


def test_notices_cover_every_open_source_runtime_dependency():
    open_source = {
        _name(c["name"])
        for c in _sbom()["components"]
        if all(
            "expression" in entry or "id" in entry.get("license", {})
            for entry in c["licenses"]
        )
    }
    notices = (ROOT / "NOTICE-3rdparty.txt").read_text(encoding="utf-8")
    listed = {
        _name(n)
        for n in re.findall(r"^Component : (\S+)$", notices, flags=re.MULTILINE)
    }
    assert listed == open_source
