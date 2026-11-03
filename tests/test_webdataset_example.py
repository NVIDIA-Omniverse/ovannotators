# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the CPU-only WebDataset capture example and check its output."""

from __future__ import annotations

import json
import runpy
import tarfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.cpu_only]

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "webdataset_capture.py"


def test_webdataset_capture_example(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLE), run_name="__main__")

    stdout = capsys.readouterr().out
    assert (
        "Packaged 5 samples from 2 episodes into 2 shards under "
        "_out/webdataset_capture/example-run"
    ) in stdout
    assert "webdataset reader decoded 5 samples" in stdout

    root = tmp_path / "_out" / "webdataset_capture"
    dataset = root / "example-run"
    assert sorted(p.name for p in dataset.iterdir()) == [
        "auxiliary",
        "manifest.json",
        "shard-000000.tar",
        "shard-000001.tar",
    ]
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert manifest["episodes"] == ["episode001", "episode002"]
    assert [s["episodes"] for s in manifest["shards"]] == [
        ["episode001"],
        ["episode002"],
    ]
    assert [s["samples"] for s in manifest["shards"]] == [3, 2]
    assert manifest["metadata"]["sample_schema"] == "coco-instance-v1"
    assert [a["path"] for a in manifest["auxiliary"]] == [
        "example-run/auxiliary/source-paths.json"
    ]
    with tarfile.open(root / manifest["shards"][1]["path"]) as archive:
        assert archive.getnames() == [
            f"example-run/episode-episode002/sample-{index}.{suffix}"
            for index in range(2)
            for suffix in ("json", "png", "metadata.json")
        ]

    # Re-running replaces the previous output instead of rejecting the namespace.
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert "Packaged 5 samples" in capsys.readouterr().out
