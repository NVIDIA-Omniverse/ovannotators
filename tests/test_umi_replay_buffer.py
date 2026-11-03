# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the UMI ReplayBuffer profile."""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

if os.environ.get("CI"):
    # Every CI lane installs dev-cpu, which includes the umi extra, so a broken
    # extra must fail here instead of silently skipping these tests.
    import zarr
else:
    zarr = pytest.importorskip("zarr", reason="install ovannotators[umi]")

from ovannotators import (  # noqa: E402
    ProfileWriterRegistry,
    UMIFieldSpec,
    UMIReplayBufferWriter,
    UMIReplayBufferWriterConfig,
)
from ovannotators.backends import DiskBackend, DiskBackendConfig  # noqa: E402


def _config() -> UMIReplayBufferWriterConfig:
    return UMIReplayBufferWriterConfig(
        frequency_hz=10,
        fields=(
            UMIFieldSpec(
                "camera.front",
                "camera0_rgb",
                "observation",
                "uint8",
                (3, 4, 3),
                rgb=True,
            ),
            UMIFieldSpec(
                "observation.state",
                "robot0_state",
                "observation",
                "float32",
                (2,),
                ("joint.position", "gripper.position"),
                ("rad", "m"),
                "absolute",
                "robot_base",
            ),
            UMIFieldSpec(
                "action",
                "action",
                "policy_signal",
                "float32",
                (2,),
                ("joint.target", "gripper.target"),
                ("rad", "m"),
                "absolute",
                "robot_base",
            ),
        ),
    )


def _write(tmp_path: Path, priority_profile_fixture):
    backend = DiskBackend(DiskBackendConfig(tmp_path))
    writer = UMIReplayBufferWriter(_config())
    for episode, samples in zip(
        priority_profile_fixture.episodes, priority_profile_fixture.samples, strict=True
    ):
        writer.begin_episode(episode, backend)
        for sample in samples:
            writer.write_sample(sample, backend)
        writer.end_episode(backend)
    writer.finalize(backend)
    return writer, backend


@pytest.mark.unit
def test_config_rejects_ambiguous_mappings():
    with pytest.raises(ValueError, match="fields"):
        UMIReplayBufferWriterConfig(10, ())
    with pytest.raises(ValueError, match="output keys"):
        replace(
            _config(),
            fields=(
                _config().fields[0],
                replace(_config().fields[1], output_key="camera0_rgb"),
            ),
        )
    with pytest.raises(ValueError, match="RGB fields"):
        UMIFieldSpec(
            "camera", "camera0_rgb", "observation", "float32", (2, 3, 3), rgb=True
        )


@pytest.mark.integration
def test_writes_canonical_zarr_tree_and_semantics(
    tmp_path: Path, priority_profile_fixture
):
    writer, backend = _write(tmp_path, priority_profile_fixture)
    assert writer.validate(backend).structural == "passed"
    payload = backend.read("dataset.zarr.zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert "meta/episode_ends/.zarray" in archive.namelist()
        assert "data/camera0_rgb/.zarray" in archive.namelist()

    path = tmp_path / "readback.zarr.zip"
    path.write_bytes(payload)
    store = zarr.ZipStore(path, mode="r")
    try:
        root = zarr.group(store=store)
        assert root["meta/episode_ends"][:].tolist() == [3, 6, 9]
        assert root["data/camera0_rgb"].shape == (9, 3, 4, 3)
        assert root["data/camera0_rgb"].chunks == (1, 3, 4, 3)
        assert root["data/camera0_rgb"].compressor.codec_id == "imagecodecs_jpegxl"
        assert root["data/robot0_state"][-1].tolist() == pytest.approx([202.1, 202.2])
        assert root["data/action"][-1].tolist() == pytest.approx([212.1, 212.2])
        assert root["data/timestamp_ns"][:3].tolist() == [
            5_000_000_000,
            5_100_000_003,
            5_200_000_006,
        ]
        manifest = {
            "arrays": {
                name: {
                    "chunks": list(array.chunks),
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
                for name, array in (
                    ("data/action", root["data/action"]),
                    ("data/camera0_rgb", root["data/camera0_rgb"]),
                    ("data/robot0_state", root["data/robot0_state"]),
                    ("data/timestamp_ns", root["data/timestamp_ns"]),
                    ("meta/episode_ends", root["meta/episode_ends"]),
                )
            },
            "episode_ends": root["meta/episode_ends"][:].tolist(),
            "status": "candidate_pending_human_approval",
        }
        golden = json.loads(
            (
                Path(__file__).parent
                / "golden/umi_replay_buffer/candidate/manifest.json"
            ).read_text()
        )
        assert manifest == golden
    finally:
        store.close()
    assert backend.exists("_ovannotators/_SUCCESS")


@pytest.mark.unit
def test_rejects_missing_signal_before_mutation(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path))
    writer = UMIReplayBufferWriter(_config())
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    sample = priority_profile_fixture.samples[0][0]
    with pytest.raises(ValueError, match="missing required signal"):
        writer.write_sample(replace(sample, policy_signals={}), backend)
    writer.write_sample(sample, backend)


@pytest.mark.unit
def test_rejects_off_grid_timestamp(tmp_path: Path, priority_profile_fixture):
    backend = DiskBackend(DiskBackendConfig(tmp_path))
    writer = UMIReplayBufferWriter(_config())
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    sample = replace(priority_profile_fixture.samples[0][0], timestamp_ns=5_000_200_000)
    with pytest.raises(ValueError, match="off the UMI timeline"):
        writer.write_sample(sample, backend)


@pytest.mark.integration
def test_finalize_is_idempotent(tmp_path: Path, priority_profile_fixture):
    writer, backend = _write(tmp_path, priority_profile_fixture)
    paths = backend.list()
    writer.finalize(backend)
    assert backend.list() == paths


@pytest.mark.unit
def test_registry_and_direct_import_agree():
    assert ProfileWriterRegistry.get("umi_replay_buffer") is UMIReplayBufferWriter


@pytest.mark.unit
def test_finalize_does_not_persist_an_invalid_payload(
    tmp_path: Path, priority_profile_fixture, monkeypatch
):
    import ovannotators.profile_writers.umi_replay_buffer as umi
    from ovannotators.profile_writers import ConformanceResult

    backend = DiskBackend(DiskBackendConfig(tmp_path))
    writer = UMIReplayBufferWriter(_config())
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, backend)
    writer.end_episode(backend)
    with monkeypatch.context() as patch:
        patch.setattr(
            umi,
            "_validate_zip_bytes",
            lambda payload: ConformanceResult(
                "passed", "failed", messages=("forced failure",)
            ),
        )
        with pytest.raises(ValueError, match="forced failure"):
            writer.finalize(backend)
    assert backend.list() == []
    writer.finalize(backend)
    assert backend.exists("_ovannotators/_SUCCESS")


@pytest.mark.integration
def test_validate_preserves_zip_result_when_success_marker_is_missing(
    tmp_path: Path, priority_profile_fixture
):
    writer, backend = _write(tmp_path, priority_profile_fixture)
    (tmp_path / "_ovannotators" / "_SUCCESS").unlink()
    result = writer.validate(backend)
    assert (result.serialization, result.structural) == ("passed", "failed")
    assert result.messages == ("missing _SUCCESS",)
    assert result.evidence == {"validator": "umi_replay_buffer@1"}

    (tmp_path / "dataset.zarr.zip").write_bytes(b"not a zip")
    result = writer.validate(backend)
    assert result.structural == "failed"
    assert any("Zarr validation failed" in item for item in result.messages)
    assert result.messages[-1] == "missing _SUCCESS"
