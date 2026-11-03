# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU structural and semantic tests for the GR00T LeRobot v2 profile."""

from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ovannotators import (
    ConfigurationError,
    GrootActionModality,
    GrootLeRobotV2Writer,
    GrootLeRobotV2WriterConfig,
    GrootNormalizationConfig,
    GrootStateModality,
    GrootVideoModality,
    NamedTensor,
    ProfileWriterRegistry,
    config_from_dict,
    config_to_dict,
    validate_groot_lerobot_v2,
)


def _config(*, horizon: int = 2) -> GrootLeRobotV2WriterConfig:
    return GrootLeRobotV2WriterConfig(
        embodiment="fixture_arm",
        video_modalities=(
            GrootVideoModality("front", "camera.front"),
            GrootVideoModality("wrist", "camera.wrist"),
        ),
        state_signal="observation.state",
        state_modalities=(
            GrootStateModality("arm", 0, 1, ("joint.position",), ("rad",)),
            GrootStateModality("gripper", 1, 2, ("gripper.position",), ("m",)),
        ),
        action_signal="action",
        action_modalities=(
            GrootActionModality(
                "arm_action",
                0,
                1,
                ("joint.target",),
                ("rad",),
                "relative",
                "non_eef",
                "default",
                state_key="arm",
            ),
            GrootActionModality(
                "gripper_action",
                1,
                2,
                ("gripper.target",),
                ("m",),
                "absolute",
                "non_eef",
                "default",
                state_key="gripper",
            ),
        ),
        action_horizon=horizon,
        normalization=GrootNormalizationConfig(
            state=(("arm", "min_max"), ("gripper", "mean_std")),
            action=(("arm_action", "min_max"), ("gripper_action", "min_max")),
            use_percentiles=True,
            clip_outliers=True,
        ),
        relative_action_policy="groot_processor",
        fps=10,
        episodes_per_chunk=1,
    )


_CAMERAS = ("camera.front", "camera.wrist")


@pytest.fixture
def priority_profile_fixture(priority_profile_fixture):
    """Pad the shared 3x4 camera frames to 4x4: H.264 4:2:0 needs even sizes."""

    def pad(tensor):
        data = np.pad(np.asarray(tensor.data), ((0, 1), (0, 0), (0, 0)), mode="edge")
        return replace(tensor, data=data)

    episodes = tuple(
        replace(
            episode,
            streams=tuple(
                replace(stream, shape=(4, 4, 3))
                if stream.stream_id in _CAMERAS
                else stream
                for stream in episode.streams
            ),
        )
        for episode in priority_profile_fixture.episodes
    )
    samples = tuple(
        tuple(
            replace(
                sample,
                observations={
                    key: pad(value) if key in _CAMERAS else value
                    for key, value in sample.observations.items()
                },
            )
            for sample in episode_samples
        )
        for episode_samples in priority_profile_fixture.samples
    )
    return replace(priority_profile_fixture, episodes=episodes, samples=samples)


def _restore_dropped_wrist(sample):
    if "camera.wrist" in sample.observations:
        return sample
    observations = dict(sample.observations)
    observations["camera.wrist"] = NamedTensor(
        np.full((4, 4, 3), 152, dtype=np.uint8),
        representation="rgb8",
        coordinate_frame="wrist",
    )
    return replace(sample, observations=observations, missing_data={})


def _write_fixture(writer, backend, fixture) -> None:
    for episode, samples in zip(fixture.episodes, fixture.samples):
        writer.begin_episode(episode, backend)
        for sample in samples:
            writer.write_sample(_restore_dropped_wrist(sample), backend)
        writer.end_episode(backend)
    writer.finalize(backend)


@pytest.mark.unit
def test_config_is_explicit_validated_and_serializable():
    config = _config()
    restored = config_from_dict(GrootLeRobotV2WriterConfig, config_to_dict(config))

    assert restored == config
    assert restored.action_horizon == 2
    with pytest.raises(ConfigurationError, match="exactly when"):
        replace(config, relative_action_policy="disabled")
    with pytest.raises(ConfigurationError, match="ordered and contiguous"):
        replace(
            config,
            state_modalities=(
                GrootStateModality("arm", 0, 1, ("joint.position",), ("rad",)),
                GrootStateModality("gripper", 2, 3, ("gripper.position",), ("m",)),
            ),
        )


@pytest.mark.unit
def test_config_rejects_ambiguous_or_incomplete_modality_contracts():
    config = _config()
    with pytest.raises(ConfigurationError, match="required"):
        GrootVideoModality("", "camera.front")
    with pytest.raises(ConfigurationError, match="invalid modality slice"):
        GrootStateModality("arm", -1, 1, ("joint.position",), ("rad",))
    with pytest.raises(ConfigurationError, match="width"):
        GrootStateModality("arm", 0, 2, ("joint.position",), ("rad",))
    with pytest.raises(ConfigurationError, match="representation"):
        replace(config.action_modalities[0], representation="invalid")
    with pytest.raises(ConfigurationError, match="control_type"):
        replace(config.action_modalities[0], control_type="invalid")
    with pytest.raises(ConfigurationError, match="action format"):
        replace(config.action_modalities[0], action_format="invalid")
    with pytest.raises(ConfigurationError, match="state normalization"):
        replace(
            config,
            normalization=replace(
                config.normalization,
                state=(("arm", "invalid"), ("gripper", "mean_std")),
            ),
        )
    with pytest.raises(ConfigurationError, match="must be explicit"):
        replace(config, video_modalities=())
    with pytest.raises(ConfigurationError, match="must be positive"):
        replace(config, action_horizon=0)
    with pytest.raises(ConfigurationError, match="unknown state key"):
        replace(
            config,
            action_modalities=(
                replace(config.action_modalities[0], state_key="missing"),
                config.action_modalities[1],
            ),
        )
    with pytest.raises(ConfigurationError, match="width must match"):
        replace(
            config,
            action_modalities=(
                GrootActionModality(
                    "arm_action",
                    0,
                    2,
                    ("joint.target", "joint.target.extra"),
                    ("rad", "rad"),
                    "relative",
                    "non_eef",
                    "default",
                    state_key="arm",
                ),
                replace(config.action_modalities[1], start=2, end=3),
            ),
            normalization=replace(
                config.normalization,
                action=(("arm_action", "min_max"), ("gripper_action", "min_max")),
            ),
        )
    with pytest.raises(ConfigurationError, match="relative EEF actions"):
        replace(
            config,
            action_modalities=(
                replace(config.action_modalities[0], control_type="eef"),
                config.action_modalities[1],
            ),
        )
    with pytest.raises(ConfigurationError, match="normalization.state"):
        replace(
            config,
            normalization=replace(config.normalization, state=(("arm", "min_max"),)),
        )
    with pytest.raises(ConfigurationError, match="normalization.action"):
        replace(
            config,
            normalization=replace(
                config.normalization, action=(("arm_action", "min_max"),)
            ),
        )
    with pytest.raises(ConfigurationError, match="video modality keys must be unique"):
        replace(
            config,
            video_modalities=(
                GrootVideoModality("front", "camera.front"),
                GrootVideoModality("front", "camera.wrist"),
            ),
        )


@pytest.mark.unit
def test_registry_and_direct_import_are_independent_from_legacy_lerobot():
    assert ProfileWriterRegistry.get("groot_lerobot_v2") is GrootLeRobotV2Writer
    assert (
        ProfileWriterRegistry.create("groot_lerobot_v2", _config()).name
        == "groot_lerobot_v2"
    )
    assert "lerobot" != GrootLeRobotV2Writer.name


@pytest.mark.integration
def test_writer_emits_pinned_layout_statistics_and_semantics(
    disk_backend, priority_profile_fixture, monkeypatch
):
    def fake_video(frames, fps):
        assert fps == 10
        return b"mock-mp4:" + bytes([int(frames[0][0, 0, 0])])

    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video", fake_video
    )
    writer = GrootLeRobotV2Writer(_config())
    _write_fixture(writer, disk_backend, priority_profile_fixture)

    paths = disk_backend.list()
    expected_layout = json.loads(
        Path("tests/golden/groot_lerobot_v2/candidate/layout.json").read_text()
    )
    for expected in expected_layout["required_paths"]:
        assert expected in paths
    assert paths.count("_ovannotators/_SUCCESS") == 1

    info = json.loads(disk_backend.read("meta/info.json"))
    assert info["codebase_version"] == "v2.1"
    assert info["codebase_version"] == expected_layout["codebase_version"]
    assert info["profile"] == "groot_lerobot_v2"
    assert info["groot_release"] == expected_layout["groot_release"]
    assert info["total_episodes"] == 3
    assert info["total_tasks"] == 2
    assert info["chunks_size"] == 1

    modality = json.loads(disk_backend.read("meta/modality.json"))
    assert modality["video"]["front"]["stream_id"] == "camera.front"
    assert modality["video"]["wrist"]["stream_id"] == "camera.wrist"
    assert modality["state"]["gripper"]["start"] == 1
    assert modality["action"]["arm_action"]["representation"] == "RELATIVE"
    assert modality["ovannotators_config"]["action_delta_indices"] == [0, 1]

    table = pq.read_table(
        io.BytesIO(disk_backend.read("data/chunk-000/episode_000000.parquet"))
    )
    state_type = table.schema.field("observation.state").type
    action_type = table.schema.field("action").type
    assert pa.types.is_fixed_size_list(state_type)
    assert state_type.list_size == 2
    assert pa.types.is_float32(state_type.value_type)
    assert pa.types.is_fixed_size_list(action_type)
    assert action_type.list_size == 2
    assert pa.types.is_float32(action_type.value_type)
    assert pa.types.is_float32(table.schema.field("timestamp").type)
    np.testing.assert_allclose(table["observation.state"].to_pylist()[2], [2.1, 2.2])
    np.testing.assert_allclose(table["action"].to_pylist()[2], [12.1, 12.2])
    assert table["timestamp_ns"].to_pylist() == [
        5_000_000_000,
        5_100_000_003,
        5_200_000_006,
    ]
    assert disk_backend.read(
        "videos/chunk-000/observation.images.front/episode_000000.mp4"
    ) != disk_backend.read(
        "videos/chunk-000/observation.images.wrist/episode_000000.mp4"
    )
    video_feature = info["features"]["observation.images.front"]
    assert {
        key: info["features"][key]["names"] for key in expected_layout["feature_names"]
    } == expected_layout["feature_names"]
    assert video_feature["info"]["video.height"] == 4
    assert video_feature["info"]["video.width"] == 4
    assert video_feature["info"]["video.channels"] == 3

    stats = json.loads(disk_backend.read("meta/stats.json"))
    assert set(stats["action"]) == {"mean", "std", "min", "max", "q01", "q99"}
    relative = json.loads(disk_backend.read("meta/relative_stats.json"))
    assert np.asarray(relative["arm_action"]["mean"]).shape == (2, 1)
    loss = json.loads(disk_backend.read("_ovannotators/loss_report.json"))
    assert any("relative" in item for item in loss["transformed"])
    conformance = json.loads(disk_backend.read("_ovannotators/conformance.json"))
    assert conformance["structural"] == "passed"
    assert conformance["canonical_loader"] == "not_run"
    profile = json.loads(disk_backend.read("_ovannotators/profile.json"))
    assert profile["schema_version"] == "2.1"


@pytest.mark.unit
def test_sample_rejects_missing_modalities_and_nonabsolute_actions(
    disk_backend, priority_profile_fixture
):
    writer = GrootLeRobotV2Writer(_config())
    writer.begin_episode(priority_profile_fixture.episodes[1], disk_backend)
    writer.write_sample(
        _restore_dropped_wrist(priority_profile_fixture.samples[1][0]), disk_backend
    )
    with pytest.raises(ConfigurationError, match="missing required observations"):
        writer.write_sample(priority_profile_fixture.samples[1][1], disk_backend)

    writer = GrootLeRobotV2Writer(_config())
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    sample = priority_profile_fixture.samples[0][0]
    policy_signals = dict(sample.policy_signals)
    policy_signals["action"] = replace(
        policy_signals["action"], representation="relative"
    )
    with pytest.raises(ConfigurationError, match="must store absolute"):
        writer.write_sample(
            replace(sample, policy_signals=policy_signals), disk_backend
        )


@pytest.mark.unit
def test_feature_reorder_is_applied_and_recorded(
    disk_backend, priority_profile_fixture, monkeypatch
):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    episode = priority_profile_fixture.episodes[0]
    writer.begin_episode(episode, disk_backend)
    sample = priority_profile_fixture.samples[0][0]
    observations = dict(sample.observations)
    state = observations["observation.state"]
    observations["observation.state"] = replace(
        state,
        data=np.asarray([state.data[1], state.data[0]], dtype=np.float32),
        feature_names=tuple(reversed(state.feature_names)),
        units=tuple(reversed(state.units)),
    )
    writer.write_sample(replace(sample, observations=observations), disk_backend)
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)

    table = pq.read_table(
        io.BytesIO(disk_backend.read("data/chunk-000/episode_000000.parquet"))
    )
    np.testing.assert_allclose(table["observation.state"].to_pylist()[0], [0.1, 0.2])
    loss = json.loads(disk_backend.read("_ovannotators/loss_report.json"))
    assert any("reordered" in item for item in loss["transformed"])


@pytest.mark.integration
def test_loss_report_records_numeric_casts_and_rgba_to_rgb(
    disk_backend, priority_profile_fixture, monkeypatch
):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    source_episode = priority_profile_fixture.episodes[0]
    episode = replace(
        source_episode,
        streams=tuple(
            replace(stream, shape=(*stream.shape[:2], 4))
            if stream.stream_id in ("camera.front", "camera.wrist")
            else stream
            for stream in source_episode.streams
        ),
    )
    writer.begin_episode(episode, disk_backend)
    sample = priority_profile_fixture.samples[0][0]
    observations = dict(sample.observations)
    for stream_id in ("camera.front", "camera.wrist"):
        rgb = np.asarray(observations[stream_id].data)
        observations[stream_id] = replace(
            observations[stream_id],
            data=np.concatenate(
                [rgb, np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)], axis=2
            ),
        )
    observations["observation.state"] = replace(
        observations["observation.state"],
        data=np.asarray(observations["observation.state"].data, dtype=np.float64),
    )
    policy_signals = dict(sample.policy_signals)
    policy_signals["action"] = replace(
        policy_signals["action"],
        data=np.asarray(policy_signals["action"].data, dtype=np.float64),
    )
    writer.write_sample(
        replace(sample, observations=observations, policy_signals=policy_signals),
        disk_backend,
    )
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)

    loss = json.loads(disk_backend.read("_ovannotators/loss_report.json"))
    assert "state values cast from float64 to float32" in loss["transformed"]
    assert "action values cast from float64 to float32" in loss["transformed"]
    assert (
        "RGBA video converted to RGB by dropping the alpha channel"
        in loss["transformed"]
    )


@pytest.mark.integration
def test_finalize_is_idempotent_and_validator_rejects_v3(
    disk_backend, priority_profile_fixture, monkeypatch
):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer = GrootLeRobotV2Writer(_config())
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, disk_backend)
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)
    before = disk_backend.list()
    writer.finalize(disk_backend)
    assert disk_backend.list() == before

    info = json.loads(disk_backend.read("meta/info.json"))
    info["codebase_version"] = "v3.0"
    disk_backend.write("meta/info.json", json.dumps(info).encode())
    result = validate_groot_lerobot_v2(disk_backend)
    assert result.structural == "failed"
    assert any("LeRobot v3 is incompatible" in message for message in result.messages)

    info["codebase_version"] = "v2.1"
    info["features"]["observation.state"]["names"] = {
        "arm": ["joint.position"],
        "gripper": ["gripper.position"],
    }
    disk_backend.write("meta/info.json", json.dumps(info).encode())
    result = validate_groot_lerobot_v2(disk_backend)
    assert result.structural == "failed"
    assert any("names must be a flat" in message for message in result.messages)


@pytest.mark.integration
def test_completed_validator_verifies_success_and_checksums(
    disk_backend, priority_profile_fixture, monkeypatch
):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    writer.write_sample(priority_profile_fixture.samples[0][0], disk_backend)
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)
    assert validate_groot_lerobot_v2(disk_backend).structural == "passed"

    disk_backend.write("meta/tasks.jsonl", b'{"task_index":0,"task":"tampered"}\n')
    result = validate_groot_lerobot_v2(disk_backend)
    assert result.serialization == "failed"
    assert any("checksum mismatch" in message for message in result.messages)


@pytest.mark.integration
def test_completed_validator_rejects_removed_completion_pair(
    disk_backend, priority_profile_fixture, monkeypatch
):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    writer.write_sample(priority_profile_fixture.samples[0][0], disk_backend)
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)

    disk_backend._abs("_ovannotators/_SUCCESS").unlink()
    disk_backend._abs("_ovannotators/payload_checksums.json").unlink()
    result = validate_groot_lerobot_v2(disk_backend)
    assert result.serialization == "failed"
    assert result.structural == "failed"
    assert any("partial completion artifacts" in item for item in result.messages)


@pytest.mark.unit
def test_finalize_requires_completed_episode(disk_backend, priority_profile_fixture):
    writer = GrootLeRobotV2Writer(_config())
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    with pytest.raises(ConfigurationError, match="end the active episode"):
        writer.finalize(disk_backend)


@pytest.mark.unit
def test_lifecycle_and_schema_errors_are_actionable(
    disk_backend, priority_profile_fixture, monkeypatch
):
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    with pytest.raises(ConfigurationError, match="begin_episode"):
        writer.write_sample(priority_profile_fixture.samples[0][0], disk_backend)
    with pytest.raises(ConfigurationError, match="empty"):
        writer.finalize(disk_backend)

    episode = priority_profile_fixture.episodes[0]
    with pytest.raises(ConfigurationError, match="embodiment"):
        writer.begin_episode(replace(episode, embodiment="other"), disk_backend)
    writer.begin_episode(episode, disk_backend)
    with pytest.raises(ConfigurationError, match="before beginning another"):
        writer.begin_episode(episode, disk_backend)
    empty_writer = GrootLeRobotV2Writer(_config())
    empty_writer.begin_episode(episode, disk_backend)
    with pytest.raises(ConfigurationError, match="at least one sample"):
        empty_writer.end_episode(disk_backend)
    with pytest.raises(ConfigurationError, match="does not match active"):
        writer.write_sample(
            replace(priority_profile_fixture.samples[0][0], episode_id="other"),
            disk_backend,
        )
    sample = priority_profile_fixture.samples[0][0]
    writer.write_sample(sample, disk_backend)
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        writer.write_sample(replace(sample, sample_id="duplicate"), disk_backend)

    bad_observations = dict(priority_profile_fixture.samples[0][1].observations)
    bad_observations["camera.front"] = replace(
        bad_observations["camera.front"], data=np.zeros((3, 4), dtype=np.uint8)
    )
    with pytest.raises(ConfigurationError, match="HxWx3/4"):
        writer.write_sample(
            replace(
                priority_profile_fixture.samples[0][1], observations=bad_observations
            ),
            disk_backend,
        )
    no_state = dict(priority_profile_fixture.samples[0][1].observations)
    no_state.pop("observation.state")
    with pytest.raises(ConfigurationError, match="missing required observations"):
        writer.write_sample(
            replace(priority_profile_fixture.samples[0][1], observations=no_state),
            disk_backend,
        )
    with pytest.raises(ConfigurationError, match="missing required policy signal"):
        writer.write_sample(
            replace(priority_profile_fixture.samples[0][1], policy_signals={}),
            disk_backend,
        )

    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)
    with pytest.raises(ConfigurationError, match="after finalize"):
        writer.begin_episode(episode, disk_backend)


@pytest.mark.unit
def test_structural_validator_reports_missing_metadata(disk_backend):
    result = validate_groot_lerobot_v2(disk_backend)
    assert result.serialization == "failed"
    assert result.structural == "failed"
    assert "missing meta/info.json" in result.messages


@pytest.mark.unit
def test_write_sample_enforces_cfr_timeline_and_contiguous_frame_index(
    disk_backend, priority_profile_fixture
):
    episode = priority_profile_fixture.episodes[0]
    first, second = priority_profile_fixture.samples[0][:2]

    writer = GrootLeRobotV2Writer(_config())
    writer.begin_episode(episode, disk_backend)
    with pytest.raises(ConfigurationError, match="must equal the per-episode"):
        writer.write_sample(replace(first, frame_id=1), disk_backend)
    writer.write_sample(first, disk_backend)
    with pytest.raises(ConfigurationError, match="must equal the per-episode"):
        writer.write_sample(replace(second, frame_id=2), disk_backend)

    # A rejected sample must not advance the timeline; the correct one still fits.
    off_grid = replace(second, timestamp_ns=second.timestamp_ns + 1_000_000)
    with pytest.raises(ConfigurationError, match="constant-frame-rate timeline"):
        writer.write_sample(off_grid, disk_backend)
    writer.write_sample(second, disk_backend)

    # The 100 us tolerance is inclusive; one nanosecond beyond it is rejected.
    third = priority_profile_fixture.samples[0][2]
    on_grid_ns = episode.source_origin_ns + 200_000_000
    with pytest.raises(ConfigurationError, match="constant-frame-rate timeline"):
        writer.write_sample(
            replace(third, timestamp_ns=on_grid_ns + 100_001), disk_backend
        )
    writer.write_sample(replace(third, timestamp_ns=on_grid_ns + 100_000), disk_backend)


@pytest.mark.unit
def test_begin_episode_rejects_duplicate_episode_id(
    disk_backend, priority_profile_fixture, monkeypatch
):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    episode = priority_profile_fixture.episodes[0]
    writer = GrootLeRobotV2Writer(_config())
    writer.begin_episode(episode, disk_backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, disk_backend)
    writer.end_episode(disk_backend)
    with pytest.raises(ConfigurationError, match="duplicate episode_id"):
        writer.begin_episode(episode, disk_backend)


@pytest.mark.integration
def test_validator_rejects_desynchronized_frame_index_and_timestamps(
    disk_backend, priority_profile_fixture, monkeypatch
):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer = GrootLeRobotV2Writer(_config())
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, disk_backend)
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)
    assert validate_groot_lerobot_v2(disk_backend).structural == "passed"

    path = "data/chunk-000/episode_000000.parquet"
    original = disk_backend.read(path)
    table = pq.read_table(io.BytesIO(original))

    def rewrite(**columns):
        updated = table
        for name, values in columns.items():
            index = updated.schema.get_field_index(name)
            updated = updated.set_column(
                index,
                updated.schema.field(name),
                pa.array(values, updated.schema.field(name).type),
            )
        buffer = io.BytesIO()
        pq.write_table(updated, buffer)
        disk_backend.write(path, buffer.getvalue())
        return validate_groot_lerobot_v2(disk_backend)

    result = rewrite(frame_index=[1, 2, 3])
    assert any("frame_index must be contiguous" in m for m in result.messages)

    stamps = table["timestamp_ns"].to_pylist()
    result = rewrite(
        timestamp_ns=[stamps[0], stamps[1] + 5_000_000, stamps[2] + 10_000_000]
    )
    assert any("constant-frame-rate timeline" in m for m in result.messages)

    result = rewrite(timestamp=[0.0, 0.25, 0.5])
    assert any("timestamp seconds do not match" in m for m in result.messages)


def _with_camera_shape(episode, shape):
    return replace(
        episode,
        streams=tuple(
            replace(stream, shape=shape)
            if stream.stream_id == "camera.front"
            else stream
            for stream in episode.streams
        ),
    )


def _finalize_loss(writer, backend, monkeypatch):
    monkeypatch.setattr(
        "ovannotators.profile_writers.groot_lerobot_v2._encode_video",
        lambda frames, fps: b"video",
    )
    writer.end_episode(backend)
    writer.finalize(backend)
    return json.loads(backend.read("_ovannotators/loss_report.json"))["transformed"]


@pytest.mark.unit
def test_rejected_sample_does_not_record_numeric_cast(
    disk_backend, priority_profile_fixture, monkeypatch
):
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    first, second = priority_profile_fixture.samples[0][:2]

    observations = dict(first.observations)
    observations["observation.state"] = replace(
        observations["observation.state"],
        data=np.asarray(observations["observation.state"].data, dtype=np.float64),
    )
    policy_signals = dict(first.policy_signals)
    policy_signals["action"] = replace(
        policy_signals["action"], representation="relative"
    )
    with pytest.raises(ConfigurationError, match="must store absolute"):
        writer.write_sample(
            replace(first, observations=observations, policy_signals=policy_signals),
            disk_backend,
        )
    writer.write_sample(first, disk_backend)
    transformed = _finalize_loss(writer, disk_backend, monkeypatch)
    assert not any("cast from" in item for item in transformed)


@pytest.mark.unit
def test_accepted_sample_records_numeric_cast(
    disk_backend, priority_profile_fixture, monkeypatch
):
    first = priority_profile_fixture.samples[0][0]
    observations = dict(first.observations)
    observations["observation.state"] = replace(
        observations["observation.state"],
        data=np.asarray(observations["observation.state"].data, dtype=np.float64),
    )
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    writer.begin_episode(priority_profile_fixture.episodes[0], disk_backend)
    writer.write_sample(replace(first, observations=observations), disk_backend)
    transformed = _finalize_loss(writer, disk_backend, monkeypatch)
    assert "state values cast from float64 to float32" in transformed


@pytest.mark.unit
def test_rejected_sample_does_not_record_alpha_drop(
    disk_backend, priority_profile_fixture, monkeypatch
):
    episode = priority_profile_fixture.episodes[0]
    first = priority_profile_fixture.samples[0][0]
    rgba = replace(
        first.observations["camera.front"],
        data=np.concatenate(
            [
                np.asarray(first.observations["camera.front"].data),
                np.full((4, 4, 1), 255, dtype=np.uint8),
            ],
            axis=2,
        ),
    )
    bad_wrist = replace(
        first.observations["camera.wrist"], data=np.zeros((2, 2, 3), dtype=np.uint8)
    )
    rejected = replace(
        first,
        observations={
            **first.observations,
            "camera.front": rgba,
            "camera.wrist": bad_wrist,
        },
    )
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    writer.begin_episode(_with_camera_shape(episode, (4, 4, 4)), disk_backend)
    with pytest.raises(ConfigurationError, match="shape changed"):
        writer.write_sample(rejected, disk_backend)
    writer.write_sample(first, disk_backend)
    transformed = _finalize_loss(writer, disk_backend, monkeypatch)
    assert not any("RGBA" in item for item in transformed)


@pytest.mark.unit
def test_accepted_sample_records_alpha_drop(
    disk_backend, priority_profile_fixture, monkeypatch
):
    episode = priority_profile_fixture.episodes[0]
    first = priority_profile_fixture.samples[0][0]
    rgba = replace(
        first.observations["camera.front"],
        data=np.concatenate(
            [
                np.asarray(first.observations["camera.front"].data),
                np.full((4, 4, 1), 255, dtype=np.uint8),
            ],
            axis=2,
        ),
    )
    writer = GrootLeRobotV2Writer(_config(horizon=1))
    writer.begin_episode(_with_camera_shape(episode, (4, 4, 4)), disk_backend)
    writer.write_sample(
        replace(first, observations={**first.observations, "camera.front": rgba}),
        disk_backend,
    )
    transformed = _finalize_loss(writer, disk_backend, monkeypatch)
    assert "RGBA video converted to RGB by dropping the alpha channel" in transformed


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(5, 4, 3), (4, 5, 3), (5, 5, 3)])
def test_begin_episode_rejects_odd_video_dimensions(
    disk_backend, priority_profile_fixture, shape
):
    episode = priority_profile_fixture.episodes[0]
    writer = GrootLeRobotV2Writer(_config())
    with pytest.raises(ConfigurationError, match="odd dimensions") as error:
        writer.begin_episode(_with_camera_shape(episode, shape), disk_backend)
    assert "camera.front" in str(error.value)
    assert f"{shape[0]}x{shape[1]}" in str(error.value)
    assert writer._active is None
    assert writer._video_shapes == {}
    assert writer._dataset_id is None
    assert disk_backend.list() == []

    writer.begin_episode(episode, disk_backend)
    assert writer._video_shapes["front"] == (4, 4, 3)


@pytest.mark.unit
def test_relative_action_policy_rejects_unknown_values():
    config = _config()
    absolute_actions = tuple(
        replace(item, representation="absolute", state_key=None)
        for item in config.action_modalities
    )
    all_absolute = replace(
        config, action_modalities=absolute_actions, relative_action_policy="disabled"
    )
    assert all_absolute.relative_action_policy == "disabled"
    with pytest.raises(
        ConfigurationError, match="'disabled' or 'groot_processor'.*'invalid'"
    ):
        replace(all_absolute, relative_action_policy="invalid")
    with pytest.raises(ConfigurationError, match="exactly when"):
        replace(all_absolute, relative_action_policy="groot_processor")
    with pytest.raises(ConfigurationError, match="exactly when"):
        replace(config, relative_action_policy="disabled")
    assert config.relative_action_policy == "groot_processor"
