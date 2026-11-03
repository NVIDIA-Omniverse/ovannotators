# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the exact Cosmos 3.5 multiview candidate profile."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ovannotators import (
    CosmosMaterializationConfig,
    CosmosMultiviewWriter,
    CosmosMultiviewWriterConfig,
    CosmosQAConfig,
    NamedTensor,
    ProfileWriterRegistry,
    config_from_dict,
    config_to_dict,
)
from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.exceptions import ConfigurationError

pa = pytest.importorskip("pyarrow", reason="requires the cosmos-multiview extra")
pq = pytest.importorskip(
    "pyarrow.parquet", reason="requires the cosmos-multiview extra"
)
lance = pytest.importorskip("lance", reason="requires the cosmos-multiview extra")


def _config(**changes) -> CosmosMultiviewWriterConfig:
    config = CosmosMultiviewWriterConfig(
        output_base_uri="s3://fixture-bucket/cosmos/release-v1",
        source_version="source-v1",
        domain="robotics",
        owner="fixture-team",
        license_uri="s3://fixture-bucket/licenses/fixture-v1.json",
        default_reference_frame="robot_base",
        supported_uri_schemes=("s3",),
        materialization=CosmosMaterializationConfig(
            version="materializer-v1",
            duration_ns=100_000_003,
            stride_ns=100_000_003,
            reference_stream_id="camera.front",
            target_rate_hz=10.0,
            selected_stream_ids=(
                "camera.front",
                "camera.wrist",
                "camera.front.depth",
                "camera.front.segmentation",
                "observation.state",
                "action",
            ),
            alignment_by_stream_type={
                "rgb": "nearest_timestamp",
                "depth": "nearest_timestamp",
                "segmentation": "nearest_timestamp",
                "state": "nearest_timestamp",
                "action": "linear_interpolation",
            },
            output_encoding_by_stream_type={
                "rgb": "npy",
                "depth": "npy",
                "segmentation": "npy",
                "state": "npy",
                "action": "npy",
            },
            tasks=("action", "transfer"),
            created_by="fixture-team",
        ),
        qa=CosmosQAConfig(
            profile_id="synthetic-fixture-qa",
            version="1.0",
            max_alignment_error_ns=100_000_003,
            geometry_reprojection_tolerance_px=0.25,
            minimum_stream_coverage=1.0,
            missing_data_policy="drop_sample",
            warning_policy="owner_approval_required",
        ),
    )
    return dataclasses.replace(config, **changes)


def _write_fixture(
    root: Path,
    priority_profile_fixture,
    config: CosmosMultiviewWriterConfig | None = None,
):
    backend = DiskBackend(DiskBackendConfig(root, overwrite=True))
    writer = CosmosMultiviewWriter(config or _config())
    for episode, samples in zip(
        priority_profile_fixture.episodes,
        priority_profile_fixture.samples,
        strict=True,
    ):
        writer.begin_episode(episode, backend)
        for sample in samples:
            writer.write_sample(sample, backend)
        writer.end_episode(backend)
    writer.register_episodes(backend)
    assert not backend.exists("_ovannotators/_SUCCESS")
    writer.materialize_samples(backend)
    writer.finalize(backend)
    writer.finalize(backend)
    return backend, writer


@pytest.mark.unit
def test_config_is_explicit_round_trippable_and_rejects_unsupported_contracts():
    config = _config()
    assert (
        config_from_dict(CosmosMultiviewWriterConfig, config_to_dict(config)) == config
    )
    with pytest.raises(ConfigurationError, match="unsupported Cosmos schema"):
        _config(schema_version="2.0")
    with pytest.raises(ConfigurationError, match="supported_uri_schemes"):
        _config(output_base_uri="https://example.test/release")
    with pytest.raises(ConfigurationError, match="Cosmos vocabulary"):
        dataclasses.replace(config.materialization, tasks=("pick_cube",))
    with pytest.raises(ConfigurationError, match="unique"):
        dataclasses.replace(config.materialization, tasks=("action", "action"))
    with pytest.raises(ConfigurationError, match="selected stream"):
        dataclasses.replace(
            config.materialization, reference_stream_id="camera.unselected"
        )
    with pytest.raises(ConfigurationError, match="object_key_policy"):
        dataclasses.replace(config.materialization, object_key_policy="sequential")
    with pytest.raises(ConfigurationError, match="missing_data_policy"):
        dataclasses.replace(config.qa, missing_data_policy="ignore")
    with pytest.raises(ConfigurationError, match="warning_policy"):
        dataclasses.replace(config.qa, warning_policy="allow")
    with pytest.raises(ConfigurationError, match="safe relative paths"):
        _config(dataset_info_path="")


@pytest.mark.unit
def test_selected_streams_must_use_the_episode_clock(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    episode = priority_profile_fixture.episodes[0]
    mismatched = dataclasses.replace(episode.streams[0], source_clock="other-clock")
    writer = CosmosMultiviewWriter(_config())

    with pytest.raises(ValueError, match="episode source clock"):
        writer.begin_episode(
            dataclasses.replace(episode, streams=(mismatched, *episode.streams[1:])),
            backend,
        )


@pytest.mark.integration
def test_unselected_streams_do_not_require_materialization_policies(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    episode = priority_profile_fixture.episodes[0]
    unselected = dataclasses.replace(
        episode.streams[0],
        stream_id="lidar.unselected",
        stream_type="lidar",
        geometry={},
        derived_from_stream_id=None,
        required=False,
    )
    writer = CosmosMultiviewWriter(_config())
    writer.begin_episode(
        dataclasses.replace(episode, streams=(*episode.streams, unselected)), backend
    )
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, backend)
    writer.end_episode(backend)

    rows = writer.register_episodes(backend)

    assert "lidar.unselected" in {stream["stream_id"] for stream in rows[0]["streams"]}


@pytest.mark.integration
def test_required_unselected_stream_must_have_episode_coverage(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    episode = priority_profile_fixture.episodes[0]
    missing_required = dataclasses.replace(
        episode.streams[0],
        stream_id="lidar.required",
        stream_type="lidar",
        geometry={},
        derived_from_stream_id=None,
    )
    writer = CosmosMultiviewWriter(_config())
    writer.begin_episode(
        dataclasses.replace(episode, streams=(*episode.streams, missing_required)),
        backend,
    )
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, backend)

    with pytest.raises(ValueError, match="required streams have no samples"):
        writer.end_episode(backend)


@pytest.mark.integration
def test_finalize_rejects_candidate_with_no_materialized_windows(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    base = _config()
    materialization = dataclasses.replace(
        base.materialization,
        duration_ns=1_000_000_000,
        stride_ns=1_000_000_000,
    )
    writer = CosmosMultiviewWriter(
        dataclasses.replace(base, materialization=materialization)
    )
    episode = priority_profile_fixture.episodes[0]
    writer.begin_episode(episode, backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, backend)
    writer.end_episode(backend)

    with pytest.raises(ValueError, match="no sample rows"):
        writer.finalize(backend)

    assert not backend.exists("samples.lance")
    assert not backend.exists("_ovannotators/_SUCCESS")


@pytest.mark.integration
def test_two_stage_output_has_exact_typed_tables_and_unverified_qa(
    tmp_path: Path, priority_profile_fixture
):
    backend, writer = _write_fixture(tmp_path, priority_profile_fixture)

    episodes = lance.dataset(tmp_path / "episodes.lance").to_table()
    samples = lance.dataset(tmp_path / "samples.lance").to_table()
    assert episodes.column_names == [
        "episode_id",
        "dataset_id",
        "source_episode_id",
        "stream_types",
        "streams",
        "dataset_info_uri",
    ]
    assert str(episodes.schema.field("streams").type).startswith("list<item: struct<")
    assert samples.column_names == [
        "sample_id",
        "episode_id",
        "task",
        "source_start_offset_s",
        "source_end_offset_s",
        "assets_json",
        "caption",
        "sample_config_uri",
        "created_by",
    ]
    assert episodes.num_rows == 3
    assert samples.num_rows == 12
    assert set(samples.column("task").to_pylist()) == {"action", "transfer"}
    assert set(samples.column("episode_id").to_pylist()) <= set(
        episodes.column("episode_id").to_pylist()
    )
    assert backend.exists("_ovannotators/_SUCCESS")
    assert writer.validate(backend).structural == "passed"
    assert CosmosMultiviewWriter(_config()).validate(backend).structural == "passed"
    episode_rows = episodes.to_pylist()
    first_stream = episode_rows[0]["streams"][0]
    source_path = first_stream["data_uri"].removeprefix(
        "s3://fixture-bucket/cosmos/release-v1/"
    )
    assert source_path.endswith("data.npy")
    assert np.load(tmp_path / source_path).shape[0] == 3
    for gate in ("episode", "sample"):
        record = json.loads(backend.read(f"validation/{gate}.json"))
        assert record["status"] == "unverified"
        assert record["started_at"] is None
        assert record["warning_approval"] is None


@pytest.mark.integration
def test_validate_uses_configured_metadata_paths(
    tmp_path: Path, priority_profile_fixture
):
    config = _config(
        dataset_info_path="custom/dataset.json",
        sample_config_path="custom/sample.json",
    )
    backend, _ = _write_fixture(tmp_path, priority_profile_fixture, config)

    assert backend.exists(config.dataset_info_path)
    assert backend.exists(config.sample_config_path)
    assert CosmosMultiviewWriter(config).validate(backend).structural == "passed"


@pytest.mark.integration
def test_validate_reads_each_sample_asset_once(
    tmp_path: Path, priority_profile_fixture, monkeypatch
):
    completed_backend, _ = _write_fixture(
        tmp_path / "completed", priority_profile_fixture
    )
    validation_root = tmp_path / "without-profile-artifacts"
    validation_backend = DiskBackend(DiskBackendConfig(validation_root, overwrite=True))
    for path in completed_backend.list():
        if not path.startswith("_ovannotators/"):
            validation_backend.write(path, completed_backend.read(path))

    expected_reads: Counter[str] = Counter()
    rows = lance.dataset(validation_root / "samples.lance").to_table().to_pylist()
    prefix = "s3://fixture-bucket/cosmos/release-v1/"
    for row in rows:
        for asset in json.loads(row["assets_json"])["assets"]:
            expected_reads[asset["uri"].removeprefix(prefix)] += 1
            expected_reads[asset["timestamps_uri"].removeprefix(prefix)] += 1

    actual_reads: Counter[str] = Counter()
    original_read = validation_backend.read

    def count_asset_read(path: str) -> bytes:
        if path in expected_reads:
            actual_reads[path] += 1
        return original_read(path)

    monkeypatch.setattr(validation_backend, "read", count_asset_read)

    result = CosmosMultiviewWriter(_config()).validate(validation_backend)

    assert result.structural == "passed"
    assert actual_reads == expected_reads


@pytest.mark.integration
def test_timestamp_windows_assets_lineage_and_task_reuse_survive_round_trip(
    tmp_path: Path, priority_profile_fixture
):
    backend, _ = _write_fixture(tmp_path, priority_profile_fixture)
    rows = lance.dataset(tmp_path / "samples.lance").to_table().to_pylist()
    paired = {}
    for row in rows:
        assert row["source_end_offset_s"] > row["source_start_offset_s"]
        manifest = json.loads(row["assets_json"])
        assert row["assets_json"] == json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        )
        assets = {asset["stream_id"]: asset for asset in manifest["assets"]}
        assert assets["camera.front.depth"]["derived_from_stream_id"] == "camera.front"
        assert (
            assets["camera.front.segmentation"]["derived_from_stream_id"]
            == "camera.front"
        )
        state = assets["observation.state"]
        relative = state["uri"].removeprefix("s3://fixture-bucket/cosmos/release-v1/")
        array = np.load(tmp_path / relative)
        assert array.shape == (2, 2)
        pair_key = (row["episode_id"], row["source_start_offset_s"])
        paired.setdefault(pair_key, []).append(
            tuple(asset["uri"] for asset in manifest["assets"])
        )
    assert all(
        len(asset_sets) == 2 and asset_sets[0] == asset_sets[1]
        for asset_sets in paired.values()
    )
    dropped = json.loads(backend.read("validation/sample.json"))["exclusions"]
    incomplete_windows = [
        item for item in dropped if "exceeds source coverage" in item["reason"]
    ]
    dropped_windows = [
        item for item in dropped if "incomplete aligned coverage" in item["reason"]
    ]
    assert len(incomplete_windows) == 3
    assert dropped_windows == []


@pytest.mark.integration
def test_stride_drives_half_open_windows_and_target_rate_sequences(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    base = _config()
    materialization = dataclasses.replace(base.materialization, stride_ns=50_000_000)
    qa = dataclasses.replace(base.qa, max_alignment_error_ns=100_000_003)
    writer = CosmosMultiviewWriter(
        dataclasses.replace(base, materialization=materialization, qa=qa)
    )
    episode = priority_profile_fixture.episodes[0]
    writer.begin_episode(episode, backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, backend)
    writer.end_episode(backend)
    writer.finalize(backend)
    rows = lance.dataset(tmp_path / "samples.lance").to_table().to_pylist()
    episode_zero = lance.dataset(tmp_path / "episodes.lance").to_table().to_pylist()[0]
    windows = sorted(
        {
            (row["source_start_offset_s"], row["source_end_offset_s"])
            for row in rows
            if row["episode_id"] == episode_zero["episode_id"]
        }
    )
    assert windows == [
        (0.0, 0.100000003),
        (0.05, 0.150000003),
        (0.1, 0.200000003),
    ]
    exclusions = json.loads(backend.read("validation/sample.json"))["exclusions"]
    assert exclusions == [
        {
            "episode_id": episode_zero["episode_id"],
            "reason": "incomplete half-open window exceeds source coverage",
            "source_start_offset_ns": "150000000",
        }
    ]
    first = next(
        row
        for row in rows
        if row["episode_id"] == episode_zero["episode_id"]
        and row["task"] == "action"
        and row["source_start_offset_s"] == 0.0
    )
    front = next(
        asset
        for asset in json.loads(first["assets_json"])["assets"]
        if asset["stream_id"] == "camera.front"
    )
    assert front["target_timestamps_ns"] == [5_000_000_000, 5_100_000_000]
    path = front["uri"].removeprefix("s3://fixture-bucket/cosmos/release-v1/")
    frames = np.load(tmp_path / path)
    assert frames.shape == (2, 3, 4, 3)
    assert np.all(frames[0] == 1)
    assert np.all(frames[1] == 2)


@pytest.mark.integration
def test_mixed_rate_nearest_and_action_interpolation_use_timestamps(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    qa = dataclasses.replace(_config().qa, max_alignment_error_ns=100_000_006)
    writer = CosmosMultiviewWriter(_config(qa=qa))
    episode = priority_profile_fixture.episodes[0]
    samples = list(priority_profile_fixture.samples[0])
    middle = samples[1]
    middle_observations = dict(middle.observations)
    middle_observations.pop("camera.wrist")
    samples[1] = dataclasses.replace(
        middle,
        observations=middle_observations,
        policy_signals={},
        missing_data={
            "camera.wrist": "not_emitted_at_mixed_rate",
            "action": "interpolate_from_authoritative_neighbors",
        },
    )
    writer.begin_episode(episode, backend)
    for sample in samples:
        writer.write_sample(sample, backend)
    writer.end_episode(backend)
    writer.finalize(backend)

    rows = lance.dataset(tmp_path / "samples.lance").to_table().to_pylist()
    middle_row = next(
        row
        for row in rows
        if row["task"] == "action" and row["source_start_offset_s"] == 0.0
    )
    assets = {
        asset["stream_id"]: asset
        for asset in json.loads(middle_row["assets_json"])["assets"]
    }
    prefix = "s3://fixture-bucket/cosmos/release-v1/"
    wrist = np.load(tmp_path / assets["camera.wrist"]["uri"].removeprefix(prefix))
    action = np.load(tmp_path / assets["action"]["uri"].removeprefix(prefix))
    assert np.all(wrist[1] == 51)
    np.testing.assert_allclose(action[1], [11.1, 11.2], rtol=1e-6)
    assert assets["camera.wrist"]["alignment_policy"] == "nearest_timestamp"
    assert assets["camera.wrist"]["source_timestamps_ns"][1] == [5_000_000_000]
    assert assets["action"]["alignment_policy"] == "linear_interpolation"
    assert assets["action"]["source_timestamps_ns"][1] == [
        5_000_000_000,
        5_200_000_006,
    ]
    timing_path = assets["action"]["timestamps_uri"].removeprefix(prefix)
    timing = pq.read_table(tmp_path / timing_path).to_pylist()
    assert timing[1]["target_timestamp_ns"] == 5_100_000_000
    assert timing[1]["source_timestamps_ns"] == [
        5_000_000_000,
        5_200_000_006,
    ]
    assert timing[1]["native_timestamp_ns"] == 5_100_000_000
    assert timing[1]["episode_time_ns"] == 100_000_000


@pytest.mark.integration
def test_linear_interpolation_rejects_targets_without_bracketing_sources(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    qa = dataclasses.replace(_config().qa, max_alignment_error_ns=100_000_003)
    writer = CosmosMultiviewWriter(_config(qa=qa))
    episode = priority_profile_fixture.episodes[0]
    samples = list(priority_profile_fixture.samples[0])
    first = samples[0]
    first_policy = dict(first.policy_signals)
    first_policy.pop("action")
    samples[0] = dataclasses.replace(
        first,
        policy_signals=first_policy,
        missing_data={"action": "not_emitted_before_target"},
    )
    writer.begin_episode(episode, backend)
    for sample in samples:
        writer.write_sample(sample, backend)
    writer.end_episode(backend)
    writer.finalize(backend)

    rows = lance.dataset(tmp_path / "samples.lance").to_table().to_pylist()
    assert all(row["source_start_offset_s"] != 0.0 for row in rows)
    exclusions = json.loads(backend.read("validation/sample.json"))["exclusions"]
    assert any(
        "action@5000000000" in item["reason"]
        for item in exclusions
        if "incomplete aligned coverage" in item["reason"]
    )


@pytest.mark.unit
def test_linear_interpolation_preserves_unsigned_values():
    writer = CosmosMultiviewWriter(_config())
    left = NamedTensor(np.asarray([200], dtype=np.uint8))
    right = NamedTensor(np.asarray([100], dtype=np.uint8))

    aligned = writer._align_signal([(0, left), (10, right)], 5, "linear_interpolation")

    assert aligned is not None
    np.testing.assert_array_equal(
        aligned.tensor.data, np.asarray([150], dtype=np.uint8)
    )
    assert aligned.tensor.data.dtype == np.dtype("uint8")


@pytest.mark.integration
def test_ids_and_logical_manifests_are_deterministic(
    tmp_path: Path, priority_profile_fixture
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_fixture(first, priority_profile_fixture)
    _write_fixture(second, priority_profile_fixture)
    for table_name in ("episodes.lance", "samples.lance"):
        left = lance.dataset(first / table_name).to_table().to_pylist()
        right = lance.dataset(second / table_name).to_table().to_pylist()
        assert left == right


@pytest.mark.integration
def test_checksum_verified_rerun_reuses_completed_immutable_tree(
    tmp_path: Path, priority_profile_fixture
):
    first_backend, _ = _write_fixture(tmp_path, priority_profile_fixture)
    before = {path: first_backend.read(path) for path in first_backend.list()}
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=False))
    writer = CosmosMultiviewWriter(_config())
    for episode, samples in zip(
        priority_profile_fixture.episodes,
        priority_profile_fixture.samples,
        strict=True,
    ):
        writer.begin_episode(episode, backend)
        for sample in samples:
            writer.write_sample(sample, backend)
        writer.end_episode(backend)
    writer.finalize(backend)
    after = {path: backend.read(path) for path in backend.list()}
    assert after == before


@pytest.mark.integration
def test_retained_candidate_golden_identity_manifest(
    tmp_path: Path, priority_profile_fixture
):
    _write_fixture(tmp_path, priority_profile_fixture)
    episodes = lance.dataset(tmp_path / "episodes.lance").to_table().to_pylist()
    samples = lance.dataset(tmp_path / "samples.lance").to_table().to_pylist()
    actual = {
        "episode_ids": [row["episode_id"] for row in episodes],
        "sample_ids": [row["sample_id"] for row in samples],
        "source_episode_ids": [row["source_episode_id"] for row in episodes],
    }
    candidate = json.loads(
        (
            Path(__file__).parent / "golden/cosmos_multiview/candidate/manifest.json"
        ).read_text()
    )
    assert actual == candidate


@pytest.mark.integration
def test_changed_asset_fails_structural_checksum_validation(
    tmp_path: Path, priority_profile_fixture
):
    backend, _ = _write_fixture(tmp_path, priority_profile_fixture)
    row = lance.dataset(tmp_path / "samples.lance").to_table().to_pylist()[0]
    uri = json.loads(row["assets_json"])["assets"][0]["uri"]
    relative = uri.removeprefix("s3://fixture-bucket/cosmos/release-v1/")
    backend.write(relative, b"changed")
    result = CosmosMultiviewWriter(_config()).validate(backend)
    assert result.structural == "failed"
    assert any("checksum changed" in message for message in result.messages)

    missing_success = DiskBackend(
        DiskBackendConfig(tmp_path / "missing-success", overwrite=True)
    )
    for path in backend.list():
        if path != "_ovannotators/_SUCCESS":
            missing_success.write(path, backend.read(path))
    missing_result = CosmosMultiviewWriter(_config()).validate(missing_success)
    assert missing_result.structural == "failed"
    assert any(
        "without the final _SUCCESS" in message for message in missing_result.messages
    )


@pytest.mark.integration
def test_persisted_validator_enforces_arrow_metadata_and_metadata_uris(
    tmp_path: Path, priority_profile_fixture
):
    schema_root = tmp_path / "schema"
    _write_fixture(schema_root, priority_profile_fixture)
    episode_path = schema_root / "episodes.lance"
    episode_table = lance.dataset(episode_path).to_table().replace_schema_metadata({})
    lance.write_dataset(
        episode_table,
        episode_path,
        mode="overwrite",
        data_storage_version="2.2",
    )
    schema_backend = DiskBackend(DiskBackendConfig(schema_root, overwrite=True))
    schema_result = CosmosMultiviewWriter(_config()).validate(schema_backend)
    assert schema_result.structural == "failed"
    assert any(
        "incompatible Arrow schema or metadata" in message
        for message in schema_result.messages
    )

    uri_root = tmp_path / "uri"
    _write_fixture(uri_root, priority_profile_fixture)
    sample_path = uri_root / "samples.lance"
    sample_table = lance.dataset(sample_path).to_table()
    bad_uris = pa.array(
        ["s3://fixture-bucket/cosmos/release-v1/metadata/missing.json"]
        * sample_table.num_rows,
        type=pa.string(),
    )
    sample_table = sample_table.set_column(
        sample_table.schema.get_field_index("sample_config_uri"),
        "sample_config_uri",
        bad_uris,
    )
    lance.write_dataset(
        sample_table,
        sample_path,
        mode="overwrite",
        data_storage_version="2.2",
    )
    episode_path = uri_root / "episodes.lance"
    episode_table = lance.dataset(episode_path).to_table()
    bad_dataset_uris = pa.array(
        ["s3://fixture-bucket/cosmos/release-v1/metadata/missing-dataset.json"]
        * episode_table.num_rows,
        type=pa.string(),
    )
    episode_table = episode_table.set_column(
        episode_table.schema.get_field_index("dataset_info_uri"),
        "dataset_info_uri",
        bad_dataset_uris,
    )
    lance.write_dataset(
        episode_table,
        episode_path,
        mode="overwrite",
        data_storage_version="2.2",
    )
    uri_backend = DiskBackend(DiskBackendConfig(uri_root, overwrite=True))
    uri_result = CosmosMultiviewWriter(_config()).validate(uri_backend)
    assert uri_result.structural == "failed"
    assert any(
        "unresolvable sample_config_uri" in message for message in uri_result.messages
    )
    assert any(
        "unresolvable dataset_info_uri" in message for message in uri_result.messages
    )


@pytest.mark.integration
def test_persisted_asset_uri_cannot_escape_the_backend_root(
    tmp_path: Path, priority_profile_fixture
):
    root = tmp_path / "candidate"
    _write_fixture(root, priority_profile_fixture)
    sample_path = root / "samples.lance"
    sample_table = lance.dataset(sample_path).to_table()
    rows = sample_table.to_pylist()
    manifest = json.loads(rows[0]["assets_json"])
    manifest["assets"][0]["uri"] = (
        "s3://fixture-bucket/cosmos/release-v1/../outside.npy"
    )
    rows[0]["assets_json"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    lance.write_dataset(
        pa.Table.from_pylist(rows, schema=sample_table.schema),
        sample_path,
        mode="overwrite",
        data_storage_version="2.2",
    )
    backend = DiskBackend(DiskBackendConfig(root, overwrite=True))

    result = CosmosMultiviewWriter(_config()).validate(backend)

    assert result.structural == "failed"
    assert any("unsafe relative path" in message for message in result.messages)


@pytest.mark.unit
def test_registry_and_direct_exports_are_distinct_from_cosmos_transfer():
    assert ProfileWriterRegistry.get("cosmos_multiview") is CosmosMultiviewWriter
    assert "cosmos_multiview" in ProfileWriterRegistry.list_names()
    from ovannotators.writers import CosmosTransferWriter

    assert CosmosTransferWriter.name == "cosmos_transfer"
    assert CosmosTransferWriter is not CosmosMultiviewWriter


@pytest.mark.unit
def test_partial_stream_coverage_policy_is_rejected():
    config = _config()
    with pytest.raises(ConfigurationError, match="must be 1.0"):
        dataclasses.replace(
            config,
            qa=dataclasses.replace(config.qa, minimum_stream_coverage=0.9),
        )


@pytest.mark.unit
def test_geometry_and_missing_selected_streams_fail_before_table_write(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    writer = CosmosMultiviewWriter(_config())
    episode = priority_profile_fixture.episodes[0]
    bad_stream = dataclasses.replace(
        episode.streams[0], geometry={"camera_model": "bad"}
    )
    writer.begin_episode(
        dataclasses.replace(episode, streams=(bad_stream, *episode.streams[1:])),
        backend,
    )
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(sample, backend)
    writer.end_episode(backend)
    with pytest.raises(ValueError, match="complete geometry"):
        writer.register_episodes(backend)

    schema_backend = DiskBackend(
        DiskBackendConfig(tmp_path / "schema-change", overwrite=True)
    )
    schema_writer = CosmosMultiviewWriter(_config())
    schema_writer.begin_episode(episode, schema_backend)
    sample = priority_profile_fixture.samples[0][0]
    observations = dict(sample.observations)
    observations["camera.front"] = dataclasses.replace(
        observations["camera.front"], data=np.zeros((2, 2, 3), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="changed shape"):
        schema_writer.write_sample(
            dataclasses.replace(sample, observations=observations), schema_backend
        )
