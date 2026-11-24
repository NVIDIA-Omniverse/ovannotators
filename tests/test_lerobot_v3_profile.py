# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native LeRobot v3.0 profile writer."""

from __future__ import annotations

import importlib.util
import io
import errno
import json
import re
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ovannotators import (
    EpisodeEnvelope,
    LeRobotLanceFeatureSpec,
    LeRobotLanceWriter,
    LeRobotLanceWriterConfig,
    LeRobotV3FeatureSpec,
    LeRobotV3Writer,
    LeRobotV3WriterConfig,
    NamedTensor,
    ProfileWriterRegistry,
    SampleEnvelope,
    StreamDescriptor,
    validate_lerobot_v3_canonical,
)
from ovannotators._internal.mp4_mux import (
    mp4_duration_s,
    mux_av1_obu,
    mux_h264_annexb,
    parse_mp4_video_index,
)
from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.profile_writers.lerobot_v3 import LEROBOT_VERSION

_SC = b"\x00\x00\x00\x01"
_SPS = b"\x67\x42\x00\x1e\xda\x01\x40"
_PPS = b"\x68\xce\x3c\x80"
_IDR = b"\x65\x88\x84\x00\x21\x00"
_NON_IDR = b"\x41\x9a\x00\x10"


def _synthetic_mp4_encoder(
    frames: list[np.ndarray],
    fps: float,
    *,
    codec: str = "h264",
    lossless: bool = False,
    preset: str | None = None,
    gop: int | None = None,
) -> bytes:
    """Build a structurally valid MP4 clip on CPU without NVENC or PyAV."""
    del lossless, preset, gop
    h, w = frames[0].shape[:2]
    if codec == "av1":
        seq = b"\x0a\x08" + b"\x00" * 8
        key = b"\x32\x04\x10\x11\x22\x33"
        inter = b"\x32\x04\x30\x44\x55\x66"
        td = b"\x12\x00"
        stream = td + seq + key + b"".join(td + inter for _ in frames[1:])
        return mux_av1_obu(stream, fps=fps, width=w, height=h)
    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR
    for _ in frames[1:]:
        stream += _SC + _NON_IDR
    return mux_h264_annexb(stream, fps=fps, width=w, height=h)


def _config(**changes: Any) -> LeRobotV3WriterConfig:
    config = LeRobotV3WriterConfig(
        fps=10,
        repo_id="team/dataset",
        camera_stream_ids=("camera.front", "camera.wrist"),
        state=LeRobotV3FeatureSpec(
            "observation.state",
            ("joint.position", "gripper.position"),
            ("rad", "m"),
            "absolute",
            (2,),
            coordinate_frame="robot_base",
        ),
        action=LeRobotV3FeatureSpec(
            "action",
            ("joint.target", "gripper.target"),
            ("rad", "m"),
            "absolute",
            (2,),
            coordinate_frame="robot_base",
        ),
    )
    return replace(config, **changes)


def _mixed_rate(sample):
    if sample.frame_id % 2 == 0:
        return sample
    observations = dict(sample.observations)
    observations.pop("camera.wrist", None)
    missing = dict(sample.missing_data)
    missing["camera.wrist"] = "not_sampled_at_source_rate"
    return replace(sample, observations=observations, missing_data=missing)


def _write_fixture(tmp_path: Path, priority_profile_fixture, **cfg_overrides: Any):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage", **cfg_overrides),
        video_encoder=_synthetic_mp4_encoder,
    )
    for episode, samples in zip(
        priority_profile_fixture.episodes,
        priority_profile_fixture.samples,
        strict=True,
    ):
        writer.begin_episode(episode, backend)
        for sample in samples:
            writer.write_sample(_mixed_rate(sample), backend)
        writer.end_episode(backend)
    writer.finalize(backend)
    return writer, backend


@pytest.mark.unit
def test_config_requires_explicit_semantics_and_valid_video_options():
    with pytest.raises(ValueError, match="camera_stream_ids"):
        replace(_config(), camera_stream_ids=())
    with pytest.raises(ValueError, match="feature_names"):
        LeRobotV3FeatureSpec("state", ("one",), ("rad",), "absolute", (2,))
    for shape in ((), (0,), (-1,), (1, 1), (2, 3)):
        with pytest.raises(ValueError, match="one positive dimension"):
            LeRobotV3FeatureSpec("state", ("a",) * 6, ("rad",), "absolute", shape)
    with pytest.raises(ValueError, match="repo_id"):
        replace(_config(), repo_id="not-a-repo-id")
    assert replace(_config(), repo_id=None).repo_id is None
    with pytest.raises(ValueError, match="chunks_size"):
        replace(_config(), chunks_size=0)
    with pytest.raises(ValueError, match="data_files_size_in_mb"):
        replace(_config(), data_files_size_in_mb=0)
    with pytest.raises(ValueError, match="video_files_size_in_mb"):
        replace(_config(), video_files_size_in_mb=0)
    with pytest.raises(ValueError, match="timestamp_tolerance_ns"):
        replace(_config(), timestamp_tolerance_ns=100_001)
    with pytest.raises(ValueError, match="must differ"):
        replace(
            _config(), action=replace(_config().action, stream_id="observation.state")
        )
    with pytest.raises(ValueError, match="video_codec"):
        replace(_config(), video_codec="vp9")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="video_gop"):
        replace(_config(), video_gop=0)
    # AV1 and explicit GOP are valid
    cfg = _config(video_codec="av1", video_preset="P4", video_gop=15)
    assert cfg.video_codec == "av1"
    assert cfg.video_gop == 15


@pytest.mark.parametrize("stream_id", ["index", "task", "source.frame_id"])
def test_reserved_state_stream_names_are_rejected(stream_id):
    with pytest.raises(ValueError, match="output keys must be unique"):
        _config(state=replace(_config().state, stream_id=stream_id))


def test_tabular_keys_must_not_look_like_image_features():
    with pytest.raises(ValueError, match="must not contain 'image'"):
        _config(state=replace(_config().state, stream_id="observation.image_state"))
    with pytest.raises(
        ValueError, match=r"camera stream IDs .*\['camera.image_left'\].*rename"
    ):
        _config(camera_stream_ids=("camera.image_left",))


@pytest.mark.parametrize(
    ("camera_ids", "error"),
    [
        ((r"camera.front\..\..\..",), "portable path component"),
        (("camera.front", "camera.FRONT"), "case-insensitive"),
    ],
)
def test_camera_video_paths_are_portable(camera_ids, error):
    with pytest.raises(ValueError, match=error):
        _config(camera_stream_ids=camera_ids)


@pytest.mark.integration
def test_native_writer_publishes_complete_v3_tree_and_metadata(
    tmp_path: Path, priority_profile_fixture
):
    writer, backend = _write_fixture(tmp_path, priority_profile_fixture)

    result = writer.validate(backend)
    assert result.serialization == "passed"
    assert result.structural == "passed"
    assert backend.exists("_ovannotators/_SUCCESS")
    assert not writer.staging_root.exists()

    # Check meta/info.json
    info = json.loads(backend.read("meta/info.json"))
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == 10
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 9
    assert info["total_tasks"] == 2
    assert info["splits"] == {"train": "0:3"}
    assert set(info["features"]) == {
        "observation.state",
        "action",
        "source.timestamp_ns",
        "source.frame_id",
        "observation.images.front",
        "source.camera.front.timestamp_ns",
        "observation.images.wrist",
        "source.camera.wrist.timestamp_ns",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    front_info = info["features"]["observation.images.front"]["info"]
    assert front_info["video.codec"] == "h264"
    assert front_info["video.fps"] == 10
    assert front_info["video.channels"] == 3
    assert front_info["video.g"] == 2  # LeRobot's default keyframe interval

    # Check data/chunk-000/file-000.parquet (one row group per episode)
    data_pf = pq.ParquetFile(
        io.BytesIO(backend.read("data/chunk-000/file-000.parquet"))
    )
    assert data_pf.num_row_groups == 3
    data_tbl = data_pf.read()
    assert data_tbl.num_rows == 9
    # Vectors are fixed-size lists, as LeRobot declares Sequence(length=N).
    assert data_tbl.schema.field("observation.state").type == pa.list_(pa.float32(), 2)
    assert data_tbl["source.timestamp_ns"][0].as_py() == 5_000_000_000
    assert data_tbl["observation.state"][-1].as_py() == pytest.approx([202.1, 202.2])
    assert data_tbl["source.camera.wrist.timestamp_ns"].to_pylist()[:3] == [
        5_000_000_000,
        5_000_000_000,
        5_200_000_006,
    ]

    # Check meta/tasks.parquet has pandas index metadata for 'task'
    tasks_tbl = pq.read_table(io.BytesIO(backend.read("meta/tasks.parquet")))
    assert tasks_tbl.num_rows == 2
    assert b"pandas" in tasks_tbl.schema.metadata
    pandas_meta = json.loads(tasks_tbl.schema.metadata[b"pandas"].decode())
    assert pandas_meta["index_columns"] == ["task"]

    # Check meta/episodes/chunk-000/file-000.parquet
    ep_tbl = pq.read_table(
        io.BytesIO(backend.read("meta/episodes/chunk-000/file-000.parquet"))
    )
    assert ep_tbl.num_rows == 3
    assert ep_tbl["dataset_from_index"].to_pylist() == [0, 3, 6]
    assert ep_tbl["dataset_to_index"].to_pylist() == [3, 6, 9]
    # Shard offsets are frame counts divided by fps, with no float accumulation.
    assert ep_tbl["videos/observation.images.front/from_timestamp"].to_pylist() == [
        0 / 10,
        3 / 10,
        6 / 10,
    ]
    assert ep_tbl["videos/observation.images.front/to_timestamp"].to_pylist() == [
        3 / 10,
        6 / 10,
        9 / 10,
    ]

    # Check concatenated MP4 video shard has all 9 frames (3 episodes x 3 frames)
    front_mp4 = backend.read("videos/observation.images.front/chunk-000/file-000.mp4")
    assert mp4_duration_s(front_mp4) == pytest.approx(0.9)
    front_idx = parse_mp4_video_index(front_mp4)
    assert front_idx["kf_indices"] == [0, 3, 6]

    # Check meta/stats.json
    stats = json.loads(backend.read("meta/stats.json"))
    assert set(stats["observation.state"]) >= {
        "min",
        "max",
        "mean",
        "std",
        "count",
        "q01",
        "q50",
        "q99",
    }
    assert stats["observation.state"]["count"] == [9]
    assert np.asarray(stats["observation.images.front"]["mean"]).shape == (3, 1, 1)
    assert stats["observation.images.front"]["count"] == [9]
    # LeRobot computes timestamp statistics from float64 values.
    assert stats["timestamp"]["max"] == [0.2]

    # Check conformance & profile artifacts
    conformance = json.loads(backend.read("_ovannotators/conformance.json"))
    assert conformance["canonical_loader"] == "not_run"
    record = json.loads(backend.read("_ovannotators/writer_record.json"))
    assert record["writer_version"] == "3"
    assert record["canonical_loader_versions"] == {}
    assert record["codec_versions"] == {"video": "h264"}
    assert all(
        not path.startswith("_ovannotators/") for path in record["output_identifiers"]
    )
    loss_report = json.loads(backend.read("_ovannotators/loss_report.json"))
    assert "sample.frame_id" in loss_report["consumed"]
    assert {
        "sample.sample_id",
        "state.units",
        "state.representation",
        "state.coordinate_frame",
        "action.units",
        "action.representation",
        "action.coordinate_frame",
    } <= set(loss_report["ignored"])


@pytest.mark.integration
def test_native_writer_supports_av1_codec_and_gop_metadata(
    tmp_path: Path, priority_profile_fixture
):
    writer, backend = _write_fixture(
        tmp_path,
        priority_profile_fixture,
        video_codec="av1",
        video_preset="P4",
        video_gop=15,
    )
    assert writer.validate(backend).structural == "passed"
    info = json.loads(backend.read("meta/info.json"))
    vinfo = info["features"]["observation.images.front"]["info"]
    assert vinfo["video.codec"] == "av1"
    assert vinfo["video.g"] == 15
    assert vinfo["video.preset"] == "P4"
    mp4 = backend.read("videos/observation.images.front/chunk-000/file-000.mp4")
    assert b"av01" in mp4
    assert b"av1C" in mp4


@pytest.mark.integration
def test_episode_retry_commits_rows_and_cameras_once(
    tmp_path, priority_profile_fixture
):
    calls = 0

    def fail_second_camera(frames, fps, **options):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("encoder unavailable")
        return _synthetic_mp4_encoder(frames, fps, **options)

    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"), video_encoder=fail_second_camera
    )
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(_mixed_rate(sample), backend)
    with pytest.raises(OSError, match="encoder unavailable"):
        writer.end_episode(backend)
    writer.end_episode(backend)
    writer.finalize(backend)

    data = pq.read_table(io.BytesIO(backend.read("data/chunk-000/file-000.parquet")))
    assert data["index"].to_pylist() == [0, 1, 2]
    for camera in ("front", "wrist"):
        clip = backend.read(
            f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        )
        assert mp4_duration_s(clip) == pytest.approx(0.3)


def _native_test_writer(stage, use_lance):
    config = _config(staging_dir=stage)
    if use_lance:
        pytest.importorskip("lance", reason="requires pylance")
        return LeRobotLanceWriter(
            LeRobotLanceWriterConfig(
                fps=config.fps,
                camera_stream_ids=config.camera_stream_ids,
                state=LeRobotLanceFeatureSpec(**asdict(config.state)),
                action=LeRobotLanceFeatureSpec(**asdict(config.action)),
                repo_id=config.repo_id,
                staging_dir=stage,
            ),
            video_encoder=_synthetic_mp4_encoder,
        )
    return LeRobotV3Writer(config, video_encoder=_synthetic_mp4_encoder)


@pytest.mark.integration
@pytest.mark.parametrize("use_lance", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_local_staging_and_output_must_be_disjoint(
    tmp_path, priority_profile_fixture, use_lance, reverse
):
    stage, output = tmp_path / "capture", tmp_path / "capture/output"
    if reverse:
        stage, output = output, stage
    writer = _native_test_writer(stage, use_lance)
    overlapping = DiskBackend(DiskBackendConfig(output))
    episode = priority_profile_fixture.episodes[0]
    with pytest.raises(ValueError, match="must not overlap"):
        writer.begin_episode(episode, overlapping)
    assert not stage.exists()
    assert not output.exists()

    safe = DiskBackend(DiskBackendConfig(tmp_path / "safe-output"))
    writer.begin_episode(episode, safe)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(_mixed_rate(sample), safe)
    writer.end_episode(safe)
    with pytest.raises(ValueError, match="must not overlap"):
        writer.finalize(overlapping)
    writer.finalize(safe)
    assert writer.validate(safe).structural == "passed"


def _stage_first_episode(writer, backend, fixture):
    writer.begin_episode(fixture.episodes[0], backend)
    for sample in fixture.samples[0]:
        writer.write_sample(_mixed_rate(sample), backend)
    writer.end_episode(backend)


@pytest.mark.integration
@pytest.mark.parametrize("parent_is_lance", [False, True])
@pytest.mark.parametrize("child_is_lance", [False, True])
def test_native_writer_rejects_nested_active_stage(
    tmp_path, priority_profile_fixture, parent_is_lance, child_is_lance
):
    stage = tmp_path / "stage"
    parent = _native_test_writer(stage, parent_is_lance)
    backend = DiskBackend(DiskBackendConfig(tmp_path / "parent-output"))
    _stage_first_episode(parent, backend, priority_profile_fixture)
    if sys.platform != "win32":
        assert stage.stat().st_mode & 0o077 == 0

    child = _native_test_writer(stage / "child", child_is_lance)
    child_output = DiskBackend(DiskBackendConfig(tmp_path / "child-output"))
    with pytest.raises(ValueError, match="nested inside another writer"):
        child.begin_episode(priority_profile_fixture.episodes[0], child_output)
    assert not child.staging_root.exists()
    assert child_output.list() == []
    parent.finalize(backend)
    assert parent.validate(backend).structural == "passed"


@pytest.mark.integration
@pytest.mark.parametrize("use_lance", [False, True])
@pytest.mark.parametrize("symlink", [False, True])
def test_native_writer_preserves_foreign_staging_entries(
    tmp_path, priority_profile_fixture, use_lance, symlink
):
    stage = tmp_path / "stage"
    writer = _native_test_writer(stage, use_lance)
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    _stage_first_episode(writer, backend, priority_profile_fixture)

    foreign = stage / "foreign"
    if symlink:
        source = tmp_path / "source"
        source.write_bytes(b"other producer")
        try:
            foreign.symlink_to(source)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are unavailable")
    else:
        foreign.write_bytes(b"other producer")
    with pytest.raises(ValueError, match="foreign entry"):
        writer.finalize(backend)
    assert foreign.read_bytes() == b"other producer"
    assert backend.list() == []


@pytest.mark.integration
@pytest.mark.parametrize("use_lance", [False, True])
@pytest.mark.parametrize("damage", [None, b"", b"{", b"[]", b'{"meaning":"complete"}'])
def test_native_completion_rejects_damaged_markers(
    tmp_path, priority_profile_fixture, use_lance, damage
):
    writer = _native_test_writer(tmp_path / "stage", use_lance)
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    _stage_first_episode(writer, backend, priority_profile_fixture)
    writer.finalize(backend)
    assert writer.validate(backend).structural == "passed"
    marker = tmp_path / "output/_ovannotators/_SUCCESS"
    if damage is None:
        marker.unlink()
    else:
        marker.write_bytes(damage)
    result = writer.validate(backend)
    assert result.structural == "failed"
    assert any("_SUCCESS" in message for message in result.messages)


@pytest.mark.integration
@pytest.mark.parametrize("use_lance", [False, True])
def test_native_finalize_ignores_a_leftover_owned_publish_temporary(
    tmp_path, priority_profile_fixture, monkeypatch, use_lance
):
    from ovannotators.backends import disk

    writer = _native_test_writer(tmp_path / "stage", use_lance)
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    _stage_first_episode(writer, backend, priority_profile_fixture)
    monkeypatch.setattr(disk, "_rename_without_overwrite", lambda *_: False)
    unlink = Path.unlink
    busy = []

    def fail_once(path, *args, **kwargs):
        if not busy and path.name.endswith(".tmp") and path.exists():
            busy.append(path)
            raise PermissionError(errno.EACCES, "temporary is busy")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    writer.finalize(backend)
    writer.finalize(backend)
    assert busy and busy[0].exists()
    assert writer.validate(backend).structural == "passed"
    assert busy[0].relative_to(backend._root).as_posix() not in backend.list()


@pytest.mark.integration
def test_lance_finalize_survives_a_work_directory_cleanup_failure(
    tmp_path, priority_profile_fixture, monkeypatch, caplog
):
    from ovannotators._internal import filesystem

    writer = _native_test_writer(tmp_path / "stage", True)
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    _stage_first_episode(writer, backend, priority_profile_fixture)
    real_rmtree = filesystem.shutil.rmtree
    failed = []

    def fail_work_directory_once(path, *args, **kwargs):
        if not failed and Path(path).name.startswith(".ovannotators-work-"):
            failed.append(Path(path))
            raise PermissionError(errno.EACCES, "work directory is busy")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(filesystem.shutil, "rmtree", fail_work_directory_once)
    with caplog.at_level("WARNING", logger="ovannotators._internal.filesystem"):
        writer.finalize(backend)
    assert failed
    assert "Could not remove work directory" in caplog.text
    assert writer.validate(backend).structural == "passed"


@pytest.mark.integration
def test_native_generation_claim_blocks_a_different_producer(
    tmp_path, priority_profile_fixture, monkeypatch
):
    output = tmp_path / "output"
    first_backend = DiskBackend(DiskBackendConfig(output, overwrite=True))
    other_backend = DiskBackend(DiskBackendConfig(output, overwrite=True))
    first = _native_test_writer(tmp_path / "first-stage", False)
    other = _native_test_writer(tmp_path / "other-stage", False)
    _stage_first_episode(first, first_backend, priority_profile_fixture)
    other_fixture = replace(
        priority_profile_fixture,
        episodes=(replace(priority_profile_fixture.episodes[0], task="other task"),),
    )
    _stage_first_episode(other, other_backend, other_fixture)
    create = first_backend.write_if_absent
    attempted = []

    def compete(path, data):
        created = create(path, data)
        if not attempted and path == "_ovannotators/payload_checksums.json":
            attempted.append(path)
            with pytest.raises(
                ValueError, match="different bytes: .*payload_checksums"
            ):
                other.finalize(other_backend)
            assert other_backend.list() == [path]
        return created

    monkeypatch.setattr(first_backend, "write_if_absent", compete)
    first.finalize(first_backend)
    assert attempted
    assert first.validate(first_backend).structural == "passed"


@pytest.mark.integration
def test_incompatible_video_starts_a_new_shard(tmp_path, priority_profile_fixture):
    episode_index = 0

    def encode(frames, fps, **options):
        clip = _synthetic_mp4_encoder(frames, fps, **options)
        if episode_index:
            return clip.replace(b"avc1", b"avc3")
        return clip

    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage", chunks_size=1), video_encoder=encode
    )
    for episode_index in range(3):
        writer.begin_episode(priority_profile_fixture.episodes[episode_index], backend)
        for sample in priority_profile_fixture.samples[episode_index]:
            writer.write_sample(_mixed_rate(sample), backend)
        writer.end_episode(backend)

    writer.finalize(backend)
    rows = pq.read_table(
        io.BytesIO(backend.read("meta/episodes/chunk-000/file-000.parquet"))
    ).to_pylist()
    for camera in ("front", "wrist"):
        key = f"videos/observation.images.{camera}"
        assert [row[f"{key}/chunk_index"] for row in rows] == [0, 1, 1]
        assert [row[f"{key}/file_index"] for row in rows] == [0, 0, 0]
        assert [row[f"{key}/from_timestamp"] for row in rows] == pytest.approx(
            [0.0, 0.0, 0.3]
        )
        assert mp4_duration_s(backend.read(f"{key}/chunk-001/file-000.mp4")) == (
            pytest.approx(0.6)
        )


@pytest.mark.integration
def test_video_shard_stays_within_the_indexed_sample_limit(
    tmp_path, priority_profile_fixture, monkeypatch
):
    from ovannotators.profile_writers import lerobot_v3

    # Two three-frame episodes fill a shard exactly; the third starts a new one.
    monkeypatch.setattr(lerobot_v3, "_MAX_INDEXED_SAMPLES", 6)
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"), video_encoder=_synthetic_mp4_encoder
    )
    for episode_index in range(3):
        writer.begin_episode(priority_profile_fixture.episodes[episode_index], backend)
        for sample in priority_profile_fixture.samples[episode_index]:
            writer.write_sample(_mixed_rate(sample), backend)
        writer.end_episode(backend)

    writer.finalize(backend)
    rows = pq.read_table(
        io.BytesIO(backend.read("meta/episodes/chunk-000/file-000.parquet"))
    ).to_pylist()
    for camera in ("front", "wrist"):
        key = f"videos/observation.images.{camera}"
        assert [row[f"{key}/file_index"] for row in rows] == [0, 0, 1]
        assert [row[f"{key}/from_timestamp"] for row in rows] == pytest.approx(
            [0.0, 0.3, 0.0]
        )
        durations = [
            mp4_duration_s(backend.read(f"{key}/chunk-000/file-00{index}.mp4"))
            for index in (0, 1)
        ]
        assert durations == pytest.approx([0.6, 0.3])


@pytest.mark.integration
@pytest.mark.parametrize(
    ("encoded_frames", "error"), [(0, "missing moov"), (1, "3 indexed camera frames")]
)
def test_incomplete_video_blocks_episode_commit(
    tmp_path, priority_profile_fixture, encoded_frames, error
):
    def incomplete_encoder(frames, fps, **options):
        if not encoded_frames:
            return b""
        return _synthetic_mp4_encoder(frames[:encoded_frames], fps, **options)

    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"), video_encoder=incomplete_encoder
    )
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(_mixed_rate(sample), backend)
    with pytest.raises(ValueError, match=error):
        writer.end_episode(backend)
    with pytest.raises(RuntimeError, match="active episode"):
        writer.finalize(backend)
    assert backend.list() == []


@pytest.mark.integration
@pytest.mark.parametrize(
    "failed_path", ["meta/info.json", "_ovannotators/payload_checksums.json"]
)
def test_finalize_retries_without_backend_overwrite(
    tmp_path, priority_profile_fixture, monkeypatch, failed_path
):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    original_write = backend.write_if_absent

    def fail_once(path, data):
        if path == failed_path:
            monkeypatch.setattr(backend, "write_if_absent", original_write)
            raise OSError("destination unavailable")
        return original_write(path, data)

    monkeypatch.setattr(backend, "write_if_absent", fail_once)
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"), video_encoder=_synthetic_mp4_encoder
    )
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(_mixed_rate(sample), backend)
    writer.end_episode(backend)

    with pytest.raises(OSError, match="destination unavailable"):
        writer.finalize(backend)
    assert writer.staging_root.is_dir()
    assert not backend.exists("_ovannotators/_SUCCESS")
    with pytest.raises(RuntimeError, match="after finalization"):
        writer.begin_episode(priority_profile_fixture.episodes[1], backend)

    writer.finalize(backend)
    assert writer.validate(backend).structural == "passed"
    assert not writer.staging_root.exists()


@pytest.mark.unit
def test_missing_signal_is_rejected_without_mutating_active_rows(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"),
        video_encoder=_synthetic_mp4_encoder,
    )
    episode = priority_profile_fixture.episodes[0]
    writer.begin_episode(episode, backend)
    sample = priority_profile_fixture.samples[0][0]
    broken = replace(sample, policy_signals={})
    with pytest.raises(ValueError, match="missing required signals"):
        writer.write_sample(broken, backend)
    assert writer._active_rows == []


@pytest.mark.unit
def test_non_uint8_camera_is_rejected_at_begin_episode(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"),
        video_encoder=_synthetic_mp4_encoder,
    )
    episode = priority_profile_fixture.episodes[0]
    float_camera = tuple(
        replace(stream, dtype="float32")
        if stream.stream_id == "camera.front"
        else stream
        for stream in episode.streams
    )
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        writer.begin_episode(replace(episode, streams=float_camera), backend)


@pytest.mark.unit
def test_timestamp_and_mixed_rate_contracts(tmp_path: Path, priority_profile_fixture):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"),
        video_encoder=_synthetic_mp4_encoder,
    )
    episode = priority_profile_fixture.episodes[0]
    samples = priority_profile_fixture.samples[0]
    writer.begin_episode(episode, backend)
    writer.write_sample(samples[0], backend)
    with pytest.raises(ValueError, match="outside its declared source cadence"):
        writer.write_sample(samples[1], backend)
    with pytest.raises(ValueError, match="CFR timeline"):
        writer.write_sample(
            replace(_mixed_rate(samples[1]), timestamp_ns=5_100_200_000), backend
        )


@pytest.mark.integration
def test_finalize_is_idempotent(tmp_path: Path, priority_profile_fixture):
    writer, backend = _write_fixture(tmp_path, priority_profile_fixture)
    paths = backend.list()
    writer.finalize(backend)
    assert backend.list() == paths


@pytest.mark.unit
def test_registry_and_direct_import_agree():
    assert ProfileWriterRegistry.get("lerobot_v3") is LeRobotV3Writer


@pytest.mark.unit
def test_checksum_corruption_fails_validation(tmp_path: Path, priority_profile_fixture):
    writer, backend = _write_fixture(tmp_path, priority_profile_fixture)
    path = tmp_path / "output" / "data/chunk-000/file-000.parquet"
    path.write_bytes(b"corrupt")
    result = writer.validate(backend)
    assert result.structural == "failed"
    assert "checksum mismatch" in " ".join(result.messages)


@pytest.mark.unit
@pytest.mark.parametrize(
    "variant",
    [
        "missing",
        "empty",
        "partial",
        "extra",
        "bad-json",
        "bad-schema",
        "bad-files",
        "bad-algorithm",
        "bad-digest",
    ],
)
def test_completed_validation_requires_complete_valid_checksum_evidence(
    tmp_path: Path, priority_profile_fixture, variant
):
    writer, backend = _write_fixture(tmp_path, priority_profile_fixture)
    assert writer.validate(backend).structural == "passed"
    path = tmp_path / "output/_ovannotators/payload_checksums.json"
    manifest = json.loads(path.read_bytes())
    if variant == "missing":
        path.unlink()
    else:
        if variant == "empty":
            manifest["files"] = {}
        elif variant == "partial":
            manifest["files"].pop("data/chunk-000/file-000.parquet")
        elif variant == "extra":
            manifest["files"]["missing.parquet"] = "0" * 64
        elif variant == "bad-schema":
            manifest = []
        elif variant == "bad-files":
            manifest["files"] = []
        elif variant == "bad-algorithm":
            manifest["algorithm"] = "md5"
        elif variant == "bad-digest":
            manifest["files"]["data/chunk-000/file-000.parquet"] = 123
        path.write_text("{" if variant == "bad-json" else json.dumps(manifest))
    result = writer.validate(backend)
    assert result.structural == "failed"
    assert not result.success_marker_allowed
    assert result.messages


@pytest.mark.unit
def test_precompletion_validation_does_not_require_completion_evidence(
    tmp_path: Path, priority_profile_fixture
):
    from ovannotators.profile_writers.lerobot_v3 import _validate_structure

    _, backend = _write_fixture(tmp_path, priority_profile_fixture)
    for path in (tmp_path / "output/_ovannotators").iterdir():
        path.unlink()
    assert _validate_structure(backend, require_success=False).structural == "passed"
    assert _validate_structure(backend, require_success=True).structural == "failed"


@pytest.mark.unit
def test_canonical_loader_is_lazy_and_directs_to_conformance_env():
    from ovannotators.profile_writers.lerobot_v3 import _official_dataset_type

    if sys.version_info < (3, 12):
        with pytest.raises(RuntimeError, match="Python 3.12"):
            _official_dataset_type()
    elif importlib.util.find_spec("lerobot") is None:
        with pytest.raises(ImportError, match="lerobot-v3-conformance"):
            _official_dataset_type()


class _RecordingDataset:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"index": index}


@pytest.fixture
def recording_loader(monkeypatch):
    from ovannotators.profile_writers import lerobot_v3

    _RecordingDataset.calls = []
    monkeypatch.setattr(
        lerobot_v3,
        "_official_dataset_type",
        lambda expected_version: (_RecordingDataset, expected_version),
    )
    # Run the loader in-process so the recording stub is visible; the offline
    # child process is covered by the _run_offline_loader tests.
    monkeypatch.setattr(
        lerobot_v3,
        "_run_offline_loader",
        lambda root, repo_id, version: lerobot_v3._load_official(
            str(root), repo_id, version
        ),
    )
    return _RecordingDataset.calls


def _rewrite_info(output: Path, change) -> None:
    """Edit meta/info.json and re-record its digest so only the edit differs."""
    from ovannotators.profile_writers._artifacts import sha256_bytes

    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    change(info)
    info_path.write_text(json.dumps(info))
    manifest_path = output / "_ovannotators/payload_checksums.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["meta/info.json"] = sha256_bytes(info_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))


def _drop_payload(output: Path, relative: str) -> None:
    """Delete a payload file and its digest so the manifest still matches."""
    (output / relative).unlink()
    manifest_path = output / "_ovannotators/payload_checksums.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"][relative]
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.unit
def test_canonical_validation_uses_supplied_repo_id(
    tmp_path: Path, priority_profile_fixture, recording_loader
):
    _write_fixture(tmp_path, priority_profile_fixture)
    output = tmp_path / "output"

    result = validate_lerobot_v3_canonical(output, repo_id="team/dataset")

    assert result.canonical_loader == "passed", result.messages
    assert result.evidence == {"loader": f"lerobot=={LEROBOT_VERSION}"}
    assert recording_loader == [{"repo_id": "team/dataset", "root": output}]

    probe = validate_lerobot_v3_canonical(
        output, repo_id="team/dataset", expected_version="99.0.0"
    )
    assert probe.evidence == {"loader": "lerobot==99.0.0"}


@pytest.mark.unit
@pytest.mark.skipif(sys.version_info < (3, 12), reason="loader needs Python 3.12")
def test_official_loader_must_match_the_expected_version(monkeypatch):
    from ovannotators.profile_writers.lerobot_v3 import _official_dataset_type

    loader_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    loader_module.LeRobotDataset = _RecordingDataset
    for name in ("lerobot", "lerobot.datasets"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", loader_module)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "99.0.0")

    with pytest.raises(
        RuntimeError,
        match=rf"requires lerobot=={re.escape(LEROBOT_VERSION)}, got 99.0.0",
    ):
        _official_dataset_type()
    assert _official_dataset_type("99.0.0") == (_RecordingDataset, "99.0.0")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda out: None, "not a directory"),
        (
            lambda out: _rewrite_info(
                out, lambda info: info.update(storage_format="lance")
            ),
            "storage_format must be 'lerobot'",
        ),
        (
            lambda out: (out / "_ovannotators/_SUCCESS").unlink(),
            "missing final _ovannotators/_SUCCESS marker",
        ),
        (
            lambda out: (out / "data/chunk-000/file-000.parquet").write_bytes(b"x"),
            "checksum mismatch: data/chunk-000/file-000.parquet",
        ),
        (
            lambda out: _rewrite_info(
                out,
                lambda info: info["features"].update(
                    {"observation.image": {"dtype": "image", "shape": [1, 1, 3]}}
                ),
            ),
            r"image features .* \['observation.image'\]",
        ),
        (
            lambda out: _rewrite_info(
                out,
                lambda info: info.update(video_path="https://example.invalid/v.mp4"),
            ),
            "video_path must be relative",
        ),
        (
            lambda out: _rewrite_info(
                out,
                lambda info: info.update(data_path="../outside/{chunk_index}.parquet"),
            ),
            "data_path must be relative",
        ),
        (
            lambda out: _rewrite_info(
                out,
                lambda info: info.update(
                    data_path="data/{chunk_index.no_such_field}.parquet"
                ),
            ),
            "data_path has an invalid template",
        ),
        (
            lambda out: _drop_payload(
                out, "videos/observation.images.wrist/chunk-000/file-000.mp4"
            ),
            "referenced shard is missing: "
            "videos/observation.images.wrist/chunk-000/file-000.mp4",
        ),
        (
            lambda out: _rewrite_info(
                out,
                lambda info: info.update(
                    video_path="videos/{video_key}/chunk-{chunk_index:03d}/x.mp4"
                ),
            ),
            r"referenced shard is missing: videos/observation\.images\.front/",
        ),
    ],
)
def test_canonical_validation_does_not_open_unverified_trees(
    tmp_path: Path, priority_profile_fixture, recording_loader, damage, message
):
    output = tmp_path / "output"
    if "not a directory" in message:
        output = tmp_path / "missing"
    else:
        _write_fixture(tmp_path, priority_profile_fixture)
        damage(output)

    result = validate_lerobot_v3_canonical(output, "team/dataset")

    assert (result.structural, result.canonical_loader) == ("failed", "not_run")
    assert any(re.search(message, m) for m in result.messages), result.messages
    assert recording_loader == []


@pytest.mark.integration
def test_canonical_validation_rejects_unlisted_episode_symlink(
    tmp_path: Path, priority_profile_fixture, recording_loader
):
    _, backend = _write_fixture(tmp_path, priority_profile_fixture)
    root = backend._root
    external = tmp_path / "outside.parquet"
    external.write_bytes(b"outside the dataset")
    shard = next((root / "meta/episodes").rglob("*.parquet"))
    try:
        shard.with_name("unlisted.parquet").symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform/user")

    result = validate_lerobot_v3_canonical(root, "team/dataset")

    assert (result.structural, result.canonical_loader) == ("failed", "not_run")
    assert any(
        "symlink resolves outside the dataset: meta/episodes/" in message
        for message in result.messages
    )
    assert recording_loader == []


@pytest.mark.integration
def test_canonical_validation_rejects_unlisted_data_symlink(
    tmp_path: Path, priority_profile_fixture, recording_loader
):
    _, backend = _write_fixture(tmp_path, priority_profile_fixture)
    root = backend._root
    external = tmp_path / "outside.parquet"
    external.write_bytes(b"outside the dataset")
    shard = next((root / "data").rglob("*.parquet"))
    try:
        shard.with_name("extra.parquet").symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform/user")

    result = validate_lerobot_v3_canonical(root, "team/dataset")

    assert (result.structural, result.canonical_loader) == ("failed", "not_run")
    assert any(
        "symlink resolves outside the dataset: data/" in message
        and message.endswith("extra.parquet")
        for message in result.messages
    ), result.messages
    assert recording_loader == []


@pytest.mark.integration
def test_canonical_validation_accepts_symlinks_that_stay_inside_the_dataset(
    tmp_path: Path, priority_profile_fixture, recording_loader
):
    _, backend = _write_fixture(tmp_path, priority_profile_fixture)
    root = backend._root
    shard = next((root / "data").rglob("*.parquet"))
    try:
        shard.with_name("alias.parquet").symlink_to(shard.name)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform/user")

    result = validate_lerobot_v3_canonical(root, "team/dataset")

    assert not any("symlink resolves" in message for message in result.messages)


@pytest.mark.parametrize("partition", ["data", "meta/episodes", "meta/tasks.parquet"])
def test_canonical_validation_rejects_parquet_decoder_metadata(
    tmp_path, priority_profile_fixture, monkeypatch, partition
):
    _write_fixture(tmp_path, priority_profile_fixture)
    root = tmp_path / "output"
    shard = root / partition
    if shard.is_dir():
        shard = next(shard.glob("*/*.parquet"))
    table = pq.read_table(shard)
    metadata = {
        b"huggingface": json.dumps(
            {"info": {"features": {"aux": {"nested": [{"_type": "Image"}]}}}}
        ).encode(),
    }
    pq.write_table(table.replace_schema_metadata(metadata), shard)
    manifest_path = root / "_ovannotators/payload_checksums.json"
    manifest = json.loads(manifest_path.read_text())
    from ovannotators.profile_writers._artifacts import sha256_bytes

    manifest["files"][shard.relative_to(root).as_posix()] = sha256_bytes(
        shard.read_bytes()
    )
    manifest_path.write_text(json.dumps(manifest))
    from ovannotators._internal import lerobot_common

    def forbidden(*args, **kwargs):
        pytest.fail("the loader subprocess opened Parquet with decoder metadata")

    monkeypatch.setattr(lerobot_common.subprocess, "run", forbidden)
    result = validate_lerobot_v3_canonical(root, "team/dataset")
    assert (result.structural, result.canonical_loader) == ("failed", "not_run")
    assert any("Image" in message for message in result.messages)


@pytest.mark.parametrize(
    ("payload", "exit_code", "error"),
    [
        ('{"version":"ok"}', 17, "exited 17"),
        ("{broken", 0, "invalid JSON"),
        ("[]", 0, "invalid result"),
        ("{}", 0, "invalid result"),
    ],
)
def test_offline_loader_requires_successful_exit_and_a_result(
    payload, exit_code, error
):
    from ovannotators._internal.lerobot_common import (
        _LOADER_RESULT_MARKER,
        run_offline_loader,
    )

    message = _LOADER_RESULT_MARKER + payload
    result = run_offline_loader(
        f"import sys; print({message!r}, flush=True); sys.exit({exit_code})"
    )
    assert error in result["error"]


def test_offline_loader_handles_signals_and_launch_failure(monkeypatch):
    from ovannotators._internal import lerobot_common

    message = lerobot_common._LOADER_RESULT_MARKER + '{"version":"ok"}'
    script = f"import os, signal; print({message!r}, flush=True); os.kill(os.getpid(), signal.SIGTERM)"
    assert "process exited" in lerobot_common.run_offline_loader(script)["error"]

    def missing_interpreter(*args, **kwargs):
        raise FileNotFoundError("missing interpreter")

    monkeypatch.setattr(lerobot_common.subprocess, "run", missing_interpreter)
    outcome = lerobot_common.run_offline_loader("")
    assert outcome == {"error": "could not launch loader process: missing interpreter"}


@pytest.mark.integration
def test_canonical_validation_rejects_success_from_a_failed_child(
    tmp_path, priority_profile_fixture, monkeypatch
):
    from ovannotators.profile_writers import lerobot_v3

    _, backend = _write_fixture(tmp_path, priority_profile_fixture)
    message = lerobot_v3._LOADER_RESULT_MARKER + '{"version":"ok"}'
    monkeypatch.setattr(
        lerobot_v3,
        "_OFFLINE_LOADER_SCRIPT",
        f"import sys; print({message!r}, flush=True); sys.exit(17)",
    )
    result = validate_lerobot_v3_canonical(backend._root, "team/dataset")
    assert (result.structural, result.canonical_loader) == ("passed", "failed")
    assert "exited 17" in result.messages[0]


@pytest.mark.unit
def test_offline_loader_runs_in_a_child_with_hub_access_disabled(
    tmp_path: Path, monkeypatch
):
    from ovannotators.profile_writers import lerobot_v3

    marker = lerobot_v3._LOADER_RESULT_MARKER
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        lerobot_v3,
        "_OFFLINE_LOADER_SCRIPT",
        "import json, os, sys\n"
        "print('loader chatter')\n"
        f"print({marker!r} + json.dumps({{'version': ':'.join(\n"
        "    [os.environ['HF_HUB_OFFLINE'], os.environ['HF_DATASETS_OFFLINE'],\n"
        "     *sys.argv[1:4]])}))\n",
    )

    outcome = lerobot_v3._run_offline_loader(tmp_path, "team/dataset", LEROBOT_VERSION)

    assert outcome == {"version": f"1:1:{tmp_path}:team/dataset:{LEROBOT_VERSION}"}


@pytest.mark.unit
def test_offline_loader_reports_child_failures(tmp_path: Path, monkeypatch):
    from ovannotators.profile_writers import lerobot_v3

    monkeypatch.setattr(
        lerobot_v3,
        "_OFFLINE_LOADER_SCRIPT",
        "import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n",
    )

    outcome = lerobot_v3._run_offline_loader(tmp_path, "team/dataset", LEROBOT_VERSION)

    assert outcome == {"error": "loader process exited 3: boom"}


@pytest.mark.unit
@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is not None,
    reason="checks the error path when lerobot is absent",
)
def test_offline_loader_child_reports_a_missing_loader(tmp_path: Path):
    from ovannotators.profile_writers import lerobot_v3

    outcome = lerobot_v3._run_offline_loader(tmp_path, "team/dataset", LEROBOT_VERSION)

    assert re.search(r"Python 3\.12|lerobot-v3-conformance", outcome["error"])


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.duration_10s
@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
@pytest.mark.parametrize("profile", ["lerobot_v3", "lerobot_lance"])
def test_native_lerobot_nvenc_round_trip(tmp_path: Path, codec, profile):
    """Exercise real NVENC and check keyframe spacing in both native profiles."""
    nvenc = pytest.importorskip("PyNvVideoCodec", reason="requires NVENC GPU runtime")
    if codec == "av1":
        try:
            encoder = nvenc.CreateEncoder(
                256, 192, "ARGB", True, codec="av1", fps="10", bf=0
            )
        except Exception as exc:
            message = str(exc).lower()
            if "codec not supported on this gpu" in message:
                pytest.skip(f"selected GPU cannot encode AV1: {exc}")
            raise
        del encoder
    shape = (192, 256, 3)
    episode = EpisodeEnvelope(
        dataset_id="native-v3-smoke",
        sequence_id="sequence-000",
        episode_id="episode-000",
        task="move",
        embodiment="test_arm",
        source_clock="sim",
        source_origin_ns=1_000_000_000,
        nominal_rate_hz=10.0,
        streams=(
            StreamDescriptor("camera.front", "rgb", "sim", 10.0, "uint8", shape),
            StreamDescriptor(
                "observation.state", "state", "sim", 10.0, "float32", (2,)
            ),
            StreamDescriptor("action", "action", "sim", 10.0, "float32", (2,)),
        ),
    )
    config = replace(
        _config(),
        camera_stream_ids=("camera.front",),
        staging_dir=tmp_path / "stage",
        video_codec=codec,
        video_gop=3,
    )
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    if profile == "lerobot_v3":
        writer = LeRobotV3Writer(config)
    else:
        lance = pytest.importorskip("lance", reason="requires pylance")
        writer = LeRobotLanceWriter(
            LeRobotLanceWriterConfig(
                fps=config.fps,
                camera_stream_ids=config.camera_stream_ids,
                state=LeRobotLanceFeatureSpec(**asdict(config.state)),
                action=LeRobotLanceFeatureSpec(**asdict(config.action)),
                repo_id=config.repo_id,
                video_codec=codec,
                video_gop=config.video_gop,
                staging_dir=config.staging_dir,
                canonical_loader_check=(
                    "required"
                    if importlib.util.find_spec("lerobot_lancedb")
                    else "skip"
                ),
            )
        )
    for current in (
        episode,
        replace(
            episode,
            episode_id="episode-001",
            task="place",
            source_origin_ns=2_000_000_000,
        ),
    ):
        writer.begin_episode(current, backend)
        for index in range(8):
            writer.write_sample(
                SampleEnvelope(
                    episode_id=current.episode_id,
                    sample_id=f"{current.episode_id}-sample-{index}",
                    frame_id=index,
                    timestamp_ns=current.source_origin_ns + index * 100_000_000,
                    observations={
                        "camera.front": NamedTensor(
                            np.full(shape, index * 3, dtype=np.uint8)
                        ),
                        "observation.state": NamedTensor(
                            np.asarray([index, index + 1], dtype=np.float32),
                            config.state.feature_names,
                            config.state.units,
                            config.state.representation,
                            config.state.coordinate_frame,
                        ),
                    },
                    policy_signals={
                        "action": NamedTensor(
                            np.asarray([index + 2, index + 3], dtype=np.float32),
                            config.action.feature_names,
                            config.action.units,
                            config.action.representation,
                            config.action.coordinate_frame,
                        )
                    },
                ),
                backend,
            )
        writer.end_episode(backend)
    writer.finalize(backend)
    assert writer.validate(backend).structural == "passed"
    if profile == "lerobot_lance":
        table = lance.dataset(str(tmp_path / "output" / "videos.lance")).to_table(
            columns=["kf_indices"]
        )
        assert table["kf_indices"].to_pylist() == [[0, 3, 6], [0, 3, 6]]
        return
    clip = backend.read("videos/observation.images.front/chunk-000/file-000.mp4")
    assert parse_mp4_video_index(clip)["kf_indices"] == [0, 3, 6, 8, 11, 14]
    if sys.version_info >= (3, 12) and importlib.util.find_spec("lerobot") is not None:
        canonical = validate_lerobot_v3_canonical(
            tmp_path / "output", repo_id=config.repo_id
        )
        assert canonical.canonical_loader == "passed", canonical.messages


@pytest.mark.unit
def test_begin_episode_accepts_equivalent_dtype_and_shape_spellings(
    tmp_path: Path, priority_profile_fixture
) -> None:
    episode = priority_profile_fixture.episodes[0]
    respelled = tuple(
        replace(stream, dtype="|u1", shape=list(stream.shape))
        if stream.stream_type == "rgb"
        else replace(stream, dtype="<f4", shape=list(stream.shape))
        if stream.stream_id == "observation.state"
        else stream
        for stream in episode.streams
    )
    writer = LeRobotV3Writer(
        _config(staging_dir=tmp_path / "stage"), video_encoder=_synthetic_mp4_encoder
    )
    writer.begin_episode(
        replace(episode, streams=respelled),
        DiskBackend(DiskBackendConfig(tmp_path / "output")),
    )


@pytest.mark.parametrize("spelling", ["f4", "<f4", "float32"])
def test_published_feature_dtypes_use_canonical_names(
    tmp_path, priority_profile_fixture, spelling
):
    config = _config()
    _, backend = _write_fixture(
        tmp_path,
        priority_profile_fixture,
        state=replace(config.state, dtype=spelling),
        action=replace(config.action, dtype=spelling),
    )
    info = json.loads(backend.read("meta/info.json"))
    stats = json.loads(backend.read("meta/stats.json"))
    shard = next((tmp_path / "output/data").glob("*/*.parquet"))
    table = pq.read_table(shard)
    for key in ("observation.state", "action"):
        assert info["features"][key]["dtype"] == "float32"
        assert info["features"][key]["shape"] == [2]
        assert str(table.schema.field(key).type.value_type) == "float"
        assert len(stats[key]["mean"]) == 2


@pytest.mark.unit
def test_data_shards_roll_over_on_parquet_size_not_arrow_size(
    tmp_path: Path, priority_profile_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ovannotators.profile_writers import lerobot_v3

    _, backend = _write_fixture(tmp_path, priority_profile_fixture)
    files = [p for p in backend.list() if p.startswith("data/")]
    assert files == ["data/chunk-000/file-000.parquet"]

    # A 1 MiB Parquet estimate per episode exceeds the 1 MB limit, so every
    # episode after the first starts a new shard.
    monkeypatch.setattr(lerobot_v3, "_parquet_size", lambda table: 1024 * 1024)
    _, backend = _write_fixture(
        tmp_path / "rolled", priority_profile_fixture, data_files_size_in_mb=1
    )
    files = sorted(p for p in backend.list() if p.startswith("data/"))
    assert len(files) == len(priority_profile_fixture.episodes)
