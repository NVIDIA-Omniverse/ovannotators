# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native LeRobot Lance profile writer."""

from __future__ import annotations

import json
import shutil
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ovannotators import (
    LeRobotLanceFeatureSpec,
    LeRobotLanceWriter,
    LeRobotLanceWriterConfig,
    ProfileWriterRegistry,
)
from ovannotators._internal.mp4_mux import mux_h264_annexb
from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.profile_writers import lerobot_lance as lance_module
from ovannotators.profile_writers._models import ConformanceResult

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
    del codec, lossless, preset, gop
    h, w = frames[0].shape[:2]
    stream = _SC + _SPS + _SC + _PPS + _SC + _IDR
    for _ in frames[1:]:
        stream += _SC + _NON_IDR
    return mux_h264_annexb(stream, fps=fps, width=w, height=h)


def _config(tmp_path: Path) -> LeRobotLanceWriterConfig:
    return LeRobotLanceWriterConfig(
        fps=10,
        repo_id="team/dataset",
        camera_stream_ids=("camera.front", "camera.wrist"),
        state=LeRobotLanceFeatureSpec(
            "observation.state",
            ("joint.position", "gripper.position"),
            ("rad", "m"),
            "absolute",
            (2,),
            coordinate_frame="robot_base",
        ),
        action=LeRobotLanceFeatureSpec(
            "action",
            ("joint.target", "gripper.target"),
            ("rad", "m"),
            "absolute",
            (2,),
            coordinate_frame="robot_base",
        ),
        staging_dir=tmp_path / "stage",
    )


def _mixed_rate(sample):
    if sample.frame_id % 2 == 0:
        return sample
    observations = dict(sample.observations)
    observations.pop("camera.wrist", None)
    missing = dict(sample.missing_data)
    missing["camera.wrist"] = "not_sampled_at_source_rate"
    return replace(sample, observations=observations, missing_data=missing)


@pytest.mark.unit
def test_config_requires_explicit_semantics(tmp_path: Path):
    with pytest.raises(ValueError, match="camera_stream_ids"):
        replace(_config(tmp_path), camera_stream_ids=())
    with pytest.raises(ValueError, match="feature_names"):
        LeRobotLanceFeatureSpec("state", ("one",), ("rad",), "absolute", (2,))
    for shape in ((), (0,), (-1,), (1, 1), (2, 3)):
        with pytest.raises(ValueError, match="one positive dimension"):
            LeRobotLanceFeatureSpec("state", ("a",) * 6, ("rad",), "absolute", shape)
    with pytest.raises(ValueError, match="timestamp_tolerance_ns"):
        replace(_config(tmp_path), timestamp_tolerance_ns=100_001)
    with pytest.raises(ValueError, match="canonical_loader_check"):
        replace(_config(tmp_path), canonical_loader_check="never")
    with pytest.raises(ValueError, match="video_codec"):
        replace(_config(tmp_path), video_codec="vp9")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="video_gop"):
        replace(_config(tmp_path), video_gop=0)
    # Default canonical_loader_check is "skip" (no lerobot-lancedb runtime dep)
    assert _config(tmp_path).canonical_loader_check == "skip"


@pytest.mark.parametrize(
    ("stream_id", "error"),
    [
        ("index", "output keys must be unique"),
        ("source_timestamp_ns", "Lance column names must be unique"),
    ],
)
def test_state_stream_cannot_collide_with_generated_columns(tmp_path, stream_id, error):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match=error):
        replace(config, state=replace(config.state, stream_id=stream_id))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fps", "rate", "stride"),
    [(10, 10.0, 1), (10, 5.0, 2), (30, 7.5, 4), (10, 4.0, None), (10, 20.0, None)],
)
def test_stride_requires_integer_divisor(fps, rate, stride):
    assert lance_module._stride(fps, rate) == stride


def _ingest(writer, backend, fixture, *, episodes=None):
    for episode, samples in zip(fixture.episodes, fixture.samples, strict=True):
        if episodes is not None and episode.episode_id not in episodes:
            continue
        writer.begin_episode(episode, backend)
        for sample in samples:
            writer.write_sample(_mixed_rate(sample), backend)
        writer.end_episode(backend)


@pytest.mark.unit
def test_mixed_rate_camera_is_zero_order_hold_aligned(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path), video_encoder=_synthetic_mp4_encoder)
    _ingest(writer, backend, priority_profile_fixture, episodes={"episode-000"})

    item = writer._episodes[0]
    wrist_times = [row["source.camera.wrist.timestamp_ns"] for row in item["rows"]]
    source_times = [row["source.timestamp_ns"] for row in item["rows"]]
    assert wrist_times == [source_times[0], source_times[0], source_times[2]]
    # Each camera clip is encoded and staged at end_episode; raw frames are
    # not retained in memory.
    assert "cameras" not in item
    staged = sorted(p.name for p in (writer.staging_root / ".episode-videos").iterdir())
    assert staged == ["000000-000.mp4", "000000-001.mp4"]
    assert [video["byte_index"]["kf_indices"] for video in item["videos"]] == [[0], [0]]


@pytest.mark.unit
def test_slow_camera_needs_marker_and_stays_on_cadence(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path), video_encoder=_synthetic_mp4_encoder)
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    first, second, _ = priority_profile_fixture.samples[0]
    writer.write_sample(first, backend)
    unmarked = replace(
        _mixed_rate(second),
        missing_data={},
    )
    with pytest.raises(ValueError, match="missing required signals"):
        writer.write_sample(unmarked, backend)
    with pytest.raises(ValueError, match="outside its cadence"):
        writer.write_sample(second, backend)


@pytest.mark.unit
def test_non_divisor_camera_rate_is_rejected(tmp_path: Path, priority_profile_fixture):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(
        replace(_config(tmp_path), fps=8), video_encoder=_synthetic_mp4_encoder
    )
    with pytest.raises(ValueError, match="rate must divide dataset fps"):
        writer.begin_episode(priority_profile_fixture.episodes[0], backend)


def _conformance(root: Path) -> dict:
    return json.loads((root / "_ovannotators/conformance.json").read_text())


@pytest.mark.integration
@pytest.mark.cpu_only
@pytest.mark.duration_10s
def test_native_lance_writer_round_trip_with_pylance(
    tmp_path: Path, priority_profile_fixture
):
    """Build real frames.lance, videos.lance (blob v2), and meta.lance on CPU."""
    lance = pytest.importorskip("lance", reason="requires pylance")
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path), video_encoder=_synthetic_mp4_encoder)
    _ingest(writer, backend, priority_profile_fixture)
    writer.finalize(backend)

    assert writer.validate(backend).structural == "passed"
    paths = backend.list()
    assert any(path.startswith("frames.lance/") for path in paths)
    assert any(path.startswith("videos.lance/") for path in paths)
    assert any(path.startswith("meta.lance/") for path in paths)
    assert not any(path.startswith("data/") for path in paths)
    assert not any(path.startswith("videos/observation") for path in paths)
    assert not any(path.startswith(".episode-videos") for path in paths)
    assert backend.exists("_ovannotators/_SUCCESS")
    assert _conformance(tmp_path / "output")["canonical_loader"] == "not_run"
    record = json.loads(
        (tmp_path / "output/_ovannotators/writer_record.json").read_text()
    )
    assert record["canonical_loader_versions"] == {}
    assert not writer.staging_root.exists()

    output = tmp_path / "output"
    info = json.loads((output / "meta/info.json").read_text())
    assert info["storage_format"] == "lance"
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 9

    frames_ds = lance.dataset(str(output / "frames.lance"))
    assert frames_ds.count_rows() == 9
    frames_tbl = frames_ds.to_table()
    assert "observation_state" in frames_tbl.column_names
    assert "source_timestamp_ns" in frames_tbl.column_names
    assert "source_camera_wrist_timestamp_ns" in frames_tbl.column_names

    videos_ds = lance.dataset(str(output / "videos.lance"))
    assert videos_ds.count_rows() == 6
    videos_meta = videos_ds.to_table(
        columns=[
            "video_key",
            "chunk_index",
            "file_index",
            "file_size",
            "moov_offset",
            "moov_size",
            "kf_indices",
            "kf_positions",
        ]
    ).to_pylist()
    assert len(videos_meta) == 6
    assert all(row["moov_offset"] > 0 and row["moov_size"] > 0 for row in videos_meta)
    assert all(row["kf_indices"] == [0] for row in videos_meta)
    blob0 = videos_ds.take_blobs("video_bytes", [0])[0].readall()
    assert blob0[4:8] == b"ftyp"
    assert len(blob0) == videos_meta[0]["file_size"]

    meta_ds = lance.dataset(str(output / "meta.lance"))
    meta_paths = sorted(meta_ds.to_table(columns=["path"])["path"].to_pylist())
    assert "info.json" in meta_paths
    assert "stats.json" in meta_paths
    assert "tasks.parquet" in meta_paths
    assert "episodes/chunk-000/file-000.parquet" in meta_paths

    # Compare manifest against the candidate golden
    manifest = {
        "frames": frames_ds.count_rows(),
        "profile": "lerobot_lance",
        "status": "candidate_pending_human_approval",
        "tables": ["frames", "meta", "videos"],
        "video_blobs": videos_ds.count_rows(),
        "video_keys": sorted(
            k for k, v in info["features"].items() if v["dtype"] == "video"
        ),
    }
    golden = json.loads(
        (
            Path(__file__).parent / "golden/lerobot_lance/candidate/manifest.json"
        ).read_text()
    )
    assert manifest == golden


def _staged_entries(writer) -> list[str]:
    return sorted(path.name for path in writer.staging_root.iterdir())


def _forbid_canonical_loader(monkeypatch):
    def _must_not_run(root):
        raise AssertionError("canonical loader must not run when skipped")

    monkeypatch.setattr(lance_module, "_validate_canonical", _must_not_run)


def _fake_native_tree(root, config, episodes, embodiment, **_):
    del config, episodes, embodiment
    for relative in (
        "frames.lance/data/0.lance",
        "frames.lance/_versions/1.manifest",
        "videos.lance/data/0.lance",
        "meta.lance/data/0.lance",
        "meta/info.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return {"pylance": "11.0.0", "pyarrow": "25.0.1"}


@pytest.fixture
def staged_writer(tmp_path: Path, priority_profile_fixture, monkeypatch):
    """Writer whose native-tree build is faked so finalize logic runs anywhere."""
    monkeypatch.setattr(lance_module, "_write_native_lance", _fake_native_tree)
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))

    def make(**overrides):
        writer = LeRobotLanceWriter(
            replace(_config(tmp_path), **overrides),
            video_encoder=_synthetic_mp4_encoder,
        )
        _ingest(writer, backend, priority_profile_fixture)
        return writer

    return make, backend, tmp_path / "output"


@pytest.mark.unit
def test_finalize_records_the_gate_result_and_removes_staging(
    staged_writer, monkeypatch
):
    make, backend, output = staged_writer
    monkeypatch.setattr(
        lance_module,
        "_validate_canonical",
        lambda root: ConformanceResult("passed", "passed", canonical_loader="passed"),
    )
    writer = make(canonical_loader_check="required")
    writer.finalize(backend)
    writer.finalize(backend)  # idempotent once finalized

    assert backend.exists("_ovannotators/_SUCCESS")
    assert _conformance(output)["canonical_loader"] == "passed"
    record = json.loads((output / "_ovannotators/writer_record.json").read_text())
    assert list(record["canonical_loader_versions"]) == ["lerobot-lancedb"]
    assert writer.validate(backend).structural == "passed"
    assert not writer.staging_root.exists()
    with pytest.raises(RuntimeError, match="after finalization starts"):
        writer.begin_episode(writer._episodes[0]["episode"], backend)


@pytest.mark.unit
@pytest.mark.parametrize("damage", ["missing", "corrupt", "extra"])
def test_validate_reports_changes_to_published_payload(staged_writer, damage):
    make, backend, output = staged_writer
    writer = make()
    writer.finalize(backend)
    assert writer.validate(backend).structural == "passed"

    fragment = output / "frames.lance/data/0.lance"
    if damage == "missing":
        fragment.unlink()
        message = "checksummed payload is missing"
    elif damage == "corrupt":
        fragment.write_bytes(b"corrupt")
        message = "checksum mismatch"
    else:
        fragment.with_name("extra.lance").write_bytes(b"extra")
        message = "checksum paths do not exactly match"
    result = writer.validate(backend)
    assert result.structural == "failed"
    assert any(message in detail for detail in result.messages)


@pytest.mark.unit
def test_skip_mode_never_claims_the_loader_passed(staged_writer, monkeypatch):
    make, backend, output = staged_writer
    _forbid_canonical_loader(monkeypatch)
    writer = make(canonical_loader_check="skip")
    writer.finalize(backend)

    conformance = _conformance(output)
    assert conformance["canonical_loader"] == "not_run"
    assert conformance["structural"] == "passed"
    assert "loader" not in conformance["evidence"]
    assert not writer.staging_root.exists()


def test_required_gate_rejects_success_from_a_failed_child(staged_writer, monkeypatch):
    make, backend, _ = staged_writer
    monkeypatch.setattr(lance_module, "_canonical_preflight", lambda root: [])
    message = lance_module._LOADER_RESULT_MARKER + '{"version":"ok"}'
    monkeypatch.setattr(
        lance_module,
        "_OFFLINE_LOADER_SCRIPT",
        f"import sys; print({message!r}, flush=True); sys.exit(17)",
    )
    writer = make(canonical_loader_check="required")
    with pytest.raises(
        ValueError, match="LanceDBDataset validation failed: .*exited 17"
    ):
        writer.finalize(backend)
    assert backend.list() == []


@pytest.mark.unit
def test_keep_staging_retains_the_native_tree(staged_writer, monkeypatch):
    make, backend, _ = staged_writer
    _forbid_canonical_loader(monkeypatch)
    writer = make(canonical_loader_check="skip", keep_staging=True)
    writer.finalize(backend)

    assert (writer.staging_root / "frames.lance").is_dir()


@pytest.mark.unit
@pytest.mark.parametrize("status", ["failed", "not_run"])
def test_unpassed_gate_blocks_publication_cleans_staging_and_allows_retry(
    staged_writer, monkeypatch, status
):
    make, backend, output = staged_writer
    results = iter(
        [
            ConformanceResult(
                "passed", "passed", canonical_loader=status, messages=("no decode",)
            ),
            ConformanceResult("passed", "passed", canonical_loader="passed"),
        ]
    )
    monkeypatch.setattr(lance_module, "_validate_canonical", lambda root: next(results))
    writer = make(canonical_loader_check="required")

    with pytest.raises(ValueError, match="LanceDBDataset validation failed"):
        writer.finalize(backend)
    assert not backend.exists("_ovannotators/_SUCCESS")
    assert backend.list() == []
    assert _staged_entries(writer) == [".episode-videos", ".ovannotators-stage-owner"]

    writer.finalize(backend)
    assert backend.exists("_ovannotators/_SUCCESS")
    assert _conformance(output)["canonical_loader"] == "passed"
    assert not writer.staging_root.exists()


@pytest.mark.unit
def test_incomplete_tree_is_not_marked_successful(staged_writer, monkeypatch):
    make, backend, _ = staged_writer

    def partial(root, config, episodes, embodiment, **_):
        _fake_native_tree(root, config, episodes, embodiment)
        shutil.rmtree(root / "meta.lance")
        return {}

    monkeypatch.setattr(lance_module, "_write_native_lance", partial)
    _forbid_canonical_loader(monkeypatch)
    writer = make(canonical_loader_check="skip")

    with pytest.raises(ValueError, match="structure is incomplete"):
        writer.finalize(backend)
    assert not backend.exists("_ovannotators/_SUCCESS")
    # Publication had started, so the built tree is kept for the retry.
    assert "frames.lance" in _staged_entries(writer)
    assert writer.validate(backend).structural == "failed"


@pytest.mark.unit
def test_finalize_requires_a_closed_episode_and_data(tmp_path: Path):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        writer.finalize(backend)


@pytest.mark.unit
def test_missing_pylance_names_the_install_target(monkeypatch):
    from ovannotators._internal.lerobot_common import require_pylance

    monkeypatch.setitem(sys.modules, "lance", None)
    with pytest.raises(ImportError, match="pylance"):
        require_pylance()


@pytest.mark.unit
def test_frame_schema_and_rows_share_the_column_mapping():
    pa = pytest.importorskip("pyarrow")
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "source.timestamp_ns": {"dtype": "int64", "shape": (1,), "names": None},
        "observation.images.front": {"dtype": "video", "shape": (3, 4, 3)},
    }

    def column(key):
        return key.replace(".", "__")

    schema = lance_module._frame_schema(pa, features, column)
    assert schema.names == ["action", "source__timestamp_ns"]
    row = lance_module._lance_row(
        {"action": [1.0, 2.0], "source.timestamp_ns": 7}, features, column
    )
    assert row == {"action": [1.0, 2.0], "source__timestamp_ns": 7}
    assert pa.Table.from_pylist([row], schema=schema).num_rows == 1


@pytest.mark.unit
def test_camera_keys_and_feature_helpers(tmp_path: Path):
    assert lance_module._camera_key("camera.front") == "observation.images.front"
    assert (
        lance_module._camera_timestamp_key("camera.rig/left")
        == "source.camera.rig.left.timestamp_ns"
    )
    spec = _config(tmp_path).state
    assert lance_module._feature(spec)["names"] == list(spec.feature_names)


@pytest.mark.unit
def test_config_and_feature_spec_validation(tmp_path: Path):
    spec = _config(tmp_path).state
    with pytest.raises(ValueError, match="stream_id"):
        replace(spec, stream_id="")
    with pytest.raises(ValueError, match="units"):
        replace(spec, units=("rad", "m", "s"))
    with pytest.raises(ValueError, match="fps"):
        replace(_config(tmp_path), fps=0)
    with pytest.raises(ValueError, match="repo_id"):
        replace(_config(tmp_path), repo_id="no-namespace")
    assert replace(_config(tmp_path), repo_id=None).repo_id is None
    with pytest.raises(ValueError, match="must differ"):
        replace(_config(tmp_path), action=spec)


@pytest.mark.unit
def test_episode_lifecycle_errors(tmp_path: Path, priority_profile_fixture):
    fixture = priority_profile_fixture
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path))
    with pytest.raises(RuntimeError, match="begin_episode"):
        writer.write_sample(fixture.samples[0][0], backend)
    with pytest.raises(RuntimeError, match="no active episode"):
        writer.end_episode(backend)
    writer.begin_episode(fixture.episodes[0], backend)
    with pytest.raises(ValueError, match="staging_dir must not already exist"):
        LeRobotLanceWriter(_config(tmp_path)).begin_episode(
            fixture.episodes[0], backend
        )
    with pytest.raises(RuntimeError, match="end the active episode"):
        writer.begin_episode(fixture.episodes[1], backend)
    with pytest.raises(ValueError, match="at least one sample"):
        writer.end_episode(backend)
    with pytest.raises(RuntimeError, match="before finalize"):
        writer.finalize(backend)


@pytest.mark.unit
def test_begin_episode_rejects_dirty_outputs_and_schema_drift(
    tmp_path: Path, priority_profile_fixture
):
    fixture = priority_profile_fixture
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    backend.write("stale.txt", b"x")
    with pytest.raises(ValueError, match="must be empty"):
        LeRobotLanceWriter(_config(tmp_path)).begin_episode(
            fixture.episodes[0], backend
        )
    backend = DiskBackend(DiskBackendConfig(tmp_path / "clean"))
    (tmp_path / "stage").mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        LeRobotLanceWriter(_config(tmp_path)).begin_episode(
            fixture.episodes[0], backend
        )
    (tmp_path / "stage").rmdir()

    writer = LeRobotLanceWriter(_config(tmp_path), video_encoder=_synthetic_mp4_encoder)
    _ingest(writer, backend, fixture, episodes={"episode-000"})
    with pytest.raises(ValueError, match="duplicate episode_id"):
        writer.begin_episode(fixture.episodes[0], backend)
    with pytest.raises(ValueError, match="must remain constant"):
        writer.begin_episode(replace(fixture.episodes[1], embodiment="other"), backend)
    narrow = tuple(
        replace(stream, nominal_rate_hz=10.0)
        if stream.stream_id == "camera.wrist"
        else stream
        for stream in fixture.episodes[1].streams
    )
    with pytest.raises(ValueError, match="stream schema changed"):
        writer.begin_episode(replace(fixture.episodes[1], streams=narrow), backend)
    relabeled = tuple(
        replace(stream, units=("deg",) * len(stream.units))
        if stream.stream_id == "action"
        else stream
        for stream in fixture.episodes[1].streams
    )
    with pytest.raises(ValueError, match="stream schema changed"):
        writer.begin_episode(replace(fixture.episodes[1], streams=relabeled), backend)
    optional_camera = tuple(
        replace(stream, required=False)
        if stream.stream_id == "camera.wrist"
        else stream
        for stream in fixture.episodes[1].streams
    )
    with pytest.raises(ValueError, match="required explicit nominal rate"):
        writer.begin_episode(
            replace(fixture.episodes[1], streams=optional_camera), backend
        )
    missing = tuple(
        stream for stream in fixture.episodes[1].streams if stream.stream_id != "action"
    )
    with pytest.raises(ValueError, match="missing required streams"):
        writer.begin_episode(replace(fixture.episodes[1], streams=missing), backend)


@pytest.mark.unit
def test_begin_episode_accepts_equivalent_spellings_and_unread_stream_changes(
    tmp_path: Path, priority_profile_fixture
):
    fixture = priority_profile_fixture
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path), video_encoder=_synthetic_mp4_encoder)
    _ingest(writer, backend, fixture, episodes={"episode-000"})
    respelled = tuple(
        replace(stream, dtype="|u1", shape=list(stream.shape))
        if stream.stream_type == "rgb"
        else replace(stream, shape=list(stream.shape), units=list(stream.units))
        if stream.stream_id == "observation.state"
        else replace(stream, units="mm")
        if stream.stream_id == "camera.front.depth"
        else stream
        for stream in fixture.episodes[1].streams
        if stream.stream_id != "camera.front.segmentation"
    )
    writer.begin_episode(replace(fixture.episodes[1], streams=respelled), backend)


@pytest.mark.unit
def test_write_sample_rejects_bad_timing_and_arrays(
    tmp_path: Path, priority_profile_fixture
):
    fixture = priority_profile_fixture
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path))
    writer.begin_episode(fixture.episodes[0], backend)
    first, second, _ = fixture.samples[0]
    writer.write_sample(first, backend)
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.write_sample(first, backend)
    late = replace(_mixed_rate(second), timestamp_ns=second.timestamp_ns + 500_000)
    with pytest.raises(ValueError, match="off the CFR timeline"):
        writer.write_sample(late, backend)
    other = replace(_mixed_rate(second), episode_id="episode-001")
    with pytest.raises(ValueError, match="does not match active episode"):
        writer.write_sample(other, backend)
    bad_frame = dict(_mixed_rate(second).observations)
    bad_frame["camera.front"] = replace(
        bad_frame["camera.front"], data=np.zeros((2, 2, 3), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="incompatible array"):
        writer.write_sample(
            replace(_mixed_rate(second), observations=bad_frame), backend
        )
    wrong_units = dict(_mixed_rate(second).observations)
    wrong_units["observation.state"] = replace(
        wrong_units["observation.state"], units=("deg", "m")
    )
    with pytest.raises(ValueError, match="semantics do not match"):
        writer.write_sample(
            replace(_mixed_rate(second), observations=wrong_units), backend
        )
    wrong_dtype = dict(_mixed_rate(second).observations)
    wrong_dtype["observation.state"] = replace(
        wrong_dtype["observation.state"],
        data=np.zeros(2, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="schema does not match"):
        writer.write_sample(
            replace(_mixed_rate(second), observations=wrong_dtype), backend
        )


@pytest.mark.unit
def test_camera_without_a_frame_inside_its_cadence_is_rejected(
    tmp_path: Path, priority_profile_fixture
):
    fixture = priority_profile_fixture
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path))
    writer.begin_episode(fixture.episodes[0], backend)
    writer.write_sample(_mixed_rate(fixture.samples[0][0]), backend)
    writer._active_camera_times["camera.wrist"][:] = [0]
    with pytest.raises(ValueError, match="no source frame within its declared cadence"):
        writer.end_episode(backend)


class _FakeDataset:
    """Stand-in loader that records whether a sample was actually decoded."""

    decoded = False
    length = 3

    def __init__(self, root, return_uint8):
        self.root = root

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        type(self).decoded = True
        return {}


@pytest.mark.unit
def test_validate_canonical_passes_only_after_a_sample_is_decoded(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(lance_module, "_canonical_preflight", lambda root: [])
    monkeypatch.setattr(
        lance_module,
        "run_offline_loader",
        lambda script, *args: lance_module._load_official(*args),
    )
    fake = types.ModuleType("lerobot_lancedb")
    fake.LanceDBDataset = _FakeDataset
    monkeypatch.setitem(sys.modules, "lerobot_lancedb", fake)
    monkeypatch.setattr(_FakeDataset, "decoded", False)
    pinned, candidate = lance_module.LEROBOT_LANCEDB_VERSION, "99.0.0"
    installed = {"lerobot-lancedb": pinned}
    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: installed[name], raising=True
    )

    result = lance_module._validate_canonical(tmp_path)
    assert result.canonical_loader == "passed"
    assert result.evidence == {"loader": f"lerobot-lancedb=={pinned}"}
    assert _FakeDataset.decoded

    # A drift probe names the candidate; the evidence records what ran.
    installed["lerobot-lancedb"] = candidate
    mismatch = lance_module._validate_canonical(tmp_path)
    assert mismatch.canonical_loader == "failed"
    assert (
        f"expected lerobot-lancedb=={pinned}, found {candidate}" in mismatch.messages[0]
    )
    probe = lance_module._validate_canonical(tmp_path, expected_version=candidate)
    assert probe.evidence == {"loader": f"lerobot-lancedb=={candidate}"}
    installed["lerobot-lancedb"] = pinned

    monkeypatch.setattr(_FakeDataset, "length", 0)
    assert lance_module._validate_canonical(tmp_path).canonical_loader == "failed"

    def broken(self, index):
        raise OSError("cannot decode")

    monkeypatch.setattr(_FakeDataset, "length", 3)
    monkeypatch.setattr(_FakeDataset, "__getitem__", broken)
    failed = lance_module._validate_canonical(tmp_path)
    assert failed.canonical_loader == "failed"
    assert "cannot decode" in failed.messages[0]


@pytest.mark.unit
def test_missing_signal_rejected_before_episode_commit(
    tmp_path: Path, priority_profile_fixture
):
    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path))
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    sample = priority_profile_fixture.samples[0][0]
    with pytest.raises(ValueError, match="missing required signals"):
        writer.write_sample(replace(sample, policy_signals={}), backend)


@pytest.mark.unit
def test_registry_and_direct_import_agree():
    assert ProfileWriterRegistry.get("lerobot_lance") is LeRobotLanceWriter


@pytest.mark.unit
def test_failed_episode_encode_keeps_the_episode_active_for_retry(
    tmp_path: Path, priority_profile_fixture
):
    calls = {"count": 0}

    def flaky(frames, fps, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("NVENC session lost")
        return _synthetic_mp4_encoder(frames, fps, **kwargs)

    backend = DiskBackend(DiskBackendConfig(tmp_path / "output"))
    writer = LeRobotLanceWriter(_config(tmp_path), video_encoder=flaky)
    writer.begin_episode(priority_profile_fixture.episodes[0], backend)
    for sample in priority_profile_fixture.samples[0]:
        writer.write_sample(_mixed_rate(sample), backend)
    with pytest.raises(RuntimeError, match="NVENC session lost"):
        writer.end_episode(backend)
    assert writer._episodes == [] and writer._tasks == {}

    writer.end_episode(backend)
    assert len(writer._episodes) == 1
    assert [row["index"] for row in writer._episodes[0]["rows"]] == [0, 1, 2]


class _FlakyDiskBackend(DiskBackend):
    """Disk backend whose first write of one path fails after earlier writes."""

    def __init__(self, config, fail_path):
        super().__init__(config)
        self.fail_path = fail_path

    def write_if_absent(self, path, data):
        if path == self.fail_path:
            self.fail_path = None
            raise OSError(f"transient failure writing {path}")
        return super().write_if_absent(path, data)


@pytest.mark.integration
@pytest.mark.cpu_only
@pytest.mark.parametrize("fail_path", ["meta/info.json", "_ovannotators/_SUCCESS"])
def test_native_lance_finalize_retries_a_partial_upload(
    tmp_path: Path, priority_profile_fixture, fail_path
):
    lance = pytest.importorskip("lance", reason="requires pylance")
    output = tmp_path / "output"
    backend = _FlakyDiskBackend(DiskBackendConfig(output), fail_path)
    writer = LeRobotLanceWriter(_config(tmp_path), video_encoder=_synthetic_mp4_encoder)
    _ingest(writer, backend, priority_profile_fixture)

    with pytest.raises(OSError, match="transient failure"):
        writer.finalize(backend)
    assert backend.list()  # some files were already published
    assert not backend.exists("_ovannotators/_SUCCESS")
    assert "videos.lance" in _staged_entries(writer)

    # The retry republishes the built tree, so no episode may be added now.
    late = replace(
        priority_profile_fixture.episodes[0], episode_id="episode-after-failure"
    )
    with pytest.raises(RuntimeError, match="after finalization starts"):
        writer.begin_episode(late, backend)

    writer.finalize(backend)
    assert backend.exists("_ovannotators/_SUCCESS")
    assert writer.validate(backend).structural == "passed"
    record = json.loads((output / "_ovannotators/writer_record.json").read_text())
    assert "episode-after-failure" not in record["input_identifiers"]
    videos = lance.dataset(str(output / "videos.lance"))
    assert videos.version == 1
    assert videos.count_rows() == 6
    assert lance.dataset(str(output / "frames.lance")).count_rows() == 9
    checksums = json.loads(
        (output / "_ovannotators/payload_checksums.json").read_text()
    )
    published = {
        path for path in backend.list() if not path.startswith("_ovannotators/")
    }
    assert set(checksums["files"]) == published
    assert not writer.staging_root.exists()


@pytest.fixture
def canonical_metadata(tmp_path: Path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    meta = tmp_path / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_tasks": 1, "total_episodes": 1}))
    (meta / "stats.json").write_text("{}")
    pq.write_table(pa.table({"task_index": [0]}), meta / "tasks.parquet")
    pq.write_table(
        pa.table({"episode_index": [0]}), meta / "episodes/chunk-000/file-000.parquet"
    )
    return tmp_path


@pytest.mark.unit
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
@pytest.mark.parametrize(
    "relative",
    [
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
    ],
)
def test_canonical_preflight_rejects_incomplete_metadata(
    canonical_metadata: Path, monkeypatch, relative, damage
):
    root = canonical_metadata
    assert lance_module._canonical_preflight(root) == []
    path = root / relative
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"broken")

    def forbidden(*args):
        raise AssertionError("loader must not open incomplete metadata")

    monkeypatch.setattr(lance_module, "run_offline_loader", forbidden)
    result = lance_module._validate_canonical(root)
    assert result.structural == "failed"
    assert result.canonical_loader == "not_run"
    assert "meta" in result.messages[0]


@pytest.mark.unit
def test_offline_loader_runs_in_a_child_with_hub_access_disabled(
    canonical_metadata: Path, monkeypatch
):
    root = canonical_metadata
    marker = lance_module._LOADER_RESULT_MARKER
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    monkeypatch.setattr(
        lance_module,
        "_OFFLINE_LOADER_SCRIPT",
        "import json, os, sys\n"
        "assert sys.flags.safe_path\n"
        "print('loader chatter')\n"
        f"print({marker!r} + json.dumps({{'version': ':'.join(\n"
        "    [os.environ['HF_HUB_OFFLINE'], os.environ['HF_DATASETS_OFFLINE'],\n"
        "     *sys.argv[1:3]])}))\n",
    )
    version = lance_module.LEROBOT_LANCEDB_VERSION
    result = lance_module._validate_canonical(root)
    assert result.canonical_loader == "passed", result.messages
    assert result.evidence == {"loader": f"lerobot-lancedb==1:1:{root}:{version}"}


@pytest.mark.unit
def test_offline_loader_child_that_hangs_is_reported_as_an_error():
    from ovannotators._internal.lerobot_common import run_offline_loader

    outcome = run_offline_loader("import time; time.sleep(30)", timeout_s=0.5)
    assert outcome == {"error": "loader process timed out after 0.5 s"}
