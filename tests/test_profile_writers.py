# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared episode-aware profile contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ovannotators import (
    ConformanceResult,
    EpisodeEnvelope,
    LossReport,
    NamedTensor,
    ProfileMetadata,
    ProfileWriterRegistry,
    SampleEnvelope,
    StreamDescriptor,
    WriterRecord,
)
from ovannotators.exceptions import ConfigurationError
from ovannotators.profile_writers import canonical_json_bytes, write_profile_artifacts

from tests.profile_fixtures import make_priority_profile_fixture


def test_priority_fixture_covers_shared_acceptance_cases():
    fixture = make_priority_profile_fixture()

    assert len(fixture.episodes) == 3
    assert {episode.task for episode in fixture.episodes} == {"pick_cube", "place_cube"}
    assert fixture.data_shard_size == 3
    assert fixture.video_shard_size == 2
    first_streams = {stream.stream_id: stream for stream in fixture.episodes[0].streams}
    assert first_streams["camera.front"].nominal_rate_hz == 10.0
    assert first_streams["camera.wrist"].nominal_rate_hz == 5.0
    assert first_streams["camera.front.depth"].derived_from_stream_id == "camera.front"
    dropped = fixture.samples[1][1]
    assert dropped.missing_data == {"camera.wrist": "dropped_at_source"}
    assert "camera.wrist" not in dropped.observations
    assert dropped.timestamp_ns == 6_100_000_003
    np.testing.assert_allclose(
        fixture.samples[2][2].policy_signals["action"].data, [212.1, 212.2]
    )


def test_envelopes_reject_invalid_identity_schema_and_time():
    stream = StreamDescriptor("cam", "rgb", "clock", 30.0, "uint8", (2, 2, 3))
    with pytest.raises(ValueError, match="stream IDs must be unique"):
        EpisodeEnvelope("d", "s", "e", "t", "r", "clock", 0, 30.0, (stream, stream))
    with pytest.raises(ValueError, match="frame_id"):
        SampleEnvelope("e", "s", -1, 0, {})
    with pytest.raises(TypeError, match="nanoseconds"):
        SampleEnvelope("e", "s", 0, 1.5, {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="units"):
        NamedTensor(np.zeros(2), ("a", "b"), ("m", "s", "rad"))


def test_profile_registry_supports_decorator_create_and_errors():
    @ProfileWriterRegistry.register
    class ExampleProfileWriter:
        name = "test_profile_registry"

        def __init__(self, value=1):
            self.value = value

    assert ProfileWriterRegistry.get("test_profile_registry") is ExampleProfileWriter
    assert ProfileWriterRegistry.create("test_profile_registry", 7).value == 7
    assert "test_profile_registry" in ProfileWriterRegistry.list_names()
    with pytest.raises(ConfigurationError, match="Available"):
        ProfileWriterRegistry.get("not-a-profile")


def test_profile_registry_rejects_duplicate_name():
    class First:
        name = "test_duplicate_profile"

    class Second:
        name = "test_duplicate_profile"

    ProfileWriterRegistry.register(First)
    with pytest.raises(ValueError, match="already registered"):
        ProfileWriterRegistry.register(Second)


def test_common_artifacts_checksum_payload(disk_backend):
    disk_backend.write("payload/data.bin", b"deterministic-payload")
    conformance = ConformanceResult("passed", "passed")
    checksums = write_profile_artifacts(
        disk_backend,
        profile=ProfileMetadata(
            "fixture", "test", "1", "1", ("rgb", "state", "action")
        ),
        writer_record=WriterRecord(
            "dataset", "fixture_writer", "1", ("source",), ("payload/data.bin",)
        ),
        loss_report=LossReport(consumed=("observations", "policy_signals")),
        conformance=conformance,
        payload_paths=("payload/data.bin",),
    )

    assert checksums == {
        "payload/data.bin": "ba93a56a8fe9befface03cc6b575aebef8cb8cde4d259452aeea11cab154fb74"
    }
    assert set(disk_backend.list("_ovannotators")) == {
        "_ovannotators/_SUCCESS",
        "_ovannotators/conformance.json",
        "_ovannotators/loss_report.json",
        "_ovannotators/payload_checksums.json",
        "_ovannotators/profile.json",
        "_ovannotators/writer_record.json",
    }
    success = json.loads(disk_backend.read("_ovannotators/_SUCCESS"))
    assert success["meaning"] == "serialization_and_structural_validation_completed"
    assert success["canonical_loader_acceptance"] == "not_run"
    assert not conformance.fully_conformant


def test_common_artifacts_write_success_last(disk_backend, monkeypatch):
    disk_backend.write("payload/data.bin", b"deterministic-payload")
    written_paths: list[str] = []
    original_write = disk_backend.write

    def record_write(path: str, data: bytes) -> None:
        written_paths.append(path)
        original_write(path, data)

    monkeypatch.setattr(disk_backend, "write", record_write)
    write_profile_artifacts(
        disk_backend,
        profile=ProfileMetadata("fixture", "test", "1", "1", ()),
        writer_record=WriterRecord("dataset", "fixture_writer", "1", (), ()),
        loss_report=LossReport(),
        conformance=ConformanceResult("passed", "passed"),
        payload_paths=("payload/data.bin",),
    )

    assert written_paths[-1] == "_ovannotators/_SUCCESS"


def test_common_artifacts_never_claim_success_after_structural_failure(disk_backend):
    disk_backend.write("payload.bin", b"x")
    with pytest.raises(ValueError, match="requires passed"):
        write_profile_artifacts(
            disk_backend,
            profile=ProfileMetadata("fixture", "test", "1", "1", ()),
            writer_record=WriterRecord("d", "w", "1", (), ("payload.bin",)),
            loss_report=LossReport(),
            conformance=ConformanceResult("passed", "failed"),
            payload_paths=("payload.bin",),
        )
    assert disk_backend.list("_ovannotators") == []
    assert not disk_backend.exists("_ovannotators/_SUCCESS")


def test_common_artifacts_reject_reserved_missing_and_duplicate_payloads(disk_backend):
    common = {
        "profile": ProfileMetadata("fixture", "test", "1", "1", ()),
        "writer_record": WriterRecord("d", "w", "1", (), ()),
        "loss_report": LossReport(),
        "conformance": ConformanceResult("passed", "passed"),
    }
    with pytest.raises(ValueError, match="reserved"):
        write_profile_artifacts(
            disk_backend,
            payload_paths=("_ovannotators/profile.json",),
            **common,
        )
    with pytest.raises(ValueError, match="missing"):
        write_profile_artifacts(disk_backend, payload_paths=("missing",), **common)
    with pytest.raises(ValueError, match="duplicates"):
        write_profile_artifacts(disk_backend, payload_paths=("a", "a"), **common)


def test_canonical_json_is_order_independent_and_numpy_aware():
    left = canonical_json_bytes({"z": np.int64(2), "a": np.asarray([1, 2])})
    right = canonical_json_bytes({"a": np.asarray([1, 2]), "z": 2})
    assert left == right == b'{"a":[1,2],"z":2}\n'


@pytest.mark.unit
def test_canonical_json_sorts_unordered_collections():
    expected = b'{"values":["alpha","beta","delta","gamma"]}\n'

    assert (
        canonical_json_bytes({"values": {"gamma", "alpha", "delta", "beta"}})
        == expected
    )
    assert (
        canonical_json_bytes({"values": frozenset({"beta", "delta", "alpha", "gamma"})})
        == expected
    )
