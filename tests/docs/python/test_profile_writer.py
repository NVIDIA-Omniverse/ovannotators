# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tested snippets for the episode-aware profile writer lifecycle."""

from __future__ import annotations

import json

from ovannotators.profile_writers import (
    ConformanceResult,
    EpisodeEnvelope,
    LossReport,
    ProfileMetadata,
    ProfileWriterRegistry,
    SampleEnvelope,
    WriterRecord,
    write_profile_artifacts,
)


# [snippet:doc-profile-writer-class]
@ProfileWriterRegistry.register
class JsonLinesProfileWriter:
    """Minimal episode-aware writer with common completion evidence."""

    name = "doc_jsonl_profile"

    def __init__(self) -> None:
        self._episode: EpisodeEnvelope | None = None
        self._payload_paths: list[str] = []
        self._finalized = False

    def begin_episode(self, episode: EpisodeEnvelope, backend) -> None:
        if self._episode is not None:
            raise ValueError("end the active episode before beginning another")
        self._episode = episode

    def write_sample(self, sample: SampleEnvelope, backend) -> None:
        if self._episode is None or sample.episode_id != self._episode.episode_id:
            raise ValueError("sample does not belong to the active episode")
        path = f"data/{sample.sample_id}.json"
        backend.write(
            path,
            json.dumps({"timestamp_ns": sample.timestamp_ns}).encode("utf-8"),
        )
        self._payload_paths.append(path)

    def end_episode(self, backend) -> None:
        if self._episode is None:
            raise ValueError("no active episode")
        self._episode = None

    def validate(self, backend) -> ConformanceResult:
        complete = self._episode is None and all(
            backend.exists(path) for path in self._payload_paths
        )
        status = "passed" if complete else "failed"
        return ConformanceResult(status, status)

    def finalize(self, backend) -> None:
        if self._finalized:
            return
        result = self.validate(backend)
        write_profile_artifacts(
            backend,
            profile=ProfileMetadata("doc_jsonl", "example", "1", "1", ()),
            writer_record=WriterRecord(
                "example", self.name, "1", (), tuple(self._payload_paths)
            ),
            loss_report=LossReport(consumed=("timestamp_ns",)),
            conformance=result,
            payload_paths=tuple(self._payload_paths),
        )
        self._finalized = True


# [/snippet:doc-profile-writer-class]


def test_profile_writer_lifecycle(disk_backend, priority_profile_fixture):
    episode = priority_profile_fixture.episodes[0]
    sample = priority_profile_fixture.samples[0][0]

    # [snippet:doc-profile-writer-lifecycle]
    writer = ProfileWriterRegistry.create("doc_jsonl_profile")
    writer.begin_episode(episode, disk_backend)
    writer.write_sample(sample, disk_backend)
    writer.end_episode(disk_backend)
    writer.finalize(disk_backend)
    writer.finalize(disk_backend)  # idempotent

    assert disk_backend.exists("_ovannotators/_SUCCESS")
    result = writer.validate(disk_backend)
    assert result.serialization == "passed"
    assert result.structural == "passed"
    # Canonical-loader and consumer acceptance are tracked separately.
    assert result.canonical_loader == "not_run"
    # [/snippet:doc-profile-writer-lifecycle]
