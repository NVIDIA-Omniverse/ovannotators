# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shared envelope fixture for exact output-profile tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ovannotators.profile_writers import (
    EpisodeEnvelope,
    NamedTensor,
    SampleEnvelope,
    StreamDescriptor,
)


@dataclass(frozen=True)
class PriorityProfileFixture:
    """Three small episodes that exercise profile boundaries and sharding."""

    episodes: tuple[EpisodeEnvelope, ...]
    samples: tuple[tuple[SampleEnvelope, ...], ...]
    data_shard_size: int = 3
    video_shard_size: int = 2


def make_priority_profile_fixture() -> PriorityProfileFixture:
    """Build the shared two-camera, mixed-rate, multi-task fixture."""
    camera_geometry = {
        "camera_model": "opencv_pinhole",
        "image_size_px": [4, 3],
        "intrinsics": {"fx_px": 4.0, "fy_px": 4.0, "cx_px": 1.5, "cy_px": 1.0},
        "reference_frame_id": "robot_base",
        "T_reference_from_sensor": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    }
    streams = (
        StreamDescriptor(
            "camera.front",
            "rgb",
            "sim_clock",
            10.0,
            "uint8",
            (3, 4, 3),
            geometry=camera_geometry,
            encoding="rgb8",
        ),
        StreamDescriptor(
            "camera.wrist",
            "rgb",
            "sim_clock",
            5.0,
            "uint8",
            (3, 4, 3),
            geometry={**camera_geometry, "reference_frame_id": "wrist"},
            encoding="rgb8",
        ),
        StreamDescriptor(
            "camera.front.depth",
            "depth",
            "sim_clock",
            10.0,
            "float32",
            (3, 4),
            units="m",
            derived_from_stream_id="camera.front",
        ),
        StreamDescriptor(
            "camera.front.segmentation",
            "segmentation",
            "sim_clock",
            10.0,
            "uint16",
            (3, 4),
            derived_from_stream_id="camera.front",
            lineage={"taxonomy": "fixture-v1"},
        ),
        StreamDescriptor(
            "observation.state",
            "state",
            "sim_clock",
            10.0,
            "float32",
            (2,),
            units=("rad", "m"),
        ),
        StreamDescriptor(
            "action",
            "action",
            "sim_clock",
            10.0,
            "float32",
            (2,),
            units=("rad", "m"),
        ),
    )
    episode_specs = (
        ("episode-000", "pick_cube"),
        ("episode-001", "place_cube"),
        ("episode-002", "pick_cube"),
    )
    episodes = tuple(
        EpisodeEnvelope(
            dataset_id="priority-profile-fixture",
            sequence_id=f"sequence-{index // 2}",
            episode_id=episode_id,
            task=task,
            embodiment="fixture_arm",
            source_clock="sim_clock",
            source_origin_ns=5_000_000_000 + index * 1_000_000_000,
            nominal_rate_hz=10.0,
            streams=streams,
            provenance={"producer": "ovannotators-test", "source_version": "1"},
        )
        for index, (episode_id, task) in enumerate(episode_specs)
    )

    samples_by_episode: list[tuple[SampleEnvelope, ...]] = []
    for episode_index, episode in enumerate(episodes):
        episode_samples = []
        for frame_index in range(3):
            sentinel = episode_index * 100 + frame_index
            front = np.full((3, 4, 3), sentinel + 1, dtype=np.uint8)
            wrist = np.full((3, 4, 3), sentinel + 51, dtype=np.uint8)
            observations = {
                "camera.front": NamedTensor(front, representation="rgb8"),
                "camera.front.depth": NamedTensor(
                    np.full((3, 4), sentinel + 0.25, dtype=np.float32),
                    units=("m",),
                    representation="distance_to_camera",
                    coordinate_frame="camera.front",
                ),
                "camera.front.segmentation": NamedTensor(
                    np.full((3, 4), sentinel + 7, dtype=np.uint16),
                    representation="semantic_id",
                ),
                "observation.state": NamedTensor(
                    np.asarray([sentinel + 0.1, sentinel + 0.2], dtype=np.float32),
                    feature_names=("joint.position", "gripper.position"),
                    units=("rad", "m"),
                    representation="absolute",
                    coordinate_frame="robot_base",
                ),
            }
            missing_data = {}
            if episode_index == 1 and frame_index == 1:
                missing_data["camera.wrist"] = "dropped_at_source"
            else:
                observations["camera.wrist"] = NamedTensor(
                    wrist, representation="rgb8", coordinate_frame="wrist"
                )
            episode_samples.append(
                SampleEnvelope(
                    episode_id=episode.episode_id,
                    sample_id=f"{episode.episode_id}-sample-{frame_index:03d}",
                    frame_id=frame_index,
                    timestamp_ns=episode.source_origin_ns + frame_index * 100_000_003,
                    observations=observations,
                    policy_signals={
                        "action": NamedTensor(
                            np.asarray(
                                [sentinel + 10.1, sentinel + 10.2], dtype=np.float32
                            ),
                            feature_names=("joint.target", "gripper.target"),
                            units=("rad", "m"),
                            representation="absolute",
                            coordinate_frame="robot_base",
                        )
                    },
                    missing_data=missing_data,
                )
            )
        samples_by_episode.append(tuple(episode_samples))
    return PriorityProfileFixture(episodes, tuple(samples_by_episode))
