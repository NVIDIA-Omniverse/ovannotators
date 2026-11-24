# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode native LeRobot output with the pinned upstream loaders.

Runs only in the isolated conformance environment
(``tests/requirements/lerobot-lance-conformance.txt``), where ``lerobot``,
``lerobot-lancedb``, and PyAV are installed. PyAV's software encoders stand in
for NVENC; the clips still go through ``ovannotators``' own MP4 muxer,
validation, concatenation, and byte index.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
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
    SampleEnvelope,
    StreamDescriptor,
    validate_lerobot_v3_canonical,
)
from ovannotators._internal.mp4_mux import (
    mux_av1_obu,
    mux_h264_annexb,
    parse_mp4_video_index,
    validate_mp4_clip,
)
from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.profile_writers.lerobot_v3 import LEROBOT_VERSION

pytestmark = [
    pytest.mark.conformance,
    pytest.mark.skipif(sys.version_info < (3, 12), reason="loaders need 3.12"),
]
av = pytest.importorskip("av", reason="requires the conformance environment")
pytest.importorskip("lerobot", reason="requires the conformance environment")

FPS = 10
REPO_ID = "conformance/native"
EPISODE_LENGTHS = (7, 1, 5)
TASKS = ("pick", "place", "pick")
FRONT, WRIST = "camera.front", "camera.wrist"
SHAPES = {FRONT: (96, 128, 3), WRIST: (64, 96, 3)}
WRIST_STRIDE = 2  # 5 Hz camera on the 10 Hz timeline
GRID = 8  # 8 bits: 2 for episode, 5 for frame, 1 for camera


def _frame(stream_id: str, episode: int, frame: int) -> np.ndarray:
    height, width, _ = SHAPES[stream_id]
    word = (episode << 6) | (frame << 1) | (stream_id == WRIST)
    image = np.full(SHAPES[stream_id], 16, dtype=np.uint8)
    cell = width // GRID
    for bit in range(GRID):
        if word >> (GRID - 1 - bit) & 1:
            image[:, bit * cell : (bit + 1) * cell] = 235
    return image


def _decode(image: np.ndarray) -> int:
    """Read the bit grid from an HWC uint8 or CHW float frame."""
    array = np.asarray(image)
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = array * 255.0
    luma = array.astype(np.float32).mean(axis=2)
    cell = luma.shape[1] // GRID
    word = 0
    for bit in range(GRID):
        block = luma[:, bit * cell + cell // 4 : (bit + 1) * cell - cell // 4]
        word = (word << 1) | int(block.mean() >= 128)
    return word


def _state(episode: int, frame: int) -> np.ndarray:
    return np.asarray([episode * 100 + frame, -frame], dtype=np.float32)


def _software_encoder(frames, fps, *, codec, lossless, preset, gop):
    """Encode with PyAV (libx264 or libsvtav1) and mux with ovannotators."""
    del lossless, preset
    height, width = frames[0].shape[:2]
    name = {"h264": "libx264", "av1": "libsvtav1"}[codec]
    context = av.CodecContext.create(name, "w")
    context.width, context.height, context.pix_fmt = width, height, "yuv420p"
    context.time_base = Fraction(1, fps)
    context.framerate = Fraction(fps, 1)
    context.gop_size = gop or 2
    context.max_b_frames = 0
    context.options = (
        {"preset": "ultrafast", "bframes": "0", "scenecut": "0"}
        if codec == "h264"
        else {"preset": "12", "svtav1-params": "pred-struct=1:scd=0"}
    )
    packets = []
    for index, frame in enumerate(frames):
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = index
        packets.extend(context.encode(video_frame))
    packets.extend(context.encode(None))
    stream = b"".join(bytes(packet) for packet in packets)
    mux = mux_h264_annexb if codec == "h264" else mux_av1_obu
    return mux(stream, fps, width, height)


_STATE = dict(
    stream_id="observation.state",
    feature_names=("joint", "gripper"),
    units=("rad", "m"),
    representation="absolute",
    shape=(2,),
)
_ACTION = {**_STATE, "stream_id": "action"}


def _tensor(spec, value) -> NamedTensor:
    return NamedTensor(value, spec.feature_names, spec.units, spec.representation, None)


def _write(writer, state_spec, action_spec, output: Path) -> None:
    backend = DiskBackend(DiskBackendConfig(output))
    for episode, length in enumerate(EPISODE_LENGTHS):
        origin = 1_000_000_000 * (episode + 1)
        envelope = EpisodeEnvelope(
            dataset_id="conformance",
            sequence_id="sequence",
            episode_id=f"episode-{episode}",
            task=TASKS[episode],
            embodiment="test_arm",
            source_clock="sim",
            source_origin_ns=origin,
            nominal_rate_hz=float(FPS),
            streams=(
                StreamDescriptor(
                    FRONT, "rgb", "sim", float(FPS), "uint8", SHAPES[FRONT]
                ),
                StreamDescriptor(
                    WRIST, "rgb", "sim", FPS / WRIST_STRIDE, "uint8", SHAPES[WRIST]
                ),
                StreamDescriptor(
                    "observation.state", "state", "sim", 10.0, "float32", (2,)
                ),
                StreamDescriptor("action", "action", "sim", 10.0, "float32", (2,)),
            ),
        )
        writer.begin_episode(envelope, backend)
        for frame in range(length):
            observations = {
                FRONT: NamedTensor(_frame(FRONT, episode, frame)),
                "observation.state": _tensor(state_spec, _state(episode, frame)),
            }
            missing = {}
            if frame % WRIST_STRIDE == 0:
                observations[WRIST] = NamedTensor(_frame(WRIST, episode, frame))
            else:
                missing[WRIST] = "not_sampled_at_source_rate"
            writer.write_sample(
                SampleEnvelope(
                    episode_id=envelope.episode_id,
                    sample_id=f"{episode}-{frame}",
                    frame_id=frame,
                    timestamp_ns=origin + frame * 1_000_000_000 // FPS,
                    observations=observations,
                    policy_signals={
                        "action": _tensor(action_spec, -_state(episode, frame))
                    },
                    missing_data=missing,
                ),
                backend,
            )
        writer.end_episode(backend)
    writer.finalize(backend)


def _expected():
    for episode, length in enumerate(EPISODE_LENGTHS):
        for frame in range(length):
            yield episode, frame


def _check_item(item, episode: int, frame: int) -> None:
    held = frame - frame % WRIST_STRIDE
    assert int(item["episode_index"]) == episode
    assert int(item["frame_index"]) == frame
    assert float(item["timestamp"]) == pytest.approx(frame / FPS)
    assert (
        np.asarray(item["observation.state"]).tolist()
        == _state(episode, frame).tolist()
    )
    np.testing.assert_array_equal(np.asarray(item["action"]), -_state(episode, frame))
    assert _decode(item["observation.images.front"]) == (episode << 6) | (frame << 1)
    assert _decode(item["observation.images.wrist"]) == (episode << 6) | (held << 1) | 1


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_lerobot_dataset_decodes_every_native_v3_frame(tmp_path: Path, codec):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    state = LeRobotV3FeatureSpec(**_STATE, dtype="f4" if codec == "h264" else "float32")
    action = LeRobotV3FeatureSpec(
        **_ACTION, dtype="<f4" if codec == "h264" else "float32"
    )
    config = LeRobotV3WriterConfig(
        fps=FPS,
        camera_stream_ids=(FRONT, WRIST),
        state=state,
        action=action,
        repo_id=REPO_ID,
        video_codec=codec,
        staging_dir=tmp_path / "stage",
    )
    output = tmp_path / "output"
    _write(
        LeRobotV3Writer(config, video_encoder=_software_encoder), state, action, output
    )

    canonical = validate_lerobot_v3_canonical(output, repo_id=REPO_ID)
    assert canonical.canonical_loader == "passed", canonical.messages
    assert canonical.evidence == {"loader": f"lerobot=={LEROBOT_VERSION}"}

    dataset = LeRobotDataset(REPO_ID, root=output, video_backend="pyav")
    expected = list(_expected())
    assert len(dataset) == len(expected)
    for index, (episode, frame) in enumerate(expected):
        _check_item(dataset[index], episode, frame)


def test_canonical_loader_never_completes_a_tree_from_the_hub(
    tmp_path: Path, monkeypatch
):
    """A shard the local checks miss makes the offline loader fail, not download."""
    from ovannotators.profile_writers import lerobot_v3

    state, action = LeRobotV3FeatureSpec(**_STATE), LeRobotV3FeatureSpec(**_ACTION)
    config = LeRobotV3WriterConfig(
        fps=FPS,
        camera_stream_ids=(FRONT, WRIST),
        state=state,
        action=action,
        repo_id=REPO_ID,
        staging_dir=tmp_path / "stage",
    )
    output = tmp_path / "output"
    _write(
        LeRobotV3Writer(config, video_encoder=_software_encoder), state, action, output
    )
    shard = "videos/observation.images.wrist/chunk-000/file-000.mp4"
    (output / shard).unlink()
    manifest_path = output / "_ovannotators/payload_checksums.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"][shard]
    manifest_path.write_text(json.dumps(manifest))
    assert any(
        "referenced shard is missing" in message
        for message in validate_lerobot_v3_canonical(output, REPO_ID).messages
    )

    # Bypass the local checks to exercise the offline loader on its own.
    monkeypatch.setattr(lerobot_v3, "_canonical_preflight", lambda root: [])
    result = validate_lerobot_v3_canonical(output, REPO_ID)

    assert result.canonical_loader == "failed", result.messages
    assert not (output / shard).exists()


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_lancedb_dataset_decodes_every_native_lance_frame(tmp_path: Path, codec):
    pytest.importorskip("lerobot_lancedb", reason="requires lerobot-lancedb")
    from lerobot_lancedb import LanceDBDataset

    state, action = (
        LeRobotLanceFeatureSpec(**_STATE),
        LeRobotLanceFeatureSpec(**_ACTION),
    )
    config = LeRobotLanceWriterConfig(
        fps=FPS,
        camera_stream_ids=(FRONT, WRIST),
        state=state,
        action=action,
        repo_id=REPO_ID,
        video_codec=codec,
        staging_dir=tmp_path / "stage",
        canonical_loader_check="required",
    )
    output = tmp_path / "output"
    writer = LeRobotLanceWriter(config, video_encoder=_software_encoder)
    _write(writer, state, action, output)

    dataset = LanceDBDataset(root=output, return_uint8=True)
    expected = list(_expected())
    assert len(dataset) == len(expected)
    for index, (episode, frame) in enumerate(expected):
        _check_item(dataset[index], episode, frame)


def test_software_encoder_matches_the_writer_contract():
    clip = _software_encoder(
        [_frame(FRONT, 0, i) for i in range(5)],
        FPS,
        codec="h264",
        lossless=False,
        preset=None,
        gop=2,
    )
    validate_mp4_clip(clip, frame_count=5, fps=FPS, width=128, height=96, codec="h264")
    assert parse_mp4_video_index(clip)["kf_indices"] == [0, 2, 4]


def test_lance_canonical_loader_never_completes_metadata_from_the_hub(
    tmp_path: Path, monkeypatch
):
    from ovannotators.profile_writers import lerobot_lance

    pytest.importorskip("lerobot_lancedb", reason="requires lerobot-lancedb")
    monkeypatch.chdir(tmp_path)
    state, action = (
        LeRobotLanceFeatureSpec(**_STATE),
        LeRobotLanceFeatureSpec(**_ACTION),
    )
    config = LeRobotLanceWriterConfig(
        fps=FPS,
        camera_stream_ids=(FRONT, WRIST),
        state=state,
        action=action,
        repo_id=REPO_ID,
        staging_dir=Path("local/stage"),
        keep_staging=True,
    )
    writer = LeRobotLanceWriter(config, video_encoder=_software_encoder)
    _write(writer, state, action, tmp_path / "output")
    root = writer.staging_root
    assert root == tmp_path.resolve() / "local/stage"
    intact = lerobot_lance._validate_canonical(root)
    assert intact.canonical_loader == "passed", intact.messages
    assert intact.evidence == {
        "loader": f"lerobot-lancedb=={lerobot_lance.LEROBOT_LANCEDB_VERSION}"
    }
    missing = root / "meta/info.json"
    missing.unlink()
    assert lerobot_lance._validate_canonical(root).canonical_loader == "not_run"
    monkeypatch.setattr(lerobot_lance, "_canonical_preflight", lambda root: [])
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")

    result = lerobot_lance._validate_canonical(root.relative_to(tmp_path.resolve()))

    assert result.canonical_loader == "failed", result.messages
    assert "OfflineModeIsEnabled" in result.messages[0]
    assert not missing.exists()
    print(result.messages[0])
