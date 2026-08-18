# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the CosmosTransferWriter -> Cosmos 3 video-transfer bridge.

These exercise the full hand-off: run :class:`CosmosTransferWriter` to produce a
manifest plus conditioning videos on disk, then convert that output (plus a
prompt) into a Cosmos 3 Framework ``video2video`` transfer spec and inference
command. The bridge has no GPU/PyTorch dependency, so the conversion is tested
on CPU. The end-to-end test that actually launches Cosmos 3 inference is gated
behind the ``gpu`` marker and skips unless a Cosmos Framework + checkpoint + GPU
are available.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.integrations.cosmos3_transfer import (
    CONDITIONING_TO_HINT,
    _nearest_aspect_ratio,
    _select_clip,
    build_inference_command,
    build_transfer_spec,
    load_manifest,
    output_video_path,
    prepare_transfer,
)
from ovannotators.writers.cosmos_transfer import (
    CosmosTransferWriter,
    CosmosTransferWriterConfig,
)

PROMPT = "A robotic arm wiping a white plate on a kitchen countertop, cinematic."


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class _FakeVideoEncoder:
    """CPU stand-in for the GPU video encoder; emits small valid-ish MP4 bytes."""

    def __init__(self) -> None:
        self._frames = 0

    def write_frame(self, frame: np.ndarray) -> None:
        self._frames += 1

    def close(self) -> bytes:
        # ftyp box header + filler so the on-disk file is a non-empty .mp4.
        return b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 128


def _fake_get_video_encoder(path, fps, w, h, **kwargs):
    return _FakeVideoEncoder()


def _write_writer_output(
    out_dir: Path,
    *,
    resolution: tuple[int, int] = (64, 48),
    num_frames: int = 3,
) -> Path:
    """Produce real CosmosTransferWriter output on disk and return its dir.

    ``resolution`` only sizes the synthetic render vars; the writer derives the
    output resolution from those frames.
    """
    backend = DiskBackend(DiskBackendConfig(root_dir=str(out_dir), overwrite=True))
    cfg = CosmosTransferWriterConfig(
        fps=24,
        conditioning_types=[
            "canny",
            "depth",
            "segmentation",
            "shaded_segmentation",
        ],
    )
    writer = CosmosTransferWriter(cfg)
    rng = np.random.default_rng(0)
    h, w = resolution[1], resolution[0]
    with patch(
        "ovannotators._internal.codecs.get_video_encoder",
        _fake_get_video_encoder,
    ):
        for i in range(num_frames):
            render_vars = {
                "LdrColor": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
                "DistanceToCameraSD": (rng.random((h, w)) * 10.0).astype(np.float32),
                "SemanticInstanceSegmentation": rng.integers(
                    0, 5, (h, w), dtype=np.int32
                ),
                "NormalSD": (rng.random((h, w, 3)) * 2.0 - 1.0).astype(np.float32),
            }
            writer.write_frame(i, render_vars, backend)
        writer.finalize(backend)
    return out_dir


@pytest.fixture
def writer_output(tmp_path) -> Path:
    return _write_writer_output(tmp_path / "writer_out")


# ---------------------------------------------------------------------------
# Writer output sanity
# ---------------------------------------------------------------------------


def test_writer_output_has_conditioning_videos(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    assert manifest["version"].startswith("cosmos-transfer")
    clip = manifest["clips"][0]
    for cond in ("canny", "depth", "segmentation", "shaded_segmentation"):
        rel = clip["conditioning"][cond]
        assert (manifest_dir / rel).is_file()


def test_load_manifest_accepts_file_and_dir(writer_output):
    m_dir, dir_base = load_manifest(writer_output)
    m_file, file_base = load_manifest(writer_output / "manifest.json")
    assert m_dir == m_file
    assert dir_base == file_base == writer_output


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("conditioning,hint", sorted(CONDITIONING_TO_HINT.items()))
def test_build_spec_for_each_conditioning(writer_output, conditioning, hint):
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest, manifest_dir, conditioning=conditioning, prompt=PROMPT
    )

    assert spec["model_mode"] == "video2video"
    assert spec["name"] == f"transfer_{hint}"
    assert spec["prompt"] == PROMPT
    # The hint block points at the matching writer conditioning video on disk.
    assert hint in spec
    control_path = Path(spec[hint]["control_path"])
    assert control_path.is_absolute()
    assert control_path.is_file()
    assert control_path.name.endswith(f"_{conditioning}.mp4")
    # Frame geometry / timing comes from the manifest clip.
    assert spec["num_frames"] == 3
    assert spec["fps"] == 24


def test_spec_requires_prompt_at_transfer_time(writer_output):
    """The prompt is supplied at transfer time; the manifest carries none."""
    manifest, manifest_dir = load_manifest(writer_output)
    assert "prompt" not in manifest["clips"][0]
    with pytest.raises(ValueError, match="non-empty prompt"):
        build_transfer_spec(manifest, manifest_dir, conditioning="depth")


def test_spec_prompt_sets_caption(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest, manifest_dir, conditioning="depth", prompt="override caption"
    )
    assert spec["prompt"] == "override caption"


def test_edge_hint_has_preset_threshold(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest, manifest_dir, conditioning="canny", prompt=PROMPT
    )
    assert spec["edge"]["preset_edge_threshold"] == "medium"


def test_seg_control_guidance_default(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest, manifest_dir, conditioning="segmentation", prompt=PROMPT
    )
    assert spec["control_guidance"] == 2.0
    assert spec["guidance"] == 3.0


def test_guidance_overrides(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest,
        manifest_dir,
        conditioning="depth",
        prompt=PROMPT,
        guidance=1.5,
        control_guidance=4.0,
        seed=7,
        num_steps=12,
    )
    assert spec["guidance"] == 1.5
    assert spec["control_guidance"] == 4.0
    assert spec["seed"] == 7
    assert spec["num_steps"] == 12


def test_resolution_and_aspect_mapping_720p(tmp_path):
    out = _write_writer_output(tmp_path / "hd_out", resolution=(1280, 720))
    manifest, manifest_dir = load_manifest(out)
    spec = build_transfer_spec(
        manifest, manifest_dir, conditioning="depth", prompt=PROMPT
    )
    assert spec["resolution"] == "720"
    assert spec["aspect_ratio"] == "16,9"


def test_relative_control_path(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest,
        manifest_dir,
        conditioning="depth",
        prompt=PROMPT,
        absolute_control_path=False,
    )
    assert spec["depth"]["control_path"] == "conditioning/000000_depth.mp4"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_conditioning_raises(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    with pytest.raises(ValueError, match="Unknown conditioning"):
        build_transfer_spec(manifest, manifest_dir, conditioning="normals")


def test_missing_conditioning_in_clip_raises(tmp_path):
    out = tmp_path / "canny_only"
    backend = DiskBackend(DiskBackendConfig(root_dir=str(out), overwrite=True))
    cfg = CosmosTransferWriterConfig(
        conditioning_types=["canny"],
    )
    writer = CosmosTransferWriter(cfg)
    rng = np.random.default_rng(1)
    with patch(
        "ovannotators._internal.codecs.get_video_encoder", _fake_get_video_encoder
    ):
        writer.write_frame(
            0,
            {
                "LdrColor": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
                "DistanceToCameraSD": rng.random((48, 64)).astype(np.float32),
                "SemanticInstanceSegmentation": rng.integers(
                    0, 5, (48, 64), dtype=np.int32
                ),
            },
            backend,
        )
        writer.finalize(backend)

    manifest, manifest_dir = load_manifest(out)
    with pytest.raises(ValueError, match="no 'depth' conditioning"):
        build_transfer_spec(manifest, manifest_dir, conditioning="depth")


def test_empty_prompt_raises(tmp_path):
    out = _write_writer_output(tmp_path / "no_prompt")
    manifest, manifest_dir = load_manifest(out)
    with pytest.raises(ValueError, match="non-empty prompt"):
        build_transfer_spec(manifest, manifest_dir, conditioning="depth", prompt="")


def test_missing_control_file_raises(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    # Remove the depth control video so resolution fails.
    (writer_output / "conditioning" / "000000_depth.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="Control video not found"):
        build_transfer_spec(manifest, manifest_dir, conditioning="depth", prompt=PROMPT)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Spec serialization + command construction
# ---------------------------------------------------------------------------


def test_prepare_transfer_writes_spec(writer_output, tmp_path):
    spec_dir = tmp_path / "specs"
    spec_path, spec = prepare_transfer(
        writer_output, spec_dir, conditioning="depth", prompt=PROMPT
    )
    assert spec_path.is_file()
    on_disk = json.loads(spec_path.read_text())
    assert on_disk == spec
    assert on_disk["model_mode"] == "video2video"
    # The control path in the written spec is a real file.
    assert Path(on_disk["depth"]["control_path"]).is_file()


def test_build_inference_command_direct(tmp_path):
    cmd = build_inference_command(
        tmp_path / "spec.json",
        tmp_path / "out",
        checkpoint="Cosmos3-Nano",
        seed=2026,
        python_executable="/venv/bin/python",
    )
    assert cmd[0] == "/venv/bin/python"
    assert cmd[1:3] == ["-m", "cosmos_framework.scripts.inference"]
    assert "--checkpoint-path" in cmd
    assert cmd[cmd.index("--checkpoint-path") + 1] == "Cosmos3-Nano"
    assert "-i" in cmd and "-o" in cmd
    assert cmd[cmd.index("--seed") + 1] == "2026"


def test_build_inference_command_torchrun_and_no_guardrails(tmp_path):
    cmd = build_inference_command(
        tmp_path / "spec.json",
        tmp_path / "out",
        use_torchrun=True,
        nproc_per_node=1,
        guardrails=False,
        python_executable="python",
    )
    assert "torch.distributed.run" in cmd
    assert "--nproc-per-node=1" in cmd
    assert "--no-guardrails" in cmd


def test_output_video_path(writer_output):
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest, manifest_dir, conditioning="depth", prompt=PROMPT
    )
    vid = output_video_path("/tmp/out", spec)
    assert vid == Path("/tmp/out/transfer_depth/vision.mp4")


# ---------------------------------------------------------------------------
# End-to-end Cosmos 3 inference (requires GPU + Cosmos Framework + checkpoint)
# ---------------------------------------------------------------------------


def _cosmos_framework_available() -> bool:
    return shutil.which("python") is not None and (
        __import__("importlib").util.find_spec("cosmos_framework") is not None
    )


@pytest.mark.e2e
@pytest.mark.gpu
def test_end_to_end_transfer_inference(writer_output, tmp_path):
    """Run real Cosmos 3 transfer on the writer's depth conditioning video.

    Skips unless ``COSMOS3_CHECKPOINT_PATH`` is set and the Cosmos Framework is
    importable (which in turn requires an NVIDIA GPU). On a suitable host this
    proves the writer output drives Cosmos 3 transfer end to end.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed (Cosmos Framework venv required)")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for Cosmos 3 inference")
    if not _cosmos_framework_available():
        pytest.skip("cosmos_framework not importable")
    checkpoint = os.environ.get("COSMOS3_CHECKPOINT_PATH")
    if not checkpoint:
        pytest.skip("set COSMOS3_CHECKPOINT_PATH to run end-to-end transfer")

    spec_dir = tmp_path / "specs"
    out_dir = tmp_path / "transfer_out"
    spec_path, spec = prepare_transfer(
        writer_output, spec_dir, conditioning="depth", prompt=PROMPT
    )
    cmd = build_inference_command(spec_path, out_dir, checkpoint=checkpoint, seed=2026)
    subprocess.run(cmd, check=True)
    assert output_video_path(out_dir, spec).is_file()


class TestBridgeInputValidation:
    """Input-validation branches in the spec/command builders."""

    def test_aspect_ratio_rejects_non_positive_height(self):
        with pytest.raises(ValueError, match="height must be positive"):
            _nearest_aspect_ratio(1920, 0)

    def test_select_clip_empty_manifest_raises(self):
        with pytest.raises(ValueError, match="no clips"):
            _select_clip({"clips": []}, 0)

    def test_select_clip_unknown_index_raises(self):
        with pytest.raises(ValueError, match="No clip with clip_id=5"):
            _select_clip({"clips": [{"clip_id": 0}]}, 5)

    def test_negative_prompt_file_embedded_in_spec(self, tmp_path: Path):
        (tmp_path / "ctrl.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        manifest = {
            "clips": [
                {
                    "clip_id": 0,
                    "conditioning": {"depth": "ctrl.mp4"},
                    "resolution": [64, 48],
                    "num_frames": 2,
                    "fps": 24,
                }
            ]
        }
        spec = build_transfer_spec(
            manifest,
            tmp_path,
            conditioning="depth",
            prompt="a caption",
            negative_prompt_file=tmp_path / "neg.json",
        )
        assert spec["negative_prompt_file"] == str(tmp_path / "neg.json")

    def test_build_inference_command_appends_extra_args(self, tmp_path: Path):
        cmd = build_inference_command(
            tmp_path / "spec.json",
            tmp_path / "out",
            python_executable="python",
            extra_args=["--foo", "bar"],
        )
        assert cmd[-2:] == ["--foo", "bar"]
