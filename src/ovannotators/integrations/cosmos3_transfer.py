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

"""Bridge :class:`CosmosTransferWriter` output to NVIDIA Cosmos 3 video transfer.

``ovannotators.writers.cosmos_transfer.CosmosTransferWriter`` emits an RGB
video, one or more conditioning videos (Canny edges, colorized depth, colorized
segmentation), and a ``manifest.json``. NVIDIA Cosmos 3 "video transfer"
(``cosmos_framework.scripts.inference`` with ``model_mode="video2video"``)
regenerates a photoreal clip from a *control* video plus a text caption.

This module converts a writer manifest into a Cosmos 3 Framework input spec so a
conditioning video can be used directly as a transfer control signal. The spec
schema and inference invocation mirror the upstream transfer cookbook
(``cookbooks/cosmos3/generator/transfer`` in ``github.com/NVIDIA/cosmos``) and
the argument parsing in ``cosmos_framework.inference.args``.

The bridge itself has no GPU or PyTorch dependency: it only reads the writer
manifest and produces JSON + a command line. Actually running the produced
command requires the Cosmos Framework, a Cosmos 3 checkpoint, and an NVIDIA GPU.

Example:
    >>> spec_path, spec = prepare_transfer(
    ...     "writer_out", "transfer_specs", conditioning="depth",
    ...     prompt="A robotic arm wiping a plate on a kitchen counter.",
    ... )
    >>> cmd = build_inference_command(
    ...     spec_path, "transfer_out", checkpoint="Cosmos3-Nano"
    ... )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Mappings mirrored from the upstream Cosmos 3 transfer pipeline
# ---------------------------------------------------------------------------

#: Writer conditioning name -> Cosmos 3 transfer hint key.
#:
#: The writer's ``canny`` conditioning is an edge map, which Cosmos 3 consumes
#: through the ``edge`` hint; ``depth`` maps to ``depth``. Both the flat
#: ``segmentation`` and the normal-shaded ``shaded_segmentation`` drive the
#: ``seg`` control branch (the latter is a richer, geometry-aware variant of
#: the same segmentation signal).
CONDITIONING_TO_HINT: dict[str, str] = {
    "canny": "edge",
    "depth": "depth",
    "segmentation": "seg",
    "shaded_segmentation": "seg",
}

#: Per-hint guidance defaults, mirrored from ``cosmos_framework.inference.args``
#: (``_TRANSFER_GUIDANCE_DEFAULTS``) and the shipped ``specs/*.json``.
_HINT_GUIDANCE_DEFAULTS: dict[str, dict[str, float]] = {
    "edge": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
    "blur": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
    "depth": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
    "seg": {"guidance": 3.0, "control_guidance": 2.0, "shift": 10.0},
    "wsm": {"guidance": 1.0, "control_guidance": 3.0, "shift": 10.0},
}

#: Resolution tiers accepted by the Cosmos 3 Framework (the ``Resolution``
#: literal in ``cosmos_framework.inference.args``). The bridge snaps the writer's
#: native height to the nearest tier.
_ALLOWED_RESOLUTIONS: tuple[int, ...] = (256, 480, 720, 768, 1080)

#: Aspect ratios accepted by the Cosmos 3 Framework (the ``AspectRatio`` literal),
#: mapped to their numeric width/height value for nearest-match selection.
_ALLOWED_ASPECT_RATIOS: dict[str, float] = {
    "1,1": 1.0,
    "4,3": 4.0 / 3.0,
    "3,4": 3.0 / 4.0,
    "16,9": 16.0 / 9.0,
    "9,16": 9.0 / 16.0,
}

#: Cosmos 3 model checkpoints, smallest first. ``Cosmos3-Nano`` (16B) is the
#: smallest publicly released model and the only one that fits on a single GPU
#: in the latency preset.
SMALLEST_CHECKPOINT: str = "Cosmos3-Nano"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_resolution(height: int) -> str:
    """Snap a pixel height to the nearest Cosmos 3 resolution tier.

    Args:
        height: Native frame height in pixels.

    Returns:
        str: The closest allowed resolution tier as a string (e.g. ``"720"``).
    """
    nearest = min(_ALLOWED_RESOLUTIONS, key=lambda tier: abs(tier - height))
    return str(nearest)


def _nearest_aspect_ratio(width: int, height: int) -> str:
    """Snap a (width, height) to the nearest Cosmos 3 aspect-ratio literal.

    Args:
        width: Native frame width in pixels.
        height: Native frame height in pixels.

    Returns:
        str: The closest allowed aspect ratio (e.g. ``"16,9"``).

    Raises:
        ValueError: If ``height`` is not positive.
    """
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    ratio = width / height
    return min(
        _ALLOWED_ASPECT_RATIOS,
        key=lambda name: abs(_ALLOWED_ASPECT_RATIOS[name] - ratio),
    )


def load_manifest(source: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a :class:`CosmosTransferWriter` ``manifest.json``.

    Args:
        source: Path to a ``manifest.json`` file or to a directory that
            contains one.

    Returns:
        tuple[dict, Path]: The parsed manifest and the directory that holds it
        (used as the base for resolving relative conditioning video paths).

    Raises:
        FileNotFoundError: If no manifest can be located at ``source``.
    """
    path = Path(source)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No Cosmos writer manifest found at: {path}")
    manifest = json.loads(path.read_text())
    return manifest, path.parent


def _select_clip(manifest: dict[str, Any], clip_index: int) -> dict[str, Any]:
    """Return the manifest clip entry with ``clip_id == clip_index``.

    Args:
        manifest: Parsed writer manifest.
        clip_index: ``clip_id`` to select.

    Returns:
        dict: The matching clip entry.

    Raises:
        ValueError: If the manifest has no clips or no clip matches.
    """
    clips = manifest.get("clips") or []
    if not clips:
        raise ValueError("Manifest contains no clips.")
    for clip in clips:
        if clip.get("clip_id") == clip_index:
            return clip
    available = [c.get("clip_id") for c in clips]
    raise ValueError(
        f"No clip with clip_id={clip_index} in manifest (available: {available})."
    )


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def build_transfer_spec(
    manifest: dict[str, Any],
    manifest_dir: str | Path,
    *,
    conditioning: str = "depth",
    prompt: str | None = None,
    clip_index: int = 0,
    name: str | None = None,
    negative_prompt_file: str | Path | None = None,
    seed: int = 2026,
    num_steps: int = 50,
    guidance: float | None = None,
    control_guidance: float | None = None,
    num_conditional_frames: int = 1,
    absolute_control_path: bool = True,
    require_control_exists: bool = True,
) -> dict[str, Any]:
    """Build a Cosmos 3 Framework transfer (``video2video``) input spec.

    The spec points its control-hint block at one of the writer's conditioning
    videos and embeds a caption, so ``cosmos_framework.scripts.inference`` can
    regenerate a photoreal clip guided by that control signal.

    Args:
        manifest: Parsed writer manifest (see :func:`load_manifest`).
        manifest_dir: Directory containing the manifest, used to resolve the
            relative conditioning video path.
        conditioning: Writer conditioning to use as the control signal. One of
            ``"canny"``, ``"depth"``, ``"segmentation"``, or
            ``"shaded_segmentation"``.
        prompt: Caption for generation, supplied at transfer time. A non-empty
            caption is required (the writer manifest does not store one).
        clip_index: ``clip_id`` of the manifest clip to convert.
        name: Output sample name (becomes the output subdirectory). Defaults to
            ``"transfer_{hint}"``.
        negative_prompt_file: Optional path to a Cosmos 3 negative-prompt JSON.
        seed: Inference seed embedded in the spec.
        num_steps: Number of diffusion sampling steps.
        guidance: Classifier-free guidance scale. Defaults to the per-hint value.
        control_guidance: Control guidance scale. Defaults to the per-hint value.
        num_conditional_frames: Autoregressive conditional-frame count.
        absolute_control_path: If ``True`` (default), store an absolute
            ``control_path``; otherwise store the path relative to
            ``manifest_dir``.
        require_control_exists: If ``True`` (default), raise when the resolved
            control video is missing on disk.

    Returns:
        dict: A Cosmos 3 Framework input spec ready to serialize to JSON.

    Raises:
        ValueError: If ``conditioning`` is unknown, the clip lacks that
            conditioning, or no non-empty prompt is available.
        FileNotFoundError: If ``require_control_exists`` and the control video
            does not exist.
    """
    if conditioning not in CONDITIONING_TO_HINT:
        raise ValueError(
            f"Unknown conditioning {conditioning!r}; expected one of "
            f"{sorted(CONDITIONING_TO_HINT)}."
        )

    clip = _select_clip(manifest, clip_index)
    conditioning_map = clip.get("conditioning") or {}
    if conditioning not in conditioning_map:
        raise ValueError(
            f"Clip {clip_index} has no {conditioning!r} conditioning video "
            f"(available: {sorted(conditioning_map)})."
        )

    hint = CONDITIONING_TO_HINT[conditioning]

    caption = prompt if prompt is not None else clip.get("prompt", "")
    caption = (caption or "").strip()
    if not caption:
        raise ValueError(
            "A non-empty prompt is required for transfer. Pass prompt=... at "
            "transfer time."
        )

    base_dir = Path(manifest_dir)
    control_rel = conditioning_map[conditioning]
    control_abs = (base_dir / control_rel).resolve()
    if require_control_exists and not control_abs.is_file():
        raise FileNotFoundError(f"Control video not found: {control_abs}")
    control_path = str(control_abs) if absolute_control_path else control_rel

    width, height = clip.get("resolution", [1280, 720])
    num_frames = int(clip.get("num_frames", 1))
    fps = int(clip.get("fps", 24))

    hint_defaults = _HINT_GUIDANCE_DEFAULTS.get(hint, {})
    spec: dict[str, Any] = {
        "name": name or f"transfer_{hint}",
        "model_mode": "video2video",
        "resolution": _nearest_resolution(int(height)),
        "aspect_ratio": _nearest_aspect_ratio(int(width), int(height)),
        "num_frames": num_frames,
        "fps": fps,
        "num_video_frames_per_chunk": num_frames,
        "num_conditional_frames": num_conditional_frames,
        "num_first_chunk_conditional_frames": 0,
        "share_vision_temporal_positions": True,
        "shift": hint_defaults.get("shift", 10.0),
        "num_steps": num_steps,
        "seed": seed,
        "guidance": guidance
        if guidance is not None
        else hint_defaults.get("guidance", 3.0),
        "control_guidance": (
            control_guidance
            if control_guidance is not None
            else hint_defaults.get("control_guidance", 1.5)
        ),
        "prompt": caption,
    }

    if negative_prompt_file is not None:
        spec["negative_prompt_file"] = str(negative_prompt_file)

    hint_block: dict[str, Any] = {"control_path": control_path}
    if hint == "edge":
        hint_block["preset_edge_threshold"] = "medium"
    spec[hint] = hint_block

    return spec


def write_transfer_spec(
    spec: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Serialize a transfer spec to a JSON file.

    Args:
        spec: Spec produced by :func:`build_transfer_spec`.
        output_path: Destination ``.json`` path (parent dirs are created).

    Returns:
        Path: The written file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2))
    return path


def output_video_path(output_dir: str | Path, spec: dict[str, Any]) -> Path:
    """Return the path Cosmos 3 writes the generated video to.

    The Framework writes one subdirectory per sample ``name`` containing
    ``vision.mp4``.

    Args:
        output_dir: Directory passed to inference via ``-o``.
        spec: The transfer spec (its ``name`` selects the subdirectory).

    Returns:
        Path: ``{output_dir}/{name}/vision.mp4``.
    """
    return Path(output_dir) / str(spec.get("name", "transfer")) / "vision.mp4"


def build_inference_command(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint: str = SMALLEST_CHECKPOINT,
    seed: int = 2026,
    python_executable: str | None = None,
    use_torchrun: bool = False,
    nproc_per_node: int = 1,
    parallelism_preset: str = "latency",
    guardrails: bool = True,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the ``cosmos_framework.scripts.inference`` command for a spec.

    Mirrors the invocation from the upstream transfer cookbook.

    Args:
        spec_path: Path to the transfer spec JSON (``-i``).
        output_dir: Output directory (``-o``).
        checkpoint: Checkpoint name or path (``--checkpoint-path``). Defaults to
            the smallest model, ``Cosmos3-Nano``.
        seed: Inference seed (``--seed``).
        python_executable: Python interpreter to use. Defaults to the current
            interpreter (e.g. the Cosmos Framework venv's ``python``).
        use_torchrun: If ``True``, launch via ``torchrun`` with
            ``--nproc-per-node``; otherwise launch the module directly.
        nproc_per_node: Process count for ``torchrun`` (single GPU -> 1).
        parallelism_preset: ``--parallelism-preset`` value.
        guardrails: If ``False``, append ``--no-guardrails``.
        extra_args: Additional CLI arguments appended verbatim.

    Returns:
        list[str]: The command argument vector, suitable for
        :func:`subprocess.run`.
    """
    import sys

    if use_torchrun:
        launcher = [
            python_executable or sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={nproc_per_node}",
        ]
    else:
        launcher = [python_executable or sys.executable]

    cmd = [
        *launcher,
        "-m",
        "cosmos_framework.scripts.inference",
        f"--parallelism-preset={parallelism_preset}",
        "-i",
        str(spec_path),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        str(checkpoint),
        "--seed",
        str(seed),
    ]
    if not guardrails:
        cmd.append("--no-guardrails")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def prepare_transfer(
    writer_output: str | Path,
    spec_output_dir: str | Path,
    *,
    conditioning: str = "depth",
    prompt: str | None = None,
    clip_index: int = 0,
    **spec_kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Convert a writer output directory into a transfer spec on disk.

    Convenience wrapper that loads the manifest, builds the spec, and writes it
    next to a recommended output directory layout.

    Args:
        writer_output: Path to the :class:`CosmosTransferWriter` output (a
            directory with ``manifest.json`` or the manifest file itself).
        spec_output_dir: Directory where the generated ``{name}.json`` spec is
            written.
        conditioning: Writer conditioning to use as the control signal.
        prompt: Caption for generation, supplied at transfer time (required).
        clip_index: ``clip_id`` to convert.
        **spec_kwargs: Forwarded to :func:`build_transfer_spec`.

    Returns:
        tuple[Path, dict]: The written spec path and the spec dict.
    """
    manifest, manifest_dir = load_manifest(writer_output)
    spec = build_transfer_spec(
        manifest,
        manifest_dir,
        conditioning=conditioning,
        prompt=prompt,
        clip_index=clip_index,
        **spec_kwargs,
    )
    spec_path = write_transfer_spec(
        spec, Path(spec_output_dir) / f"{spec['name']}.json"
    )
    return spec_path, spec
