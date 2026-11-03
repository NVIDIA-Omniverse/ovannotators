# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental perceptually separated palettes for segmentation images."""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ovannotators.exceptions import MissingRenderVarError
from ovannotators.writers._helpers import _to_numpy, id_label_map_from_render_vars

ColorAccessibilityMode = Literal["none", "protanopia", "deuteranopia", "tritanopia"]
SegmentationColorKey = Literal[
    "instance_id", "stable_id", "semantic_id", "semantic_label"
]

SEGMENTATION_VAR_BY_KEY: dict[SegmentationColorKey, str] = {
    "instance_id": "SemanticInstanceSegmentation",
    "stable_id": "StableIdSegmentation",
    "semantic_id": "SemanticSegmentation",
    "semantic_label": "SemanticSegmentation",
}

_GRID_LEVELS = 24
_PALETTE_STATE_VERSION = 1
_DISTINCT_PALETTE_VERSION = 1

# Full-severity linear-RGB simulation matrices from Machado et al. (2009).
_ACCESSIBILITY_MATRICES = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        dtype=np.float32,
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
        dtype=np.float32,
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ],
        dtype=np.float32,
    ),
}


def validate_accessibility_mode(value: object) -> None:
    """Raise ``ValueError`` unless ``value`` is a supported accessibility mode.

    Shared by the writer configs so invalid modes fail when the config is built.
    """
    if value not in {"none", *_ACCESSIBILITY_MATRICES}:
        raise ValueError(
            "segmentation_palette_accessibility must be 'none', 'protanopia', "
            f"'deuteranopia', or 'tritanopia', got {value!r}"
        )


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB values to linear RGB."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB values to the perceptually uniform Oklab space."""
    rgb = np.asarray(rgb, dtype=np.float32)
    lms = rgb @ np.array(
        [
            [0.4122214708, 0.2119034982, 0.0883024619],
            [0.5363325363, 0.6806995451, 0.2817188376],
            [0.0514459929, 0.1073969566, 0.6299787005],
        ],
        dtype=np.float32,
    )
    lms = np.cbrt(lms)
    return lms @ np.array(
        [
            [0.2104542553, 1.9779984951, 0.0259040371],
            [0.7936177850, -2.4285922050, 0.7827717662],
            [-0.0040720468, 0.4505937099, -0.8086757660],
        ],
        dtype=np.float32,
    )


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB values to Oklab."""
    return _linear_rgb_to_oklab(_srgb_to_linear(rgb))


def _simulate_accessibility_oklab(
    rgb: np.ndarray, accessibility: ColorAccessibilityMode
) -> np.ndarray:
    """Return Oklab coordinates after the selected vision simulation."""
    if accessibility == "none":
        return _srgb_to_oklab(rgb)
    if accessibility not in _ACCESSIBILITY_MATRICES:
        raise ValueError(f"unsupported color accessibility mode: {accessibility!r}")
    simulated = np.clip(
        _srgb_to_linear(rgb) @ _ACCESSIBILITY_MATRICES[accessibility].T, 0.0, 1.0
    )
    return _linear_rgb_to_oklab(simulated)


@lru_cache(maxsize=4)
def _candidate_colors(
    accessibility: ColorAccessibilityMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a compact RGB candidate grid and its comparison coordinates."""
    if accessibility != "none" and accessibility not in _ACCESSIBILITY_MATRICES:
        raise ValueError(f"unsupported color accessibility mode: {accessibility!r}")

    levels = np.linspace(0, 255, _GRID_LEVELS, dtype=np.uint8)
    r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
    rgb8 = np.stack((r, g, b), axis=-1).reshape(-1, 3)
    srgb = rgb8.astype(np.float32) / 255.0
    linear = _srgb_to_linear(srgb)
    oklab = _linear_rgb_to_oklab(linear)

    chroma = np.linalg.norm(oklab[:, 1:], axis=1)
    usable = (oklab[:, 0] >= 0.35) & (oklab[:, 0] <= 0.90) & (chroma >= 0.05)
    rgb8 = rgb8[usable]
    oklab = oklab[usable]

    if accessibility == "none":
        perceived = oklab
    else:
        perceived = _simulate_accessibility_oklab(
            rgb8.astype(np.float32) / 255.0, accessibility
        )

    return rgb8, oklab, perceived


class DistinctColorPalette:
    """Lazily generate a deterministic farthest-point color sequence.

    Candidates come from a compact sRGB grid rather than all 16.7 million RGB
    triplets. Each extension updates a cached nearest-color distance array, so
    producing ``n`` colors costs O(n * candidates) once per dataset session.
    """

    def __init__(self, accessibility: ColorAccessibilityMode = "none") -> None:
        """Initialize an empty palette optimized for the requested vision mode."""
        self._accessibility = accessibility
        self._rgb, self._oklab, self._perceived = _candidate_colors(accessibility)
        self._selected: list[int] = []

        black_white = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        reserved = _srgb_to_oklab(black_white)
        normal_distance = np.min(
            np.sum((self._oklab[:, None, :] - reserved[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        if accessibility == "none":
            self._min_distance = normal_distance
        else:
            perceived_distance = np.min(
                np.sum(
                    (self._perceived[:, None, :] - reserved[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
            self._min_distance = np.minimum(normal_distance, perceived_distance)

    @property
    def capacity(self) -> int:
        """Return the number of unique colors in the candidate grid."""
        return len(self._rgb)

    def colors(self, count: int) -> np.ndarray:
        """Return the first ``count`` colors, extending the sequence as needed.

        Counts beyond the finite candidate grid cycle deterministically. At the
        intended hundreds-to-thousands scale, every returned color is unique.
        """
        if count < 0:
            raise ValueError("color count must be non-negative")

        unique_count = min(count, self.capacity)
        while len(self._selected) < unique_count:
            index = int(np.argmax(self._min_distance))
            self._selected.append(index)

            normal_distance = np.sum((self._oklab - self._oklab[index]) ** 2, axis=1)
            if self._accessibility == "none":
                distance = normal_distance
            else:
                perceived_distance = np.sum(
                    (self._perceived - self._perceived[index]) ** 2, axis=1
                )
                distance = np.minimum(normal_distance, perceived_distance)
            np.minimum(self._min_distance, distance, out=self._min_distance)

        if count == 0:
            return np.empty((0, 3), dtype=np.uint8)
        selected = self._rgb[np.asarray(self._selected, dtype=np.intp)]
        if count <= self.capacity:
            return selected[:count]
        return selected[np.arange(count) % self.capacity]


class SegmentationColorizer:
    """Assign deterministic, optionally persisted colors to segmentation keys."""

    def __init__(
        self,
        *,
        accessibility: ColorAccessibilityMode = "none",
        path: str | Path | None = None,
        allowed_key_modes: set[SegmentationColorKey] | None = None,
    ) -> None:
        """Initialize a colorizer and load its optional state file."""
        if accessibility not in {"none", *set(_ACCESSIBILITY_MATRICES)}:
            raise ValueError(
                f"unsupported segmentation palette accessibility: {accessibility!r}"
            )
        self._accessibility = accessibility
        self._path = None if path is None else Path(path)
        self._allowed_key_modes = (
            set(SEGMENTATION_VAR_BY_KEY)
            if allowed_key_modes is None
            else allowed_key_modes
        )
        invalid_modes = self._allowed_key_modes - set(SEGMENTATION_VAR_BY_KEY)
        if invalid_modes:
            raise ValueError(f"unsupported segmentation color keys: {invalid_modes}")
        self._slots: dict[tuple[SegmentationColorKey, tuple[int, ...] | str], int] = {}
        self._semantic_labels: dict[int, str] = {}
        self._distinct_palette: DistinctColorPalette | None = None
        self._dirty = False
        self._load()

    @property
    def assignments(
        self,
    ) -> dict[tuple[SegmentationColorKey, tuple[int, ...] | str], int]:
        """Return a copy of the typed key-to-slot assignments."""
        return dict(self._slots)

    def prepare_frame(
        self,
        seg: Any,
        render_vars: dict[str, Any],
        key_mode: SegmentationColorKey,
        *,
        writer_name: str,
        frame_id: int,
    ) -> np.ndarray:
        """Convert a frame to NumPy and capture metadata required by its key mode."""
        self._validate_key_mode(key_mode)
        seg_array = _to_numpy(seg)
        if key_mode != "semantic_label":
            return seg_array

        labels = id_label_map_from_render_vars(render_vars)
        if labels is not None:
            for key, value in labels.items():
                semantic_id = int(key)
                label = str(value).strip()
                if not label:
                    raise ValueError(f"semantic id {semantic_id} has an empty label")
                previous = self._semantic_labels.get(semantic_id)
                if previous is not None and previous != label:
                    raise ValueError(
                        f"semantic id {semantic_id} changed label from "
                        f"{previous!r} to {label!r}"
                    )
                self._semantic_labels[semantic_id] = label

        scalar_ids = seg_array[..., 0] if seg_array.ndim > 2 else seg_array
        missing_ids = [
            int(value)
            for value in np.unique(scalar_ids)
            if int(value) != 0 and int(value) not in self._semantic_labels
        ]
        if missing_ids:
            raise MissingRenderVarError(writer_name, "SemanticIdMap", frame_id=frame_id)
        return seg_array

    def colorize(self, seg: np.ndarray, key_mode: SegmentationColorKey) -> np.ndarray:
        """Colorize one segmentation identity frame using typed assignments."""
        self._validate_key_mode(key_mode)
        ids = np.asarray(seg)
        if ids.ndim == 2:
            ids = ids[..., np.newaxis]
        if ids.ndim != 3 or ids.shape[2] == 0:
            raise ValueError("segmentation must have shape (H, W) or (H, W, C)")

        if ids.shape[2] == 1:
            unique_ids, inverse = np.unique(ids[..., 0], return_inverse=True)
            unique_ids = unique_ids[:, np.newaxis]
        else:
            flat = np.ascontiguousarray(ids.reshape(-1, ids.shape[2]))
            # Most rendered instance-ID buffers have unique first channels.
            # A scalar unique is much cheaper than sorting multichannel rows.
            # Check every pixel against its representative so a first-channel
            # collision takes the exact full-row path instead of merging IDs.
            _, representatives, scalar_inverse = np.unique(
                flat[:, 0], return_index=True, return_inverse=True
            )
            candidate_ids = flat[representatives]
            if np.array_equal(flat, candidate_ids[scalar_inverse]):
                unique_ids, inverse = candidate_ids, scalar_inverse
            else:
                unique_ids, inverse = np.unique(flat, axis=0, return_inverse=True)

        slots_for_unique = np.full(len(unique_ids), -1, dtype=np.int64)
        for i, uid in enumerate(unique_ids):
            raw_key = tuple(int(channel) for channel in uid)
            if not any(raw_key):
                continue
            value: tuple[int, ...] | str = raw_key
            if key_mode == "semantic_label":
                semantic_id = raw_key[0]
                try:
                    value = self._semantic_labels[semantic_id]
                except KeyError as exc:
                    raise ValueError(
                        f"semantic label is unavailable for semantic id {semantic_id}"
                    ) from exc
            key = (key_mode, value)
            if key not in self._slots:
                self._slots[key] = len(self._slots)
                self._dirty = True
            slots_for_unique[i] = self._slots[key]

        unique_colors = np.zeros((len(unique_ids), 3), dtype=np.uint8)
        foreground = slots_for_unique >= 0
        if foreground.any():
            colors = self._colors(len(self._slots))
            unique_colors[foreground] = colors[slots_for_unique[foreground]]
        return unique_colors[inverse].reshape(*ids.shape[:2], 3)

    def save(self) -> None:
        """Atomically persist assignments when a state path is configured."""
        if self._path is None or not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        assignments = [
            {
                "key_type": key_type,
                "key": value if isinstance(value, str) else list(value),
                "slot": slot,
            }
            for (key_type, value), slot in sorted(
                self._slots.items(), key=lambda item: item[1]
            )
        ]
        state = {
            "format_version": _PALETTE_STATE_VERSION,
            "algorithm": self._algorithm,
            "accessibility": self._accessibility,
            "colors": self._colors(len(assignments)).tolist(),
            "assignments": assignments,
        }
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(state, stream, indent=2, sort_keys=True)
                stream.write("\n")
                temp_path = Path(stream.name)
            os.replace(temp_path, self._path)
            self._dirty = False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @property
    def _algorithm(self) -> str:
        return f"distinct-v{_DISTINCT_PALETTE_VERSION}"

    def _validate_key_mode(self, key_mode: SegmentationColorKey) -> None:
        if key_mode not in self._allowed_key_modes:
            raise ValueError(f"segmentation color key {key_mode!r} is not enabled")

    def _colors(self, count: int) -> np.ndarray:
        if self._distinct_palette is None:
            self._distinct_palette = DistinctColorPalette(self._accessibility)
        return self._distinct_palette.colors(count)

    def _parse_key(
        self, key_type: Any, value: Any
    ) -> tuple[SegmentationColorKey, tuple[int, ...] | str]:
        if key_type not in self._allowed_key_modes:
            raise ValueError(f"palette state key type {key_type!r} is not enabled")
        if key_type == "semantic_label":
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "semantic_label palette keys must be non-empty strings"
                )
            return key_type, value
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(channel, int) or channel < 0 for channel in value)
        ):
            raise ValueError("numeric palette keys must be non-empty integer lists")
        numeric = tuple(value)
        if not any(numeric):
            raise ValueError("the all-zero background cannot have a palette assignment")
        return key_type, numeric

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise TypeError("palette state must be a JSON object")
            expected_metadata = {
                "format_version": _PALETTE_STATE_VERSION,
                "algorithm": self._algorithm,
                "accessibility": self._accessibility,
            }
            for name, expected in expected_metadata.items():
                if state.get(name) != expected:
                    raise ValueError(
                        f"palette state {name!r} is {state.get(name)!r}; expected {expected!r}"
                    )

            assignments = state.get("assignments")
            colors = state.get("colors")
            if not isinstance(assignments, list) or not isinstance(colors, list):
                raise TypeError(
                    "palette state requires list 'assignments' and 'colors'"
                )

            loaded: dict[tuple[SegmentationColorKey, tuple[int, ...] | str], int] = {}
            used_slots: set[int] = set()
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    raise TypeError("each palette assignment must be an object")
                key = self._parse_key(assignment.get("key_type"), assignment.get("key"))
                slot = assignment.get("slot")
                if not isinstance(slot, int) or slot < 0:
                    raise ValueError(
                        "palette assignment slots must be non-negative integers"
                    )
                if key in loaded or slot in used_slots:
                    raise ValueError(
                        "palette assignments contain duplicate keys or slots"
                    )
                loaded[key] = slot
                used_slots.add(slot)

            if used_slots != set(range(len(loaded))):
                raise ValueError(
                    "palette assignment slots must be contiguous from zero"
                )
            stored_colors = np.asarray(colors)
            if stored_colors.shape != (len(loaded), 3):
                raise ValueError("palette colors must have shape (assignment_count, 3)")
            if not np.issubdtype(stored_colors.dtype, np.integer) or np.any(
                (stored_colors < 0) | (stored_colors > 255)
            ):
                raise ValueError("palette colors must contain RGB integers in [0, 255]")
            # Colors are 8-bit grid points, so rounding drift cannot shift them
            # by one level; only an argmax tie flip changes the sequence. A
            # tolerance would not absorb that, so the comparison stays exact
            # and the golden-color tests pin the sequence on every CI platform.
            if not np.array_equal(
                stored_colors.astype(np.uint8), self._colors(len(loaded))
            ):
                raise ValueError("palette colors do not match the recorded algorithm")
            self._slots = loaded
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid segmentation palette state {self._path}: {exc}"
            ) from exc
