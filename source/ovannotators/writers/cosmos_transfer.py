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

"""CosmosTransferWriter — NVIDIA Cosmos Transfer training data writer.

Produces RGB video + conditioning signal videos (canny edge, depth,
segmentation, normal-shaded segmentation) for fine-tuning or evaluating Cosmos
Transfer world-generation models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from ovannotators.exceptions import MissingRenderVarError
from ovannotators.writers._helpers import _to_numpy, validate_required_vars
from ovannotators.writers._registry import WriterRegistry

if TYPE_CHECKING:
    pass


def _generate_seg_palette(n: int = 256) -> np.ndarray:
    """Generate a deterministic segmentation color palette.

    Uses a simple hue-rotation scheme to produce visually distinct colors.
    Index 0 is always black (background).
    """
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        # Deterministic pseudo-random color via bit manipulation
        r = ((i * 157) % 255) + 1
        g = ((i * 89) % 255) + 1
        b = ((i * 213) % 255) + 1
        palette[i] = [r, g, b]
    return palette


_SEG_PALETTE = _generate_seg_palette()


# ---------------------------------------------------------------------------
# Ordered-dither matrix (8x8 Bayer) for banding-free 8-bit depth quantization
# ---------------------------------------------------------------------------

#: Bayer thresholds normalized to ``(0, 1)``. Adding these per-pixel offsets
#: before flooring to 8-bit spreads quantization error spatially, so a smooth
#: depth ramp degrades into fine dither noise instead of visible contour bands.
#: This matters because the H.264 luma channel only carries 256 levels and a
#: clip-global normalized depth often occupies only a fraction of that range.
_BAYER_8X8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float64,
)
_BAYER_8X8_THRESHOLD = (_BAYER_8X8 + 0.5) / 64.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Cosmos accepts only one valid form for each of these, so they are fixed
# constants rather than config fields.
_DEPTH_CLIP_PERCENTILE: float = 0.5
_SHADED_SEG_AMBIENT: float = 0.35


@dataclass
class CosmosTransferWriterConfig:
    """Configuration for :class:`CosmosTransferWriter`.

    Attributes:
        fps: Frames per second for output videos.
        clip_index: Starting index used to name output files, zero-padded to
            six digits (e.g. ``clip_index=0`` → ``videos/000000_rgb.mp4``,
            ``clip_index=42`` → ``videos/000042_rgb.mp4``). A single writer
            instance can emit multiple clips: each :meth:`CosmosTransferWriter.cut`
            (and the final :meth:`CosmosTransferWriter.finalize`) writes one clip
            and advances ``clip_index`` automatically.
        conditioning_types: Which conditioning signal videos to produce.
            Defaults to ``["canny", "depth", "segmentation"]``. Valid values:
            ``"canny"``, ``"depth"``, ``"segmentation"``,
            ``"shaded_segmentation"``. ``"shaded_segmentation"`` combines the
            colorized segmentation with ``NormalSD`` (which must be present in
            the render vars) to produce a per-instance palette shaded by the
            surface normals.
        canny_source: Which render var the canny edges are derived from.
            ``"rgb"`` (default) detects edges in the LdrColor image,
            ``"normals"`` detects surface-orientation discontinuities in
            NormalSD, ``"segmentation"`` detects object boundaries in
            SemanticInstanceSegmentation, and ``"shaded_segmentation"`` detects
            edges in the normal-shaded segmentation (object boundaries plus
            shading relief). The selected source must be present in the render
            vars when ``"canny"`` is in ``conditioning_types``.
        shaded_seg_light_dir: Light direction ``(x, y, z)`` used to shade the
            segmentation for ``"shaded_segmentation"`` (and the matching canny
            source). Normalized internally.
        canny_sigma: Gaussian sigma for canny edge detection.
        canny_low_threshold: Low hysteresis threshold for canny (0-1 range).
        canny_high_threshold: High hysteresis threshold for canny (0-1 range).
    """

    fps: int = 24
    clip_index: int = 0
    conditioning_types: list[str] = field(
        default_factory=lambda: ["canny", "depth", "segmentation"]
    )
    canny_source: Literal["rgb", "normals", "segmentation", "shaded_segmentation"] = (
        "rgb"
    )
    canny_sigma: float = 1.0
    canny_low_threshold: float = 0.1
    canny_high_threshold: float = 0.2
    shaded_seg_light_dir: tuple[float, float, float] = (0.3, 0.5, 0.81)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_REQUIRED_VARS: frozenset[str] = frozenset(
    {"LdrColor", "DistanceToCameraSD", "SemanticInstanceSegmentation"}
)


@WriterRegistry.register
class CosmosTransferWriter:
    """Writes training data for NVIDIA Cosmos Transfer models.

    Produces RGB video plus conditioning signal videos (canny edges, depth
    colorization, segmentation colorization) and a JSON manifest describing
    every clip.

    A single writer can emit multiple clips: accumulate frames with
    :meth:`write_frame`, then call :meth:`cut` to close the current clip and
    begin a new one (e.g. to slice a long capture into sequential clips, or to
    render one clip per scene variation). :meth:`finalize` flushes the last
    clip. Each clip's files are named with its own ``clip_id`` and all clips
    share one ``manifest.json``.

    Output structure::

        {output_dir}/
        ├── videos/
        │   └── {clip_id:06d}_rgb.mp4
        ├── conditioning/
        │   ├── {clip_id:06d}_canny.mp4
        │   ├── {clip_id:06d}_depth.mp4
        │   ├── {clip_id:06d}_segmentation.mp4
        │   └── {clip_id:06d}_shaded_segmentation.mp4
        ├── thumbnails/
        │   └── {clip_id:06d}.jpg
        └── manifest.json

    Edge conditioning uses a device-agnostic Warp Canny kernel (runs on CUDA or
    CPU). Uses ``PyNvVideoCodec`` for GPU-accelerated video encoding.
    """

    name: ClassVar[str] = "cosmos_transfer"
    required_render_vars: ClassVar[frozenset[str]] = _REQUIRED_VARS

    def __init__(self, config: CosmosTransferWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (CosmosTransferWriterConfig | None): Writer options; defaults
                are used when ``None``.
        """
        self._config = config or CosmosTransferWriterConfig()
        self._rgb_frames: list[np.ndarray] = []
        self._depth_frames: list[np.ndarray] = []
        self._seg_frames: list[np.ndarray] = []
        self._normal_frames: list[np.ndarray] = []
        self._shaded_seg_frames: list[np.ndarray] | None = None
        # Output resolution is taken from the incoming render vars (the render
        # settings decide it), captured from the first LdrColor frame.
        self._resolution: tuple[int, int] | None = None
        # Index used to name the next clip's output files; advances each time a
        # clip is flushed via cut() or finalize().
        self._clip_index = self._config.clip_index
        # Manifest entries for every clip flushed so far.
        self._clips: list[dict[str, Any]] = []
        self._finalized = False

    def _output_resolution(self) -> tuple[int, int]:
        """Return the output ``(width, height)`` derived from the input frames.

        The resolution follows the render settings rather than a config value:
        it is captured from the first ``LdrColor`` frame in
        :meth:`write_frame`, and falls back to inferring from any accumulated
        frame so the colorization helpers can be exercised in isolation.

        Returns:
            tuple[int, int]: The output ``(width, height)`` in pixels.
        """
        if self._resolution is not None:
            return self._resolution
        for frames in (
            self._rgb_frames,
            self._depth_frames,
            self._seg_frames,
            self._normal_frames,
        ):
            if frames:
                h0, w0 = np.asarray(frames[0]).shape[:2]
                return (int(w0), int(h0))
        return (0, 0)

    # ------------------------------------------------------------------
    # Frame accumulation
    # ------------------------------------------------------------------

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Any,
    ) -> None:
        """Accumulate one frame's conditioning data for encoding in finalize().

        Args:
            frame_id (int): Zero-based frame index within the clip.
            render_vars (dict[str, np.ndarray]): Frame render vars; must include
                this writer's ``required_render_vars`` (and ``NormalSD`` when a
                normals-derived conditioning type is enabled).
            backend (Any): Destination backend (used at finalize time).

        Raises:
            MissingRenderVarError: If a required render var is missing.
        """
        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )

        cfg = self._config

        # Frames are accumulated on the host for later video encoding; copy any
        # Warp (e.g. ovrtx) views to NumPy at ingest.
        # RGB — kept at its native (render) resolution, which sets the output
        # resolution for every conditioning video.
        rgb = _to_numpy(render_vars["LdrColor"])
        if self._resolution is None:
            h0, w0 = rgb.shape[:2]
            self._resolution = (int(w0), int(h0))
        self._rgb_frames.append(rgb)

        # Depth
        if "depth" in cfg.conditioning_types:
            depth = render_vars.get("DistanceToCameraSD")
            if depth is not None:
                self._depth_frames.append(_to_numpy(depth))

        canny_enabled = "canny" in cfg.conditioning_types
        shaded_seg_enabled = "shaded_segmentation" in cfg.conditioning_types
        canny_from_seg = canny_enabled and cfg.canny_source == "segmentation"
        canny_from_shaded = canny_enabled and cfg.canny_source == "shaded_segmentation"
        canny_from_normals = canny_enabled and cfg.canny_source == "normals"

        # Segmentation — needed for the segmentation conditioning video, the
        # shaded-segmentation conditioning video, and/or when canny edges are
        # derived from (shaded) segmentation.
        need_seg = (
            "segmentation" in cfg.conditioning_types
            or shaded_seg_enabled
            or canny_from_seg
            or canny_from_shaded
        )
        if need_seg:
            seg = render_vars.get("SemanticInstanceSegmentation")
            if seg is not None:
                self._seg_frames.append(_to_numpy(seg))

        # Normals — needed to shade the segmentation and/or when canny edges are
        # derived from normals or the shaded segmentation.
        need_normals = canny_from_normals or shaded_seg_enabled or canny_from_shaded
        if need_normals:
            normals = render_vars.get("NormalSD")
            if normals is None:
                raise MissingRenderVarError(self.name, "NormalSD", frame_id=frame_id)
            self._normal_frames.append(_to_numpy(normals))

    # ------------------------------------------------------------------
    # Clip boundaries — encode the buffered frames and start a new clip
    # ------------------------------------------------------------------

    def cut(self, backend: Any) -> int | None:
        """Finalize the buffered frames into a clip and start a new one.

        Encodes the frames accumulated so far into their own RGB +
        conditioning videos, appends the clip to the manifest, and resets the
        frame buffers so subsequent :meth:`write_frame` calls accumulate a fresh
        clip with the next ``clip_index``. The manifest on disk is updated after
        every cut so partial datasets stay readable if the run is interrupted.

        Use this to slice a long capture into several sequential clips, or to
        emit one clip per scene variation (modify the scene, render more frames,
        then ``cut`` again).

        No-op when no frames are buffered (so back-to-back cuts never produce
        empty clips).

        Args:
            backend: Storage backend the clip videos and manifest are written
                to.

        Returns:
            int | None: The ``clip_id`` of the clip that was written, or
            ``None`` when there were no buffered frames to flush.
        """
        clip_id = self._flush_clip(backend)
        if clip_id is None:
            return None
        self._write_manifest(backend)
        return clip_id

    # ------------------------------------------------------------------
    # Finalization — encode any remaining frames and write the manifest
    # ------------------------------------------------------------------

    def finalize(self, backend: Any) -> None:
        """Encode the remaining buffered frames and write the manifest.

        Flushes any frames accumulated since the last :meth:`cut` (or since the
        writer was created) into a final clip, then writes the manifest
        describing every clip produced. Idempotent: repeated calls are no-ops.

        Args:
            backend: Storage backend the clip videos and manifest are written
                to.
        """
        if self._finalized:
            return
        self._finalized = True

        self._flush_clip(backend)
        if self._clips:
            self._write_manifest(backend)

    def _flush_clip(self, backend: Any) -> int | None:
        """Encode the buffered frames into one clip and reset for the next.

        Args:
            backend: Storage backend the clip videos are written to.

        Returns:
            int | None: The ``clip_id`` that was written, or ``None`` when no
            frames were buffered.
        """
        cfg = self._config
        num_frames = len(self._rgb_frames)
        if num_frames == 0:
            return None

        clip_id = self._clip_index
        clip_str = f"{clip_id:06d}"
        w, h = self._output_resolution()

        # --- Thumbnail (first RGB frame as JPEG) ---
        thumbnail_path = f"thumbnails/{clip_str}.jpg"
        thumbnail_data = self._encode_jpeg(self._rgb_frames[0])
        backend.write(thumbnail_path, thumbnail_data)

        # --- RGB video ---
        rgb_path = f"videos/{clip_str}_rgb.mp4"
        rgb_bytes = self._encode_video(self._rgb_frames, w, h)
        backend.write(rgb_path, rgb_bytes)

        # --- Conditioning videos ---
        conditioning_manifest: dict[str, str] = {}

        if "canny" in cfg.conditioning_types:
            canny_frames = self._compute_canny_frames()
            canny_path = f"conditioning/{clip_str}_canny.mp4"
            canny_bytes = self._encode_video(canny_frames, w, h)
            backend.write(canny_path, canny_bytes)
            conditioning_manifest["canny"] = canny_path

        if "depth" in cfg.conditioning_types and self._depth_frames:
            depth_color_frames = self._colorize_depth_frames()
            depth_path = f"conditioning/{clip_str}_depth.mp4"
            depth_bytes = self._encode_video(depth_color_frames, w, h)
            backend.write(depth_path, depth_bytes)
            conditioning_manifest["depth"] = depth_path

        if "segmentation" in cfg.conditioning_types and self._seg_frames:
            seg_color_frames = self._colorize_seg_frames()
            seg_path = f"conditioning/{clip_str}_segmentation.mp4"
            seg_bytes = self._encode_video(seg_color_frames, w, h)
            backend.write(seg_path, seg_bytes)
            conditioning_manifest["segmentation"] = seg_path

        if (
            "shaded_segmentation" in cfg.conditioning_types
            and self._seg_frames
            and self._normal_frames
        ):
            shaded_frames = self._compute_shaded_seg_frames()
            shaded_path = f"conditioning/{clip_str}_shaded_segmentation.mp4"
            shaded_bytes = self._encode_video(shaded_frames, w, h)
            backend.write(shaded_path, shaded_bytes)
            conditioning_manifest["shaded_segmentation"] = shaded_path

        self._clips.append(
            {
                "clip_id": clip_id,
                "rgb_video": rgb_path,
                "conditioning": conditioning_manifest,
                "thumbnail": thumbnail_path,
                "num_frames": num_frames,
                "fps": cfg.fps,
                "resolution": [w, h],
                "conditioning_types": [
                    k for k in cfg.conditioning_types if k in conditioning_manifest
                ],
            }
        )

        # Advance to the next clip and release the buffered frames. Resolution
        # is reset so the next clip can follow its own render settings.
        self._clip_index += 1
        self._resolution = None
        self._rgb_frames.clear()
        self._depth_frames.clear()
        self._seg_frames.clear()
        self._normal_frames.clear()
        self._shaded_seg_frames = None
        return clip_id

    def _write_manifest(self, backend: Any) -> None:
        """Write ``manifest.json`` describing every clip produced so far.

        Args:
            backend: Storage backend the manifest is written to.
        """
        manifest = {
            "version": "cosmos-transfer-2.5",
            "clips": self._clips,
        }
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        backend.write("manifest.json", manifest_bytes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resize_frame(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize frame to target resolution using Pillow bilinear."""
        from PIL import Image

        h, w = frame.shape[:2]
        if h == target_h and w == target_w:
            return frame.astype(np.uint8) if frame.dtype != np.uint8 else frame

        img = Image.fromarray(
            frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
        )
        resized = img.resize((target_w, target_h), Image.BILINEAR)
        return np.asarray(resized)

    def _compute_canny_frames(self) -> list[np.ndarray]:
        """Compute canny edge maps from the configured ``canny_source``.

        The source (``"rgb"``, ``"normals"``, ``"segmentation"`` or
        ``"shaded_segmentation"``) is reduced to a grayscale image in ``[0, 1]``
        and run through the Warp Canny pipeline (Gaussian blur → Sobel →
        non-max suppression → hysteresis) on the current Warp device.
        """
        cfg = self._config
        w, h = self._output_resolution()
        result: list[np.ndarray] = []

        gray_frames = self._canny_source_gray_frames()

        from ovannotators.writers.canny import canny

        for gray in gray_frames:
            edges = canny(
                gray,
                sigma=cfg.canny_sigma,
                low_threshold=cfg.canny_low_threshold,
                high_threshold=cfg.canny_high_threshold,
            )
            edge_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            edge_rgb[edges] = 255
            result.append(edge_rgb)

        return result

    def _canny_source_gray_frames(self) -> list[np.ndarray]:
        """Return per-frame grayscale ``[0, 1]`` images for Canny detection.

        The image is derived from ``config.canny_source`` and is sized to the
        output resolution so the resulting edge map matches the other
        conditioning videos.
        """
        cfg = self._config
        w, h = self._output_resolution()
        source = cfg.canny_source

        if source == "rgb":
            return [self._to_gray01(rgb) for rgb in self._rgb_frames]

        if source == "normals":
            grays: list[np.ndarray] = []
            for normals in self._normal_frames:
                vis = self._normals_to_rgb(normals)
                vis = self._resize_frame(vis, w, h)
                grays.append(self._to_gray01(vis))
            return grays

        if source == "shaded_segmentation":
            # Edges in the normal-shaded segmentation capture both object
            # boundaries (palette color steps) and shading relief (normal
            # discontinuities). Reuse the cached shaded frames.
            return [
                self._to_gray01(shaded) for shaded in self._compute_shaded_seg_frames()
            ]

        if source == "segmentation":
            # Distinct gray level per region guarantees a detectable step at
            # every object boundary (unlike palette colors, which can collide
            # in luminance).
            grays = []
            for seg in self._seg_frames:
                seg2d = np.asarray(seg)
                if seg2d.ndim > 2:
                    seg2d = seg2d[..., 0]
                seg2d = seg2d.astype(np.int64)
                unique_ids, inverse = np.unique(seg2d, return_inverse=True)
                denom = max(1, unique_ids.size - 1)
                levels = np.arange(unique_ids.size, dtype=np.float64) / denom
                gray = levels[inverse].reshape(seg2d.shape)
                if gray.shape != (h, w):
                    from PIL import Image

                    gray = (
                        np.asarray(
                            Image.fromarray(
                                (gray * 255.0).astype(np.uint8), mode="L"
                            ).resize((w, h), Image.NEAREST),
                            dtype=np.float64,
                        )
                        / 255.0
                    )
                grays.append(gray)
            return grays

        raise ValueError(  # pragma: no cover - guarded by Literal typing
            f"Unknown canny_source: {source!r}"
        )

    @staticmethod
    def _to_gray01(rgb: np.ndarray) -> np.ndarray:
        """Convert an RGB(A) uint8/float image to grayscale float64 in [0, 1]."""
        gray = np.dot(rgb[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
        if gray.max() > 1.0:
            gray = gray / 255.0
        return gray

    @staticmethod
    def _normals_to_rgb(normals: np.ndarray) -> np.ndarray:
        """Map a NormalSD vector field to an ``uint8`` RGB visualization.

        Components in ``[-1, 1]`` are mapped to ``[0, 255]`` via
        ``(n * 0.5 + 0.5)``, matching the BasicWriter normal colorization.
        """
        n = np.asarray(normals).astype(np.float32)
        if n.ndim > 2:
            n = n[..., :3]
        vis = np.clip(n * 0.5 + 0.5, 0.0, 1.0)
        return (vis * 255.0).astype(np.uint8)

    @staticmethod
    def _quantize_with_dither(values01: np.ndarray) -> np.ndarray:
        """Quantize ``[0, 1]`` floats to ``uint8`` ``[0, 255]`` indices.

        Applies 8x8 ordered (Bayer) dithering before flooring so smooth
        gradients do not collapse into visible 8-bit contour bands.

        Args:
            values01: Float array with values in ``[0, 1]``.

        Returns:
            np.ndarray: ``uint8`` array of the same shape with values in
            ``[0, 255]``.
        """
        scaled = values01 * 255.0
        h, w = scaled.shape[:2]
        reps = ((h + 7) // 8, (w + 7) // 8)
        tile = np.tile(_BAYER_8X8_THRESHOLD, reps)[:h, :w]
        scaled = scaled + tile
        return np.clip(np.floor(scaled), 0.0, 255.0).astype(np.uint8)

    def _colorize_depth_frames(self) -> list[np.ndarray]:
        """Encode depth frames into a Cosmos-compatible depth control signal.

        ``DistanceToCameraSD`` (metric distance, possibly ``+inf`` for sky) is
        first mapped into inverse-depth/disparity space (``1 / z``), then
        normalized with a single percentile-robust min/max that is **anchored on
        the first frame** and reused unchanged for every subsequent frame. This
        keeps the depth-to-brightness mapping fixed for the whole clip, so a
        static camera viewing a static object renders that object at a constant
        brightness instead of drifting. Anchoring on the first frame (rather
        than the whole-clip min/max) also keeps the mapping causal: later frames
        cannot retroactively change how earlier ones were encoded. A few outlier
        pixels are excluded via the percentile so they do not collapse the
        usable dynamic range. The signal is oriented so near surfaces are bright
        (the inverse-depth convention Cosmos/DepthAnything use) and quantized to
        8-bit with ordered dithering to avoid contour banding. It is stored in
        luma only (R=G=B), the format Cosmos expects.

        The trade-off of first-frame anchoring is that geometry which moves
        nearer or farther than anything present in frame 0 saturates (clips) to
        full bright/dark; this is intentional, preferring a stable absolute
        scale over per-frame range adaptation.

        Disparity encoding both matches the DepthAnything-style modality the
        depth branch was trained on and concentrates the limited 8-bit code
        levels on near geometry, which is the main lever against the visible
        banding that otherwise propagates into the transfer output.

        Returns:
            list[np.ndarray]: Per-frame ``(H, W, 3)`` ``uint8`` RGB arrays.
        """
        w, h = self._output_resolution()

        # Map every frame into disparity space, marking pixels that have no
        # valid surface (``+inf``/non-positive distance, i.e. sky or background)
        # as NaN so they are excluded from the normalization bounds.
        space_maps: list[np.ndarray] = []
        for depth in self._depth_frames:
            a = np.asarray(depth, dtype=np.float64)
            if a.ndim > 2:
                a = a[..., 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(np.isfinite(a) & (a > 0.0), 1.0 / a, np.nan)
            space_maps.append(m)

        # Anchor the normalization bounds on the first frame that contains any
        # valid surface, then reuse them for every frame. This fixes the
        # depth->brightness mapping for the whole clip (a static object keeps a
        # constant brightness) and stays causal (later frames cannot change how
        # earlier ones are encoded). Robust [p, 100 - p] percentiles keep a few
        # outlier pixels from collapsing the usable dynamic range.
        pct = _DEPTH_CLIP_PERCENTILE
        lo, hi = 0.0, 1.0
        for m in space_maps:
            finite = m[np.isfinite(m)]
            if finite.size:
                lo = float(np.percentile(finite, pct))
                hi = float(np.percentile(finite, 100.0 - pct))
                break
        span = hi - lo

        # Pixels with no valid surface are the far extreme of disparity space:
        # the smallest disparity value.
        far_fill = lo

        result: list[np.ndarray] = []
        for m in space_maps:
            filled = np.where(np.isfinite(m), m, far_fill)
            resized = self._resize_depth(filled.astype(np.float32), w, h).astype(
                np.float64
            )

            if span < 1e-12:
                near_bright = np.zeros_like(resized)
            else:
                # In disparity space large values are near, so the normalized
                # signal is already "near = bright" — the inverse-depth
                # convention Cosmos expects.
                near_bright = np.clip((resized - lo) / span, 0.0, 1.0)

            idx = self._quantize_with_dither(near_bright)
            colored = np.repeat(idx[:, :, np.newaxis], 3, axis=2)
            result.append(colored)

        return result

    def _colorize_seg_frames(self) -> list[np.ndarray]:
        """Colorize segmentation frames using a deterministic palette.

        ``SemanticInstanceSegmentation`` pixels are arbitrary 128-bit instance
        IDs (stored as ``uint32``), not small contiguous labels. They are
        compactified to consecutive palette indices via :func:`numpy.unique`
        (sorted, so the mapping is deterministic) before indexing the fixed
        palette. Colorization happens at native resolution and the resulting
        RGB is resized with nearest-neighbor, so large IDs never overflow an
        ``int32`` resize buffer.
        """
        w, h = self._output_resolution()
        result: list[np.ndarray] = []
        for seg in self._seg_frames:
            colored = self._seg_to_color(seg)
            result.append(self._resize_rgb_nearest(colored, w, h))
        return result

    @staticmethod
    def _seg_to_color(seg: np.ndarray) -> np.ndarray:
        """Map a raw instance-id frame to native-resolution palette RGB.

        Instance IDs are compactified to consecutive palette slots via
        :func:`numpy.unique` (sorted, so deterministic): background id ``0``
        stays black, every other id gets a distinct non-zero slot cycling
        through ``1..255``.

        Args:
            seg: Raw ``SemanticInstanceSegmentation`` frame ``(H, W)`` or
                ``(H, W, 1)``.

        Returns:
            np.ndarray: ``(H, W, 3)`` ``uint8`` palette colorization at the
            input's native resolution.
        """
        seg2d = np.asarray(seg)
        if seg2d.ndim > 2:
            seg2d = seg2d[..., 0]
        seg2d = seg2d.astype(np.int64)

        unique_ids, inverse = np.unique(seg2d, return_inverse=True)
        palette_for_unique = np.zeros(unique_ids.size, dtype=np.int64)
        counter = 0
        for i, uid in enumerate(unique_ids):
            if uid == 0:
                continue
            palette_for_unique[i] = 1 + (counter % 255)
            counter += 1

        palette_idx = palette_for_unique[inverse].reshape(seg2d.shape)
        return _SEG_PALETTE[palette_idx]

    @staticmethod
    def _resize_rgb_nearest(
        rgb: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Nearest-neighbor resize of an ``(H, W, 3)`` uint8 image."""
        if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
            return rgb
        from PIL import Image

        return np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((target_w, target_h), Image.NEAREST)
        )

    def _compute_shaded_seg_frames(self) -> list[np.ndarray]:
        """Shade the colorized segmentation with surface normals (cached).

        Each instance keeps its palette color but is modulated by a Lambertian
        term computed from ``NormalSD``, giving a control signal that is both
        semantically labelled and geometrically shaded. The fused shading runs
        through a Warp kernel (GPU when available, else Warp CPU) for
        performance; see :func:`ovannotators.writers._gpu_shade.shade_segmentation`.

        Shading is performed at the segmentation's native resolution (where the
        palette boundaries are crisp), then the shaded RGB is resized to the
        output resolution. The result is cached so the
        ``"shaded_segmentation"`` conditioning video and a
        ``canny_source="shaded_segmentation"`` edge map share one computation.

        Returns:
            list[np.ndarray]: Per-frame ``(H, W, 3)`` ``uint8`` RGB arrays at
            the configured output resolution.
        """
        if self._shaded_seg_frames is not None:
            return self._shaded_seg_frames

        from ovannotators.writers._gpu_shade import shade_segmentation

        cfg = self._config
        w, h = self._output_resolution()
        result: list[np.ndarray] = []
        for seg, normals in zip(self._seg_frames, self._normal_frames):
            seg_color = self._seg_to_color(seg)
            n = np.asarray(normals)
            if n.shape[:2] != seg_color.shape[:2]:
                # Geometry should match (same render), but guard against a
                # mismatch by snapping normals to the segmentation grid.
                n = self._resize_normals(n, seg_color.shape[1], seg_color.shape[0])
            shaded = shade_segmentation(
                seg_color,
                n,
                light_dir=cfg.shaded_seg_light_dir,
                ambient=_SHADED_SEG_AMBIENT,
            )
            result.append(self._resize_frame(shaded, w, h))

        self._shaded_seg_frames = result
        return result

    @staticmethod
    def _resize_normals(
        normals: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Bilinearly resize an ``(H, W, C>=3)`` normal field, per channel."""
        from PIL import Image

        n = np.asarray(normals, dtype=np.float32)[..., :3]
        if n.shape[0] == target_h and n.shape[1] == target_w:
            return n
        channels = [
            np.asarray(
                Image.fromarray(n[..., c], mode="F").resize(
                    (target_w, target_h), Image.BILINEAR
                ),
                dtype=np.float32,
            )
            for c in range(3)
        ]
        return np.stack(channels, axis=-1)

    def _resize_depth(
        self, depth: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize depth map using Pillow bilinear interpolation."""
        from PIL import Image

        h, w = depth.shape[:2]
        if depth.ndim > 2:
            depth = depth[..., 0]
        if h == target_h and w == target_w:
            return depth

        img = Image.fromarray(depth.astype(np.float32), mode="F")
        resized = img.resize((target_w, target_h), Image.BILINEAR)
        return np.asarray(resized)

    def _encode_video(self, frames: list[np.ndarray], w: int, h: int) -> bytes:
        """Encode a list of RGB frames into H.264 MP4 bytes.

        Args:
            frames: Per-frame ``(H, W, 3)`` ``uint8`` RGB arrays.
            w: Output width in pixels.
            h: Output height in pixels.

        Returns:
            bytes: The encoded MP4 payload.
        """
        from ovannotators._internal.codecs import get_video_encoder

        cfg = self._config
        enc = get_video_encoder(
            path="",
            fps=cfg.fps,
            w=w,
            h=h,
        )
        for frame in frames:
            # Ensure uint8 RGB, correct shape
            f = frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
            if f.shape[:2] != (h, w):
                f = self._resize_frame(f, w, h)
            enc.write_frame(f)
        return enc.close()

    def _encode_jpeg(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """Encode a single frame as JPEG bytes."""
        from ovannotators._internal.codecs import get_image_codec

        codec = get_image_codec()
        return codec.encode(frame, "jpeg", quality=quality)
