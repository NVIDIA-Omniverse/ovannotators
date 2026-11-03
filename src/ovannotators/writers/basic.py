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

"""BasicWriter — flat per-frame dump of all render vars."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import warp as wp

from ovannotators._internal.video_config import (
    VideoCodec,
    VideoPreset,
    validate_video_options,
    video_frame_rate,
)

from ovannotators.writers._helpers import (
    OVRTX_CAMERA_PARAMS_DTYPE,
    OVRTX_INSTANCE_MAP_DTYPE,
    _to_numpy,
    decode_bbox_2d,
    decode_bbox_3d,
    decode_camera_params,
    decode_id_string_map,
    decode_instance_map,
    decode_stable_id_semantic_id_map,
    id_label_map_from_render_vars,
    id_prim_path_map_from_render_vars,
    iter_bboxes_2d,
    iter_bboxes_3d,
    to_rgb_uint8,
    validate_required_vars,
)
from ovannotators.writers._registry import WriterRegistry
from ovannotators.writers._segmentation_palette import (
    ColorAccessibilityMode,
    SegmentationColorizer,
    SegmentationColorKey,
    validate_accessibility_mode,
)

if TYPE_CHECKING:
    from ovannotators.backends._base import Backend

logger = logging.getLogger(__name__)


def _array_is_empty(array: Any) -> bool:  # noqa: ANN401
    """Return True when an array (NumPy or Warp) has no elements.

    Uses the cheap ``shape`` attribute so GPU-resident Warp views are not copied
    to host just to test for emptiness.

    Args:
        array (Any): NumPy or Warp array (or any object exposing ``shape``).

    Returns:
        bool: True when the array has zero total elements.
    """
    shape = getattr(array, "shape", None)
    if shape is None:
        return False
    return int(np.prod(shape)) == 0


# ---------------------------------------------------------------------------
# Render-var → (filename stem, encoding) map
# ---------------------------------------------------------------------------

_VAR_FILE_MAP: dict[str, tuple[str, str]] = {
    # Core render vars (ovrtx naming convention: lowercase filenames)
    "LdrColor": ("ldr_color", "image"),
    "HdrColor": ("hdr_color", "exr"),
    "DistanceToCameraSD": ("distance_to_camera", "depth"),
    "DistanceToImagePlaneSD": ("distance_to_image_plane", "depth"),
    "DepthSD": ("depth", "depth"),
    "DepthLinearized": ("depth_linearized", "depth"),
    "NormalSD": ("normal", "exr"),
    "SmoothNormal": ("smooth_normal", "exr"),
    "BumpNormal": ("bump_normal", "exr"),
    "AmbientOcclusion": ("ambient_occlusion", "exr"),
    "DiffuseAlbedoSD": ("diffuse_albedo", "image"),
    "Camera3dPositionSD": ("camera_3d_position", "exr"),
    "TargetMotionSD": ("target_motion", "exr"),
    "SemanticInstanceSegmentation": ("instance_segmentation", "npy"),
    "NonStableInstanceSegmentation": ("non_stable_instance_segmentation", "npy"),
    "StableIdSegmentation": ("stable_id_segmentation", "npy"),
    "SemanticSegmentation": ("semantic_segmentation", "npy"),
    "SemanticIdMap": ("semantic_id_map", "json"),
    "StableIdMap": ("stable_id_map", "json"),
    "StableIdSemanticIdMap": ("stable_id_semantic_id_map", "json"),
    "InstanceMap": ("instance_map", "json"),
    "SemanticBoundingBox2DTight": ("bbox_2d_tight", "json"),
    "SemanticBoundingBox2DLoose": ("bbox_2d_loose", "json"),
    "SemanticBoundingBox3D": ("bbox_3d", "json"),
    "CameraParams": ("camera_params", "json"),
    "Motion2d": ("motion_2d", "exr"),
    "PointCloud": ("point_cloud", "npz"),
}

_SEGMENTATION_KEY_BY_VAR: dict[str, SegmentationColorKey] = {
    "SemanticInstanceSegmentation": "instance_id",
    "StableIdSegmentation": "stable_id",
    "SemanticSegmentation": "semantic_id",
}

# ovrtx 0.5 robot-scene readback: uint8 display images and float image tensors.
# Keep the sourceName spelling (including PtDirectIllumation) as the lookup key.
_ADDITIONAL_IMAGE_VARS = frozenset(
    "LdrColorSD SimpleShadingSD DiffuseAlbedo SpecularAlbedo Roughness "
    "DepthSensorPointCloudColor".split()
)
_ADDITIONAL_EXR_VARS = frozenset(
    "Depth HighResDepth XRDepth DepthSensorDistance DepthSensorImager "
    "DepthSensorPointCloudPosition BumpNormalResearch "
    "DiffuseLightingDirectAndIndirect DisocclusionMask DlssInput "
    "EmissionAndForegroundMask HighResDisocclusionMask HighResMotion2d "
    "HighResMotion2dXYZ HighResReflectionsMotion2d LightingComposited "
    "Motion2dDilated Motion2dXYZ Motion2dXYZDilated MotionWorld "
    "ReflectionsFirstBounceDiffuseAlbedo ReflectionsFirstBounceNormal "
    "ReflectionsFirstBounceRayDirAndHitT ReprojectionQuality "
    "SpecularLightingDirectAndIndirect ViewspacePosition "
    "PtBackground PtDiffuseFilter PtDirectIllumation PtGlobalIllumination "
    "PtIlluminance PtLuminance PtMotion PtPreDenoisedResult "
    "PtReflectionFilter PtReflections PtRefractionFilter PtRefractions "
    "PtSelfIllumination PtSubsurfaceFilter PtSubsurfaceScattering "
    "PtViewNormal PtVolumes PtWorldNormal PtWorldPos PtZDepth "
    "PtLpe0 PtLpe1 PtLpe2 PtLpe3 PtLpe4 PtLpe5 PtLpe6 PtLpe7 PtLpe8".split()
)
_ADDITIONAL_RAW_VARS = frozenset(
    "StableId Exposure ViewMatricesAux BoundingBox2DLooseSD "
    "BoundingBox2DTightSD BoundingBox3DSD InstanceIdTokenMapSD OcclusionSD "
    "SemanticIdMapDeltas StableIdMapDeltas TruncationSD "
    "Caustics CrossCorrespondenceSD DeveloperDebugTexture DirectDiffuse "
    "DirectSpecular DlssInputWithoutTransparency GBufferDepth "
    "IndirectDiffuse IndirectDiffuseFirstBounceNormal "
    "IndirectDiffuseFirstBounceRayDirAndHitT Irradiance Reflections "
    "ReflectionsFirstBounceWorldPos ReflectionsHitT ReflectionsMotion2d "
    "ReflectionsMotionWorld StableDepth SubsurfaceScattering "
    "PtMultiMatte0 PtMultiMatte1 PtMultiMatte2 PtMultiMatte3 "
    "PtMultiMatte4 PtMultiMatte5 PtMultiMatte6 PtMultiMatte7".split()
)


def _source_name_stem(name: str) -> str:
    """Convert a known ovrtx sourceName to a stable snake_case file stem."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


for _source_name in _ADDITIONAL_IMAGE_VARS:
    _VAR_FILE_MAP[_source_name] = (_source_name_stem(_source_name), "image")
for _source_name in _ADDITIONAL_EXR_VARS:
    _VAR_FILE_MAP[_source_name] = (_source_name_stem(_source_name), "exr")
for _source_name in _ADDITIONAL_RAW_VARS:
    _VAR_FILE_MAP[_source_name] = (_source_name_stem(_source_name), "npy")
# These renderer names coexist with existing SD writer stems in one frame.
_VAR_FILE_MAP["Depth"] = ("depth_native", "exr")
_VAR_FILE_MAP["DiffuseAlbedo"] = ("diffuse_albedo_native", "image")
_VAR_FILE_MAP["BoundingBox2DTightSD"] = ("bounding_box_2d_tight_sd", "npy")
_VAR_FILE_MAP["BoundingBox2DLooseSD"] = ("bounding_box_2d_loose_sd", "npy")
_VAR_FILE_MAP["BoundingBox3DSD"] = ("bounding_box_3d_sd", "npy")

_COLORIZABLE_VARS: frozenset[str] = frozenset(
    {
        "DistanceToCameraSD",
        "DistanceToImagePlaneSD",
        "NormalSD",
        "SemanticInstanceSegmentation",
        "StableIdSegmentation",
        "SemanticSegmentation",
    }
)

# Render vars that can be encoded as a video track: directly viewable RGB(A)
# images, plus every var that has a colorized RGB visualization (the colorized
# frame is what gets encoded for those).
_VIDEO_VARS: frozenset[str] = frozenset({"LdrColor", "DiffuseAlbedoSD"})
_VIDEO_CAPABLE_VARS: frozenset[str] = _VIDEO_VARS | _COLORIZABLE_VARS


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars/arrays/structured arrays and dataclasses."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        """Convert numpy and dataclass objects to JSON-serializable values.

        Args:
            obj (Any): Object the base encoder could not serialize (numpy
                scalar/array or a dataclass instance such as ``BBox2D``).

        Returns:
            Any: A JSON-serializable representation (list/dict/number/str).
        """
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, np.ndarray):
            if obj.dtype.names:
                if obj.ndim == 0:
                    return {n: self.default(obj[n]) for n in obj.dtype.names}
                return [
                    {n: self.default(row[n]) for n in obj.dtype.names} for row in obj
                ]
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bytes_, bytes)):
            return obj.decode("utf-8", errors="replace")
        return super().default(obj)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BasicWriterConfig:
    """Configuration for :class:`BasicWriter`.

    Attributes:
        render_vars: Render vars to write.  ``None`` writes all present vars.
        depth_format: Storage format for depth arrays.
        exr_compression: OpenEXR compression scheme for EXR outputs. ``None``
            uses OpenEXR's default ZIP compression. An unknown name raises
            ``ValueError`` when the config is created.
        image_format: Image encoding for LdrColor and similar image AOVs.
        frame_id_digits: Zero-padding width for the frame directory name.
        colorize: When True, write colorized PNG visualizations alongside raw
            data for DistanceToCameraSD, NormalSD, SemanticInstanceSegmentation,
            and SemanticSegmentation.
        video: When True, also accumulate the frames of every video-capable
            render var and write one MP4 per var (H.264 by default) on
            :meth:`BasicWriter.finalize`. Requires ``PyNvVideoCodec``.
        video_fps: Frame rate of the written MP4 files; fractional rates
            such as 29.97 are supported.
        video_vars: Render vars to encode as video.  ``None`` encodes every
            video-capable var present (``LdrColor``, ``DiffuseAlbedoSD``, and
            the colorized depth/normal/segmentation visualizations).  Names not
            in that set are ignored.
        video_dir: Directory (relative to the backend root) the MP4 files are
            written to.  An empty string writes them at the root.
        video_codec: ``"h264"`` (default) or ``"hevc"``.
        video_lossless: Preserve normalized uint8 RGB pixels exactly with
            HEVC 4:4:4. Requires even, fixed dimensions; alpha is not retained.
        video_preset: Optional NVENC ``P1`` through ``P7`` preset. None keeps
            the encoder default, or uses P5 for lossless HEVC.
        segmentation_palette_accessibility: Optional palette accessibility
            mode: ``"none"``, ``"protanopia"``, ``"deuteranopia"``, or
            ``"tritanopia"``.
        semantic_color_key: Color semantic segmentation by numeric ID or label.
        segmentation_palette_path: Optional JSON file used to persist typed
            segmentation color assignments across writer sessions.
    """

    render_vars: list[str] | None = None
    depth_format: Literal["npy", "exr"] = "exr"
    image_format: Literal["png", "jpg"] = "png"
    frame_id_digits: int = 4
    colorize: bool = True
    depth_near_bright: bool = True
    video: bool = False
    video_fps: float = 24
    video_vars: list[str] | None = None
    video_dir: str = "videos"
    segmentation_palette_accessibility: ColorAccessibilityMode = "none"
    semantic_color_key: Literal["semantic_id", "semantic_label"] = "semantic_id"
    segmentation_palette_path: str | Path | None = None
    exr_compression: str | None = None
    video_codec: VideoCodec = "h264"
    video_lossless: bool = False
    video_preset: VideoPreset | None = None

    def __post_init__(self) -> None:
        """Validate palette options, semantic key selection and ``exr_compression``."""
        from ovannotators._internal.codecs import _check_exr_compression

        validate_accessibility_mode(self.segmentation_palette_accessibility)
        validate_video_options(self.video_codec, self.video_lossless, self.video_preset)
        if self.video:
            video_frame_rate(self.video_fps, "video_fps")
        if self.semantic_color_key not in {"semantic_id", "semantic_label"}:
            raise ValueError(
                "semantic_color_key must be 'semantic_id' or 'semantic_label', got "
                f"{self.semantic_color_key!r}"
            )
        _check_exr_compression(self.exr_compression)


# ---------------------------------------------------------------------------
# Colorization helpers
# ---------------------------------------------------------------------------


wp.config.verbose = False
wp.init()


@wp.kernel
def _apply_lut_kernel(
    norm: wp.array2d(dtype=wp.float32),
    valid_mask: wp.array2d(dtype=wp.uint8),
    lut: wp.array2d(dtype=wp.float32),
    lut_size: int,
    out_rgb: wp.array3d(dtype=wp.uint8),
):
    x, y = wp.tid()
    h = norm.shape[0]
    w = norm.shape[1]
    if x >= w or y >= h:
        return
    if valid_mask[y, x] == wp.uint8(0):
        out_rgb[y, x, 0] = wp.uint8(0)
        out_rgb[y, x, 1] = wp.uint8(0)
        out_rgb[y, x, 2] = wp.uint8(0)
        return

    n = wp.clamp(norm[y, x], 0.0, 1.0)
    scaled = n * float(lut_size - 1)
    i0 = int(wp.floor(scaled))
    i1 = wp.min(i0 + 1, lut_size - 1)
    t = scaled - float(i0)

    c0 = lut[i0, 0] * (1.0 - t) + lut[i1, 0] * t
    c1 = lut[i0, 1] * (1.0 - t) + lut[i1, 1] * t
    c2 = lut[i0, 2] * (1.0 - t) + lut[i1, 2] * t
    out_rgb[y, x, 0] = wp.uint8(wp.clamp(c0, 0.0, 255.0))
    out_rgb[y, x, 1] = wp.uint8(wp.clamp(c1, 0.0, 255.0))
    out_rgb[y, x, 2] = wp.uint8(wp.clamp(c2, 0.0, 255.0))


@wp.kernel
def _normals_to_rgb_kernel(
    normals_xyz: wp.array3d(dtype=Any),
    out_rgb: wp.array3d(dtype=wp.uint8),
):
    x, y = wp.tid()
    h = normals_xyz.shape[0]
    w = normals_xyz.shape[1]
    if x >= w or y >= h:
        return

    nx = float(normals_xyz[y, x, 0])
    ny = float(normals_xyz[y, x, 1])
    nz = float(normals_xyz[y, x, 2])
    if nx == 0.0 and ny == 0.0 and nz == 0.0:
        out_rgb[y, x, 0] = wp.uint8(0)
        out_rgb[y, x, 1] = wp.uint8(0)
        out_rgb[y, x, 2] = wp.uint8(0)
        return

    rx = wp.clamp(0.5 + nx * 0.5, 0.0, 1.0) * 255.0
    ry = wp.clamp(0.5 + ny * 0.5, 0.0, 1.0) * 255.0
    rz = wp.clamp(0.5 + nz * 0.5, 0.0, 1.0) * 255.0
    out_rgb[y, x, 0] = wp.uint8(rx)
    out_rgb[y, x, 1] = wp.uint8(ry)
    out_rgb[y, x, 2] = wp.uint8(rz)


def _warp_device() -> wp.context.Device:
    """Return the current default Warp device (CUDA when present, else CPU).

    Respects :func:`warp.ScopedDevice`/:func:`warp.set_device`, so multi-GPU
    callers can target a specific device without this module hard-coding one.
    """
    return wp.get_device()


@lru_cache(maxsize=1)
def _turbo_lut() -> np.ndarray:
    """Build a float32 RGB LUT in [0, 255] for depth colorization."""
    try:
        import matplotlib.cm as _cm  # type: ignore[import-untyped]

        samples = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
        if hasattr(_cm, "colormaps"):
            cmap = _cm.colormaps["turbo"]
        else:
            cmap = _cm.get_cmap("turbo")
        rgb = (cmap(samples)[..., :3] * 255.0).astype(np.float32)
        return rgb
    except ImportError:
        grey = np.linspace(0.0, 255.0, 4096, dtype=np.float32)
        return np.stack([grey, grey, grey], axis=1)


def _colorize_depth(
    arr: np.ndarray, near_bright: bool = True, keep_device: bool = False
) -> Any:
    """Colorize depth using Warp kernels on CPU/GPU."""
    a = arr.astype(np.float32, copy=False)
    if a.ndim == 3:
        a = a[..., 0]
    inf_mask = ~np.isfinite(a)
    finite_vals = a[~inf_mask]
    if finite_vals.size == 0:
        return np.zeros((*a.shape, 3), dtype=np.uint8)
    lo, hi = float(finite_vals.min()), float(finite_vals.max())
    norm = np.where(inf_mask, 0.0, np.clip((a - lo) / max(hi - lo, 1e-8), 0.0, 1.0))
    if near_bright:
        norm = 1.0 - norm
    h, w = norm.shape
    device = _warp_device()
    lut = _turbo_lut()
    norm_wp = wp.array(norm, dtype=wp.float32, shape=(h, w), device=device)
    valid_wp = wp.array((~inf_mask).astype(np.uint8), dtype=wp.uint8, device=device)
    lut_wp = wp.array(lut, dtype=wp.float32, shape=lut.shape, device=device)
    out_wp = wp.zeros((h, w, 3), dtype=wp.uint8, device=device)
    wp.launch(
        _apply_lut_kernel,
        dim=(w, h),
        inputs=[norm_wp, valid_wp, lut_wp, lut.shape[0], out_wp],
        device=device,
    )
    return out_wp if keep_device and device.is_cuda else out_wp.numpy()


def _colorize_normals(arr: Any, keep_device: bool = False) -> Any:
    """Colorize normal vectors using Warp kernels on CPU/GPU."""
    from ovannotators._internal.video_frames import (
        as_scalar_channels,
        as_warp,
        is_device_frame,
    )

    if is_device_frame(arr):
        xyz_wp = as_scalar_channels(as_warp(arr))
        if xyz_wp.ndim != 3 or xyz_wp.shape[2] < 3:
            raise ValueError(
                f"NormalSD must have shape (H, W, C>=3), got {tuple(xyz_wp.shape)}"
            )
        device = xyz_wp.device
        wp.synchronize_device(device)
        h, w = xyz_wp.shape[:2]
    else:
        xyz = _to_numpy(arr)[..., :3].astype(np.float32, copy=False)
        h, w, _ = xyz.shape
        device = _warp_device()
        xyz_wp = wp.array(xyz, dtype=wp.float32, shape=(h, w, 3), device=device)
    out_wp = wp.zeros((h, w, 3), dtype=wp.uint8, device=device)
    wp.launch(
        _normals_to_rgb_kernel,
        dim=(w, h),
        inputs=[xyz_wp, out_wp],
        device=device,
    )
    return out_wp if keep_device and device.is_cuda else out_wp.numpy()


def _crop_to_even(frame: np.ndarray) -> np.ndarray:
    """Crop a frame to even width and height for H.264 4:2:0 chroma subsampling.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` ``uint8`` RGB frame.

    Returns:
        np.ndarray: The frame, cropped by at most one row/column so both
        dimensions are even.
    """
    height, width = frame.shape[:2]
    if height % 2 == 0 and width % 2 == 0:
        return frame
    return np.ascontiguousarray(frame[: height - (height % 2), : width - (width % 2)])


def _compute_colorized(
    var_name: str,
    arr: np.ndarray,
    depth_near_bright: bool = True,
    keep_device: bool = False,
) -> Any:
    if var_name in ("DistanceToCameraSD", "DistanceToImagePlaneSD"):
        return _colorize_depth(
            _to_numpy(arr), near_bright=depth_near_bright, keep_device=keep_device
        )
    if var_name == "NormalSD":
        return _colorize_normals(arr, keep_device=keep_device)
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@WriterRegistry.register
class BasicWriter:
    """Writes all present render vars per frame in a flat layout.

    Output structure (colorize=True)::

        {output_dir}/{frame_id:04d}/
        ├── ldr_color.png
        ├── distance_to_camera.exr
        ├── distance_to_camera_colorized.png
        ├── normal.exr
        ├── normal_colorized.png
        ├── instance_segmentation.npy
        ├── instance_segmentation_colorized.png
        ├── stable_id_segmentation.npy
        ├── stable_id_segmentation_colorized.png
        ├── semantic_segmentation.npy
        ├── semantic_segmentation_colorized.png
        ├── semantic_id_map.json
        ├── stable_id_map.json
        ├── stable_id_semantic_id_map.json
        ├── instance_map.json
        ├── bbox_2d_tight.json
        ├── bbox_3d.json
        ├── camera_params.json
        ├── motion_2d.exr
        ├── target_motion.exr
        └── point_cloud.npz

    Named ovrtx 0.5 AOVs are also written with explicit filenames: six uint8
    image outputs as PNG/JPEG, 55 float image outputs (``Pt*`` lighting passes,
    motion, positions, depth) as EXR, and 37 integer, packed-record, and
    auxiliary buffers as NPY (for example ``pt_world_pos.exr`` and
    ``stable_id.npy``). EXR keeps float16, float32, and uint32 data without
    narrowing. The "ovrtx render outputs" documentation page lists every name.
    Unknown render-var names are stored as the lowercased name with a ``.npy``
    extension.

    With ``video=True`` the per-frame files above are still written, and each
    video-capable render var is additionally encoded into one H.264 MP4 spanning
    every frame the writer saw::

        {output_dir}/videos/
        ├── ldr_color.mp4
        ├── distance_to_camera_colorized.mp4
        ├── normal_colorized.mp4
        ├── stable_id_segmentation_colorized.mp4
        └── semantic_segmentation_colorized.mp4
    """

    name: ClassVar[str] = "basic"
    required_render_vars: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: BasicWriterConfig | None = None) -> None:
        """Initialize the writer.

        Args:
            config (BasicWriterConfig | None): Output options; a default
                ``BasicWriterConfig`` is used when omitted.
        """
        self.config: BasicWriterConfig = config or BasicWriterConfig()
        self._finalized: bool = False
        # Video streams are opened lazily on the first frame of each var and
        # kept open until finalize(), so raw frames are never buffered: only the
        # encoder's compressed output accumulates in memory.
        self._video_encoders: dict[str, Any] = {}
        self._video_sizes: dict[str, tuple[int, int]] = {}
        self._seg_colorizer = SegmentationColorizer(
            accessibility=self.config.segmentation_palette_accessibility,
            path=self.config.segmentation_palette_path,
        )

    def write_frame(
        self,
        frame_id: int,
        render_vars: dict[str, np.ndarray],
        backend: Backend,
    ) -> None:
        """Write one per-frame directory containing the configured render vars.

        ``BasicWriter`` has no required render vars, so it writes whatever subset
        of vars the frame provides (intersected with ``config.render_vars`` when
        set).

        Args:
            frame_id (int): Zero-based frame index; names the frame directory.
            render_vars (dict[str, np.ndarray]): Frame render vars (NumPy or
                renderer-agnostic Warp arrays, e.g. zero-copy ovrtx views).
            backend (Backend): Destination backend for the frame's files.
        """
        from ovannotators._internal.video_frames import as_warp, is_device_frame

        validate_required_vars(
            self.name, self.required_render_vars, render_vars, frame_id
        )
        # Renderer-agnostic arrays (e.g. zero-copy Warp views from OvrtxOutput)
        # are passed straight through to the backend/codec, which keeps GPU
        # arrays on-device for zero-copy image encoding. Conversion to NumPy is
        # deferred to the few paths that genuinely require it (JSON/NPZ
        # serialization and colorization).
        frame_dir = f"{frame_id:0{self.config.frame_id_digits}d}"
        backend.makedirs(frame_dir)

        vars_to_write = (
            self.config.render_vars
            if self.config.render_vars is not None
            else list(render_vars.keys())
        )

        for var_name in vars_to_write:
            if var_name not in render_vars:
                continue
            stem, enc = _VAR_FILE_MAP.get(
                var_name, (var_name.lower().replace(" ", "_"), "npy")
            )
            value = render_vars[var_name]
            if is_device_frame(value):
                # Alias protocol-only inputs before host serialization as well
                # as video preparation; np.asarray cannot import CUDA DLPack.
                value = as_warp(value)
            wants_video = self._wants_video(var_name)
            if wants_video and self.config.video_lossless and _array_is_empty(value):
                raise ValueError("Lossless video requires positive frame dimensions")
            # The colorized RGB frame is both the PNG visualization and the
            # video frame, so compute it whenever either output needs it.
            colorize = var_name in _COLORIZABLE_VARS and (
                self.config.colorize or wants_video
            )
            if colorize:
                # Colorization needs host data and these vars are stored as
                # npy/exr, which copies to host anyway: read a GPU array back
                # once here rather than once per consumer.
                value = _to_numpy(value)
            self._write_var(
                frame_dir,
                var_name,
                stem,
                enc,
                value,
                backend,
                render_vars,
            )
            if _array_is_empty(value):
                continue

            if var_name in _COLORIZABLE_VARS:
                if not colorize:
                    continue
                if var_name in _SEGMENTATION_KEY_BY_VAR:
                    key_mode = self._segmentation_key(var_name)
                    seg = self._seg_colorizer.prepare_frame(
                        value,
                        render_vars,
                        key_mode,
                        writer_name=self.name,
                        frame_id=frame_id,
                    )
                    colorized = self._seg_colorizer.colorize(seg, key_mode)
                else:
                    # NormalSD stays on the device for video when the source is
                    # a CUDA frame; depth colorization needs host data.
                    on_device = (
                        var_name == "NormalSD"
                        and wants_video
                        and is_device_frame(render_vars[var_name])
                    )
                    colorized = _compute_colorized(
                        var_name,
                        render_vars[var_name] if on_device else value,
                        depth_near_bright=self.config.depth_near_bright,
                        keep_device=on_device,
                    )
                if colorized is None:  # pragma: no cover
                    continue
                if self.config.colorize:
                    backend.write_array(
                        f"{frame_dir}/{stem}_colorized.png", colorized, encoding="png"
                    )
                if wants_video:
                    self._append_video_frame(
                        f"{stem}_colorized", colorized, borrowed=False
                    )
            elif wants_video:
                self._append_video_frame(stem, value)

    def finalize(self, backend: Backend) -> None:
        """Close any open video streams and write their MP4 files.

        Per-frame data is written eagerly in :meth:`write_frame`, so this is a
        no-op unless ``config.video`` is enabled.  Repeated calls are no-ops.

        Every video stream is attempted and the segmentation palette is saved
        even if a stream fails. A failed stream cannot be retried, because its
        frames were encoded as they arrived rather than buffered.

        Args:
            backend (Backend): Destination backend for the MP4 files.

        Raises:
            RuntimeError: If any video stream failed to encode or write.
        """
        if self._finalized:
            return
        self._finalized = True
        try:
            self._flush_videos(backend)
        finally:
            self._seg_colorizer.save()

    def _segmentation_key(self, var_name: str) -> SegmentationColorKey:
        """Return the natural typed color key for a segmentation render var."""
        if var_name == "SemanticSegmentation":
            return self.config.semantic_color_key
        return _SEGMENTATION_KEY_BY_VAR[var_name]

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    def _wants_video(self, var_name: str) -> bool:
        """Return True when *var_name* should be accumulated into a video.

        Args:
            var_name (str): ovrtx render-var name.

        Returns:
            bool: True when video output is enabled, the var can be encoded as
            video, and it is selected by ``config.video_vars``.
        """
        if not self.config.video or var_name not in _VIDEO_CAPABLE_VARS:
            return False
        selected = self.config.video_vars
        return selected is None or var_name in selected

    def _append_video_frame(
        self, stream: str, array: Any, *, borrowed: bool = True
    ) -> None:
        """Encode one frame into the named video stream, opening it if needed.

        The stream's resolution is fixed by its first frame; later frames of a
        different resolution are skipped with a warning rather than corrupting
        the stream (a renderer can change resolution mid-capture).

        Args:
            stream (str): Output stem, e.g. ``"ldr_color"``.
            array (Any): Frame image array (NumPy or Warp).
            borrowed (bool): Whether a CUDA frame may still be in production
                on another stream. Pass ``False`` for frames this writer just
                produced on the current Warp stream to skip the device sync.
        """
        from ovannotators._internal.video_frames import (
            as_warp,
            device_rgb,
            is_device_frame,
        )

        lossless = self.config.video_lossless
        if not is_device_frame(array):
            frame = to_rgb_uint8(array)
            if not lossless:
                frame = _crop_to_even(frame)
            height, width = frame.shape[:2]
        elif lossless:
            # NVENC's planar packing needs normalized uint8 RGB first.
            frame = device_rgb(array, synchronize=borrowed)
            height, width = frame.shape[:2]
        else:
            # The encoder normalizes, crops and packs BGRA in one kernel.
            frame = as_warp(array)
            if borrowed and frame.device.is_cuda:
                wp.synchronize_device(frame.device)
            height, width = frame.shape[:2]
            height, width = height - height % 2, width - width % 2
        encoder = self._video_encoders.get(stream)
        if encoder is None:
            if width == 0 or height == 0:
                if self.config.video_lossless:
                    raise ValueError(
                        "Lossless video requires positive frame dimensions"
                    )
                logger.warning(
                    "Skipping video '%s': first frame has zero-size dimensions %s.",
                    stream,
                    (height, width),
                )
                return
            from ovannotators._internal.codecs import get_video_encoder

            encoder = get_video_encoder(
                path="",
                fps=self.config.video_fps,
                w=width,
                h=height,
                codec=self.config.video_codec,
                lossless=self.config.video_lossless,
                preset=self.config.video_preset,
            )
            self._video_encoders[stream] = encoder
            self._video_sizes[stream] = (height, width)
        elif self._video_sizes[stream] != (height, width):
            if self.config.video_lossless:
                raise ValueError("Lossless video requires a fixed frame resolution")
            logger.warning(
                "Skipping a frame of video '%s': resolution %s does not match the "
                "stream's %s.",
                stream,
                (height, width),
                self._video_sizes[stream],
            )
            return
        encoder.write_frame(frame)

    def _flush_videos(self, backend: Backend) -> None:
        """Close every open video stream and write the encoded MP4 files.

        Every stream is closed and written even when another fails, so one
        encoder error cannot discard the remaining videos. Encoders stream
        frames as they arrive, so a failed stream cannot be re-encoded.

        Args:
            backend (Backend): Destination backend for the MP4 files.

        Raises:
            RuntimeError: If any stream failed to close or write; the original
                error of the first failed stream is chained as its cause.
        """
        if not self._video_encoders:
            return
        video_dir = self.config.video_dir.strip("/")
        if video_dir:
            backend.makedirs(video_dir)
        failed: dict[str, BaseException] = {}
        for stream, encoder in self._video_encoders.items():
            try:
                payload = encoder.close()
            except Exception as exc:
                failed[stream] = exc
                continue
            if not payload:
                logger.warning(
                    "Video '%s' produced no encoded data; no file written.", stream
                )
                continue
            path = f"{video_dir}/{stream}.mp4" if video_dir else f"{stream}.mp4"
            try:
                backend.write(path, payload)
            except Exception as exc:
                failed[stream] = exc
        self._video_encoders.clear()
        self._video_sizes.clear()
        if failed:
            raise RuntimeError(
                f"Failed to write video streams: {', '.join(failed)}"
            ) from next(iter(failed.values()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_var(
        self,
        frame_dir: str,
        var_name: str,
        stem: str,
        enc: str,
        array: np.ndarray,
        backend: Backend,
        render_vars: dict[str, np.ndarray],
    ) -> None:
        # Image/EXR/depth/NPY arrays (NumPy or Warp) are passed straight to the
        # backend, which converts or keeps them on-device as appropriate. Only
        # the JSON/NPZ paths require host NumPy here.
        #
        # Skip render vars the renderer emitted with no data (zero-size
        # buffers): there is nothing to encode and image/EXR codecs cannot
        # represent an empty array. JSON/NPZ still run so empty annotators
        # (e.g. a frame with zero bounding boxes) serialize an empty payload.
        if enc not in ("json", "npz") and _array_is_empty(array):
            logger.debug(
                "Skipping render var '%s': empty buffer (shape %s).",
                var_name,
                getattr(array, "shape", None),
            )
            return
        if enc == "image":
            fmt = self.config.image_format
            backend.makedirs(frame_dir)
            backend.write_array(f"{frame_dir}/{stem}.{fmt}", array, encoding=fmt)
        elif enc == "exr":
            kwargs = (
                {"exr_compression": self.config.exr_compression}
                if self.config.exr_compression is not None
                else {}
            )
            backend.write_array(
                f"{frame_dir}/{stem}.exr", array, encoding="exr", **kwargs
            )
        elif enc == "depth":
            fmt = self.config.depth_format
            kwargs = (
                {"exr_compression": self.config.exr_compression}
                if fmt == "exr" and self.config.exr_compression is not None
                else {}
            )
            backend.write_array(
                f"{frame_dir}/{stem}.{fmt}", array, encoding=fmt, **kwargs
            )
        elif enc == "json":
            data = _json_payload(var_name, array, render_vars)
            payload = json.dumps(data, cls=_NumpyEncoder, indent=2).encode()
            backend.write(f"{frame_dir}/{stem}.json", payload)
        elif enc == "npz":
            backend.write(f"{frame_dir}/{stem}.npz", _npz_bytes(array))
        else:
            backend.write_array(f"{frame_dir}/{stem}.npy", array, encoding="npy")


def _npz_bytes(array: Any) -> bytes:
    """Serialize a render var into an in-memory ``.npz`` archive.

    Composite sensor outputs (``PointCloud``) arrive as the ``{"tensors": ...,
    "params": ...}`` mapping :class:`~ovannotators.renderer_outputs.ovrtx.OvrtxOutput`
    builds, so each named tensor and param becomes one archive member. Param
    names are prefixed with ``param_`` so a tensor and a param that share a
    name cannot collide in the flat archive namespace. A structured array is
    split into one member per field, and any other array is stored as ``data``.

    Args:
        array: A render var value: the composite mapping, a structured array,
            or a plain array.

    Returns:
        bytes: The compressed ``.npz`` archive.
    """
    if isinstance(array, dict):
        members = {
            name: _to_numpy(value)
            for name, value in (array.get("tensors") or {}).items()
        }
        members.update(
            {
                f"param_{name}": _to_numpy(value)
                for name, value in (array.get("params") or {}).items()
            }
        )
    else:
        as_numpy = _to_numpy(array)
        members = (
            {name: as_numpy[name] for name in as_numpy.dtype.names}
            if as_numpy.dtype.names
            else {"data": as_numpy}
        )
    buf = io.BytesIO()
    np.savez_compressed(buf, **members)
    return buf.getvalue()


def _is_camera_params_array(array: np.ndarray) -> bool:
    """Return True if *array* looks like an OVRTX_CAMERA_PARAMS_DTYPE struct."""
    return isinstance(array, np.ndarray) and array.dtype == OVRTX_CAMERA_PARAMS_DTYPE


def _format_camera_params(array: np.ndarray) -> dict[str, Any]:
    """Format a ``CameraParams`` structured array into a well-structured dict.

    Outputs all raw fields from the ovrtx payload in snake_case, plus derived
    pinhole intrinsics and world-to-camera extrinsics for direct downstream
    consumption.  Matrices are stored as flattened lists (column-major) matching
    the ovrtx/USD convention.

    The output mirrors the ovrtx payload semantics:

    - ``view_transform``: 4×4 camera-to-world matrix (flattened, column-major).
    - ``projection``: 4×4 GL/NDC projection matrix (flattened, column-major).
    - ``camera_model``: projection model string (e.g. ``"pinhole"``).
    - Derived ``intrinsics`` block with ``fx``, ``fy``, ``cx``, ``cy`` and
      ``K`` (3×3 flattened row-major) for immediate use by ML pipelines.
    - Derived ``extrinsics`` block with ``R_world_to_camera`` (3×3 flattened
      row-major) and ``t_world_to_camera`` (3-element list).
    """
    arr = np.asarray(array)
    scalar = arr.reshape(-1)[0] if arr.ndim > 0 else arr

    # --- Raw fields (snake_case, matching OVRTX_CAMERA_PARAMS_DTYPE) ---
    view_transform = np.asarray(scalar["view_transform"], dtype=np.float64).reshape(
        4, 4
    )
    projection = np.asarray(scalar["projection"], dtype=np.float64).reshape(4, 4)
    resolution = np.asarray(scalar["resolution"]).reshape(2)
    width, height = int(resolution[0]), int(resolution[1])
    clip = np.asarray(scalar["clipping_plane"]).reshape(2)
    near, far = float(clip[0]), float(clip[1])
    fov_x = float(scalar["fov_x"])
    focal_length = float(scalar["focal_length"])
    focal_distance = float(scalar["focal_distance"])
    f_stop = float(scalar["f_stop"])
    meters_per_scene_unit = float(scalar["meters_per_scene_unit"])

    # Projection model (strip NUL padding from fixed-width bytes field).
    raw_model = scalar["projection_model"]
    camera_model = (
        raw_model.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(raw_model, (bytes, np.bytes_))
        else str(raw_model)
    )

    # Fisheye / distortion fields
    fisheye_nominal_width = float(scalar["fisheye_nominal_width"])
    fisheye_nominal_height = float(scalar["fisheye_nominal_height"])
    fisheye_optical_centre = np.asarray(scalar["fisheye_optical_centre"]).tolist()
    fisheye_max_fov = float(scalar["fisheye_max_fov"])
    fisheye_polynomial = np.asarray(scalar["fisheye_polynomial"]).tolist()
    lens_p = np.asarray(scalar["lens_p"]).tolist()
    lens_s = np.asarray(scalar["lens_s"]).tolist()

    # --- Derived intrinsics (pinhole, square pixels, centred principal point) ---
    fx = (
        (width / 2.0) / math.tan(fov_x / 2.0)
        if fov_x > 0 and width > 0
        else float(width)
    )
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]  # row-major 3x3

    # --- Derived extrinsics (world-to-camera) ---
    try:
        world_to_cam = np.linalg.inv(view_transform)
    except np.linalg.LinAlgError:
        world_to_cam = np.eye(4, dtype=np.float64)

    R_w2c = world_to_cam[:3, :3].ravel().tolist()  # row-major 3x3
    t_w2c = world_to_cam[:3, 3].tolist()

    return {
        # Raw payload fields
        "view_transform": view_transform.ravel(order="F").tolist(),  # column-major 16
        "projection": projection.ravel(order="F").tolist(),  # column-major 16
        "camera_model": camera_model,
        "focal_length": focal_length,
        "focal_distance": focal_distance,
        "f_stop": f_stop,
        "fov_x": fov_x,
        "meters_per_scene_unit": meters_per_scene_unit,
        "resolution": [width, height],
        "near_far": [near, far],
        # Fisheye / distortion
        "fisheye_nominal_width": fisheye_nominal_width,
        "fisheye_nominal_height": fisheye_nominal_height,
        "fisheye_optical_centre": fisheye_optical_centre,
        "fisheye_max_fov": fisheye_max_fov,
        "fisheye_polynomial": fisheye_polynomial,
        "lens_p": lens_p,
        "lens_s": lens_s,
        # Derived pinhole intrinsics
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "K": K,
        },
        # Derived extrinsics (world-to-camera)
        "extrinsics": {
            "R_world_to_camera": R_w2c,
            "t_world_to_camera": t_w2c,
        },
    }


def _json_payload(
    var_name: str,
    array: np.ndarray,
    render_vars: dict[str, np.ndarray],
) -> Any:
    """Build the JSON-serializable payload for a ``json``-encoded render var.

    Bounding-box and camera vars arrive as raw ovrtx byte buffers; they are
    decoded into structured arrays and expanded into per-entry records here.
    Labels come from ``SemanticIdMap`` and prim paths from ``StableIdMap``.

    Args:
        var_name (str): ovrtx render-var name being written.
        array (np.ndarray): Raw render-var buffer (NumPy or Warp array).
        render_vars (dict[str, np.ndarray]): All render vars for this frame,
            used to resolve the label and prim-path id maps.

    Returns:
        Any: A JSON-serializable structure (list of per-entry dicts for bbox
        vars, a dict for ``CameraParams``, otherwise a generic array dump).
    """
    if var_name in ("SemanticBoundingBox2DTight", "SemanticBoundingBox2DLoose"):
        id_to_label = id_label_map_from_render_vars(render_vars)
        id_to_prim_path = id_prim_path_map_from_render_vars(render_vars)
        decoded = decode_bbox_2d(array)
        return list(iter_bboxes_2d(decoded, id_to_label, id_to_prim_path))
    if var_name == "SemanticBoundingBox3D":
        id_to_label = id_label_map_from_render_vars(render_vars)
        id_to_prim_path = id_prim_path_map_from_render_vars(render_vars)
        decoded = decode_bbox_3d(array)
        return list(iter_bboxes_3d(decoded, id_to_label, id_to_prim_path))
    if var_name == "CameraParams":
        if _is_camera_params_array(array):
            return _format_camera_params(array)
        return _format_camera_params(_to_numpy(decode_camera_params(array)))
    if var_name in ("SemanticIdMap", "StableIdMap"):
        # Already-decoded dicts pass through; raw ovrtx buffers (e.g. from
        # extract(decode=False)) are decoded here rather than dumped as bytes.
        if isinstance(array, dict):
            return array
        return decode_id_string_map(array)
    if var_name == "StableIdSemanticIdMap":
        return _to_json_serializable(_to_numpy(decode_stable_id_semantic_id_map(array)))
    if var_name == "InstanceMap":
        decoded = _to_numpy(array)
        # decode_instance_map is not idempotent: re-decoding a decoded table
        # reinterprets the (index, value) pairs as a flat uint32 stream, so
        # guard on dtype the way the bbox decoders do.
        if decoded.dtype != OVRTX_INSTANCE_MAP_DTYPE:
            decoded = _to_numpy(decode_instance_map(decoded))
        return _to_json_serializable(decoded)
    if isinstance(array, (dict, list)):
        return array
    return _to_json_serializable(_to_numpy(array))


def _to_json_serializable(array: np.ndarray) -> Any:
    """Recursively convert a numpy array to a JSON-serialisable structure."""
    if array.dtype.names:
        if array.ndim == 0:
            return {n: _to_json_serializable(array[n]) for n in array.dtype.names}
        return [
            {n: _to_json_serializable(row[n]) for n in array.dtype.names}
            for row in array
        ]
    return array.tolist()
