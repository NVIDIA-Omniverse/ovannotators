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

import io
import json
import logging
import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import math

import numpy as np
import warp as wp

from ovannotators.writers._helpers import (
    OVRTX_CAMERA_PARAMS_DTYPE,
    _to_numpy,
    decode_bbox_2d,
    decode_bbox_3d,
    decode_camera_params,
    id_label_map_from_render_vars,
    id_prim_path_map_from_render_vars,
    iter_bboxes_2d,
    iter_bboxes_3d,
    validate_required_vars,
)
from ovannotators.writers._registry import WriterRegistry

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
    "GenericModelOutput": ("generic_model_output", "npz"),
}

_COLORIZABLE_VARS: frozenset[str] = frozenset(
    {
        "DistanceToCameraSD",
        "DistanceToImagePlaneSD",
        "NormalSD",
        "SemanticInstanceSegmentation",
        "SemanticSegmentation",
    }
)


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
        image_format: Image encoding for LdrColor and similar image AOVs.
        frame_id_digits: Zero-padding width for the frame directory name.
        colorize: When True, write colorized PNG visualizations alongside raw
            data for DistanceToCameraSD, NormalSD, SemanticInstanceSegmentation,
            and SemanticSegmentation.
    """

    render_vars: list[str] | None = None
    depth_format: Literal["npy", "exr"] = "exr"
    image_format: Literal["png", "jpg"] = "png"
    frame_id_digits: int = 4
    colorize: bool = True
    depth_near_bright: bool = True


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
    normals_xyz: wp.array3d(dtype=wp.float32),
    out_rgb: wp.array3d(dtype=wp.uint8),
):
    x, y = wp.tid()
    h = normals_xyz.shape[0]
    w = normals_xyz.shape[1]
    if x >= w or y >= h:
        return

    nx = normals_xyz[y, x, 0]
    ny = normals_xyz[y, x, 1]
    nz = normals_xyz[y, x, 2]
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


@wp.kernel
def _segmentation_to_rgb_kernel(
    compact_ids: wp.array2d(dtype=wp.int32),
    palette: wp.array2d(dtype=wp.uint8),
    out_rgb: wp.array3d(dtype=wp.uint8),
):
    x, y = wp.tid()
    h = compact_ids.shape[0]
    w = compact_ids.shape[1]
    if x >= w or y >= h:
        return

    cid = compact_ids[y, x]
    out_rgb[y, x, 0] = palette[cid, 0]
    out_rgb[y, x, 1] = palette[cid, 1]
    out_rgb[y, x, 2] = palette[cid, 2]


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


def _colorize_depth(arr: np.ndarray, near_bright: bool = True) -> np.ndarray:
    """Colorize depth using Warp kernels on CPU/GPU."""
    a = arr.astype(np.float32)
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
    return out_wp.numpy()


def _colorize_normals(arr: np.ndarray) -> np.ndarray:
    """Colorize normal vectors using Warp kernels on CPU/GPU."""
    xyz = arr.astype(np.float32)[..., :3]
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
    return out_wp.numpy()


def _colorize_segmentation(arr: np.ndarray) -> np.ndarray:
    """Colorize segmentation IDs using Warp kernels on CPU/GPU."""
    seg = arr.astype(np.int64)
    if seg.ndim > 2:
        seg = seg[..., 0]
    H, W = seg.shape
    unique_ids, inverse = np.unique(seg, return_inverse=True)
    palette = np.zeros((unique_ids.size, 3), dtype=np.uint8)
    for i, uid in enumerate(unique_ids):
        if uid == 0:
            continue
        rng = np.random.default_rng(int(uid) & 0xFFFFFFFF)
        color = rng.integers(64, 255, size=3, dtype=np.uint8)
        palette[i] = color

    compact = inverse.reshape(H, W).astype(np.int32)
    device = _warp_device()
    compact_wp = wp.array(compact, dtype=wp.int32, shape=(H, W), device=device)
    palette_wp = wp.array(
        palette, dtype=wp.uint8, shape=(palette.shape[0], 3), device=device
    )
    out_wp = wp.zeros((H, W, 3), dtype=wp.uint8, device=device)
    wp.launch(
        _segmentation_to_rgb_kernel,
        dim=(W, H),
        inputs=[compact_wp, palette_wp, out_wp],
        device=device,
    )
    return out_wp.numpy()


def _compute_colorized(
    var_name: str,
    arr: np.ndarray,
    depth_near_bright: bool = True,
) -> np.ndarray | None:
    if var_name in ("DistanceToCameraSD", "DistanceToImagePlaneSD"):
        return _colorize_depth(arr, near_bright=depth_near_bright)
    if var_name == "NormalSD":
        return _colorize_normals(arr)
    if var_name in ("SemanticInstanceSegmentation", "SemanticSegmentation"):
        return _colorize_segmentation(arr)
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
        └── generic_model_output.npz

    Unknown render-var names are stored as ``<name>.npy``.
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
            self._write_var(
                frame_dir,
                var_name,
                stem,
                enc,
                render_vars[var_name],
                backend,
                render_vars,
            )
            if (
                self.config.colorize
                and var_name in _COLORIZABLE_VARS
                and not _array_is_empty(render_vars[var_name])
            ):
                colorized = _compute_colorized(
                    var_name,
                    _to_numpy(render_vars[var_name]),
                    depth_near_bright=self.config.depth_near_bright,
                )
                if colorized is not None:
                    backend.write_array(
                        f"{frame_dir}/{stem}_colorized.png", colorized, encoding="png"
                    )

    def finalize(self, backend: Backend) -> None:
        """No-op; BasicWriter writes all data eagerly in :meth:`write_frame`.

        Args:
            backend (Backend): Unused; present for :class:`Writer` compatibility.
        """
        self._finalized = True

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
            backend.write_array(f"{frame_dir}/{stem}.exr", array, encoding="exr")
        elif enc == "depth":
            fmt = self.config.depth_format
            backend.write_array(f"{frame_dir}/{stem}.{fmt}", array, encoding=fmt)
        elif enc == "json":
            data = _json_payload(var_name, array, render_vars)
            payload = json.dumps(data, cls=_NumpyEncoder, indent=2).encode()
            backend.write(f"{frame_dir}/{stem}.json", payload)
        elif enc == "npz":
            array = _to_numpy(array)
            buf = io.BytesIO()
            if array.dtype.names:
                np.savez_compressed(buf, **{n: array[n] for n in array.dtype.names})
            else:
                np.savez_compressed(buf, data=array)
            backend.write(f"{frame_dir}/{stem}.npz", buf.getvalue())
        else:
            backend.write_array(f"{frame_dir}/{stem}.npy", array, encoding="npy")


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
