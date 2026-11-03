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

"""Shared helpers for writers: bbox/camera extraction, RLE encoding, etc."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterator, Sequence

import numpy as np
import warp as wp

from ovannotators.exceptions import MissingRenderVarError
from ovannotators.types import (
    BBox2D,
    BBox3D,
    CameraParams,
    PointCloudResult,
    Resolution,
)

try:  # Warp is required; initialise once so kernels can launch on CPU or CUDA.
    wp.config.verbose = False
    wp.init()
except Exception:  # pragma: no cover - init is idempotent / already done elsewhere
    pass


# ---------------------------------------------------------------------------
# Per-frame validation
# ---------------------------------------------------------------------------


def validate_required_vars(
    writer_name: str,
    required: frozenset[str] | set[str] | Sequence[str],
    render_vars: dict[str, Any],
    frame_id: int | None = None,
) -> None:
    """Ensure every required render var is present in a frame.

    Centralizes the per-frame validation contract shared by all writers (see
    :class:`ovannotators.writers._base.Writer`), so the check and its error
    message stay consistent across implementations.

    Args:
        writer_name (str): ``name`` of the calling writer, used in the error.
        required (frozenset[str] | set[str] | Sequence[str]): Render-var names
            the frame must provide. May be empty, in which case this is a no-op.
        render_vars (dict[str, Any]): The frame's render vars, keyed by name.
        frame_id (int | None): Frame index, included in the error when provided.

    Raises:
        MissingRenderVarError: For the first required var missing from
            ``render_vars`` (sorted for deterministic messages).
    """
    for var in sorted(required):
        if var not in render_vars:
            raise MissingRenderVarError(writer_name, var, frame_id=frame_id)


# ---------------------------------------------------------------------------
# ovrtx annotator payload dtypes
# ---------------------------------------------------------------------------

# 128-bit ids are transported as four little-endian uint32 words.
_ID128 = ("<u4", (4,))

OVRTX_BBOX2D_DTYPE = np.dtype(
    [
        ("instance_id", *_ID128),
        ("semantic_id", *_ID128),
        ("x_min", "<i4"),
        ("y_min", "<i4"),
        ("x_max", "<i4"),
        ("y_max", "<i4"),
    ]
)

OVRTX_BBOX3D_DTYPE = np.dtype(
    [
        ("instance_id", *_ID128),
        ("semantic_id", *_ID128),
        ("x_min", "<f4"),
        ("y_min", "<f4"),
        ("z_min", "<f4"),
        ("x_max", "<f4"),
        ("y_max", "<f4"),
        ("z_max", "<f4"),
        ("transform", "<f4", (4, 4)),
    ]
)

OVRTX_CAMERA_PARAMS_DTYPE = np.dtype(
    [
        ("view_transform", "<f4", (4, 4)),
        ("projection", "<f4", (4, 4)),
        ("meters_per_scene_unit", "<f4"),
        ("resolution", "<u4", (2,)),
        ("clipping_plane", "<f4", (2,)),
        ("projection_model", "<S28"),
        ("focal_length", "<f4"),
        ("focal_distance", "<f4"),
        ("f_stop", "<f4"),
        ("fov_x", "<f4"),
        ("fisheye_nominal_width", "<f4"),
        ("fisheye_nominal_height", "<f4"),
        ("fisheye_optical_centre", "<f4", (2,)),
        ("fisheye_max_fov", "<f4"),
        ("fisheye_polynomial", "<f4", (6,)),
        ("lens_p", "<f4", (2,)),
        ("lens_s", "<f4", (4,)),
    ]
)

OVRTX_STABLE_SEMANTIC_ID_DTYPE = np.dtype(
    [
        ("stable_id", *_ID128),
        ("semantic_id", *_ID128),
    ]
)

OVRTX_INSTANCE_MAP_DTYPE = np.dtype(
    [
        ("instance_index", "<u4"),
        ("stable_semantic_index", "<u4"),
    ]
)


def _as_id(value: Any) -> int:
    """Return a Python int id from a (4,) uint32 128-bit word array.

    The low (first) word is used; ovrtx semantic/instance ids fit in 32 bits.
    """
    return int(np.asarray(value).reshape(-1)[0])


def _to_numpy(data: Any) -> np.ndarray:
    """Convert supported array inputs to a NumPy array.

    Args:
        data: Array-like input, including NumPy arrays and Warp arrays.

    Returns:
        A NumPy array view/copy of the input.

    Raises:
        TypeError: If the input cannot be interpreted as an array.
    """
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "numpy"):
        return data.numpy()
    try:
        return np.asarray(data)
    except Exception as exc:
        raise TypeError(f"Unsupported array input type: {type(data)!r}") from exc


def to_rgb_uint8(data: Any) -> np.ndarray:
    """Convert an image array to a contiguous ``(H, W, 3)`` ``uint8`` RGB array.

    Grayscale inputs are replicated across three channels, alpha is dropped, and
    floating-point inputs are assumed to be in ``[0, 1]`` and scaled to
    ``[0, 255]``.  Used by writers that hand frames to an image or video codec.

    Args:
        data (Any): Image array (NumPy or Warp), shaped ``(H, W)``,
            ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``.

    Returns:
        np.ndarray: An ``(H, W, 3)`` ``uint8`` RGB array.
    """
    arr = _to_numpy(data)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_warp_array(data: np.ndarray) -> Any:
    """Convert a NumPy array to a Warp CPU array when possible."""
    try:
        return wp.array(data, device="cpu")
    except Exception:
        # Warp does not support every numpy dtype (e.g. structured dtypes).
        return data


def _sanitize_semantic_label(label: str) -> str:
    """Clean an ovrtx semantic label for use in dataset formats.

    Omniverse USD semantic labels often include a schema prefix and trailing
    punctuation (e.g. ``"class: plane;"``).  This strips the prefix and
    trailing semicolons to yield a clean name (``"plane"``).
    """
    import re

    cleaned = re.sub(r"^(class|type)\s*:\s*", "", label, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip("; \t")
    return cleaned.strip() or label


def _label_for(semantic_id: int, id_to_label: dict[int, str] | None) -> str:
    """Resolve a class label for ``semantic_id`` via the optional id->label map."""
    if id_to_label is not None and semantic_id in id_to_label:
        return _sanitize_semantic_label(id_to_label[semantic_id])
    return str(semantic_id)


def _prim_path_for(
    instance_id: int, id_to_prim_path: dict[int, str] | None
) -> str | None:
    """Resolve a prim path for ``instance_id`` via the optional id->path map."""
    if id_to_prim_path is not None:
        return id_to_prim_path.get(instance_id)
    return None


# ---------------------------------------------------------------------------
# SemanticIdMap decoding
# ---------------------------------------------------------------------------


def decode_id_string_map(data: Any) -> dict[int, str]:
    """Decode an ovrtx id-to-string map payload into a ``{id: string}`` dict.

    Shared binary layout used by ``SemanticIdMap`` (id -> label) and
    ``StableIdMap`` (id -> prim path).

    Args:
        data: Raw buffer as ``bytes`` or a ``uint8`` numpy array. The trailing
            4 bytes hold the entry count; each entry is a 128-bit id (four
            ``uint32``) plus ``label_length`` and ``label_offset`` ``uint32``
            fields, followed by a string table.

    Returns:
        Mapping from id to its decoded string. Returns an empty dict for empty
        or malformed input.
    """
    if data is None:
        return {}
    raw = _to_bytes(data)
    if len(raw) < 4:
        return {}
    num_entries = int.from_bytes(raw[-4:], "little")
    if num_entries <= 0:
        return {}
    entry_dtype = np.dtype(
        [("semantic_id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    needed = num_entries * entry_dtype.itemsize
    if needed > len(raw):
        return {}
    entries = np.frombuffer(raw[:needed], dtype=entry_dtype)
    mapping: dict[int, str] = {}
    for entry in entries:
        sid = int(entry["semantic_id"][0])
        length = int(entry["label_length"])
        offset = int(entry["label_offset"])
        if offset + length > len(raw):
            continue
        mapping[sid] = raw[offset : offset + length].decode("utf-8", "replace").rstrip()
    return mapping


def decode_semantic_id_map(data: Any) -> dict[int, str]:
    """Decode a ``SemanticIdMap`` payload into a ``{semantic_id: label}`` dict.

    Args:
        data: Raw ``SemanticIdMap`` buffer as ``bytes`` or a ``uint8`` numpy
            array. See :func:`decode_id_string_map` for the binary layout.

    Returns:
        Mapping from semantic id to its decoded label string. Returns an empty
        dict for empty or malformed input.
    """
    return decode_id_string_map(data)


def decode_stable_id_semantic_id_map(data: Any) -> np.ndarray:
    """Decode ``StableIdSemanticIdMap`` into stable/semantic 128-bit ID pairs.

    The ovrtx 0.4 runtime emits a tightly packed table of 32-byte records with
    no trailing count: four little-endian uint32 words for the stable ID,
    followed by four words for the semantic ID.

    Args:
        data: Raw uint8 buffer, bytes, NumPy array, or Warp array.

    Returns:
        Structured NumPy array using :data:`OVRTX_STABLE_SEMANTIC_ID_DTYPE`.

    Raises:
        ValueError: If the byte count is not a whole number of records.
    """
    raw = _to_bytes(data)
    itemsize = OVRTX_STABLE_SEMANTIC_ID_DTYPE.itemsize
    if len(raw) % itemsize:
        raise ValueError(
            f"StableIdSemanticIdMap has {len(raw)} bytes; expected a multiple of {itemsize}"
        )
    return np.frombuffer(raw, dtype=OVRTX_STABLE_SEMANTIC_ID_DTYPE).copy()


def decode_instance_map(data: Any) -> np.ndarray:
    """Decode ``InstanceMap`` into instance-to-stable-semantic table indices.

    Each uint32 word is indexed by its position in the buffer and references a
    record in ``StableIdSemanticIdMap``. The value ``0xffffffff`` is the
    renderer's invalid-entry sentinel and is preserved in the decoded array.

    Args:
        data: Raw uint8 buffer, bytes, NumPy array, or Warp array.

    Returns:
        Structured NumPy array using :data:`OVRTX_INSTANCE_MAP_DTYPE`.

    Raises:
        ValueError: If the byte count is not uint32-aligned.
    """
    raw = _to_bytes(data)
    word_size = np.dtype("<u4").itemsize
    if len(raw) % word_size:
        raise ValueError(f"InstanceMap has {len(raw)} bytes; expected uint32 alignment")
    values = np.frombuffer(raw, dtype="<u4")
    decoded = np.empty(values.size, dtype=OVRTX_INSTANCE_MAP_DTYPE)
    decoded["instance_index"] = np.arange(values.size, dtype=np.uint32)
    decoded["stable_semantic_index"] = values
    return decoded


def _to_bytes(data: Any) -> bytes:
    """Coerce a raw payload (bytes, NumPy, Warp) to bytes.

    Args:
        data: Raw payload as bytes-like, NumPy array, or Warp array.

    Returns:
        Raw payload bytes.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, np.ndarray):
        return data.tobytes()
    if hasattr(data, "numpy"):
        return data.numpy().tobytes()
    return bytes(data)


def decode_bbox_2d(data: Any) -> Any:
    """Decode a ``SemanticBoundingBox2D*`` payload into a structured array.

    Accepts either an already-decoded array or the raw ovrtx byte payload, whose
    trailing 4 bytes encode the box count.

    Args:
        data (Any): Raw bbox payload (bytes/buffer) or an
            :data:`OVRTX_BBOX2D_DTYPE` array.

    Returns:
        Any: A Warp array of :data:`OVRTX_BBOX2D_DTYPE` records (empty when the
        payload holds no boxes).
    """
    if isinstance(data, np.ndarray) and data.dtype == OVRTX_BBOX2D_DTYPE:
        return _to_warp_array(data)
    raw = _to_bytes(data)
    if len(raw) < 4:
        return _to_warp_array(np.zeros(0, dtype=OVRTX_BBOX2D_DTYPE))
    num_boxes = int.from_bytes(raw[-4:], "little")
    size = num_boxes * OVRTX_BBOX2D_DTYPE.itemsize
    decoded = np.frombuffer(raw[:size], dtype=OVRTX_BBOX2D_DTYPE).copy()
    return _to_warp_array(decoded)


def decode_bbox_3d(data: Any) -> Any:
    """Decode a ``SemanticBoundingBox3D`` payload into a structured array.

    Accepts either an already-decoded array or the raw ovrtx byte payload, whose
    trailing 4 bytes encode the box count.

    Args:
        data (Any): Raw bbox payload (bytes/buffer) or an
            :data:`OVRTX_BBOX3D_DTYPE` array.

    Returns:
        Any: A Warp array of :data:`OVRTX_BBOX3D_DTYPE` records (empty when the
        payload holds no boxes).
    """
    if isinstance(data, np.ndarray) and data.dtype == OVRTX_BBOX3D_DTYPE:
        return _to_warp_array(data)
    raw = _to_bytes(data)
    if len(raw) < 4:
        return _to_warp_array(np.zeros(0, dtype=OVRTX_BBOX3D_DTYPE))
    num_boxes = int.from_bytes(raw[-4:], "little")
    size = num_boxes * OVRTX_BBOX3D_DTYPE.itemsize
    decoded = np.frombuffer(raw[:size], dtype=OVRTX_BBOX3D_DTYPE).copy()
    return _to_warp_array(decoded)


def decode_camera_params(data: Any) -> Any:
    """Decode a ``CameraParams`` payload into a 1-element structured array.

    Accepts either an already-decoded array or the raw ovrtx byte payload.

    Args:
        data (Any): Raw camera-params payload (bytes/buffer) or an
            :data:`OVRTX_CAMERA_PARAMS_DTYPE` array.

    Returns:
        Any: A 1-element Warp array of :data:`OVRTX_CAMERA_PARAMS_DTYPE`.
    """
    if isinstance(data, np.ndarray) and data.dtype == OVRTX_CAMERA_PARAMS_DTYPE:
        return _to_warp_array(data)
    raw = _to_bytes(data)
    itemsize = OVRTX_CAMERA_PARAMS_DTYPE.itemsize
    decoded = np.frombuffer(raw[:itemsize], dtype=OVRTX_CAMERA_PARAMS_DTYPE).copy()
    return _to_warp_array(decoded)


def id_label_map_from_render_vars(
    render_vars: dict[str, Any],
) -> dict[int, str] | None:
    """Return the decoded ``SemanticIdMap`` from ``render_vars`` if present.

    Args:
        render_vars (dict[str, Any]): Extracted render vars for a frame.

    Returns:
        dict[int, str] | None: Mapping from semantic id to label, or ``None``
        when ``SemanticIdMap`` is absent.
    """
    sid_map = render_vars.get("SemanticIdMap")
    if sid_map is None:
        return None
    if isinstance(sid_map, dict):
        return sid_map
    return decode_semantic_id_map(sid_map)


def id_prim_path_map_from_render_vars(
    render_vars: dict[str, Any],
) -> dict[int, str] | None:
    """Return the decoded ``StableIdMap`` (``{id: prim_path}``) if present.

    ovrtx ``StableIdMap`` shares the id-to-string binary layout decoded by
    :func:`decode_id_string_map`, mapping each stable/instance id to its USD
    prim path.

    Args:
        render_vars (dict[str, Any]): Extracted render vars for a frame.

    Returns:
        dict[int, str] | None: Mapping from id to prim path, or ``None`` when
        ``StableIdMap`` is absent.
    """
    stable_map = render_vars.get("StableIdMap")
    if stable_map is None:
        return None
    if isinstance(stable_map, dict):
        return stable_map
    return decode_id_string_map(stable_map)


# ---------------------------------------------------------------------------
# Bounding-box iterators
# ---------------------------------------------------------------------------


def iter_bboxes_2d(
    arr: Any,
    id_to_label: dict[int, str] | None = None,
    id_to_prim_path: dict[int, str] | None = None,
) -> Iterator[BBox2D]:
    """Yield :class:`~ovannotators.types.BBox2D` from a 2-D bbox render var.

    Args:
        arr: Structured array following :data:`OVRTX_BBOX2D_DTYPE`.
        id_to_label: Optional ``{semantic_id: label}`` map (from
            :func:`decode_semantic_id_map`). When absent, ``label`` falls back
            to the stringified ``semantic_id``.
        id_to_prim_path: Optional ``{instance_id: prim_path}`` map (from
            :func:`id_prim_path_map_from_render_vars`). When provided, each box's
            ``prim_path`` is resolved from its ``instance_id`` (``None`` when
            unmatched).

    Yields:
        BBox2D: One dataclass per box. ``occlusion_ratio`` is ``0.0`` since the
        ovrtx payload carries no occlusion information.
    """
    if arr is None:
        return
    arr = _to_numpy(arr)
    if arr.size == 0:
        return
    for i in range(len(arr)):
        row = arr[i]
        semantic_id = _as_id(row["semantic_id"])
        instance_id = _as_id(row["instance_id"])
        yield BBox2D(
            x_min=float(row["x_min"]),
            y_min=float(row["y_min"]),
            x_max=float(row["x_max"]),
            y_max=float(row["y_max"]),
            semantic_id=semantic_id,
            instance_id=instance_id,
            label=_label_for(semantic_id, id_to_label),
            prim_path=_prim_path_for(instance_id, id_to_prim_path),
        )


def iter_bboxes_3d(
    arr: Any,
    id_to_label: dict[int, str] | None = None,
    id_to_prim_path: dict[int, str] | None = None,
) -> Iterator[BBox3D]:
    """Yield :class:`~ovannotators.types.BBox3D` from a 3-D bbox render var.

    The ovrtx payload (:data:`OVRTX_BBOX3D_DTYPE`) stores per-box object-space
    ``[x/y/z]_min`` / ``[x/y/z]_max`` extents plus a 4x4 local-to-world
    ``transform``. These are converted to the ``center`` / ``extent`` /
    ``rotation`` representation the writers consume, using the convention
    ``world = R @ local + T`` where ``R = transform[:3, :3]`` and
    ``T = transform[:3, 3]``. The resulting boxes are therefore in world space
    (``coordinate_system="world"``).

    Args:
        arr: Structured array following :data:`OVRTX_BBOX3D_DTYPE`.
        id_to_label: Optional ``{semantic_id: label}`` map; see
            :func:`iter_bboxes_2d`.
        id_to_prim_path: Optional ``{instance_id: prim_path}`` map; see
            :func:`iter_bboxes_2d`.

    Yields:
        BBox3D: One dataclass per box, with ``coordinate_system="world"``.
    """
    if arr is None:
        return
    arr = _to_numpy(arr)
    if arr.size == 0:
        return
    for i in range(len(arr)):
        row = arr[i]
        semantic_id = _as_id(row["semantic_id"])
        instance_id = _as_id(row["instance_id"])
        lo = np.array([row["x_min"], row["y_min"], row["z_min"]], dtype=np.float64)
        hi = np.array([row["x_max"], row["y_max"], row["z_max"]], dtype=np.float64)
        transform = np.asarray(row["transform"], dtype=np.float64).reshape(4, 4)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        center = rotation @ ((lo + hi) / 2.0) + translation
        yield BBox3D(
            center=center,
            extent=(hi - lo) / 2.0,
            rotation=rotation,
            semantic_id=semantic_id,
            instance_id=instance_id,
            label=_label_for(semantic_id, id_to_label),
            coordinate_system="world",
            prim_path=_prim_path_for(instance_id, id_to_prim_path),
        )


# ---------------------------------------------------------------------------
# Camera params extraction
# ---------------------------------------------------------------------------


def extract_camera_params(arr: Any) -> CameraParams:
    """Build a :class:`~ovannotators.types.CameraParams` from a render var.

    The ovrtx payload (:data:`OVRTX_CAMERA_PARAMS_DTYPE`) provides a 4x4
    camera-to-world ``view_transform``, a 4x4 ``projection`` matrix, the render
    ``resolution``, ``clipping_plane`` (near/far) and a horizontal ``fov_x``.
    The pinhole intrinsics ``K`` are derived from ``fov_x`` and ``resolution``
    (square pixels, principal point at the image centre); the world-to-camera
    ``R`` / ``t`` are derived by inverting ``view_transform``; ``fov_y`` is
    derived from ``fov_x`` and the aspect ratio (square pixels).

    Args:
        arr: ``CameraParams`` render var (structured array or raw buffer).

    Returns:
        CameraParams: Intrinsics (``K``), world-to-camera extrinsics
        (``R``/``t``), ``resolution``, ``near``/``far`` and ``fov_x``/``fov_y``.
    """
    arr = _to_numpy(arr)
    scalar = arr.reshape(-1)[0] if arr.ndim > 0 else arr
    resolution = np.asarray(scalar["resolution"]).reshape(2).astype(np.float64)
    width, height = int(resolution[0]), int(resolution[1])
    clip = np.asarray(scalar["clipping_plane"]).reshape(2).astype(np.float64)
    near, far = float(clip[0]), float(clip[1])
    fov_x = float(scalar["fov_x"])

    # Intrinsics from horizontal FOV (square pixels, centred principal point).
    fx = (
        (width / 2.0) / math.tan(fov_x / 2.0)
        if fov_x > 0 and width > 0
        else float(width)
    )
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Vertical FOV from the aspect ratio (square pixels => fy == fx).
    fov_y = 2.0 * math.atan((height / 2.0) / fx) if fx > 0 else 0.0

    # view_transform is camera-to-world; world-to-camera is its inverse.
    view = np.asarray(scalar["view_transform"], dtype=np.float64).reshape(4, 4)
    try:
        world_to_cam = np.linalg.inv(view)
    except np.linalg.LinAlgError:
        world_to_cam = np.eye(4, dtype=np.float64)

    return CameraParams(
        K=K,
        R=world_to_cam[:3, :3],
        t=world_to_cam[:3, 3],
        resolution=Resolution(width=width, height=height),
        near=near,
        far=far,
        fov_x=fov_x,
        fov_y=fov_y,
    )


def point_cloud_from_depth(
    depth: Any,
    camera: CameraParams,
    colors: Any = None,
    semantic_ids: Any = None,
    *,
    device: str = "cuda",
    output_device: str = "cpu",
    reuse_gpu_outputs: bool = False,
) -> PointCloudResult:
    """Back-project a depth buffer into a camera-space point cloud.

    Each valid pixel ``(u, v)`` with depth ``z`` is unprojected through the
    pinhole intrinsics ``camera.K`` as ``x = (u - cx) * z / fx`` and
    ``y = (v - cy) * z / fy``. Pixels with non-finite or non-positive depth are
    dropped, and any per-pixel ``colors``/``semantic_ids`` are filtered to match.

    Args:
        depth (Any): ``(H, W)`` (or ``(H, W, 1)``) depth image, in scene units
            (NumPy or Warp array).
        camera (CameraParams): Intrinsics used for back-projection; its ``K``
            supplies ``fx``, ``fy``, ``cx`` and ``cy``.
        colors (Any): Optional ``(H, W, C)`` per-pixel colors carried onto the
            surviving points.
        semantic_ids (Any): Optional ``(H, W)`` per-pixel semantic ids carried
            onto the surviving points. Attributes must have Warp-supported
            numeric dtypes.
        device (str): Construction device, CUDA by default. A CUDA device
            such as ``"cuda:0"`` selects GPU compaction and backprojection.
            Pass ``"cpu"`` explicitly for CPU construction without CUDA.
            CUDA Warp inputs remain on the device during construction; NumPy
            inputs are uploaded. CPU delivery returns compact Warp CPU arrays.
        output_device (str): ``"cpu"`` (default) returns compact Warp CPU arrays;
            ``"cuda"`` returns owned capacity-sized CUDA buffers with a
            device-side valid count. CUDA outputs require CUDA construction.
        reuse_gpu_outputs (bool): Opt in to bounded pooled storage for returned
            CUDA arrays. Retained arrays/views protect the entire batch bank;
            last disposal fences device work before recycling. Disabled by
            default to avoid its synchronization cost. Temporary CUDA output
            buffers for CPU delivery are always pooled.

    Returns:
        PointCloudResult: Compact CPU ``(N, 3)``
        points, or capacity-sized CUDA points and a device-side valid count.
        Positions are float32 camera-space coordinates. Optional colors and
        IDs retain input dtypes; normals are left as None.
    """
    if not isinstance(reuse_gpu_outputs, bool):
        raise ValueError("reuse_gpu_outputs must be a bool")
    if output_device not in {"cpu", "cuda"}:
        raise ValueError("output_device must be cpu or cuda")
    if device == "cpu" and output_device != "cpu":
        raise ValueError("CUDA outputs require CUDA construction")
    if device != "cpu":
        from ovannotators.writers._pointcloud_gpu import _point_cloud_gpu

        return _point_cloud_gpu(
            depth,
            camera,
            colors,
            semantic_ids,
            device,
            output_device,
            reuse_gpu_outputs,
        )
    from ovannotators.writers._pointcloud_cpu import _point_cloud_cpu

    return _point_cloud_cpu(depth, camera, colors, semantic_ids)


# ---------------------------------------------------------------------------
# Batched instance-mask extraction (Warp)
# ---------------------------------------------------------------------------


def _mask_device() -> wp.context.Device:
    """Return the current default Warp device (CUDA if present, else CPU).

    Respects :func:`warp.ScopedDevice`/:func:`warp.set_device`, so multi-GPU
    callers can target a specific device without this module hard-coding one.
    """
    return wp.get_device()


@wp.kernel
def _instance_masks_kernel(
    inst_seg: wp.array2d(dtype=wp.int64),
    ids: wp.array(dtype=wp.int64),
    out: wp.array3d(dtype=wp.uint8),
):
    """Set ``out[m, y, x] = 1`` where ``inst_seg[y, x]`` equals ``ids[m]``."""
    m, y, x = wp.tid()
    if inst_seg[y, x] == ids[m]:
        out[m, y, x] = wp.uint8(1)
    else:
        out[m, y, x] = wp.uint8(0)


def instance_masks(inst_seg: Any, instance_ids: Sequence[int]) -> dict[int, np.ndarray]:
    """Extract per-instance binary masks in a single parallel Warp pass.

    Replaces ``N`` separate ``inst_seg == id`` full-image comparisons with one
    Warp kernel launch over ``(num_ids, H, W)``. The kernel runs on CUDA when a
    device is available and otherwise on the Warp CPU device.

    Args:
        inst_seg: 2-D instance-id image (NumPy or Warp array). A trailing
            channel axis (``H x W x 1``) is squeezed automatically.
        instance_ids: Instance ids to extract masks for. Duplicates are
            collapsed and ids absent from the image yield all-zero masks.

    Returns:
        Mapping ``{instance_id: mask}`` where each ``mask`` is a 2-D ``uint8``
        array holding ``1`` where ``inst_seg == instance_id`` and ``0``
        elsewhere. Returns an empty dict when ``instance_ids`` is empty.
    """
    unique_ids = list(dict.fromkeys(int(i) for i in instance_ids))
    if not unique_ids:
        return {}
    seg = _to_numpy(inst_seg)
    if seg.ndim > 2:
        seg = seg[..., 0]
    h, w = seg.shape[:2]
    device = _mask_device()
    seg_wp = wp.array(
        np.ascontiguousarray(seg, dtype=np.int64), dtype=wp.int64, device=device
    )
    ids_wp = wp.array(
        np.asarray(unique_ids, dtype=np.int64), dtype=wp.int64, device=device
    )
    out = wp.empty((len(unique_ids), h, w), dtype=wp.uint8, device=device)
    wp.launch(
        _instance_masks_kernel,
        dim=(len(unique_ids), h, w),
        inputs=[seg_wp, ids_wp, out],
        device=device,
    )
    masks = out.numpy()
    return {iid: masks[k] for k, iid in enumerate(unique_ids)}


# ---------------------------------------------------------------------------
# Mask encoding
# ---------------------------------------------------------------------------


def mask_to_rle(mask: Any) -> dict[str, Any]:
    """Encode a binary mask as COCO uncompressed RLE (Fortran/column-major order).

    Args:
        mask: 2-D boolean or uint array.

    Returns:
        ``{"counts": [...], "size": [height, width]}``
    """
    mask = _to_numpy(mask)
    h, w = mask.shape[:2]
    m = np.asfortranarray(mask > 0).ravel(order="F").astype(np.uint8)
    if m.size == 0:
        return {"counts": [], "size": [h, w]}

    padded = np.concatenate([[0], m, [0]])
    diff = np.diff(padded.astype(np.int8))
    starts: np.ndarray = np.where(diff > 0)[0]
    ends: np.ndarray = np.where(diff < 0)[0]

    counts: list[int] = []
    prev = 0
    for s, e in zip(starts, ends):
        counts.append(int(s - prev))
        counts.append(int(e - s))
        prev = e
    counts.append(int(len(m) - prev))
    return {"counts": counts, "size": [h, w]}


# ---------------------------------------------------------------------------
# Mask -> tight polygon contour (Warp marching squares + host ring-linking)
# ---------------------------------------------------------------------------
#
# Unlike a convex hull (e.g. scipy/QuickHull or GPU variants such as gHull,
# Tang et al. 2012), marching squares recovers a *tight*, concavity-preserving
# contour, which is what instance-segmentation polygon labels (e.g. YOLO-seg)
# require. The per-cell case classification is embarrassingly parallel and runs
# as a Warp kernel on CPU or CUDA; the inherently sequential parts (linking
# segments into rings, simplification) run on the host where the data volume is
# tiny.
#
# Algorithm credits:
#   * Marching squares — the 2-D analogue of marching cubes (Lorensen & Cline,
#     "Marching Cubes", SIGGRAPH 1987); 16-case edge table per the standard
#     formulation (cf. Maple/Wikipedia "Marching squares").
#   * Polyline simplification — Ramer (1972) / Douglas & Peucker (1973).

# Per-case edge pairs for marching squares. Edges are indexed
# ``0=top, 1=right, 2=bottom, 3=left``; each case lists up to two segments as
# ``(e0a, e0b, e1a, e1b)`` with ``-1`` marking an absent segment. The case index
# is ``TL*8 + TR*4 + BR*2 + BL`` over the four cell corners. Cases 5 and 10 are
# saddles, resolved by separating the diagonally-opposite "inside" corners.
_MARCHING_SQUARES_TABLE = np.array(
    [
        [-1, -1, -1, -1],  # 0: empty
        [3, 2, -1, -1],  # 1: BL
        [2, 1, -1, -1],  # 2: BR
        [3, 1, -1, -1],  # 3: BL+BR
        [0, 1, -1, -1],  # 4: TR
        [0, 3, 2, 1],  # 5: TR+BL (saddle)
        [0, 2, -1, -1],  # 6: TR+BR
        [0, 3, -1, -1],  # 7: TR+BR+BL
        [0, 3, -1, -1],  # 8: TL
        [0, 2, -1, -1],  # 9: TL+BL
        [0, 1, 2, 3],  # 10: TL+BR (saddle)
        [0, 1, -1, -1],  # 11: TL+BR+BL
        [3, 1, -1, -1],  # 12: TL+TR
        [2, 1, -1, -1],  # 13: TL+TR+BL
        [3, 2, -1, -1],  # 14: TL+TR+BR
        [-1, -1, -1, -1],  # 15: full
    ],
    dtype=np.int32,
)


@wp.func
def _ms_edge_xy(e: int, i: int, j: int) -> wp.vec2:
    """Return the ``(x, y)`` midpoint of cell ``(i, j)``'s edge ``e``.

    Coordinates are in the (cropped) mask's pixel-center frame, where pixel
    ``(r, c)`` sits at ``(x=c, y=r)``. Cell ``(i, j)`` spans the padded-grid
    corners around that pixel, so edge midpoints land on half-integer offsets.
    """
    fi = float(i)
    fj = float(j)
    if e == 0:  # top
        return wp.vec2(fj - 0.5, fi - 1.0)
    if e == 1:  # right
        return wp.vec2(fj, fi - 0.5)
    if e == 2:  # bottom
        return wp.vec2(fj - 0.5, fi)
    return wp.vec2(fj - 1.0, fi - 0.5)  # left


@wp.kernel
def _marching_squares_kernel(
    padded: wp.array2d(dtype=wp.int32),
    seg_table: wp.array2d(dtype=wp.int32),
    out_segs: wp.array4d(dtype=wp.float32),
    out_count: wp.array2d(dtype=wp.int32),
):
    """Classify each 2x2 cell and emit its 0-2 iso-contour segments."""
    i, j = wp.tid()
    tl = padded[i, j]
    tr = padded[i, j + 1]
    bl = padded[i + 1, j]
    br = padded[i + 1, j + 1]
    case = tl * 8 + tr * 4 + br * 2 + bl
    n = int(0)
    for s in range(2):
        e0 = seg_table[case, 2 * s]
        e1 = seg_table[case, 2 * s + 1]
        if e0 >= 0:
            p0 = _ms_edge_xy(e0, i, j)
            p1 = _ms_edge_xy(e1, i, j)
            out_segs[i, j, n, 0] = p0[0]
            out_segs[i, j, n, 1] = p0[1]
            out_segs[i, j, n, 2] = p1[0]
            out_segs[i, j, n, 3] = p1[1]
            n = n + 1
    out_count[i, j] = n


def _marching_squares_segments(mask: np.ndarray) -> np.ndarray:
    """Return iso-contour segments of a binary mask via a Warp kernel.

    The mask is cropped to its set-pixel bounding box (so the kernel only
    touches the relevant region of a large frame) and zero-padded by one pixel
    so border objects produce closed contours.

    Args:
        mask: 2-D array; non-zero pixels are foreground.

    Returns:
        ``(N, 4)`` float32 array of ``(ax, ay, bx, by)`` segment endpoints in
        the original mask's pixel-center coordinate frame. Empty when the mask
        has no foreground pixels.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    sub = mask[y0 : y1 + 1, x0 : x1 + 1] > 0
    padded = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), dtype=np.int32)
    padded[1:-1, 1:-1] = sub.astype(np.int32)

    rows, cols = padded.shape[0] - 1, padded.shape[1] - 1
    device = _mask_device()
    padded_wp = wp.array(padded, dtype=wp.int32, device=device)
    table_wp = wp.array(_MARCHING_SQUARES_TABLE, dtype=wp.int32, device=device)
    out_segs = wp.empty((rows, cols, 2, 4), dtype=wp.float32, device=device)
    out_count = wp.empty((rows, cols), dtype=wp.int32, device=device)
    wp.launch(
        _marching_squares_kernel,
        dim=(rows, cols),
        inputs=[padded_wp, table_wp, out_segs, out_count],
        device=device,
    )

    counts = out_count.numpy()
    segs = out_segs.numpy()
    pieces = [segs[:, :, s, :][counts > s] for s in range(2) if np.any(counts > s)]
    if not pieces:  # pragma: no cover - a non-empty crop always emits segments
        return np.empty((0, 4), dtype=np.float32)
    seg_arr = np.concatenate(pieces).astype(np.float32)
    # Offset cropped-frame coordinates back into the full mask frame.
    seg_arr[:, 0] += x0
    seg_arr[:, 2] += x0
    seg_arr[:, 1] += y0
    seg_arr[:, 3] += y0
    return seg_arr


def _link_rings(segments: np.ndarray) -> list[np.ndarray]:
    """Stitch undirected contour segments into closed rings.

    Endpoints lie on a half-integer grid, so scaling by two yields exact integer
    keys; each grid point is shared by at most two segments, making the linkage
    unambiguous without floating-point tolerances.

    Args:
        segments: ``(N, 4)`` array of ``(ax, ay, bx, by)`` endpoints.

    Returns:
        A list of rings, each an ``(M, 2)`` array of ordered ``(x, y)`` vertices.
    """

    def _key(x: float, y: float) -> tuple[int, int]:
        return (int(round(x * 2.0)), int(round(y * 2.0)))

    adjacency: dict[
        tuple[int, int], list[tuple[int, tuple[int, int], tuple[float, float]]]
    ] = defaultdict(list)
    for idx, (ax, ay, bx, by) in enumerate(segments):
        ka, kb = _key(ax, ay), _key(bx, by)
        adjacency[ka].append((idx, kb, (float(bx), float(by))))
        adjacency[kb].append((idx, ka, (float(ax), float(ay))))

    used: set[int] = set()
    rings: list[np.ndarray] = []
    for start_idx in range(len(segments)):
        if start_idx in used:
            continue
        ax, ay, bx, by = segments[start_idx]
        start_key = _key(ax, ay)
        ring: list[tuple[float, float]] = [(float(ax), float(ay))]
        used.add(start_idx)
        next_key, next_xy = _key(bx, by), (float(bx), float(by))
        while True:
            ring.append(next_xy)
            if next_key == start_key:
                ring.pop()  # closed; drop the duplicate of the start vertex
                break
            advanced = False
            for seg_id, other_key, other_xy in adjacency[next_key]:
                if seg_id in used:
                    continue
                used.add(seg_id)
                next_key, next_xy = other_key, other_xy
                advanced = True
                break
            if not advanced:  # pragma: no cover - marching-squares rings close
                break
        if len(ring) >= 3:
            rings.append(np.asarray(ring, dtype=np.float64))
    return rings


def _polygon_area(ring: np.ndarray) -> float:
    """Return the absolute shoelace area of a closed ``(M, 2)`` ring."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _douglas_peucker(points: np.ndarray, eps: float) -> np.ndarray:
    """Simplify an ordered polyline with the Ramer-Douglas-Peucker algorithm.

    Args:
        points: ``(M, 2)`` ordered vertices.
        eps: Maximum perpendicular deviation (in pixels) allowed when dropping a
            vertex. Larger values yield coarser polygons.

    Returns:
        The simplified ``(K, 2)`` subset of ``points`` (endpoints preserved).
    """
    n = len(points)
    if n < 3 or eps <= 0.0:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a, b = points[start], points[end]
        seg = b - a
        seg_len = float(np.hypot(seg[0], seg[1]))
        between = points[start + 1 : end]
        if seg_len == 0.0:
            dist = np.hypot(between[:, 0] - a[0], between[:, 1] - a[1])
        else:
            # 2-D cross product (z-component); avoids np.cross on 2-vectors,
            # which is deprecated in NumPy 2.0.
            rel = between - a
            cross = seg[0] * rel[:, 1] - seg[1] * rel[:, 0]
            dist = np.abs(cross) / seg_len
        local = int(np.argmax(dist))
        if dist[local] > eps:
            split = start + 1 + local
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return points[keep]


def _simplify_ring(ring: np.ndarray, eps: float) -> np.ndarray:
    """Apply Ramer-Douglas-Peucker to a *closed* ring without flattening it.

    The ring is anchored at an extreme vertex and the vertex farthest from it,
    then each of the two arcs between those anchors is simplified independently.
    Anchoring keeps two well-separated vertices so a loop can never collapse to
    a line, unlike running the open-polyline variant across the seam.

    Args:
        ring: ``(M, 2)`` ordered closed-ring vertices (start not repeated).
        eps: Maximum perpendicular deviation in pixels.

    Returns:
        The simplified ``(K, 2)`` ring.
    """
    n = len(ring)
    if n < 4 or eps <= 0.0:
        return ring
    start = int(np.lexsort((ring[:, 1], ring[:, 0]))[0])
    rolled = np.roll(ring, -start, axis=0)
    far = int(
        np.argmax(np.hypot(rolled[:, 0] - rolled[0, 0], rolled[:, 1] - rolled[0, 1]))
    )
    if far == 0:
        return ring
    first = _douglas_peucker(rolled[: far + 1], eps)
    second = _douglas_peucker(np.vstack([rolled[far:], rolled[:1]]), eps)
    return np.vstack([first[:-1], second[:-1]])


def mask_to_polygon(mask: Any, simplify_tol: float = 1.0) -> list[float]:
    """Convert a binary mask to a normalized tight polygon ``[x1, y1, x2, y2, …]``.

    Extracts the object's outer contour with a device-agnostic Warp
    marching-squares pass, links the segments into closed rings, keeps the
    largest ring, and simplifies it (Ramer-Douglas-Peucker). The result hugs
    concavities, unlike a convex hull, and carries no optional dependency.

    Args:
        mask (Any): 2-D binary mask (NumPy or Warp array); non-zero is
            foreground. A trailing channel axis is squeezed.
        simplify_tol (float): Maximum perpendicular vertex deviation in pixels
            for polyline simplification. ``0`` disables simplification.

    Returns:
        list[float]: Flat ``[x1, y1, …, xn, yn]`` polygon with coordinates
        normalized to ``[0, 1]``. Empty when the mask has no foreground pixels.
    """
    mask = _to_numpy(mask)
    if mask.ndim > 2:
        mask = mask[..., 0]
    height, width = mask.shape[:2]

    segments = _marching_squares_segments(mask)
    rings = _link_rings(segments)
    if not rings:
        return []

    ring = max(rings, key=_polygon_area)
    simplified = _simplify_ring(ring, simplify_tol)
    if len(simplified) >= 3:  # never emit a degenerate (<3-vertex) polygon
        ring = simplified

    norm = ring / np.array([width, height], dtype=np.float64)
    norm = np.clip(norm, 0.0, 1.0)
    return norm.reshape(-1).tolist()


def point_clouds_from_depth(
    depths: list[Any],
    cameras: list[CameraParams],
    colors: list[Any] | None = None,
    semantic_ids: list[Any] | None = None,
    *,
    device: str = "cuda",
    output_device: str = "cpu",
    reuse_gpu_outputs: bool = False,
) -> list[PointCloudResult]:
    """Construct multiple clouds, batching CUDA count and channel transfers.

    Args:
        depths: Depth images, one per camera, as in ``point_cloud_from_depth``.
        cameras: Intrinsics, one per image.
        colors: Optional aligned list of color images (entries may be None).
        semantic_ids: Optional aligned list of ID images (entries may be None).
        device: Construction device; CUDA by default, or explicit ``"cpu"``.
        output_device: ``"cpu"`` returns compact Warp CPU arrays; ``"cuda"``
            returns ready, independently owned CUDA buffers and GPU valid counts.
        reuse_gpu_outputs: Opt in to pooled returned CUDA arrays and a device
            fence on last batch-array disposal. Temporary CUDA output buffers
            for CPU delivery are always pooled.

    Returns:
        Clouds in input order. GPU scratch and pinned transfer staging are
        bounded and reused. CPU storage is leased until all returned arrays and
        their subviews are released; live outputs are never overwritten.

    Raises:
        ValueError: List lengths differ or the output/construction mode is invalid.
    """
    colors = [None] * len(depths) if colors is None else colors
    semantic_ids = [None] * len(depths) if semantic_ids is None else semantic_ids
    if not len(depths) == len(cameras) == len(colors) == len(semantic_ids):
        raise ValueError("Pointcloud batch inputs must have equal lengths")
    if not isinstance(reuse_gpu_outputs, bool):
        raise ValueError("reuse_gpu_outputs must be a bool")
    if output_device not in {"cpu", "cuda"}:
        raise ValueError("output_device must be cpu or cuda")
    if device == "cpu":
        if output_device != "cpu":
            raise ValueError("CUDA outputs require CUDA construction")
        return [
            point_cloud_from_depth(d, c, rgb, ids, device="cpu")
            for d, c, rgb, ids in zip(depths, cameras, colors, semantic_ids)
        ]
    from ovannotators.writers._pointcloud_gpu import _point_clouds_gpu

    return _point_clouds_gpu(
        depths, cameras, colors, semantic_ids, device, output_device, reuse_gpu_outputs
    )
