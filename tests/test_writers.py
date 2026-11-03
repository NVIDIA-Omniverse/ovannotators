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

"""Tests for all built-in Writer implementations.

Render vars are keyed by their ovrtx ``sourceName`` (the single source of truth);
bounding-box and camera vars use the structured ovrtx payload dtypes from
``ovannotators.writers._helpers``.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pytest

from ovannotators.exceptions import ConfigurationError, MissingRenderVarError
from ovannotators.writers import (
    BasicWriter,
    BasicWriterConfig,
    BOPWriter,
    BOPWriterConfig,
    COCOWriter,
    COCOWriterConfig,
    KITTIWriter,
    KITTIWriterConfig,
    LeRobotWriter,
    LeRobotWriterConfig,
    WriterRegistry,
    YOLOWriter,
    YOLOWriterConfig,
)
from ovannotators.writers._helpers import (
    OVRTX_BBOX2D_DTYPE,
    OVRTX_BBOX3D_DTYPE,
    OVRTX_CAMERA_PARAMS_DTYPE,
    _to_numpy,
)


def _has_functional_nvvideocodec() -> bool:
    """Check if PyNvVideoCodec is importable AND can create an encoder."""
    try:
        from ovannotators._internal.codecs import get_video_encoder

        enc = get_video_encoder("", 24, 256, 256)
        enc.write_frame(np.zeros((256, 256, 3), dtype=np.uint8))
        enc.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# MockBackend — avoids imageio/cv2 dependencies in writer tests
# ---------------------------------------------------------------------------


class MockBackend:
    """In-memory backend that records all writes without performing real I/O."""

    def __init__(self) -> None:
        self.written: dict[str, Any] = {}
        self.dirs: set[str] = set()

    def write(self, path: str, data: bytes) -> None:
        self.written[path] = data

    def write_array(self, path: str, array: np.ndarray, encoding: str = "npy") -> None:
        # Mirror real backends: accept NumPy or Warp arrays by materializing to
        # host NumPy before recording.
        np_array = array if isinstance(array, np.ndarray) else _to_numpy(array)
        self.written[path] = (np.asarray(np_array).copy(), encoding)

    def read(self, path: str) -> bytes:
        return self.written[path]  # type: ignore[return-value]

    def list(self, prefix: str = "") -> list[str]:
        return [p for p in self.written if p.startswith(prefix)]

    def exists(self, path: str) -> bool:
        return path in self.written

    def makedirs(self, path: str) -> None:
        self.dirs.add(path)


# ---------------------------------------------------------------------------
# Local builders for custom-label cases (e.g. unknown-class fallback)
# ---------------------------------------------------------------------------


def _id128(value: int) -> tuple[int, int, int, int]:
    return (int(value), 0, 0, 0)


def _make_bbox2d(boxes: list[tuple[int, int, int, int, int, int]]) -> np.ndarray:
    """Build an OVRTX_BBOX2D_DTYPE array from (iid, sid, xmin, ymin, xmax, ymax)."""
    arr = np.zeros(len(boxes), dtype=OVRTX_BBOX2D_DTYPE)
    for i, (iid, sid, x0, y0, x1, y1) in enumerate(boxes):
        arr[i]["instance_id"] = _id128(iid)
        arr[i]["semantic_id"] = _id128(sid)
        arr[i]["x_min"], arr[i]["y_min"], arr[i]["x_max"], arr[i]["y_max"] = (
            x0,
            y0,
            x1,
            y1,
        )
    return arr


def _make_id_map(labels: dict[int, str]) -> np.ndarray:
    """Pack ``{semantic_id: label}`` into a SemanticIdMap uint8 buffer."""
    entry_dtype = np.dtype(
        [("semantic_id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    num = len(labels)
    header_size = num * entry_dtype.itemsize
    items = list(labels.items())
    strings = b""
    offsets: list[int] = []
    lengths: list[int] = []
    for _sid, label in items:
        encoded = label.encode("utf-8")
        offsets.append(header_size + len(strings))
        lengths.append(len(encoded))
        strings += encoded
    entries = np.zeros(num, dtype=entry_dtype)
    for i, (sid, _label) in enumerate(items):
        entries[i]["semantic_id"] = _id128(sid)
        entries[i]["label_length"] = lengths[i]
        entries[i]["label_offset"] = offsets[i]
    raw = entries.tobytes() + strings + int(num).to_bytes(4, "little")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _make_bbox3d(boxes):
    """Build an OVRTX_BBOX3D_DTYPE array from (iid, sid, lo, hi, translate)."""
    arr = np.zeros(len(boxes), dtype=OVRTX_BBOX3D_DTYPE)
    for i, (iid, sid, lo, hi, translate) in enumerate(boxes):
        arr[i]["instance_id"] = _id128(iid)
        arr[i]["semantic_id"] = _id128(sid)
        arr[i]["x_min"], arr[i]["y_min"], arr[i]["z_min"] = lo
        arr[i]["x_max"], arr[i]["y_max"], arr[i]["z_max"] = hi
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = translate
        arr[i]["transform"] = transform
    return arr


def _make_camera_params():
    """Build a 1-element OVRTX_CAMERA_PARAMS_DTYPE array."""
    arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
    arr[0]["view_transform"] = np.eye(4, dtype=np.float32)
    arr[0]["projection"] = np.eye(4, dtype=np.float32)
    arr[0]["meters_per_scene_unit"] = 1.0
    arr[0]["resolution"] = (64, 48)
    arr[0]["clipping_plane"] = (0.01, 1000.0)
    arr[0]["fov_x"] = np.radians(60.0)
    return arr


def _decode_panoptic_ids(rgb: np.ndarray) -> np.ndarray:
    """Decode COCO panoptic RGB ids into uint32 segment ids."""
    encoded = rgb.astype(np.uint32)
    return encoded[..., 0] + (encoded[..., 1] << 8) + (encoded[..., 2] << 16)


def _make_coco_panoptic_render_vars() -> dict[str, np.ndarray]:
    """Build a compact render-vars sample with two labeled instances."""
    h, w = 24, 32
    ldr = np.zeros((h, w, 4), dtype=np.uint8)
    inst_seg = np.zeros((h, w), dtype=np.uint32)
    inst_seg[2:12, 3:11] = 101
    inst_seg[8:20, 15:26] = 202
    return {
        "LdrColor": ldr,
        "SemanticBoundingBox2DTight": _make_bbox2d(
            [
                (101, 1, 3, 2, 11, 12),
                (202, 2, 15, 8, 26, 20),
            ]
        ),
        "SemanticInstanceSegmentation": inst_seg,
        "SemanticIdMap": _make_id_map({1: "car", 2: "person"}),
    }


# ---------------------------------------------------------------------------
# WriterRegistry
# ---------------------------------------------------------------------------


def test_registry_list_names_has_all_builtins() -> None:
    names = set(WriterRegistry.list_names())
    assert {
        "basic",
        "coco",
        "kitti",
        "yolo",
        "bop",
        "lerobot",
        "cosmos_transfer",
    }.issubset(names)


def test_registry_get_basic() -> None:
    assert WriterRegistry.get("basic") is BasicWriter


def test_registry_get_coco() -> None:
    assert WriterRegistry.get("coco") is COCOWriter


def test_registry_get_kitti() -> None:
    assert WriterRegistry.get("kitti") is KITTIWriter


def test_registry_get_yolo() -> None:
    assert WriterRegistry.get("yolo") is YOLOWriter


def test_registry_get_bop() -> None:
    assert WriterRegistry.get("bop") is BOPWriter


def test_registry_get_lerobot() -> None:
    assert WriterRegistry.get("lerobot") is LeRobotWriter


def test_registry_get_unknown_raises() -> None:
    with pytest.raises(ConfigurationError, match="No writer registered"):
        WriterRegistry.get("_nonexistent_writer_xyz")


def test_registry_create_returns_instance() -> None:
    writer = WriterRegistry.create("coco")
    assert isinstance(writer, COCOWriter)


def test_top_level_exports_writers_and_configs() -> None:
    import ovannotators

    for name in (
        "COCOWriter",
        "COCOWriterConfig",
        "KITTIWriter",
        "KITTIWriterConfig",
        "YOLOWriter",
        "YOLOWriterConfig",
        "BOPWriter",
        "BOPWriterConfig",
        "LeRobotWriter",
        "LeRobotWriterConfig",
        "CosmosTransferWriter",
        "CosmosTransferWriterConfig",
        "BasicWriter",
        "BasicWriterConfig",
    ):
        assert hasattr(ovannotators, name), name
        assert name in ovannotators.__all__, name


def test_registry_register_custom_writer() -> None:
    class _CustomWriter:
        name = "_test_custom_writer_xyz"
        required_render_vars = frozenset()

        def write_frame(self, frame_id, render_vars, backend):
            pass

        def finalize(self, backend):
            pass

    try:
        result = WriterRegistry.register(_CustomWriter)
        assert result is _CustomWriter
        assert WriterRegistry.get("_test_custom_writer_xyz") is _CustomWriter
    finally:
        WriterRegistry._registry.pop("_test_custom_writer_xyz", None)


# ---------------------------------------------------------------------------
# BasicWriter
# ---------------------------------------------------------------------------


def test_basic_writer_name() -> None:
    assert BasicWriter.name == "basic"


def test_basic_writer_has_no_required_vars() -> None:
    assert BasicWriter.required_render_vars == frozenset()


def test_basic_writer_writes_ldr_color(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, sample_render_vars, backend)
    assert any("ldr_color" in k for k in backend.written)


def test_basic_writer_accepts_warp_array_inputs() -> None:
    """Warp array inputs (e.g. zero-copy ovrtx views) are normalized to NumPy.

    Regression test for ``OvrtxOutput.extract`` returning Warp arrays: the
    encode and colorize paths must not call NumPy-only methods (``.astype``)
    on Warp arrays.
    """
    import warp as wp

    ldr = np.zeros((8, 8, 4), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    render_vars = {
        "LdrColor": wp.array(ldr, device="cpu"),
        "DistanceToCameraSD": wp.array(depth, device="cpu"),
    }

    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(0, render_vars, backend)

    assert backend.exists("0000/ldr_color.png")
    assert backend.exists("0000/distance_to_camera.exr")
    assert backend.exists("0000/distance_to_camera_colorized.png")


class _HostReadbackCounter:
    """Device-array stand-in that counts host readbacks via ``numpy()``."""

    def __init__(self, host: np.ndarray) -> None:
        self._host = host
        self.shape = host.shape
        self.readbacks = 0

    def numpy(self) -> np.ndarray:
        self.readbacks += 1
        return self._host


@pytest.mark.parametrize(
    ("var_name", "host", "expected_keys"),
    [
        (
            "DistanceToCameraSD",
            np.linspace(0.5, 4.0, 64, dtype=np.float32).reshape(8, 8),
            {"0000/distance_to_camera.exr", "0000/distance_to_camera_colorized.png"},
        ),
        (
            "NormalSD",
            np.full((8, 8, 3), 0.5, dtype=np.float32),
            {"0000/normal.exr", "0000/normal_colorized.png"},
        ),
    ],
)
def test_basic_writer_reads_colorizable_var_back_once(
    var_name, host, expected_keys
) -> None:
    """A colorized var is read back from the device once and output is unchanged."""
    from_device = _HostReadbackCounter(host)
    device_backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0, {var_name: from_device}, device_backend
    )
    assert from_device.readbacks == 1

    host_backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0, {var_name: host}, host_backend
    )
    assert expected_keys <= set(device_backend.written)
    assert set(device_backend.written) == set(host_backend.written)
    for path, (array, encoding) in device_backend.written.items():
        host_array, host_encoding = host_backend.written[path]
        assert encoding == host_encoding
        np.testing.assert_array_equal(array, host_array)


def test_basic_writer_frame_dir_zero_padded(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(frame_id_digits=4)).write_frame(
        7, sample_render_vars, backend
    )
    assert any(k.startswith("0007/") for k in backend.written)


def test_basic_writer_custom_digits(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(frame_id_digits=6)).write_frame(
        3, sample_render_vars, backend
    )
    assert any(k.startswith("000003/") for k in backend.written)


def test_basic_writer_depth_stored_as_npy(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(depth_format="npy")).write_frame(
        0, {"DistanceToCameraSD": sample_render_vars["DistanceToCameraSD"]}, backend
    )
    assert any("distance_to_camera.npy" in k for k in backend.written)


def test_basic_writer_depth_stored_as_exr(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(depth_format="exr", colorize=False)).write_frame(
        0, {"DistanceToCameraSD": sample_render_vars["DistanceToCameraSD"]}, backend
    )
    key = next(k for k in backend.written if "distance_to_camera.exr" in k)
    _arr, enc = backend.written[key]
    assert enc == "exr"


def test_basic_writer_image_format_png_default(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, {"LdrColor": sample_render_vars["LdrColor"]}, backend)
    key = next(k for k in backend.written if "ldr_color" in k)
    assert key.endswith("ldr_color.png")


def test_basic_writer_image_format_jpg(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(image_format="jpg")).write_frame(
        0, {"LdrColor": sample_render_vars["LdrColor"]}, backend
    )
    key = next(k for k in backend.written if "ldr_color" in k)
    assert key.endswith("ldr_color.jpg")
    _arr, enc = backend.written[key]
    assert enc == "jpg"


def test_basic_writer_normals_stored_as_exr(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, sample_render_vars, backend)
    key = next(k for k in backend.written if "normal.exr" in k)
    _arr, enc = backend.written[key]
    assert enc == "exr"


def test_basic_writer_bbox2d_stored_as_json(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, sample_render_vars, backend)
    assert any("bbox_2d_tight.json" in k for k in backend.written)


def test_basic_writer_bbox2d_json_decoded_per_entry() -> None:
    """Bbox JSON is a list of per-entry records with label, prim_path, coords."""
    backend = MockBackend()
    render_vars = {
        "SemanticBoundingBox2DTight": _make_bbox2d([(7, 1, 10, 20, 100, 200)]),
        "SemanticIdMap": _make_id_map({1: "car"}),
        "StableIdMap": _make_id_map({7: "/World/Car"}),
    }
    BasicWriter().write_frame(0, render_vars, backend)
    key = next(k for k in backend.written if "bbox_2d_tight.json" in k)
    entries = json.loads(backend.written[key])
    assert isinstance(entries, list) and len(entries) == 1
    entry = entries[0]
    assert entry["label"] == "car"
    assert entry["prim_path"] == "/World/Car"
    assert entry["instance_id"] == 7
    assert (entry["x_min"], entry["y_min"], entry["x_max"], entry["y_max"]) == (
        10.0,
        20.0,
        100.0,
        200.0,
    )


def test_basic_writer_bbox3d_stored_as_json(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, sample_render_vars, backend)
    assert any("bbox_3d.json" in k for k in backend.written)


def test_basic_writer_camera_params_stored_as_json(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, sample_render_vars, backend)
    key = next(k for k in backend.written if "camera_params.json" in k)
    payload = json.loads(backend.written[key])
    # New format: single dict (not wrapped in a list)
    assert isinstance(payload, dict)
    # Raw fields present
    assert "view_transform" in payload
    assert "projection" in payload
    assert "camera_model" in payload
    assert "focal_length" in payload
    assert "fov_x" in payload
    assert "meters_per_scene_unit" in payload
    assert "resolution" in payload
    assert "near_far" in payload
    # Derived intrinsics present and correct shape
    assert "intrinsics" in payload
    intrinsics = payload["intrinsics"]
    assert "fx" in intrinsics and "fy" in intrinsics
    assert "cx" in intrinsics and "cy" in intrinsics
    assert len(intrinsics["K"]) == 9  # 3x3 flattened
    # Derived extrinsics present
    assert "extrinsics" in payload
    extrinsics = payload["extrinsics"]
    assert len(extrinsics["R_world_to_camera"]) == 9  # 3x3 flattened
    assert len(extrinsics["t_world_to_camera"]) == 3
    # Matrices are flattened 16-element lists
    assert len(payload["view_transform"]) == 16
    assert len(payload["projection"]) == 16
    # Resolution matches fixture
    assert payload["resolution"] == [64, 48] or payload["resolution"] == [640, 480]
    # Intrinsics consistency: fx derived from fov_x and width
    import math

    w = payload["resolution"][0]
    fov_x = payload["fov_x"]
    if fov_x > 0 and w > 0:
        expected_fx = (w / 2.0) / math.tan(fov_x / 2.0)
        assert abs(intrinsics["fx"] - expected_fx) < 1e-4


def test_basic_writer_camera_params_non_identity_extrinsics() -> None:
    """Verify extrinsics are correctly derived from a non-identity view_transform."""
    import math as _math

    backend = MockBackend()
    arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
    # Camera at (3, 4, 5), rotated 45 deg around Y
    c = _math.cos(_math.radians(45))
    s = _math.sin(_math.radians(45))
    arr[0]["view_transform"] = [
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [3, 4, 5, 1],
    ]  # camera-to-world with translation in last row (row-vector convention)
    arr[0]["projection"] = np.eye(4)
    arr[0]["resolution"] = (640, 480)
    arr[0]["clipping_plane"] = (0.1, 100.0)
    arr[0]["projection_model"] = b"pinhole"
    arr[0]["fov_x"] = np.radians(90.0)
    BasicWriter().write_frame(0, {"CameraParams": arr}, backend)
    key = next(k for k in backend.written if "camera_params.json" in k)
    payload = json.loads(backend.written[key])
    ext = payload["extrinsics"]
    # world-to-camera t should recover -R^T @ translation
    # With this view_transform (translation in last row), inv gives us
    # the world-to-camera transform
    view = np.array(arr[0]["view_transform"], dtype=np.float64).reshape(4, 4)
    expected_w2c = np.linalg.inv(view)
    np.testing.assert_allclose(
        ext["R_world_to_camera"],
        expected_w2c[:3, :3].ravel().tolist(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ext["t_world_to_camera"],
        expected_w2c[:3, 3].tolist(),
        atol=1e-5,
    )
    # Intrinsics: fov=90 deg, width=640 → fx = 320 / tan(45°) = 320
    assert abs(payload["intrinsics"]["fx"] - 320.0) < 1e-4


def test_basic_writer_skips_empty_exr_aov() -> None:
    """An AOV emitted as an empty buffer is skipped, not encoded as EXR."""
    backend = MockBackend()
    render_vars = {
        "NormalSD": np.empty((0,), dtype=np.uint8),
        "LdrColor": np.zeros((8, 8, 4), dtype=np.uint8),
    }
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(0, render_vars, backend)
    assert not any("normal" in k for k in backend.written)
    assert backend.exists("0000/ldr_color.png")


def test_basic_writer_motion2d_stored_as_exr() -> None:
    backend = MockBackend()
    motion = np.zeros((8, 8, 4), dtype=np.float32)
    BasicWriter().write_frame(0, {"Motion2d": motion}, backend)
    key = next(k for k in backend.written if "motion_2d.exr" in k)
    _arr, enc = backend.written[key]
    assert enc == "exr"


def test_basic_writer_exports_runtime_verified_ovrtx_additions() -> None:
    backend = MockBackend()
    stable_semantic = np.zeros(
        1,
        dtype=[("stable_id", "<u4", (4,)), ("semantic_id", "<u4", (4,))],
    )
    instance_map = np.array(
        [(0, 0), (1, 0xFFFFFFFF)],
        dtype=[("instance_index", "<u4"), ("stable_semantic_index", "<u4")],
    )
    render_vars = {
        "TargetMotionSD": np.zeros((8, 8, 4), dtype=np.float32),
        "NonStableInstanceSegmentation": np.zeros((8, 8, 1), dtype=np.uint32),
        "StableIdSegmentation": np.zeros((8, 8, 4), dtype=np.uint32),
        "SemanticIdMap": {7: "class: robot;"},
        "StableIdMap": {101: "/World/Robot"},
        "StableIdSemanticIdMap": stable_semantic,
        "InstanceMap": instance_map,
    }

    BasicWriter(BasicWriterConfig(colorize=False)).write_frame(0, render_vars, backend)

    assert backend.exists("0000/target_motion.exr")
    assert backend.exists("0000/non_stable_instance_segmentation.npy")
    assert backend.exists("0000/stable_id_segmentation.npy")

    # Content, not just existence: a byte-dump of the raw buffer would also
    # satisfy exists().
    assert _read_json(backend, "0000/semantic_id_map.json") == {"7": "class: robot;"}
    assert _read_json(backend, "0000/stable_id_map.json") == {"101": "/World/Robot"}
    assert _read_json(backend, "0000/stable_id_semantic_id_map.json") == [
        {"stable_id": [0, 0, 0, 0], "semantic_id": [0, 0, 0, 0]}
    ]
    assert _read_json(backend, "0000/instance_map.json") == [
        {"instance_index": 0, "stable_semantic_index": 0},
        {"instance_index": 1, "stable_semantic_index": 0xFFFFFFFF},
    ]


def _read_json(backend, path: str):
    """Decode a JSON artifact written to ``backend``."""
    raw = backend.written[path]
    return json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)


def test_basic_writer_exports_raw_ovrtx_buffers_decoded() -> None:
    """Undecoded ovrtx buffers are decoded, not dumped as byte lists.

    ``OvrtxOutput.extract(decode=False)`` (and any non-ovrtx producer) hands the
    writer raw payloads. Those must land as the same JSON the decoded form
    produces, rather than as a list of uint8 values.
    """
    labels = {7: "class: robot;"}
    entry = np.dtype([("sid", "<u4", (4,)), ("len", "<u4"), ("off", "<u4")])
    entries = np.zeros(len(labels), dtype=entry)
    table = b""
    header_size = len(labels) * entry.itemsize
    for i, (sid, label) in enumerate(labels.items()):
        entries[i]["sid"][0] = sid
        entries[i]["len"] = len(label)
        entries[i]["off"] = header_size + len(table)
        table += label.encode()
    raw_id_map = np.frombuffer(
        entries.tobytes() + table + len(labels).to_bytes(4, "little"), dtype=np.uint8
    ).copy()

    raw_stable_semantic = np.zeros(
        1, dtype=[("stable_id", "<u4", (4,)), ("semantic_id", "<u4", (4,))]
    ).view(np.uint8)
    raw_instance_map = np.array([3, 0xFFFFFFFF], dtype="<u4").view(np.uint8)

    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=False)).write_frame(
        0,
        {
            "SemanticIdMap": raw_id_map,
            "StableIdSemanticIdMap": raw_stable_semantic,
            "InstanceMap": raw_instance_map,
        },
        backend,
    )

    assert _read_json(backend, "0000/semantic_id_map.json") == {"7": "class: robot;"}
    assert _read_json(backend, "0000/stable_id_semantic_id_map.json") == [
        {"stable_id": [0, 0, 0, 0], "semantic_id": [0, 0, 0, 0]}
    ]
    assert _read_json(backend, "0000/instance_map.json") == [
        {"instance_index": 0, "stable_semantic_index": 3},
        {"instance_index": 1, "stable_semantic_index": 0xFFFFFFFF},
    ]


def test_basic_writer_point_cloud_stored_as_npz() -> None:
    backend = MockBackend()
    flat = np.arange(32, dtype=np.uint8)
    BasicWriter().write_frame(0, {"PointCloud": flat}, backend)
    assert any("point_cloud.npz" in k for k in backend.written)


def test_basic_writer_composite_point_cloud_keeps_named_tensors() -> None:
    """A composite sensor output lands as one npz member per tensor/param."""
    backend = MockBackend()
    point_cloud = {
        "tensors": {
            "Coordinates": np.arange(9, dtype=np.float32).reshape(3, 3),
            "Intensity": np.arange(3, dtype=np.float32),
        },
        "params": {"frameId": np.array([7], dtype=np.uint64)},
    }
    BasicWriter().write_frame(0, {"PointCloud": point_cloud}, backend)

    archive = np.load(io.BytesIO(backend.read("0000/point_cloud.npz")))
    assert sorted(archive.files) == ["Coordinates", "Intensity", "param_frameId"]
    np.testing.assert_array_equal(
        archive["Coordinates"], point_cloud["tensors"]["Coordinates"]
    )
    np.testing.assert_array_equal(archive["param_frameId"], np.array([7]))


def test_basic_writer_semantic_id_map_written(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, sample_render_vars, backend)
    assert backend.exists("0000/semantic_id_map.json")


def test_basic_writer_instance_seg_stored_as_npy(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(0, sample_render_vars, backend)
    assert any("instance_segmentation.npy" in k for k in backend.written)


def test_basic_writer_respects_render_vars_list(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(render_vars=["LdrColor"])).write_frame(
        0, sample_render_vars, backend
    )
    # Only LdrColor.* should be written; no depth, normals, etc.
    non_image = [k for k in backend.written if "ldr_color" not in k]
    assert non_image == []


def test_basic_writer_skips_var_not_in_render_vars(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(
        BasicWriterConfig(render_vars=["LdrColor", "DistanceToCameraSD"])
    ).write_frame(0, sample_render_vars, backend)
    # NormalSD should NOT be written
    assert not any("normal" in k for k in backend.written)


def test_basic_writer_finalize_is_noop(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BasicWriter()
    writer.write_frame(0, sample_render_vars, backend)
    count_before = len(backend.written)
    writer.finalize(backend)
    assert len(backend.written) == count_before


def test_basic_writer_colorize_true_writes_depth_colorized(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0, {"DistanceToCameraSD": sample_render_vars["DistanceToCameraSD"]}, backend
    )
    assert any("distance_to_camera_colorized.png" in k for k in backend.written)


def test_basic_writer_colorize_true_writes_normals_colorized(
    sample_render_vars,
) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0, {"NormalSD": sample_render_vars["NormalSD"]}, backend
    )
    assert any("normal_colorized.png" in k for k in backend.written)


def test_basic_writer_colorize_true_writes_instance_seg_colorized(
    sample_render_vars,
) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0,
        {
            "SemanticInstanceSegmentation": sample_render_vars[
                "SemanticInstanceSegmentation"
            ]
        },
        backend,
    )
    assert any("instance_segmentation_colorized.png" in k for k in backend.written)


def test_basic_writer_colorize_true_writes_semantic_seg_colorized() -> None:
    rng = np.random.default_rng(7)
    sem_seg = rng.integers(0, 4, (48, 64), dtype=np.uint32)
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0, {"SemanticSegmentation": sem_seg}, backend
    )
    assert any("semantic_segmentation_colorized.png" in k for k in backend.written)


def test_basic_writer_colorize_false_skips_colorized(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=False)).write_frame(
        0, sample_render_vars, backend
    )
    assert not any("_colorized" in k for k in backend.written)


def test_basic_writer_colorize_default_is_true(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter().write_frame(
        0, {"DistanceToCameraSD": sample_render_vars["DistanceToCameraSD"]}, backend
    )
    assert any("distance_to_camera_colorized.png" in k for k in backend.written)


def test_basic_writer_colorized_output_is_uint8_rgb(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0, {"DistanceToCameraSD": sample_render_vars["DistanceToCameraSD"]}, backend
    )
    key = next(k for k in backend.written if "distance_to_camera_colorized.png" in k)
    arr, enc = backend.written[key]
    assert enc == "png"
    assert arr.dtype == np.uint8
    assert arr.ndim == 3 and arr.shape[2] == 3


def test_basic_writer_colorize_raw_still_written(sample_render_vars) -> None:
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True)).write_frame(
        0, {"DistanceToCameraSD": sample_render_vars["DistanceToCameraSD"]}, backend
    )
    assert any("distance_to_camera.exr" in k for k in backend.written)


def test_basic_writer_depth_near_bright_true_near_pixels_brighter() -> None:
    depth = np.array([[1.0, 100.0]], dtype=np.float32)
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True, depth_near_bright=True)).write_frame(
        0, {"DistanceToCameraSD": depth}, backend
    )
    key = next(k for k in backend.written if "distance_to_camera_colorized.png" in k)
    arr, _ = backend.written[key]
    near_brightness = int(arr[0, 0].sum())
    far_brightness = int(arr[0, 1].sum())
    assert near_brightness > far_brightness, (
        f"near pixel ({near_brightness}) should be brighter than far ({far_brightness})"
    )


def test_basic_writer_depth_near_bright_false_far_pixels_brighter() -> None:
    depth = np.array([[1.0, 100.0]], dtype=np.float32)
    backend = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True, depth_near_bright=False)).write_frame(
        0, {"DistanceToCameraSD": depth}, backend
    )
    key = next(k for k in backend.written if "distance_to_camera_colorized.png" in k)
    arr, _ = backend.written[key]
    near_brightness = int(arr[0, 0].sum())
    far_brightness = int(arr[0, 1].sum())
    assert far_brightness > near_brightness, (
        f"far pixel ({far_brightness}) should be brighter than near ({near_brightness})"
    )


def test_basic_writer_depth_near_bright_default_is_true() -> None:
    assert BasicWriterConfig().depth_near_bright is True


def test_basic_writer_depth_near_bright_modes_produce_different_output() -> None:
    depth = np.linspace(1.0, 50.0, 16, dtype=np.float32).reshape(4, 4)
    backend_a = MockBackend()
    backend_b = MockBackend()
    BasicWriter(BasicWriterConfig(colorize=True, depth_near_bright=True)).write_frame(
        0, {"DistanceToCameraSD": depth}, backend_a
    )
    BasicWriter(BasicWriterConfig(colorize=True, depth_near_bright=False)).write_frame(
        0, {"DistanceToCameraSD": depth}, backend_b
    )
    key = "0000/distance_to_camera_colorized.png"
    arr_a, _ = backend_a.written[key]
    arr_b, _ = backend_b.written[key]
    assert not np.array_equal(arr_a, arr_b)


# ---------------------------------------------------------------------------
# COCOWriter
# ---------------------------------------------------------------------------


def test_coco_writer_name() -> None:
    assert COCOWriter.name == "coco"


def test_coco_writer_required_vars() -> None:
    assert "LdrColor" in COCOWriter.required_render_vars
    assert "SemanticBoundingBox2DTight" in COCOWriter.required_render_vars


def test_coco_writer_missing_ldr_raises(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    incomplete = {k: v for k, v in sample_render_vars.items() if k != "LdrColor"}
    with pytest.raises(MissingRenderVarError) as exc_info:
        writer.write_frame(0, incomplete, backend)
    assert exc_info.value.writer_name == "coco"


def test_coco_writer_missing_bbox_raises(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    incomplete = {
        k: v for k, v in sample_render_vars.items() if k != "SemanticBoundingBox2DTight"
    }
    with pytest.raises(MissingRenderVarError):
        writer.write_frame(0, incomplete, backend)


def test_coco_writer_writes_image_file(sample_render_vars) -> None:
    backend = MockBackend()
    COCOWriter(COCOWriterConfig(image_format="jpg")).write_frame(
        0, sample_render_vars, backend
    )
    assert "images/000000.jpg" in backend.written


def test_coco_writer_image_format_png(sample_render_vars) -> None:
    backend = MockBackend()
    COCOWriter(COCOWriterConfig(image_format="png")).write_frame(
        0, sample_render_vars, backend
    )
    assert "images/000000.png" in backend.written


def test_coco_writer_finalize_writes_json(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    assert "annotations/instances_train.json" in backend.written
    coco = json.loads(backend.written["annotations/instances_train.json"])
    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco


def test_coco_writer_annotation_bbox_is_xywh(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    coco = json.loads(backend.written["annotations/instances_train.json"])
    for ann in coco["annotations"]:
        assert len(ann["bbox"]) == 4
        assert ann["bbox"][2] > 0
        assert ann["bbox"][3] > 0


def test_coco_writer_annotation_has_required_keys(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    coco = json.loads(backend.written["annotations/instances_train.json"])
    ann = coco["annotations"][0]
    for key in (
        "id",
        "image_id",
        "category_id",
        "bbox",
        "area",
        "iscrowd",
        "segmentation",
    ):
        assert key in ann, f"missing key: {key}"


def test_coco_writer_segmentation_empty_when_disabled(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter(COCOWriterConfig(include_segmentation=False))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    coco = json.loads(backend.written["annotations/instances_train.json"])
    for ann in coco["annotations"]:
        assert ann["segmentation"] == []


def test_coco_writer_accumulates_across_frames(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.write_frame(1, sample_render_vars, backend)
    writer.finalize(backend)
    coco = json.loads(backend.written["annotations/instances_train.json"])
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 4  # 2 frames × 2 bboxes


def test_coco_writer_custom_split(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter(COCOWriterConfig(split="val"))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    assert "annotations/instances_val.json" in backend.written


def test_coco_writer_finalize_idempotent(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    count = len(backend.written)
    writer.finalize(backend)
    assert len(backend.written) == count


def test_coco_writer_categories_populated(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    coco = json.loads(backend.written["annotations/instances_train.json"])
    category_names = {c["name"] for c in coco["categories"]}
    assert "car" in category_names
    assert "person" in category_names


def _coco_first_bbox_area(sample_render_vars, bbox_mode: str) -> float:
    backend = MockBackend()
    writer = COCOWriter(COCOWriterConfig(bbox_mode=bbox_mode))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    coco = json.loads(backend.written["annotations/instances_train.json"])
    return float(coco["annotations"][0]["area"])


def test_coco_writer_loose_bbox_larger_than_tight(sample_render_vars) -> None:
    tight_area = _coco_first_bbox_area(sample_render_vars, "tight")
    loose_area = _coco_first_bbox_area(sample_render_vars, "loose")
    assert loose_area > tight_area, (
        f"loose area {loose_area} should exceed tight area {tight_area}"
    )


def test_coco_writer_with_segmentation_rle(sample_render_vars) -> None:
    backend = MockBackend()
    writer = COCOWriter(COCOWriterConfig(include_segmentation=True))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    coco = json.loads(backend.written["annotations/instances_train.json"])
    assert any(
        isinstance(ann["segmentation"], dict) and ann["segmentation"].get("counts")
        for ann in coco["annotations"]
    ), "expected at least one RLE segmentation"


def test_coco_writer_panoptic_requires_instance_segmentation() -> None:
    backend = MockBackend()
    render_vars = _make_coco_panoptic_render_vars()
    render_vars.pop("SemanticInstanceSegmentation")
    writer = COCOWriter(COCOWriterConfig(include_panoptic=True))
    with pytest.raises(MissingRenderVarError) as exc_info:
        writer.write_frame(0, render_vars, backend)
    assert exc_info.value.missing_render_var == "SemanticInstanceSegmentation"


def test_coco_writer_panoptic_outputs_json_and_mask() -> None:
    backend = MockBackend()
    writer = COCOWriter(COCOWriterConfig(include_panoptic=True))
    render_vars = _make_coco_panoptic_render_vars()
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)

    assert "annotations/panoptic_train.json" in backend.written
    assert "annotations/panoptic_train/000000.png" in backend.written
    panoptic = json.loads(backend.written["annotations/panoptic_train.json"])
    assert {"images", "annotations", "categories"}.issubset(panoptic.keys())
    assert len(panoptic["annotations"]) == 1


def test_coco_writer_panoptic_segments_match_encoded_mask() -> None:
    backend = MockBackend()
    writer = COCOWriter(COCOWriterConfig(include_panoptic=True))
    writer.write_frame(0, _make_coco_panoptic_render_vars(), backend)
    writer.finalize(backend)

    panoptic = json.loads(backend.written["annotations/panoptic_train.json"])
    panoptic_ann = panoptic["annotations"][0]
    panoptic_img, encoding = backend.written["annotations/panoptic_train/000000.png"]
    assert encoding == "png"
    decoded_ids = _decode_panoptic_ids(panoptic_img)

    expected_segment_ids = {int(seg["id"]) for seg in panoptic_ann["segments_info"]}
    image_segment_ids = {int(v) for v in np.unique(decoded_ids) if int(v) > 0}
    assert image_segment_ids == expected_segment_ids

    for segment in panoptic_ann["segments_info"]:
        seg_id = int(segment["id"])
        mask = decoded_ids == seg_id
        ys, xs = np.where(mask)
        assert ys.size > 0 and xs.size > 0
        assert int(np.count_nonzero(mask)) == int(segment["area"])
        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        assert bbox == [int(v) for v in segment["bbox"]]


# ---------------------------------------------------------------------------
# KITTIWriter
# ---------------------------------------------------------------------------


def test_kitti_writer_name() -> None:
    assert KITTIWriter.name == "kitti"


def test_kitti_writer_required_vars() -> None:
    req = KITTIWriter.required_render_vars
    assert {
        "LdrColor",
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox3D",
        "CameraParams",
    }.issubset(req)


def test_kitti_writer_missing_camera_params_raises(sample_render_vars) -> None:
    backend = MockBackend()
    writer = KITTIWriter()
    incomplete = {k: v for k, v in sample_render_vars.items() if k != "CameraParams"}
    with pytest.raises(MissingRenderVarError) as exc_info:
        writer.write_frame(0, incomplete, backend)
    assert exc_info.value.missing_render_var == "CameraParams"


def test_kitti_writer_writes_image(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    assert any("image_2/000000.png" in k for k in backend.written)


def test_kitti_writer_writes_label_file(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    assert "label_2/000000.txt" in backend.written


def test_kitti_writer_label_has_content(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    label_txt = backend.written["label_2/000000.txt"].decode()
    assert len(label_txt.strip()) > 0


def test_kitti_writer_label_format(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    label_txt = backend.written["label_2/000000.txt"].decode().strip()
    first_line = label_txt.splitlines()[0]
    parts = first_line.split()
    assert len(parts) == 15
    for tok in parts[1:]:
        float(tok)


def test_kitti_writer_label_uses_resolved_class_names(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    label_txt = backend.written["label_2/000000.txt"].decode().strip()
    types = {ln.split()[0] for ln in label_txt.splitlines()}
    assert types == {"car", "person"}


def test_kitti_writer_writes_calib_file(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    assert "calib/000000.txt" in backend.written
    calib = backend.written["calib/000000.txt"].decode()
    assert "P2:" in calib


def test_kitti_calib_r0_rect_is_identity(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    calib = backend.written["calib/000000.txt"].decode()
    r0_line = next(ln for ln in calib.splitlines() if ln.startswith("R0_rect:"))
    vals = list(map(float, r0_line.split(":")[1].split()))
    assert len(vals) == 9
    mat = np.array(vals).reshape(3, 3)
    np.testing.assert_allclose(mat, np.eye(3), atol=1e-6)


def test_kitti_calib_tr_is_identity_transform(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    calib = backend.written["calib/000000.txt"].decode()
    tr_line = next(ln for ln in calib.splitlines() if ln.startswith("Tr_velo_to_cam:"))
    vals = list(map(float, tr_line.split(":")[1].split()))
    assert len(vals) == 12
    mat = np.array(vals).reshape(3, 4)
    expected = np.hstack([np.eye(3), np.zeros((3, 1))])
    np.testing.assert_allclose(mat, expected, atol=1e-6)


def test_kitti_label_alpha_in_range(sample_render_vars) -> None:
    import math

    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    label_txt = backend.written["label_2/000000.txt"].decode().strip()
    for line in label_txt.splitlines():
        parts = line.split()
        alpha = float(parts[3])
        assert -math.pi <= alpha <= math.pi, f"alpha={alpha} out of [-pi, pi]"


def test_kitti_writer_no_depth_by_default(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter().write_frame(0, sample_render_vars, backend)
    assert not any("depth/" in k for k in backend.written)


def test_kitti_writer_with_depth(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter(KITTIWriterConfig(write_depth=True)).write_frame(
        0, sample_render_vars, backend
    )
    assert any("depth/" in k for k in backend.written)


def test_kitti_writer_depth_scale_encodes_value(sample_render_vars) -> None:
    rv = dict(sample_render_vars)
    h, w = rv["DistanceToCameraSD"].shape[:2]
    rv["DistanceToCameraSD"] = np.full((h, w), 2.0, dtype=np.float32)
    backend = MockBackend()
    KITTIWriter(KITTIWriterConfig(write_depth=True, depth_scale=500.0)).write_frame(
        0, rv, backend
    )
    key = next(k for k in backend.written if k.startswith("depth/"))
    arr, enc = backend.written[key]
    assert enc == "png"
    assert arr.dtype == np.uint16
    # depth_uint16 = round(2.0 * 500.0) = 1000
    assert int(arr.flat[0]) == 1000


def test_kitti_writer_class_filter(sample_render_vars) -> None:
    backend = MockBackend()
    KITTIWriter(KITTIWriterConfig(classes=["car"])).write_frame(
        0, sample_render_vars, backend
    )
    label_txt = backend.written["label_2/000000.txt"].decode().strip()
    assert len(label_txt) > 0
    for line in label_txt.splitlines():
        assert line.startswith("car")


def test_kitti_writer_finalize_is_noop(sample_render_vars) -> None:
    backend = MockBackend()
    writer = KITTIWriter()
    writer.write_frame(0, sample_render_vars, backend)
    count_before = len(backend.written)
    writer.finalize(backend)
    assert len(backend.written) == count_before


# ---------------------------------------------------------------------------
# YOLOWriter
# ---------------------------------------------------------------------------


def test_yolo_writer_name() -> None:
    assert YOLOWriter.name == "yolo"


def test_yolo_writer_required_vars() -> None:
    assert {"LdrColor", "SemanticBoundingBox2DTight"}.issubset(
        YOLOWriter.required_render_vars
    )


def test_yolo_writer_segment_task_requires_instance_seg() -> None:
    writer = YOLOWriter(YOLOWriterConfig(task="segment"))
    assert "SemanticInstanceSegmentation" in writer.required_render_vars


def test_yolo_writer_missing_bbox_raises(sample_render_vars) -> None:
    backend = MockBackend()
    writer = YOLOWriter()
    incomplete = {"LdrColor": sample_render_vars["LdrColor"]}
    with pytest.raises(MissingRenderVarError):
        writer.write_frame(0, incomplete, backend)


def test_yolo_writer_writes_image(sample_render_vars) -> None:
    backend = MockBackend()
    YOLOWriter().write_frame(1, sample_render_vars, backend)  # frame 1 → train
    assert any("images/" in k and ".jpg" in k for k in backend.written)


def test_yolo_writer_image_format_png(sample_render_vars) -> None:
    backend = MockBackend()
    YOLOWriter(YOLOWriterConfig(image_format="png", val_split=0.01)).write_frame(
        1, sample_render_vars, backend
    )
    assert any("images/" in k and k.endswith(".png") for k in backend.written)


def test_yolo_writer_writes_label_file(sample_render_vars) -> None:
    backend = MockBackend()
    YOLOWriter().write_frame(1, sample_render_vars, backend)
    assert any("labels/" in k and ".txt" in k for k in backend.written)


def test_yolo_writer_labels_are_normalized(sample_render_vars) -> None:
    backend = MockBackend()
    YOLOWriter(YOLOWriterConfig(val_split=0.01)).write_frame(
        1, sample_render_vars, backend
    )
    label_key = next(k for k in backend.written if "labels/" in k and ".txt" in k)
    label_txt = backend.written[label_key].decode().strip()
    for line in label_txt.splitlines():
        parts = list(map(float, line.split()))
        assert 0.0 <= parts[1] <= 1.0
        assert 0.0 <= parts[2] <= 1.0
        assert 0.0 <= parts[3] <= 1.0
        assert 0.0 <= parts[4] <= 1.0


def test_yolo_writer_val_split_assigns_frames(sample_render_vars) -> None:
    backend = MockBackend()
    writer = YOLOWriter(YOLOWriterConfig(val_split=0.5))
    writer.write_frame(0, sample_render_vars, backend)  # val (0 % 2 == 0)
    writer.write_frame(1, sample_render_vars, backend)  # train
    assert any("images/val/" in k for k in backend.written)
    assert any("images/train/" in k for k in backend.written)


def test_yolo_writer_finalize_writes_data_yaml(sample_render_vars) -> None:
    backend = MockBackend()
    writer = YOLOWriter()
    writer.write_frame(1, sample_render_vars, backend)
    writer.finalize(backend)
    assert "data.yaml" in backend.written
    yaml_txt = backend.written["data.yaml"].decode()
    assert "nc:" in yaml_txt
    assert "names:" in yaml_txt


def test_yolo_writer_data_yaml_has_correct_class_count(sample_render_vars) -> None:
    backend = MockBackend()
    writer = YOLOWriter()
    writer.write_frame(1, sample_render_vars, backend)
    writer.finalize(backend)
    yaml_txt = backend.written["data.yaml"].decode()
    nc_line = next(ln for ln in yaml_txt.splitlines() if ln.startswith("nc:"))
    assert nc_line == "nc: 2"


def test_yolo_writer_explicit_classes_fixed_mapping(sample_render_vars) -> None:
    backend = MockBackend()
    writer = YOLOWriter(YOLOWriterConfig(classes=["person", "car"], val_split=0.01))
    writer.write_frame(1, sample_render_vars, backend)
    writer.finalize(backend)
    label_key = next(k for k in backend.written if "labels/" in k and ".txt" in k)
    label_txt = backend.written[label_key].decode().strip()
    class_ids = {int(ln.split()[0]) for ln in label_txt.splitlines()}
    # person -> 0, car -> 1 (fixed by the provided list order)
    assert class_ids == {0, 1}
    yaml_txt = backend.written["data.yaml"].decode()
    assert "['person', 'car']" in yaml_txt


def test_yolo_writer_unknown_label_maps_to_class_zero() -> None:
    backend = MockBackend()
    rng = np.random.default_rng(1)
    rv = {
        "LdrColor": rng.integers(0, 255, (64, 64, 4), dtype=np.uint8),
        "SemanticBoundingBox2DTight": _make_bbox2d([(1, 9, 5, 5, 20, 20)]),
        "SemanticIdMap": _make_id_map({9: "unicorn"}),
    }
    writer = YOLOWriter(YOLOWriterConfig(classes=["car", "person"], val_split=0.01))
    writer.write_frame(1, rv, backend)
    label_key = next(k for k in backend.written if "labels/" in k and ".txt" in k)
    label_txt = backend.written[label_key].decode().strip()
    # "unicorn" is not in the fixed class list -> falls back to class 0
    assert label_txt.split()[0] == "0"


def test_yolo_writer_segment_task(sample_render_vars) -> None:
    backend = MockBackend()
    config = YOLOWriterConfig(task="segment", image_format="png", val_split=0.01)
    writer = YOLOWriter(config)
    writer.write_frame(1, sample_render_vars, backend)
    writer.finalize(backend)
    assert any("labels/" in k and ".txt" in k for k in backend.written)


def test_yolo_writer_finalize_idempotent(sample_render_vars) -> None:
    backend = MockBackend()
    writer = YOLOWriter()
    writer.write_frame(1, sample_render_vars, backend)
    writer.finalize(backend)
    count_before = len(backend.written)
    writer.finalize(backend)
    assert len(backend.written) == count_before


# ---------------------------------------------------------------------------
# BOPWriter
# ---------------------------------------------------------------------------


def test_bop_writer_name() -> None:
    assert BOPWriter.name == "bop"


def test_bop_writer_required_vars() -> None:
    required = BOPWriter.required_render_vars
    for var in (
        "LdrColor",
        "DistanceToCameraSD",
        "SemanticBoundingBox2DTight",
        "SemanticBoundingBox3D",
        "SemanticInstanceSegmentation",
        "CameraParams",
    ):
        assert var in required


def test_bop_writer_missing_depth_raises(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BOPWriter()
    incomplete = {
        k: v for k, v in sample_render_vars.items() if k != "DistanceToCameraSD"
    }
    with pytest.raises(MissingRenderVarError) as exc_info:
        writer.write_frame(0, incomplete, backend)
    assert exc_info.value.missing_render_var == "DistanceToCameraSD"


def test_bop_writer_writes_rgb(sample_render_vars) -> None:
    backend = MockBackend()
    BOPWriter(BOPWriterConfig(scene_id=0)).write_frame(0, sample_render_vars, backend)
    assert any("rgb/000000.png" in k for k in backend.written)


def test_bop_writer_writes_depth(sample_render_vars) -> None:
    backend = MockBackend()
    BOPWriter().write_frame(0, sample_render_vars, backend)
    assert any("depth/000000.png" in k for k in backend.written)


def test_bop_writer_writes_masks(sample_render_vars) -> None:
    backend = MockBackend()
    BOPWriter().write_frame(0, sample_render_vars, backend)
    masks = {k: v for k, v in backend.written.items() if "/mask/" in k}
    assert masks
    for path, payload in masks.items():
        visib = backend.written[path.replace("/mask/", "/mask_visib/")]
        # No occlusion model: both masks are the same PNG.
        assert visib == payload
        assert payload.startswith(b"\x89PNG")


def test_bop_writer_finalize_writes_scene_json(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BOPWriter(BOPWriterConfig(scene_id=0))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    sd = "000000"
    assert f"{sd}/scene_camera.json" in backend.written
    assert f"{sd}/scene_gt.json" in backend.written
    assert f"{sd}/scene_gt_info.json" in backend.written


def test_bop_writer_scene_camera_has_cam_K(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BOPWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    sd = f"{writer.config.scene_id:06d}"
    cam_data = json.loads(backend.written[f"{sd}/scene_camera.json"])
    frame_entry = cam_data["0"]
    assert "cam_K" in frame_entry
    assert len(frame_entry["cam_K"]) == 9


def test_bop_writer_depth_scale_in_scene_camera(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BOPWriter(BOPWriterConfig(depth_scale=5.0))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    sd = f"{writer.config.scene_id:06d}"
    cam_data = json.loads(backend.written[f"{sd}/scene_camera.json"])
    assert cam_data["0"]["depth_scale"] == 5.0


def test_bop_writer_object_id_map(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BOPWriter(BOPWriterConfig(object_id_map={"car": 42, "person": 7}))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    sd = f"{writer.config.scene_id:06d}"
    gt = json.loads(backend.written[f"{sd}/scene_gt.json"])
    obj_ids = {entry["obj_id"] for entry in gt["0"]}
    assert obj_ids == {42, 7}
    assert any(f"{sd}/mask/000000_000042.png" == k for k in backend.written)


def test_bop_writer_finalize_idempotent(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BOPWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    count_before = len(backend.written)
    writer.finalize(backend)
    assert len(backend.written) == count_before


def test_bop_writer_custom_scene_id(sample_render_vars) -> None:
    backend = MockBackend()
    writer = BOPWriter(BOPWriterConfig(scene_id=5))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    assert any("000005/" in k for k in backend.written)


# ---------------------------------------------------------------------------
# LeRobotWriter
# ---------------------------------------------------------------------------


def test_lerobot_writer_name() -> None:
    assert LeRobotWriter.name == "lerobot"


def test_lerobot_writer_required_vars() -> None:
    assert "LdrColor" in LeRobotWriter.required_render_vars
    # CameraParams is not consumed by the writer and is no longer required.
    assert "CameraParams" not in LeRobotWriter.required_render_vars


def test_lerobot_writer_missing_ldr_raises() -> None:
    backend = MockBackend()
    writer = LeRobotWriter()
    with pytest.raises(MissingRenderVarError) as exc_info:
        writer.write_frame(0, {}, backend)
    assert exc_info.value.missing_render_var == "LdrColor"


def test_lerobot_writer_accumulates_rows(sample_render_vars) -> None:
    backend = MockBackend()
    writer = LeRobotWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.write_frame(1, sample_render_vars, backend)
    assert len(writer._rows) == 2


def test_lerobot_writer_row_frame_index_increments(sample_render_vars) -> None:
    backend = MockBackend()
    writer = LeRobotWriter()
    writer.write_frame(0, sample_render_vars, backend)
    writer.write_frame(1, sample_render_vars, backend)
    assert writer._rows[0]["frame_index"] == 0
    assert writer._rows[1]["frame_index"] == 1


def test_lerobot_writer_row_has_required_schema(sample_render_vars) -> None:
    backend = MockBackend()
    writer = LeRobotWriter()
    writer.write_frame(0, sample_render_vars, backend)
    row = writer._rows[0]
    for key in ("index", "episode_index", "frame_index", "timestamp", "task_index"):
        assert key in row, f"missing parquet column: {key}"


def test_lerobot_writer_accumulates_images(sample_render_vars) -> None:
    backend = MockBackend()
    writer = LeRobotWriter(LeRobotWriterConfig(camera_keys=["top"]))
    writer.write_frame(0, sample_render_vars, backend)
    writer.write_frame(1, sample_render_vars, backend)
    assert len(writer._frames) == 2


@pytest.mark.skipif(
    not _has_functional_nvvideocodec(),
    reason="PyNvVideoCodec not functional (missing CUDA libs or NVENC)",
)
def test_lerobot_writer_finalize_writes_meta(sample_render_vars) -> None:
    backend = MockBackend()
    writer = LeRobotWriter(LeRobotWriterConfig(fps=5))
    writer.write_frame(0, sample_render_vars, backend)
    writer.finalize(backend)
    assert "meta/info.json" in backend.written


# ===========================================================================
# writers __init__ entry-point loading
# ===========================================================================


def test_load_entry_points_no_crash():
    """Ensure _load_entry_points runs without error in the test env."""
    from ovannotators.writers import WriterRegistry, _load_entry_points

    _load_entry_points()  # Should not raise

    # The built-in writers stay registered after loading entry points.
    names = WriterRegistry.list_names()
    assert isinstance(names, list)
    assert "coco" in names


def test_load_entry_points_handles_exception():
    """If entry_points raises, it's caught and the registry is left intact."""
    from unittest.mock import patch

    with patch(
        "ovannotators.writers._imeta.entry_points",
        side_effect=RuntimeError("boom"),
    ):
        from ovannotators.writers import WriterRegistry, _load_entry_points

        assert _load_entry_points() is None  # swallows the exception
        assert "coco" in WriterRegistry.list_names()


def test_load_entry_points_handles_broken_plugin(caplog):
    """If a specific entry point fails to load, others still work."""
    from unittest.mock import MagicMock, patch

    broken_ep = MagicMock()
    broken_ep.name = "broken_writer"
    broken_ep.load.side_effect = ImportError("broken plugin")

    with patch(
        "ovannotators.writers._imeta.entry_points",
        return_value=[broken_ep],
    ):
        from ovannotators.writers import WriterRegistry, _load_entry_points

        with caplog.at_level("WARNING", logger="ovannotators.writers"):
            assert _load_entry_points() is None  # broken plugin is swallowed
        # The broken plugin never registered, but built-ins remain.
        assert "coco" in WriterRegistry.list_names()
    # The failure is reported rather than silently dropped.
    assert "broken_writer" in caplog.text


# ---------------------------------------------------------------------------
# BOPWriter — bbox derived from mask when 2D box absent
# ---------------------------------------------------------------------------


def test_bop_bbox_from_mask_and_empty(tmp_path):
    h, w = 16, 24
    inst_seg = np.zeros((h, w), dtype=np.uint32)
    inst_seg[2:6, 3:9] = 99  # instance 99 has pixels but no 2D box
    render_vars = {
        "LdrColor": np.zeros((h, w, 4), dtype=np.uint8),
        "DistanceToCameraSD": np.ones((h, w), dtype=np.float32),
        "CameraParams": _make_camera_params(),
        "SemanticInstanceSegmentation": inst_seg,
        # Only instance 1 has a 2D box; 99 and 77 do not.
        "SemanticBoundingBox2DTight": _make_bbox2d([(1, 1, 1, 1, 5, 5)]),
        "SemanticBoundingBox3D": _make_bbox3d(
            [
                (1, 1, (-1, -1, -1), (1, 1, 1), (0, 0, 0)),
                (99, 2, (-1, -1, -1), (1, 1, 1), (1, 1, 1)),
                (77, 3, (-1, -1, -1), (1, 1, 1), (2, 2, 2)),
            ]
        ),
        "SemanticIdMap": _make_id_map({1: "car", 2: "person", 3: "tree"}),
    }
    backend = MockBackend()
    writer = BOPWriter(BOPWriterConfig())
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)
    # scene_gt_info is written; instance 99's bbox comes from its mask,
    # instance 77 (no pixels) yields the zero bbox.
    info = writer._scene_gt_info["0"]
    bboxes = [obj["bbox_visib"] for obj in info]
    assert [0, 0, 0, 0] in bboxes  # empty-mask fallback
    # The mask-derived bbox for instance 99: x in [3,8], y in [2,5].
    assert [3, 2, 6, 4] in bboxes


# ---------------------------------------------------------------------------
# COCOWriter — panoptic guards
# ---------------------------------------------------------------------------


def test_coco_panoptic_requires_instance_seg():
    from ovannotators.exceptions import MissingRenderVarError

    writer = COCOWriter(COCOWriterConfig(include_panoptic=True))
    # The constructor adds SemanticInstanceSegmentation to required_render_vars
    # for panoptic mode; drop it so the dedicated panoptic guard (not the
    # generic required-var loop) is what fires.
    writer.required_render_vars = frozenset({"LdrColor", "SemanticBoundingBox2DTight"})
    render_vars = {
        "LdrColor": np.zeros((8, 8, 4), dtype=np.uint8),
        "SemanticBoundingBox2DTight": _make_bbox2d([(1, 1, 0, 0, 4, 4)]),
        "SemanticIdMap": _make_id_map({1: "car"}),
    }
    with pytest.raises(MissingRenderVarError):
        writer.write_frame(0, render_vars, MockBackend())


def test_coco_panoptic_skips_uncategorized_instance():
    writer = COCOWriter(COCOWriterConfig(include_panoptic=True))
    inst_seg = np.zeros((8, 8), dtype=np.uint32)
    inst_seg[1:4, 1:4] = 1
    inst_seg[5:7, 5:7] = 555  # present in seg but not mapped to any category
    render_vars = {
        "LdrColor": np.zeros((8, 8, 4), dtype=np.uint8),
        "SemanticBoundingBox2DTight": _make_bbox2d([(1, 1, 1, 1, 4, 4)]),
        "SemanticInstanceSegmentation": inst_seg,
        "SemanticIdMap": _make_id_map({1: "car"}),
    }
    backend = MockBackend()
    writer.write_frame(0, render_vars, backend)
    writer.finalize(backend)  # should not raise

    # Two instances are present in the segmentation (1 and 555), but only the
    # categorized one (1 -> "car") yields a panoptic segment; the uncategorized
    # instance 555 must be skipped entirely.
    segments = writer._panoptic_annotations[0]["segments_info"]
    assert len(segments) == 1
    assert segments[0]["category_id"] == writer._categories["car"]
    # 555 was never mapped to a category, so "car" is the only known label.
    assert set(writer._categories) == {"car"}


def test_coco_panoptic_uncategorized_frame_is_empty():
    from ovannotators.writers import coco as coco_mod

    inst_seg = np.zeros((4, 5), dtype=np.uint32)
    inst_seg[1:3, 1:3] = 7  # present but has no category
    segments, panoptic = coco_mod.COCOWriter._build_panoptic_segments(inst_seg, {3: 11})
    assert segments == []
    assert panoptic.dtype == np.uint32 and panoptic.shape == (4, 5)
    assert not panoptic.any()


def test_coco_panoptic_segments_bbox_area_and_ids():
    """Segments follow ascending instance id; uncategorized ids stay 0."""
    from ovannotators.writers import coco as coco_mod

    inst_seg = np.zeros((6, 8), dtype=np.uint32)
    inst_seg[1:4, 2:5] = 9  # 3x3 block
    inst_seg[4, 0:8] = 3  # full-width row
    inst_seg[0, 7] = 5  # uncategorized
    segments, panoptic = coco_mod.COCOWriter._build_panoptic_segments(
        inst_seg, {3: 11, 9: 22}
    )
    assert segments == [
        {"id": 1, "category_id": 11, "area": 8, "bbox": [0, 4, 8, 1], "iscrowd": 0},
        {"id": 2, "category_id": 22, "area": 9, "bbox": [2, 1, 3, 3], "iscrowd": 0},
    ]
    expected = np.zeros_like(inst_seg)
    expected[inst_seg == 3] = 1
    expected[inst_seg == 9] = 2
    assert panoptic.dtype == np.uint32
    np.testing.assert_array_equal(panoptic, expected)


# ---------------------------------------------------------------------------
# CosmosTransferWriter
# ---------------------------------------------------------------------------


def test_canny_flat_image_returns_no_edges():
    from ovannotators.writers.canny import canny

    flat = np.full((16, 16), 0.5, dtype=np.float32)
    edges = canny(flat)
    assert edges.shape == (16, 16) and not edges.any()


def test_cosmos_colorize_seg_squeezes_extra_dims(monkeypatch):
    from ovannotators.writers import cosmos_transfer as ct

    writer = ct.CosmosTransferWriter(ct.CosmosTransferWriterConfig())
    # Provide a 3-D segmentation frame so the squeeze guard (seg2d[..., 0])
    # executes and the result is still a valid (h, w, 3) RGB map.
    writer._seg_frames = [np.zeros((8, 8, 2), dtype=np.int32)]
    out = writer._colorize_seg_frames()
    assert out[0].shape == (8, 8, 3)


def test_cosmos_encode_video_resizes_mismatched_frame(monkeypatch):
    from ovannotators._internal import codecs
    from ovannotators.writers import cosmos_transfer as ct

    class _FakeEncoder:
        def __init__(self):
            self.frames = []

        def write_frame(self, frame):
            self.frames.append(frame)

        def close(self):
            return b"FAKE"

    fake = _FakeEncoder()
    monkeypatch.setattr(codecs, "get_video_encoder", lambda **kwargs: fake)

    writer = ct.CosmosTransferWriter(ct.CosmosTransferWriterConfig())
    # Frame shape differs from (h, w) → triggers the resize branch.
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    data = writer._encode_video([frame], w=16, h=16)
    assert data == b"FAKE"
    # The recorded frame was resized to the target (h, w).
    assert fake.frames[0].shape[:2] == (16, 16)


# ---------------------------------------------------------------------------
# LeRobotWriter
# ---------------------------------------------------------------------------


def test_lerobot_to_rgb_int_and_video_empty():
    from ovannotators.writers import lerobot as lr

    # Non-float, non-uint8 dtype → astype(uint8) branch.
    arr = np.array([[1, 2], [3, 4]], dtype=np.int16)
    out = lr._to_rgb(arr)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    # Empty frame list → empty bytes.
    assert lr._encode_video([], fps=5) == b""


# ---------------------------------------------------------------------------
# WriterRegistry — duplicate name conflict
# ---------------------------------------------------------------------------


def test_registry_duplicate_name_raises():
    class _A:
        name = "_dup_cov_name"
        required_render_vars = frozenset()

        def write_frame(self, *a):  # pragma: no cover - protocol stub
            ...

        def finalize(self, *a):  # pragma: no cover - protocol stub
            ...

    class _B:
        name = "_dup_cov_name"
        required_render_vars = frozenset()

        def write_frame(self, *a):  # pragma: no cover - protocol stub
            ...

        def finalize(self, *a):  # pragma: no cover - protocol stub
            ...

    WriterRegistry.register(_A)
    try:
        with pytest.raises(ValueError):
            WriterRegistry.register(_B)
    finally:
        WriterRegistry._registry.pop("_dup_cov_name", None)


# ---------------------------------------------------------------------------
# Coverage: channel-axis squeeze, missing 3D match, and real video encode
# ---------------------------------------------------------------------------

# Minimal parseable H.264 Annex-B stream (SPS + PPS + IDR), mirroring the
# fixture used in tests/test_codecs.py, to drive the MP4 muxing path.
_START = b"\x00\x00\x00\x01"
_FAKE_ANNEXB = (
    _START
    + b"\x67\x42\x00\x1e\xda\x01\x40"  # SPS
    + _START
    + b"\x68\xce\x3c\x80"  # PPS
    + _START
    + b"\x65\x88\x84\x00\x21\x00"  # IDR slice
)


def test_coco_writer_squeezes_3d_inst_seg(sample_render_vars) -> None:
    """Instance segmentation with an extra channel axis is squeezed to 2-D."""
    render_vars = dict(sample_render_vars)
    inst_seg = sample_render_vars["SemanticInstanceSegmentation"]
    render_vars["SemanticInstanceSegmentation"] = inst_seg[..., np.newaxis]
    backend = MockBackend()
    # Should not raise despite the (H, W, 1) instance-seg input.
    COCOWriter().write_frame(0, render_vars, backend)
    assert any("images/000000" in k for k in backend.written)


def test_kitti_writer_2d_box_without_3d_match(sample_render_vars) -> None:
    """A 2D box whose instance id has no 3D box falls back to KITTI sentinels."""
    render_vars = dict(sample_render_vars)
    bbox3d = sample_render_vars["SemanticBoundingBox3D"]
    # No 3D boxes at all → every 2D box takes the "no 3D match" branch.
    render_vars["SemanticBoundingBox3D"] = np.zeros(0, dtype=bbox3d.dtype)
    backend = MockBackend()
    KITTIWriter().write_frame(0, render_vars, backend)
    label_txt = backend.written["label_2/000000.txt"].decode().strip()
    first = label_txt.splitlines()[0].split()
    # KITTI sentinels for unknown 3D geometry: dims -1, location -1000, ry -10.
    assert first[8:11] == ["-1.0000", "-1.0000", "-1.0000"]
    assert first[11:14] == ["-1000.0000", "-1000.0000", "-1000.0000"]
    assert first[14] == "-10.0000"


def test_lerobot_encode_video_non_empty():
    """A non-empty frame list runs the real GPU-encoder dispatch (mocked codec)."""
    from unittest.mock import MagicMock, patch

    from ovannotators.writers import lerobot as lr

    mock_pynv = MagicMock()
    mock_encoder_inst = MagicMock()
    mock_encoder_inst.Encode.return_value = _FAKE_ANNEXB
    mock_encoder_inst.EndEncode.return_value = b""
    mock_pynv.CreateEncoder.return_value = mock_encoder_inst

    frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
    with patch.dict("sys.modules", {"PyNvVideoCodec": mock_pynv}):
        out = lr._encode_video(frames, fps=10)

    # A real MP4 container is produced from the encoded frames.
    assert out[4:8] == b"ftyp"
