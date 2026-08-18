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

"""Shared fixtures for the ovannotators test suite."""

from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

# Make the package importable without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

import numpy as np
import pytest

from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
from ovannotators._internal.optional_deps import ovrtx_available
from ovannotators.writers._helpers import (
    OVRTX_BBOX2D_DTYPE,
    OVRTX_BBOX3D_DTYPE,
    OVRTX_CAMERA_PARAMS_DTYPE,
    decode_bbox_2d,
    decode_bbox_3d,
    decode_camera_params,
)

# Canonical labels shared by the bbox arrays and the SemanticIdMap buffer.
SAMPLE_SEMANTIC_LABELS: dict[int, str] = {1: "car", 2: "person"}

# Universal test scene and the RenderProduct that requests every ovrtx AOV.
TEST_SCENE_USDA = Path(__file__).parent / "data" / "test_scene.usda"
ALL_ANNOTATORS_PRODUCT = "/Render/RenderProduct_001"
WARMUP_FRAMES = 40

# Raw ovrtx annotator payloads decoded into the structured arrays writers expect.
_ANNOTATOR_DECODERS = {
    "SemanticBoundingBox2DTight": decode_bbox_2d,
    "SemanticBoundingBox2DLoose": decode_bbox_2d,
    "SemanticBoundingBox3D": decode_bbox_3d,
    "CameraParams": decode_camera_params,
}

# Guard on the real renderer API, not bare importability: a content-less stub
# ``ovrtx`` can import without exposing ``Renderer``.
_OVRTX_AVAILABLE = ovrtx_available()

requires_ovrtx = pytest.mark.skipif(
    not _OVRTX_AVAILABLE,
    reason="ovrtx not installed — skipping GPU render tests",
)


# ---------------------------------------------------------------------------
# Helpers — build numpy structured arrays in the real ovrtx payload layouts.
# ---------------------------------------------------------------------------


def _id128(value: int) -> tuple[int, int, int, int]:
    """Return a 128-bit id as four little-endian uint32 words (low word first)."""
    return (value, 0, 0, 0)


def _make_bbox2d_array() -> np.ndarray:
    """Return a 2-box ``SemanticBoundingBox2DTight`` array (OVRTX_BBOX2D_DTYPE)."""
    arr = np.zeros(2, dtype=OVRTX_BBOX2D_DTYPE)
    arr[0]["instance_id"] = _id128(1)
    arr[0]["semantic_id"] = _id128(1)
    arr[0]["x_min"], arr[0]["y_min"], arr[0]["x_max"], arr[0]["y_max"] = (
        10,
        20,
        100,
        200,
    )
    arr[1]["instance_id"] = _id128(2)
    arr[1]["semantic_id"] = _id128(2)
    arr[1]["x_min"], arr[1]["y_min"], arr[1]["x_max"], arr[1]["y_max"] = (
        50,
        60,
        150,
        160,
    )
    return arr


def _make_bbox3d_array() -> np.ndarray:
    """Return a 2-box ``SemanticBoundingBox3D`` array (OVRTX_BBOX3D_DTYPE).

    Each box is symmetric about its local origin, with a pure-translation
    local-to-world ``transform`` so the world-space centre equals the
    translation (5, 2, 10) and (3, 1.5, 8) respectively.
    """
    arr = np.zeros(2, dtype=OVRTX_BBOX3D_DTYPE)
    specs = [
        (1, (-1.0, -1.0, -2.0), (1.0, 1.0, 2.0), (5.0, 2.0, 10.0)),
        (2, (-0.5, -1.5, -0.5), (0.5, 1.5, 0.5), (3.0, 1.5, 8.0)),
    ]
    for i, (sid, lo, hi, translate) in enumerate(specs):
        arr[i]["instance_id"] = _id128(sid)
        arr[i]["semantic_id"] = _id128(sid)
        arr[i]["x_min"], arr[i]["y_min"], arr[i]["z_min"] = lo
        arr[i]["x_max"], arr[i]["y_max"], arr[i]["z_max"] = hi
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = translate
        arr[i]["transform"] = transform
    return arr


def _make_camera_params_array() -> np.ndarray:
    """Return a 1-element ``CameraParams`` array (OVRTX_CAMERA_PARAMS_DTYPE)."""
    arr = np.zeros(1, dtype=OVRTX_CAMERA_PARAMS_DTYPE)
    arr[0]["view_transform"] = np.eye(4, dtype=np.float32)
    arr[0]["projection"] = np.eye(4, dtype=np.float32)
    arr[0]["meters_per_scene_unit"] = 1.0
    arr[0]["resolution"] = (640, 480)
    arr[0]["clipping_plane"] = (0.01, 1000.0)
    arr[0]["projection_model"] = b"pinhole"
    arr[0]["focal_length"] = 24.0
    arr[0]["fov_x"] = math.radians(60.0)
    return arr


def _make_semantic_id_map(labels: dict[int, str]) -> np.ndarray:
    """Pack ``{semantic_id: label}`` into a ``SemanticIdMap`` uint8 buffer.

    Mirrors the ovrtx layout decoded by
    :func:`ovannotators.writers._helpers.decode_semantic_id_map`: a table of
    ``(semantic_id, label_length, label_offset)`` entries, a string table, and
    a trailing 4-byte little-endian entry count.
    """
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def disk_backend(tmp_path: Path) -> DiskBackend:
    return DiskBackend(DiskBackendConfig(root_dir=tmp_path, overwrite=True))


def _own_extracted(extracted: object, decoder: object) -> object:
    """Copy an extracted render var into renderer-independent host memory.

    ovrtx returns zero-copy Warp arrays (single-tensor vars) or a nested
    ``{"tensors": ..., "params": ...}`` dict of views (composite/sensor vars),
    each of which keeps the underlying render-var mapping alive. This copies the
    data out so the mapping can be released before the renderer is destroyed.

    Args:
        extracted: The value returned by :meth:`OvrtxOutput.extract`.
        decoder: Optional annotator decoder for this var; when set it is applied
            to the host array and already returns owned NumPy data.

    Returns:
        An owned NumPy array (or a nested dict of owned NumPy arrays) holding a
        copy of the extracted data.
    """
    if isinstance(extracted, dict):
        owned: dict = {}
        for key, sub in extracted.items():
            if isinstance(sub, dict):
                owned[key] = {
                    name: np.array(v.numpy() if hasattr(v, "numpy") else v, copy=True)
                    for name, v in sub.items()
                }
            else:
                owned[key] = sub
        return owned
    array = extracted.numpy() if hasattr(extracted, "numpy") else extracted
    if decoder is not None:
        return decoder(array)
    return np.array(array, copy=True)


@pytest.fixture(scope="session")
def rendered_scene_vars() -> dict:
    """Render ``test_scene.usda`` once and return all extracted render vars.

    Renders frame 0 of the ``AllAnnotators`` RenderProduct (which requests every
    ovrtx AOV) and extracts each var to a CPU numpy array via
    :class:`OvrtxOutput`.  Raw annotator byte payloads (bounding boxes, camera
    params) are decoded into the structured arrays the writers consume; the
    ``SemanticIdMap`` buffer is left raw for label resolution.

    Skipped when ovrtx is unavailable.
    """
    if not _OVRTX_AVAILABLE:
        pytest.skip("ovrtx not installed — skipping GPU render tests")

    import ovrtx

    from ovannotators.renderer_outputs.ovrtx import OvrtxOutput

    usda = TEST_SCENE_USDA.read_text()
    renderer = ovrtx.Renderer()
    try:
        renderer.open_usd_from_string(usda)

        # RT2 warmup: allow texture streaming and sample accumulation to settle.
        for _ in range(WARMUP_FRAMES):
            renderer.step(
                render_products={ALL_ANNOTATORS_PRODUCT},
                delta_time=1 / 60,
            )

        products = renderer.step(
            render_products={ALL_ANNOTATORS_PRODUCT},
            delta_time=1 / 60,
        )

        output = OvrtxOutput()
        render_vars: dict = {}
        for _product_name, product in products.items():
            for frame in product.frames:
                for source_name in output.available_vars(frame):
                    try:
                        extracted = output.extract(frame, source_name)
                    except Exception:
                        # Some vars (e.g. GenericModelOutput / point clouds) are
                        # lidar/radar sensor outputs that produce no buffer on a
                        # camera product. Tolerate absent payloads per the scene
                        # plan.
                        continue
                    decoder = _ANNOTATOR_DECODERS.get(source_name)
                    # Copy into owned host memory before the renderer is torn
                    # down. ovrtx hands back zero-copy Warp/NumPy views that keep
                    # the underlying render-var mapping alive; if those views
                    # outlive the renderer it logs "leaking mapped rendered
                    # output" errors (and can hang/segfault at interpreter exit).
                    # Owning the data lets every mapping release here so the
                    # renderer destroys cleanly below.
                    render_vars[source_name] = _own_extracted(extracted, decoder)
                    del extracted
                break
            break
        del products
    finally:
        # Destroy the renderer deterministically while the interpreter is fully
        # alive, instead of leaving it to interpreter-shutdown GC (which races
        # ovrtx/omni atexit teardown and can hang). All extracted data is owned
        # above, so there are no live mappings to leak.
        del renderer
        gc.collect()
    return render_vars


@pytest.fixture
def sample_render_vars() -> dict:
    """Complete render-vars dict (keyed by ovrtx sourceName) for all writers."""
    rng = np.random.default_rng(42)
    H, W = 480, 640

    # Instance / semantic segmentation: region 1 = car, region 2 = person
    inst_seg = np.zeros((H, W), dtype=np.uint32)
    inst_seg[20:200, 10:100] = 1
    inst_seg[60:160, 50:150] = 2
    sem_seg = inst_seg.copy()

    return {
        "LdrColor": rng.integers(0, 255, (H, W, 4), dtype=np.uint8),
        "DistanceToCameraSD": (rng.random((H, W)) * 50.0).astype(np.float32),
        "NormalSD": (rng.random((H, W, 4)).astype(np.float32) * 2.0 - 1.0),
        "SemanticBoundingBox2DTight": _make_bbox2d_array(),
        "SemanticBoundingBox3D": _make_bbox3d_array(),
        "SemanticInstanceSegmentation": inst_seg,
        "SemanticSegmentation": sem_seg,
        "CameraParams": _make_camera_params_array(),
        "SemanticIdMap": _make_semantic_id_map(SAMPLE_SEMANTIC_LABELS),
    }
