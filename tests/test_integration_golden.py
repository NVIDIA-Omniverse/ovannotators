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

"""Integration test: render ``test_scene.usda`` and compare golden outputs."""

from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests._golden_metrics import (
    ImageMetricThresholds,
    assert_image_metrics,
    load_rgb_u8,
)

# ---------------------------------------------------------------------------
# ovrtx availability guard
# ---------------------------------------------------------------------------

from ovannotators._internal.optional_deps import ovrtx_available

_OVRTX_AVAILABLE = ovrtx_available()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _OVRTX_AVAILABLE,
        reason="ovrtx not installed — skipping GPU integration tests",
    ),
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "golden" / "basic_writer_scene"
EXPECTED_DIR = GOLDEN_DIR / "expected"
SCENE_USDA = Path(__file__).parent / "data" / "test_scene.usda"
ALL_ANNOTATORS_PRODUCT = "/Render/RenderProduct_001"
WARMUP_FRAMES = 40
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixture: render one frame and write via BasicWriter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered_frame_dir():
    """Render frame 0 from ``test_scene.usda`` and return output dir path."""
    import ovrtx

    from ovannotators.backends.disk import DiskBackend, DiskBackendConfig
    from ovannotators.renderer_outputs.ovrtx import OvrtxOutput
    from ovannotators.writers.basic import BasicWriter, BasicWriterConfig

    usda = SCENE_USDA.read_text()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        backend = DiskBackend(DiskBackendConfig(root_dir=str(out_dir), overwrite=True))
        writer = BasicWriter(BasicWriterConfig(colorize=True))

        renderer = ovrtx.Renderer()
        renderer.open_usd_from_string(usda)

        # RT2 warmup per ovrtx guidance: textures + convergence.
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
        render_vars_out: dict[str, np.ndarray] = {}

        # Prefer broad AOV coverage from AllAnnotators.
        if ALL_ANNOTATORS_PRODUCT in products:
            product = products[ALL_ANNOTATORS_PRODUCT]
            frame = product.frames[0]
            for source_name in output.available_vars(frame):
                try:
                    extracted = output.extract(frame, source_name)
                except Exception:
                    # Sensor-style outputs (e.g. GenericModelOutput) may be
                    # absent on a camera render product; ignore those.
                    continue
                # Copy into owned host memory: ovrtx returns zero-copy views
                # that keep the render-var mapping alive, which would otherwise
                # leak (and hang/segfault) when the renderer is torn down.
                if hasattr(extracted, "numpy"):
                    render_vars_out[source_name] = np.array(
                        extracted.numpy(), copy=True
                    )
                else:
                    render_vars_out[source_name] = extracted
                del extracted

        writer.write_frame(
            frame_id=0,
            render_vars=render_vars_out,
            backend=backend,
        )
        # Store raw vars for stat checks (module-level via yield trick)
        pytest._golden_render_vars = render_vars_out  # type: ignore[attr-defined]

        # Destroy the renderer deterministically now that all data is owned, so
        # it does not race ovrtx/omni atexit teardown (which can hang) at the
        # end of the session.
        del products
        del renderer
        gc.collect()

        frame_dir = out_dir / "0000"
        if UPDATE_GOLDEN:
            EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
            _write_stats_json(render_vars_out, EXPECTED_DIR / "stats.json")
            for name in (
                "ldr_color.png",
                "distance_to_camera_colorized.png",
                "normal_colorized.png",
                "instance_segmentation_colorized.png",
                "semantic_segmentation_colorized.png",
                "bbox_2d_tight.json",
                "bbox_3d.json",
            ):
                src = frame_dir / name
                if src.exists():
                    (EXPECTED_DIR / name).write_bytes(src.read_bytes())

        # Yield inside the TemporaryDirectory context so files survive the test.
        yield frame_dir


def _assert_json_close(actual, expected, tol: float = 1e-3) -> None:
    """Recursively compare JSON structures with a numeric tolerance.

    Strings (labels, prim paths) and ``None`` must match exactly; numbers
    (bbox coordinates, transforms) may differ by up to ``tol`` to absorb tiny
    render-to-render variation.
    """
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert set(actual) == set(expected), (
            f"key mismatch: {sorted(actual)} vs {sorted(expected)}"
        )
        for key in actual:
            _assert_json_close(actual[key], expected[key], tol=tol)
        return
    if isinstance(actual, list):
        assert isinstance(expected, list)
        assert len(actual) == len(expected), (
            f"length mismatch: {len(actual)} vs {len(expected)}"
        )
        for a_item, e_item in zip(actual, expected):
            _assert_json_close(a_item, e_item, tol=tol)
        return
    if isinstance(actual, bool) or isinstance(expected, bool):
        assert actual == expected
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert abs(float(actual) - float(expected)) <= tol, (
            f"value mismatch: {actual} vs {expected} (tol={tol})"
        )
        return
    assert actual == expected, f"value mismatch: {actual!r} vs {expected!r}"


def _write_stats_json(render_vars: dict[str, np.ndarray], path: Path) -> None:
    depth = render_vars["DistanceToCameraSD"]
    normals = render_vars["NormalSD"]
    finite_vals = depth[np.isfinite(depth)]
    payload = {
        "depth": {
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "min_finite": float(finite_vals.min()),
            "max_finite": float(finite_vals.max()),
        },
        "normals": {
            "shape": list(normals.shape),
            "dtype": str(normals.dtype),
        },
    }
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Image comparison tests
# ---------------------------------------------------------------------------


class TestGoldenImages:
    def test_ldr_color(self, rendered_frame_dir: Path) -> None:
        rendered_path = rendered_frame_dir / "ldr_color.png"
        golden_path = EXPECTED_DIR / "ldr_color.png"
        assert_image_metrics(
            rendered_path,
            golden_path,
            ImageMetricThresholds(mae_max=10.0, rmse_max=18.0, psnr_min=22.0),
        )

    def test_depth_colorized(self, rendered_frame_dir: Path) -> None:
        rendered_path = rendered_frame_dir / "distance_to_camera_colorized.png"
        golden_path = EXPECTED_DIR / "distance_to_camera_colorized.png"
        assert_image_metrics(
            rendered_path,
            golden_path,
            ImageMetricThresholds(mae_max=10.0, rmse_max=18.0, psnr_min=22.0),
        )

    def test_normals_colorized(self, rendered_frame_dir: Path) -> None:
        rendered_path = rendered_frame_dir / "normal_colorized.png"
        golden_path = EXPECTED_DIR / "normal_colorized.png"
        assert_image_metrics(
            rendered_path,
            golden_path,
            ImageMetricThresholds(mae_max=5.0, rmse_max=10.0, psnr_min=28.0),
        )

    def test_instance_segmentation_colorized(self, rendered_frame_dir: Path) -> None:
        rendered = load_rgb_u8(
            rendered_frame_dir / "instance_segmentation_colorized.png"
        )
        golden_path = EXPECTED_DIR / "instance_segmentation_colorized.png"
        assert golden_path.exists(), (
            f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
        )
        golden = load_rgb_u8(golden_path)
        assert rendered.shape == golden.shape, (
            "instance_segmentation_colorized shape mismatch: "
            f"{rendered.shape} vs {golden.shape}"
        )
        assert np.array_equal(rendered, golden), (
            "instance_segmentation_colorized: expected exact match "
            "(deterministic colors)"
        )

    def test_semantic_segmentation_colorized(self, rendered_frame_dir: Path) -> None:
        rendered = load_rgb_u8(
            rendered_frame_dir / "semantic_segmentation_colorized.png"
        )
        golden_path = EXPECTED_DIR / "semantic_segmentation_colorized.png"
        assert golden_path.exists(), (
            f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
        )
        golden = load_rgb_u8(golden_path)
        assert rendered.shape == golden.shape, (
            "semantic_segmentation_colorized shape mismatch: "
            f"{rendered.shape} vs {golden.shape}"
        )
        assert np.array_equal(rendered, golden), (
            "semantic_segmentation_colorized: expected exact match "
            "(deterministic colors)"
        )


# ---------------------------------------------------------------------------
# Bounding box tests
# ---------------------------------------------------------------------------


class TestGoldenBoundingBoxes:
    """Golden coverage for the BasicWriter bounding-box JSON outputs."""

    def test_bbox_2d_tight(self, rendered_frame_dir: Path) -> None:
        rendered_path = rendered_frame_dir / "bbox_2d_tight.json"
        golden_path = EXPECTED_DIR / "bbox_2d_tight.json"
        assert rendered_path.exists(), "BasicWriter did not emit bbox_2d_tight.json"
        assert golden_path.exists(), (
            f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
        )
        _assert_json_close(
            json.loads(rendered_path.read_text()),
            json.loads(golden_path.read_text()),
        )

    def test_bbox_3d(self, rendered_frame_dir: Path) -> None:
        rendered_path = rendered_frame_dir / "bbox_3d.json"
        golden_path = EXPECTED_DIR / "bbox_3d.json"
        assert rendered_path.exists(), "BasicWriter did not emit bbox_3d.json"
        assert golden_path.exists(), (
            f"Golden missing: {golden_path}. Run UPDATE_GOLDEN=1 pytest to generate."
        )
        _assert_json_close(
            json.loads(rendered_path.read_text()),
            json.loads(golden_path.read_text()),
        )


# ---------------------------------------------------------------------------
# Raw data stats tests
# ---------------------------------------------------------------------------


class TestGoldenStats:
    @pytest.fixture(autouse=True)
    def _load_stats(self) -> None:
        with open(EXPECTED_DIR / "stats.json") as f:
            self.stats = json.load(f)

    def _render_vars(self) -> dict:
        return pytest._golden_render_vars  # type: ignore[attr-defined]

    def test_depth_shape(self, rendered_frame_dir: Path) -> None:
        depth = self._render_vars()["DistanceToCameraSD"]
        expected_shape = tuple(self.stats["depth"]["shape"])
        assert depth.shape == expected_shape, (
            f"depth shape {depth.shape} != expected {expected_shape}"
        )

    def test_depth_dtype(self, rendered_frame_dir: Path) -> None:
        depth = self._render_vars()["DistanceToCameraSD"]
        expected_dtype = self.stats["depth"]["dtype"]
        assert str(depth.dtype) == expected_dtype, (
            f"depth dtype {depth.dtype} != expected {expected_dtype}"
        )

    def test_depth_finite_range(self, rendered_frame_dir: Path) -> None:
        depth = self._render_vars()["DistanceToCameraSD"]
        finite_vals = depth[np.isfinite(depth)]
        assert finite_vals.size > 0, "depth has no finite values"
        tol = 1.0
        assert (
            abs(float(finite_vals.min()) - self.stats["depth"]["min_finite"]) < tol
        ), (
            f"depth min_finite {finite_vals.min():.4f} differs from golden "
            f"{self.stats['depth']['min_finite']:.4f} by >{tol}"
        )
        assert (
            abs(float(finite_vals.max()) - self.stats["depth"]["max_finite"]) < tol
        ), (
            f"depth max_finite {finite_vals.max():.4f} differs from golden "
            f"{self.stats['depth']['max_finite']:.4f} by >{tol}"
        )

    def test_normals_shape(self, rendered_frame_dir: Path) -> None:
        normals = self._render_vars()["NormalSD"]
        expected_shape = tuple(self.stats["normals"]["shape"])
        assert normals.shape == expected_shape, (
            f"normals shape {normals.shape} != expected {expected_shape}"
        )

    def test_normals_dtype(self, rendered_frame_dir: Path) -> None:
        normals = self._render_vars()["NormalSD"]
        expected_dtype = self.stats["normals"]["dtype"]
        assert str(normals.dtype) == expected_dtype, (
            f"normals dtype {normals.dtype} != expected {expected_dtype}"
        )
