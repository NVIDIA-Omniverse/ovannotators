# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metric-based image golden comparison helpers.

This module centralizes best-practice image regression checks:
- exact comparison for deterministic buffers when needed
- tolerance-based comparison for rendered or compressed images

The default metric policy uses multiple complementary signals:
- MAE (absolute drift)
- RMSE (energy of error)
- PSNR (signal fidelity; higher is better)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageMetricThresholds:
    """Thresholds for tolerant image golden checks.

    Attributes:
        mae_max: Maximum allowed mean absolute error in uint8 pixel units.
        rmse_max: Maximum allowed root-mean-square error in uint8 units.
        psnr_min: Minimum allowed PSNR in dB.
    """

    mae_max: float
    rmse_max: float
    psnr_min: float


def load_rgb_u8(path: Path) -> np.ndarray:
    """Load an image as a uint8 RGB numpy array.

    Args:
        path: Path to an image file.

    Returns:
        Image array of shape ``(H, W, 3)`` and dtype ``uint8``.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def write_golden_image(src: Path, dst: Path) -> None:
    """Copy a produced image to the golden location.

    Args:
        src: Produced image path.
        dst: Golden image target path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def compute_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Compute core pixel-quality metrics.

    Args:
        actual: Produced RGB image.
        expected: Golden RGB image.

    Returns:
        Dict containing ``mae``, ``rmse``, and ``psnr``.
    """
    diff = actual.astype(np.float32) - expected.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    psnr = float(99.0 if rmse < 1e-12 else 20.0 * np.log10(255.0 / rmse))
    return {"mae": mae, "rmse": rmse, "psnr": psnr}


def assert_image_metrics(
    actual_path: Path,
    expected_path: Path,
    thresholds: ImageMetricThresholds,
) -> None:
    """Assert an image matches a golden image under metric thresholds.

    Args:
        actual_path: Produced image path.
        expected_path: Golden image path.
        thresholds: Metric thresholds for pass/fail.
    """
    assert expected_path.exists(), (
        f"Golden missing: {expected_path}. Run UPDATE_GOLDEN=1 pytest to generate."
    )
    actual = load_rgb_u8(actual_path)
    expected = load_rgb_u8(expected_path)
    assert actual.shape == expected.shape, (
        f"image shape mismatch: {actual.shape} vs {expected.shape}"
    )

    metrics = compute_metrics(actual, expected)
    assert metrics["mae"] <= thresholds.mae_max, (
        f"MAE {metrics['mae']:.4f} > {thresholds.mae_max:.4f}"
    )
    assert metrics["rmse"] <= thresholds.rmse_max, (
        f"RMSE {metrics['rmse']:.4f} > {thresholds.rmse_max:.4f}"
    )
    assert metrics["psnr"] >= thresholds.psnr_min, (
        f"PSNR {metrics['psnr']:.4f} < {thresholds.psnr_min:.4f}"
    )
