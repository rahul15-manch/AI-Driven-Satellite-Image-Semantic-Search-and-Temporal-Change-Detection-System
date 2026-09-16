"""Geometric misregistration perturbation.

Simulates sensor orthorectification discrepancies, satellite attitude estimation errors,
and spatial alignment mismatches between temporal acquisitions.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.ndimage import shift

from src.false_alarm.base import BasePerturbation


class GeometricMisregistrationPerturbation(BasePerturbation):
    """Perturbation Family 3: Geometric Misregistration."""

    # Documented, fixed spatial displacement (dy, dx) across severity tiers
    SEVERITY_SHIFTS: Dict[str, Tuple[int, int]] = {
        "mild": (1, 1),    # 1 pixel translation (subtle co-registration error)
        "medium": (3, 3),  # 3 pixels translation (moderate orthorectification error)
        "strong": (5, 5),  # 5 pixels translation (severe spatial displacement)
    }

    def __init__(self):
        super().__init__(
            name="geometric_misregistration",
            family="misregistration",
            supported_severities=list(self.SEVERITY_SHIFTS.keys()),
        )

    def _apply_transformation(
        self,
        image: np.ndarray,
        severity: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_dtype = image.dtype
        dy, dx = self.SEVERITY_SHIFTS[severity]

        arr = image.astype(np.float64)

        if arr.ndim == 3:
            shifted = shift(arr, shift=(dy, dx, 0), order=0, mode="reflect")
        else:
            shifted = shift(arr, shift=(dy, dx), order=0, mode="reflect")

        clamped = self._validate_and_clamp(shifted, orig_dtype)

        pixel_abs_diff = np.abs(clamped.astype(np.float64) - image.astype(np.float64))
        scale = 1.0 if np.issubdtype(orig_dtype, np.floating) else 255.0

        meta = {
            "shift_dy": dy,
            "shift_dx": dx,
            "euclidean_pixel_shift": float(np.sqrt(dx**2 + dy**2)),
            "mean_pixel_divergence": float(np.mean(pixel_abs_diff) / scale),
        }

        return clamped, meta
