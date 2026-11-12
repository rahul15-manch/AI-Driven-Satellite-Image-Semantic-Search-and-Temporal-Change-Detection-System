"""Gaussian blur spatial degradation perturbation.

Simulates atmospheric turbulence, optical defocus, sensor point-spread function (PSF)
variations, and high-frequency textural softening.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from src.false_alarm.base import BasePerturbation


class GaussianBlurPerturbation(BasePerturbation):
    """Perturbation Family 2: Gaussian Blur."""

    # Documented, fixed parameter values across severity tiers
    SEVERITY_SIGMAS: Dict[str, float] = {
        "mild": 1.0,    # Subtle optical defocus
        "medium": 2.0,  # Moderate atmospheric turbulence
        "strong": 4.0,  # Severe haze / high-frequency texture attenuation
    }

    def __init__(self):
        super().__init__(
            name="gaussian_blur",
            family="blur",
            supported_severities=list(self.SEVERITY_SIGMAS.keys()),
        )

    def _apply_transformation(
        self,
        image: np.ndarray,
        severity: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_dtype = image.dtype
        sigma = self.SEVERITY_SIGMAS[severity]

        # Convert to float for accurate filtering
        arr = image.astype(np.float64)

        if arr.ndim == 3:
            # Apply 2D spatial Gaussian blur independently across color channels
            blurred = np.zeros_like(arr)
            for c in range(arr.shape[2]):
                blurred[:, :, c] = gaussian_filter(arr[:, :, c], sigma=sigma, mode="reflect")
        else:
            blurred = gaussian_filter(arr, sigma=sigma, mode="reflect")

        clamped = self._validate_and_clamp(blurred, orig_dtype)

        # Compute empirical difference statistics
        pixel_abs_diff = np.abs(clamped.astype(np.float64) - image.astype(np.float64))
        scale = 1.0 if np.issubdtype(orig_dtype, np.floating) else 255.0

        meta = {
            "sigma": sigma,
            "mean_pixel_divergence": float(np.mean(pixel_abs_diff) / scale),
            "max_pixel_divergence": float(np.max(pixel_abs_diff) / scale),
        }

        return clamped, meta
