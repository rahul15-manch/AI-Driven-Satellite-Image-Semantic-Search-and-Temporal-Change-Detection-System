"""Global illumination shift perturbation.

Simulates temporal variations in solar elevation angle, seasonal irradiance,
and global atmospheric transmission shifts by applying controlled radiometric offsets.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.false_alarm.base import BasePerturbation


class IlluminationShiftPerturbation(BasePerturbation):
    """Perturbation Family 1: Global Illumination Shift."""

    # Documented, fixed parameter values across severity tiers
    SEVERITY_OFFSETS: Dict[str, float] = {
        "mild": 0.05,    # +5% radiometric dynamic range
        "medium": 0.15,  # +15% radiometric dynamic range
        "strong": 0.25,  # +25% radiometric dynamic range
    }

    def __init__(self):
        super().__init__(
            name="global_illumination_shift",
            family="illumination",
            supported_severities=list(self.SEVERITY_OFFSETS.keys()),
        )

    def _apply_transformation(
        self,
        image: np.ndarray,
        severity: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_dtype = image.dtype
        offset_fraction = self.SEVERITY_OFFSETS[severity]

        if np.issubdtype(orig_dtype, np.floating):
            # Working in float [0.0, 1.0]
            transformed = image + offset_fraction
            clamped = self._validate_and_clamp(transformed, orig_dtype)
            mean_shift = float(np.mean(clamped - image))
        else:
            # Working in integer [0, 255]
            int_offset = offset_fraction * 255.0
            transformed = image.astype(np.float32) + int_offset
            clamped = self._validate_and_clamp(transformed, orig_dtype)
            mean_shift = float(np.mean(clamped.astype(np.float32) - image.astype(np.float32))) / 255.0

        meta = {
            "offset_fraction": offset_fraction,
            "actual_mean_shift": mean_shift,
            "effective_clipped_pixels": int(np.count_nonzero(clamped >= (1.0 if np.issubdtype(orig_dtype, np.floating) else 255))),
        }

        return clamped, meta
