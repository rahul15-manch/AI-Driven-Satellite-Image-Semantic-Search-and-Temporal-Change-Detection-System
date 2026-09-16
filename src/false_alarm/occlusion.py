"""Localized occlusion and cloud shadow perturbation.

Simulates transient non-ground atmospheric obstructions, cloud shadows,
and localized solar occlusions by attenuating optical reflectance over a localized spatial zone.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.false_alarm.base import BasePerturbation


class LocalizedOcclusionShadowPerturbation(BasePerturbation):
    """Perturbation Family 4: Localized Occlusion / Shadow Effect."""

    # Documented, fixed area coverage fractions across severity tiers
    SEVERITY_AREA_FRACTIONS: Dict[str, float] = {
        "mild": 0.02,    # ~2% scene area (e.g. 145x145 on 1024x1024)
        "medium": 0.05,  # ~5% scene area (e.g. 230x230 on 1024x1024)
        "strong": 0.10,  # ~10% scene area (e.g. 324x324 on 1024x1024)
    }

    # Intensity transmission / shadow attenuation factor
    ATTENUATION_FACTOR: float = 0.40  # 60% intensity reduction in shadow

    def __init__(self, attenuation_factor: float = ATTENUATION_FACTOR):
        super().__init__(
            name="localized_occlusion_shadow",
            family="occlusion",
            supported_severities=list(self.SEVERITY_AREA_FRACTIONS.keys()),
        )
        self.attenuation_factor = attenuation_factor

    def _apply_transformation(
        self,
        image: np.ndarray,
        severity: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_dtype = image.dtype
        h, w = image.shape[:2]
        area_frac = self.SEVERITY_AREA_FRACTIONS[severity]

        # Calculate square patch dimension corresponding to target area fraction
        total_pixels = h * w
        target_patch_pixels = total_pixels * area_frac
        patch_side = int(np.round(np.sqrt(target_patch_pixels)))
        patch_side = max(1, min(patch_side, min(h, w)))

        # Deterministic spatial placement
        if seed is not None:
            rng = np.random.default_rng(seed)
            y0 = int(rng.integers(0, max(1, h - patch_side)))
            x0 = int(rng.integers(0, max(1, w - patch_side)))
        else:
            # Canonical deterministic offset (approximately upper-left-center)
            y0 = int(np.round((h - patch_side) * 0.35))
            x0 = int(np.round((w - patch_side) * 0.35))

        y1 = min(h, y0 + patch_side)
        x1 = min(w, x0 + patch_side)

        transformed = image.astype(np.float64)

        # Apply intensity attenuation to the localized patch
        if transformed.ndim == 3:
            transformed[y0:y1, x0:x1, :] *= self.attenuation_factor
        else:
            transformed[y0:y1, x0:x1] *= self.attenuation_factor

        clamped = self._validate_and_clamp(transformed, orig_dtype)

        affected_pixels = (y1 - y0) * (x1 - x0)
        actual_area_fraction = float(affected_pixels / total_pixels)

        meta = {
            "area_fraction_target": area_frac,
            "actual_area_fraction": actual_area_fraction,
            "patch_top_left": (y0, x0),
            "patch_bottom_right": (y1, x1),
            "patch_height": y1 - y0,
            "patch_width": x1 - x0,
            "attenuation_factor": self.attenuation_factor,
            "affected_pixel_count": affected_pixels,
        }

        return clamped, meta
