"""Method B3: Change Vector Analysis (CVA) Detector.

Computes the Euclidean magnitude of change vectors in spectral/feature space between
pre- and post-change satellite imagery.

Formulation for RGB imagery:
    D_CVA(x, y) = sqrt((R2 - R1)^2 + (G2 - G1)^2 + (B2 - B1)^2) / sqrt(3)
Normalized to [0.0, 1.0] for pixel values in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from src.change_detection.base import BaseChangeDetector


class CVADetector(BaseChangeDetector):
    """Method B3: Change Vector Analysis Detector."""

    def __init__(self, normalize: bool = True):
        """Initializes CVADetector.

        Args:
            normalize: Whether to scale Euclidean distance by 1/sqrt(C) to ensure [0, 1] range.
        """
        super().__init__(name="B3_CVA", params={"normalize": normalize})
        self.normalize = normalize

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Computes pixel-wise CVA magnitude map in [0.0, 1.0].

        Args:
            img1: Pre-change optical image (H, W, 3) or (H, W).
            img2: Post-change optical image (H, W, 3) or (H, W).

        Returns:
            2D numpy array (H, W) float32 with continuous values in [0.0, 1.0].
        """
        arr1 = self._ensure_float_rgb(img1)
        arr2 = self._ensure_float_rgb(img2)

        if arr1.shape != arr2.shape:
            raise ValueError(f"Shape mismatch between img1 {arr1.shape} and img2 {arr2.shape}")

        diff = arr2 - arr1

        if diff.ndim == 3:
            num_channels = diff.shape[2]
            # Euclidean norm across spectral channels
            cva_mag = np.linalg.norm(diff, axis=2)
            if self.normalize and num_channels > 0:
                cva_mag = cva_mag / np.sqrt(num_channels)
        else:
            cva_mag = np.abs(diff)

        return np.clip(cva_mag, 0.0, 1.0).astype(np.float32)
