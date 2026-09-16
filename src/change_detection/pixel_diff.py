"""Method B1: Absolute Pixel Differencing Change Detector.

Computes absolute pixel differences between pre- and post-change satellite imagery,
aggregating across color channels deterministically.

Primary formulation:
    D_mean(x, y) = 1/3 * sum_{c in {R, G, B}} |I2(x, y, c) - I1(x, y, c)|
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from src.change_detection.base import BaseChangeDetector


class PixelDiffDetector(BaseChangeDetector):
    """Method B1: Absolute Pixel Differencing Detector."""

    def __init__(self, aggregation_mode: str = "mean"):
        """Initializes PixelDiffDetector.

        Args:
            aggregation_mode: Channel aggregation method ('mean' [primary, default] or 'max').
        """
        if aggregation_mode not in {"mean", "max"}:
            raise ValueError(f"aggregation_mode must be 'mean' or 'max', got {aggregation_mode}")
        super().__init__(name="B1_Pixel_Diff", params={"aggregation_mode": aggregation_mode})
        self.aggregation_mode = aggregation_mode

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Computes pixel-wise absolute difference map normalized to [0.0, 1.0].

        Args:
            img1: Pre-change image (H, W, 3) or (H, W).
            img2: Post-change image (H, W, 3) or (H, W).

        Returns:
            2D array (H, W) float32 in [0.0, 1.0].
        """
        arr1 = self._ensure_float_rgb(img1)
        arr2 = self._ensure_float_rgb(img2)

        if arr1.shape != arr2.shape:
            raise ValueError(f"Shape mismatch between img1 {arr1.shape} and img2 {arr2.shape}")

        abs_diff = np.abs(arr2 - arr1)

        if abs_diff.ndim == 3 and abs_diff.shape[2] > 1:
            if self.aggregation_mode == "mean":
                diff_map = np.mean(abs_diff, axis=2)
            else:
                diff_map = np.max(abs_diff, axis=2)
        elif abs_diff.ndim == 3 and abs_diff.shape[2] == 1:
            diff_map = abs_diff.squeeze(2)
        else:
            diff_map = abs_diff

        return np.clip(diff_map, 0.0, 1.0).astype(np.float32)
