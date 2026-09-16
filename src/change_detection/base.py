"""Abstract base class for classical bi-temporal change detection algorithms.

Provides standardized interface for computing continuous difference maps in [0, 1]
and generating binary change masks given a calibrated threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


class BaseChangeDetector(ABC):
    """Abstract base class for all change detection methods."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Computes continuous change magnitude/dissimilarity map in [0.0, 1.0].

        Args:
            img1: Pre-change optical image array (H, W, 3) or (H, W), dtype uint8 [0, 255] or float [0, 1].
            img2: Post-change optical image array (H, W, 3) or (H, W), dtype uint8 [0, 255] or float [0, 1].

        Returns:
            2D numpy array of shape (H, W), float32 with values normalized to [0.0, 1.0].
        """
        pass

    def predict(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates binary change mask and returns difference map using a calibrated threshold.

        Args:
            img1: Pre-change optical image array.
            img2: Post-change optical image array.
            threshold: Calibrated decision threshold in [0.0, 1.0].

        Returns:
            Tuple of:
                - binary_mask: (H, W) uint8 array with values in {0, 1} (1 = changed).
                - diff_map: (H, W) float32 array with continuous values in [0.0, 1.0].
        """
        diff_map = self.compute_difference_map(img1, img2)
        binary_mask = (diff_map >= threshold).astype(np.uint8)
        return binary_mask, diff_map

    @staticmethod
    def _ensure_float_rgb(img: np.ndarray) -> np.ndarray:
        """Converts input image to float32 in [0.0, 1.0] with shape (H, W, 3) or (H, W)."""
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr
