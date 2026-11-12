"""Abstract base class and registry for controlled image perturbations.

All perturbations operate on bi-temporal optical pairs (T1, T2) and return
modified pairs while preserving data types, spatial dimensions, and value bounds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BasePerturbation(ABC):
    """Abstract base class for all non-ground visual perturbations."""

    def __init__(self, name: str, family: str, supported_severities: Optional[List[str]] = None):
        self.name = name
        self.family = family
        self.supported_severities = supported_severities or ["mild", "medium", "strong"]

    @abstractmethod
    def _apply_transformation(
        self,
        image: np.ndarray,
        severity: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies the specific visual transformation to a single image array.

        Args:
            image: Single optical image array (H, W, 3) or (H, W).
            severity: 'mild', 'medium', or 'strong'.
            seed: Optional random seed for reproducible stochastic components.

        Returns:
            Tuple of (transformed_image, transformation_metadata).
        """
        pass

    def apply(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        severity: str = "medium",
        target_image: str = "t2",
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Applies controlled perturbation to one temporal image in a bi-temporal pair.

        Args:
            img1: Pre-change optical image array (T1).
            img2: Post-change optical image array (T2).
            severity: 'mild', 'medium', or 'strong'.
            target_image: Which temporal acquisition to perturb ('t1' or 't2', default 't2').
            seed: Optional integer seed for reproducibility.

        Returns:
            Tuple of:
                - out_img1: Array for T1 (unmodified if target_image is 't2').
                - out_img2: Array for T2 (unmodified if target_image is 't1').
                - meta: Dictionary describing perturbation parameters and affected statistics.
        """
        if severity not in self.supported_severities:
            raise ValueError(
                f"Unsupported severity '{severity}' for {self.name}. "
                f"Supported: {self.supported_severities}"
            )
        if target_image not in {"t1", "t2"}:
            raise ValueError(f"target_image must be 't1' or 't2', got '{target_image}'")

        # Guarantee functional purity: do not mutate inputs
        t1_copy = np.copy(img1)
        t2_copy = np.copy(img2)

        if target_image == "t2":
            perturbed, meta = self._apply_transformation(t2_copy, severity=severity, seed=seed)
            out_img1 = t1_copy
            out_img2 = perturbed
        else:
            perturbed, meta = self._apply_transformation(t1_copy, severity=severity, seed=seed)
            out_img1 = perturbed
            out_img2 = t2_copy

        meta.update({
            "perturbation_name": self.name,
            "perturbation_family": self.family,
            "severity": severity,
            "target_image": target_image,
            "seed": seed,
        })

        return out_img1, out_img2, meta

    @staticmethod
    def _validate_and_clamp(arr: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
        """Clamps transformed values to valid ranges based on dtype."""
        if np.issubdtype(original_dtype, np.floating):
            return np.clip(arr, 0.0, 1.0).astype(original_dtype)
        elif np.issubdtype(original_dtype, np.integer):
            return np.clip(np.round(arr), 0, 255).astype(original_dtype)
        return arr
