"""Validation-only threshold selection and calibration module.

MANDATORY LEAKAGE-CONTROL GUARANTEE:
Thresholds are calibrated EXCLUSIVELY on the validation split.
Test split data is NEVER exposed to this module. Once calibrated,
thresholds are strictly frozen before evaluation on test imagery.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import skimage.filters


@dataclass
class ThresholdSearchResult:
    """Encapsulates the results of a validation threshold optimization sweep."""
    best_threshold: float
    best_f1: float
    best_precision: float
    best_recall: float
    best_iou: float
    candidate_thresholds: List[float]
    f1_curve: List[float]
    precision_curve: List[float]
    recall_curve: List[float]
    num_candidates: int
    tie_breaker_rule: str = "smallest_threshold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_threshold": float(self.best_threshold),
            "best_f1": float(self.best_f1),
            "best_precision": float(self.best_precision),
            "best_recall": float(self.best_recall),
            "best_iou": float(self.best_iou),
            "num_candidates": self.num_candidates,
            "tie_breaker_rule": self.tie_breaker_rule,
        }


class ValidationThresholdOptimizer:
    """Finds optimal decision threshold maximizing F1 exclusively on validation data."""

    def __init__(
        self,
        min_threshold: float = 0.005,
        max_threshold: float = 0.995,
        num_candidates: int = 199,
        num_bins: int = 2000,
    ):
        """Initializes the validation threshold optimizer.

        Args:
            min_threshold: Lower bound for threshold search grid (default 0.005).
            max_threshold: Upper bound for threshold search grid (default 0.995).
            num_candidates: Number of uniformly spaced candidate thresholds (default 199, step ~0.005).
            num_bins: Fine histogram resolution for cumulative binning (default 2000).
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.num_candidates = num_candidates
        self.num_bins = num_bins
        self.candidates = np.linspace(min_threshold, max_threshold, num_candidates, dtype=np.float32)

        # Histograms over [0.0, 1.0]
        self.hist_changed = np.zeros(num_bins, dtype=np.int64)
        self.hist_unchanged = np.zeros(num_bins, dtype=np.int64)
        self.bin_edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float32)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self.total_pairs_observed = 0

    def update(self, diff_map: np.ndarray, gt_mask: np.ndarray) -> None:
        """Updates internal histograms with a validation difference map and ground-truth mask.

        Args:
            diff_map: Continuous difference map in [0, 1].
            gt_mask: Binary ground-truth mask in {0, 1}.
        """
        d = np.clip(diff_map.ravel(), 0.0, 1.0)
        g = gt_mask.ravel() > 0

        # Accumulate into fine-grained histograms
        h_ch, _ = np.histogram(d[g], bins=self.bin_edges)
        h_unch, _ = np.histogram(d[~g], bins=self.bin_edges)

        self.hist_changed += h_ch
        self.hist_unchanged += h_unch
        self.total_pairs_observed += 1

    def optimize(self) -> ThresholdSearchResult:
        """Computes Precision, Recall, and F1 across candidate thresholds and selects best tau*.

        Tie-breaking rule: Chooses the smallest threshold among equal-F1 candidates.

        Returns:
            ThresholdSearchResult containing optimal threshold and search curves.
        """
        if self.total_pairs_observed == 0:
            raise ValueError("Cannot optimize threshold without observing any validation pairs.")

        total_changed = int(np.sum(self.hist_changed))
        total_unchanged = int(np.sum(self.hist_unchanged))

        # Reverse cumulative sums: counts for pixels with value >= bin_edge
        cum_tp = np.cumsum(self.hist_changed[::-1])[::-1]
        cum_fp = np.cumsum(self.hist_unchanged[::-1])[::-1]

        f1_curve: List[float] = []
        p_curve: List[float] = []
        r_curve: List[float] = []
        iou_curve: List[float] = []

        best_f1 = -1.0
        best_tau = self.candidates[0]
        best_p = 0.0
        best_r = 0.0
        best_iou = 0.0

        for tau in self.candidates:
            # Find the bin index corresponding to tau
            idx = int(np.digitize(tau, self.bin_edges[:-1]) - 1)
            idx = max(0, min(idx, self.num_bins - 1))

            tp = float(cum_tp[idx])
            fp = float(cum_fp[idx])
            fn = float(total_changed - tp)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            p_curve.append(p)
            r_curve.append(r)
            f1_curve.append(f1)
            iou_curve.append(iou)

            # Strict greater-than enforces tie-breaking: smallest threshold chosen
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
                best_p = p
                best_r = r
                best_iou = iou

        return ThresholdSearchResult(
            best_threshold=best_tau,
            best_f1=best_f1,
            best_precision=best_p,
            best_recall=best_r,
            best_iou=best_iou,
            candidate_thresholds=[float(c) for c in self.candidates],
            f1_curve=f1_curve,
            precision_curve=p_curve,
            recall_curve=r_curve,
            num_candidates=self.num_candidates,
            tie_breaker_rule="smallest_threshold",
        )

    def compute_otsu_threshold(self) -> float:
        """Computes Otsu threshold using the aggregated validation difference histogram.

        Returns:
            Threshold float in [0.0, 1.0].
        """
        combined_hist = self.hist_changed + self.hist_unchanged
        if np.sum(combined_hist) == 0:
            raise ValueError("No validation data recorded for Otsu threshold computation.")

        # skimage threshold_otsu requires 256 bins typically, so rebin to 256
        h_256, edges_256 = np.histogram(self.bin_centers, bins=256, weights=combined_hist, range=(0.0, 1.0))
        centers_256 = (edges_256[:-1] + edges_256[1:]) / 2.0

        try:
            otsu_val = float(skimage.filters.threshold_otsu(hist=(h_256, centers_256)))
        except Exception:
            # Fallback to mean if histogram is degenerate
            otsu_val = float(np.average(self.bin_centers, weights=combined_hist))

        return otsu_val
