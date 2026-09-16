"""Evaluation metrics and confusion matrix utilities for change detection.

Implements pixel-level Precision, Recall, F1, and IoU for the CHANGED class (1),
with explicit zero-division guards and audit totals (TP, FP, FN, TN).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Union
import numpy as np


@dataclass
class ConfusionMatrix:
    """Accumulator for pixel-level classification outcomes."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def changed_gt(self) -> int:
        return self.tp + self.fn

    @property
    def changed_pred(self) -> int:
        return self.tp + self.fp

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Updates confusion counts from binary prediction and target arrays.

        Args:
            pred: Binary prediction mask {0, 1}.
            target: Binary ground-truth mask {0, 1}.
        """
        p = pred.astype(bool)
        t = target.astype(bool)

        self.tp += int(np.count_nonzero(p & t))
        self.fp += int(np.count_nonzero(p & ~t))
        self.fn += int(np.count_nonzero(~p & t))
        self.tn += int(np.count_nonzero(~p & ~t))

    def add(self, other: ConfusionMatrix) -> ConfusionMatrix:
        """Merges another ConfusionMatrix into a new instance."""
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "total_pixels": self.total,
            "gt_changed_pixels": self.changed_gt,
            "pred_changed_pixels": self.changed_pred,
        }


class ChangeDetectionEvaluator:
    """Computes research-standard evaluation metrics from confusion totals."""

    @staticmethod
    def evaluate(cm: ConfusionMatrix) -> Dict[str, Union[float, int]]:
        """Computes Precision, Recall, F1, IoU, and change-pixel ratios for the CHANGED class.

        Handles edge cases (zero division) explicitly by returning 0.0.

        Args:
            cm: ConfusionMatrix containing pixel counts.

        Returns:
            Dictionary containing metrics and raw audit counts.
        """
        tp = cm.tp
        fp = cm.fp
        fn = cm.fn
        tn = cm.tn
        total = cm.total

        # Precision = TP / (TP + FP)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # F1 = 2 * P * R / (P + R) == 2 * TP / (2 * TP + FP + FN)
        f1_denom = 2 * tp + fp + fn
        f1 = float(2 * tp / f1_denom) if f1_denom > 0 else 0.0

        # IoU = TP / (TP + FP + FN)
        iou_denom = tp + fp + fn
        iou = float(tp / iou_denom) if iou_denom > 0 else 0.0

        # Overall accuracy for reference (not primary metric due to severe class imbalance)
        accuracy = float((tp + tn) / total) if total > 0 else 0.0

        # Change ratios
        gt_ratio = float((tp + fn) / total) if total > 0 else 0.0
        pred_ratio = float((tp + fp) / total) if total > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_pixels": total,
            "gt_changed_pixels": tp + fn,
            "pred_changed_pixels": tp + fp,
            "gt_changed_ratio": gt_ratio,
            "pred_changed_ratio": pred_ratio,
        }
