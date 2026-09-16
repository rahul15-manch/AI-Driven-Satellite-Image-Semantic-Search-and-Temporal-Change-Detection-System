"""Robustness and false-alarm degradation metrics for change detection under perturbations.

Computes relative degradation and additional false positive metrics with strict
zero-division handling.
"""

from __future__ import annotations

from typing import Dict, Any, Union
import numpy as np


class RobustnessMetricsCalculator:
    """Calculates comparative degradation metrics between control and perturbed evaluations."""

    @staticmethod
    def compute_degradation(
        control_metrics: Dict[str, Union[float, int]],
        perturbed_metrics: Dict[str, Union[float, int]],
    ) -> Dict[str, Union[float, int]]:
        """Computes absolute and relative degradation metrics.

        Args:
            control_metrics: Metrics dictionary from the unperturbed control evaluation.
            perturbed_metrics: Metrics dictionary from the perturbed condition.

        Returns:
            Dictionary containing absolute differences, relative degradation percentages,
            and false-positive amplification metrics.
        """
        f1_ctrl = float(control_metrics.get("f1", 0.0))
        f1_pert = float(perturbed_metrics.get("f1", 0.0))

        iou_ctrl = float(control_metrics.get("iou", 0.0))
        iou_pert = float(perturbed_metrics.get("iou", 0.0))

        p_ctrl = float(control_metrics.get("precision", 0.0))
        p_pert = float(perturbed_metrics.get("precision", 0.0))

        r_ctrl = float(control_metrics.get("recall", 0.0))
        r_pert = float(perturbed_metrics.get("recall", 0.0))

        fp_ctrl = int(control_metrics.get("fp", 0))
        fp_pert = int(perturbed_metrics.get("fp", 0))

        tp_ctrl = int(control_metrics.get("tp", 0))
        tp_pert = int(perturbed_metrics.get("tp", 0))

        # Absolute changes: pert - ctrl
        delta_f1 = f1_pert - f1_ctrl
        delta_iou = iou_pert - iou_ctrl
        delta_precision = p_pert - p_ctrl
        delta_recall = r_pert - r_ctrl
        delta_fp = fp_pert - fp_ctrl
        delta_tp = tp_pert - tp_ctrl

        # Relative F1 degradation (%): ((F1_ctrl - F1_pert) / F1_ctrl) * 100
        # Positive values represent degradation; negative values indicate improvement
        if f1_ctrl > 0.0:
            rel_f1_degradation = float(((f1_ctrl - f1_pert) / f1_ctrl) * 100.0)
        else:
            rel_f1_degradation = 0.0

        # Relative IoU degradation (%): ((IoU_ctrl - IoU_pert) / IoU_ctrl) * 100
        if iou_ctrl > 0.0:
            rel_iou_degradation = float(((iou_ctrl - iou_pert) / iou_ctrl) * 100.0)
        else:
            rel_iou_degradation = 0.0

        # Relative FP increase (%): ((FP_pert - FP_ctrl) / FP_ctrl) * 100
        if fp_ctrl > 0:
            rel_fp_increase = float((delta_fp / fp_ctrl) * 100.0)
        else:
            rel_fp_increase = 0.0

        # Ground-truth unchanged pixel count (TN_ctrl + FP_ctrl)
        total_unchanged_pixels = int(control_metrics.get("tn", 0)) + fp_ctrl
        if total_unchanged_pixels > 0:
            # Fraction of true non-change pixels incorrectly converted to false positives by perturbation
            fp_generation_rate = float((delta_fp / total_unchanged_pixels) * 100.0)
        else:
            fp_generation_rate = 0.0

        return {
            "f1_control": f1_ctrl,
            "f1_perturbed": f1_pert,
            "f1_delta": delta_f1,
            "relative_f1_degradation_pct": rel_f1_degradation,
            "iou_control": iou_ctrl,
            "iou_perturbed": iou_pert,
            "iou_delta": delta_iou,
            "relative_iou_degradation_pct": rel_iou_degradation,
            "precision_control": p_ctrl,
            "precision_perturbed": p_pert,
            "precision_delta": delta_precision,
            "recall_control": r_ctrl,
            "recall_perturbed": r_pert,
            "recall_delta": delta_recall,
            "fp_control": fp_ctrl,
            "fp_perturbed": fp_pert,
            "additional_false_positives": delta_fp,
            "relative_fp_increase_pct": rel_fp_increase,
            "fp_generation_rate_pct": fp_generation_rate,
            "tp_control": tp_ctrl,
            "tp_perturbed": tp_pert,
            "delta_tp": delta_tp,
        }
