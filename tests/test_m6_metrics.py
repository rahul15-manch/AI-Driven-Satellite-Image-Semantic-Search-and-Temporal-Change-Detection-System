"""Unit tests for RobustnessMetricsCalculator."""

import pytest
from src.false_alarm.robustness_metrics import RobustnessMetricsCalculator


def test_robustness_metrics_standard():
    ctrl = {
        "f1": 0.50,
        "iou": 0.3333,
        "precision": 0.60,
        "recall": 0.4285,
        "fp": 1000,
        "tp": 1500,
        "tn": 8000,
        "fn": 2000,
    }

    pert = {
        "f1": 0.25,
        "iou": 0.1428,
        "precision": 0.20,
        "recall": 0.3333,
        "fp": 4000,
        "tp": 1000,
        "tn": 5000,
        "fn": 2500,
    }

    deg = RobustnessMetricsCalculator.compute_degradation(ctrl, pert)

    # Relative F1 degradation: ((0.50 - 0.25) / 0.50) * 100 = 50.0%
    assert deg["f1_delta"] == pytest.approx(-0.25)
    assert deg["relative_f1_degradation_pct"] == pytest.approx(50.0)

    # Delta FP: 4000 - 1000 = 3000
    assert deg["additional_false_positives"] == 3000
    # Relative FP increase: (3000 / 1000) * 100 = 300.0%
    assert deg["relative_fp_increase_pct"] == pytest.approx(300.0)

    # FP generation rate: 3000 / (8000 + 1000) = 3000 / 9000 = 33.33%
    assert deg["fp_generation_rate_pct"] == pytest.approx(3000.0 / 9000.0 * 100.0)


def test_robustness_metrics_zero_denominators():
    ctrl = {"f1": 0.0, "iou": 0.0, "fp": 0, "tn": 0}
    pert = {"f1": 0.0, "iou": 0.0, "fp": 0, "tn": 0}

    deg = RobustnessMetricsCalculator.compute_degradation(ctrl, pert)

    assert deg["relative_f1_degradation_pct"] == 0.0
    assert deg["relative_iou_degradation_pct"] == 0.0
    assert deg["relative_fp_increase_pct"] == 0.0
    assert deg["fp_generation_rate_pct"] == 0.0
