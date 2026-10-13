"""Unit and algorithmic validation tests for Milestone M11 benchmarking harness.

Validates:
- Confusion matrix arithmetic and zero-division behavior.
- Total pixel accounting integrity: TP + FP + FN + TN == 134,217,728 on LEVIR-CD.
- Exact ground-truth changed pixel count: 6,837,404.
- Scene-level paired bootstrap reproducibility and 95% CI monotonicity.
- Paired Wilcoxon hypothesis testing consistency.
- QAT-CD ablation arithmetic and false-alarm reduction calculation.
- Schema conformity for all exported M11 machine-readable artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from src.change_detection.benchmark_harness import BenchmarkHarness


def test_confusion_matrix_arithmetic_and_zero_guards():
    """Verifies confusion matrix arithmetic and division-by-zero guards."""
    cm = ConfusionMatrix(tp=100, fp=20, fn=30, tn=850)
    assert cm.total == 1000
    assert cm.changed_gt == 130
    assert cm.changed_pred == 120

    m = ChangeDetectionEvaluator.evaluate(cm)
    assert pytest.approx(m["precision"], rel=1e-4) == 100 / 120
    assert pytest.approx(m["recall"], rel=1e-4) == 100 / 130
    assert pytest.approx(m["f1"], rel=1e-4) == (2 * (100 / 120) * (100 / 130)) / ((100 / 120) + (100 / 130))
    assert pytest.approx(m["iou"], rel=1e-4) == 100 / (100 + 20 + 30)
    assert pytest.approx(m["accuracy"], rel=1e-4) == 950 / 1000

    # Zero denominator edge cases
    cm_zero = ConfusionMatrix(tp=0, fp=0, fn=0, tn=1000)
    m_zero = ChangeDetectionEvaluator.evaluate(cm_zero)
    assert m_zero["precision"] == 0.0
    assert m_zero["recall"] == 0.0
    assert m_zero["f1"] == 0.0
    assert m_zero["iou"] == 0.0
    assert m_zero["accuracy"] == 1.0


def test_confusion_matrix_update():
    """Verifies ConfusionMatrix.update with binary numpy arrays."""
    cm = ConfusionMatrix()
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    cm.update(pred, target)
    # (1,1): pred=1, target=1 -> TP=1
    # (0,1): pred=0, target=1 -> FN=1
    # (1,0): pred=1, target=0 -> FP=1
    # (0,0): pred=0, target=0 -> TN=1
    assert cm.tp == 1
    assert cm.fp == 1
    assert cm.fn == 1
    assert cm.tn == 1
    assert cm.total == 4


def test_recompute_confusion_metrics():
    """Verifies BenchmarkHarness.recompute_confusion_metrics static helper."""
    metrics = BenchmarkHarness.recompute_confusion_metrics(tp=3707080, fp=1965041, fn=3130324, tn=125415283)
    assert metrics["total_pixels"] == 134217728
    assert pytest.approx(metrics["precision"], rel=1e-3) == 0.6536
    assert pytest.approx(metrics["recall"], rel=1e-3) == 0.5422
    assert pytest.approx(metrics["f1"], rel=1e-3) == 0.5927
    assert pytest.approx(metrics["iou"], rel=1e-3) == 0.4211


def test_full_scene_pixel_accounting_integrity():
    """Verifies all LEVIR-CD test scenes sum to exactly 134,217,728 pixels with 6,837,404 GT changes."""
    summary_p = Path("experiments/results/m11/benchmark_summary.json")
    if not summary_p.exists():
        pytest.skip("M11 benchmark summary not yet generated.")

    with open(summary_p, "r", encoding="utf-8") as f:
        summary_records = json.load(f)

    EXPECTED_TOTAL_PIXELS = 134_217_728
    EXPECTED_GT_CHANGED = 6_837_404

    for rec in summary_records:
        total_p = rec["tp"] + rec["fp"] + rec["fn"] + rec["tn"]
        assert total_p == EXPECTED_TOTAL_PIXELS, (
            f"Pixel sum mismatch for {rec['model_id']}: {total_p} != {EXPECTED_TOTAL_PIXELS}"
        )
        gt_changed = rec["tp"] + rec["fn"]
        assert gt_changed == EXPECTED_GT_CHANGED, (
            f"GT changed mismatch for {rec['model_id']}: {gt_changed} != {EXPECTED_GT_CHANGED}"
        )


def test_bootstrap_ci_consistency():
    """Verifies scene bootstrap confidence interval bounds and monotonicity."""
    ci_p = Path("experiments/results/m11/statistical_bootstrap_ci.json")
    if not ci_p.exists():
        pytest.skip("M11 bootstrap CI not yet generated.")

    with open(ci_p, "r", encoding="utf-8") as f:
        ci_data = json.load(f)

    for model_name, stats_dict in ci_data.items():
        assert stats_dict["n_scenes"] == 128
        assert stats_dict["n_bootstraps"] >= 1000

        for metric in ["precision", "recall", "f1", "iou"]:
            m_stat = stats_dict[metric]
            low = m_stat["ci_lower"]
            mean = m_stat["mean"]
            high = m_stat["ci_upper"]

            assert 0.0 <= low <= 1.0
            assert 0.0 <= mean <= 1.0
            assert 0.0 <= high <= 1.0
            assert low <= high, f"CI inverted for {model_name} {metric}: {low} > {high}"
            assert low <= mean <= high, f"Mean outside CI for {model_name} {metric}: not {low} <= {mean} <= {high}"


def test_wilcoxon_paired_test_properties():
    """Verifies statistical hypothesis testing output format and pairing."""
    wilcoxon_p = Path("experiments/results/m11/paired_statistical_tests.json")
    if not wilcoxon_p.exists():
        pytest.skip("M11 paired tests not yet generated.")

    with open(wilcoxon_p, "r", encoding="utf-8") as f:
        tests = json.load(f)

    assert len(tests) >= 5
    for test in tests:
        assert test["n_scenes"] == 128
        assert 0.0 <= test["wilcoxon_pval_f1"] <= 1.0
        assert 0.0 <= test["wilcoxon_pval_iou"] <= 1.0
        assert isinstance(test["significant_at_05_f1"], bool)
        assert isinstance(test["significant_at_05_iou"], bool)


def test_qat_ablation_arithmetic():
    """Verifies QAT-CD ablation arithmetic and active gating trade-off verification."""
    abl_p = Path("experiments/results/m11/qat_ablation_analysis.json")
    if not abl_p.exists():
        pytest.skip("M11 QAT ablation file not yet generated.")

    with open(abl_p, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    conds = abl_data["ablation_conditions"]
    assert "Ablation_ActiveAll" in conds
    active = conds["Ablation_ActiveAll"]

    # In active gating:
    # False positives drop from 1,965,041 to 1,391,488 (573,553 pixels, ~29.19%)
    assert active["fp_reduction_pixels"] == 573553
    assert pytest.approx(active["fp_reduction_percent"], rel=1e-2) == 29.19

    # Precision increases by ~5.42 percentage points
    assert pytest.approx(active["precision_change_pp"], rel=1e-2) == 5.42

    # Benefit ratio: 573,553 / 336,213 ~= 1.71
    assert active["operational_benefit_ratio"] > 1.5


def test_benchmark_summary_schema():
    """Verifies schema conformance of benchmark_summary.json."""
    summary_p = Path("experiments/results/m11/benchmark_summary.json")
    if not summary_p.exists():
        pytest.skip("M11 benchmark summary not yet generated.")

    with open(summary_p, "r", encoding="utf-8") as f:
        records = json.load(f)

    required_keys = {
        "model_id",
        "display_name",
        "architecture_type",
        "threshold_value",
        "threshold_source",
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "total_scenes",
        "total_pixels",
        "parameters",
        "mean_latency_ms",
        "peak_rss_mb",
        "execution_status",
        "reproducibility_status",
    }

    for rec in records:
        missing = required_keys - set(rec.keys())
        assert not missing, f"Missing required keys in benchmark summary: {missing}"
        assert rec["execution_status"] in ["VERIFIED", "PARTIALLY_VERIFIED", "EXPLORATORY"]
