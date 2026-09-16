"""Unit tests for ChangeDetectionEvaluator, ConfusionMatrix, and ValidationThresholdOptimizer."""

import numpy as np
import pytest

from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from src.change_detection.thresholding import ValidationThresholdOptimizer


def test_confusion_matrix_perfect_prediction():
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    cm = ConfusionMatrix()
    cm.update(pred, gt)

    assert cm.tp == 2
    assert cm.fp == 0
    assert cm.fn == 0
    assert cm.tn == 2
    assert cm.total == 4

    metrics = ChangeDetectionEvaluator.evaluate(cm)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_confusion_matrix_zero_division_guards():
    # All zeros in prediction and ground truth (no change present, none predicted)
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)

    cm = ConfusionMatrix()
    cm.update(pred, gt)
    metrics = ChangeDetectionEvaluator.evaluate(cm)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["accuracy"] == 1.0
    assert not np.isnan(metrics["f1"])


def test_validation_threshold_optimizer_synthetic():
    opt = ValidationThresholdOptimizer(min_threshold=0.1, max_threshold=0.9, num_candidates=9, num_bins=100)

    # Let changed pixels have difference = 0.8, unchanged pixels have difference = 0.2
    diff = np.array([[0.8, 0.2], [0.8, 0.2]], dtype=np.float32)
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    opt.update(diff, gt)
    res = opt.optimize()

    # Optimal threshold should cleanly separate 0.8 from 0.2 (e.g. between 0.3 and 0.8)
    assert 0.2 < res.best_threshold <= 0.8
    assert res.best_f1 == 1.0
    assert res.best_precision == 1.0
    assert res.best_recall == 1.0


def test_validation_threshold_optimizer_empty_raises():
    opt = ValidationThresholdOptimizer()
    with pytest.raises(ValueError, match="without observing"):
        opt.optimize()
