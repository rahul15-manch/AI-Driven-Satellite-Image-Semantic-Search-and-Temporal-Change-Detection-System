"""Unit tests for Milestone M10 Quality-Aware Temporal Change Detection (QAT-CD).

Tests:
1. Mathematical invariants: quality signals strictly bounded in [0.0, 1.0].
2. Temporal symmetry: swapping T1 and T2 produces identical quality maps and predictions.
3. Edge case stability: constant images, identical pairs, extreme contrast, zeros.
4. Detector uncertainty formulation: peaks at P=0.5, vanishes at P=0.0 and P=1.0.
5. Gating behavior: identity when gamma=0.0, confident predictions unattenuated.
6. Detector interface: conforms to BaseChangeDetector contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.change_detection.base import BaseChangeDetector
from src.change_detection.qat_cd import (
    QualityAwareChangeDetector,
    QATChangeDetector,
    extract_registration_discrepancy,
    extract_illumination_discrepancy,
    extract_sharpness_discrepancy,
    compute_composite_defect_index,
    compute_detector_uncertainty,
    quality_aware_gate,
)


class DummyBaseDetector(BaseChangeDetector):
    """Deterministic dummy base change detector for unit tests."""

    def __init__(self, constant_prob: float = 0.50):
        super().__init__(name="DummyBaseDetector")
        self.constant_prob = constant_prob

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        h, w = img1.shape[:2]
        return np.full((h, w), self.constant_prob, dtype=np.float32)


class SymmetricBaseDetector(BaseChangeDetector):
    """Symmetric difference detector based on L1 difference."""

    def __init__(self):
        super().__init__(name="SymmetricBaseDetector")

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        arr1 = self._ensure_float_rgb(img1)
        arr2 = self._ensure_float_rgb(img2)
        diff = np.abs(arr1 - arr2).mean(axis=-1)
        return np.clip(diff, 0.0, 1.0).astype(np.float32)


class TestQATMathematicalInvariants:
    """Verifies bounds, ranges, and mathematical properties of individual signals."""

    def test_registration_discrepancy_bounds(self):
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        s_reg = extract_registration_discrepancy(img1, img2)
        assert s_reg.shape == (64, 64)
        assert s_reg.dtype == np.float32
        assert np.all(s_reg >= 0.0)
        assert np.all(s_reg <= 1.0)
        assert not np.isnan(s_reg).any()
        assert not np.isinf(s_reg).any()

    def test_illumination_discrepancy_bounds(self):
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        s_illum = extract_illumination_discrepancy(img1, img2)
        assert s_illum.shape == (64, 64)
        assert s_illum.dtype == np.float32
        assert np.all(s_illum >= 0.0)
        assert np.all(s_illum <= 1.0)
        assert not np.isnan(s_illum).any()

    def test_sharpness_discrepancy_bounds(self):
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        s_sharp = extract_sharpness_discrepancy(img1, img2)
        assert s_sharp.shape == (64, 64)
        assert s_sharp.dtype == np.float32
        assert np.all(s_sharp >= 0.0)
        assert np.all(s_sharp <= 1.0)
        assert not np.isnan(s_sharp).any()

    def test_composite_defect_index_bounds(self):
        shape = (64, 64)
        s1 = np.full(shape, 0.8, dtype=np.float32)
        s2 = np.full(shape, 0.4, dtype=np.float32)
        s3 = np.full(shape, 0.2, dtype=np.float32)

        q = compute_composite_defect_index(s1, s2, s3, weights=(0.34, 0.33, 0.33))
        assert q.shape == shape
        assert np.all(q >= 0.0)
        assert np.all(q <= 1.0)
        expected = 0.34 * 0.8 + 0.33 * 0.4 + 0.33 * 0.2
        np.testing.assert_allclose(q, expected, atol=1e-5)

    def test_detector_uncertainty_behavior(self):
        p_vals = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32)
        u_vals = compute_detector_uncertainty(p_vals)

        # U(0.0) == 0.0
        assert u_vals[0] == pytest.approx(0.0, abs=1e-5)
        # U(1.0) == 0.0
        assert u_vals[4] == pytest.approx(0.0, abs=1e-5)
        # U(0.5) == 4 * 0.5 * 0.5 == 1.0 (maximal uncertainty)
        assert u_vals[2] == pytest.approx(1.0, abs=1e-5)
        # Symmetry: U(0.1) == U(0.9) == 4 * 0.1 * 0.9 == 0.36
        assert u_vals[1] == pytest.approx(0.36, abs=1e-5)
        assert u_vals[3] == pytest.approx(0.36, abs=1e-5)


class TestTemporalSymmetry:
    """Verifies that swapping T1 and T2 produces identical quality signals and predictions."""

    def test_signals_temporal_symmetry(self):
        np.random.seed(123)
        img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        s_reg_12 = extract_registration_discrepancy(img1, img2)
        s_reg_21 = extract_registration_discrepancy(img2, img1)
        np.testing.assert_allclose(s_reg_12, s_reg_21, atol=1e-6)

        s_illum_12 = extract_illumination_discrepancy(img1, img2)
        s_illum_21 = extract_illumination_discrepancy(img2, img1)
        np.testing.assert_allclose(s_illum_12, s_illum_21, atol=1e-6)

        s_sharp_12 = extract_sharpness_discrepancy(img1, img2)
        s_sharp_21 = extract_sharpness_discrepancy(img2, img1)
        np.testing.assert_allclose(s_sharp_12, s_sharp_21, atol=1e-6)

    def test_qat_detector_temporal_symmetry(self):
        np.random.seed(456)
        img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        base_det = SymmetricBaseDetector()
        qat_det = QualityAwareChangeDetector(base_detector=base_det, gamma=0.30)

        diff_12 = qat_det.compute_difference_map(img1, img2)
        diff_21 = qat_det.compute_difference_map(img2, img1)

        np.testing.assert_allclose(diff_12, diff_21, atol=1e-6)

        mask_12, _ = qat_det.predict(img1, img2, threshold=0.40)
        mask_21, _ = qat_det.predict(img2, img1, threshold=0.40)
        np.testing.assert_array_equal(mask_12, mask_21)


class TestEdgeCasesAndStability:
    """Verifies numerical stability on edge case inputs."""

    def test_identical_images(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        s_reg = extract_registration_discrepancy(img, img)
        s_illum = extract_illumination_discrepancy(img, img)
        s_sharp = extract_sharpness_discrepancy(img, img)

        assert np.all(s_reg == 0.0)
        assert np.all(s_illum == 0.0)
        assert np.all(s_sharp == 0.0)

    def test_all_black_images(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        s_reg = extract_registration_discrepancy(img, img)
        s_illum = extract_illumination_discrepancy(img, img)
        s_sharp = extract_sharpness_discrepancy(img, img)

        assert not np.isnan(s_reg).any()
        assert not np.isnan(s_illum).any()
        assert not np.isnan(s_sharp).any()

    def test_extreme_contrast_and_illumination(self):
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.full((64, 64, 3), 255, dtype=np.uint8)

        s_illum = extract_illumination_discrepancy(img1, img2)
        assert not np.isnan(s_illum).any()
        assert np.all(s_illum <= 1.0)
        assert np.all(s_illum >= 0.0)

    def test_zero_gamma_identity_ablation(self):
        base_p = np.array([[0.2, 0.5], [0.8, 0.95]], dtype=np.float32)
        defect_q = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)

        gated = quality_aware_gate(base_p, defect_q, gamma=0.0, use_uncertainty=True)
        np.testing.assert_array_equal(gated, base_p)

    def test_confident_change_preservation(self):
        # A confident prediction (P=0.999) has uncertainty U ~= 0.004
        base_p = np.array([0.999], dtype=np.float32)
        defect_q = np.array([1.0], dtype=np.float32)  # High defect risk

        gated = quality_aware_gate(base_p, defect_q, gamma=0.50, use_uncertainty=True)
        # Should remain highly confident (> 0.99)
        assert gated[0] > 0.99


class TestQATDetectorIntegration:
    """Verifies QualityAwareChangeDetector wrapper contracts."""

    def test_diagnostics_dictionary_keys(self):
        img1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
        img2 = np.ones((64, 64, 3), dtype=np.uint8) * 150

        base_det = DummyBaseDetector(constant_prob=0.60)
        qat_det = QualityAwareChangeDetector(base_detector=base_det, gamma=0.25)

        diag = qat_det.compute_quality_diagnostics(img1, img2)
        required_keys = {
            "base_prob",
            "s_reg",
            "s_illum",
            "s_sharp",
            "uncertainty",
            "defect_index",
            "gated_prob",
        }
        assert required_keys.issubset(diag.keys())

        for k, v in diag.items():
            assert isinstance(v, np.ndarray)
            assert v.shape == (64, 64)
            assert np.all(v >= 0.0)
            assert np.all(v <= 1.0)

    def test_predict_contract(self):
        img1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        base_det = DummyBaseDetector(constant_prob=0.50)
        qat_det = QATChangeDetector(base_detector=base_det, gamma=0.20)

        mask, diff_map = qat_det.predict(img1, img2, threshold=0.45)
        assert mask.shape == (32, 32)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert diff_map.shape == (32, 32)
        assert diff_map.dtype == np.float32
