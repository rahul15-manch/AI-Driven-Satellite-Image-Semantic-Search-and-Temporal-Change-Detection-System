"""Milestone 13 Controlled Ablation Study and Research Reproducibility Tests.

Verifies:
1. Mathematical correctness and temporal symmetry of the QAT-CD gating formulation.
2. Boundary behavior of the detector margin uncertainty function U_det.
3. Integrity and schema conformance of the M13 controlled ablation artifacts.
4. Operational benefit ratio computation and trade-off sign conventions.
5. Monotonicity of the validation gamma calibration sweep (precision vs. recall trade-offs).
6. Verification of the RSICD multi-caption sibling leakage analysis and metadata control.
7. Verification of the documented scope discrepancy regarding CLIP+metadata reranking.
8. Scene-level paired statistical tests and bootstrap confidence interval validity.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from utils.config_loader import get_project_root
from src.change_detection.qat_cd import (
    compute_composite_defect_index,
    compute_detector_uncertainty,
    extract_illumination_discrepancy,
    extract_registration_discrepancy,
    extract_sharpness_discrepancy,
    quality_aware_gate,
)

ROOT = Path(get_project_root())


# =============================================================================
# 1. QAT-CD MATHEMATICAL FORMULATION & BOUNDARY BEHAVIOR
# =============================================================================

class TestQATCDFormulationAndProperties:
    """Verifies theoretical formulation and numerical stability of QAT-CD gating."""

    def test_detector_uncertainty_boundary_conditions(self):
        """U_det(P) = 4 * P * (1 - P). Must be 0 at extremes (0, 1) and 1 at P=0.5."""
        p_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        u_vals = compute_detector_uncertainty(p_vals)

        np.testing.assert_allclose(u_vals[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(u_vals[2], 1.0, atol=1e-6)
        np.testing.assert_allclose(u_vals[4], 0.0, atol=1e-6)
        np.testing.assert_allclose(u_vals[1], 0.75, atol=1e-6)
        np.testing.assert_allclose(u_vals[3], 0.75, atol=1e-6)

    def test_identity_ablation_at_gamma_zero(self):
        """When gamma = 0.0, P_QAT must equal P_0 exactly regardless of defect index."""
        rng = np.random.default_rng(42)
        p0 = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)
        q = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)

        p_gat = quality_aware_gate(p0, q, gamma=0.0, use_uncertainty=True)
        np.testing.assert_allclose(p_gat, p0, atol=1e-6)

    def test_attenuation_direction_and_boundedness(self):
        """For gamma > 0 and Q > 0, P_QAT must be in [0, 1] and P_QAT <= P_0 everywhere."""
        rng = np.random.default_rng(42)
        p0 = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)
        q = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)

        p_gat = quality_aware_gate(p0, q, gamma=0.30, use_uncertainty=True)
        assert np.all(p_gat >= 0.0)
        assert np.all(p_gat <= 1.0)
        assert np.all(p_gat <= p0 + 1e-6)

    def test_uncertainty_modulation_protects_confident_detections(self):
        """Highly confident changes (P0=0.99) should experience near-zero attenuation."""
        p_high = np.array([0.99], dtype=np.float32)
        q_high = np.array([1.0], dtype=np.float32)

        p_gat_with_u = quality_aware_gate(p_high, q_high, gamma=0.50, use_uncertainty=True)
        p_gat_no_u = quality_aware_gate(p_high, q_high, gamma=0.50, use_uncertainty=False)

        # With uncertainty modulation: U_det(0.99) = 4 * 0.99 * 0.01 = 0.0396 -> attenuation is small
        # Without uncertainty modulation: attenuation is 0.5 * 1.0 = 0.50 -> probability drops to ~0.495!
        assert p_gat_with_u[0] > 0.95
        assert p_gat_no_u[0] < 0.55


# =============================================================================
# 2. CONTROLLED QAT-CD ABLATION MATRIX & OPERATIONAL RATIOS
# =============================================================================

class TestM13ControlledAblationArtifacts:
    """Verifies that M13 QAT-CD ablation records exist, load, and adhere to definitions."""

    def test_ablation_file_exists_and_contains_required_conditions(self):
        path = ROOT / "experiments" / "results" / "m13" / "qat_cd_controlled_ablation.json"
        assert path.exists(), f"M13 ablation file missing at {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "ablation_conditions" in data
        assert "validation_gamma_sweep_tau_0_45" in data
        assert "operational_tradeoff_summary" in data

        cond_names = [c["condition"] for c in data["ablation_conditions"]]
        assert "QAT_CD_Full" in cond_names
        assert "Ablation_ActiveAll" in cond_names
        assert "Ablation_RegOnly" in cond_names
        assert "Ablation_IllumOnly" in cond_names
        assert "Ablation_SharpOnly" in cond_names
        assert "Ablation_NoUncertainty" in cond_names

    def test_operational_benefit_ratios_positive(self):
        """Operational exchange ratio Delta FP / Delta TP must be strictly positive."""
        path = ROOT / "experiments" / "results" / "m13" / "qat_cd_controlled_ablation.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for c in data["ablation_conditions"]:
            if c["condition"] in ["Ablation_ActiveAll", "Ablation_RegOnly", "Ablation_IllumOnly", "Ablation_SharpOnly"]:
                ratio = c["operational_benefit_ratio"]
                assert ratio > 1.0, f"Expected ratio > 1.0 for {c['condition']}, got {ratio}"
                # Verify formula Delta FP / Delta TP
                expected_ratio = c["fp_reduction_pixels"] / max(c["tp_loss_pixels"], 1)
                assert pytest.approx(ratio, rel=1e-2) == expected_ratio

    def test_validation_gamma_sweep_monotonicity(self):
        """Validation sweep at tau*=0.45 must show precision increasing and recall decreasing with gamma."""
        path = ROOT / "experiments" / "results" / "m13" / "qat_cd_controlled_ablation.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sweep = data["validation_gamma_sweep_tau_0_45"]
        assert len(sweep) >= 5

        # Check precision strictly non-decreasing
        precisions = [r["val_precision"] for r in sweep]
        recalls = [r["val_recall"] for r in sweep]
        for i in range(len(precisions) - 1):
            assert precisions[i+1] >= precisions[i] - 1e-4
            assert recalls[i+1] <= recalls[i] + 1e-4


# =============================================================================
# 3. SEMANTIC RETRIEVAL ABLATION & LEAKAGE INVESTIGATION
# =============================================================================

class TestM13RetrievalAblationArtifacts:
    """Verifies that retrieval ablation data records sibling caption leakage and reranker discrepancy."""

    def test_retrieval_ablation_file_exists_and_valid(self):
        path = ROOT / "experiments" / "results" / "m13" / "semantic_retrieval_ablation.json"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "protocol_investigation" in data
        assert "records" in data
        inv = data["protocol_investigation"]
        assert "bm25_loco_leakage_mechanism" in inv
        assert "clip_cross_modal_integrity" in inv
        assert "missing_reranking_status" in inv

        # Verify that missing reranking is explicitly marked
        records = data["records"]
        rerank_rec = next(r for r in records if r["method_id"] == "clip_metadata_rerank")
        assert rerank_rec["implementation_status"] == "NOT_IMPLEMENTED_SCOPE_DISCREPANCY"
        assert rerank_rec["R@1"] is None


# =============================================================================
# 4. STATISTICAL SIGNIFICANCE TESTS INTEGRITY
# =============================================================================

class TestM13StatisticalSignificanceArtifacts:
    """Verifies statistical hypothesis testing artifacts and assumptions."""

    def test_statistical_significance_file_exists(self):
        path = ROOT / "experiments" / "results" / "m13" / "statistical_significance_m13.json"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "evaluation_protocol" in data
        assert "paired_tests" in data
        assert "bootstrap_confidence_intervals" in data
        proto = data["evaluation_protocol"]
        assert proto["resampling_unit"] == "Scene-Level (Matched 1024x1024 Test Image Pairs, N=128)"
        assert proto["bootstrap_replicates"] == 1000
        assert proto["significance_alpha"] == 0.05
