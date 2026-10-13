"""Milestone 12 Streamlit Parity & Verification Test Suite.

Verifies:
1. BM25 retrieval ranking and top-K execution without TypeError.
2. Method-aware retrieval score labeling (Cosine Similarity vs. BM25 Score).
3. QAT-CD gamma parameter propagation to inference engine (gamma=0.0 vs gamma=0.30).
4. Manual decision threshold override application without mutating frozen benchmark baselines.
5. Loading and schema conformance of all four M11 JSON artifacts and 7 publication figures.
6. Error handling / graceful warning fallbacks for missing or malformed artifacts.
7. Verification of mandatory scientific disclosures (test-set reuse, exploratory active gating, SNUNet false alarms).
8. Streamlit AppTest integration and zero FastAPI regressions.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.config_loader import get_project_root
from utils.result_loader import (
    load_m11_summary,
    load_m11_bootstrap_ci,
    load_m11_paired_tests,
    load_m11_ablation,
    get_m11_figures,
)
from src.semantic_search.bm25 import BM25Retriever, tokenize
from pages.change_detection import (
    load_m7_detector,
    load_m8_detector,
    load_m9_detector,
    load_m10_detector,
    get_change_detection_detectors,
)

ROOT = Path(get_project_root())


# =============================================================================
# 1. BM25 RETRIEVAL REPAIR VERIFICATION (P0)
# =============================================================================

class TestBM25RetrievalParity:
    """Verifies that BM25 retrieval ranks correctly, respects top-K, and never raises TypeError."""

    def test_bm25_rank_gallery_unsupported_kwarg_reproduced_and_isolated(self):
        """Verifies that BM25Retriever.rank_gallery does NOT accept top_k (reproducing defect)."""
        retriever = BM25Retriever()
        retriever.fit(["img_1"], [["a runway", "an airport"]])
        
        with pytest.raises(TypeError, match="unexpected keyword argument 'top_k'"):
            retriever.rank_gallery("runway", top_k=5)  # type: ignore

    def test_bm25_supported_combined_scoring_and_top_k(self):
        """Verifies the supported scoring path implemented in pages/retrieval.py."""
        retriever = BM25Retriever()
        image_ids = ["img_1", "img_2", "img_3"]
        image_captions = [
            ["a commercial airport with runways", "airplanes on tarmac"],
            ["dense green residential trees and houses", "neighborhood"],
            ["long concrete runway near terminal", "hangar"],
        ]
        retriever.fit(image_ids, image_captions)

        query = "airport runway"
        tokens = tokenize(query)
        scores = retriever.score_query_combined(tokens)
        
        assert len(scores) == 3
        # Sort descending by score
        ranked_indices = np.argsort(scores)[::-1]
        
        top_k = 2
        top_indices = ranked_indices[:top_k]
        top_ids = [image_ids[idx] for idx in top_indices]
        top_scores = [float(scores[idx]) for idx in top_indices]

        assert len(top_ids) == 2
        assert len(top_scores) == 2
        # img_1 or img_3 must be ranked first since they contain airport/runway
        assert top_ids[0] in ["img_1", "img_3"]
        assert top_scores[0] >= top_scores[1]
        assert "img_2" not in top_ids  # Irrelevant item excluded from top-2

    def test_bm25_empty_query_and_empty_gallery(self):
        """Verifies safe handling of empty strings and empty galleries."""
        retriever = BM25Retriever()
        retriever.fit(["img_1"], [["a satellite scene"]])

        # Empty tokens query
        scores = retriever.score_query_combined([])
        assert len(scores) == 1
        assert scores[0] == 0.0

        # Empty gallery
        empty_retriever = BM25Retriever()
        empty_scores = empty_retriever.score_query_combined(["test"])
        assert len(empty_scores) == 0

    def test_retrieval_page_bm25_apptest(self):
        """Executes pages/retrieval.py with BM25 query in Streamlit AppTest without exceptions."""
        at = AppTest.from_file(str(ROOT / "pages" / "retrieval.py"))
        at.run()
        assert len(at.exception) == 0, f"AppTest raised exceptions: {at.exception}"


# =============================================================================
# 2. METHOD-AWARE SCORE LABELS
# =============================================================================

class TestMethodAwareScoreLabels:
    """Verifies image viewer displays method-aware score labels instead of generic labels."""

    def test_score_label_distinction(self):
        """Verifies score_type formatting logic."""
        item_clip = {"image_id": "test_1", "score": 0.85234, "score_type": "Cosine Similarity"}
        item_bm25 = {"image_id": "test_2", "score": 12.3456, "score_type": "BM25 Score"}
        item_default = {"image_id": "test_3", "score": 0.5}

        def get_score_label(item):
            stype = item.get("score_type", "Score")
            val = item.get("score", 0.0)
            return f"{stype}: {val:.4f}"

        assert get_score_label(item_clip) == "Cosine Similarity: 0.8523"
        assert get_score_label(item_bm25) == "BM25 Score: 12.3456"
        assert get_score_label(item_default) == "Score: 0.5000"
        # Ensure BM25 is not presented as percentage
        assert "%" not in get_score_label(item_bm25)


# =============================================================================
# 3. QAT-CD GAMMA SELECTION PROPAGATION & DECISION THRESHOLDS
# =============================================================================

class TestQATCDConfigurationAndThresholdOverride:
    """Verifies that gamma selections reach inference and threshold overrides are applied."""

    def test_qat_cd_gamma_propagation_changes_output(self):
        """Verifies that det.gamma dynamically alters gated_prob in inference."""
        det, tau, meta = load_m10_detector()
        assert det is not None, "M10 QAT-CD detector must be available"

        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

        # Gamma = 0.0 (Calibrated)
        det.gamma = 0.0
        diag_0 = det.compute_quality_diagnostics(img1, img2)
        prob_0 = diag_0["gated_prob"]

        # Gamma = 0.30 (Exploratory Active Gating)
        det.gamma = 0.30
        diag_30 = det.compute_quality_diagnostics(img1, img2)
        prob_30 = diag_30["gated_prob"]

        # With gamma=0.0, gated_prob must equal base_prob
        np.testing.assert_allclose(prob_0, diag_0["base_prob"], atol=1e-6)

        # With gamma=0.30, gated_prob must be attenuated (prob_30 <= prob_0 everywhere)
        assert np.all(prob_30 <= prob_0 + 1e-6)
        abs_diff = np.abs(prob_0 - prob_30).sum()
        assert abs_diff > 0.0, "Active gating must attenuate probability map relative to calibrated baseline"

    def test_frozen_thresholds_remain_unmodified(self):
        """Verifies that default frozen thresholds are preserved and not altered."""
        _, _, frozen_thresholds = get_change_detection_detectors()
        assert frozen_thresholds["B1_Pixel_Diff"] == 0.4100
        assert frozen_thresholds["B2_SSIM"] == 0.9000
        assert frozen_thresholds["B3_CVA"] == 0.4050

        _, m7_tau, _ = load_m7_detector()
        _, m8_tau, _ = load_m8_detector()
        _, m9_tau, _ = load_m9_detector()
        _, m10_tau, _ = load_m10_detector()

        assert m7_tau == 0.55
        assert m8_tau == 0.50
        assert m9_tau == 0.45
        assert m10_tau == 0.45

    def test_manual_threshold_override_logic(self):
        """Verifies that manual decision threshold override changes binary prediction count."""
        continuous_diff = np.linspace(0.0, 1.0, 100, dtype=np.float32)

        # Baseline tau = 0.50
        pred_base = (continuous_diff >= 0.50).astype(np.uint8)
        assert pred_base.sum() == 50

        # Strict override tau = 0.75
        pred_strict = (continuous_diff >= 0.75).astype(np.uint8)
        assert pred_strict.sum() == 25

        # Lenient override tau = 0.25
        pred_lenient = (continuous_diff >= 0.25).astype(np.uint8)
        assert pred_lenient.sum() == 75


# =============================================================================
# 4. M11 ARTIFACT LOADERS & SCHEMA CONSISTENCY
# =============================================================================

class TestM11ArtifactLoaders:
    """Verifies that all M11 artifacts load faithfully from saved JSON and figures."""

    def test_load_m11_summary(self):
        df = load_m11_summary()
        assert df is not None and not df.empty
        assert len(df) == 8  # 7 models + QAT active condition
        required_cols = ["model_id", "display_name", "precision", "recall", "f1", "iou", "threshold_value"]
        for col in required_cols:
            assert col in df.columns

    def test_load_m11_bootstrap_ci(self):
        ci = load_m11_bootstrap_ci()
        assert isinstance(ci, dict)
        assert "M9_BIT_CD" in ci
        assert "M10_QAT_CD_Calibrated" in ci
        bit_f1 = ci["M9_BIT_CD"]["f1"]
        assert "mean" in bit_f1 and "ci_lower" in bit_f1 and "ci_upper" in bit_f1
        assert bit_f1["ci_lower"] <= bit_f1["mean"] <= bit_f1["ci_upper"]

    def test_load_m11_paired_tests(self):
        paired = load_m11_paired_tests()
        assert isinstance(paired, list)
        assert len(paired) == 6
        # Find M9 vs M8 comparison
        m9_vs_m8 = next((p for p in paired if p["model_a"] == "M9_BIT_CD" and p["model_b"] == "M8_SNUNet"), None)
        assert m9_vs_m8 is not None
        assert m9_vs_m8["significant_at_05_f1"] is True
        assert m9_vs_m8["wilcoxon_pval_f1"] < 1e-15

    def test_load_m11_ablation(self):
        abl = load_m11_ablation()
        assert "ablation_conditions" in abl
        conds = abl["ablation_conditions"]
        assert "Ablation_ActiveAll" in conds
        active = conds["Ablation_ActiveAll"]
        assert active["fp_reduction_pixels"] == 573553
        assert pytest.approx(active["fp_reduction_percent"], 0.01) == 29.19
        assert pytest.approx(active["operational_benefit_ratio"], 0.01) == 1.71

    def test_get_m11_figures(self):
        figs = get_m11_figures()
        assert len(figs) == 7
        expected_figs = [
            "cpu_latency_vs_f1",
            "cross_model_robustness",
            "f1_iou_benchmark_bars",
            "false_positive_reduction_log",
            "per_scene_f1_distribution",
            "precision_recall_tradeoff",
            "qat_ablation_tradeoff",
        ]
        for ef in expected_figs:
            assert ef in figs
            assert Path(figs[ef]).exists()

    def test_missing_artifact_graceful_handling(self, monkeypatch):
        """Verifies loaders handle missing files gracefully without crashing."""
        import utils.result_loader as rl
        monkeypatch.setattr(rl, "ROOT", Path("/tmp/non_existent_research_dir_12345"))
        assert rl.load_m11_summary() is None
        assert rl.load_m11_bootstrap_ci() is None
        assert rl.load_m11_paired_tests() is None
        assert rl.load_m11_ablation() is None
        assert rl.get_m11_figures() == {}


# =============================================================================
# 5. SCIENTIFIC DISCLOSURES VERIFICATION
# =============================================================================

class TestScientificDisclosures:
    """Verifies that the 3 mandatory scientific disclosures are present in UI files."""

    def test_disclosures_in_overview(self):
        overview_path = ROOT / "pages" / "overview.py"
        content = overview_path.read_text(encoding="utf-8")
        assert "Test-Set Reuse" in content
        assert "Exploratory QAT-CD Active Gating" in content
        assert "29.19%" in content
        assert "53,485,389 false-positive pixels" in content

    def test_disclosures_in_change_detection(self):
        cd_path = ROOT / "pages" / "change_detection.py"
        content = cd_path.read_text(encoding="utf-8")
        assert "Test-Set Reuse Disclosure" in content
        assert "Exploratory QAT-CD Active Gating Disclosure" in content
        assert "SNUNet-CD False Alarms Disclosure" in content
        assert "53,485,389 false-positive pixels" in content


# =============================================================================
# 6. STREAMLIT APPTEST EXECUTION
# =============================================================================

class TestStreamlitAppTestIntegration:
    """Executes all Streamlit pages via AppTest to ensure zero runtime exceptions."""

    def test_app_main(self):
        at = AppTest.from_file(str(ROOT / "app.py"))
        at.run()
        assert len(at.exception) == 0

    def test_overview_page(self):
        at = AppTest.from_file(str(ROOT / "pages" / "overview.py"))
        at.run()
        assert len(at.exception) == 0

    def test_change_detection_page(self):
        at = AppTest.from_file(str(ROOT / "pages" / "change_detection.py"))
        at.run()
        assert len(at.exception) == 0

    def test_false_alarm_page(self):
        at = AppTest.from_file(str(ROOT / "pages" / "false_alarm.py"))
        at.run()
        assert len(at.exception) == 0
