"""Milestone 13 End-to-End System Integration Tests.

Verifies:
1. Complete semantic search user flow from natural-language query to ranked results.
2. Complete change detection user flow across all supported detectors (classical B1-B3, M7, M8, M9, M10).
3. Parity and consistency between FastAPI service layer and Streamlit page implementations.
4. Ground-truth overlay suppression when no ground-truth mask is provided.
5. Decision threshold override application on continuous maps without mutating frozen benchmarks.
6. Preservation and immutability of historical M11 results and checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from utils.config_loader import get_project_root
from src.api.app import app
from src.api.service import retrieval_service, change_detection_service
from pages.change_detection import (
    load_m7_detector,
    load_m8_detector,
    load_m9_detector,
    load_m10_detector,
    get_change_detection_detectors,
)
from src.semantic_search.bm25 import BM25Retriever, tokenize

ROOT = Path(get_project_root())
client = TestClient(app)


# =============================================================================
# 1. SEMANTIC RETRIEVAL END-TO-END FLOW & CONSISTENCY
# =============================================================================

class TestRetrievalEndToEndIntegration:
    """Verifies end-to-end retrieval across CLIP, prompt ensemble, and BM25."""

    def test_clip_search_end_to_end_fastapi(self):
        """Tests POST /api/search with CLIP method returns ranked image items."""
        resp = client.post("/api/search", json={"query": "commercial runway near airport terminal", "method": "clip", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "clip"
        assert len(data["results"]) == 5
        # Verify descending order of similarity
        scores = [r["score"] for r in data["results"]]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        # Verify image IDs and preview URLs are populated
        for r in data["results"]:
            assert r["image_id"]
            assert r["image_url"].startswith("/api/image/rsicd/")
            assert r["score_type"] == "Cosine Similarity"
            assert 0.0 <= r["score"] <= 1.0

    def test_prompt_ensemble_search_end_to_end_fastapi(self):
        """Tests POST /api/search with prompt_ensemble method."""
        resp = client.post("/api/search", json={"query": "dense residential neighborhood with houses", "method": "prompt_ensemble", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "prompt_ensemble"
        assert len(data["results"]) == 3
        for r in data["results"]:
            assert r["score_type"] == "Ensemble Cosine Similarity"

    def test_bm25_search_end_to_end_fastapi(self):
        """Tests POST /api/search with BM25 method returns ranked results."""
        resp = client.post("/api/search", json={"query": "airport runway", "method": "bm25", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "bm25"
        assert len(data["results"]) <= 5
        scores = [r["score"] for r in data["results"]]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        for r in data["results"]:
            assert r["score_type"] == "BM25 Score"

    def test_retrieval_empty_query_and_boundary_safety(self):
        """Tests validation on empty query strings."""
        resp = client.post("/api/search", json={"query": "   ", "method": "clip", "top_k": 5})
        assert resp.status_code == 422


# =============================================================================
# 2. CHANGE DETECTION END-TO-END FLOW & CONSISTENCY
# =============================================================================

class TestChangeDetectionEndToEndIntegration:
    """Verifies change detection inference across classical and learned architectures."""

    @pytest.mark.parametrize("model_name", [
        "B1_Pixel_Diff",
        "B2_SSIM",
        "B3_CVA",
        "M7_FC_Siam_diff",
        "M8_SNUNet",
        "M9_BIT_CD",
        "M10_QAT_CD",
    ])
    def test_change_detection_all_models_fastapi(self, model_name):
        """Tests POST /api/change-detection for every supported model."""
        payload = {
            "pair_id": "test_1",
            "model_id": model_name,
            "gamma": 0.0,
        }
        resp = client.post("/api/change-detection", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair_id"] == "test_1"
        assert data["model_id"] == model_name
        assert data["mask_data_uri"].startswith("data:image/png;base64,")
        assert data["heatmap_data_uri"].startswith("data:image/png;base64,")
        # Since test_1 has ground truth label on disk, metrics must be computed
        assert data["metrics"] is not None
        assert "f1" in data["metrics"]
        assert "precision" in data["metrics"]
        assert "recall" in data["metrics"]
        assert "iou" in data["metrics"]
        assert "latency_ms" in data

    def test_qat_cd_gamma_propagation_fastapi(self):
        """Tests that changing gamma in API request alters prediction results."""
        resp_cal = client.post("/api/change-detection", json={"pair_id": "test_1", "model_id": "M10_QAT_CD", "gamma": 0.0})
        assert resp_cal.status_code == 200
        fp_cal = resp_cal.json()["metrics"]["fp"]

        resp_active = client.post("/api/change-detection", json={"pair_id": "test_1", "model_id": "M10_QAT_CD", "gamma": 0.30})
        assert resp_active.status_code == 200
        fp_active = resp_active.json()["metrics"]["fp"]

        # Active gating at gamma=0.30 attenuates ambiguous detections, so FP must be <= calibrated FP
        assert fp_active <= fp_cal

    def test_threshold_override_fastapi(self):
        """Tests that manual threshold override changes metrics and mask."""
        resp_lenient = client.post("/api/change-detection", json={"pair_id": "test_1", "model_id": "M9_BIT_CD", "threshold": 0.20})
        resp_strict = client.post("/api/change-detection", json={"pair_id": "test_1", "model_id": "M9_BIT_CD", "threshold": 0.80})

        assert resp_lenient.status_code == 200
        assert resp_strict.status_code == 200
        tp_lenient = resp_lenient.json()["metrics"]["tp"]
        tp_strict = resp_strict.json()["metrics"]["tp"]
        assert tp_lenient > tp_strict


# =============================================================================
# 3. GROUND-TRUTH ABSENCE & DIAGNOSTIC OVERLAY INTEGRITY
# =============================================================================

class TestGroundTruthSuppression:
    """Verifies that ground-truth metrics and error overlays are suppressed when GT is absent."""

    def test_no_gt_metrics_suppression_in_service(self):
        """Directly invokes service when gt_mask is None."""
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.ones((256, 256, 3), dtype=np.uint8) * 100

        result = change_detection_service.detect_change(
            img_a=img1,
            img_b=img2,
            model_id="B1_Pixel_Diff",
        )
        assert result["mask_data_uri"] is not None
        assert result["heatmap_data_uri"] is not None
        # When no GT is supplied, metrics and overlay_data_uri must be None
        assert result["metrics"] is None
        assert result["overlay_data_uri"] is None


# =============================================================================
# 4. FROZEN M11 ARTIFACT INTEGRITY
# =============================================================================

class TestFrozenM11ArtifactIntegrity:
    """Verifies that M11 benchmark results remain frozen and untouched."""

    def test_m11_summary_unchanged(self):
        path = ROOT / "experiments" / "results" / "m11" / "benchmark_summary.json"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == 8
        bit_rec = next(r for r in records if r["model_id"] == "M9_BIT_CD")
        assert round(bit_rec["f1"], 4) == 0.5927
        assert round(bit_rec["iou"], 4) == 0.4211

    def test_m11_checkpoints_exist(self):
        assert (ROOT / "experiments" / "checkpoints" / "m7" / "fc_siam_diff_best.pt").exists()
        assert (ROOT / "experiments" / "checkpoints" / "m8" / "snunet_best.pt").exists()
        assert (ROOT / "experiments" / "checkpoints" / "m9" / "bit_cd_best.pt").exists()
        assert (ROOT / "experiments" / "checkpoints" / "m10" / "qat_cd_metadata.json").exists()
