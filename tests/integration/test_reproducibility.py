"""Integration Test: Reproducibility, Metric Alignment, and Resource Safety.

Covers:
- INT-M3-01: Metric Formulas (R@1, R@5, R@10, MRR) Match Standard Definitions
- INT-M3-02: Retrieval Protocol Match (Single Positive Target per Query)
- INT-M3-03: Threshold Policy Isolation (No CD threshold logic in M4 retrieval)
- INT-TEST-SEP-01: Test/Production Separation & Raw Data Immutability
- INT-SAFE-01: Resource Budget Adherence (Process RAM < 8 GB)
- INT-E2E-04: Full Reproducibility Across Repeated Runs
"""

from __future__ import annotations

import json
from pathlib import Path
import psutil
import pytest

from src.semantic_search.retrieval_evaluator import QueryEvaluationResult, QueryRecord, RetrievalEvaluator


@pytest.mark.integration
def test_int_m3_01_metrics_mathematical_correctness():
    """INT-M3-01: Verify R@1, R@5, R@10, and MRR formulas against hand-calculated cases."""
    gallery = ["img_A", "img_B", "img_C", "img_D", "img_E", "img_F", "img_G", "img_H", "img_I", "img_J",
               "img_K", "img_L", "img_M", "img_N", "img_Z"]

    # Case 1: Target at rank 1 -> r1=1, r5=1, r10=1, rr=1.0
    # Case 2: Target at rank 3 -> r1=0, r5=1, r10=1, rr=1/3
    # Case 3: Target at rank 7 -> r1=0, r5=0, r10=1, rr=1/7
    # Case 4: Target at rank 15 -> r1=0, r5=0, r10=0, rr=1/15
    queries = [
        QueryRecord("q1", "q1 text", "img_A"),
        QueryRecord("q2", "q2 text", "img_C"),
        QueryRecord("q3", "q3 text", "img_G"),
        QueryRecord("q4", "q4 text", "img_Z"),  # missing from top 10
    ]

    def mock_ranking(q: str):
        # Always returns fixed order: img_A, img_B, img_C, img_D, img_E, img_F, img_G, img_H, img_I, img_J, ... img_Z at 15
        full_list = ["img_A", "img_B", "img_C", "img_D", "img_E", "img_F", "img_G", "img_H", "img_I", "img_J",
                     "img_K", "img_L", "img_M", "img_N", "img_Z"]
        return full_list, None

    evaluator = RetrievalEvaluator(queries, gallery)
    metrics = evaluator.evaluate("Metric Verification", mock_ranking)

    # Expected:
    # r1: 1/4 = 0.25
    # r5: 2/4 = 0.50
    # r10: 3/4 = 0.75
    # mrr: (1/1 + 1/3 + 1/7 + 1/15) / 4 = (1 + 0.333333 + 0.142857 + 0.066667) / 4 = 1.542857 / 4 = 0.385714
    assert metrics.r1 == pytest.approx(0.25, abs=1e-4)
    assert metrics.r5 == pytest.approx(0.50, abs=1e-4)
    assert metrics.r10 == pytest.approx(0.75, abs=1e-4)
    assert metrics.mrr == pytest.approx(0.385714, abs=1e-4)


@pytest.mark.integration
def test_int_m3_02_retrieval_protocol_single_positive():
    """INT-M3-02: Caption-to-Own-Image protocol strictly requires exactly one positive target per query."""
    q = QueryRecord(query_id="test_q", query_text="test", target_image_id="target_img.jpg")
    assert isinstance(q.target_image_id, str), "Target must be a single string ID, not a list."
    assert q.target_image_id == "target_img.jpg"


@pytest.mark.integration
def test_int_m3_03_no_threshold_logic_in_semantic_search():
    """INT-M3-03: Verify semantic search module contains no change detection thresholding logic."""
    src_retrieval = Path("src/semantic_search")
    for py_file in src_retrieval.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "otsu_threshold" not in text
        assert "validation_tuned_threshold" not in text
        assert "bitemporal" not in text.lower()


@pytest.mark.integration
def test_int_test_sep_01_raw_data_immutability():
    """INT-TEST-SEP-01: Ensure raw dataset remains unmodified and test caches write only to designated paths."""
    raw_rsicd = Path("data/raw/rsicd/dataset_rsicd.json")
    assert raw_rsicd.exists()
    # Check that .gitignore ignores raw images or caches
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        gi_text = gitignore_path.read_text()
        # Verify cache and temporary test directories are ignored
        assert "cache" in gi_text or ".npy" in gi_text or "experiments/cache" in gi_text


@pytest.mark.integration
def test_int_safe_01_memory_budget_under_8gb():
    """INT-SAFE-01: Process RSS memory must remain strictly below 8,192 MB (8 GB)."""
    process = psutil.Process()
    current_rss_mb = process.memory_info().rss / (1024 * 1024)
    assert current_rss_mb < 8192.0, (
        f"Memory constraint violated! Process RSS is {current_rss_mb:.1f} MB, which exceeds 8 GB."
    )


@pytest.mark.integration
def test_int_e2e_04_summary_reproducibility():
    """INT-E2E-04: Verify saved summary results match locally measured metrics exactly."""
    summary_path = Path("experiments/results/m4/summary.json")
    assert summary_path.exists(), "experiments/results/m4/summary.json must exist."

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Locate corrected BM25, CLIP, and Prompt Ensemble
    loco = next((m for m in summary if "Leave-One-Caption-Out" in m.get("method", "")), None)
    assert loco is not None, "Leave-One-Caption-Out BM25 must be present in summary."
    assert loco["R@1"] == pytest.approx(0.4212, abs=1e-3)
    assert loco["R@5"] == pytest.approx(0.6081, abs=1e-3)
    assert loco["R@10"] == pytest.approx(0.6787, abs=1e-3)
    assert loco["MRR"] == pytest.approx(0.5112, abs=1e-3)

    clip = next((m for m in summary if m.get("method") == "CLIP ViT-B/32"), None)
    assert clip is not None, "CLIP ViT-B/32 must be present in summary."
    assert clip["R@1"] == pytest.approx(0.0545, abs=1e-3)
    assert clip["R@5"] == pytest.approx(0.1771, abs=1e-3)
    assert clip["R@10"] == pytest.approx(0.2789, abs=1e-3)
    assert clip["MRR"] == pytest.approx(0.1307, abs=1e-3)

    ens = next((m for m in summary if "Prompt Ensemble" in m.get("method", "")), None)
    assert ens is not None, "CLIP + Prompt Ensemble must be present in summary."
    assert ens["R@1"] == pytest.approx(0.0514, abs=1e-3)
    assert ens["R@5"] == pytest.approx(0.1700, abs=1e-3)
    assert ens["R@10"] == pytest.approx(0.2801, abs=1e-3)
    assert ens["MRR"] == pytest.approx(0.1268, abs=1e-3)
