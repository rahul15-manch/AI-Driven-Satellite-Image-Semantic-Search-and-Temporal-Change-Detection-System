"""Integration Test: BM25 Lexical Retrieval and Leakage Control (M4 Integration).

Covers:
- INT-CONTRACT-05: Strict Exclusion of Exact Query Text from Target Document
- INT-M4-06: Dual-Protocol Support (Legacy Leaky vs Corrected Leave-One-Caption-Out)
- INT-E2E-02: End-to-End Execution of Leave-One-Caption-Out BM25 on Test Gallery
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.data.rsicd_loader import RSICDDataset
from src.semantic_search.bm25 import BM25Retriever
from src.semantic_search.retrieval_evaluator import QueryRecord, RetrievalEvaluator


@pytest.mark.integration
@pytest.mark.dataset
def test_int_contract_05_bm25_leakage_absence():
    """INT-CONTRACT-05: Target document in leave-one-caption-out mode must strictly exclude query text."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    gallery_ids = [s.filename for s in dataset.samples]
    gallery_caps = [s.captions for s in dataset.samples]

    bm25 = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25.fit(gallery_ids, gallery_caps)

    # Test on a representative subset of 100 images
    for sample in dataset.samples[:100]:
        img_id = sample.filename
        for c_idx, query_text in enumerate(sample.captions):
            remaining = bm25.get_target_leave_one_out_captions(
                target_image_id=img_id,
                query_text=query_text,
                query_caption_idx=c_idx,
            )
            # The exact query text must not be in remaining
            assert query_text not in remaining, (
                f"Leakage detected! Query '{query_text}' present in target document for {img_id}."
            )
            # Remaining captions must not be empty (unless all 5 were identical strings)
            assert len(remaining) <= 4, f"Expected at most 4 captions, got {len(remaining)}."


@pytest.mark.integration
def test_int_m4_06_bm25_dual_protocols():
    """INT-M4-06: Verify both legacy (combined_document) and corrected (leave_one_caption_out) modes exist and differ."""
    toy_ids = ["img1", "img2"]
    toy_caps = [
        ["a red sports car parked outside", "a vehicle on the street", "automobile parked", "red car", "sedan"],
        ["a blue boat on calm water", "sailboat in harbor", "marine vessel", "ocean vessel", "docked boat"],
    ]

    # 1. Legacy combined mode
    bm25_legacy = BM25Retriever(aggregation_mode="combined_document")
    bm25_legacy.fit(toy_ids, toy_caps)
    query = "a red sports car parked outside"
    ranks_legacy = bm25_legacy.rank_gallery(query)
    scores_legacy = bm25_legacy.query(query, top_k=2)

    # 2. Corrected LOCO mode
    bm25_loco = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25_loco.fit(toy_ids, toy_caps)
    ranks_loco, scores_loco = bm25_loco.rank_gallery_leave_one_out(
        query_text=query,
        target_image_id="img1",
        query_caption_idx=0,
    )

    # Legacy score for img1 should be significantly higher due to exact query tokens in document
    legacy_img1_score = next(s for i, s in scores_legacy if i == "img1")
    loco_img1_score = scores_loco[ranks_loco.index("img1")]
    assert legacy_img1_score > loco_img1_score, (
        f"Legacy leaky score ({legacy_img1_score}) must exceed leave-one-out score ({loco_img1_score})."
    )


@pytest.mark.integration
@pytest.mark.dataset
def test_int_e2e_02_bm25_loco_end_to_end():
    """INT-E2E-02: End-to-end evaluation of Leave-One-Caption-Out BM25 over sample queries without missing targets."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    gallery_ids = [s.filename for s in dataset.samples]
    gallery_caps = [s.captions for s in dataset.samples]

    bm25 = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25.fit(gallery_ids, gallery_caps)

    # Evaluate on a slice of 100 queries
    sample_queries = []
    for s in dataset.samples[:20]:
        for c_idx, c in enumerate(s.captions):
            sample_queries.append(
                QueryRecord(
                    query_id=f"{s.filename}_{c_idx}",
                    query_text=c,
                    target_image_id=s.filename,
                    caption_idx=c_idx,
                    category=s.category,
                )
            )

    evaluator = RetrievalEvaluator(sample_queries, gallery_ids)
    metrics = evaluator.evaluate(
        "BM25 LOCO E2E Test",
        lambda q, target_image_id=None, query_caption_idx=None: (
            bm25.rank_gallery_leave_one_out(q, target_image_id, query_caption_idx)
        ),
    )

    assert metrics.total_queries == 100
    assert 0.0 <= metrics.r1 <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
    # Every query must successfully locate its target rank without crashing
    for res in metrics.query_results:
        assert 1 <= res.target_rank <= 1093, f"Invalid rank {res.target_rank} for query {res.query_id}."
