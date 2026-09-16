"""Unit tests for RetrievalEvaluator under Caption-to-Own-Image protocol."""

import pytest
from src.semantic_search.retrieval_evaluator import (
    QueryRecord,
    RetrievalEvaluator,
)


def test_retrieval_evaluator_synthetic_ranks():
    gallery_ids = ["img_1.jpg", "img_2.jpg", "img_3.jpg", "img_4.jpg", "img_5.jpg", "img_6.jpg", "img_7.jpg", "img_8.jpg", "img_9.jpg", "img_10.jpg", "img_11.jpg"]

    # 3 synthetic queries:
    # Q1: Target is img_1 (will rank 1st -> R@1=1, R@5=1, R@10=1, RR=1.0)
    # Q2: Target is img_5 (will rank 5th -> R@1=0, R@5=1, R@10=1, RR=0.2)
    # Q3: Target is img_11 (will rank 11th -> R@1=0, R@5=0, R@10=0, RR=1/11)
    queries = [
        QueryRecord(query_id="q1", query_text="text1", target_image_id="img_1.jpg", category="airport"),
        QueryRecord(query_id="q2", query_text="text2", target_image_id="img_5.jpg", category="airport"),
        QueryRecord(query_id="q3", query_text="text3", target_image_id="img_11.jpg", category="river"),
    ]

    evaluator = RetrievalEvaluator(queries=queries, gallery_image_ids=gallery_ids)

    # Fixed ranking function returning the gallery in order
    def fixed_rank_fn(query_text: str):
        return gallery_ids, [1.0 / (i + 1) for i in range(len(gallery_ids))]

    metrics = evaluator.evaluate("Synthetic Test", fixed_rank_fn)

    # R@1: 1 out of 3 = 1/3
    assert pytest.approx(metrics.r1, 0.001) == 1.0 / 3.0
    # R@5: 2 out of 3 = 2/3
    assert pytest.approx(metrics.r5, 0.001) == 2.0 / 3.0
    # R@10: 2 out of 3 = 2/3
    assert pytest.approx(metrics.r10, 0.001) == 2.0 / 3.0
    # MRR: (1.0 + 0.2 + (1.0/11)) / 3
    expected_mrr = (1.0 + 0.2 + (1.0 / 11.0)) / 3.0
    assert pytest.approx(metrics.mrr, 0.001) == expected_mrr

    # Category breakdown check
    assert "airport" in metrics.category_breakdown
    assert metrics.category_breakdown["airport"]["count"] == 2
    assert metrics.category_breakdown["airport"]["R@1"] == 0.5


def test_target_missing_from_gallery():
    queries = [QueryRecord(query_id="q1", query_text="text1", target_image_id="missing.jpg")]
    with pytest.raises(ValueError):
        RetrievalEvaluator(queries=queries, gallery_image_ids=["img1.jpg"])
