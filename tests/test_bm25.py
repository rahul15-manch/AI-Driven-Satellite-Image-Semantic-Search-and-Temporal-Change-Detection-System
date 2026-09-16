"""Unit tests for BM25 Lexical Retrieval Baseline."""

import pytest
from src.semantic_search.bm25 import BM25Retriever, tokenize


def test_tokenize():
    text = "An airport with several airplanes parked near the runway!"
    tokens = tokenize(text)
    assert "airport" in tokens
    assert "airplanes" in tokens
    assert "runway" in tokens
    assert "!" not in tokens
    # Ensure lowercase
    for t in tokens:
        assert t == t.lower()


def test_bm25_fit_and_query():
    retriever = BM25Retriever(k1=1.5, b=0.75)
    image_ids = ["img_airport.jpg", "img_river.jpg", "img_forest.jpg"]
    captions = [
        ["An airport with many planes on the tarmac.", "Planes at an airport runway."],
        ["A winding river flows through green valleys.", "Water body in countryside."],
        ["Dense green forest with tall trees.", "Woodland area from above."],
    ]

    retriever.fit(image_ids, captions)
    assert retriever.num_docs == 3

    # Query matching airport
    results = retriever.query("airport planes", top_k=3)
    assert len(results) == 3
    assert results[0][0] == "img_airport.jpg"
    assert results[0][1] > results[1][1]

    # Query matching river
    results_river = retriever.query("winding river", top_k=3)
    assert results_river[0][0] == "img_river.jpg"


def test_bm25_aggregation_modes():
    image_ids = ["img_1.jpg", "img_2.jpg"]
    captions = [
        ["commercial airport", "terminal building"],
        ["residential houses", "green gardens"],
    ]

    # Combined document mode
    bm25_comb = BM25Retriever(aggregation_mode="combined_document")
    bm25_comb.fit(image_ids, captions)
    ranks_comb = bm25_comb.rank_gallery("terminal")
    assert ranks_comb[0] == "img_1.jpg"

    # Max caption mode
    bm25_max = BM25Retriever(aggregation_mode="max_caption")
    bm25_max.fit(image_ids, captions)
    ranks_max = bm25_max.rank_gallery("terminal")
    assert ranks_max[0] == "img_1.jpg"


def test_bm25_leave_one_caption_out():
    image_ids = ["img_1.jpg", "img_2.jpg"]
    captions = [
        ["commercial airport runway", "airplane parked near terminal"],
        ["residential houses", "green suburban lawns"],
    ]

    bm25 = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25.fit(image_ids, captions)

    # When querying "airplane parked near terminal" with target img_1.jpg:
    # Target img_1.jpg only has "commercial airport runway" in its document!
    # Because "airplane" does not appear in "commercial airport runway", score of img_1.jpg drops to 0!
    ranks, scores = bm25.rank_gallery_leave_one_out(
        "airplane parked near terminal",
        target_image_id="img_1.jpg",
        query_caption_idx=1,
    )
    # Score for img_1.jpg should be 0.0 because the query terms do not appear in the other caption
    target_idx = ranks.index("img_1.jpg")
    assert scores[target_idx] == 0.0

    # But if query is "airport" (which appears in the remaining caption), img_1.jpg gets a positive score
    ranks_airport, scores_airport = bm25.rank_gallery_leave_one_out(
        "airport",
        target_image_id="img_1.jpg",
        query_caption_idx=1,
    )
    target_idx_airport = ranks_airport.index("img_1.jpg")
    assert scores_airport[target_idx_airport] > 0.0


def test_bm25_fit_metadata():
    image_ids = ["img_1.jpg", "img_2.jpg"]
    categories = ["airport", "residential"]

    bm25 = BM25Retriever(aggregation_mode="category_metadata")
    bm25.fit_metadata(image_ids, categories)
    assert bm25.aggregation_mode == "category_metadata"

    ranks = bm25.rank_gallery("airport runway")
    assert ranks[0] == "img_1.jpg"


def test_bm25_dimension_mismatch():
    retriever = BM25Retriever()
    with pytest.raises(ValueError):
        retriever.fit(["img1.jpg"], [["cap1"], ["cap2"]])
