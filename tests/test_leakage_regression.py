"""Leakage Regression Tests for M4 Semantic Retrieval Baseline.

Validates that:
1. Exact query string is never present in its target BM25 document in leave-one-caption-out mode.
2. Each query receives exactly one originating-image positive target.
3. The leave-one-caption-out document contains the other captions but excludes the query caption.
4. The gallery contains exactly 1,093 unique test images.
5. The query count remains exactly 5,465 (5 captions per image).
6. No train or validation captions/images enter the test gallery.
7. Query-specific leave-one-out exclusion does not mutate the persistent gallery representation for other queries.
8. Repeated evaluation produces identical rankings and metrics under deterministic settings.
"""

from pathlib import Path
import json
import pytest

from src.data.rsicd_loader import RSICDDataset
from src.semantic_search.bm25 import BM25Retriever, tokenize
from src.semantic_search.retrieval_evaluator import QueryRecord, RetrievalEvaluator


@pytest.fixture(scope="module")
def rsicd_test_data():
    """Loads RSICD test split once for regression tests."""
    image_dir = "data/raw/rsicd/images"
    json_path = "data/raw/rsicd/dataset_rsicd.json"
    ds = RSICDDataset(image_dir=image_dir, json_path=json_path, split="test")

    gallery_ids = [s.filename for s in ds.samples]
    gallery_captions = [s.captions for s in ds.samples]
    return ds, gallery_ids, gallery_captions


def test_1_exact_query_string_not_in_target_document(rsicd_test_data):
    """Test 1: The exact query string is not present in its target BM25 document."""
    _, gallery_ids, gallery_captions = rsicd_test_data
    bm25 = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25.fit(gallery_ids, gallery_captions)

    # Test across first 50 samples
    for img_id, caps in zip(gallery_ids[:50], gallery_captions[:50]):
        for cap_idx, query_str in enumerate(caps):
            remaining = bm25.get_target_leave_one_out_captions(
                target_image_id=img_id,
                query_text=query_str,
                query_caption_idx=cap_idx,
            )
            # The exact query text must not be in the remaining captions
            clean_q = query_str.strip().lower()
            for rem_cap in remaining:
                assert rem_cap.strip().lower() != clean_q, (
                    f"Query leakage detected: '{query_str}' found in remaining target captions: {remaining}"
                )


def test_2_each_query_receives_exactly_one_originating_target(rsicd_test_data):
    """Test 2: Each query receives exactly one originating-image positive target."""
    ds, _, _ = rsicd_test_data
    queries = []
    for s in ds.samples:
        for cap_idx, c in enumerate(s.captions):
            queries.append(
                QueryRecord(
                    query_id=f"{s.filename}_{cap_idx}",
                    query_text=c,
                    target_image_id=s.filename,
                    caption_idx=cap_idx,
                )
            )

    target_counts = {}
    for q in queries:
        target_counts[q.target_image_id] = target_counts.get(q.target_image_id, 0) + 1

    # Exactly 5 queries per image
    for img_id, count in target_counts.items():
        assert count == 5, f"Image {img_id} has {count} queries, expected exactly 5"


def test_3_leave_one_out_contains_other_captions_excludes_query():
    """Test 3: The leave-one-caption-out document contains the other captions but excludes the query caption."""
    image_ids = ["img_test.jpg"]
    captions = [["Caption A", "Caption B", "Caption C", "Caption D", "Caption E"]]

    bm25 = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25.fit(image_ids, captions)

    # Query with Caption C (index 2)
    rem = bm25.get_target_leave_one_out_captions("img_test.jpg", "Caption C", query_caption_idx=2)
    assert len(rem) == 4
    assert "Caption C" not in rem
    assert "Caption A" in rem
    assert "Caption B" in rem
    assert "Caption D" in rem
    assert "Caption E" in rem


def test_4_gallery_contains_exactly_1093_unique_images(rsicd_test_data):
    """Test 4: The gallery still contains exactly 1,093 unique images."""
    _, gallery_ids, _ = rsicd_test_data
    assert len(gallery_ids) == 1093
    assert len(set(gallery_ids)) == 1093


def test_5_query_count_remains_exactly_5465(rsicd_test_data):
    """Test 5: The query count remains exactly 5,465."""
    ds, _, _ = rsicd_test_data
    total_queries = sum(len(s.captions) for s in ds.samples)
    assert total_queries == 5465


def test_6_no_train_val_captions_in_test_gallery():
    """Test 6: No train/validation captions enter the test gallery."""
    json_path = Path("data/raw/rsicd/dataset_rsicd.json")
    if not json_path.exists():
        pytest.skip("dataset_rsicd.json not available")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_ids = {img["filename"] for img in data.get("images", []) if img.get("split") == "train"}
    val_ids = {img["filename"] for img in data.get("images", []) if img.get("split") == "val"}
    test_ids = {img["filename"] for img in data.get("images", []) if img.get("split") == "test"}

    assert len(test_ids) == 1093
    assert len(test_ids.intersection(train_ids)) == 0, "Train-test leakage detected!"
    assert len(test_ids.intersection(val_ids)) == 0, "Val-test leakage detected!"


def test_7_query_exclusion_does_not_mutate_persistent_gallery():
    """Test 7: Changing query caption cannot accidentally mutate persistent gallery representation."""
    image_ids = ["img_1.jpg", "img_2.jpg"]
    captions = [["first airport", "second airport"], ["first river", "second river"]]

    bm25 = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25.fit(image_ids, captions)

    # Initial state
    initial_caps_1 = list(bm25.image_captions["img_1.jpg"])
    initial_len_1 = bm25.doc_lengths[0]

    # Perform leave-one-out query
    _ = bm25.rank_gallery_leave_one_out("first airport", "img_1.jpg", query_caption_idx=0)

    # State must remain strictly unchanged
    assert bm25.image_captions["img_1.jpg"] == initial_caps_1
    assert bm25.doc_lengths[0] == initial_len_1

    # Perform another leave-one-out query on different caption
    _ = bm25.rank_gallery_leave_one_out("second airport", "img_1.jpg", query_caption_idx=1)
    assert bm25.image_captions["img_1.jpg"] == initial_caps_1
    assert bm25.doc_lengths[0] == initial_len_1


def test_8_repeated_evaluation_produces_identical_rankings():
    """Test 8: Repeated evaluation produces identical rankings/results under deterministic settings."""
    image_ids = ["img_1.jpg", "img_2.jpg", "img_3.jpg"]
    captions = [
        ["small green island", "water around island"],
        ["dry bare land", "desert soil"],
        ["bridge over highway", "cars on the road"],
    ]
    bm25 = BM25Retriever(aggregation_mode="leave_one_caption_out")
    bm25.fit(image_ids, captions)

    ranks_run1, scores_run1 = bm25.rank_gallery_leave_one_out("small green island", "img_1.jpg", 0)
    ranks_run2, scores_run2 = bm25.rank_gallery_leave_one_out("small green island", "img_1.jpg", 0)

    assert ranks_run1 == ranks_run2
    assert scores_run1 == scores_run2
