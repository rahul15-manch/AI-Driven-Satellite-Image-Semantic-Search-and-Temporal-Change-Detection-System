"""Unit tests for FAISS IndexFlatIP exact vector retrieval."""

import numpy as np
import pytest
from src.semantic_search.faiss_index import FAISSFlatIPIndex


def test_faiss_index_lifecycle():
    dimension = 4
    index = FAISSFlatIPIndex(dimension=dimension)

    # 3 dummy L2-normalized vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = np.vstack([v1, v2, v3])
    ids = ["img1.jpg", "img2.jpg", "img3.jpg"]

    index.add(embeddings, ids)
    assert index.ntotal == 3

    # Query identical to v2
    retrieved_ids, scores = index.search(v2, top_k=2)
    assert len(retrieved_ids) == 2
    assert retrieved_ids[0] == "img2.jpg"
    assert np.isclose(scores[0], 1.0, atol=1e-5)

    # Full gallery ranking
    ranked = index.rank_gallery(v3)
    assert ranked[0] == "img3.jpg"
    assert len(ranked) == 3


def test_faiss_dimension_error():
    index = FAISSFlatIPIndex(dimension=512)
    wrong_emb = np.zeros((2, 128), dtype=np.float32)
    with pytest.raises(ValueError):
        index.add(wrong_emb, ["a", "b"])
