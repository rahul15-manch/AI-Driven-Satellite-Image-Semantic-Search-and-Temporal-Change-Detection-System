"""Unit tests for persistent embedding cache."""

import numpy as np
import pytest
from src.semantic_search.embedding_cache import EmbeddingCache, compute_ids_hash


def test_embedding_cache_save_load_valid(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    embeddings = np.random.randn(5, 16).astype(np.float32)
    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    image_ids = [f"img_{i}.jpg" for i in range(5)]

    # Before saving, should be invalid
    assert not cache.is_valid("model_test", "RSICD", "test", 5, 16, image_ids)

    # Save
    cache.save(embeddings, image_ids, "model_test", "RSICD", "test")

    # Now should be valid
    assert cache.is_valid("model_test", "RSICD", "test", 5, 16, image_ids)

    # Incompatible parameters should return False
    assert not cache.is_valid("different_model", "RSICD", "test", 5, 16, image_ids)
    assert not cache.is_valid("model_test", "RSICD", "test", 10, 16, image_ids)
    assert not cache.is_valid("model_test", "RSICD", "test", 5, 32, image_ids)

    # Load and verify contents
    loaded_emb, loaded_ids, meta = cache.load()
    assert np.allclose(embeddings, loaded_emb)
    assert loaded_ids == image_ids
    assert meta["model"] == "model_test"


def test_compute_ids_hash_consistency():
    ids1 = ["a.jpg", "b.jpg", "c.jpg"]
    ids2 = ["a.jpg", "b.jpg", "c.jpg"]
    ids3 = ["b.jpg", "a.jpg", "c.jpg"]

    assert compute_ids_hash(ids1) == compute_ids_hash(ids2)
    assert compute_ids_hash(ids1) != compute_ids_hash(ids3)
