"""Integration Test: CLIP Dual-Encoder and FAISS Search Pipeline (M4 Integration).

Covers:
- INT-M4-01: Dataset Loader -> CLIP Image Encoder (512D, L2-normalized)
- INT-M4-02: Caption Loader -> CLIP Text Encoder (512D, L2-normalized)
- INT-M4-03: Image Embeddings -> FAISS IndexFlatIP (dim=512, exact inner product)
- INT-M4-04: Query Embedding -> FAISS Search
- INT-M4-05: Retrieval Evaluator Logic
- INT-M4-07: Persistent Embedding Cache Validation & Integrity Checking
- INT-E2E-01: Complete Zero-Shot CLIP Pipeline End-to-End
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.data.rsicd_loader import RSICDDataset
from src.semantic_search.clip_model import CLIPRetriever
from src.semantic_search.embedding_cache import EmbeddingCache
from src.semantic_search.faiss_index import FAISSFlatIPIndex
from src.semantic_search.retrieval_evaluator import QueryRecord, RetrievalEvaluator


@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.cpu
def test_int_m4_01_image_encoder_l2_normalized():
    """INT-M4-01: Image encoder outputs 512-dim float32 L2-normalized embeddings on CPU."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    # Test on first 3 images
    sample_paths = [dataset.samples[i].image_path for i in range(3)]
    clip = CLIPRetriever(model_name="openai/clip-vit-base-patch32", device="cpu", batch_size=3)

    embs = clip.encode_images(sample_paths, batch_size=3)
    assert embs.shape == (3, 512), f"Expected shape (3, 512), got {embs.shape}."
    assert embs.dtype == np.float32, f"Expected dtype float32, got {embs.dtype}."

    # L2-normalization check (norm == 1.0 within numerical precision)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Embeddings not L2-normalized: norms={norms}"


@pytest.mark.integration
@pytest.mark.cpu
def test_int_m4_02_text_encoder_l2_normalized():
    """INT-M4-02: Text encoder outputs 512-dim float32 L2-normalized embeddings on CPU."""
    queries = ["an airport with several airplanes", "a dense residential area", "a bridge over a river"]
    clip = CLIPRetriever(model_name="openai/clip-vit-base-patch32", device="cpu", batch_size=3)

    embs = clip.encode_text(queries, batch_size=3)
    assert embs.shape == (3, 512), f"Expected shape (3, 512), got {embs.shape}."
    assert embs.dtype == np.float32, f"Expected dtype float32, got {embs.dtype}."

    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Text embeddings not L2-normalized: norms={norms}"


@pytest.mark.integration
def test_int_m4_03_and_04_faiss_index_and_search():
    """INT-M4-03 & INT-M4-04: FAISS IndexFlatIP indexing and inner-product cosine search."""
    dim = 512
    num_vectors = 100
    rng = np.random.RandomState(42)

    # Generate synthetic unit vectors
    raw = rng.randn(num_vectors, dim).astype(np.float32)
    vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    image_ids = [f"img_{i}.jpg" for i in range(num_vectors)]

    index = FAISSFlatIPIndex(dimension=dim)
    index.add(vectors, image_ids)
    assert index.ntotal == num_vectors
    assert index.dimension == dim

    # Search with the first vector (must return itself at rank 1 with score ~ 1.0)
    query_vec = vectors[0:1]
    ranked_ids, scores = index.search(query_vec, top_k=5)

    assert ranked_ids[0] == "img_0.jpg"
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    # Cosine similarities must be in [-1, 1]
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)


@pytest.mark.integration
def test_int_m4_07_embedding_cache_integrity(tmp_path):
    """INT-M4-07: EmbeddingCache validates count, dim, model, split, and ordered ID hash."""
    cache = EmbeddingCache(tmp_path)
    embs = np.random.randn(10, 512).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    ids = [f"img_{i}.jpg" for i in range(10)]

    cache.save(embs, ids, model="openai/clip-vit-base-patch32", dataset="RSICD", split="test")

    # Valid check
    assert cache.is_valid(
        expected_model="openai/clip-vit-base-patch32",
        expected_dataset="RSICD",
        expected_split="test",
        expected_count=10,
        expected_dim=512,
        expected_image_ids=ids,
    )

    # Rejection on mismatched image count
    assert not cache.is_valid(
        expected_model="openai/clip-vit-base-patch32",
        expected_dataset="RSICD",
        expected_split="test",
        expected_count=11,
        expected_dim=512,
        expected_image_ids=ids,
    )

    # Rejection on altered image ID order (SHA-256 hash mismatch)
    permuted_ids = list(reversed(ids))
    assert not cache.is_valid(
        expected_model="openai/clip-vit-base-patch32",
        expected_dataset="RSICD",
        expected_split="test",
        expected_count=10,
        expected_dim=512,
        expected_image_ids=permuted_ids,
    )


@pytest.mark.integration
@pytest.mark.dataset
def test_int_e2e_01_complete_clip_zero_shot_pipeline():
    """INT-E2E-01: End-to-end evaluation of CLIP zero-shot pipeline on verified cache."""
    cache_path = Path("experiments/cache/clip/image_embeddings.npy")
    assert cache_path.exists(), "experiments/cache/clip/image_embeddings.npy must exist."

    embs = np.load(cache_path)
    assert embs.shape == (1093, 512), f"Expected (1093, 512), got {embs.shape}."

    # Load test split
    dataset = RSICDDataset(image_dir="data/raw/rsicd/images", json_path="data/raw/rsicd/dataset_rsicd.json", split="test")
    gallery_ids = [s.filename for s in dataset.samples]

    faiss_index = FAISSFlatIPIndex(dimension=512)
    faiss_index.add(embs, gallery_ids)
    assert faiss_index.ntotal == 1093

    # Run on a sample of 20 queries
    sample_queries = []
    for s in dataset.samples[:4]:
        for c_idx, c in enumerate(s.captions):
            sample_queries.append(QueryRecord(query_id=f"{s.filename}_{c_idx}", query_text=c, target_image_id=s.filename, caption_idx=c_idx, category=s.category))

    clip = CLIPRetriever(model_name="openai/clip-vit-base-patch32", device="cpu", batch_size=20)
    q_texts = [q.query_text for q in sample_queries]
    q_embs = clip.encode_text(q_texts, batch_size=20)

    evaluator = RetrievalEvaluator(sample_queries, gallery_ids)
    metrics = evaluator.evaluate(
        "CLIP E2E Sample",
        lambda q: (faiss_index.search(clip.encode_single_query(q), top_k=1093)[0], None)
    )

    assert metrics.total_queries == 20
    assert 0.0 <= metrics.r1 <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
