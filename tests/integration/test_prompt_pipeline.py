"""Integration Test: Domain Prompt Ensembling Pipeline (M4 Integration).

Covers:
- INT-M4-08: Frozen Remote-Sensing Prompt Template Formatting & Ensembling
- INT-E2E-03: End-to-End Execution of Prompt Ensemble Retrieval
"""

from __future__ import annotations

import numpy as np
import pytest

from src.semantic_search.clip_model import CLIPRetriever
from src.semantic_search.faiss_index import FAISSFlatIPIndex
from src.semantic_search.prompt_ensembler import FROZEN_REMOTE_SENSING_TEMPLATES, PromptEnsembler
from src.semantic_search.retrieval_evaluator import QueryRecord, RetrievalEvaluator


@pytest.mark.integration
def test_int_m4_08_prompt_ensemble_templates_and_normalization():
    """INT-M4-08: Verify 5 frozen RS prompt templates and unit normalization."""
    ensembler = PromptEnsembler()
    assert len(ensembler.templates) == 5, f"Expected 5 templates, got {len(ensembler.templates)}."
    assert ensembler.templates == FROZEN_REMOTE_SENSING_TEMPLATES

    query = "a circular athletic field"
    prompts = ensembler.generate_prompts(query)
    assert len(prompts) == 5
    assert prompts[0] == "a circular athletic field"
    assert "satellite image" in prompts[1]
    assert "remote sensing" in prompts[2]
    assert "overhead image" in prompts[3]
    assert "aerial photograph" in prompts[4]

    # Test averaging logic on synthetic embeddings
    mock_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0] / np.sqrt(2),
        [0.0, 1.0, 1.0] / np.sqrt(2),
    ], dtype=np.float32)

    ens_vec = ensembler.encode_ensemble(query, encode_fn=lambda p_list: mock_embeddings)
    assert ens_vec.shape == (1, 3)
    norm = float(np.linalg.norm(ens_vec))
    assert np.isclose(norm, 1.0, atol=1e-5), f"Ensembled vector must have unit norm, got {norm}."


@pytest.mark.integration
@pytest.mark.cpu
def test_int_e2e_03_prompt_ensemble_retrieval_end_to_end():
    """INT-E2E-03: End-to-end prompt ensembling with CLIP and FAISS."""
    clip = CLIPRetriever(model_name="openai/clip-vit-base-patch32", device="cpu", batch_size=5)
    ensembler = PromptEnsembler()

    query = "an airport runway with planes"
    ens_vec = ensembler.encode_ensemble(
        query,
        encode_fn=lambda p_list: clip.encode_text(p_list, batch_size=5)
    )

    assert ens_vec.shape == (1, 512)
    assert np.isclose(float(np.linalg.norm(ens_vec)), 1.0, atol=1e-5)

    # Build toy FAISS index
    index = FAISSFlatIPIndex(dimension=512)
    # Add ens_vec itself and a random vector
    rng = np.random.RandomState(42)
    other_vec = rng.randn(1, 512).astype(np.float32)
    other_vec /= np.linalg.norm(other_vec, axis=1, keepdims=True)

    gallery_vecs = np.vstack([other_vec, ens_vec])
    index.add(gallery_vecs, ["other.jpg", "target.jpg"])

    ranked_ids, scores = index.search(ens_vec, top_k=2)
    assert ranked_ids[0] == "target.jpg", "Exact match must be retrieved at rank 1."
    assert np.isclose(scores[0], 1.0, atol=1e-5)
