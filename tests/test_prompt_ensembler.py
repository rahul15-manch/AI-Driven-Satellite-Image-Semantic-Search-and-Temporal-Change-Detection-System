"""Unit tests for PromptEnsembler."""

import numpy as np
import pytest
from src.semantic_search.prompt_ensembler import (
    FROZEN_REMOTE_SENSING_TEMPLATES,
    PromptEnsembler,
)


def test_prompt_generation():
    ensembler = PromptEnsembler(templates=FROZEN_REMOTE_SENSING_TEMPLATES)
    prompts = ensembler.generate_prompts("airplanes parked at airport.")

    assert len(prompts) == len(FROZEN_REMOTE_SENSING_TEMPLATES)
    assert prompts[0] == "airplanes parked at airport"
    assert "a satellite image of airplanes parked at airport" in prompts
    assert "a remote sensing image of airplanes parked at airport" in prompts


def test_encode_ensemble():
    ensembler = PromptEnsembler(templates=["{query}", "photo of {query}"])

    # Mock encoder function returning unit vectors
    def mock_encode(prompt_list):
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    ens_vec = ensembler.encode_ensemble("test query", mock_encode)

    assert ens_vec.shape == (1, 2)
    # L2 norm must be 1.0
    norm = np.linalg.norm(ens_vec)
    assert np.isclose(norm, 1.0, atol=1e-5)
    # Vectors (1,0) and (0,1) average to (0.5, 0.5), normalized to (1/sqrt(2), 1/sqrt(2))
    assert np.isclose(ens_vec[0, 0], 1.0 / np.sqrt(2), atol=1e-4)
    assert np.isclose(ens_vec[0, 1], 1.0 / np.sqrt(2), atol=1e-4)
