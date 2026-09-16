"""Domain Prompt Ensembling Ablation Module (Method A3).

Implements context-guided prompt expansion and embedding ensembling for zero-shot satellite TIR.
Uses a frozen set of literature-grounded remote sensing prompt templates.
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np

# Frozen, immutable prompt templates for Method A3 ablation
FROZEN_REMOTE_SENSING_TEMPLATES: List[str] = [
    "{query}",
    "a satellite image of {query}",
    "a remote sensing image of {query}",
    "an overhead image showing {query}",
    "an aerial photograph of {query}",
]


class PromptEnsembler:
    """Expands queries using fixed prompt templates and aggregates embeddings."""

    def __init__(
        self,
        templates: Sequence[str] = FROZEN_REMOTE_SENSING_TEMPLATES,
    ) -> None:
        """Initializes PromptEnsembler with a frozen template set.

        Args:
            templates: Sequence of format string templates containing '{query}'.
        """
        self.templates = list(templates)

    def generate_prompts(self, query: str) -> List[str]:
        """Generates the prompt ensemble list for a given raw query string.

        Args:
            query: Input natural-language query.

        Returns:
            List of formatted prompt strings.
        """
        # Clean query of trailing punctuation and whitespace
        clean_q = query.strip().rstrip(".!?")
        return [template.format(query=clean_q) for template in self.templates]

    def encode_ensemble(
        self,
        query: str,
        encode_fn,
    ) -> np.ndarray:
        """Encodes an ensembled query via prompt expansion and L2-normalized averaging.

        Formulation:
            u_m = L2_norm(E_text(template_m(query)))
            u_mean = (1 / M) * sum_{m=1}^M u_m
            u_ens = u_mean / ||u_mean||_2

        Args:
            query: Input query text.
            encode_fn: Callable taking List[str] and returning (M, D) float32 numpy array.

        Returns:
            Float32 array of shape (1, D) with unit L2 norm.
        """
        prompts = self.generate_prompts(query)
        prompt_embeddings = encode_fn(prompts)  # Shape: (M, D)

        # Average across prompt embeddings
        mean_embedding = np.mean(prompt_embeddings, axis=0, keepdims=True)

        # Re-normalize to unit sphere
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            ens_embedding = mean_embedding / norm
        else:
            ens_embedding = mean_embedding

        return ens_embedding.astype(np.float32)
