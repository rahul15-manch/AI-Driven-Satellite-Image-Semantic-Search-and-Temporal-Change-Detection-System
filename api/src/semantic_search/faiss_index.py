"""FAISS Exact Inner-Product Index (IndexFlatIP) for Semantic Vector Retrieval.

Implements exact cosine similarity search over L2-normalized image embeddings.
Strictly CPU-only execution without approximation distortion.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None  # Handled gracefully for unit testing / mocking


class FAISSFlatIPIndex:
    """FAISS exact inner product (IndexFlatIP) search structure."""

    def __init__(self, dimension: int = 512) -> None:
        """Initializes FAISSFlatIPIndex.

        Args:
            dimension: Latent embedding dimension (512 for CLIP ViT-B/32).
        """
        self.dimension = dimension
        self.image_ids: List[str] = []
        self.index = None

        if faiss is not None:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            logger.warning("faiss package not found. Using NumPy fallback for inner product search.")

    def add(self, embeddings: np.ndarray, image_ids: List[str]) -> None:
        """Adds L2-normalized embeddings to the index.

        Args:
            embeddings: Float32 array of shape (N, dimension).
            image_ids: Parallel list of N image identifier strings.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (N, {self.dimension}), got {embeddings.shape}"
            )
        if len(image_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: len(image_ids)={len(image_ids)} != embeddings.shape[0]={embeddings.shape[0]}"
            )

        # Enforce float32
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Validate L2 normalization
        norms = np.linalg.norm(embeddings_f32, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            logger.warning("Embeddings are not strictly L2 normalized. Re-normalizing in-place.")
            embeddings_f32 = embeddings_f32 / (norms[:, np.newaxis] + 1e-9)

        self.image_ids = list(image_ids)

        if self.index is not None:
            self.index.reset()
            self.index.add(embeddings_f32)
        self._raw_embeddings = embeddings_f32

    @property
    def ntotal(self) -> int:
        """Returns the number of indexed vectors."""
        if self.index is not None:
            return self.index.ntotal
        return len(self.image_ids)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """Performs exact nearest-neighbor search for a query embedding.

        Args:
            query_embedding: Array of shape (1, dimension) or (dimension,).
            top_k: Number of top results to retrieve.

        Returns:
            Tuple of (retrieved_image_ids, similarity_scores).
        """
        q = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, self.dimension)
        q_norm = np.linalg.norm(q)
        if not np.isclose(q_norm, 1.0, atol=1e-3) and q_norm > 0:
            q = q / q_norm

        top_k = min(top_k, self.ntotal)

        if self.index is not None:
            distances, indices = self.index.search(q, top_k)
            retrieved_ids = [self.image_ids[idx] for idx in indices[0]]
            return retrieved_ids, distances[0]
        else:
            # Fallback inner product search
            scores = np.dot(self._raw_embeddings, q.T).flatten()
            sorted_indices = np.argsort(-scores)[:top_k]
            retrieved_ids = [self.image_ids[idx] for idx in sorted_indices]
            return retrieved_ids, scores[sorted_indices]

    def rank_gallery(self, query_embedding: np.ndarray) -> List[str]:
        """Returns full gallery ranking ordered from highest to lowest similarity.

        Args:
            query_embedding: Array of shape (1, dimension) or (dimension,).

        Returns:
            List of all indexed image IDs sorted by descending cosine similarity.
        """
        q = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, self.dimension)
        q_norm = np.linalg.norm(q)
        if not np.isclose(q_norm, 1.0, atol=1e-3) and q_norm > 0:
            q = q / q_norm

        if self.index is not None:
            _, indices = self.index.search(q, self.ntotal)
            return [self.image_ids[idx] for idx in indices[0]]
        else:
            scores = np.dot(self._raw_embeddings, q.T).flatten()
            sorted_indices = np.argsort(-scores)
            return [self.image_ids[idx] for idx in sorted_indices]

    def get_memory_bytes(self) -> int:
        """Returns the approximate memory footprint of the index in bytes."""
        return self.ntotal * self.dimension * 4
