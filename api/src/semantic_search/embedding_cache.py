"""Persistent Embedding Cache for Precomputed Vision-Language Representations.

Manages serialization, deserialization, and integrity validation for image embeddings.
Prevents redundant recomputation while ensuring strict compatibility verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def compute_ids_hash(image_ids: List[str]) -> str:
    """Computes a deterministic SHA-256 hash across an ordered list of image IDs."""
    hasher = hashlib.sha256()
    for img_id in image_ids:
        hasher.update(img_id.encode("utf-8"))
    return hasher.hexdigest()


class EmbeddingCache:
    """Handles disk caching and metadata verification of gallery embeddings."""

    def __init__(self, cache_dir: Path | str) -> None:
        """Initializes EmbeddingCache.

        Args:
            cache_dir: Directory path for caching embeddings and metadata.
        """
        self.cache_dir = Path(cache_dir)
        self.embeddings_path = self.cache_dir / "image_embeddings.npy"
        self.metadata_path = self.cache_dir / "metadata.json"

    def is_valid(
        self,
        expected_model: str,
        expected_dataset: str,
        expected_split: str,
        expected_count: int,
        expected_dim: int,
        expected_image_ids: Optional[List[str]] = None,
    ) -> bool:
        """Validates whether the cached embeddings match the expected experimental configuration."""
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            return False

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("model") != expected_model:
                logger.info(f"Cache miss: model mismatch ({meta.get('model')} != {expected_model})")
                return False
            if meta.get("dataset") != expected_dataset:
                logger.info(f"Cache miss: dataset mismatch ({meta.get('dataset')} != {expected_dataset})")
                return False
            if meta.get("split") != expected_split:
                logger.info(f"Cache miss: split mismatch ({meta.get('split')} != {expected_split})")
                return False
            if meta.get("count") != expected_count:
                logger.info(f"Cache miss: count mismatch ({meta.get('count')} != {expected_count})")
                return False
            if meta.get("embedding_dim") != expected_dim:
                logger.info(f"Cache miss: dimension mismatch ({meta.get('embedding_dim')} != {expected_dim})")
                return False

            if expected_image_ids is not None:
                expected_hash = compute_ids_hash(expected_image_ids)
                if meta.get("image_ids_hash") != expected_hash:
                    logger.info("Cache miss: image IDs sequence mismatch")
                    return False

            return True
        except Exception as e:
            logger.warning(f"Failed to validate cache metadata: {e}")
            return False

    def save(
        self,
        embeddings: np.ndarray,
        image_ids: List[str],
        model: str,
        dataset: str,
        split: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serializes embeddings and companion metadata to disk.

        Args:
            embeddings: Float32 array of shape (N, D).
            image_ids: List of N image identifiers.
            model: Model identifier string (e.g., 'openai/clip-vit-base-patch32').
            dataset: Dataset identifier (e.g., 'RSICD').
            split: Split identifier (e.g., 'test').
            extra_meta: Optional additional metadata dictionary.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        np.save(self.embeddings_path, embeddings_f32)

        metadata: Dict[str, Any] = {
            "model": model,
            "dataset": dataset,
            "split": split,
            "count": int(embeddings_f32.shape[0]),
            "embedding_dim": int(embeddings_f32.shape[1]),
            "image_ids_hash": compute_ids_hash(image_ids),
            "image_ids": list(image_ids),
        }
        if extra_meta:
            metadata.update(extra_meta)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved {embeddings_f32.shape[0]} embeddings to cache: {self.embeddings_path}"
        )

    def load(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Loads cached embeddings, image IDs, and metadata from disk.

        Returns:
            Tuple of (embeddings_matrix, image_ids, metadata_dict).

        Raises:
            FileNotFoundError: If cache files are missing.
        """
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f"Cache not found in {self.cache_dir}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        embeddings = np.load(self.embeddings_path)
        image_ids = metadata.get("image_ids", [])

        logger.info(
            f"Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]} dim) from {self.embeddings_path}"
        )
        return embeddings, image_ids, metadata
