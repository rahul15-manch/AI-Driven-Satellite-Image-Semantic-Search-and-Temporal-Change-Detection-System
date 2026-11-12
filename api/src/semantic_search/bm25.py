"""BM25 Lexical Retrieval Baseline (Method A1).

Implements standard Okapi BM25 scoring for cross-modal image retrieval on RSICD.
Supports:
1. Leave-One-Caption-Out BM25 (Mode A - Leakage-Controlled Primary Protocol):
   Excludes the query caption from its target gallery document to prevent query-in-document leakage.
2. Category / Metadata Lexical Baseline (Mode B - Auxiliary Metadata Protocol):
   Indexes only dataset category labels without human captions.
3. Combined-Document Multi-Caption Indexing (Legacy Diagnostic Protocol):
   Preserved for historical comparison.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words.

    Args:
        text: Input string.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class BM25Retriever:
    """Okapi BM25 index and retrieval engine.

    Formulation:
        BM25(D, Q) = sum_{t in Q} IDF(t) * (f(t, D) * (k1 + 1)) / (f(t, D) + k1 * (1 - b + b * (|D| / avgdl)))
        IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1.0)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        aggregation_mode: str = "leave_one_caption_out",
    ) -> None:
        """Initializes BM25Retriever.

        Args:
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
            aggregation_mode: Strategy for multi-caption images:
                - 'leave_one_caption_out': Primary leakage-controlled protocol (Mode A).
                - 'combined_document': Legacy protocol concatenating all captions (Mode A legacy).
                - 'max_caption': Evaluates captions independently, taking max score.
                - 'category_metadata': Auxiliary baseline indexing only category labels (Mode B).
        """
        self.k1 = k1
        self.b = b
        self.aggregation_mode = aggregation_mode

        self.image_ids: List[str] = []
        self.image_to_idx: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_term_freqs: List[Counter] = []
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.num_docs: int = 0
        self.idf_cache: Dict[str, float] = {}

        # Raw caption storage for leave-one-caption-out protocol
        self.image_captions: Dict[str, List[str]] = {}

        # Secondary index structures for 'max_caption' mode
        self.caption_to_image_id: List[str] = []
        self.caption_lengths: List[int] = []
        self.avg_caption_len: float = 0.0
        self.caption_term_freqs: List[Counter] = []
        self.caption_doc_freq: Dict[str, int] = defaultdict(int)
        self.num_captions: int = 0
        self.caption_idf_cache: Dict[str, float] = {}

    def fit(self, image_ids: Sequence[str], image_captions: Sequence[Sequence[str]]) -> None:
        """Builds the BM25 index from images and their associated caption lists.

        Args:
            image_ids: List of unique image identifier strings (e.g., 'airport_348.jpg').
            image_captions: Parallel sequence of caption lists for each image.
        """
        if len(image_ids) != len(image_captions):
            raise ValueError(
                f"Mismatch: len(image_ids)={len(image_ids)} != len(image_captions)={len(image_captions)}"
            )

        self.image_ids = list(image_ids)
        self.image_to_idx = {img_id: idx for idx, img_id in enumerate(self.image_ids)}
        self.num_docs = len(self.image_ids)

        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freq = defaultdict(int)
        self.image_captions = {}

        self.caption_to_image_id = []
        self.caption_lengths = []
        self.caption_term_freqs = []
        self.caption_doc_freq = defaultdict(int)

        total_doc_len = 0
        total_caption_len = 0

        for img_id, captions in zip(self.image_ids, image_captions):
            # Store immutable copy of captions for leave-one-caption-out
            self.image_captions[img_id] = list(captions)

            # Combined document representation: concatenate all captions for this image
            combined_text = " ".join(captions)
            doc_tokens = tokenize(combined_text)
            doc_len = len(doc_tokens)
            self.doc_lengths.append(doc_len)
            total_doc_len += doc_len

            term_freq = Counter(doc_tokens)
            self.doc_term_freqs.append(term_freq)
            for term in term_freq.keys():
                self.doc_freq[term] += 1

            # Individual caption representation for max_caption mode
            for cap in captions:
                cap_tokens = tokenize(cap)
                cap_len = len(cap_tokens)
                self.caption_to_image_id.append(img_id)
                self.caption_lengths.append(cap_len)
                total_caption_len += cap_len

                c_freq = Counter(cap_tokens)
                self.caption_term_freqs.append(c_freq)
                for term in c_freq.keys():
                    self.caption_doc_freq[term] += 1

        self.avg_doc_len = total_doc_len / self.num_docs if self.num_docs > 0 else 0.0
        self.num_captions = len(self.caption_to_image_id)
        self.avg_caption_len = (
            total_caption_len / self.num_captions if self.num_captions > 0 else 0.0
        )

        # Precompute IDFs for all observed terms
        self.idf_cache = {
            t: self._compute_idf(df, self.num_docs) for t, df in self.doc_freq.items()
        }
        self.caption_idf_cache = {
            t: self._compute_idf(df, self.num_captions)
            for t, df in self.caption_doc_freq.items()
        }

    def fit_metadata(self, image_ids: Sequence[str], categories: Sequence[str]) -> None:
        """Builds a Category/Metadata BM25 index (Mode B).

        Args:
            image_ids: List of image IDs.
            categories: List of category strings (e.g. 'airport', 'residential').
        """
        category_docs = [[cat] for cat in categories]
        self.aggregation_mode = "category_metadata"
        self.fit(image_ids, category_docs)

    @staticmethod
    def _compute_idf(doc_freq: int, num_docs: int) -> float:
        """Computes probabilistic IDF with floor protection."""
        return math.log(((num_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1.0)

    def score_query_combined(self, query_tokens: List[str]) -> List[float]:
        """Scores all images using the combined document representation."""
        scores = [0.0] * self.num_docs
        query_terms = [t for t in query_tokens if t in self.doc_freq]

        if not query_terms:
            return scores

        k1 = self.k1
        b = self.b
        avgdl = self.avg_doc_len

        for term in query_terms:
            idf = self.idf_cache[term]
            for doc_idx in range(self.num_docs):
                tf = self.doc_term_freqs[doc_idx].get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lengths[doc_idx]
                    denom = tf + k1 * (1.0 - b + b * (doc_len / (avgdl + 1e-9)))
                    scores[doc_idx] += idf * ((tf * (k1 + 1.0)) / (denom + 1e-9))

        return scores

    def score_query_max_caption(self, query_tokens: List[str]) -> List[float]:
        """Scores all images by taking the max score across each image's individual captions."""
        image_scores: Dict[str, float] = {img_id: 0.0 for img_id in self.image_ids}
        query_terms = [t for t in query_tokens if t in self.caption_doc_freq]

        if not query_terms:
            return [0.0] * self.num_docs

        k1 = self.k1
        b = self.b
        avgdl = self.avg_caption_len

        for cap_idx in range(self.num_captions):
            cap_score = 0.0
            cap_len = self.caption_lengths[cap_idx]
            cap_freq = self.caption_term_freqs[cap_idx]

            for term in query_terms:
                tf = cap_freq.get(term, 0)
                if tf > 0:
                    idf = self.caption_idf_cache[term]
                    denom = tf + k1 * (1.0 - b + b * (cap_len / (avgdl + 1e-9)))
                    cap_score += idf * ((tf * (k1 + 1.0)) / (denom + 1e-9))

            img_id = self.caption_to_image_id[cap_idx]
            if cap_score > image_scores[img_id]:
                image_scores[img_id] = cap_score

        return [image_scores[img_id] for img_id in self.image_ids]

    def get_target_leave_one_out_captions(
        self, target_image_id: str, query_text: str, query_caption_idx: Optional[int] = None
    ) -> List[str]:
        """Extracts the remaining captions for the target image, strictly excluding query_text.

        Args:
            target_image_id: Identifier of the target image.
            query_text: Exact query caption string.
            query_caption_idx: Optional index of the caption being queried (0 to 4).

        Returns:
            List of remaining captions, guaranteed to exclude query_text.
        """
        all_caps = self.image_captions.get(target_image_id, [])
        clean_q = query_text.strip().lower()

        # Filter by index if provided, and ensure exact query string is absent
        remaining = []
        for idx, cap in enumerate(all_caps):
            if query_caption_idx is not None and idx == query_caption_idx:
                continue
            if cap.strip().lower() == clean_q:
                # Exclude exact string match to prevent query leakage
                continue
            remaining.append(cap)

        # Edge case fallback: if all remaining were exact string duplicates, exclude by index
        if not remaining and query_caption_idx is not None:
            remaining = [c for idx, c in enumerate(all_caps) if idx != query_caption_idx]

        return remaining

    def rank_gallery_leave_one_out(
        self,
        query_text: str,
        target_image_id: str,
        query_caption_idx: Optional[int] = None,
    ) -> Tuple[List[str], List[float]]:
        """Ranks gallery with the query strictly excluded from its target document.

        Methodological Rule:
            For all images j != target: D_j contains all standard captions.
            For target image i: D_i contains strictly the remaining captions (excluding query_text).

        Args:
            query_text: Natural language query string.
            target_image_id: True originating target image ID.
            query_caption_idx: Optional caption index (0 to 4).

        Returns:
            Tuple of (sorted_image_ids, sorted_scores).
        """
        query_tokens = tokenize(query_text)
        # 1. Compute baseline scores for all gallery images
        scores = self.score_query_combined(query_tokens)

        # 2. Re-score target image using only its remaining captions
        if target_image_id in self.image_to_idx:
            target_idx = self.image_to_idx[target_image_id]
            remaining_captions = self.get_target_leave_one_out_captions(
                target_image_id, query_text, query_caption_idx
            )

            rem_tokens = tokenize(" ".join(remaining_captions))
            rem_len = len(rem_tokens)
            rem_tf = Counter(rem_tokens)

            new_target_score = 0.0
            k1 = self.k1
            b = self.b
            avgdl = self.avg_doc_len

            query_terms = [t for t in query_tokens if t in self.doc_freq]
            for term in query_terms:
                tf = rem_tf.get(term, 0)
                if tf > 0:
                    idf = self.idf_cache[term]
                    denom = tf + k1 * (1.0 - b + b * (rem_len / (avgdl + 1e-9)))
                    new_target_score += idf * ((tf * (k1 + 1.0)) / (denom + 1e-9))

            scores[target_idx] = new_target_score

        # 3. Deterministic sort descending by score, tie-breaking by index
        ranked_indices = sorted(
            range(self.num_docs),
            key=lambda idx: (scores[idx], -idx),
            reverse=True,
        )
        ranked_ids = [self.image_ids[idx] for idx in ranked_indices]
        ranked_scores = [float(scores[idx]) for idx in ranked_indices]
        return ranked_ids, ranked_scores

    def rank_gallery(
        self,
        query_text: str,
        target_image_id: Optional[str] = None,
        query_caption_idx: Optional[int] = None,
    ) -> List[str]:
        """Ranks all gallery images in descending order of BM25 score.

        If aggregation_mode is 'leave_one_caption_out' and target_image_id is provided,
        it routes to rank_gallery_leave_one_out to prevent query leakage.

        Args:
            query_text: Query string.
            target_image_id: Optional target image ID for leave-one-out mode.
            query_caption_idx: Optional caption index for leave-one-out mode.

        Returns:
            List of image IDs sorted descending by relevance score.
        """
        if self.aggregation_mode == "leave_one_caption_out" and target_image_id is not None:
            ranked_ids, _ = self.rank_gallery_leave_one_out(
                query_text, target_image_id, query_caption_idx
            )
            return ranked_ids

        query_tokens = tokenize(query_text)
        if self.aggregation_mode == "max_caption":
            scores = self.score_query_max_caption(query_tokens)
        else:
            scores = self.score_query_combined(query_tokens)

        ranked_indices = sorted(
            range(self.num_docs),
            key=lambda idx: (scores[idx], -idx),
            reverse=True,
        )
        return [self.image_ids[idx] for idx in ranked_indices]

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Queries the index and returns top-K retrieved images and scores."""
        query_tokens = tokenize(query_text)
        if self.aggregation_mode == "max_caption":
            scores = self.score_query_max_caption(query_tokens)
        else:
            scores = self.score_query_combined(query_tokens)

        ranked_indices = sorted(
            range(self.num_docs),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        return [(self.image_ids[idx], float(scores[idx])) for idx in ranked_indices]
