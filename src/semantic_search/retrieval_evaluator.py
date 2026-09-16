"""Evaluation Module for Cross-Modal Satellite Image Retrieval.

Implements standard Caption-to-Own-Image evaluation protocol:
- Recall@1 (R@1)
- Recall@5 (R@5)
- Recall@10 (R@10)
- Mean Reciprocal Rank (MRR)
- Category-level breakdown and per-query diagnostic logs for failure analysis.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """Represents an evaluation query record."""
    query_id: str
    query_text: str
    target_image_id: str
    category: str = "unclassified"
    caption_idx: Optional[int] = None


@dataclass
class QueryEvaluationResult:
    """Individual per-query evaluation result."""
    query_id: str
    query_text: str
    target_image_id: str
    category: str
    target_rank: int
    reciprocal_rank: float
    is_r1: bool
    is_r5: bool
    is_r10: bool
    top_5_retrieved: List[str] = field(default_factory=list)
    top_5_scores: List[float] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval benchmark metrics."""
    method_name: str
    total_queries: int
    r1: float
    r5: float
    r10: float
    mrr: float
    category_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    query_results: List[QueryEvaluationResult] = field(default_factory=list)

    def to_dict(self, include_per_query: bool = False) -> Dict[str, Any]:
        """Converts metrics to a dictionary for JSON serialization."""
        d: Dict[str, Any] = {
            "method": self.method_name,
            "total_queries": self.total_queries,
            "R@1": round(self.r1, 6),
            "R@5": round(self.r5, 6),
            "R@10": round(self.r10, 6),
            "MRR": round(self.mrr, 6),
            "category_breakdown": self.category_breakdown,
        }
        if include_per_query:
            d["query_results"] = [asdict(r) for r in self.query_results]
        return d


class RetrievalEvaluator:
    """Evaluates ranking functions under Caption-to-Own-Image protocol."""

    def __init__(
        self,
        queries: Sequence[QueryRecord],
        gallery_image_ids: Sequence[str],
    ) -> None:
        """Initializes RetrievalEvaluator.

        Args:
            queries: Sequence of QueryRecord instances.
            gallery_image_ids: Full ordered list of gallery image identifiers.
        """
        self.queries = list(queries)
        self.gallery_image_ids = list(gallery_image_ids)
        self.gallery_set = set(self.gallery_image_ids)

        # Validate that all targets exist in the gallery
        missing = [q.target_image_id for q in self.queries if q.target_image_id not in self.gallery_set]
        if missing:
            raise ValueError(
                f"{len(missing)} query target images are not present in the gallery. Examples: {missing[:5]}"
            )

    def evaluate(
        self,
        method_name: str,
        ranking_fn: Callable[..., Tuple[List[str], Optional[List[float]]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> RetrievalMetrics:
        """Evaluates a retrieval method over all queries.

        Args:
            method_name: Identifier string for reporting (e.g., 'BM25', 'CLIP ViT-B/32').
            ranking_fn: Callable taking query string (and optionally target_image_id, query_caption_idx),
                        returning tuple of (sorted_gallery_image_ids, optional_scores).
            progress_callback: Optional callback(current, total) for tracking progress.

        Returns:
            RetrievalMetrics containing aggregate R@1, R@5, R@10, MRR, and breakdown.
        """
        total_queries = len(self.queries)
        query_results: List[QueryEvaluationResult] = []

        r1_count = 0
        r5_count = 0
        r10_count = 0
        reciprocal_ranks: List[float] = []

        category_stats: Dict[str, Dict[str, float]] = {}

        for idx, q_rec in enumerate(self.queries):
            try:
                ranked_ids, scores = ranking_fn(
                    q_rec.query_text,
                    target_image_id=q_rec.target_image_id,
                    query_caption_idx=q_rec.caption_idx,
                )
            except TypeError:
                ranked_ids, scores = ranking_fn(q_rec.query_text)

            # Find 1-based rank of the target image
            try:
                rank = ranked_ids.index(q_rec.target_image_id) + 1
            except ValueError:
                rank = len(ranked_ids) + 1


            rr = 1.0 / rank
            reciprocal_ranks.append(rr)

            is_r1 = rank <= 1
            is_r5 = rank <= 5
            is_r10 = rank <= 10

            if is_r1:
                r1_count += 1
            if is_r5:
                r5_count += 1
            if is_r10:
                r10_count += 1

            # Top 5 details for qualitative logs
            top_5_ids = ranked_ids[:5]
            top_5_scores = [float(s) for s in scores[:5]] if scores else []

            res = QueryEvaluationResult(
                query_id=q_rec.query_id,
                query_text=q_rec.query_text,
                target_image_id=q_rec.target_image_id,
                category=q_rec.category,
                target_rank=rank,
                reciprocal_rank=rr,
                is_r1=is_r1,
                is_r5=is_r5,
                is_r10=is_r10,
                top_5_retrieved=top_5_ids,
                top_5_scores=top_5_scores,
            )
            query_results.append(res)

            # Category tracking
            cat = q_rec.category
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "r1": 0, "r5": 0, "r10": 0, "rr_sum": 0.0}
            category_stats[cat]["count"] += 1
            if is_r1:
                category_stats[cat]["r1"] += 1
            if is_r5:
                category_stats[cat]["r5"] += 1
            if is_r10:
                category_stats[cat]["r10"] += 1
            category_stats[cat]["rr_sum"] += rr

            if progress_callback and (idx + 1) % 500 == 0:
                progress_callback(idx + 1, total_queries)

        r1 = r1_count / total_queries if total_queries > 0 else 0.0
        r5 = r5_count / total_queries if total_queries > 0 else 0.0
        r10 = r10_count / total_queries if total_queries > 0 else 0.0
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        # Compute per-category averages
        category_breakdown: Dict[str, Dict[str, float]] = {}
        for cat, stats in sorted(category_stats.items()):
            n = stats["count"]
            category_breakdown[cat] = {
                "count": n,
                "R@1": round(stats["r1"] / n, 4),
                "R@5": round(stats["r5"] / n, 4),
                "R@10": round(stats["r10"] / n, 4),
                "MRR": round(stats["rr_sum"] / n, 4),
            }

        return RetrievalMetrics(
            method_name=method_name,
            total_queries=total_queries,
            r1=r1,
            r5=r5,
            r10=r10,
            mrr=mrr,
            category_breakdown=category_breakdown,
            query_results=query_results,
        )
