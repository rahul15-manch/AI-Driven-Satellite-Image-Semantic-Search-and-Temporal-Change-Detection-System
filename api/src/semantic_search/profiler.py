"""CPU Latency and Memory Profiling for Semantic Retrieval.

Measures:
- Per-query latency: text preprocessing, encoding, vector search, end-to-end.
- Statistical percentiles: Mean, Median (p50), 95th percentile (p95).
- Memory tracking: Process Resident Set Size (RSS in MB) via psutil and peak heap via tracemalloc.
"""

from __future__ import annotations

import logging
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class LatencyProfile:
    """Statistical summary of execution latency in milliseconds."""
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class MemoryProfile:
    """Summary of process and heap memory consumption in megabytes."""
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    heap_peak_mb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rss_start_mb": round(self.rss_start_mb, 2),
            "rss_end_mb": round(self.rss_end_mb, 2),
            "rss_peak_mb": round(self.rss_peak_mb, 2),
            "heap_peak_mb": round(self.heap_peak_mb, 2),
        }


class RetrievalProfiler:
    """Profiles latency and memory across retrieval components on CPU."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())

    def get_current_rss_mb(self) -> float:
        """Returns the current process Resident Set Size in megabytes."""
        return self.process.memory_info().rss / (1024.0 * 1024.0)

    def profile_latencies(
        self,
        callable_fn: Callable[[str], Any],
        queries: List[str],
        warmup_runs: int = 5,
    ) -> Tuple[LatencyProfile, List[float]]:
        """Profiles the execution latency of a retrieval callable over a query set.

        Args:
            callable_fn: Callable taking query string.
            queries: List of test query strings.
            warmup_runs: Number of initial discarded runs for instruction cache stabilization.

        Returns:
            Tuple of (LatencyProfile, raw_latencies_in_ms).
        """
        # Warm-up phase
        for i in range(min(warmup_runs, len(queries))):
            callable_fn(queries[i])

        latencies_ms: List[float] = []

        for q in queries:
            t0 = time.perf_counter()
            callable_fn(q)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        arr = np.array(latencies_ms, dtype=np.float64)
        profile = LatencyProfile(
            count=len(latencies_ms),
            mean_ms=float(np.mean(arr)),
            p50_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
        )
        return profile, latencies_ms

    def run_with_memory_tracking(
        self,
        task_fn: Callable[[], Any],
    ) -> Tuple[Any, MemoryProfile]:
        """Executes a function while monitoring RSS and tracemalloc heap peak."""
        rss_start = self.get_current_rss_mb()
        tracemalloc.start()

        result = task_fn()

        _, heap_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_end = self.get_current_rss_mb()
        profile = MemoryProfile(
            rss_start_mb=rss_start,
            rss_end_mb=rss_end,
            rss_peak_mb=max(rss_start, rss_end),
            heap_peak_mb=heap_peak_bytes / (1024.0 * 1024.0),
        )
        return result, profile
