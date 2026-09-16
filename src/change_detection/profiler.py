"""CPU and memory profiling utilities for change detection pipelines.

Measures wall-clock latency (perf_counter) and peak resident set size (RSS)
in compliance with strict CPU-only and <= 8 GB RAM constraints.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np


@dataclass
class StageProfile:
    """Latency profile for a specific pipeline stage."""
    total_time_s: float = 0.0
    call_count: int = 0
    per_item_times_ms: List[float] = field(default_factory=list)

    def record(self, elapsed_s: float) -> None:
        self.total_time_s += elapsed_s
        self.call_count += 1
        self.per_item_times_ms.append(elapsed_s * 1000.0)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.per_item_times_ms)) if self.per_item_times_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.per_item_times_ms, 50)) if self.per_item_times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.per_item_times_ms, 95)) if self.per_item_times_ms else 0.0


class ChangeDetectionProfiler:
    """Tracks latency breakdown across stages and hardware resource utilization."""

    def __init__(self):
        self.stages: Dict[str, StageProfile] = {
            "preprocessing": StageProfile(),
            "difference_computation": StageProfile(),
            "thresholding": StageProfile(),
            "reconstruction": StageProfile(),
            "per_pair_total": StageProfile(),
        }
        self.start_time: float = 0.0
        self.total_pipeline_time_s: float = 0.0

    def start_pipeline(self) -> None:
        self.start_time = time.perf_counter()

    def end_pipeline(self) -> float:
        self.total_pipeline_time_s = time.perf_counter() - self.start_time
        return self.total_pipeline_time_s

    def record_stage(self, stage_name: str, elapsed_s: float) -> None:
        if stage_name not in self.stages:
            self.stages[stage_name] = StageProfile()
        self.stages[stage_name].record(elapsed_s)

    @staticmethod
    def get_peak_rss_mb() -> float:
        """Returns peak Resident Set Size (RSS) in Megabytes."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS, ru_maxrss is reported in bytes; on Linux, in kilobytes
        if os.uname().sysname == "Darwin":
            return float(usage.ru_maxrss / (1024 * 1024))
        return float(usage.ru_maxrss / 1024)

    def summary(self) -> Dict[str, Any]:
        """Generates comprehensive latency and memory summary dict."""
        peak_rss = self.get_peak_rss_mb()
        stage_summaries = {}
        for name, stage in self.stages.items():
            stage_summaries[name] = {
                "total_time_s": stage.total_time_s,
                "call_count": stage.call_count,
                "mean_ms": stage.mean_ms,
                "p50_ms": stage.p50_ms,
                "p95_ms": stage.p95_ms,
            }

        per_pair = self.stages["per_pair_total"]
        return {
            "total_runtime_s": self.total_pipeline_time_s,
            "peak_rss_mb": peak_rss,
            "mean_pair_latency_ms": per_pair.mean_ms,
            "p50_pair_latency_ms": per_pair.p50_ms,
            "p95_pair_latency_ms": per_pair.p95_ms,
            "stages": stage_summaries,
        }
