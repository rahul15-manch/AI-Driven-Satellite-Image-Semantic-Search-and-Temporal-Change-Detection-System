"""Machine-readable results and artifacts loader for Milestones M1-M6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from utils.config_loader import get_project_root

ROOT = get_project_root()


# =============================================================================
# DATASET & METADATA LOADERS
# =============================================================================

def load_rsicd_splits() -> Optional[Dict[str, Any]]:
    """Loads RSICD official split manifest."""
    path = ROOT / "data" / "splits" / "rsicd" / "rsicd_splits.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_levir_splits() -> Optional[Dict[str, Any]]:
    """Loads LEVIR-CD official split manifest."""
    path = ROOT / "data" / "splits" / "levir_cd" / "levir_splits.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rsicd_validation_report() -> Optional[Dict[str, Any]]:
    """Loads RSICD dataset validation audit report."""
    path = ROOT / "data" / "metadata" / "rsicd" / "validation_report.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_rsicd_images() -> List[str]:
    """Returns sorted list of available RSICD raw image filenames."""
    img_dir = ROOT / "data" / "raw" / "rsicd" / "images"
    if not img_dir.exists():
        return []
    return sorted([f.name for f in img_dir.glob("*.jpg")])


def get_available_levir_test_pairs() -> List[str]:
    """Returns sorted list of available LEVIR-CD test sample IDs (e.g. 'test_1', 'test_20')."""
    test_a_dir = ROOT / "data" / "raw" / "levir_cd" / "test" / "A"
    if not test_a_dir.exists():
        return []
    # Sort numerically by index
    stems = [p.stem for p in test_a_dir.glob("*.png")]
    def sort_key(s: str) -> int:
        try:
            return int(s.split("_")[-1])
        except (ValueError, IndexError):
            return 9999
    return sorted(stems, key=sort_key)


# =============================================================================
# MILESTONE 4: SEMANTIC RETRIEVAL RESULTS
# =============================================================================

def load_m4_summary() -> Optional[pd.DataFrame]:
    """Loads Milestone 4 summary benchmark results."""
    json_path = ROOT / "experiments" / "results" / "m4" / "summary.json"
    csv_path = ROOT / "experiments" / "results" / "m4" / "summary.csv"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            return df
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_m4_qualitative_samples() -> Optional[List[Dict[str, Any]]]:
    """Loads M4 qualitative retrieval samples."""
    path = ROOT / "experiments" / "results" / "m4" / "qualitative_samples.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_m4_failure_analysis() -> Optional[Dict[str, Any]]:
    """Loads M4 CLIP failure case breakdown."""
    path = ROOT / "experiments" / "results" / "m4" / "failure_analysis.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_m4_detailed_metrics(method_key: str) -> Optional[Dict[str, Any]]:
    """Loads detailed per-query and aggregate metrics for a specific M4 method.

    Supported keys:
    - 'corrected_bm25'
    - 'category_metadata_bm25'
    - 'clip'
    - 'prompt_ensemble'
    - 'legacy_bm25'
    """
    path_map = {
        "corrected_bm25": ROOT / "experiments" / "results" / "m4" / "corrected_bm25_results.json",
        "category_metadata_bm25": ROOT / "experiments" / "results" / "m4" / "category_metadata_bm25_results.json",
        "clip": ROOT / "experiments" / "results" / "m4" / "clip_results.json",
        "prompt_ensemble": ROOT / "experiments" / "results" / "m4" / "prompt_ensemble_results.json",
        "legacy_bm25": ROOT / "experiments" / "results" / "m4" / "legacy_caption_indexed" / "bm25_results.json",
    }
    path = path_map.get(method_key)
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# =============================================================================
# MILESTONE 5: CHANGE DETECTION RESULTS
# =============================================================================

def load_m5_summary() -> Optional[pd.DataFrame]:
    """Loads Milestone 5 summary evaluation table."""
    json_path = ROOT / "experiments" / "results" / "m5" / "summary.json"
    csv_path = ROOT / "experiments" / "results" / "m5" / "summary.csv"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return pd.DataFrame(data)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_m5_thresholds() -> Optional[Dict[str, Any]]:
    """Loads validation threshold calibration curves and optimal tau*."""
    path = ROOT / "experiments" / "results" / "m5" / "thresholds.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_m5_confusion_totals() -> Optional[Dict[str, Any]]:
    """Loads M5 confusion matrices totals (TP, FP, FN, TN)."""
    path = ROOT / "experiments" / "results" / "m5" / "confusion_totals.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_m5_per_image_metrics() -> Optional[pd.DataFrame]:
    """Loads per-image evaluation metrics on the 128 LEVIR-CD test pairs."""
    path = ROOT / "experiments" / "results" / "m5" / "per_image_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# =============================================================================
# MILESTONE 6: FALSE-ALARM & ROBUSTNESS RESULTS
# =============================================================================

def load_m6_results() -> Optional[pd.DataFrame]:
    """Loads Milestone 6 comprehensive perturbation benchmark results."""
    path = ROOT / "experiments" / "results" / "m6" / "m6_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_m6_summary() -> Optional[Dict[str, Any]]:
    """Loads Milestone 6 JSON summary report."""
    path = ROOT / "experiments" / "results" / "m6" / "m6_summary.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_m6_degradation_table() -> Optional[pd.DataFrame]:
    """Loads M6 degradation and false-alarm increase table."""
    path = ROOT / "experiments" / "results" / "m6" / "degradation_table.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# =============================================================================
# MILESTONE 7: LEARNED CHANGE DETECTION (FC-SIAM-DIFF)
# =============================================================================

def load_m7_results() -> Optional[pd.DataFrame]:
    """Loads Milestone 7 learned baseline test results table."""
    path = ROOT / "experiments" / "results" / "m7" / "m7_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_m7_summary() -> Optional[Dict[str, Any]]:
    """Loads Milestone 7 JSON summary report with training and evaluation stats."""
    path = ROOT / "experiments" / "results" / "m7" / "m7_summary.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# FIGURE PATH HELPERS
# =============================================================================

def get_m5_figures() -> Dict[str, Path]:
    """Returns mapping of available M5 figure files."""
    m5_dir = ROOT / "experiments" / "figures" / "m5"
    if not m5_dir.exists():
        return {}
    return {f.stem: f for f in m5_dir.glob("*.png")}


def get_m6_figures() -> Dict[str, Path]:
    """Returns mapping of available M6 figure files."""
    m6_dir = ROOT / "experiments" / "figures" / "m6"
    if not m6_dir.exists():
        return {}
    return {f.stem: f for f in m6_dir.glob("*.png")}


def get_m7_figures() -> Dict[str, Path]:
    """Returns mapping of available M7 figure files."""
    m7_dir = ROOT / "experiments" / "figures" / "m7"
    if not m7_dir.exists():
        return {}
    return {f.stem: f for f in m7_dir.glob("*.png")}

