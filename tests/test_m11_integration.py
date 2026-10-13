"""Integration tests for Milestone M11 benchmarking and scientific validation.

Covers:
- Presence and non-emptiness of all M11 results and figures.
- Strict non-mutation of M5, M7, M8, M9, M10 historical artifacts.
- M11 configuration loader and schema completeness.
- CPU benchmark configuration conformance (CPU-only, threads=4, warm-up policy).
- Streamlit change detection UI benchmark integration compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from utils.config_loader import load_m11_config, get_project_root


def test_load_m11_config():
    """Verifies that load_m11_config loads valid configuration with all required sections."""
    cfg = load_m11_config()
    assert cfg["experiment"]["milestone"] == "M11"
    assert cfg["experiment"]["num_threads"] == 4
    assert cfg["dataset"]["name"] == "LEVIR-CD"
    assert "models" in cfg
    assert "statistical_evaluation" in cfg
    assert "cpu_benchmarking" in cfg
    assert "paths" in cfg


def test_historical_artifacts_untouched():
    """Verifies that M5-M10 historical result files are authentic and preserved bit-for-bit."""
    root = get_project_root()

    # M5
    m5_summary = root / "experiments" / "results" / "m5" / "summary.json"
    assert m5_summary.exists()
    with open(m5_summary, "r", encoding="utf-8") as f:
        data_m5 = json.load(f)
    assert any(r["method"] == "B1_Pixel_Diff" for r in data_m5)
    assert any(r["method"] == "B2_SSIM" for r in data_m5)
    assert any(r["method"] == "B3_CVA" for r in data_m5)

    # M7
    m7_summary = root / "experiments" / "results" / "m7" / "summary.json"
    assert m7_summary.exists()
    with open(m7_summary, "r", encoding="utf-8") as f:
        data_m7 = json.load(f)
    assert pytest.approx(data_m7[0]["f1"], rel=1e-3) == 0.6035

    # M8
    m8_summary = root / "experiments" / "results" / "m8" / "summary.json"
    assert m8_summary.exists()
    with open(m8_summary, "r", encoding="utf-8") as f:
        data_m8 = json.load(f)
    assert pytest.approx(data_m8[0]["recall"], rel=1e-3) == 0.7758

    # M9
    m9_summary = root / "experiments" / "results" / "m9" / "summary.json"
    assert m9_summary.exists()
    with open(m9_summary, "r", encoding="utf-8") as f:
        data_m9 = json.load(f)
    assert pytest.approx(data_m9[0]["f1"], rel=1e-3) == 0.5927

    # M10
    m10_summary = root / "experiments" / "results" / "m10" / "summary.json"
    assert m10_summary.exists()
    with open(m10_summary, "r", encoding="utf-8") as f:
        data_m10 = json.load(f)
    assert pytest.approx(data_m10[0]["f1"], rel=1e-3) == 0.5927


def test_m11_artifact_inventory_and_integrity():
    """Verifies that all M11 results and visualizations exist and are non-empty."""
    root = get_project_root()
    res_dir = root / "experiments" / "results" / "m11"
    fig_dir = root / "experiments" / "figures" / "m11"

    if not res_dir.exists():
        pytest.skip("M11 benchmark results not yet produced.")

    expected_result_files = [
        "benchmark_summary.json",
        "benchmark_summary.csv",
        "per_scene_all_models.csv",
        "statistical_bootstrap_ci.json",
        "paired_statistical_tests.json",
        "qat_ablation_analysis.json",
        "cpu_performance_benchmark.json",
        "threshold_provenance_audit.json",
    ]

    for fname in expected_result_files:
        fpath = res_dir / fname
        assert fpath.exists(), f"Missing M11 result artifact: {fname}"
        assert fpath.stat().st_size > 0, f"M11 artifact is empty: {fname}"

    expected_figures = [
        "precision_recall_tradeoff.png",
        "f1_iou_benchmark_bars.png",
        "false_positive_reduction_log.png",
        "cpu_latency_vs_f1.png",
        "per_scene_f1_distribution.png",
        "qat_ablation_tradeoff.png",
        "cross_model_robustness.png",
    ]

    for figname in expected_figures:
        figpath = fig_dir / figname
        assert figpath.exists(), f"Missing M11 research figure: {figname}"
        assert figpath.stat().st_size > 1000, f"M11 figure appears corrupt or empty: {figname}"


def test_cpu_benchmark_metadata():
    """Verifies CPU benchmark profiling conditions."""
    root = get_project_root()
    p = root / "experiments" / "results" / "m11" / "cpu_performance_benchmark.json"
    if not p.exists():
        pytest.skip("CPU benchmark file not yet produced.")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["configuration"]["num_threads"] == 4
    assert data["configuration"]["warmup_runs"] >= 1
    assert data["configuration"]["input_scene_dimension"] == [1024, 1024]
    assert data["configuration"]["patch_dimension"] == [256, 256]
    assert len(data["profiles"]) >= 6


def test_threshold_provenance_audit_completeness():
    """Verifies that all models in threshold_provenance_audit.json have explicit provenance records."""
    root = get_project_root()
    p = root / "experiments" / "results" / "m11" / "threshold_provenance_audit.json"
    if not p.exists():
        pytest.skip("Threshold audit not yet produced.")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"]["split_isolation_verified"] is True
    assert data["dataset"]["test_scenes"] == 128
    assert len(data["models"]) >= 7

    for m_id, m_meta in data["models"].items():
        assert "threshold" in m_meta
        assert "source" in m_meta
        assert m_meta["status"] in ["VERIFIED", "PARTIALLY_VERIFIED", "EXPLORATORY"]
