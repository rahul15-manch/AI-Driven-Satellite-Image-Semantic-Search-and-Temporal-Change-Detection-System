"""Unit and smoke tests for the Streamlit research frontend and data loaders."""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from utils.config_loader import load_m4_config, load_m5_config, load_m6_config, get_project_root
from utils.result_loader import (
    load_rsicd_splits,
    load_levir_splits,
    load_m4_summary,
    load_m4_qualitative_samples,
    load_m4_failure_analysis,
    load_m5_summary,
    load_m5_thresholds,
    load_m5_confusion_totals,
    load_m6_results,
    load_m6_summary,
    load_m6_degradation_table,
    get_available_rsicd_images,
    get_available_levir_test_pairs,
    get_m5_figures,
    get_m6_figures,
)
from ui.charts import (
    plot_m4_recall_comparison,
    plot_m4_latency_mrr,
    plot_m5_detector_comparison,
    plot_m6_f1_vs_severity,
)
from ui.image_viewer import create_error_overlay
import numpy as np


class TestConfigLoaders:
    """Verifies that all M4, M5, and M6 experiment configs load properly."""

    def test_load_m4_config(self):
        cfg = load_m4_config()
        assert "dataset" in cfg
        assert "bm25" in cfg
        assert "clip" in cfg

    def test_load_m5_config(self):
        cfg = load_m5_config()
        assert "dataset" in cfg
        assert "methods" in cfg
        assert "thresholding" in cfg

    def test_load_m6_config(self):
        cfg = load_m6_config()
        assert "dataset" in cfg
        assert "frozen_thresholds" in cfg
        assert "perturbations" in cfg


class TestResultLoaders:
    """Verifies that all underlying machine-readable result files exist and contain valid data."""

    def test_rsicd_splits_manifest(self):
        splits = load_rsicd_splits()
        assert splits is not None
        assert splits["dataset_name"] == "RSICD"
        assert splits["counts"]["test"] == 1093
        assert splits["counts"]["total"] == 10921

    def test_levir_splits_manifest(self):
        splits = load_levir_splits()
        assert splits is not None
        assert splits["dataset_name"] == "LEVIR-CD"
        assert splits["counts"]["test"] == 128
        assert splits["counts"]["val"] == 64
        assert splits["counts"]["total"] == 637

    def test_m4_summary_table(self):
        df = load_m4_summary()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 5
        assert "R@1" in df.columns
        assert "MRR" in df.columns
        assert "protocol" in df.columns
        # Verify Leave-One-Caption-Out is present
        assert any("Leave-One-Caption-Out" in m for m in df["method"])
        # Verify CLIP is present
        assert any("CLIP" in m for m in df["method"])

    def test_m4_detailed_artifacts(self):
        samples = load_m4_qualitative_samples()
        assert samples is not None
        assert len(samples) > 0
        assert "query_text" in samples[0]

        failures = load_m4_failure_analysis()
        assert failures is not None
        assert "observed_failure_patterns_percentages" in failures
        assert "observed_failure_patterns_counts" in failures

    def test_m5_summary_table(self):
        df = load_m5_summary()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 6
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert "iou" in df.columns
        # Verify B1, B2, B3 are evaluated
        methods = df["method"].unique().tolist()
        assert "B1_Pixel_Diff" in methods
        assert "B2_SSIM" in methods
        assert "B3_CVA" in methods

    def test_m5_thresholds(self):
        thresh = load_m5_thresholds()
        assert thresh is not None
        assert "B1_Pixel_Diff" in thresh
        assert "B2_SSIM" in thresh
        assert "B3_CVA" in thresh
        assert "validation_f1_optimal" in thresh["B1_Pixel_Diff"]

    def test_m6_results(self):
        df = load_m6_results()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 39  # 1 control + 12 conditions x 3 detectors
        assert "perturbation" in df.columns
        assert "severity" in df.columns
        assert "relative_f1_degradation_pct" in df.columns
        assert "additional_false_positives" in df.columns

        # Verify SSIM misregistration additional FP is recorded accurately
        ssim_misreg = df[
            (df["detector"] == "B2_SSIM")
            & (df["perturbation"] == "geometric_misregistration")
            & (df["severity"] == "strong")
        ]
        assert len(ssim_misreg) == 1
        assert ssim_misreg.iloc[0]["additional_false_positives"] == 5679832

    def test_dataset_sample_availability(self):
        rsicd_imgs = get_available_rsicd_images()
        assert len(rsicd_imgs) > 0

        levir_pairs = get_available_levir_test_pairs()
        assert len(levir_pairs) == 128
        assert "test_1" in levir_pairs
        assert "test_20" in levir_pairs
        assert "test_50" in levir_pairs
        assert "test_100" in levir_pairs

    def test_figure_artifacts(self):
        m5_figs = get_m5_figures()
        assert "validation_threshold_curves" in m5_figs
        assert "qualitative_comparison_test_1" in m5_figs

        m6_figs = get_m6_figures()
        assert "f1_vs_severity" in m6_figs
        assert "additional_fp_vs_severity" in m6_figs
        assert "relative_f1_degradation" in m6_figs
        assert "detector_perturbation_heatmap" in m6_figs


class TestUIComponents:
    """Verifies that chart generators and UI components execute without rendering errors."""

    def test_m4_charts(self):
        summary_df = load_m4_summary()
        assert summary_df is not None
        fig1 = plot_m4_recall_comparison(summary_df)
        assert fig1 is not None

        fig2 = plot_m4_latency_mrr(summary_df)
        assert fig2 is not None

    def test_m5_charts(self):
        m5_df = load_m5_summary()
        assert m5_df is not None
        fig = plot_m5_detector_comparison(m5_df)
        assert fig is not None

    def test_m6_charts(self):
        m6_df = load_m6_results()
        assert m6_df is not None
        fig = plot_m6_f1_vs_severity(m6_df, "geometric_misregistration")
        assert fig is not None

    def test_error_overlay(self):
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        pred[2:5, 2:5] = 1
        gt[3:6, 3:6] = 1

        overlay = create_error_overlay(pred, gt)
        assert overlay.shape == (10, 10, 3)
        assert overlay.dtype == np.uint8
        # TP pixel at (3, 3) -> Green [34, 197, 94]
        assert list(overlay[3, 3]) == [34, 197, 94]
        # FP pixel at (2, 2) -> Red [239, 68, 68]
        assert list(overlay[2, 2]) == [239, 68, 68]
        # FN pixel at (5, 5) -> Blue [59, 130, 246]
        assert list(overlay[5, 5]) == [59, 130, 246]

    def test_all_pages_apptest(self):
        from streamlit.testing.v1 import AppTest
        root = get_project_root()
        pages = [
            root / "app.py",
            root / "pages" / "overview.py",
            root / "pages" / "dataset.py",
            root / "pages" / "retrieval.py",
            root / "pages" / "change_detection.py",
            root / "pages" / "false_alarm.py",
        ]
        for p in pages:
            at = AppTest.from_file(str(p), default_timeout=30).run()
            assert len(at.exception) == 0, f"Page {p} raised exceptions: {at.exception}"


class TestMilestoneBoundaries:
    """Ensures strict governance: no future milestones (M7-M10+) are represented as implemented."""

    def test_no_unimplemented_models_in_pages(self):
        app_file = Path(get_project_root()) / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Should not import or expose M7-M10
        assert "fc_siam_diff" not in content.lower()
        assert "snunet" not in content.lower()
        assert "qat_cd" not in content.lower()
        assert "quality_aware" not in content.lower()
