"""Page 4: Milestone 5 - Classical Bi-Temporal Change Detection (B1, B2, B3)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from ui.components import (
    render_header,
    render_frozen_threshold_notice,
    render_hardware_constraint_badge,
    render_metric_card,
)
from ui.charts import plot_m5_detector_comparison, display_research_figure
from ui.image_viewer import render_change_detection_panel
from utils.result_loader import (
    load_m5_summary,
    load_m5_thresholds,
    load_m5_confusion_totals,
    load_m5_per_image_metrics,
    get_available_levir_test_pairs,
    get_m5_figures,
)
from utils.config_loader import get_project_root, load_m5_config

# Import existing M5 modules
from src.data.patch_extractor import PatchExtractor
from src.change_detection.pixel_diff import PixelDiffDetector
from src.change_detection.ssim_detector import SSIMDetector
from src.change_detection.cva_detector import CVADetector
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix

ROOT = get_project_root()


# =============================================================================
# CACHED DETECTOR INSTANCES & PATCH EXTRACTOR
# =============================================================================

@st.cache_resource
def get_change_detection_detectors():
    """Initializes and caches the three M5 classical detectors matching configuration."""
    config = load_m5_config()
    patch_size = config["dataset"]["patch_size"]
    stride = config["dataset"]["stride"]
    padding_mode = config["dataset"]["padding_mode"]

    extractor = PatchExtractor(patch_size=patch_size, stride=stride, padding_mode=padding_mode)

    detectors = {
        "B1_Pixel_Diff": PixelDiffDetector(
            aggregation_mode=config["methods"]["b1_pixel_diff"]["aggregation_mode"]
        ),
        "B2_SSIM": SSIMDetector(
            win_size=config["methods"]["b2_ssim"]["win_size"],
            sigma=config["methods"]["b2_ssim"]["sigma"],
            channel_mode=config["methods"]["b2_ssim"]["channel_mode"],
        ),
        "B3_CVA": CVADetector(
            normalize=config["methods"]["b3_cva"]["normalize"]
        ),
    }

    # Frozen thresholds from M5 validation calibration
    frozen_thresholds = {
        "B1_Pixel_Diff": 0.4100,
        "B2_SSIM": 0.9000,
        "B3_CVA": 0.4050,
    }

    return extractor, detectors, frozen_thresholds


def render_change_detection_page():
    """Renders the Bi-Temporal Change Detection page."""
    render_header(
        title="Bi-Temporal Change Detection",
        subtitle="Classical Non-Learned Baselines on LEVIR-CD (1024×1024 Image Pairs, 0.5m GSD)",
        badge_text="Milestone 5 Implemented",
    )

    render_hardware_constraint_badge()

    extractor, detectors, frozen_thresholds = get_change_detection_detectors()

    # =========================================================================
    # SECTION 1: INTERACTIVE CHANGE DETECTOR
    # =========================================================================
    st.markdown("### Interactive Bi-Temporal Pair Detection")
    st.markdown(
        """
        Select a real **LEVIR-CD test pair** and evaluate the classical change detection algorithms. 
        Each detector operates patch-by-patch ($256 \\times 256$) with seamless $1024 \\times 1024$ mosaic reconstruction.
        """
    )

    available_pairs = get_available_levir_test_pairs()
    if not available_pairs:
        st.warning("LEVIR-CD test image pairs not found in data/raw/levir_cd/test/.")
        return

    # Prioritize well-known benchmark pairs first
    preferred_pairs = ["test_1", "test_20", "test_50", "test_100"]
    ordered_pairs = [p for p in preferred_pairs if p in available_pairs] + [
        p for p in available_pairs if p not in preferred_pairs
    ]

    col_select_pair, col_select_det, col_diag_toggle = st.columns([1.5, 2, 1.2])

    with col_select_pair:
        selected_pair = st.selectbox("Select LEVIR-CD Test Pair:", ordered_pairs, index=0)

    with col_select_det:
        detector_key = st.selectbox(
            "Select Classical Detector:",
            [
                "B1: Pixel Difference (L1 Mean Intensity)",
                "B2: SSIM Dissimilarity (1.0 - SSIM, 11x11 Gaussian)",
                "B3: Change Vector Analysis (Normalized L2 Magnitude)",
            ],
            index=0,
        )
        det_name_map = {
            "B1: Pixel Difference (L1 Mean Intensity)": "B1_Pixel_Diff",
            "B2: SSIM Dissimilarity (1.0 - SSIM, 11x11 Gaussian)": "B2_SSIM",
            "B3: Change Vector Analysis (Normalized L2 Magnitude)": "B3_CVA",
        }
        current_det_name = det_name_map[detector_key]
        current_tau = frozen_thresholds[current_det_name]

    with col_diag_toggle:
        st.write("")
        st.write("")
        show_diagnostics = st.checkbox("Show Error Overlay", value=True)

    render_frozen_threshold_notice(threshold_val=current_tau, detector_name=current_det_name)

    # Load pair images
    p_t1 = ROOT / "data" / "raw" / "levir_cd" / "test" / "A" / f"{selected_pair}.png"
    p_t2 = ROOT / "data" / "raw" / "levir_cd" / "test" / "B" / f"{selected_pair}.png"
    p_gt = ROOT / "data" / "raw" / "levir_cd" / "test" / "label" / f"{selected_pair}.png"

    if not (p_t1.exists() and p_t2.exists() and p_gt.exists()):
        st.error(f"Image files for {selected_pair} are missing on disk.")
        return

    # Load into memory
    img_a = np.array(Image.open(p_t1).convert("RGB"), dtype=np.uint8)
    img_b = np.array(Image.open(p_t2).convert("RGB"), dtype=np.uint8)
    gt_mask = (np.array(Image.open(p_gt).convert("L"), dtype=np.uint8) > 128).astype(np.uint8)

    # Execute Live Change Detection on CPU
    t0 = time.perf_counter()
    detector = detectors[current_det_name]

    # Extract 16 patches
    patches = extractor.extract_from_pair(img_a, img_b, gt_mask, parent_stem=selected_pair)
    diff_patches = []
    for p_a, p_b, p_lbl, p_meta in patches:
        d_patch = detector.compute_difference_map(p_a, p_b)
        diff_patches.append((d_patch, p_meta))

    # Reconstruct full 1024x1024 continuous difference map
    full_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)

    # Apply frozen decision threshold
    pred_mask = (full_diff >= current_tau).astype(np.uint8)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Compute pair-level confusion matrix & metrics
    cm = ConfusionMatrix()
    cm.update(pred_mask, gt_mask)
    pair_metrics = ChangeDetectionEvaluator.evaluate(cm)

    # Display 4-column visualizer
    render_change_detection_panel(
        img_t1=img_a,
        img_t2=img_b,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        diff_map=full_diff,
        show_error_overlay=show_diagnostics,
    )

    # Display Pair Metrics Summary
    st.markdown("#### Sample Pair Evaluation Metrics")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        render_metric_card("Precision", f"{pair_metrics['precision']*100:.2f}%")
    with m_col2:
        render_metric_card("Recall", f"{pair_metrics['recall']*100:.2f}%")
    with m_col3:
        render_metric_card("F1 Score", f"{pair_metrics['f1']:.4f}")
    with m_col4:
        render_metric_card("IoU", f"{pair_metrics['iou']*100:.2f}%")
    with m_col5:
        render_metric_card("CPU Latency", f"{elapsed_ms:.1f} ms")
    with m_col6:
        render_metric_card("False Positives", f"{cm.fp:,} px")

    # =========================================================================
    # SECTION 2: OFFICIAL M5 BENCHMARK RESULTS TABLE
    # =========================================================================
    st.markdown("---")
    st.markdown("### Official Benchmark Results (Milestone 5)")
    st.markdown(
        """
        Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
        Loaded dynamically from `experiments/results/m5/summary.json`.
        """
    )

    m5_summary_df = load_m5_summary()
    if m5_summary_df is not None:
        # Filter and format
        disp_df = m5_summary_df.copy()
        disp_df["Precision (%)"] = disp_df["precision"].apply(lambda v: f"{v*100:.2f}%")
        disp_df["Recall (%)"] = disp_df["recall"].apply(lambda v: f"{v*100:.2f}%")
        disp_df["F1 Score"] = disp_df["f1"].apply(lambda v: f"{v:.4f}")
        disp_df["IoU (%)"] = disp_df["iou"].apply(lambda v: f"{v*100:.2f}%")
        disp_df["Accuracy (%)"] = disp_df["accuracy"].apply(lambda v: f"{v*100:.2f}%")
        disp_df["Mean Latency (ms)"] = disp_df["mean_pair_latency_ms"].apply(lambda v: f"{v:.1f}")
        disp_df["Peak RAM (MB)"] = disp_df["peak_rss_mb"].apply(lambda v: f"{v:.1f}")
        disp_df["Threshold tau"] = disp_df["threshold_value"].apply(lambda v: f"{v:.4f}")

        st.dataframe(
            disp_df[[
                "method",
                "threshold_source",
                "Threshold tau",
                "Precision (%)",
                "Recall (%)",
                "F1 Score",
                "IoU (%)",
                "Mean Latency (ms)",
                "Peak RAM (MB)",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        # Plot comparison chart
        st.pyplot(plot_m5_detector_comparison(m5_summary_df))
    else:
        st.warning("M5 summary results not found at experiments/results/m5/summary.json.")

    # =========================================================================
    # SECTION 3: VALIDATION CURVES & QUALITATIVE FIGURES
    # =========================================================================
    st.markdown("### Precomputed Research Figures (Milestone 5)")
    m5_figs = get_m5_figures()

    tab_curves, tab_samples = st.tabs(["Validation Threshold Curves", "Qualitative Test Comparisons"])

    with tab_curves:
        thresh_fig = m5_figs.get("validation_threshold_curves")
        if thresh_fig:
            display_research_figure(
                thresh_fig,
                caption="Validation Sweep Curves across B1, B2, and B3 showing optimal F1 thresholds (crimson curve) calibrated strictly on the 64 validation pairs."
            )
        else:
            st.info("Validation threshold curves figure not found.")

    with tab_samples:
        q_options = [k for k in m5_figs.keys() if "qualitative_comparison" in k]
        if q_options:
            q_sel = st.selectbox("Select Qualitative Research Figure:", q_options, index=0)
            display_research_figure(
                m5_figs[q_sel],
                caption=f"Deterministic Qualitative Research Comparison ({q_sel})."
            )
        else:
            st.info("No precomputed qualitative comparison figures found in experiments/figures/m5/.")


if __name__ == "__main__":
    render_change_detection_page()
