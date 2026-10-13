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


@st.cache_resource
def load_m7_detector():
    """Initializes and caches the trained M7 FC-Siam-diff detector and frozen threshold."""
    ckpt_path = ROOT / "experiments" / "checkpoints" / "m7" / "fc_siam_diff_best.pt"
    if not ckpt_path.exists():
        return None, None, None
    try:
        from src.change_detection.fc_siam_diff import FCSiamDiffDetector, load_checkpoint
        model, metadata = load_checkpoint(ckpt_path, device="cpu")
        detector = FCSiamDiffDetector(model=model, device="cpu", name="M7_FC_Siam_diff")
        tau = float(metadata.get("frozen_threshold", 0.55))
        return detector, tau, metadata
    except Exception as e:
        return None, None, {"error": str(e)}


@st.cache_resource
def load_m8_detector():
    """Initializes and caches the trained M8 SNUNet detector and frozen threshold."""
    ckpt_path = ROOT / "experiments" / "checkpoints" / "m8" / "snunet_best.pt"
    if not ckpt_path.exists():
        return None, None, None
    try:
        from src.change_detection.snunet import SNUNetDetector, load_checkpoint
        model, metadata = load_checkpoint(ckpt_path, device="cpu")
        detector = SNUNetDetector(model=model, device="cpu", name="M8_SNUNet")
        tau = float(metadata.get("frozen_threshold", 0.50))
        return detector, tau, metadata
    except Exception as e:
        return None, None, {"error": str(e)}


@st.cache_resource
def load_m9_detector():
    """Initializes and caches the trained M9 BIT-CD detector and frozen threshold."""
    ckpt_path = ROOT / "experiments" / "checkpoints" / "m9" / "bit_cd_best.pt"
    if not ckpt_path.exists():
        return None, None, None
    try:
        from src.change_detection.bit_cd import BITChangeDetector, load_checkpoint
        model, metadata = load_checkpoint(ckpt_path, device="cpu")
        detector = BITChangeDetector(model=model, device="cpu", name="M9_BIT_CD")
        tau = float(metadata.get("frozen_threshold", 0.50))
        return detector, tau, metadata
    except Exception as e:
        return None, None, {"error": str(e)}


@st.cache_resource
def load_m10_detector():
    """Initializes and caches the trained M10 QAT-CD detector and frozen parameters."""
    m9_ckpt = ROOT / "experiments" / "checkpoints" / "m9" / "bit_cd_best.pt"
    m10_meta_path = ROOT / "experiments" / "checkpoints" / "m10" / "qat_cd_metadata.json"
    if not (m9_ckpt.exists() and m10_meta_path.exists()):
        return None, None, None
    try:
        from src.change_detection.bit_cd import BITChangeDetector, load_checkpoint as load_m9_checkpoint
        from src.change_detection.qat_cd import QualityAwareChangeDetector, load_qat_metadata
        base_model, _ = load_m9_checkpoint(m9_ckpt, device="cpu")
        base_detector = BITChangeDetector(model=base_model, device="cpu", name="M9_BIT_CD")
        m10_meta = load_qat_metadata(m10_meta_path)
        gamma = float(m10_meta.get("calibrated_gamma", 0.0))
        tau = float(m10_meta.get("calibrated_threshold", 0.45))
        w = m10_meta.get("weights", {"reg": 0.34, "illum": 0.33, "sharp": 0.33})
        weights = (w["reg"], w["illum"], w["sharp"])
        detector = QualityAwareChangeDetector(
            base_detector=base_detector,
            gamma=gamma,
            weights=weights,
            use_reg=True,
            use_illum=True,
            use_sharp=True,
            use_uncertainty=True,
            name="M10_QAT_CD",
        )
        return detector, tau, m10_meta
    except Exception as e:
        return None, None, {"error": str(e)}


def render_change_detection_page():
    """Renders the Bi-Temporal Change Detection page."""
    render_header(
        title="Bi-Temporal Change Detection",
        subtitle="Classical Baselines, FC-Siam-diff, SNUNet-CD, BIT-CD & Quality-Aware QAT-CD on LEVIR-CD (1024×1024 Pairs, 0.5m GSD)",
        badge_text="Milestones 5, 7, 8, 9 & 10 Verified",
    )

    render_hardware_constraint_badge()

    extractor, detectors, frozen_thresholds = get_change_detection_detectors()
    m7_det, m7_tau, m7_meta = load_m7_detector()
    m8_det, m8_tau, m8_meta = load_m8_detector()
    m9_det, m9_tau, m9_meta = load_m9_detector()
    m10_det, m10_tau, m10_meta = load_m10_detector()


    # =========================================================================
    # SECTION 1: INTERACTIVE CHANGE DETECTOR
    # =========================================================================
    st.markdown("### Interactive Bi-Temporal Pair Detection")
    st.markdown(
        """
        Select a real **LEVIR-CD test pair** and evaluate the change detection algorithms. 
        Each detector operates patch-by-patch ($256 \\times 256$) with seamless $1024 \\times 1024$ mosaic reconstruction on CPU.
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

    det_options = [
        "M10: QAT-CD (Quality-Aware Temporal Change Detection)",
        "M9: BIT-CD (Bitemporal Image Transformer + Cross-Attention)",
        "M8: SNUNet-CD (Densely Connected Siamese + ECAM)",
        "M7: FC-Siam-diff (Learned Siamese U-Net)",
        "B1: Pixel Difference (L1 Mean Intensity)",
        "B2: SSIM Dissimilarity (1.0 - SSIM, 11x11 Gaussian)",
        "B3: Change Vector Analysis (Normalized L2 Magnitude)",
    ]

    with col_select_det:
        default_idx = 0 if m10_det is not None else (1 if m9_det is not None else (2 if m8_det is not None else (3 if m7_det is not None else 4)))
        detector_key = st.selectbox(
            "Select Change Detector:",
            det_options,
            index=default_idx,
        )
        det_name_map = {
            "M10: QAT-CD (Quality-Aware Temporal Change Detection)": "M10_QAT_CD",
            "M9: BIT-CD (Bitemporal Image Transformer + Cross-Attention)": "M9_BIT_CD",
            "M8: SNUNet-CD (Densely Connected Siamese + ECAM)": "M8_SNUNet",
            "M7: FC-Siam-diff (Learned Siamese U-Net)": "M7_FC_Siam_diff",
            "B1: Pixel Difference (L1 Mean Intensity)": "B1_Pixel_Diff",
            "B2: SSIM Dissimilarity (1.0 - SSIM, 11x11 Gaussian)": "B2_SSIM",
            "B3: Change Vector Analysis (Normalized L2 Magnitude)": "B3_CVA",
        }
        current_det_name = det_name_map[detector_key]
        if current_det_name == "M10_QAT_CD":
            if m10_det is None:
                err_msg = m10_meta.get("error") if m10_meta else "Metadata not found at experiments/checkpoints/m10/qat_cd_metadata.json"
                st.error(f"M10 QAT-CD detector unavailable: {err_msg}")
                return
            current_tau = m10_tau
        elif current_det_name == "M9_BIT_CD":
            if m9_det is None:
                err_msg = m9_meta.get("error") if m9_meta else "Checkpoint not found at experiments/checkpoints/m9/bit_cd_best.pt"
                st.error(f"M9 BIT-CD detector unavailable: {err_msg}")
                return
            current_tau = m9_tau
        elif current_det_name == "M8_SNUNet":
            if m8_det is None:
                err_msg = m8_meta.get("error") if m8_meta else "Checkpoint not found at experiments/checkpoints/m8/snunet_best.pt"
                st.error(f"M8 SNUNet detector unavailable: {err_msg}")
                return
            current_tau = m8_tau
        elif current_det_name == "M7_FC_Siam_diff":
            if m7_det is None:
                err_msg = m7_meta.get("error") if m7_meta else "Checkpoint not found at experiments/checkpoints/m7/fc_siam_diff_best.pt"
                st.error(f"M7 FC-Siam-diff detector unavailable: {err_msg}")
                return
            current_tau = m7_tau
        else:
            current_tau = frozen_thresholds[current_det_name]

    with col_diag_toggle:
        st.write("")
        st.write("")
        show_diagnostics = st.checkbox("Show Error Overlay", value=True)

    render_frozen_threshold_notice(threshold_val=current_tau, detector_name=current_det_name)
    if current_det_name == "M10_QAT_CD" and m10_meta:
        st.caption(
            f"Architecture: QAT-CD (Base: BIT-CD 552,097 params, 0 learned quality params) | "
            f"Calibrated gamma*: {m10_meta.get('calibrated_gamma', 0.0):.2f} | "
            f"Frozen Threshold tau*: {current_tau:.4f}"
        )
    elif current_det_name == "M9_BIT_CD" and m9_meta:
        st.caption(
            f"Architecture: BIT-CD (552,097 parameters) | Best Val F1: {m9_meta.get('best_val_f1', 0.0):.4f} | "
            f"Frozen Threshold tau*: {current_tau:.4f}"
        )
    elif current_det_name == "M8_SNUNet" and m8_meta:
        st.caption(
            f"Architecture: SNUNet-CD (563,377 parameters) | Best Val F1: {m8_meta.get('best_val_f1', 0.0):.4f} | "
            f"Frozen Threshold tau*: {current_tau:.4f}"
        )
    elif current_det_name == "M7_FC_Siam_diff" and m7_meta:
        st.caption(
            f"Architecture: FC-Siam-diff (1,471,409 parameters) | Best Val F1: {m7_meta.get('best_val_f1', 0.0):.4f} | "
            f"Frozen Threshold tau*: {current_tau:.4f}"
        )

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
    if current_det_name == "M10_QAT_CD":
        detector = m10_det.base_detector
    elif current_det_name == "M9_BIT_CD":
        detector = m9_det
    elif current_det_name == "M8_SNUNet":
        detector = m8_det
    elif current_det_name == "M7_FC_Siam_diff":
        detector = m7_det
    else:
        detector = detectors[current_det_name]

    # Extract 16 patches
    patches = extractor.extract_from_pair(img_a, img_b, gt_mask, parent_stem=selected_pair)
    diff_patches = []
    for p_a, p_b, p_lbl, p_meta in patches:
        d_patch = detector.compute_difference_map(p_a, p_b)
        diff_patches.append((d_patch, p_meta))

    # Reconstruct full 1024x1024 continuous difference map
    base_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)

    diag = None
    if current_det_name == "M10_QAT_CD":
        diag = m10_det.compute_quality_diagnostics(img_a, img_b, base_prob=base_diff)
        full_diff = diag["gated_prob"]
    else:
        full_diff = base_diff

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

    # Quality Reliability Indicators & Diagnostics Panel (M10 QAT-CD)
    if current_det_name == "M10_QAT_CD" and diag is not None:
        with st.expander("Quality Reliability Indicators & Diagnostics", expanded=False):
            st.markdown(
                """
                **Local Quality Discrepancy & Detector Uncertainty Breakdown**:
                These signals capture misregistration, illumination disparity, and optical blur differences
                to attenuate ambiguous change detections near the decision boundary.
                """
            )
            q_c1, q_c2, q_c3 = st.columns(3)
            with q_c1:
                st.image(
                    diag["s_reg"],
                    caption=f"Registration Discrepancy S_reg (Mean: {diag['s_reg'].mean():.4f})",
                    use_container_width=True,
                    clamp=True,
                )
            with q_c2:
                st.image(
                    diag["s_illum"],
                    caption=f"Illumination Discrepancy S_illum (Mean: {diag['s_illum'].mean():.4f})",
                    use_container_width=True,
                    clamp=True,
                )
            with q_c3:
                st.image(
                    diag["s_sharp"],
                    caption=f"Sharpness Discrepancy S_sharp (Mean: {diag['s_sharp'].mean():.4f})",
                    use_container_width=True,
                    clamp=True,
                )

            q_c4, q_c5, q_c6 = st.columns(3)
            with q_c4:
                st.image(
                    diag["defect_index"],
                    caption=f"Composite Defect Risk Q (Mean: {diag['defect_index'].mean():.4f})",
                    use_container_width=True,
                    clamp=True,
                )
            with q_c5:
                st.image(
                    diag["uncertainty"],
                    caption=f"Margin Uncertainty U_det (Mean: {diag['uncertainty'].mean():.4f})",
                    use_container_width=True,
                    clamp=True,
                )
            with q_c6:
                st.image(
                    diag["base_prob"],
                    caption=f"Prior Base Probability P0 (Mean: {diag['base_prob'].mean():.4f})",
                    use_container_width=True,
                    clamp=True,
                )

    # =========================================================================
    # SECTION 2: OFFICIAL BENCHMARK RESULTS TABLE
    # =========================================================================
    st.markdown("---")
    m11_comp_file = ROOT / "experiments" / "results" / "m11" / "benchmark_summary.json"
    m10_comp_file = ROOT / "experiments" / "results" / "m10" / "comparison_m5_to_m10.json"
    m9_comp_file = ROOT / "experiments" / "results" / "m9" / "comparison_m5_m7_m8_m9.json"
    m8_comp_file = ROOT / "experiments" / "results" / "m8" / "comparison_m5_m7_m8.json"
    m7_comp_file = ROOT / "experiments" / "results" / "m7" / "comparison_m5_m7.json"

    if m11_comp_file.exists():
        import json
        with open(m11_comp_file, "r", encoding="utf-8") as f:
            comp_records = json.load(f)
        summary_df = pd.DataFrame(comp_records)
        if "method" not in summary_df.columns and "display_name" in summary_df.columns:
            summary_df["method"] = summary_df["display_name"]
        if "mean_pair_latency_ms" not in summary_df.columns and "mean_latency_ms" in summary_df.columns:
            summary_df["mean_pair_latency_ms"] = summary_df["mean_latency_ms"]
        st.markdown("### Official Benchmark Results (Milestone M11 Comprehensive Evaluation)")
        st.markdown(
            """
            Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
            Compares classical non-learned baselines (B1 Pixel Diff, B2 SSIM, B3 CVA) with learned Siamese U-Net (M7 FC-Siam-diff), dense Siamese network (M8 SNUNet-CD), attention-based Bitemporal Image Transformer (M9 BIT-CD), and Quality-Aware Temporal Change Detection (M10 QAT-CD) under standardized CPU benchmarking and strictly frozen validation thresholds.
            Loaded dynamically from `experiments/results/m11/benchmark_summary.json`.
            """
        )
    elif m10_comp_file.exists():
        import json
        with open(m10_comp_file, "r", encoding="utf-8") as f:
            comp_records = json.load(f)
        summary_df = pd.DataFrame(comp_records)
        st.markdown("### Official Benchmark Results (Classical vs. FC-Siam-diff vs. SNUNet-CD vs. BIT-CD vs. QAT-CD)")
        st.markdown(
            """
            Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
            Compares classical non-learned baselines (B1, B2, B3) with learned Siamese U-Net (M7 FC-Siam-diff), dense Siamese network (M8 SNUNet-CD), attention-based Bitemporal Image Transformer (M9 BIT-CD), and Quality-Aware Temporal Change Detection (M10 QAT-CD) under strictly frozen validation parameters.
            Loaded dynamically from `experiments/results/m10/comparison_m5_to_m10.json`.
            """
        )
    elif m9_comp_file.exists():
        import json
        with open(m9_comp_file, "r", encoding="utf-8") as f:
            comp_records = json.load(f)
        summary_df = pd.DataFrame(comp_records)
        st.markdown("### Official Benchmark Results (Classical vs. FC-Siam-diff vs. SNUNet-CD vs. BIT-CD)")
        st.markdown(
            """
            Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
            Compares classical non-learned baselines (B1, B2, B3) with learned Siamese U-Net (M7 FC-Siam-diff), dense Siamese network (M8 SNUNet-CD), and attention-based Bitemporal Image Transformer (M9 BIT-CD) under strictly frozen validation thresholds $\\tau^*$.
            Loaded dynamically from `experiments/results/m9/comparison_m5_m7_m8_m9.json`.
            """
        )

    elif m8_comp_file.exists():
        import json
        with open(m8_comp_file, "r", encoding="utf-8") as f:
            comp_records = json.load(f)
        summary_df = pd.DataFrame(comp_records)
        st.markdown("### Official Benchmark Results (Classical Baselines vs. M7 FC-Siam-diff vs. M8 SNUNet-CD)")
        st.markdown(
            """
            Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
            Compares classical non-learned baselines (B1, B2, B3) with learned Siamese U-Net (M7 FC-Siam-diff) and dense Siamese network (M8 SNUNet-CD) under strictly frozen validation thresholds $\\tau^*$.
            Loaded dynamically from `experiments/results/m8/comparison_m5_m7_m8.json`.
            """
        )
    elif m7_comp_file.exists():
        import json
        with open(m7_comp_file, "r", encoding="utf-8") as f:
            comp_records = json.load(f)
        summary_df = pd.DataFrame(comp_records)
        st.markdown("### Official Benchmark Results (Classical Baselines vs. M7 FC-Siam-diff)")
        st.markdown(
            """
            Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
            Compares classical non-learned baselines (B1, B2, B3) with learned Siamese U-Net (M7 FC-Siam-diff) under strictly frozen validation thresholds $\\tau^*$.
            Loaded dynamically from `experiments/results/m7/comparison_m5_m7.json`.
            """
        )
    else:
        summary_df = load_m5_summary()
        st.markdown("### Official Benchmark Results (Milestone 5)")
        st.markdown(
            """
            Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
            Loaded dynamically from `experiments/results/m5/summary.json`.
            """
        )

    if summary_df is not None and not summary_df.empty:
        # Filter and format
        disp_df = summary_df.copy()
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
        st.pyplot(plot_m5_detector_comparison(summary_df))
    else:
        st.warning("Benchmark results not found.")

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
