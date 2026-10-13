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
    load_m11_summary,
    load_m11_bootstrap_ci,
    load_m11_paired_tests,
    load_m11_ablation,
    get_m11_figures,
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

    # QAT-CD Gating Configuration Control
    selected_gamma = 0.0
    if current_det_name == "M10_QAT_CD" and m10_det is not None:
        col_gamma_sel, col_gamma_exp = st.columns([1.5, 2.5])
        with col_gamma_sel:
            gamma_options = [
                "Calibrated: gamma = 0.0 (Optimal F1 on LEVIR-CD Val)",
                "Exploratory active gating: gamma = 0.30 (FP Reduction Trade-off)",
            ]
            gamma_choice = st.selectbox(
                "QAT-CD Gating Configuration (gamma):",
                gamma_options,
                index=0,
                help="Calibrated gamma=0.0 maintains unweighted F1. Exploratory gamma=0.30 attenuates detections near boundary using defect discrepancy."
            )
            selected_gamma = 0.0 if "0.0" in gamma_choice else 0.30
            m10_det.gamma = selected_gamma

        with col_gamma_exp:
            if selected_gamma > 0.0:
                st.info(
                    "**Exploratory Active Gating Note:** As documented in Milestone M11, active gating (gamma = 0.30) "
                    "reduces false-positive pixels by approximately **29.19%** relative to M9 BIT-CD, but sacrifices recall "
                    "and unweighted F1. This is an operational trade-off for high-precision scenarios, not an unconditional performance improvement."
                )
            else:
                st.caption(
                    "Calibrated configuration (gamma = 0.0): Evaluated at validation-calibrated operating point for optimal unweighted F1."
                )

    # Optional Manual Decision-Threshold Override
    col_ov_check, col_ov_val = st.columns([1.5, 2.5])
    with col_ov_check:
        override_threshold = st.checkbox(
            "Manual Threshold Override (Experimental)",
            value=False,
            help="Override the validation-calibrated decision threshold tau for this test inference."
        )

    if override_threshold:
        with col_ov_val:
            effective_tau = st.slider(
                "Decision Threshold Override (tau):",
                min_value=0.0,
                max_value=1.0,
                value=float(current_tau),
                step=0.01,
                help="Manual decision threshold override. Valid range: [0.0, 1.0]. Does not modify saved benchmark artifacts."
            )
        st.warning(
            f"⚠️ **Experimental Threshold Override Active:** Evaluating with tau = **{effective_tau:.4f}** "
            f"(Validation-selected threshold was tau* = **{current_tau:.4f}**). "
            f"This user override does not modify saved validation thresholds or benchmark results."
        )
    else:
        effective_tau = current_tau
        render_frozen_threshold_notice(threshold_val=current_tau, detector_name=current_det_name)

    if current_det_name == "M10_QAT_CD" and m10_meta:
        tau_label = f"Overridden tau: {effective_tau:.4f} (Calibrated tau*: {current_tau:.4f})" if override_threshold else f"Frozen Threshold tau*: {current_tau:.4f}"
        gamma_label = f"Active gamma: {selected_gamma:.2f}" if selected_gamma > 0.0 else f"Calibrated gamma*: {m10_meta.get('calibrated_gamma', 0.0):.2f}"
        st.caption(
            f"Architecture: QAT-CD (Base: BIT-CD 552,097 params, 0 learned quality params) | "
            f"{gamma_label} | {tau_label}"
        )
    elif current_det_name == "M9_BIT_CD" and m9_meta:
        tau_label = f"Overridden tau: {effective_tau:.4f} (Frozen tau*: {current_tau:.4f})" if override_threshold else f"Frozen Threshold tau*: {current_tau:.4f}"
        st.caption(
            f"Architecture: BIT-CD (552,097 parameters) | Best Val F1: {m9_meta.get('best_val_f1', 0.0):.4f} | "
            f"{tau_label}"
        )
    elif current_det_name == "M8_SNUNet" and m8_meta:
        tau_label = f"Overridden tau: {effective_tau:.4f} (Frozen tau*: {current_tau:.4f})" if override_threshold else f"Frozen Threshold tau*: {current_tau:.4f}"
        st.caption(
            f"Architecture: SNUNet-CD (563,377 parameters) | Best Val F1: {m8_meta.get('best_val_f1', 0.0):.4f} | "
            f"{tau_label}"
        )
    elif current_det_name == "M7_FC_Siam_diff" and m7_meta:
        tau_label = f"Overridden tau: {effective_tau:.4f} (Frozen tau*: {current_tau:.4f})" if override_threshold else f"Frozen Threshold tau*: {current_tau:.4f}"
        st.caption(
            f"Architecture: FC-Siam-diff (1,471,409 parameters) | Best Val F1: {m7_meta.get('best_val_f1', 0.0):.4f} | "
            f"{tau_label}"
        )
    else:
        tau_label = f"Overridden tau: {effective_tau:.4f} (Frozen tau*: {current_tau:.4f})" if override_threshold else f"Frozen Threshold tau*: {current_tau:.4f}"
        st.caption(f"Method: {current_det_name} | {tau_label}")

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

    # Apply decision threshold to continuous difference/probability map
    pred_mask = (full_diff >= effective_tau).astype(np.uint8)
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
    # SECTION 2: MILESTONE M11 COMPREHENSIVE RESEARCH EVIDENCE & BENCHMARKS
    # =========================================================================
    st.markdown("---")
    st.markdown("### Milestone M11: Comprehensive Research Evidence & Statistical Benchmarking")
    st.markdown(
        """
        Evaluated on the full **128 LEVIR-CD test pairs** ($134,217,728$ evaluation pixels; $6,837,404$ ground-truth changed pixels). 
        All models operate under standardized CPU benchmarking, identical patch reconstruction pipelines, and strictly frozen validation-calibrated parameters.
        """
    )

    # Mandatory Research Disclosures
    with st.expander("ℹ️ Mandatory Research Disclosures & Scientific Governance", expanded=True):
        st.markdown(
            """
            1. **Test-Set Reuse Disclosure:** The LEVIR-CD test split was inspected and referenced during earlier development milestones; the reported test results reflect standard reproducible benchmark evaluation on the designated partition rather than untouched, independent out-of-distribution generalization.
            2. **Exploratory QAT-CD Active Gating Disclosure:** The active gating configuration ($\gamma = 0.30$) represents an operational trade-off: it eliminates 573,553 false-positive pixels (-29.19% relative to M9 BIT-CD) at the cost of 336,213 true-positive pixels (-9.07%), increasing precision to 70.78% (+5.43 pp) but sacrificing recall (49.30%, -4.92 pp) and unweighted F1 (0.5812 vs 0.5927). It is an operational tool for high-precision scenarios, not an unconditional benchmark improvement.
            3. **SNUNet-CD False Alarms Disclosure:** Under standardized CPU benchmarking with frozen validation thresholding, SNUNet-CD generated **53,485,389 false-positive pixels** (a 39.85% false alarm rate across the test set), highlighting severe over-prediction when deployed out-of-domain.
            """
        )

    tab_bench, tab_ci, tab_paired, tab_abl, tab_figs = st.tabs([
        "Benchmark Comparison",
        "Bootstrap Confidence Intervals",
        "Paired Statistical Tests",
        "QAT-CD Ablation Analysis",
        "Publication Figures Gallery",
    ])

    with tab_bench:
        summary_df = load_m11_summary()
        if summary_df is not None and not summary_df.empty:
            disp_df = summary_df.copy()
            if "method" not in disp_df.columns and "display_name" in disp_df.columns:
                disp_df["method"] = disp_df["display_name"]
            if "mean_pair_latency_ms" not in disp_df.columns and "mean_latency_ms" in disp_df.columns:
                disp_df["mean_pair_latency_ms"] = disp_df["mean_latency_ms"]

            disp_df["Precision (%)"] = disp_df["precision"].apply(lambda v: f"{v*100:.2f}%")
            disp_df["Recall (%)"] = disp_df["recall"].apply(lambda v: f"{v*100:.2f}%")
            disp_df["F1 Score"] = disp_df["f1"].apply(lambda v: f"{v:.4f}")
            disp_df["IoU (%)"] = disp_df["iou"].apply(lambda v: f"{v*100:.2f}%")
            disp_df["Accuracy (%)"] = disp_df["accuracy"].apply(lambda v: f"{v*100:.2f}%")
            disp_df["Mean Latency (ms)"] = disp_df["mean_pair_latency_ms"].apply(lambda v: f"{v:.1f}")
            disp_df["Peak RAM (MB)"] = disp_df["peak_rss_mb"].apply(lambda v: f"{v:.1f}")
            disp_df["Threshold tau"] = disp_df["threshold_value"].apply(lambda v: f"{v:.4f}")

            cols_to_show = [
                "display_name",
                "architecture_type",
                "parameters",
                "threshold_source",
                "Threshold tau",
                "Precision (%)",
                "Recall (%)",
                "F1 Score",
                "IoU (%)",
                "Accuracy (%)",
                "Mean Latency (ms)",
                "Peak RAM (MB)",
            ]
            valid_cols = [c for c in cols_to_show if c in disp_df.columns]
            st.dataframe(disp_df[valid_cols], use_container_width=True, hide_index=True)

            st.pyplot(plot_m5_detector_comparison(disp_df))
        else:
            st.warning("Benchmark summary artifact not found at experiments/results/m11/benchmark_summary.json.")

    with tab_ci:
        st.markdown("#### 95% Confidence Intervals via Scene-Level Bootstrap")
        st.caption(
            "Computed from B = 1,000 bootstrap resamples of the N = 128 LEVIR-CD test scenes. "
            "Reported values represent empirical 2.5th and 97.5th percentiles of scene-level aggregations."
        )
        ci_dict = load_m11_bootstrap_ci()
        if ci_dict:
            ci_rows = []
            for model_key, data in ci_dict.items():
                for metric_name in ["precision", "recall", "f1", "iou", "macro_f1"]:
                    if metric_name in data:
                        m = data[metric_name]
                        ci_rows.append({
                            "Model": model_key.replace("_", " "),
                            "Metric": metric_name.upper().replace("_", "-"),
                            "Mean": round(float(m.get("mean", 0.0)), 4),
                            "95% CI Lower": round(float(m.get("ci_lower", 0.0)), 4),
                            "95% CI Upper": round(float(m.get("ci_upper", 0.0)), 4),
                        })
            if ci_rows:
                df_ci = pd.DataFrame(ci_rows)
                st.dataframe(df_ci, use_container_width=True, hide_index=True)
            else:
                st.warning("Bootstrap confidence interval records are empty.")
        else:
            st.warning("Bootstrap confidence interval artifact not found at experiments/results/m11/statistical_bootstrap_ci.json.")

    with tab_paired:
        st.markdown("#### Paired Wilcoxon Signed-Rank Hypothesis Tests")
        st.caption(
            "Two-sided non-parametric paired tests evaluated across the matched N = 128 test scenes. "
            "Significance threshold alpha = 0.05. Tests whether performance differences between model pairs are statistically significant."
        )
        paired_list = load_m11_paired_tests()
        if paired_list:
            p_rows = []
            for item in paired_list:
                p_rows.append({
                    "Model A": item.get("model_a", "").replace("_", " "),
                    "Model B": item.get("model_b", "").replace("_", " "),
                    "N Scenes": item.get("n_scenes", 128),
                    "Mean F1 (A)": round(float(item.get("mean_f1_a", 0.0)), 4),
                    "Mean F1 (B)": round(float(item.get("mean_f1_b", 0.0)), 4),
                    "F1 Delta (A - B)": f"{float(item.get('f1_delta_mean', 0.0)):+.4f}",
                    "Wilcoxon W": round(float(item.get("wilcoxon_stat_f1", 0.0)), 1),
                    "p-value": f"{float(item.get('wilcoxon_pval_f1', 0.0)):.2e}",
                    "Significant (p < 0.05)": "Yes" if item.get("significant_at_05_f1") else "No",
                })
            if p_rows:
                df_p = pd.DataFrame(p_rows)
                st.dataframe(df_p, use_container_width=True, hide_index=True)
            else:
                st.warning("Paired test records are empty.")
        else:
            st.warning("Paired hypothesis test artifact not found at experiments/results/m11/paired_statistical_tests.json.")

    with tab_abl:
        st.markdown("#### Quality-Aware Gating (QAT-CD) Ablation Study")
        st.caption(
            "Evaluation of individual defect discrepancy signals (Registration, Illumination, Sharpness) and "
            "active gating strength against the baseline BIT-CD (gamma = 0.0) on 128 LEVIR-CD test pairs (134.2M pixels)."
        )
        abl_dict = load_m11_ablation()
        if abl_dict and "ablation_conditions" in abl_dict:
            abl_rows = []
            for cond_name, cdata in abl_dict["ablation_conditions"].items():
                m = cdata.get("metrics", {})
                abl_rows.append({
                    "Ablation Condition": cond_name.replace("_", " "),
                    "Precision": f"{float(m.get('precision', 0.0))*100:.2f}%",
                    "Recall": f"{float(m.get('recall', 0.0))*100:.2f}%",
                    "F1 Score": f"{float(m.get('f1', 0.0)):.4f}",
                    "IoU": f"{float(m.get('iou', 0.0))*100:.2f}%",
                    "FP Reduction (px)": f"{int(cdata.get('fp_reduction_pixels', 0)):,}",
                    "FP Reduction (%)": f"{float(cdata.get('fp_reduction_percent', 0.0)):.2f}%",
                    "TP Loss (px)": f"{int(cdata.get('tp_loss_pixels', 0)):,}",
                    "TP Loss (%)": f"{float(cdata.get('tp_loss_percent', 0.0)):.2f}%",
                    "Benefit Ratio (FP/TP)": f"{float(cdata.get('operational_benefit_ratio', 0.0)):.2f}x",
                })
            df_abl = pd.DataFrame(abl_rows)
            st.dataframe(df_abl, use_container_width=True, hide_index=True)

            interp = abl_dict.get("operational_interpretation", "")
            if interp:
                st.info(f"**Operational Analysis:** {interp}")
        else:
            st.warning("QAT ablation artifact not found at experiments/results/m11/qat_ablation_analysis.json.")

    with tab_figs:
        st.markdown("#### Publication-Grade Empirical Research Figures")
        m11_figs = get_m11_figures()
        if m11_figs:
            captions_map = {
                "f1_iou_benchmark_bars": "Figure 1: Comparative Test Benchmark (F1 score and IoU across all 7 evaluated detectors on LEVIR-CD test split).",
                "cpu_latency_vs_f1": "Figure 2: CPU Latency vs. F1 Trade-off on Commodity Hardware (ms per 1024x1024 pair on CPU).",
                "precision_recall_tradeoff": "Figure 3: Precision-Recall Trade-off Curves across classical baselines and learned architectures.",
                "false_positive_reduction_log": "Figure 4: False-Alarm Magnitude (Log-Scale False Positive Pixels), showing SNUNet-CD false alarms vs. learned models.",
                "per_scene_f1_distribution": "Figure 5: Scene-Level F1 Score Distribution boxplots across the 128 LEVIR-CD test pairs.",
                "qat_ablation_tradeoff": "Figure 6: Quality Gating Trade-Off: False Positive Reduction vs. True Positive Loss across ablation conditions.",
                "cross_model_robustness": "Figure 7: Non-Ground Perturbation Sensitivity under illumination, blur, and registration shift perturbations.",
            }
            fig_options = list(m11_figs.keys())
            selected_fig_key = st.selectbox(
                "Select Milestone M11 Publication Figure:",
                fig_options,
                format_func=lambda k: captions_map.get(k, k.replace("_", " ").title()),
                index=0,
            )
            fig_path = m11_figs[selected_fig_key]
            display_research_figure(
                fig_path,
                caption=captions_map.get(selected_fig_key, fig_path.name)
            )
        else:
            st.warning("Publication figures not found in experiments/figures/m11/.")

        # Preserved M5 Historical Figures Expander
        with st.expander("Historical Milestone 5 Precomputed Figures (Threshold Curves & Baselines)", expanded=False):
            m5_figs = get_m5_figures()
            if m5_figs:
                m5_sub_curves, m5_sub_samples = st.tabs(["Validation Threshold Curves", "Qualitative Comparisons"])
                with m5_sub_curves:
                    thresh_fig = m5_figs.get("validation_threshold_curves")
                    if thresh_fig:
                        display_research_figure(
                            thresh_fig,
                            caption="Validation Sweep Curves across B1, B2, and B3 showing optimal F1 thresholds calibrated on validation pairs."
                        )
                with m5_sub_samples:
                    q_options = [k for k in m5_figs.keys() if "qualitative_comparison" in k]
                    if q_options:
                        q_sel = st.selectbox("Select Qualitative Research Figure (M5):", q_options, index=0)
                        display_research_figure(
                            m5_figs[q_sel],
                            caption=f"Deterministic Qualitative Research Comparison ({q_sel})."
                        )


if __name__ == "__main__":
    render_change_detection_page()
