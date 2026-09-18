"""Page 5: Milestone 6 - False-Alarm Analysis & Perturbation Robustness Evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from ui.components import (
    render_header,
    render_hardware_constraint_badge,
    render_metric_card,
)
from ui.charts import plot_m6_f1_vs_severity, display_research_figure
from utils.result_loader import (
    load_m6_results,
    load_m6_summary,
    load_m6_degradation_table,
    get_m6_figures,
)
from utils.config_loader import get_project_root

ROOT = get_project_root()


def render_false_alarm_page():
    """Renders the False-Alarm & Perturbation Robustness page."""
    render_header(
        title="False-Alarm & Robustness Analysis",
        subtitle="Empirical Vulnerability Audit of Classical Detectors Under Controlled Non-Ground Visual Perturbations",
        badge_text="Milestone 6 Implemented",
    )

    render_hardware_constraint_badge()

    # Introduction
    st.markdown(
        """
        Milestone 6 systematically evaluates how classical bi-temporal change detectors (B1 Pixel Diff, B2 SSIM, B3 CVA) 
        respond to non-ground visual variations using strictly **frozen M5 decision thresholds** 
        ($\\tau^*_{\\text{B1}}=0.4100, \\tau^*_{\\text{B2}}=0.9000, \\tau^*_{\\text{B3}}=0.4050$) across all $134,217,728$ evaluation pixels.
        """
    )

    m6_df = load_m6_results()
    if m6_df is None:
        st.error("Milestone 6 results not found at experiments/results/m6/m6_results.csv.")
        return

    # =========================================================================
    # SECTION 1: PERTURBATION CONDITION SELECTOR
    # =========================================================================
    st.markdown("### Interactive Perturbation & Detector Query")
    st.markdown("Select a perturbation family, severity tier, and detector to inspect the official empirical measurements:")

    col_pert, col_sev, col_det = st.columns(3)

    pert_options = {
        "Control (No Perturbation)": "control",
        "Global Illumination Shift": "global_illumination_shift",
        "Gaussian Blur (Defocus)": "gaussian_blur",
        "Geometric Misregistration": "geometric_misregistration",
        "Localized Occlusion / Shadow": "localized_occlusion_shadow",
    }

    with col_pert:
        pert_label = st.selectbox("Perturbation Family:", list(pert_options.keys()), index=3)  # Default: Misregistration
        selected_pert = pert_options[pert_label]

    with col_sev:
        if selected_pert == "control":
            sev_options = ["none"]
        else:
            sev_options = ["mild", "medium", "strong"]
        selected_sev = st.selectbox("Severity Tier:", sev_options, index=len(sev_options) - 1)

    with col_det:
        det_options = {
            "B1: Pixel Difference": "B1_Pixel_Diff",
            "B2: SSIM Dissimilarity": "B2_SSIM",
            "B3: Change Vector Analysis": "B3_CVA",
        }
        det_label = st.selectbox("Classical Detector:", list(det_options.keys()), index=1)  # Default: SSIM
        selected_det = det_options[det_label]

    # Query matching row
    match = m6_df[
        (m6_df["perturbation"] == selected_pert)
        & (m6_df["severity"] == selected_sev)
        & (m6_df["detector"] == selected_det)
    ]

    if not match.empty:
        row = match.iloc[0]
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.8rem 0;">
                <strong>Selected Condition:</strong> {det_label} under <strong>{pert_label}</strong> ({selected_sev.upper()}) | 
                Frozen Threshold &tau;* = <code>{row['threshold']:.4f}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Precision", f"{row['precision']*100:.2f}%")
        with m2:
            render_metric_card("Recall", f"{row['recall']*100:.2f}%")
        with m3:
            f1_deg = row["relative_f1_degradation_pct"]
            deg_label = f"{f1_deg:+.2f}%" if selected_pert != "control" else "Baseline"
            render_metric_card("F1 Score", f"{row['f1']:.4f}", delta=deg_label, delta_color="inverse" if f1_deg > 0 else "normal")
        with m4:
            render_metric_card("IoU", f"{row['iou']*100:.2f}%")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            render_metric_card("Additional False Positives", f"{int(row['additional_false_positives']):+,} px" if selected_pert != "control" else "0 px")
        with r2:
            render_metric_card("Relative FP Increase", f"{row['relative_fp_increase_pct']:+.2f}%" if selected_pert != "control" else "0.00%")
        with r3:
            render_metric_card("Total False Positives", f"{int(row['fp']):,} px")
        with r4:
            render_metric_card("Mean Latency", f"{row['latency_ms_per_pair']:.1f} ms/pair")

    # =========================================================================
    # SECTION 2: OFFICIAL RESEARCH CHARTS & HEATMAP
    # =========================================================================
    st.markdown("---")
    st.markdown("### Official Research Figures (Milestone 6)")

    m6_figs = get_m6_figures()
    fig_tabs = st.tabs([
        "Relative F1 Degradation",
        "Additional False Positives",
        "Detector × Perturbation Heatmap",
        "F1 vs. Severity",
    ])

    with fig_tabs[0]:
        if "relative_f1_degradation" in m6_figs:
            display_research_figure(
                m6_figs["relative_f1_degradation"],
                caption="Relative F1 Degradation (%) across all 4 perturbation families and 3 classical detectors.",
            )

    with fig_tabs[1]:
        if "additional_fp_vs_severity" in m6_figs:
            display_research_figure(
                m6_figs["additional_fp_vs_severity"],
                caption="Additional False Positive Pixels vs. Perturbation Severity across all 13 conditions.",
            )

    with fig_tabs[2]:
        if "detector_perturbation_heatmap" in m6_figs:
            display_research_figure(
                m6_figs["detector_perturbation_heatmap"],
                caption="Detector × Perturbation Sensitivity Matrix summarizing impact across all evaluated conditions.",
            )

    with fig_tabs[3]:
        if "f1_vs_severity" in m6_figs:
            display_research_figure(
                m6_figs["f1_vs_severity"],
                caption="F1 Score vs. Severity across all 3 classical detectors under frozen validation thresholds.",
            )

    # =========================================================================
    # SECTION 3: QUALITATIVE VISUAL COMPARISONS
    # =========================================================================
    st.markdown("---")
    st.markdown("### Qualitative Perturbation Visualizations")
    st.markdown("Inspect precomputed high-resolution qualitative research figures comparing original vs perturbed predictions:")

    qual_figs = {k: v for k, v in m6_figs.items() if k.startswith("qualitative_test_")}
    if qual_figs:
        q_choice = st.selectbox("Select Qualitative Research Figure:", sorted(list(qual_figs.keys())), index=0)
        display_research_figure(
            qual_figs[q_choice],
            caption=f"Official qualitative perturbation analysis figure: {q_choice}.",
        )
    else:
        st.info("No qualitative perturbation figures found in experiments/figures/m6/.")

    # =========================================================================
    # SECTION 4: CORE RESEARCH FINDINGS (SUPPORTED DATA ONLY)
    # =========================================================================
    st.markdown("---")
    st.markdown("### Core Empirical Findings on Classical Vulnerabilities")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #ef4444; border-radius: 6px; padding: 1rem; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #991b1b; font-size: 0.95rem;">
                    1. Misregistration-Induced False Alarm Runaway (B2 SSIM)
                </h5>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    SSIM is extremely sensitive to sub-building translational shifts. A <strong>5-pixel misregistration</strong> produced 
                    <strong>+5,679,832 additional false positive pixels</strong> (+11.83% increase), driving total B2 false alarms 
                    to 53.69M pixels (40.0% of the entire test dataset). Rigid displacement creates wide mismatch bands along all building edges.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #eab308; border-radius: 6px; padding: 1rem; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #854d0e; font-size: 0.95rem;">
                    2. Defocus Recall Collapse (SSIM Structural Blindness)
                </h5>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    Under Gaussian blur (&sigma; = 4.0), SSIM suffered the worst relative F1 degradation in the entire benchmark: 
                    <strong>+12.82% relative degradation</strong> (F1 dropped from 0.1246 to 0.1087). 
                    Blur obliterates high-frequency structural edges, dropping true building recall from 53.32% down to 37.26%.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 1rem; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #1e40af; font-size: 0.95rem;">
                    3. Spectral Cloud Shadow Sensitivity (B1 & B3)
                </h5>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    Localized cloud shadows (10% area, 0.4&times; attenuation) produced 
                    <strong>+1,971,579 additional FP for B1</strong> (+20.61% relative increase) and 
                    <strong>+2,067,512 additional FP for B3</strong> (+20.44% relative increase). 
                    Radiometric drops create spectral difference vectors that cross frozen thresholds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #10b981; border-radius: 6px; padding: 1rem; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #065f46; font-size: 0.95rem;">
                    4. CPU Execution Feasibility & Bitwise Determinism
                </h5>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    The full 13-condition benchmark over 134.2M test pixels completed in <strong>261.2 seconds</strong> on CPU. 
                    Peak resident memory remained strictly at <strong>432.8 MB</strong> (&le; 5.4% of budget). 
                    Repeated independent runs confirmed 100% bitwise identical confusion matrices and metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_false_alarm_page()
