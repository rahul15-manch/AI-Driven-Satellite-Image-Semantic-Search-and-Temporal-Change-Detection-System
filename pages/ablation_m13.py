"""Page 6: Milestone 13 - Master System Integration, Controlled Ablation Studies & Reproducibility.

Presents Workstreams B (Retrieval Ablation & Leakage Audit), C (Unified Change Detection & Pareto Frontier),
D (QAT-CD Quality Gating & Sensitivity Sweep), and E (Statistical Significance & Scene-Level Resampling).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from ui.components import (
    render_header,
    render_hardware_constraint_badge,
    render_metric_card,
)
from ui.charts import display_research_figure
from utils.result_loader import (
    load_m13_retrieval_ablation,
    load_m13_cd_comparison,
    load_m13_qat_ablation,
    load_m13_statistical_significance,
    get_m13_figures,
)
from utils.config_loader import get_project_root

ROOT = get_project_root()


def render_m13_ablation_page():
    """Renders the comprehensive Milestone 13 Master Integration & Ablation dashboard."""
    render_header(
        title="Master Integration & Ablation Studies (M13)",
        subtitle="End-to-End System Evaluation, Information-Theoretic Leakage Audit, QAT-CD Gating Trade-Offs & Publication Figures",
        badge_text="Milestone M13 End-to-End Verified",
    )

    render_hardware_constraint_badge()

    # Introduction & Research Summary
    st.markdown(
        """
        **Milestone 13 (M13)** provides the final scientific synthesis and ablation package for the 
        AI-Driven Satellite Image Semantic Search and Bi-Temporal Change Detection System. M13 reconciles 
        retrieval protocol boundaries, unifies all 8 change detection architectures into an empirical Pareto frontier, 
        evaluates active quality gating operational trade-offs, and validates experimental rigor via scene-level hypothesis testing.
        """
    )

    # Top KPI Metrics Cards
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_metric_card(
            label="Peak Cross-Modal F1",
            value="0.6035",
            delta="FC-Siam-diff (CPU)",
            help_text="Highest balanced F1 achieved on 128 LEVIR-CD test pairs.",
        )
    with kpi_cols[1]:
        render_metric_card(
            label="Active Gating Benefit Ratio",
            value="1.71 FP / TP",
            delta="+5.43 pp Precision",
            help_text="1.71 false-positive pixels eliminated per 1.0 true-positive pixel conceded.",
        )
    with kpi_cols[2]:
        render_metric_card(
            label="Statistical Resampling",
            value="N = 128 Scenes",
            delta="1,000 Bootstrap Replicates",
            help_text="Evaluated at scene level to prevent spatial autocorrelation p-value deflation.",
        )
    with kpi_cols[3]:
        render_metric_card(
            label="Max CPU Memory (RSS)",
            value="877.3 MB",
            delta="Budget <= 8,192 MB",
            help_text="Peak resident set size observed across all benchmarks.",
        )

    # 5 Tab Navigation
    tab_retrieval, tab_cd, tab_qat, tab_stats, tab_figures = st.tabs([
        "Workstream B: Retrieval Ablation & Leakage Audit",
        "Workstream C: Unified CD & Pareto Frontier",
        "Workstream D: QAT-CD Quality Gating Ablation",
        "Workstream E: Statistical Significance",
        "Publication Figures Gallery",
    ])

    # =========================================================================
    # TAB 1: WORKSTREAM B - SEMANTIC RETRIEVAL ABLATION
    # =========================================================================
    with tab_retrieval:
        st.markdown("### Cross-Modal Semantic Retrieval Protocol Audit & Ablation")
        st.markdown(
            """
            Prior development phases documented an apparent paradox: lexical BM25 (LOCO R@1 = 42.12%) 
            outperformed multimodal CLIP ViT-B/32 (R@1 = 5.45%). Milestone 13 formally investigates the 
            **asymmetric information boundaries** between lexical and multimodal evaluation protocols.
            """
        )

        with st.expander("ℹ️ Root-Cause Investigation: Sibling Caption Lexical Overlap vs Zero-Shot Multimodal Retrieval", expanded=True):
            st.markdown(
                """
                1. **Leave-One-Caption-Out (LOCO) BM25 Leakage Mechanism:**  
                   In RSICD, each 224x224 image has 5 human captions. When a query caption is tested under LOCO, it is held out, 
                   but the **4 remaining human captions for that identical scene remain in the candidate candidate pool**. 
                   Crowd-workers repeatedly reused distinct lexical tokens (e.g., *'many airplanes parked at terminal gates'*). 
                   BM25 easily identifies the matching image by matching overlapping text between sibling human captions, yielding an inflated **R@1 = 42.12%**.
                2. **Category Metadata Control Protocol:**  
                   When all human captions are purged from the candidate gallery and only coarse category labels are indexed 
                   (e.g., *'airport'*, *'parking'*), BM25 lexical recall collapses to **R@1 = 1.50%** (MRR = 0.0618).
                3. **Multimodal CLIP Zero-Shot Integrity:**  
                   OpenAI CLIP ViT-B/32 searches directly against **512-dimensional visual image embeddings with zero human text**. 
                   It translates free-form natural language into visual semantic space, achieving **R@1 = 5.45%** and **R@10 = 27.89%** 
                   with pure cross-modal transfer.
                4. **Formal Scope Discrepancy Disclosure (CLIP + Metadata Reranking):**  
                   Early conceptual roadmaps envisioned a two-stage *'CLIP + Metadata Reranking'* pipeline. 
                   A complete codebase forensic audit confirmed that this method was **never implemented or benchmarked**; 
                   Milestone M7 was executed strictly as learned change detection (FC-Siam-diff). This discrepancy is formally 
                   disclosed in scientific governance reports and marked **NOT IMPLEMENTED**.
                """
            )

        df_ret, doc_ret = load_m13_retrieval_ablation()
        if df_ret is not None and not df_ret.empty:
            st.markdown("#### Retrieval Ablation Summary Table")
            disp_ret = df_ret.copy()
            disp_ret["R@1 (%)"] = disp_ret["R@1"].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "N/A")
            disp_ret["R@5 (%)"] = disp_ret["R@5"].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "N/A")
            disp_ret["R@10 (%)"] = disp_ret["R@10"].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "N/A")
            disp_ret["MRR"] = disp_ret["MRR"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "N/A")
            disp_ret["CPU Latency (ms)"] = disp_ret["mean_latency_ms"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
            disp_ret["Peak RAM (MB)"] = disp_ret["peak_rss_mb"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A")

            cols_to_show = [
                "display_name",
                "modality_type",
                "protocol",
                "R@1 (%)",
                "R@5 (%)",
                "R@10 (%)",
                "MRR",
                "CPU Latency (ms)",
                "Peak RAM (MB)",
                "implementation_status",
            ]
            st.dataframe(disp_ret[cols_to_show], hide_index=True)

        m13_figs = get_m13_figures()
        if "m13_retrieval_tradeoff" in m13_figs:
            st.markdown("#### Publication Figure: Semantic Retrieval Trade-Off (Recall vs CPU Latency)")
            display_research_figure(
                m13_figs["m13_retrieval_tradeoff"],
                caption="Figure M13-1: Retrieval trade-off illustrating asymmetric information boundaries, prompt ensemble scaling, and CPU latency across protocols.",
            )

    # =========================================================================
    # TAB 2: WORKSTREAM C - UNIFIED CHANGE DETECTION COMPARISON
    # =========================================================================
    with tab_cd:
        st.markdown("### Unified Bi-Temporal Change Detection Benchmark & Pareto Frontier")
        st.markdown(
            """
            Milestone 13 synthesizes all 8 evaluated change detection configurations on the standardized 
            **128 LEVIR-CD test pairs** (134,217,728 total pixels; 6,837,404 changed pixels). All methods 
            utilize strictly frozen validation-calibrated parameters and CPU-only inference.
            """
        )

        df_cd = load_m13_cd_comparison()
        if df_cd is not None and not df_cd.empty:
            disp_cd = df_cd.copy()
            disp_cd["Precision (%)"] = disp_cd["precision"].apply(lambda v: f"{v*100:.2f}%")
            disp_cd["Recall (%)"] = disp_cd["recall"].apply(lambda v: f"{v*100:.2f}%")
            disp_cd["F1 Score"] = disp_cd["f1"].apply(lambda v: f"{v:.4f}")
            disp_cd["IoU (%)"] = disp_cd["iou"].apply(lambda v: f"{v*100:.2f}%")
            disp_cd["Accuracy (%)"] = disp_cd["accuracy"].apply(lambda v: f"{v*100:.2f}%")
            disp_cd["Latency (ms/pair)"] = disp_cd["mean_latency_ms"].apply(lambda v: f"{v:.1f}")
            disp_cd["Peak RAM (MB)"] = disp_cd["peak_rss_mb"].apply(lambda v: f"{v:.1f}")
            disp_cd["Threshold tau"] = disp_cd["threshold_value"].apply(lambda v: f"{v:.4f}")
            disp_cd["Parameters"] = disp_cd["parameters"].apply(lambda v: f"{v:,}" if v > 0 else "0 (Classical)")

            show_cols = [
                "display_name",
                "architecture_type",
                "Precision (%)",
                "Recall (%)",
                "F1 Score",
                "IoU (%)",
                "Accuracy (%)",
                "Latency (ms/pair)",
                "Peak RAM (MB)",
                "Parameters",
                "Threshold tau",
            ]
            st.dataframe(disp_cd[show_cols], hide_index=True)

            # Comparative Insights
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(
                    "**Classical vs. Learned Gap:**  \n"
                    "Classical baselines (B1, B2, B3) operate at ultra-low latency (12–47 ms/pair) but yield poor F1 (0.125–0.132) "
                    "due to excessive false alarms (up to 48.0M FP pixels)."
                )
            with c2:
                st.warning(
                    "**SNUNet-CD Vulnerability:**  \n"
                    "Despite high recall (77.58%), SNUNet-CD generated **53.1M false-positive pixels** (39.56% error rate), "
                    "collapsing precision to 9.08% and F1 to 0.1626."
                )
            with c3:
                st.success(
                    "**Pareto Frontier Champions:**  \n"
                    "**FC-Siam-diff** achieves the highest raw F1 (0.6035) at 603 ms CPU latency.  \n"
                    "**QAT-CD Active Gating** achieves the highest precision (70.78%) with a 29.19% FP suppression."
                )

        if "m13_cd_pareto_frontier" in m13_figs:
            st.markdown("#### Publication Figure: Change Detection Pareto Frontier (F1 vs CPU Inference Latency)")
            display_research_figure(
                m13_figs["m13_cd_pareto_frontier"],
                caption="Figure M13-2: Empirical Pareto frontier contrasting F1 detection quality against single-pair CPU runtime on commodity hardware.",
            )

    # =========================================================================
    # TAB 3: WORKSTREAM D - QAT-CD ABLATION & QUALITY GATING
    # =========================================================================
    with tab_qat:
        st.markdown("### Controlled Quality-Aware Change Detection (QAT-CD) Gating Ablation")
        st.markdown(
            """
            QAT-CD introduces an explicit, analytical quality-attenuation mechanism designed to suppress 
            false alarms induced by sub-pixel misregistration, illumination fluctuations, and optical blur.
            """
        )

        st.markdown(
            r"""
            #### Mathematical Formulation
            The quality-attenuated posterior change probability $P_{\text{QAT}}(x, y)$ is computed as:
            $$P_{\text{QAT}}(x, y) = \text{clip}\left(P_0(x, y) \cdot \left[1 - \gamma \cdot U_{\text{det}}(x, y) \cdot Q_{\text{defect}}(x, y)\right],\, 0.0,\, 1.0\right)$$
            Where:
            - **Detector Uncertainty:** $U_{\text{det}}(x, y) = 4 \cdot P_0(x, y) \cdot (1 - P_0(x, y)) \in [0, 1]$, maximizing suppression strictly on ambiguous decision boundaries while leaving confident predictions unaltered ($U_{\text{det}}=0$ at $P=0$ and $P=1$).
            - **Composite Defect Index:** $Q_{\text{defect}}(x, y) = 0.34 \cdot S_{\text{reg}}(x, y) + 0.33 \cdot S_{\text{illum}}(x, y) + 0.33 \cdot S_{\text{sharp}}(x, y)$.
            - **Attenuating Signals:** $S_{\text{reg}}$ (Sobel gradient magnitude ratio), $S_{\text{illum}}$ (box-filtered luminance disparity), $S_{\text{sharp}}$ (Laplacian energy disparity).
            """
        )

        df_qat, doc_qat = load_m13_qat_ablation()
        if df_qat is not None and not df_qat.empty:
            st.markdown("#### Controlled Ablation Matrix (Test Partition)")
            disp_qat = df_qat.copy()
            disp_qat["Precision (%)"] = disp_qat["precision"].apply(lambda v: f"{v*100:.2f}%")
            disp_qat["Recall (%)"] = disp_qat["recall"].apply(lambda v: f"{v*100:.2f}%")
            disp_qat["F1 Score"] = disp_qat["f1"].apply(lambda v: f"{v:.4f}")
            disp_qat["IoU (%)"] = disp_qat["iou"].apply(lambda v: f"{v*100:.2f}%")
            disp_qat["FP Suppressed"] = disp_qat["fp_reduction_pixels"].apply(lambda v: f"{v:,} ({disp_qat.loc[disp_qat['fp_reduction_pixels']==v, 'fp_reduction_percent'].values[0]:.2f}%)" if v > 0 else "Baseline")
            disp_qat["TP Conceded"] = disp_qat["tp_loss_pixels"].apply(lambda v: f"{v:,} ({disp_qat.loc[disp_qat['tp_loss_pixels']==v, 'tp_loss_percent'].values[0]:.2f}%)" if v > 0 else "Baseline")
            disp_qat["Benefit Ratio"] = disp_qat["operational_benefit_ratio"].apply(lambda v: f"{v:.2f} FP/TP" if v > 0 else "N/A")

            qat_cols = [
                "condition",
                "Precision (%)",
                "Recall (%)",
                "F1 Score",
                "IoU (%)",
                "FP Suppressed",
                "TP Conceded",
                "Benefit Ratio",
            ]
            st.dataframe(disp_qat[qat_cols], hide_index=True)

        if doc_qat and "operational_tradeoff_summary" in doc_qat:
            tradeoff = doc_qat["operational_tradeoff_summary"]
            st.markdown(
                f"""
                <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-left: 5px solid #2F3E32; padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin: 1rem 0; font-size: 0.9rem; color: #3E3B39;">
                    <strong style="color: #4A2E2B; font-size: 1rem;">Operational Benefit Ratio: {tradeoff['operational_benefit_ratio']:.4f} FP Suppressed per TP Conceded</strong><br>
                    Under nominal gating (&gamma; = 0.30, &tau; = 0.45), QAT-CD suppresses <strong>{tradeoff['fp_pixels_eliminated']:,} false-positive pixels (-{tradeoff['fp_percent_reduction']:.2f}%)</strong> 
                    while conceding <strong>{tradeoff['tp_pixels_conceded']:,} true-positive pixels (-{tradeoff['tp_percent_conceded']:.2f}%)</strong>, 
                    driving test precision to <strong>70.78% (+{tradeoff['precision_gain_pp']:.2f} pp)</strong>. 
                    In high-consequence satellite surveillance workflows where analyst false alarm fatigue is paramount, active gating offers clear operational utility.
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_w, col_g = st.columns(2)
        with col_w:
            if "m13_qat_ablation_waterfall" in m13_figs:
                display_research_figure(
                    m13_figs["m13_qat_ablation_waterfall"],
                    caption="Figure M13-3: Component ablation waterfall illustrating false-positive reduction across defect signals.",
                )
        with col_g:
            if "m13_gamma_sensitivity_curve" in m13_figs:
                display_research_figure(
                    m13_figs["m13_gamma_sensitivity_curve"],
                    caption="Figure M13-4: Quality attenuation parameter gamma sensitivity sweep on validation calibration.",
                )

    # =========================================================================
    # TAB 4: WORKSTREAM E - STATISTICAL SIGNIFICANCE & RESAMPLING
    # =========================================================================
    with tab_stats:
        st.markdown("### Scene-Level Hypothesis Testing & Non-Parametric Bootstrap")
        st.markdown(
            """
            Remote sensing pixels exhibit severe spatial autocorrelation: neighboring pixels within an optical tile 
            are non-independent. To uphold statistical validity, Milestone 13 conducts all significance hypothesis testing 
            and bootstrap resampling strictly at the **scene level** ($N=128$ matched 1024x1024 image pairs).
            """
        )

        stat_data = load_m13_statistical_significance()
        if stat_data:
            paired_tests = stat_data.get("paired_tests", [])
            bootstrap_ci = stat_data.get("bootstrap_confidence_intervals", {})

            st.markdown("#### Scene-Level Paired Wilcoxon Signed-Rank Hypothesis Tests (N = 128 Pairs)")
            if paired_tests:
                df_paired = pd.DataFrame(paired_tests)
                disp_paired = df_paired.copy()
                disp_paired["Mean Difference"] = disp_paired["mean_diff"].apply(lambda v: f"{v:+.4f}")
                disp_paired["Median Difference"] = disp_paired["median_diff"].apply(lambda v: f"{v:+.4f}")
                disp_paired["W Statistic"] = disp_paired["statistic"].apply(lambda v: f"{v:.1f}")
                disp_paired["p-value"] = disp_paired["p_value"].apply(lambda v: f"{v:.4e}" if v < 1e-4 else f"{v:.4f}")
                disp_paired["Significant (alpha=0.05)"] = disp_paired["significant_alpha_0_05"].apply(lambda v: "✓ Significant" if v else "Not Significant")

                p_cols = ["comparison", "metric", "Mean Difference", "Median Difference", "W Statistic", "p-value", "Significant (alpha=0.05)"]
                st.dataframe(disp_paired[p_cols], hide_index=True)

            st.markdown("#### 1,000-Replicate Bootstrap 95% Confidence Intervals (Scene-Level Resampling)")
            if bootstrap_ci:
                ci_rows = []
                for model_key, m_info in bootstrap_ci.items():
                    m_name = m_info.get("display_name", model_key)
                    for metric_key, c_vals in m_info.get("metrics", {}).items():
                        ci_rows.append({
                            "Model": m_name,
                            "Metric": metric_key.upper(),
                            "Mean Estimate": f"{c_vals.get('mean', 0.0):.4f}",
                            "95% CI Lower": f"{c_vals.get('ci_lower_2_5', 0.0):.4f}",
                            "95% CI Upper": f"{c_vals.get('ci_upper_97_5', 0.0):.4f}",
                            "CI Width": f"{(c_vals.get('ci_upper_97_5', 0.0) - c_vals.get('ci_lower_2_5', 0.0)):.4f}",
                        })
                df_ci = pd.DataFrame(ci_rows)
                st.dataframe(df_ci, hide_index=True)

    # =========================================================================
    # TAB 5: PUBLICATION RESEARCH FIGURES GALLERY
    # =========================================================================
    with tab_figures:
        st.markdown("### High-Resolution Publication Research Figures (Milestone 13)")
        st.markdown(
            """
            All 4 figures are generated at 300 DPI directly from machine-readable benchmark outputs 
            using the standardized Earth-inspired research color system (Forest Green, Terracotta, Yellow Ochre, Deep Bark, Alabaster).
            """
        )

        fig_descs = [
            (
                "m13_retrieval_tradeoff",
                "M13 Figure 1: Semantic Retrieval Trade-Off (Recall vs CPU Latency)",
                "Contrasts cross-modal zero-shot CLIP ViT-B/32 against lexical BM25 (LOCO vs Metadata Control vs Diagnostic Leaky), highlighting asymmetric information boundaries and prompt ensemble compute overhead.",
            ),
            (
                "m13_cd_pareto_frontier",
                "M13 Figure 2: Change Detection Pareto Frontier (F1 Score vs CPU Inference Latency)",
                "Maps all 8 change detection methods on single-pair CPU runtime, illustrating the Pareto boundary formed by classical baselines, FC-Siam-diff, and BIT-CD.",
            ),
            (
                "m13_qat_ablation_waterfall",
                "M13 Figure 3: QAT-CD Component Ablation Waterfall Analysis",
                "Step-by-step pixel error attribution demonstrating false-positive suppression contributed by misregistration, illumination disparity, and optical blur signals.",
            ),
            (
                "m13_gamma_sensitivity_curve",
                "M13 Figure 4: Quality Attenuation Parameter Sensitivity Curve",
                "Validation calibration parameter sweep on gamma across [0.0, 0.6] at frozen tau=0.45, establishing the operational trade-off boundary.",
            ),
        ]

        for stem, title, desc in fig_descs:
            if stem in m13_figs:
                st.markdown(f"#### {title}")
                st.caption(desc)
                display_research_figure(m13_figs[stem], caption=title)
                st.markdown("---")


if __name__ == "__main__":
    render_m13_ablation_page()
