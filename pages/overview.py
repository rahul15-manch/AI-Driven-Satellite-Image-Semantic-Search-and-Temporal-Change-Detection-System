"""Page 1: Project Overview, Research Questions, Pipeline, and Implementation Status."""

from __future__ import annotations

import streamlit as st
from ui.components import (
    render_header,
    render_milestone_status_bar,
    render_hardware_constraint_badge,
)
from utils.result_loader import load_rsicd_splits, load_levir_splits


def render_overview_page():
    """Renders the Home / Overview dashboard page."""
    render_header(
        title="AI-Driven Satellite Image Semantic Search & Temporal Change Detection",
        subtitle="Empirical Investigation of Vision-Language Cross-Modal Retrieval and Classical & Learned Bi-Temporal Change Detection",
        badge_text="Milestones M1–M12 Verified",
    )

    render_milestone_status_bar()
    render_hardware_constraint_badge()

    # Executive Overview
    st.markdown("### Research Objectives")
    st.markdown(
        """
        This project investigates whether natural-language vision-language representations and temporal change detection 
        can be effectively and rigorously deployed under strict **CPU-only commodity hardware constraints** (&le; 8 GB RAM). 
        The system evaluates cross-modal semantic search on uncaptioned satellite imagery alongside classical bi-temporal 
        change detection baselines on high-resolution optical remote-sensing datasets.
        """
    )

    # 4 Research Questions
    st.markdown("### Research Questions & Empirical Status")
    rq_cols = st.columns(2)
    with rq_cols[0]:
        st.markdown(
            """
            <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-radius: 8px; padding: 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(62,59,57,0.06);">
                <h4 style="margin: 0 0 0.5rem 0; color: #4A2E2B; font-size: 1rem; font-weight: 700;">
                    RQ1: Vision-Language vs. Lexical Retrieval (M4)
                </h4>
                <p style="font-size: 0.85rem; color: #3E3B39; margin: 0; line-height: 1.5;">
                    <em>"How effectively can vision-language embeddings retrieve semantically relevant satellite images from natural-language queries, compared with a lexical retrieval baseline, under CPU-only constraints?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 700; color: #6B7A65;">
                    Status: Verified locally across 5,465 queries on 1,093 RSICD test gallery images.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-radius: 8px; padding: 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(62,59,57,0.06);">
                <h4 style="margin: 0 0 0.5rem 0; color: #4A2E2B; font-size: 1rem; font-weight: 700;">
                    RQ3: Non-Ground Perturbation Sensitivity (M6)
                </h4>
                <p style="font-size: 0.85rem; color: #3E3B39; margin: 0; line-height: 1.5;">
                    <em>"How sensitive are classical bi-temporal change-detection methods to non-ground visual variations such as illumination changes, blur, geometric misregistration, and localized occlusion/shadow effects?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 700; color: #6B7A65;">
                    Status: Quantified across 4 perturbation families &times; 3 severities on 134.2M test pixels.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rq_cols[1]:
        st.markdown(
            """
            <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-radius: 8px; padding: 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(62,59,57,0.06);">
                <h4 style="margin: 0 0 0.5rem 0; color: #4A2E2B; font-size: 1rem; font-weight: 700;">
                    RQ2: Classical vs. Learned Change Detection (M5)
                </h4>
                <p style="font-size: 0.85rem; color: #3E3B39; margin: 0; line-height: 1.5;">
                    <em>"How do classical image-comparison techniques and lightweight learned models compare for bi-temporal satellite-image change detection under CPU-only constraints?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 700; color: #6B7A65;">
                    Status: Classical baselines (B1, B2, B3) fully established on 128 LEVIR-CD test pairs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-radius: 8px; padding: 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(62,59,57,0.06);">
                <h4 style="margin: 0 0 0.5rem 0; color: #4A2E2B; font-size: 1rem; font-weight: 700;">
                    RQ4: CPU Feasibility & Pareto Trade-offs (M4-M6)
                </h4>
                <p style="font-size: 0.85rem; color: #3E3B39; margin: 0; line-height: 1.5;">
                    <em>"What are the latency, memory footprint, and retrieval accuracy trade-offs of deploying semantic search and bi-temporal change detection entirely on CPU architectures?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 700; color: #6B7A65;">
                    Status: All operations profiled; peak RAM &le; 884 MB, latencies &le; 152 ms per pair.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Research Pipeline Diagram
    st.markdown("### Implemented Research Pipeline (Milestones 1–13)")
    st.markdown(
        """
        <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(62,59,57,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; text-align: center; gap: 0.8rem;">
                <div style="flex: 1; min-width: 130px; background: #FAF7F2; padding: 0.85rem 0.5rem; border-radius: 6px; border-top: 3px solid #2F3E32; border: 1px solid #D9C5B2; border-top-width: 3px;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #4A2E2B;">1. Datasets (M2)</div>
                    <div style="font-size: 0.72rem; color: #8A7E72;">RSICD & LEVIR-CD</div>
                </div>
                <div style="color: #6B7A65; font-weight: bold; font-size: 1.1rem;">&rarr;</div>
                <div style="flex: 1; min-width: 130px; background: #FAF7F2; padding: 0.85rem 0.5rem; border-radius: 6px; border-top: 3px solid #6B7A65; border: 1px solid #D9C5B2; border-top-width: 3px;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #4A2E2B;">2. Retrieval (M4)</div>
                    <div style="font-size: 0.72rem; color: #8A7E72;">BM25 & CLIP FAISS</div>
                </div>
                <div style="color: #6B7A65; font-weight: bold; font-size: 1.1rem;">&rarr;</div>
                <div style="flex: 1; min-width: 130px; background: #FAF7F2; padding: 0.85rem 0.5rem; border-radius: 6px; border-top: 3px solid #E1A95F; border: 1px solid #D9C5B2; border-top-width: 3px;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #4A2E2B;">3. Classical CD (M5)</div>
                    <div style="font-size: 0.72rem; color: #8A7E72;">B1, B2, B3 Baselines</div>
                </div>
                <div style="color: #6B7A65; font-weight: bold; font-size: 1.1rem;">&rarr;</div>
                <div style="flex: 1; min-width: 130px; background: #FAF7F2; padding: 0.85rem 0.5rem; border-radius: 6px; border-top: 3px solid #C86B4D; border: 1px solid #D9C5B2; border-top-width: 3px;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #4A2E2B;">4. Learned CD (M7-M9)</div>
                    <div style="font-size: 0.72rem; color: #8A7E72;">FC-Siam, SNUNet, BIT</div>
                </div>
                <div style="color: #6B7A65; font-weight: bold; font-size: 1.1rem;">&rarr;</div>
                <div style="flex: 1; min-width: 130px; background: #FAF7F2; padding: 0.85rem 0.5rem; border-radius: 6px; border-top: 3px solid #4A2E2B; border: 1px solid #D9C5B2; border-top-width: 3px;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #4A2E2B;">5. Quality Gating (M10)</div>
                    <div style="font-size: 0.72rem; color: #8A7E72;">QAT-CD Attenuation</div>
                </div>
                <div style="color: #6B7A65; font-weight: bold; font-size: 1.1rem;">&rarr;</div>
                <div style="flex: 1; min-width: 130px; background: #FAF7F2; padding: 0.85rem 0.5rem; border-radius: 6px; border-top: 3px solid #2F3E32; border: 1px solid #D9C5B2; border-top-width: 3px;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #4A2E2B;">6. Integration (M11-13)</div>
                    <div style="font-size: 0.72rem; color: #8A7E72;">Ablation & Parity</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mandatory Scientific Disclosures
    st.markdown("### Mandatory Research Disclosures & Scientific Governance")
    st.markdown(
        """
        <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-left: 4px solid #E1A95F; border-radius: 6px; padding: 1.1rem; margin-bottom: 1.5rem; font-size: 0.88rem; color: #3E3B39; box-shadow: 0 1px 3px rgba(62,59,57,0.06);">
            <strong style="color: #4A2E2B; font-size: 0.95rem;">Critical Research Disclosures:</strong>
            <ol style="margin: 0.5rem 0 0 1.2rem; padding: 0; line-height: 1.6;">
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #2F3E32;">Test-Set Reuse:</strong> The LEVIR-CD test split was inspected and referenced during earlier development milestones; the reported test results reflect standard reproducible benchmark evaluation on the designated partition rather than untouched, independent out-of-distribution generalization.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #C86B4D;">Exploratory QAT-CD Active Gating:</strong> Active gating (&gamma; = 0.30) is an exploratory trade-off configuration: it reduces false-positive pixels by approximately <strong>29.19%</strong> relative to M9 BIT-CD, but sacrifices recall and unweighted F1. It is not an unconditional performance improvement across all operational metrics.
                </li>
                <li>
                    <strong style="color: #4A2E2B;">SNUNet-CD False Alarms:</strong> Under standardized CPU benchmarking with frozen validation thresholding, SNUNet-CD generated <strong>53,485,389 false-positive pixels</strong> (a 39.85% false alarm rate across the test set), demonstrating severe vulnerability to false alarms when evaluated out-of-domain.
                </li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Research Boundaries & Milestone Status
    st.markdown("### Authoritative Milestone Verification Status")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Foundations, Retrieval & Classical Baselines (M1–M6):**
            - **M1:** Problem definition, system architecture & leakage control design.
            - **M2:** RSICD & LEVIR-CD acquisition, verification audits & patching pipeline.
            - **M3:** Remote-sensing literature review, baseline registry & evaluation framework.
            - **M4:** Leakage-controlled BM25 (LOCO & Metadata), CLIP ViT-B/32, and Prompt Ensemble on CPU.
            - **M5:** Classical bi-temporal baselines (B1 Pixel Diff, B2 SSIM, B3 CVA) with frozen validation thresholds.
            - **M6:** False-alarm diagnostic framework & 4 controlled non-ground perturbation families.
            """
        )
    with col_b:
        st.markdown(
            """
            **Learned Models, Quality Gating & Evaluation (M7–M12):**
            - **M7:** Learned Siamese U-Net (FC-Siam-diff) with frozen validation thresholding.
            - **M8:** Advanced Dense Siamese Network (SNUNet-CD) with ECAM channel attention on CPU.
            - **M9:** Attention-based Bitemporal Image Transformer (BIT-CD) with cross-attention token fusion.
            - **M10:** Quality-Aware Temporal Change Detection (QAT-CD) with registration, illumination, and blur gating.
            - **M11:** Standardized comparative evaluation, bootstrap confidence intervals, paired hypothesis tests, and ablation analysis.
            - **M12:** FastAPI backend and dual-interface frontend parity (Streamlit dashboard & Web UI).
            """
        )


if __name__ == "__main__":
    render_overview_page()
