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
        subtitle="Empirical Investigation of Vision-Language Cross-Modal Retrieval and Classical Bi-Temporal Change Detection",
        badge_text="Implemented through Milestone 6",
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
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1rem;">
                    RQ1: Vision-Language vs. Lexical Retrieval (M4)
                </h4>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    <em>"How effectively can vision-language embeddings retrieve semantically relevant satellite images from natural-language queries, compared with a lexical retrieval baseline, under CPU-only constraints?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 600; color: #0369a1;">
                    Status: Verified locally across 5,465 queries on 1,093 RSICD test gallery images.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1rem;">
                    RQ3: Non-Ground Perturbation Sensitivity (M6)
                </h4>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    <em>"How sensitive are classical bi-temporal change-detection methods to non-ground visual variations such as illumination changes, blur, geometric misregistration, and localized occlusion/shadow effects?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 600; color: #0369a1;">
                    Status: Quantified across 4 perturbation families &times; 3 severities on 134.2M test pixels.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rq_cols[1]:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1rem;">
                    RQ2: Classical vs. Learned Change Detection (M5)
                </h4>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    <em>"How do classical image-comparison techniques and lightweight learned models compare for bi-temporal satellite-image change detection under CPU-only constraints?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 600; color: #0369a1;">
                    Status: Classical baselines (B1, B2, B3) fully established on 128 LEVIR-CD test pairs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1rem;">
                    RQ4: CPU Feasibility & Pareto Trade-offs (M4-M6)
                </h4>
                <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                    <em>"What are the latency, memory footprint, and retrieval accuracy trade-offs of deploying semantic search and bi-temporal change detection entirely on CPU architectures?"</em>
                </p>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; font-weight: 600; color: #0369a1;">
                    Status: All operations profiled; peak RAM &le; 884 MB, latencies &le; 152 ms per pair.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Research Pipeline Diagram
    st.markdown("### Implemented Research Pipeline (Milestones 1–6)")
    st.markdown(
        """
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; text-align: center; gap: 0.8rem;">
                <div style="flex: 1; min-width: 140px; background: #f1f5f9; padding: 0.8rem 0.5rem; border-radius: 6px; border-top: 3px solid #3b82f6;">
                    <div style="font-weight: 700; font-size: 0.85rem; color: #1e293b;">1. Datasets (M2)</div>
                    <div style="font-size: 0.75rem; color: #64748b;">RSICD & LEVIR-CD</div>
                </div>
                <div style="color: #94a3b8; font-weight: bold; font-size: 1.2rem;">&rarr;</div>
                <div style="flex: 1; min-width: 140px; background: #f1f5f9; padding: 0.8rem 0.5rem; border-radius: 6px; border-top: 3px solid #3b82f6;">
                    <div style="font-weight: 700; font-size: 0.85rem; color: #1e293b;">2. Pipeline (M2)</div>
                    <div style="font-size: 0.75rem; color: #64748b;">256&times;256 Patching</div>
                </div>
                <div style="color: #94a3b8; font-weight: bold; font-size: 1.2rem;">&rarr;</div>
                <div style="flex: 1; min-width: 140px; background: #f1f5f9; padding: 0.8rem 0.5rem; border-radius: 6px; border-top: 3px solid #3b82f6;">
                    <div style="font-weight: 700; font-size: 0.85rem; color: #1e293b;">3. Retrieval (M4)</div>
                    <div style="font-size: 0.75rem; color: #64748b;">BM25 & CLIP FAISS</div>
                </div>
                <div style="color: #94a3b8; font-weight: bold; font-size: 1.2rem;">&rarr;</div>
                <div style="flex: 1; min-width: 140px; background: #f1f5f9; padding: 0.8rem 0.5rem; border-radius: 6px; border-top: 3px solid #3b82f6;">
                    <div style="font-weight: 700; font-size: 0.85rem; color: #1e293b;">4. Change Det. (M5)</div>
                    <div style="font-size: 0.75rem; color: #64748b;">B1, B2, B3 Baselines</div>
                </div>
                <div style="color: #94a3b8; font-weight: bold; font-size: 1.2rem;">&rarr;</div>
                <div style="flex: 1; min-width: 140px; background: #f1f5f9; padding: 0.8rem 0.5rem; border-radius: 6px; border-top: 3px solid #3b82f6;">
                    <div style="font-weight: 700; font-size: 0.85rem; color: #1e293b;">5. Robustness (M6)</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Perturbation Audit</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Research Boundaries
    st.markdown("### Strict Milestone Governance & Boundaries")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Fully Implemented in Current System:**
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
            **Planned / Not Implemented in Current Frontend:**
            - *M7:* Pretraining domain adaptation.
            - *M8:* Learned Siamese CNN change detection models (FC-Siam-diff, SNUNet).
            - *M9:* Quality-aware gating modules (QAT-CD).
            - *M10+:* Integrated multi-task query-to-change pipeline.
            
            *(These future milestones do not have working weights or evaluations in the repository and are intentionally absent from interactive features.)*
            """
        )


if __name__ == "__main__":
    render_overview_page()
