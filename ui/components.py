"""Reusable academic UI components, status badges, and research callouts.
Styled with the unified Earth-inspired visual system.
"""

from __future__ import annotations

import streamlit as st
from typing import Optional, Dict, Any


def render_header(
    title: str,
    subtitle: Optional[str] = None,
    badge_text: str = "Milestones M1–M13 Implemented",
):
    """Renders a clean, editorial academic header with Earth-inspired styling."""
    st.markdown(
        f"""
        <div style="padding: 1.25rem 1.5rem; background: linear-gradient(135deg, #2F3E32 0%, #4A2E2B 100%); border-radius: 9px; border-left: 5px solid #C86B4D; margin-bottom: 1.5rem; color: #F1EBE4; box-shadow: 0 4px 12px rgba(47, 62, 50, 0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.6rem;">
                <h1 style="margin: 0; font-size: 1.65rem; font-weight: 700; color: #F1EBE4; letter-spacing: -0.02em;">
                    {title}
                </h1>
                <span style="background: rgba(200, 107, 77, 0.25); border: 1px solid #C86B4D; color: #E3DAC9; padding: 0.28rem 0.8rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    {badge_text}
                </span>
            </div>
            {f'<p style="margin: 0.5rem 0 0 0; color: #9DB198; font-size: 0.95rem; font-weight: 400;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_milestone_status_bar():
    """Renders visual milestone progress indicator strictly distinguishing completed vs planned."""
    st.markdown(
        """
        <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(62, 59, 57, 0.06);">
            <div style="font-size: 0.75rem; font-weight: 700; color: #4A2E2B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;">
                Research Implementation Status (Strict Verification Boundary)
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M1: Architecture</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M2: Datasets & Pipeline</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M3: Literature & Metrics</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M4: Retrieval Baselines</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M5: Classical CD</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M6: Robustness Audit</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M7: FC-Siam-diff</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M8: SNUNet-CD</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M9: BIT-CD</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M10: QAT-CD</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M11: Evaluation</span>
                <span style="background: #6B7A65; color: #F1EBE4; border: 1px solid #2F3E32; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M12: Web Interface</span>
                <span style="background: #C86B4D; color: #F1EBE4; border: 1px solid #4A2E2B; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M13: Integration & Ablation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hardware_constraint_badge():
    """Renders the explicit CPU execution constraint notice."""
    st.markdown(
        """
        <div style="background: #D9C5B2; border-left: 4px solid #2F3E32; border: 1px solid #BFAF9E; border-left-width: 4px; padding: 0.65rem 0.95rem; border-radius: 0 6px 6px 0; margin-bottom: 1rem; font-size: 0.85rem; color: #3E3B39;">
            <strong style="color: #2F3E32;">Hardware Constraint Profile:</strong> Strict CPU execution (&le; 8 GB RAM budget). No CUDA / GPU acceleration utilized in benchmarks.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_frozen_threshold_notice(threshold_val: float, detector_name: str):
    """Renders the mandatory validation threshold frozen policy notice."""
    st.markdown(
        f"""
        <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-left: 4px solid #E1A95F; padding: 0.7rem 0.95rem; border-radius: 0 6px 6px 0; margin: 0.85rem 0; font-size: 0.85rem; color: #3E3B39;">
            <strong style="color: #4A2E2B;">Decision Threshold Frozen:</strong> <code style="background: #FAF7F2; border: 1px solid #BFAF9E; color: #3E3B39; padding: 0.15rem 0.35rem; border-radius: 3px;">&tau;* = {threshold_val:.4f}</code> for <strong>{detector_name}</strong> was optimized strictly on the 64 LEVIR-CD validation pairs to maximize validation F1, and frozen prior to test-split evaluation. Zero test-set threshold tuning occurred.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_leakage_audit_notice():
    """Renders the scientific audit note for M4 BM25."""
    st.markdown(
        """
        <div style="background: #FAF7F2; border: 1px solid #9DB198; border-left: 4px solid #6B7A65; padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0; font-size: 0.875rem; color: #3E3B39;">
            <strong style="color: #2F3E32;">Methodological Rigor Note (DEC-024 / DEC-025):</strong>
            The primary lexical benchmark is <strong>Mode A (Leave-One-Caption-Out BM25)</strong>, where the query caption is strictly excluded from the target image's gallery representation to prevent text self-overlap leakage. 
            The <em>legacy diagnostic BM25</em> (85.65% R@1) is preserved for auditable historical comparison and must not be cited as the true cross-modal baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str | float,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
):
    """Renders a structured metric card with clean styling."""
    formatted_val = f"{value:.4f}" if isinstance(value, float) else str(value)
    st.metric(label=label, value=formatted_val, delta=delta, delta_color=delta_color, help=help_text)
