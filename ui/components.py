"""Reusable academic UI components, status badges, and research callouts."""

from __future__ import annotations

import streamlit as st
from typing import Optional, Dict, Any


def render_header(
    title: str,
    subtitle: Optional[str] = None,
    badge_text: str = "Milestones M1–M6 Implemented",
):
    """Renders a clean, academic header with research status."""
    st.markdown(
        f"""
        <div style="padding: 1.2rem 1.4rem; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 1.5rem; color: #f8fafc;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <h1 style="margin: 0; font-size: 1.65rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">
                    {title}
                </h1>
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #93c5fd; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    {badge_text}
                </span>
            </div>
            {f'<p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.95rem; font-weight: 400;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_milestone_status_bar():
    """Renders visual milestone progress indicator strictly distinguishing completed vs planned."""
    st.markdown(
        """
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;">
                Research Implementation Status (Strict Verification Boundary)
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
                <span style="background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M1: Architecture</span>
                <span style="background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M2: Datasets & Pipeline</span>
                <span style="background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M3: Literature & Metric Framework</span>
                <span style="background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M4: Semantic Retrieval Baselines</span>
                <span style="background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M5: Classical Bi-Temporal CD</span>
                <span style="background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">✓ M6: False-Alarm & Robustness</span>
                <span style="background: #f1f5f9; color: #94a3b8; border: 1px dashed #cbd5e1; padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500;">○ M7+: Planned / Not Implemented</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hardware_constraint_badge():
    """Renders the explicit CPU execution constraint notice."""
    st.markdown(
        """
        <div style="background: #eff6ff; border-left: 4px solid #3b82f6; padding: 0.6rem 0.9rem; border-radius: 0 6px 6px 0; margin-bottom: 1rem; font-size: 0.85rem; color: #1e40af;">
            <strong>Hardware Constraint Profile:</strong> Strict CPU execution (&le; 8 GB RAM budget). No CUDA / GPU acceleration utilized in benchmarks.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_frozen_threshold_notice(threshold_val: float, detector_name: str):
    """Renders the mandatory validation threshold frozen policy notice."""
    st.markdown(
        f"""
        <div style="background: #fefce8; border: 1px solid #fef08a; border-left: 4px solid #eab308; padding: 0.65rem 0.9rem; border-radius: 0 6px 6px 0; margin: 0.8rem 0; font-size: 0.85rem; color: #854d0e;">
            <strong>Decision Threshold Frozen:</strong> <code>&tau;* = {threshold_val:.4f}</code> for <strong>{detector_name}</strong> was optimized strictly on the 64 LEVIR-CD validation pairs to maximize validation F1, and frozen prior to test-split evaluation. Zero test-set threshold tuning occurred.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_leakage_audit_notice():
    """Renders the scientific audit note for M4 BM25."""
    st.markdown(
        """
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #22c55e; padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0; font-size: 0.875rem; color: #166534;">
            <strong>Methodological Rigor Note (DEC-024 / DEC-025):</strong>
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
