"""Main Streamlit Application Entrypoint.

AI-Driven Satellite Image Semantic Search and Temporal Change Detection System.
Visualizes and interacts exclusively with implemented Milestones M1–M6.
"""

from __future__ import annotations

import os
import psutil
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Satellite AI — Research Dashboard (M1–M6)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import page render functions
from pages.overview import render_overview_page
from pages.dataset import render_dataset_page
from pages.retrieval import render_retrieval_page
from pages.change_detection import render_change_detection_page
from pages.false_alarm import render_false_alarm_page


# =============================================================================
# SIDEBAR SYSTEM STATUS & METADATA
# =============================================================================

def render_sidebar_metadata():
    """Renders persistent academic project metadata and hardware telemetry in sidebar."""
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0; margin-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0;">
            <h3 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                🛰️ Satellite AI System
            </h3>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.2rem;">
                B.Tech CSE Capstone Research
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Process RSS Memory Telemetry
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        mem_str = f"{mem_mb:.1f} MB"
    except Exception:
        mem_str = "N/A"

    st.sidebar.markdown(
        f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.8rem; margin-top: 1.5rem; font-size: 0.75rem; color: #334155;">
            <div style="font-weight: 700; color: #475569; margin-bottom: 0.3rem;">SYSTEM TELEMETRY</div>
            <div><strong>Profile:</strong> Strict CPU-Only</div>
            <div><strong>Active RSS:</strong> {mem_str} (&le; 8 GB)</div>
            <div><strong>Boundary:</strong> M1–M6 Verified</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="margin-top: 1.5rem; font-size: 0.75rem; color: #94a3b8; line-height: 1.4;">
            <strong>Implementation Governance:</strong><br>
            All displayed metrics originate directly from local benchmark files. 
            Future milestones (M7+) remain unexecuted.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STREAMLIT MULTIPAGE NAVIGATION
# =============================================================================

pages_nav = {
    "Project": [
        st.Page(render_overview_page, title="Overview", icon="🔬", default=True),
        st.Page(render_dataset_page, title="Dataset & Preprocessing", icon="📁"),
        st.Page(render_retrieval_page, title="Semantic Retrieval", icon="🔍"),
        st.Page(render_change_detection_page, title="Change Detection", icon="🛰️"),
        st.Page(render_false_alarm_page, title="False-Alarm Analysis", icon="📊"),
    ]
}

render_sidebar_metadata()

router = st.navigation(pages_nav)
router.run()
