"""Plotly and Matplotlib research charts for M4, M5, and M6 results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

# We use Matplotlib / Seaborn or Streamlit native bar charts for 100% stable rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_m4_recall_comparison(summary_df: pd.DataFrame) -> plt.Figure:
    """Generates grouped bar chart of R@1, R@5, R@10 across evaluated M4 methods."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

    # Filter out empty or duplicate rows if any
    methods = summary_df["method"].tolist()
    r1 = [v * 100 for v in summary_df["R@1"]]
    r5 = [v * 100 for v in summary_df["R@5"]]
    r10 = [v * 100 for v in summary_df["R@10"]]

    x = range(len(methods))
    width = 0.25

    rects1 = ax.bar([i - width for i in x], r1, width, label="R@1 (%)", color="#2563eb")
    rects2 = ax.bar([i for i in x], r5, width, label="R@5 (%)", color="#38bdf8")
    rects3 = ax.bar([i + width for i in x], r10, width, label="R@10 (%)", color="#93c5fd")

    ax.set_ylabel("Recall Score (%)", fontsize=10, fontweight="bold")
    ax.set_title("Milestone 4: Semantic Retrieval Recall Comparison (RSICD Test Gallery)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    
    # Shorten method labels for readability
    short_labels = [
        m.replace(" (Leave-One-Caption-Out)", "\n(LOCO Primary)")
        .replace(" (Category Metadata)", "\n(Metadata Control)")
        .replace(" (Original Diagnostic / Leaky)", "\n(Legacy Leaky)")
        .replace("CLIP + Prompt Ensemble", "CLIP +\nPrompt Ens.")
        for m in methods
    ]
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)

    # Annotate R@1 bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_m4_latency_mrr(summary_df: pd.DataFrame) -> plt.Figure:
    """Generates Latency vs MRR Pareto trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)

    for _, row in summary_df.iterrows():
        lat = row["latency_mean_ms"]
        mrr = row["MRR"]
        name = row["method"]
        color = "#dc2626" if "Leaky" in name else ("#16a34a" if "CLIP" in name else "#2563eb")
        
        ax.scatter(lat, mrr, s=120, color=color, alpha=0.85, edgecolors="black", linewidth=1.2)
        offset_y = 0.03 if "Leaky" not in name else -0.05
        ax.annotate(
            name.split("(")[0].strip(),
            xy=(lat, mrr),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

    ax.set_xlabel("Mean Latency on CPU (ms)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=10, fontweight="bold")
    ax.set_title("Milestone 4: Latency vs. Retrieval Accuracy Trade-off (CPU)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_m5_detector_comparison(m5_summary_df: pd.DataFrame) -> plt.Figure:
    """Generates comparison bar chart for M5 classical change detection baselines."""
    # Filter to Frozen Validation F1 threshold rows
    df_frozen = m5_summary_df[m5_summary_df["threshold_source"].str.contains("Frozen", na=False)]
    if df_frozen.empty:
        df_frozen = m5_summary_df

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    methods = df_frozen["method"].tolist()
    prec = [v * 100 for v in df_frozen["precision"]]
    rec = [v * 100 for v in df_frozen["recall"]]
    f1 = [v * 100 for v in df_frozen["f1"]]
    iou = [v * 100 for v in df_frozen["iou"]]

    x = range(len(methods))
    width = 0.18

    ax.bar([i - 1.5 * width for i in x], prec, width, label="Precision (%)", color="#3b82f6")
    ax.bar([i - 0.5 * width for i in x], rec, width, label="Recall (%)", color="#10b981")
    ax.bar([i + 0.5 * width for i in x], f1, width, label="F1 Score (%)", color="#ef4444")
    ax.bar([i + 1.5 * width for i in x], iou, width, label="IoU (%)", color="#8b5cf6")

    ax.set_ylabel("Metric Score (%)", fontsize=10, fontweight="bold")
    ax.set_title("Milestone 5: Classical Detector Test Performance (Frozen tau*)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    clean_labels = [m.replace("B1_", "B1: ").replace("B2_", "B2: ").replace("B3_", "B3: ").replace("_", " ") for m in methods]
    ax.set_xticklabels(clean_labels, fontsize=9, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


def plot_m6_f1_vs_severity(m6_df: pd.DataFrame, perturbation_name: str) -> Optional[plt.Figure]:
    """Plots F1 score vs severity for a selected perturbation across all 3 detectors."""
    sub_df = m6_df[m6_df["perturbation"] == perturbation_name]
    if sub_df.empty:
        return None

    # Get control row
    ctrl_df = m6_df[m6_df["perturbation"] == "control"]

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=120)
    severities = ["control", "mild", "medium", "strong"]
    color_map = {"B1_Pixel_Diff": "#2563eb", "B2_SSIM": "#dc2626", "B3_CVA": "#16a34a"}

    for det in ["B1_Pixel_Diff", "B2_SSIM", "B3_CVA"]:
        f1_vals = []
        # control
        c_val = ctrl_df[ctrl_df["detector"] == det]["f1"].values
        f1_vals.append(c_val[0] if len(c_val) > 0 else None)
        
        for sev in ["mild", "medium", "strong"]:
            v = sub_df[(sub_df["detector"] == det) & (sub_df["severity"] == sev)]["f1"].values
            f1_vals.append(v[0] if len(v) > 0 else None)

        ax.plot(severities, f1_vals, marker="o", lw=2, label=det.replace("_", " "), color=color_map.get(det, "#000000"))

    ax.set_xlabel("Perturbation Severity", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=10, fontweight="bold")
    ax.set_title(f"F1 vs. Severity: {perturbation_name.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    return fig


def display_research_figure(figure_path: Path, caption: Optional[str] = None):
    """Displays a static precomputed research figure from experiments/figures/."""
    if figure_path and figure_path.exists():
        st.image(str(figure_path), caption=caption or figure_path.name, use_container_width=True)
    else:
        st.warning(f"Research figure not found at path: {figure_path}")
