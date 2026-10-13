"""Milestone 13 Evaluation, Ablation Synthesis & Visualization Generator.

Executes:
1. Workstream B: Semantic retrieval ablation compilation and RSICD multi-caption leakage analysis.
2. Workstream C: Unified change detection comparative evaluation across all 8 configurations.
3. Workstream D: Controlled QAT-CD quality gating ablation matrix with operational exchange ratios.
4. Workstream E: Statistical significance hypothesis tests and bootstrap confidence intervals.
5. Publication Figures: Generates 4 high-resolution research figures in experiments/figures/m13/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_workstream_b_retrieval_ablation() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compiles the retrieval ablation dataset, including protocol disclosures and leakage risks."""
    m4_summary_path = REPO_ROOT / "experiments" / "results" / "m4" / "summary.json"
    with open(m4_summary_path, "r", encoding="utf-8") as f:
        m4_data = json.load(f)

    records = []
    for item in m4_data:
        m_name = item["method"]
        records.append({
            "method_id": m_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("+", "plus").replace("/", "_"),
            "display_name": m_name,
            "modality_type": "Cross-Modal (Zero-Shot)" if "CLIP" in m_name else "Lexical (Text-Text)",
            "gallery_information": item.get("information_available", "N/A"),
            "protocol": item.get("protocol", "Standard"),
            "R@1": float(item["R@1"]),
            "R@5": float(item["R@5"]),
            "R@10": float(item["R@10"]),
            "MRR": float(item["MRR"]),
            "mean_latency_ms": float(item.get("latency_mean_ms", 0.0)),
            "peak_rss_mb": float(item.get("peak_rss_mb", 0.0)),
            "implementation_status": "VERIFIED_IMPLEMENTED",
        })

    # Record the documented scope discrepancy: CLIP + Metadata Reranking was planned in early roadmaps but never implemented
    records.append({
        "method_id": "clip_metadata_rerank",
        "display_name": "CLIP + Metadata Reranking",
        "modality_type": "Hybrid Cross-Modal Reranking",
        "gallery_information": "Visual image embeddings + Category metadata text reranking",
        "protocol": "Zero-shot cross-modal retrieval followed by category metadata rescoring",
        "R@1": None,
        "R@5": None,
        "R@10": None,
        "MRR": None,
        "mean_latency_ms": None,
        "peak_rss_mb": None,
        "implementation_status": "NOT_IMPLEMENTED_SCOPE_DISCREPANCY",
    })

    df = pd.DataFrame(records)

    analysis_doc = {
        "dataset": "RSICD Test Gallery (1,093 images, 5,465 human captions)",
        "protocol_investigation": {
            "finding": "Asymmetric Information Boundaries between Lexical and Multimodal Protocols",
            "bm25_loco_leakage_mechanism": (
                "In RSICD, each image has 5 captions written by human annotators. In the Leave-One-Caption-Out "
                "(LOCO) BM25 protocol, the query caption is held out, but the 4 remaining human captions for "
                "that exact target image remain in the candidate text gallery. Because crowd-workers frequently "
                "repeat distinctive phrasing (e.g. 'many airplanes parked at terminal gates'), BM25 LOCO achieves "
                "R@1=42.12% through sibling caption lexical overlap."
            ),
            "bm25_metadata_control": (
                "When all human captions are removed from the gallery and only category metadata labels are indexed "
                "(e.g. 'airport', 'parking'), BM25 lexical recall drops sharply to R@1=1.50% (MRR=0.0618)."
            ),
            "clip_cross_modal_integrity": (
                "CLIP ViT-B/32 operates under genuine cross-modal conditions where the candidate gallery contains "
                "zero human text (only 512-dimensional visual image embeddings). Query text is mapped directly into "
                "visual latent space, achieving R@1=5.45% (R@10=27.89%, MRR=0.1307) without text-text leakage."
            ),
            "prompt_ensemble_effect": (
                "Averaging 5 domain-specific remote sensing prompt templates yields R@1=5.14% and R@10=28.01% with "
                "14.71 ms CPU latency (vs 7.46 ms single prompt), demonstrating template robustness at 1.97x compute cost."
            ),
            "missing_reranking_status": (
                "CLIP + Metadata Reranking does not exist in the repository codebase. Milestone M7 was designated "
                "and executed strictly as Learned Change Detection (FC-Siam-diff). The retrieval reranking feature "
                "remains unexecuted and is formally marked NOT IMPLEMENTED."
            )
        },
        "records": records,
    }

    out_json = REPO_ROOT / "experiments" / "results" / "m13" / "semantic_retrieval_ablation.json"
    out_csv = REPO_ROOT / "experiments" / "results" / "m13" / "semantic_retrieval_ablation.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(analysis_doc, f, indent=2)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_json} and {out_csv}")
    return df, analysis_doc


def run_workstream_c_cd_comparison() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Synthesizes unified comparative change detection evaluation across all 8 configurations."""
    m11_summary_path = REPO_ROOT / "experiments" / "results" / "m11" / "benchmark_summary.json"
    with open(m11_summary_path, "r", encoding="utf-8") as f:
        m11_records = json.load(f)

    df = pd.DataFrame(m11_records)
    out_json = REPO_ROOT / "experiments" / "results" / "m13" / "change_detection_comparison.json"
    out_csv = REPO_ROOT / "experiments" / "results" / "m13" / "change_detection_comparison.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(m11_records, f, indent=2)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_json} and {out_csv}")
    return df, {"records": m11_records}


def run_workstream_d_qat_ablation() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Synthesizes QAT-CD ablation conditions, validation gamma sweep, and operational benefit ratios."""
    # Load M11 ablation analysis
    m11_abl_path = REPO_ROOT / "experiments" / "results" / "m11" / "qat_ablation_analysis.json"
    with open(m11_abl_path, "r", encoding="utf-8") as f:
        m11_abl = json.load(f)

    # Load M10 validation calibration thresholds
    m10_thresh_path = REPO_ROOT / "experiments" / "results" / "m10" / "thresholds.json"
    with open(m10_thresh_path, "r", encoding="utf-8") as f:
        m10_thresh = json.load(f)

    conds = m11_abl.get("ablation_conditions", {})
    records = []
    for cond_name, cdata in conds.items():
        m = cdata["metrics"]
        records.append({
            "condition": cond_name,
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "iou": float(m["iou"]),
            "accuracy": float(m["accuracy"]),
            "tp_pixels": int(m["tp"]),
            "fp_pixels": int(m["fp"]),
            "fn_pixels": int(m["fn"]),
            "tn_pixels": int(m["tn"]),
            "fp_reduction_pixels": int(cdata.get("fp_reduction_pixels", 0)),
            "fp_reduction_percent": float(cdata.get("fp_reduction_percent", 0.0)),
            "tp_loss_pixels": int(cdata.get("tp_loss_pixels", 0)),
            "tp_loss_percent": float(cdata.get("tp_loss_percent", 0.0)),
            "operational_benefit_ratio": float(cdata.get("operational_benefit_ratio", 0.0)),
        })

    df = pd.DataFrame(records)

    # Extract validation gamma sweep at tau=0.45
    val_sweep = []
    for r in m10_thresh.get("grid_results", []):
        if abs(r["threshold"] - 0.45) < 1e-5:
            val_sweep.append({
                "gamma": float(r["gamma"]),
                "threshold": float(r["threshold"]),
                "val_precision": float(r["precision"]),
                "val_recall": float(r["recall"]),
                "val_f1": float(r["f1"]),
                "val_iou": float(r["iou"]),
            })

    output_data = {
        "mathematical_formulation": {
            "gating_equation": "P_QAT(x, y) = clip(P_0(x, y) * (1 - gamma * U_det(x, y) * Q_defect(x, y)), 0.0, 1.0)",
            "detector_uncertainty": "U_det(x, y) = 4 * P_0(x, y) * (1 - P_0(x, y)) [Range: 0 to 1, max at P=0.5, zero at P=0 and P=1]",
            "composite_defect_index": "Q_defect(x, y) = w_reg * S_reg + w_illum * S_illum + w_sharp * S_sharp [Weights: 0.34, 0.33, 0.33]",
            "signals": {
                "S_reg": "Sobel gradient magnitude difference ratio (subpixel misregistration)",
                "S_illum": "Uniform filtered local luminance disparity (illumination & shadow disparity)",
                "S_sharp": "Laplacian energy disparity (optical blur differences)"
            }
        },
        "ablation_conditions": records,
        "validation_gamma_sweep_tau_0_45": val_sweep,
        "operational_tradeoff_summary": {
            "nominal_gamma": 0.30,
            "nominal_threshold": 0.45,
            "fp_pixels_eliminated": 573553,
            "fp_percent_reduction": 29.19,
            "tp_pixels_conceded": 336213,
            "tp_percent_conceded": 9.07,
            "precision_gain_pp": 5.43,
            "recall_drop_pp": -4.92,
            "f1_delta": -0.0115,
            "operational_benefit_ratio": 1.7059,
            "interpretation": (
                "For every 1.0 true-positive changed pixel conceded, 1.71 false-positive pixels are suppressed. "
                "In high-consequence operational satellite monitoring where false alarms carry substantial analyst verification "
                "costs, active gating offers positive utility despite lowering unweighted F1 by 0.0115."
            )
        }
    }

    out_json = REPO_ROOT / "experiments" / "results" / "m13" / "qat_cd_controlled_ablation.json"
    out_csv = REPO_ROOT / "experiments" / "results" / "m13" / "qat_cd_controlled_ablation.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_json} and {out_csv}")
    return df, output_data


def run_workstream_e_statistical_analysis() -> Dict[str, Any]:
    """Compiles scene-level paired statistical tests and bootstrap confidence intervals."""
    m11_paired_path = REPO_ROOT / "experiments" / "results" / "m11" / "paired_statistical_tests.json"
    m11_ci_path = REPO_ROOT / "experiments" / "results" / "m11" / "statistical_bootstrap_ci.json"

    with open(m11_paired_path, "r", encoding="utf-8") as f:
        paired_tests = json.load(f)
    with open(m11_ci_path, "r", encoding="utf-8") as f:
        bootstrap_ci = json.load(f)

    stat_data = {
        "evaluation_protocol": {
            "resampling_unit": "Scene-Level (Matched 1024x1024 Test Image Pairs, N=128)",
            "independence_disclosure": (
                "Individual pixels within a remote sensing scene exhibit strong spatial autocorrelation and are NOT "
                "statistically independent. Statistical hypothesis testing and bootstrap resampling are strictly conducted "
                "at the scene level (N=128 pairs), preventing severe p-value deflation."
            ),
            "test_set_reuse_disclosure": (
                "The LEVIR-CD test split was inspected in earlier milestones. Reported p-values and confidence intervals "
                "reflect standardized evaluation of this historical partition rather than untouched independent generalization."
            ),
            "bootstrap_replicates": 1000,
            "confidence_level": 0.95,
            "significance_alpha": 0.05,
            "test_type": "Two-sided Wilcoxon signed-rank test on matched scene pairs"
        },
        "paired_tests": paired_tests,
        "bootstrap_confidence_intervals": bootstrap_ci,
    }

    out_json = REPO_ROOT / "experiments" / "results" / "m13" / "statistical_significance_m13.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stat_data, f, indent=2)
    print(f"Saved: {out_json}")
    return stat_data


def generate_publication_figures():
    """Generates 4 high-resolution research figures in experiments/figures/m13/."""
    figures_dir = REPO_ROOT / "experiments" / "figures" / "m13"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 1: Retrieval Recall vs Latency Trade-Off
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    methods = [
        "CLIP ViT-B/32",
        "CLIP + Prompt Ens.",
        "BM25 (LOCO)",
        "BM25 (Metadata)",
    ]
    r1 = [5.45, 5.14, 42.12, 1.50]
    r10 = [27.89, 28.01, 67.87, 14.35]
    latencies = [7.46, 14.71, 1.50, 0.13]
    colors = ["#2563eb", "#0284c7", "#16a34a", "#94a3b8"]

    for m, r, lat, c in zip(methods, r10, latencies, colors):
        ax1.scatter(lat, r, s=180, color=c, zorder=3, edgecolors="#0f172a", lw=1.2)
        offset_y = 1.8 if m != "CLIP + Prompt Ens." else -3.2
        ax1.annotate(f"{m}\n(R@10={r:.1f}%, {lat:.1f}ms)", (lat, r),
                     textcoords="offset points", xytext=(8, offset_y), fontsize=8.5, fontweight="bold")

    ax1.set_xlabel("Mean CPU Query Latency (ms)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Recall@10 Retrieval Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Milestone 13: Semantic Retrieval Latency vs. Accuracy Trade-off (RSICD Test Split)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 80)
    ax1.set_xlim(-0.5, 18.5)
    plt.tight_layout()
    p1 = figures_dir / "m13_retrieval_tradeoff.png"
    fig1.savefig(p1)
    plt.close(fig1)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------------
    # Figure 2: Change Detection Pareto Frontier (Precision vs Recall & Latency vs F1)
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    models = ["B1 Pixel Diff", "B2 SSIM", "B3 CVA", "M7 FC-Siam", "M8 SNUNet", "M9 BIT-CD", "M10 QAT (Cal.)", "M10 QAT (Act.)"]
    prec = [10.64, 7.06, 10.56, 62.81, 9.08, 65.36, 65.36, 70.78]
    rec = [16.66, 53.32, 17.46, 58.07, 77.58, 54.22, 54.22, 49.30]
    f1 = [12.99, 12.46, 13.16, 60.35, 16.26, 59.27, 59.27, 58.12]
    lat = [13.0, 47.0, 26.5, 96.6, 95.8, 116.1, 116.1, 116.1]
    cd_colors = ["#94a3b8", "#64748b", "#475569", "#f59e0b", "#ef4444", "#10b981", "#3b82f6", "#8b5cf6"]

    # Ax2a: Precision vs Recall
    for m, p, r, c in zip(models, prec, rec, cd_colors):
        ax2a.scatter(r, p, s=140, color=c, zorder=3, edgecolors="#0f172a", lw=1.1)
        ax2a.annotate(m, (r, p), textcoords="offset points", xytext=(5, 4), fontsize=7.5, fontweight="bold")
    ax2a.set_xlabel("Recall (%)", fontsize=9.5, fontweight="bold")
    ax2a.set_ylabel("Precision (%)", fontsize=9.5, fontweight="bold")
    ax2a.set_title("Precision vs. Recall Trade-off (134.2M Test Pixels)", fontsize=10.5, fontweight="bold")
    ax2a.grid(True, linestyle="--", alpha=0.5)

    # Ax2b: CPU Latency vs F1
    for m, l, score, c in zip(models, lat, f1, cd_colors):
        ax2b.scatter(l, score, s=140, color=c, zorder=3, edgecolors="#0f172a", lw=1.1)
        ax2b.annotate(m, (l, score), textcoords="offset points", xytext=(5, -6), fontsize=7.5, fontweight="bold")
    ax2b.set_xlabel("CPU Latency (ms / 1024x1024 pair)", fontsize=9.5, fontweight="bold")
    ax2b.set_ylabel("F1 Score (%)", fontsize=9.5, fontweight="bold")
    ax2b.set_title("CPU Latency vs. Test F1 Score", fontsize=10.5, fontweight="bold")
    ax2b.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p2 = figures_dir / "m13_cd_pareto_frontier.png"
    fig2.savefig(p2)
    plt.close(fig2)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------------
    # Figure 3: QAT-CD Ablation Waterfall (FP Reduction vs TP Loss)
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 4.5), dpi=300)
    abl_names = ["Active All (γ=0.30)", "Registration Only", "Illumination Only", "Sharpness Only", "No Uncertainty"]
    fp_red = [573553, 642387, 346192, 677653, 576168]
    tp_loss = [336213, 428333, 177045, 436681, 338483]
    x_pos = np.arange(len(abl_names))
    bar_width = 0.35

    rects1 = ax3.bar(x_pos - bar_width/2, [v / 1000.0 for v in fp_red], bar_width, label="False Positives Eliminated (k px)", color="#10b981")
    rects2 = ax3.bar(x_pos + bar_width/2, [v / 1000.0 for v in tp_loss], bar_width, label="True Positives Conceded (k px)", color="#ef4444")

    ax3.set_ylabel("Thousands of Pixels (k px)", fontsize=10, fontweight="bold")
    ax3.set_title("QAT-CD Ablations: False Alarm Elimination vs. True Positive Loss", fontsize=11, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(abl_names, fontsize=8.5, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for r1, r2 in zip(rects1, rects2):
        ratio = r1.get_height() / max(r2.get_height(), 1e-3)
        ax3.annotate(f"{ratio:.2f}x", xy=(r1.get_x() + bar_width, max(r1.get_height(), r2.get_height())),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#0f172a")

    plt.tight_layout()
    p3 = figures_dir / "m13_qat_ablation_waterfall.png"
    fig3.savefig(p3)
    plt.close(fig3)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------------
    # Figure 4: Gamma Sensitivity Curve on Validation Split (tau = 0.45)
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    val_prec = [58.14, 59.89, 61.64, 63.40, 65.11, 66.82, 68.39]
    val_rec = [53.61, 51.70, 49.61, 47.32, 44.81, 42.33, 39.91]
    val_f1 = [55.78, 55.49, 54.98, 54.19, 53.09, 51.83, 50.41]

    ax4.plot(gammas, val_prec, marker="s", lw=2, color="#3b82f6", label="Validation Precision (%)")
    ax4.plot(gammas, val_rec, marker="^", lw=2, color="#ef4444", label="Validation Recall (%)")
    ax4.plot(gammas, val_f1, marker="o", lw=2.5, color="#10b981", label="Validation F1 Score (%)")

    ax4.axvline(0.0, color="#10b981", linestyle=":", alpha=0.7, label="Calibrated Operating Point (γ*=0.0)")
    ax4.axvline(0.3, color="#8b5cf6", linestyle="--", alpha=0.7, label="Exploratory Active Gating (γ=0.30)")

    ax4.set_xlabel("Quality Gating Attenuation Strength (gamma)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Metric Score (%)", fontsize=10, fontweight="bold")
    ax4.set_title("Validation Calibration Sweep: Precision/Recall/F1 vs. Gamma (tau*=0.45, 64 Val Scenes)", fontsize=10.5, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="center right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    p4 = figures_dir / "m13_gamma_sensitivity_curve.png"
    fig4.savefig(p4)
    plt.close(fig4)
    print(f"Saved: {p4}")


def main():
    print("=== Milestone 13 Evaluation & Ablation Synthesis ===")
    run_workstream_b_retrieval_ablation()
    run_workstream_c_cd_comparison()
    run_workstream_d_qat_ablation()
    run_workstream_e_statistical_analysis()
    generate_publication_figures()
    print("All Milestone 13 results and figures successfully generated!")


if __name__ == "__main__":
    main()
