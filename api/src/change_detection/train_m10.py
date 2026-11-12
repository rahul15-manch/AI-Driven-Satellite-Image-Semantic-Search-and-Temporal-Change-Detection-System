"""Calibration, Scientific Evaluation & Ablation Pipeline for Milestone M10 QAT-CD.

Executes reproducible, leakage-free QAT-CD workflow on LEVIR-CD:
1. Base Detector: Loads frozen M9 BIT-CD checkpoint (and compares against M7 FC-Siam-diff).
2. Validation Calibration: Calibrates quality attenuation strength gamma* and decision threshold tau*
   strictly on the official 64 validation scenes (zero test leakage).
3. Freezes all parameters and decision threshold tau*.
4. Held-Out Test Evaluation: Benchmarks official 128 test scenes (134.2M pixels).
5. Ablation Studies:
   - Unadjusted baseline (gamma = 0.0, base tau* = 0.45)
   - Individual quality signal ablations (Registration only, Illumination only, Sharpness only)
   - Margin uncertainty ablation (linear attenuation vs. uncertainty-modulated attenuation)
6. Compiles comprehensive comparison table from M5 classical baselines (B1, B2, B3) through M7, M8, M9, and M10.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.change_detection.bit_cd import load_checkpoint as load_m9_checkpoint, BITChangeDetector
from src.change_detection.fc_siam_diff import load_checkpoint as load_m7_checkpoint, FCSiamDiffDetector
from src.change_detection.qat_cd import (
    QualityAwareChangeDetector,
    extract_registration_discrepancy,
    extract_illumination_discrepancy,
    extract_sharpness_discrepancy,
    compute_composite_defect_index,
    compute_detector_uncertainty,
    quality_aware_gate,
    save_qat_metadata,
)
from src.data.levir_loader import LEVIRDataset
from src.data.patch_extractor import PatchExtractor
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from utils.config_loader import load_m10_config


def calibrate_qat_parameters(
    base_detector: Any,
    val_dataset: LEVIRDataset,
    patch_extractor: PatchExtractor,
    gamma_candidates: List[float],
    threshold_candidates: List[float],
    weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
) -> Tuple[float, float, Dict[str, Any]]:
    """Calibrates gamma* and tau* strictly on the 64 validation scenes."""
    print(f"\n[Step 1/4] Precomputing base predictions and quality signals for {len(val_dataset)} validation scenes...")
    t0_pre = time.perf_counter()

    val_records = []
    for idx in range(len(val_dataset)):
        t1_t, t2_t, lbl_t, meta = val_dataset[idx]
        img1 = (t1_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img2 = (t2_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt = lbl_t.squeeze(0).numpy().astype(np.uint8)

        stem = meta["sample_id"] if isinstance(meta, dict) else getattr(meta, "sample_id", f"scene_{idx}")
        patches = patch_extractor.extract_from_pair(img1, img2, gt, parent_stem=stem)
        diff_patches = []
        for p1, p2, _, p_meta in patches:
            p_prob = base_detector.compute_difference_map(p1, p2)
            diff_patches.append((p_prob, p_meta))

        full_prob = patch_extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)

        # Full-scene quality signals
        s_reg = extract_registration_discrepancy(img1, img2)
        s_illum = extract_illumination_discrepancy(img1, img2)
        s_sharp = extract_sharpness_discrepancy(img1, img2)
        q_defect = compute_composite_defect_index(s_reg, s_illum, s_sharp, weights=weights)

        val_records.append({
            "p0": full_prob,
            "q_defect": q_defect,
            "gt": gt,
        })

        if (idx + 1) % 16 == 0 or (idx + 1) == len(val_dataset):
            print(f"  Processed {idx + 1}/{len(val_dataset)} validation scenes (elapsed: {time.perf_counter() - t0_pre:.1f}s)")

    print(f"\n[Step 2/4] Executing grid search over gamma x tau ({len(gamma_candidates)} x {len(threshold_candidates)} = {len(gamma_candidates)*len(threshold_candidates)} combinations)...")

    grid_results = []
    best_f1 = -1.0
    best_gamma = 0.0
    best_tau = 0.50
    best_metrics: Dict[str, Any] = {}

    for gamma in gamma_candidates:
        # Precompute gated probabilities for this gamma
        gated_probs = []
        for r in val_records:
            p_gated = quality_aware_gate(r["p0"], r["q_defect"], gamma=gamma, use_uncertainty=True)
            gated_probs.append(p_gated)

        for tau in threshold_candidates:
            cm = ConfusionMatrix()
            for p_gated, r in zip(gated_probs, val_records):
                pred_bin = (p_gated >= tau).astype(np.uint8)
                cm.update(pred_bin, r["gt"])

            m = ChangeDetectionEvaluator.evaluate(cm)
            record = {
                "gamma": float(gamma),
                "threshold": float(tau),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "iou": float(m["iou"]),
            }
            grid_results.append(record)

            if m["f1"] > best_f1:
                best_f1 = float(m["f1"])
                best_gamma = float(gamma)
                best_tau = float(tau)
                best_metrics = m

    print(
        f"[Calibration Complete] Optimal parameters selected: gamma* = {best_gamma:.2f}, tau* = {best_tau:.4f}\n"
        f"  Validation F1: {best_f1:.4f} | IoU: {best_metrics['iou']:.4f} | "
        f"Precision: {best_metrics['precision']:.4f} | Recall: {best_metrics['recall']:.4f}"
    )

    calibration_metadata = {
        "best_gamma": best_gamma,
        "best_threshold": best_tau,
        "best_val_f1": best_f1,
        "best_val_metrics": best_metrics,
        "grid_results": grid_results,
    }
    return best_gamma, best_tau, calibration_metadata


def evaluate_qat_test_set(
    qat_detector: QualityAwareChangeDetector,
    test_dataset: LEVIRDataset,
    patch_extractor: PatchExtractor,
    frozen_tau: float,
    unadjusted_tau: float = 0.45,
    ablation_gamma: float = 0.30,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Evaluates official 128 held-out test scenes for QAT-CD and ablations."""
    print(f"\n[Step 3/4] Evaluating {len(test_dataset)} test scenes with frozen parameters (gamma* = {qat_detector.gamma:.2f}, tau* = {frozen_tau:.4f}, ablation_gamma = {ablation_gamma:.2f})...")

    cm_full = ConfusionMatrix()
    cm_no_quality = ConfusionMatrix()
    cm_active_all = ConfusionMatrix()
    cm_reg_only = ConfusionMatrix()
    cm_illum_only = ConfusionMatrix()
    cm_sharp_only = ConfusionMatrix()
    cm_no_uncertainty = ConfusionMatrix()

    per_image_rows = []
    latencies_ms = []

    process = psutil.Process(os.getpid())
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    t_start = time.perf_counter()

    for idx in range(len(test_dataset)):
        t0_pair = time.perf_counter()
        t1_t, t2_t, lbl_t, meta = test_dataset[idx]
        img1 = (t1_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img2 = (t2_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt = lbl_t.squeeze(0).numpy().astype(np.uint8)

        stem = meta["sample_id"] if isinstance(meta, dict) else getattr(meta, "sample_id", f"scene_{idx}")

        # Base patch inference
        patches = patch_extractor.extract_from_pair(img1, img2, gt, parent_stem=stem)
        diff_patches = []
        for p1, p2, _, p_meta in patches:
            p_prob = qat_detector.base_detector.compute_difference_map(p1, p2)
            diff_patches.append((p_prob, p_meta))

        p0_full = patch_extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)

        # Diagnostics & quality signals
        diag = qat_detector.compute_quality_diagnostics(img1, img2, base_prob=p0_full)
        p_gated = diag["gated_prob"]
        bin_pred = (p_gated >= frozen_tau).astype(np.uint8)

        pair_latency_ms = (time.perf_counter() - t0_pair) * 1000.0
        latencies_ms.append(pair_latency_ms)

        # Update primary QAT-CD confusion matrix
        cm_full.update(bin_pred, gt)

        # Ablation 1: No Quality Adjustment (Pure base model at unadjusted tau*)
        bin_no_qual = (p0_full >= unadjusted_tau).astype(np.uint8)
        cm_no_quality.update(bin_no_qual, gt)

        # Ablation 2: Active Gating with All Signals (Nominal gamma = 0.30)
        p_active_all = quality_aware_gate(p0_full, diag["defect_index"], gamma=ablation_gamma, use_uncertainty=True)
        cm_active_all.update((p_active_all >= frozen_tau).astype(np.uint8), gt)

        # Ablation 3: Registration Only (gamma = ablation_gamma)
        q_reg_only = compute_composite_defect_index(diag["s_reg"], None, None, weights=(1.0, 0.0, 0.0))
        p_reg_only = quality_aware_gate(p0_full, q_reg_only, gamma=ablation_gamma, use_uncertainty=True)
        cm_reg_only.update((p_reg_only >= frozen_tau).astype(np.uint8), gt)

        # Ablation 4: Illumination Only (gamma = ablation_gamma)
        q_illum_only = compute_composite_defect_index(None, diag["s_illum"], None, weights=(0.0, 1.0, 0.0))
        p_illum_only = quality_aware_gate(p0_full, q_illum_only, gamma=ablation_gamma, use_uncertainty=True)
        cm_illum_only.update((p_illum_only >= frozen_tau).astype(np.uint8), gt)

        # Ablation 5: Sharpness Only (gamma = ablation_gamma)
        q_sharp_only = compute_composite_defect_index(None, None, diag["s_sharp"], weights=(0.0, 0.0, 1.0))
        p_sharp_only = quality_aware_gate(p0_full, q_sharp_only, gamma=ablation_gamma, use_uncertainty=True)
        cm_sharp_only.update((p_sharp_only >= frozen_tau).astype(np.uint8), gt)

        # Ablation 6: Without Margin Uncertainty Gating (gamma = ablation_gamma)
        p_no_unc = quality_aware_gate(p0_full, diag["defect_index"], gamma=ablation_gamma, use_uncertainty=False)
        cm_no_uncertainty.update((p_no_unc >= frozen_tau).astype(np.uint8), gt)


        # Per-image tracking
        cm_img = ConfusionMatrix()
        cm_img.update(bin_pred, gt)
        m_img = ChangeDetectionEvaluator.evaluate(cm_img)

        current_rss = process.memory_info().rss / (1024 * 1024)
        if current_rss > peak_rss_mb:
            peak_rss_mb = current_rss

        per_image_rows.append({
            "image_id": stem,
            "threshold": frozen_tau,
            "gamma": qat_detector.gamma,
            "tp": int(cm_img.tp),
            "fp": int(cm_img.fp),
            "fn": int(cm_img.fn),
            "tn": int(cm_img.tn),
            "precision": float(m_img["precision"]),
            "recall": float(m_img["recall"]),
            "f1": float(m_img["f1"]),
            "iou": float(m_img["iou"]),
            "accuracy": float(m_img["accuracy"]),
            "latency_ms": round(pair_latency_ms, 2),
        })

        if (idx + 1) % 32 == 0 or (idx + 1) == len(test_dataset):
            print(f"  Processed {idx + 1}/{len(test_dataset)} test scenes (elapsed: {time.perf_counter() - t_start:.1f}s)")

    global_m = ChangeDetectionEvaluator.evaluate(cm_full)
    mean_latency = float(np.mean(latencies_ms))

    qat_test_metrics = {
        "method": "M10_QAT_CD",
        "threshold_source": "Validation F1-Optimal (Frozen)",
        "threshold_value": float(frozen_tau),
        "gamma_attenuation": float(qat_detector.gamma),
        "f1": float(global_m["f1"]),
        "iou": float(global_m["iou"]),
        "precision": float(global_m["precision"]),
        "recall": float(global_m["recall"]),
        "accuracy": float(global_m["accuracy"]),
        "tp": int(cm_full.tp),
        "fp": int(cm_full.fp),
        "fn": int(cm_full.fn),
        "tn": int(cm_full.tn),
        "total_scenes": len(test_dataset),
        "total_pixels": int(cm_full.tp + cm_full.fp + cm_full.fn + cm_full.tn),
        "gt_change_prevalence": float((cm_full.tp + cm_full.fn) / (cm_full.tp + cm_full.fp + cm_full.fn + cm_full.tn)),
        "predicted_change_prevalence": float((cm_full.tp + cm_full.fp) / (cm_full.tp + cm_full.fp + cm_full.fn + cm_full.tn)),
        "latency_mean_ms": round(mean_latency, 2),
        "latency_per_patch_ms": round(mean_latency / 16.0, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }

    ablations = {
        "QAT_CD_Full": global_m,
        "Ablation_NoQuality": ChangeDetectionEvaluator.evaluate(cm_no_quality),
        "Ablation_ActiveAll": ChangeDetectionEvaluator.evaluate(cm_active_all),
        "Ablation_RegOnly": ChangeDetectionEvaluator.evaluate(cm_reg_only),
        "Ablation_IllumOnly": ChangeDetectionEvaluator.evaluate(cm_illum_only),
        "Ablation_SharpOnly": ChangeDetectionEvaluator.evaluate(cm_sharp_only),
        "Ablation_NoUncertainty": ChangeDetectionEvaluator.evaluate(cm_no_uncertainty),
    }

    return qat_test_metrics, pd.DataFrame(per_image_rows), ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 QAT-CD Calibration & Evaluation")
    parser.add_argument("--config", type=str, default="experiments/configs/m10_qat_cd.yaml")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip validation calibration and use existing thresholds.json")
    args = parser.parse_args()

    config = load_m10_config()
    root = REPO_ROOT

    ckpt_dir = root / config["paths"]["checkpoint_dir"]
    results_dir = root / config["paths"]["results_dir"]
    figures_dir = root / config["paths"]["figures_dir"]

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = root / config["dataset"]["raw_dir"]
    val_raw_ds = LEVIRDataset(root_dir=raw_dir, split=config["dataset"]["val_split"])
    test_raw_ds = LEVIRDataset(root_dir=raw_dir, split=config["dataset"]["test_split"])
    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")

    assert len(val_raw_ds) == 64, f"Expected 64 val scenes, found {len(val_raw_ds)}"
    assert len(test_raw_ds) == 128, f"Expected 128 test scenes, found {len(test_raw_ds)}"

    # Load frozen base detector (M9 BIT-CD)
    m9_ckpt_path = root / config["base_detector"]["checkpoint_path"]
    assert m9_ckpt_path.exists(), f"Base detector checkpoint not found: {m9_ckpt_path}"
    base_model, base_meta = load_m9_checkpoint(m9_ckpt_path, device="cpu")
    base_detector = BITChangeDetector(model=base_model, device="cpu", name="M9_BIT_CD")
    base_frozen_tau = float(config["base_detector"].get("frozen_threshold", 0.45))

    w_reg = config["quality_estimator"]["signals"]["registration"]["weight"]
    w_illum = config["quality_estimator"]["signals"]["illumination"]["weight"]
    w_sharp = config["quality_estimator"]["signals"]["sharpness"]["weight"]
    weights = (w_reg, w_illum, w_sharp)

    thresh_file = results_dir / "thresholds.json"
    if args.skip_calibration and thresh_file.exists():
        print(f"\n[Step 1/4] Loading existing validation calibration from {thresh_file}...")
        with open(thresh_file, "r", encoding="utf-8") as f:
            calib_meta = json.load(f)
        best_gamma = float(calib_meta["best_gamma"])
        best_tau = float(calib_meta["best_threshold"])
        gamma_candidates = config["calibration"]["gamma_candidates"]
    else:
        # Calibration on Validation Split (64 scenes)
        gamma_candidates = config["calibration"]["gamma_candidates"]
        tau_candidates = config["calibration"]["threshold_candidates"]

        best_gamma, best_tau, calib_meta = calibrate_qat_parameters(
            base_detector=base_detector,
            val_dataset=val_raw_ds,
            patch_extractor=extractor,
            gamma_candidates=gamma_candidates,
            threshold_candidates=tau_candidates,
            weights=weights,
        )

        with open(thresh_file, "w", encoding="utf-8") as f:
            json.dump(calib_meta, f, indent=2)

    # Initialize frozen QAT-CD detector
    qat_detector = QualityAwareChangeDetector(
        base_detector=base_detector,
        gamma=best_gamma,
        weights=weights,
        use_reg=True,
        use_illum=True,
        use_sharp=True,
        use_uncertainty=True,
        name="M10_QAT_CD",
    )

    # Official Held-Out Test Evaluation (128 scenes)
    test_metrics, per_image_df, ablations = evaluate_qat_test_set(
        qat_detector=qat_detector,
        test_dataset=test_raw_ds,
        patch_extractor=extractor,
        frozen_tau=best_tau,
        unadjusted_tau=base_frozen_tau,
        ablation_gamma=float(config["gating"].get("attenuation_strength", 0.30)),
    )

    # Save artifacts
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump([test_metrics], f, indent=2)

    pd.DataFrame([test_metrics]).to_csv(results_dir / "summary.csv", index=False)
    per_image_df.to_csv(results_dir / "per_image_metrics.csv", index=False)

    with open(results_dir / "ablations.json", "w", encoding="utf-8") as f:
        json.dump(ablations, f, indent=2)

    # Metadata record
    qat_meta = {
        "architecture": "Quality-Aware Temporal Change Detection (QAT-CD)",
        "base_detector": "M9_BIT_CD",
        "base_parameters": 552097,
        "quality_parameters": 0,  # Non-trainable decision layer
        "total_parameters": 552097,
        "calibrated_gamma": best_gamma,
        "calibrated_threshold": best_tau,
        "validation_f1": calib_meta["best_val_f1"],
        "test_f1": test_metrics["f1"],
        "test_iou": test_metrics["iou"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "weights": {"reg": w_reg, "illum": w_illum, "sharp": w_sharp},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ckpt_dir / "qat_cd_metadata.json", "w", encoding="utf-8") as f:
        json.dump(qat_meta, f, indent=2)

    # Step 4: Comprehensive Comparison Table (B1, B2, B3, M7, M8, M9, M10)
    comparison_records = []
    m5_summary_path = root / "experiments" / "results" / "m5" / "summary.json"
    if m5_summary_path.exists():
        with open(m5_summary_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                if "Frozen" in r.get("threshold_source", ""):
                    comparison_records.append(r)

    m7_summary_path = root / "experiments" / "results" / "m7" / "summary.json"
    if m7_summary_path.exists():
        with open(m7_summary_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                comparison_records.append(r)

    m8_summary_path = root / "experiments" / "results" / "m8" / "summary.json"
    if m8_summary_path.exists():
        with open(m8_summary_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                comparison_records.append(r)

    m9_summary_path = root / "experiments" / "results" / "m9" / "summary.json"
    if m9_summary_path.exists():
        with open(m9_summary_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                comparison_records.append(r)

    comparison_records.append(test_metrics)

    with open(results_dir / "comparison_m5_to_m10.json", "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    pd.DataFrame(comparison_records).to_csv(results_dir / "comparison_m5_to_m10.csv", index=False)

    # Step 5: Generate Research Figures
    # 1. Validation Surface / Curves
    plt.figure(figsize=(8, 5))
    grid_df = pd.DataFrame(calib_meta["grid_results"])
    for g_val in gamma_candidates:
        sub = grid_df[grid_df["gamma"] == g_val].sort_values("threshold")
        plt.plot(sub["threshold"], sub["f1"], marker="o", label=f"gamma = {g_val:.1f}")
    plt.axvline(best_tau, color="red", linestyle="--", label=f"Optimal tau* = {best_tau:.4f}")
    plt.xlabel("Decision Threshold tau")
    plt.ylabel("Validation F1 Score")
    plt.title("M10 QAT-CD Validation Calibration: Gamma vs Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "validation_calibration_surface.png", dpi=300)
    plt.close()

    # 2. Detector Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    methods = [r.get("method", f"M_{i}") for i, r in enumerate(comparison_records)]
    f1_scores = [r.get("f1", 0.0) for r in comparison_records]
    precisions = [r.get("precision", 0.0) for r in comparison_records]
    recalls = [r.get("recall", 0.0) for r in comparison_records]

    x = np.arange(len(methods))
    w = 0.25
    plt.bar(x - w, precisions, width=w, label="Precision", color="#3498db")
    plt.bar(x, recalls, width=w, label="Recall", color="#e67e22")
    plt.bar(x + w, f1_scores, width=w, label="F1 Score", color="#2ecc71")
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Score [0, 1]")
    plt.title("LEVIR-CD Benchmark: Classical Baselines vs Learned Models vs M10 QAT-CD")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "detector_comparison_m5_to_m10.png", dpi=300)
    plt.close()

    # 3. Ablation Comparison Bar Chart
    plt.figure(figsize=(9, 4.5))
    ab_names = list(ablations.keys())
    ab_f1s = [ablations[k]["f1"] for k in ab_names]
    ab_ious = [ablations[k]["iou"] for k in ab_names]
    ab_prec = [ablations[k]["precision"] for k in ab_names]

    xa = np.arange(len(ab_names))
    wa = 0.25
    plt.bar(xa - wa, ab_prec, width=wa, label="Precision", color="#2980b9")
    plt.bar(xa, ab_f1s, width=wa, label="F1 Score", color="#27ae60")
    plt.bar(xa + wa, ab_ious, width=wa, label="IoU", color="#8e44ad")
    plt.xticks(xa, ab_names, rotation=20, ha="right")
    plt.ylabel("Metric Score")
    plt.title("M10 QAT-CD Test Ablations: Quality Signals & Uncertainty Modulation")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_f1_comparison.png", dpi=300)
    plt.close()

    print("\n[M10 Complete] All artifacts successfully generated and saved to experiments/results/m10/ and experiments/figures/m10/.")


if __name__ == "__main__":
    main()
