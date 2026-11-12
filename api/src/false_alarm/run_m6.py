"""Milestone 6: False-Alarm Analysis & Perturbation Robustness Benchmark Runner.

Executes:
1. Unperturbed Control evaluation on 128 LEVIR-CD test pairs using frozen M5 thresholds.
2. Controlled perturbation evaluation across 4 families x 3 severity tiers (12 conditions).
3. Degradation and additional false-positive metric calculations.
4. CPU profiling and memory measurement.
5. Generation of machine-readable artifacts and research figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Dict, List, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.levir_loader import LEVIRDataset
from src.data.patch_extractor import PatchExtractor
from src.change_detection.pixel_diff import PixelDiffDetector
from src.change_detection.ssim_detector import SSIMDetector
from src.change_detection.cva_detector import CVADetector
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from src.change_detection.profiler import ChangeDetectionProfiler

from src.false_alarm.illumination import IlluminationShiftPerturbation
from src.false_alarm.blur import GaussianBlurPerturbation
from src.false_alarm.misregistration import GeometricMisregistrationPerturbation
from src.false_alarm.occlusion import LocalizedOcclusionShadowPerturbation
from src.false_alarm.robustness_metrics import RobustnessMetricsCalculator


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_benchmark(config_path: str = "experiments/configs/m6_false_alarm.yaml") -> Dict[str, Any]:
    config = load_config(config_path)

    raw_dir = Path(config["dataset"]["raw_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to results directory for exact archival
    shutil.copy(config_path, results_dir / "m6_config.yaml")

    patch_size = config["dataset"]["patch_size"]
    stride = config["dataset"]["stride"]
    padding_mode = config["dataset"]["padding_mode"]
    extractor = PatchExtractor(patch_size=patch_size, stride=stride, padding_mode=padding_mode)

    print("=================================================================")
    print("MILESTONE 6: FALSE-ALARM ANALYSIS & PERTURBATION ROBUSTNESS")
    print("=================================================================")
    print(f"Loading LEVIR-CD test dataset from: {raw_dir}")

    test_dataset = LEVIRDataset(root_dir=raw_dir, split=config["dataset"]["split"])
    print(f"Test pairs loaded: {len(test_dataset)} (Expected: 128)")
    assert len(test_dataset) == 128, f"Expected 128 test pairs, got {len(test_dataset)}"

    # Instantiate detectors matching M5 configurations
    detectors = {
        "B1_Pixel_Diff": PixelDiffDetector(
            aggregation_mode=config["methods"]["B1_Pixel_Diff"]["aggregation_mode"]
        ),
        "B2_SSIM": SSIMDetector(
            win_size=config["methods"]["B2_SSIM"]["win_size"],
            sigma=config["methods"]["B2_SSIM"]["sigma"],
            channel_mode=config["methods"]["B2_SSIM"]["channel_mode"],
        ),
        "B3_CVA": CVADetector(
            normalize=config["methods"]["B3_CVA"]["normalize"]
        ),
    }

    # Frozen M5 validation thresholds
    frozen_thresholds: Dict[str, float] = config["frozen_thresholds"]
    print("\nEnforcing Frozen M5 Validation Thresholds:")
    for det_name, tau in frozen_thresholds.items():
        print(f"  {det_name:15s} -> tau* = {tau:.4f}")

    # Instantiate perturbation families
    perturbation_registry = {
        "global_illumination_shift": IlluminationShiftPerturbation(),
        "gaussian_blur": GaussianBlurPerturbation(),
        "geometric_misregistration": GeometricMisregistrationPerturbation(),
        "localized_occlusion_shadow": LocalizedOcclusionShadowPerturbation(),
    }

    severity_levels = ["mild", "medium", "strong"]

    # Storage for all evaluation rows
    all_experiment_records: List[Dict[str, Any]] = []

    # =========================================================================
    # STAGE 1: UNPERTURBED CONTROL EVALUATION
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("STAGE 1: CONTROL EVALUATION (128 UNMODIFIED TEST PAIRS)")
    print("-----------------------------------------------------------------")

    control_metrics: Dict[str, Dict[str, Any]] = {}
    ctrl_confusion: Dict[str, ConfusionMatrix] = {name: ConfusionMatrix() for name in detectors.keys()}
    profiler_ctrl = ChangeDetectionProfiler()
    profiler_ctrl.start_pipeline()

    for idx in range(len(test_dataset)):
        t1, t2, target, meta = test_dataset[idx]
        img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_mask = target.numpy().astype(np.uint8)

        patches = extractor.extract_from_pair(img_a, img_b, gt_mask, parent_stem=meta["sample_id"])

        for name, detector in detectors.items():
            tau = frozen_thresholds[name]
            diff_patches = []
            for p_a, p_b, p_lbl, p_meta in patches:
                d_patch = detector.compute_difference_map(p_a, p_b)
                diff_patches.append((d_patch, p_meta))

            full_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
            pred = (full_diff >= tau).astype(np.uint8)
            ctrl_confusion[name].update(pred, gt_mask)

    profiler_ctrl.end_pipeline()
    ctrl_prof_summary = profiler_ctrl.summary()

    for name in detectors.keys():
        m = ChangeDetectionEvaluator.evaluate(ctrl_confusion[name])
        control_metrics[name] = m
        record = {
            "detector": name,
            "perturbation": "control",
            "severity": "none",
            "split": "test",
            "threshold_policy": "frozen_m5_val_f1",
            "threshold": frozen_thresholds[name],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "iou": m["iou"],
            "accuracy": m["accuracy"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
            "f1_delta": 0.0,
            "relative_f1_degradation_pct": 0.0,
            "iou_delta": 0.0,
            "relative_iou_degradation_pct": 0.0,
            "additional_false_positives": 0,
            "relative_fp_increase_pct": 0.0,
            "fp_generation_rate_pct": 0.0,
            "latency_ms_per_pair": ctrl_prof_summary["mean_pair_latency_ms"],
            "peak_ram_mb": ctrl_prof_summary["peak_rss_mb"],
            "seed": config.get("seed", 42),
        }
        all_experiment_records.append(record)
        print(f"  [CONTROL] {name:15s} -> F1: {m['f1']:.4f}, IoU: {m['iou']:.4f}, P: {m['precision']:.4f}, R: {m['recall']:.4f}, FP: {m['fp']:,}")

    # =========================================================================
    # STAGE 2: PERTURBATION ROBUSTNESS EVALUATION (12 CONDITIONS)
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("STAGE 2: PERTURBED EVALUATIONS (4 FAMILIES x 3 SEVERITIES x 3 DETECTORS)")
    print("-----------------------------------------------------------------")

    qualitative_target_ids = {"test_1", "test_20", "test_50", "test_100"}
    qualitative_figures_data: Dict[str, Dict[str, Any]] = {pid: {} for pid in qualitative_target_ids}

    seed = config.get("seed", 42)

    for pert_name, pert_obj in perturbation_registry.items():
        print(f"\nEvaluating Perturbation Family: {pert_name}")

        for severity in severity_levels:
            t0_cond = time.perf_counter()
            pert_confusion: Dict[str, ConfusionMatrix] = {name: ConfusionMatrix() for name in detectors.keys()}
            prof_pert = ChangeDetectionProfiler()
            prof_pert.start_pipeline()

            for idx in range(len(test_dataset)):
                t_pair_start = time.perf_counter()
                t1, t2, target, meta = test_dataset[idx]
                sample_id = meta["sample_id"]

                img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                gt_mask = target.numpy().astype(np.uint8)

                # Apply perturbation to T2
                p_img_a, p_img_b, pert_meta = pert_obj.apply(
                    img_a, img_b, severity=severity, target_image="t2", seed=seed + idx
                )

                # Extract patches from perturbed pair
                patches = extractor.extract_from_pair(p_img_a, p_img_b, gt_mask, parent_stem=sample_id)

                sample_preds = {}

                for name, detector in detectors.items():
                    tau = frozen_thresholds[name]
                    diff_patches = []
                    for p_a, p_b, p_lbl, p_meta in patches:
                        d_patch = detector.compute_difference_map(p_a, p_b)
                        diff_patches.append((d_patch, p_meta))

                    full_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
                    pred = (full_diff >= tau).astype(np.uint8)
                    pert_confusion[name].update(pred, gt_mask)

                    if sample_id in qualitative_target_ids and severity == "medium":
                        sample_preds[name] = pred

                pair_elapsed = time.perf_counter() - t_pair_start
                prof_pert.record_stage("per_pair_total", pair_elapsed)

                # Save qualitative medium samples
                if sample_id in qualitative_target_ids and severity == "medium":
                    qualitative_figures_data[sample_id][pert_name] = {
                        "img_a": img_a,
                        "img_b_orig": img_b,
                        "img_b_pert": p_img_b,
                        "gt_mask": gt_mask,
                        "preds": sample_preds,
                    }

            prof_pert.end_pipeline()
            pert_prof_summary = prof_pert.summary()
            cond_elapsed = time.perf_counter() - t0_cond

            # Calculate metrics and degradation for each detector
            for name in detectors.keys():
                m = ChangeDetectionEvaluator.evaluate(pert_confusion[name])
                deg = RobustnessMetricsCalculator.compute_degradation(control_metrics[name], m)

                record = {
                    "detector": name,
                    "perturbation": pert_name,
                    "severity": severity,
                    "split": "test",
                    "threshold_policy": "frozen_m5_val_f1",
                    "threshold": frozen_thresholds[name],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "iou": m["iou"],
                    "accuracy": m["accuracy"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tn": m["tn"],
                    "f1_delta": deg["f1_delta"],
                    "relative_f1_degradation_pct": deg["relative_f1_degradation_pct"],
                    "iou_delta": deg["iou_delta"],
                    "relative_iou_degradation_pct": deg["relative_iou_degradation_pct"],
                    "additional_false_positives": deg["additional_false_positives"],
                    "relative_fp_increase_pct": deg["relative_fp_increase_pct"],
                    "fp_generation_rate_pct": deg["fp_generation_rate_pct"],
                    "latency_ms_per_pair": pert_prof_summary["mean_pair_latency_ms"],
                    "peak_ram_mb": pert_prof_summary["peak_rss_mb"],
                    "seed": seed,
                }
                all_experiment_records.append(record)

                print(
                    f"  [{severity.upper():6s}] {name:15s} | "
                    f"F1: {m['f1']:.4f} (deg: {deg['relative_f1_degradation_pct']:+6.1f}%) | "
                    f"FP: {m['fp']:,} (+{deg['additional_false_positives']:,}) | "
                    f"IoU: {m['iou']:.4f}"
                )

    # =========================================================================
    # STAGE 3: EXPORT MACHINE-READABLE RESULTS
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("STAGE 3: EXPORTING MACHINE-READABLE ARTIFACTS")
    print("-----------------------------------------------------------------")

    df_results = pd.DataFrame(all_experiment_records)
    csv_path = results_dir / "m6_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  Saved full results table: {csv_path}")

    # Build concise degradation summary table
    degradation_cols = [
        "detector", "perturbation", "severity", "precision", "recall",
        "f1", "relative_f1_degradation_pct", "iou", "relative_iou_degradation_pct",
        "fp", "additional_false_positives", "fp_generation_rate_pct"
    ]
    df_degradation = df_results[degradation_cols].copy()
    degradation_csv = results_dir / "degradation_table.csv"
    df_degradation.to_csv(degradation_csv, index=False)
    print(f"  Saved degradation summary: {degradation_csv}")

    # Summary JSON with nested structure
    summary_json_path = results_dir / "m6_summary.json"
    summary_dict = {
        "control_metrics": control_metrics,
        "experiments": all_experiment_records,
        "config": config,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"  Saved summary JSON: {summary_json_path}")

    # =========================================================================
    # STAGE 4: RESEARCH VISUALIZATIONS
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("STAGE 4: GENERATING RESEARCH VISUALIZATIONS")
    print("-----------------------------------------------------------------")

    # 1. F1 vs Perturbation Severity across families
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, pert_name in enumerate(perturbation_registry.keys()):
        ax = axes[i]
        sub = df_results[df_results["perturbation"] == pert_name]
        ctrl_sub = df_results[df_results["perturbation"] == "control"]

        for det_name, color, marker in [
            ("B1_Pixel_Diff", "#1f77b4", "o"),
            ("B2_SSIM", "#2ca02c", "s"),
            ("B3_CVA", "#d62728", "^"),
        ]:
            c_val = ctrl_sub[ctrl_sub["detector"] == det_name]["f1"].values[0]
            p_vals = sub[sub["detector"] == det_name]["f1"].values.tolist()
            x_ticks = ["Control", "Mild", "Medium", "Strong"]
            y_vals = [c_val] + p_vals
            ax.plot(x_ticks, y_vals, marker=marker, lw=2, color=color, label=det_name)

        ax.set_title(f"F1 vs Severity: {pert_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Test F1 Score (Changed Class)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_ylim(-0.01, 0.16)

    plt.tight_layout()
    plt.savefig(figures_dir / "f1_vs_severity.png", dpi=150)
    plt.close()
    print("  Saved: f1_vs_severity.png")

    # 2. Additional False Positives vs Severity
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, pert_name in enumerate(perturbation_registry.keys()):
        ax = axes[i]
        sub = df_results[df_results["perturbation"] == pert_name]

        for det_name, color, marker in [
            ("B1_Pixel_Diff", "#1f77b4", "o"),
            ("B2_SSIM", "#2ca02c", "s"),
            ("B3_CVA", "#d62728", "^"),
        ]:
            delta_fp_vals = (sub[sub["detector"] == det_name]["additional_false_positives"].values / 1e6).tolist()
            x_ticks = ["Mild", "Medium", "Strong"]
            ax.plot(x_ticks, delta_fp_vals, marker=marker, lw=2, color=color, label=det_name)

        ax.set_title(f"Additional FP Pixels (Millions): {pert_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Delta FP Pixels (Millions)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(figures_dir / "additional_fp_vs_severity.png", dpi=150)
    plt.close()
    print("  Saved: additional_fp_vs_severity.png")

    # 3. Relative F1 Degradation Bar Chart (Strong Severity)
    df_strong = df_results[df_results["severity"] == "strong"].copy()
    fig, ax = plt.subplots(figsize=(14, 6))

    pert_order = list(perturbation_registry.keys())
    x = np.arange(len(pert_order))
    width = 0.25

    for idx, (det_name, color) in enumerate([
        ("B1_Pixel_Diff", "#1f77b4"),
        ("B2_SSIM", "#2ca02c"),
        ("B3_CVA", "#d62728"),
    ]):
        vals = []
        for p in pert_order:
            sub = df_strong[(df_strong["detector"] == det_name) & (df_strong["perturbation"] == p)]
            deg = sub["relative_f1_degradation_pct"].values[0] if len(sub) > 0 else 0.0
            vals.append(deg)
        ax.bar(x + idx * width, vals, width, label=det_name, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([p.replace("_", "\n").title() for p in pert_order], fontsize=11)
    ax.set_ylabel("Relative F1 Degradation (%) [Higher = More Vulnerable]", fontsize=11)
    ax.set_title("Relative F1 Degradation under Strong Perturbations", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figures_dir / "relative_f1_degradation.png", dpi=150)
    plt.close()
    print("  Saved: relative_f1_degradation.png")

    # 4. Detector x Perturbation Sensitivity Heatmap (Relative F1 Degradation %)
    pert_conditions = []
    for p in perturbation_registry.keys():
        for s in severity_levels:
            pert_conditions.append((p, s))

    heatmap_data = np.zeros((len(detectors), len(pert_conditions)))
    det_names = list(detectors.keys())

    for i, d in enumerate(det_names):
        for j, (p, s) in enumerate(pert_conditions):
            match = df_results[(df_results["detector"] == d) & (df_results["perturbation"] == p) & (df_results["severity"] == s)]
            if len(match) > 0:
                heatmap_data[i, j] = match["relative_f1_degradation_pct"].values[0]

    fig, ax = plt.subplots(figsize=(18, 5))
    cax = ax.matshow(heatmap_data, cmap="YlOrRd", vmin=0, vmax=100)
    fig.colorbar(cax, ax=ax, label="Relative F1 Degradation (%)")

    ax.set_xticks(np.arange(len(pert_conditions)))
    ax.set_yticks(np.arange(len(det_names)))
    labels = [f"{p.split('_')[0]}\n{s[0].upper()}" for p, s in pert_conditions]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(det_names, fontsize=10)

    for i in range(len(det_names)):
        for j in range(len(pert_conditions)):
            val = heatmap_data[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color="black" if val < 50 else "white", fontsize=8)

    ax.set_title("Detector Vulnerability Matrix (Relative F1 Degradation % across Conditions)", pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "detector_perturbation_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: detector_perturbation_heatmap.png")

    # 5. Qualitative Representative Comparisons (T1, T2_orig, T2_pert, GT, B1, B2, B3)
    for sample_id, p_dict in qualitative_figures_data.items():
        for pert_name, sdata in p_dict.items():
            img_a = sdata["img_a"]
            img_b_orig = sdata["img_b_orig"]
            img_b_pert = sdata["img_b_pert"]
            gt = sdata["gt_mask"]
            preds = sdata["preds"]

            fig, axes = plt.subplots(1, 7, figsize=(28, 4))
            axes[0].imshow(img_a)
            axes[0].set_title(f"T1 ({sample_id})")
            axes[0].axis("off")

            axes[1].imshow(img_b_orig)
            axes[1].set_title("Original T2")
            axes[1].axis("off")

            axes[2].imshow(img_b_pert)
            axes[2].set_title(f"Perturbed T2\n({pert_name} Med)")
            axes[2].axis("off")

            axes[3].imshow(gt, cmap="gray")
            axes[3].set_title("Ground Truth Mask")
            axes[3].axis("off")

            axes[4].imshow(preds["B1_Pixel_Diff"], cmap="gray")
            axes[4].set_title("B1 Prediction")
            axes[4].axis("off")

            axes[5].imshow(preds["B2_SSIM"], cmap="gray")
            axes[5].set_title("B2 Prediction")
            axes[5].axis("off")

            axes[6].imshow(preds["B3_CVA"], cmap="gray")
            axes[6].set_title("B3 Prediction")
            axes[6].axis("off")

            plt.tight_layout()
            out_path = figures_dir / f"qualitative_{sample_id}_{pert_name}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

    print("  Saved representative qualitative figures.")

    print("\n=================================================================")
    print("M6 BENCHMARK COMPLETE — SUMMARY REPORT OF MEASURED DEGRADATIONS")
    print("=================================================================")
    print(df_degradation.to_string(index=False))
    print("=================================================================\n")

    return {
        "results_table": all_experiment_records,
        "control_metrics": control_metrics,
        "degradation_df": df_degradation,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 6 False-Alarm & Perturbation Benchmark.")
    parser.add_argument("--config", type=str, default="experiments/configs/m6_false_alarm.yaml", help="Path to config yaml")
    args = parser.parse_args()
    run_benchmark(args.config)
