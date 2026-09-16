"""Milestone 5: Classical Bi-Temporal Change Detection Baselines Benchmark Runner.

Executes:
1. Validation-only threshold optimization (B1, B2, B3, and secondary Otsu) on the 64 LEVIR-CD validation pairs.
2. Freezes thresholds.
3. Evaluates frozen thresholds on the 128 LEVIR-CD test pairs patch-by-patch (reconstructed to 1024x1024).
4. Measures CPU execution latency and peak memory usage.
5. Saves all machine-readable artifacts and qualitative figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.levir_loader import LEVIRDataset
from src.data.patch_extractor import PatchExtractor
from src.change_detection.pixel_diff import PixelDiffDetector
from src.change_detection.ssim_detector import SSIMDetector
from src.change_detection.cva_detector import CVADetector
from src.change_detection.thresholding import (
    ValidationThresholdOptimizer,
    ThresholdSearchResult,
)
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from src.change_detection.profiler import ChangeDetectionProfiler


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_benchmark(config_path: str = "experiments/configs/m5_change_detection.yaml") -> Dict[str, Any]:
    config = load_config(config_path)

    raw_dir = Path(config["dataset"]["raw_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    patch_size = config["dataset"]["patch_size"]
    stride = config["dataset"]["stride"]
    padding_mode = config["dataset"]["padding_mode"]
    extractor = PatchExtractor(patch_size=patch_size, stride=stride, padding_mode=padding_mode)

    print("=================================================================")
    print("MILESTONE 5: CLASSICAL BI-TEMPORAL CHANGE DETECTION BENCHMARK")
    print("=================================================================")
    print(f"Loading LEVIR-CD dataset from: {raw_dir}")

    # Load validation and test datasets
    val_dataset = LEVIRDataset(root_dir=raw_dir, split="val")
    test_dataset = LEVIRDataset(root_dir=raw_dir, split="test")

    print(f"Validation pairs loaded: {len(val_dataset)} (Expected: 64)")
    print(f"Test pairs loaded:       {len(test_dataset)} (Expected: 128)")
    assert len(val_dataset) == 64, f"Expected 64 val pairs, got {len(val_dataset)}"
    assert len(test_dataset) == 128, f"Expected 128 test pairs, got {len(test_dataset)}"

    # Instantiate detectors
    detectors = {
        "B1_Pixel_Diff": PixelDiffDetector(
            aggregation_mode=config["methods"]["b1_pixel_diff"]["aggregation_mode"]
        ),
        "B2_SSIM": SSIMDetector(
            win_size=config["methods"]["b2_ssim"]["win_size"],
            sigma=config["methods"]["b2_ssim"]["sigma"],
            channel_mode=config["methods"]["b2_ssim"]["channel_mode"],
        ),
        "B3_CVA": CVADetector(
            normalize=config["methods"]["b3_cva"]["normalize"]
        ),
    }

    # =========================================================================
    # PHASE 1: VALIDATION THRESHOLD CALIBRATION (STRICTLY VALIDATION ONLY)
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("PHASE 1: VALIDATION THRESHOLD CALIBRATION (64 PAIRS / 1,024 PATCHES)")
    print("-----------------------------------------------------------------")

    val_optimizers = {
        name: ValidationThresholdOptimizer(
            min_threshold=config["thresholding"]["min_threshold"],
            max_threshold=config["thresholding"]["max_threshold"],
            num_candidates=config["thresholding"]["num_candidates"],
        )
        for name in detectors.keys()
    }

    t0_val = time.perf_counter()
    for idx in range(len(val_dataset)):
        t1, t2, target, meta = val_dataset[idx]
        # Convert tensors to numpy
        img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_mask = target.numpy().astype(np.uint8)

        # Extract 256x256 patches
        patches = extractor.extract_from_pair(img_a, img_b, gt_mask, parent_stem=meta["sample_id"])

        for name, detector in detectors.items():
            diff_patches = []
            for p_a, p_b, p_lbl, p_meta in patches:
                diff_patch = detector.compute_difference_map(p_a, p_b)
                diff_patches.append((diff_patch, p_meta))

            # Reconstruct full 1024x1024 difference map
            full_diff = extractor.reconstruct_image(
                diff_patches, original_height=1024, original_width=1024, dtype=np.float32
            )
            # Update validation optimizer
            val_optimizers[name].update(full_diff, gt_mask)

        if (idx + 1) % 16 == 0 or (idx + 1) == len(val_dataset):
            print(f"  Processed {idx + 1}/64 validation pairs...")

    val_time = time.perf_counter() - t0_val
    print(f"Validation calibration complete in {val_time:.2f}s.")

    # Compute optimal validation thresholds and Otsu thresholds
    thresholds_dict: Dict[str, Any] = {}
    for name, opt in val_optimizers.items():
        res = opt.optimize()
        otsu_val = opt.compute_otsu_threshold() if config["thresholding"]["evaluate_otsu_secondary"] else None
        thresholds_dict[name] = {
            "validation_f1_optimal": res.to_dict(),
            "otsu_secondary_threshold": otsu_val,
            "f1_curve": res.f1_curve,
            "candidate_thresholds": res.candidate_thresholds,
            "precision_curve": res.precision_curve,
            "recall_curve": res.recall_curve,
        }
        print(f"\n[{name}] Validation Calibration:")
        print(f"  Optimal tau* (F1 max): {res.best_threshold:.4f} (Val F1: {res.best_f1:.4f}, P: {res.best_precision:.4f}, R: {res.best_recall:.4f}, IoU: {res.best_iou:.4f})")
        if otsu_val is not None:
            print(f"  Secondary Otsu tau:   {otsu_val:.4f}")

    # Plot and save validation threshold curves
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, data) in enumerate(thresholds_dict.items()):
        cand = data["candidate_thresholds"]
        ax[i].plot(cand, data["f1_curve"], label="Validation F1", color="crimson", lw=2)
        ax[i].plot(cand, data["precision_curve"], label="Precision", color="dodgerblue", lw=1.5, ls="--")
        ax[i].plot(cand, data["recall_curve"], label="Recall", color="forestgreen", lw=1.5, ls=":")
        best_tau = data["validation_f1_optimal"]["best_threshold"]
        best_f1 = data["validation_f1_optimal"]["best_f1"]
        ax[i].axvline(best_tau, color="black", linestyle="--", label=f"tau*={best_tau:.3f}")
        if data["otsu_secondary_threshold"] is not None:
            ax[i].axvline(data["otsu_secondary_threshold"], color="purple", linestyle="-.", label=f"Otsu={data['otsu_secondary_threshold']:.3f}")
        ax[i].set_title(f"{name} Validation Sweep (Best F1={best_f1:.3f})")
        ax[i].set_xlabel("Threshold tau")
        ax[i].set_ylabel("Metric")
        ax[i].grid(True, alpha=0.3)
        ax[i].legend(loc="best")
    plt.tight_layout()
    plt.savefig(figures_dir / "validation_threshold_curves.png", dpi=150)
    plt.close()

    # Save thresholds.json
    with open(results_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)

    # =========================================================================
    # PHASE 2: TEST EVALUATION (128 PAIRS / 2,048 PATCHES WITH FROZEN TAU)
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("PHASE 2: TEST SPLIT EVALUATION (128 PAIRS / 2,048 PATCHES / 134M PIXELS)")
    print("-----------------------------------------------------------------")

    # Metrics containers
    confusion_totals: Dict[str, Dict[str, ConfusionMatrix]] = {
        name: {
            "validation_f1": ConfusionMatrix(),
            "otsu": ConfusionMatrix(),
        }
        for name in detectors.keys()
    }

    profilers: Dict[str, ChangeDetectionProfiler] = {
        name: ChangeDetectionProfiler() for name in detectors.keys()
    }

    per_image_records: List[Dict[str, Any]] = []
    qualitative_samples: Dict[str, Dict[str, Any]] = {}

    # Target test pair IDs for deterministic qualitative analysis
    qualitative_target_ids = {"test_1", "test_20", "test_50", "test_100"}

    for name in detectors.keys():
        profilers[name].start_pipeline()

    t0_test = time.perf_counter()

    for idx in range(len(test_dataset)):
        t_pair_start = time.perf_counter()
        t1, t2, target, meta = test_dataset[idx]
        sample_id = meta["sample_id"]

        # Preprocessing: convert tensors to uint8 RGB
        t_prep_start = time.perf_counter()
        img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_mask = target.numpy().astype(np.uint8)
        prep_time = time.perf_counter() - t_prep_start

        # Extract 16 patches
        patches = extractor.extract_from_pair(img_a, img_b, gt_mask, parent_stem=sample_id)

        sample_preds_for_viz = {}

        for name, detector in detectors.items():
            prof = profilers[name]
            prof.record_stage("preprocessing", prep_time)

            frozen_tau = thresholds_dict[name]["validation_f1_optimal"]["best_threshold"]
            otsu_tau = thresholds_dict[name]["otsu_secondary_threshold"]

            # Difference computation patch-by-patch
            t_diff_start = time.perf_counter()
            diff_patches = []
            for p_a, p_b, p_lbl, p_meta in patches:
                d_patch = detector.compute_difference_map(p_a, p_b)
                diff_patches.append((d_patch, p_meta))
            diff_time = time.perf_counter() - t_diff_start
            prof.record_stage("difference_computation", diff_time)

            # Reconstruct full 1024x1024 difference map
            t_rec_start = time.perf_counter()
            full_diff = extractor.reconstruct_image(
                diff_patches, original_height=1024, original_width=1024, dtype=np.float32
            )
            rec_time = time.perf_counter() - t_rec_start
            prof.record_stage("reconstruction", rec_time)

            # Thresholding stage
            t_thresh_start = time.perf_counter()
            pred_f1 = (full_diff >= frozen_tau).astype(np.uint8)
            pred_otsu = (full_diff >= otsu_tau).astype(np.uint8) if otsu_tau is not None else None
            thresh_time = time.perf_counter() - t_thresh_start
            prof.record_stage("thresholding", thresh_time)

            # Update confusion matrix
            confusion_totals[name]["validation_f1"].update(pred_f1, gt_mask)
            if pred_otsu is not None:
                confusion_totals[name]["otsu"].update(pred_otsu, gt_mask)

            # Compute per-image metrics for audit
            img_cm = ConfusionMatrix()
            img_cm.update(pred_f1, gt_mask)
            img_eval = ChangeDetectionEvaluator.evaluate(img_cm)
            per_image_records.append({
                "sample_id": sample_id,
                "method": name,
                "threshold_policy": "validation_f1",
                "threshold_value": frozen_tau,
                "precision": img_eval["precision"],
                "recall": img_eval["recall"],
                "f1": img_eval["f1"],
                "iou": img_eval["iou"],
                "tp": img_eval["tp"],
                "fp": img_eval["fp"],
                "fn": img_eval["fn"],
                "tn": img_eval["tn"],
                "gt_changed_pixels": img_eval["gt_changed_pixels"],
                "pred_changed_pixels": img_eval["pred_changed_pixels"],
            })

            pair_total_time = time.perf_counter() - t_pair_start
            prof.record_stage("per_pair_total", pair_total_time)

            sample_preds_for_viz[name] = {
                "pred_f1": pred_f1,
                "diff_map": full_diff,
            }

        # Store qualitative samples if target ID
        if sample_id in qualitative_target_ids:
            qualitative_samples[sample_id] = {
                "img_a": img_a,
                "img_b": img_b,
                "gt_mask": gt_mask,
                "preds": sample_preds_for_viz,
            }

        if (idx + 1) % 32 == 0 or (idx + 1) == len(test_dataset):
            print(f"  Evaluated {idx + 1}/128 test pairs...")

    for name in detectors.keys():
        profilers[name].end_pipeline()

    test_total_time = time.perf_counter() - t0_test
    print(f"\nTest evaluation complete in {test_total_time:.2f}s.")

    # =========================================================================
    # PHASE 3: AGGREGATE RESULTS & COMPOSE AUDIT ARTIFACTS
    # =========================================================================
    summary_rows: List[Dict[str, Any]] = []
    method_json_results: Dict[str, Any] = {}
    confusion_dict: Dict[str, Any] = {}

    for name in detectors.keys():
        prof_summary = profilers[name].summary()
        method_json_results[name] = {}

        # Primary result: validation-F1 frozen threshold
        cm_val = confusion_totals[name]["validation_f1"]
        eval_val = ChangeDetectionEvaluator.evaluate(cm_val)
        tau_val = thresholds_dict[name]["validation_f1_optimal"]["best_threshold"]

        row_val = {
            "method": name,
            "threshold_source": "Validation F1 (Frozen)",
            "threshold_value": tau_val,
            "precision": eval_val["precision"],
            "recall": eval_val["recall"],
            "f1": eval_val["f1"],
            "iou": eval_val["iou"],
            "accuracy": eval_val["accuracy"],
            "tp": eval_val["tp"],
            "fp": eval_val["fp"],
            "fn": eval_val["fn"],
            "tn": eval_val["tn"],
            "total_pixels": eval_val["total_pixels"],
            "gt_changed_pixels": eval_val["gt_changed_pixels"],
            "pred_changed_pixels": eval_val["pred_changed_pixels"],
            "gt_changed_ratio": eval_val["gt_changed_ratio"],
            "pred_changed_ratio": eval_val["pred_changed_ratio"],
            "mean_pair_latency_ms": prof_summary["mean_pair_latency_ms"],
            "p50_pair_latency_ms": prof_summary["p50_pair_latency_ms"],
            "p95_pair_latency_ms": prof_summary["p95_pair_latency_ms"],
            "total_runtime_s": prof_summary["total_runtime_s"],
            "peak_rss_mb": prof_summary["peak_rss_mb"],
        }
        summary_rows.append(row_val)
        method_json_results[name]["primary_validation_f1"] = row_val

        # Secondary result: Otsu threshold
        if config["thresholding"]["evaluate_otsu_secondary"]:
            cm_otsu = confusion_totals[name]["otsu"]
            eval_otsu = ChangeDetectionEvaluator.evaluate(cm_otsu)
            tau_otsu = thresholds_dict[name]["otsu_secondary_threshold"]
            row_otsu = {
                "method": name,
                "threshold_source": "Validation Otsu (Secondary)",
                "threshold_value": tau_otsu,
                "precision": eval_otsu["precision"],
                "recall": eval_otsu["recall"],
                "f1": eval_otsu["f1"],
                "iou": eval_otsu["iou"],
                "accuracy": eval_otsu["accuracy"],
                "tp": eval_otsu["tp"],
                "fp": eval_otsu["fp"],
                "fn": eval_otsu["fn"],
                "tn": eval_otsu["tn"],
                "total_pixels": eval_otsu["total_pixels"],
                "gt_changed_pixels": eval_otsu["gt_changed_pixels"],
                "pred_changed_pixels": eval_otsu["pred_changed_pixels"],
                "gt_changed_ratio": eval_otsu["gt_changed_ratio"],
                "pred_changed_ratio": eval_otsu["pred_changed_ratio"],
                "mean_pair_latency_ms": prof_summary["mean_pair_latency_ms"],
                "p50_pair_latency_ms": prof_summary["p50_pair_latency_ms"],
                "p95_pair_latency_ms": prof_summary["p95_pair_latency_ms"],
                "total_runtime_s": prof_summary["total_runtime_s"],
                "peak_rss_mb": prof_summary["peak_rss_mb"],
            }
            summary_rows.append(row_otsu)
            method_json_results[name]["secondary_otsu"] = row_otsu

        confusion_dict[name] = {
            "validation_f1": cm_val.to_dict(),
            "otsu": confusion_totals[name]["otsu"].to_dict() if config["thresholding"]["evaluate_otsu_secondary"] else None,
        }

    # Save summary.csv
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(results_dir / "summary.csv", index=False)

    # Save summary.json
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    # Save per-method results
    with open(results_dir / "b1_results.json", "w", encoding="utf-8") as f:
        json.dump(method_json_results["B1_Pixel_Diff"], f, indent=2)
    with open(results_dir / "b2_results.json", "w", encoding="utf-8") as f:
        json.dump(method_json_results["B2_SSIM"], f, indent=2)
    with open(results_dir / "b3_results.json", "w", encoding="utf-8") as f:
        json.dump(method_json_results["B3_CVA"], f, indent=2)

    # Save confusion_totals.json
    with open(results_dir / "confusion_totals.json", "w", encoding="utf-8") as f:
        json.dump(confusion_dict, f, indent=2)

    # Save per_image_metrics.csv
    df_per_image = pd.DataFrame(per_image_records)
    df_per_image.to_csv(results_dir / "per_image_metrics.csv", index=False)

    # =========================================================================
    # PHASE 4: GENERATE QUALITATIVE COMPARISON FIGURES
    # =========================================================================
    print("\n-----------------------------------------------------------------")
    print("PHASE 4: GENERATING QUALITATIVE COMPARISON FIGURES")
    print("-----------------------------------------------------------------")

    for sample_id, sample_data in qualitative_samples.items():
        img_a = sample_data["img_a"]
        img_b = sample_data["img_b"]
        gt_mask = sample_data["gt_mask"]
        preds = sample_data["preds"]

        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        axes[0].imshow(img_a)
        axes[0].set_title(f"T1 ({sample_id})")
        axes[0].axis("off")

        axes[1].imshow(img_b)
        axes[1].set_title("T2")
        axes[1].axis("off")

        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

        axes[3].imshow(preds["B1_Pixel_Diff"]["pred_f1"], cmap="gray")
        axes[3].set_title("B1 Pixel Diff Pred")
        axes[3].axis("off")

        axes[4].imshow(preds["B2_SSIM"]["pred_f1"], cmap="gray")
        axes[4].set_title("B2 SSIM Pred")
        axes[4].axis("off")

        axes[5].imshow(preds["B3_CVA"]["pred_f1"], cmap="gray")
        axes[5].set_title("B3 CVA Pred")
        axes[5].axis("off")

        plt.tight_layout()
        plt.savefig(figures_dir / f"qualitative_comparison_{sample_id}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Generated qualitative figure for: {sample_id}")

    print("\n=================================================================")
    print("M5 BENCHMARK COMPLETE — EMPIRICAL RESULTS SUMMARY")
    print("=================================================================")
    print(df_summary[["method", "threshold_source", "threshold_value", "precision", "recall", "f1", "iou", "mean_pair_latency_ms", "peak_rss_mb"]].to_string(index=False))
    print("=================================================================\n")

    return {
        "summary": summary_rows,
        "thresholds": thresholds_dict,
        "confusion_totals": confusion_dict,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 5 classical change detection benchmark.")
    parser.add_argument("--config", type=str, default="experiments/configs/m5_change_detection.yaml", help="Path to config yaml")
    args = parser.parse_args()
    run_benchmark(args.config)
