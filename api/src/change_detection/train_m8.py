"""Training and Evaluation Pipeline for Milestone M8 SNUNet Change Detection Model.

Executes end-to-end reproducible workflow on LEVIR-CD:
1. Balanced patch training on official train split (445 scenes = 7,120 patches).
2. Validation split loss and F1 monitoring (64 scenes = 1,024 patches).
3. Best checkpoint selection strictly based on validation F1 score.
4. Validation threshold search (19 candidate thresholds) to freeze optimal tau*.
5. Official held-out test split evaluation (128 scenes = 2,048 patches).
6. Comprehensive comparison against B1, B2, B3, and M7 baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.change_detection.snunet import (
    SNUNet,
    SNUNetDetector,
    save_checkpoint,
    load_checkpoint,
)
from src.change_detection.losses import BCEDiceLoss
from src.change_detection.patch_dataset import LEVIRPatchDataset
from src.data.levir_loader import LEVIRDataset
from src.data.patch_extractor import PatchExtractor
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from utils.config_loader import get_project_root, load_m8_config


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BalancedPatchSampler(Sampler[int]):
    """Samples equal number of positive and negative change patches per epoch."""

    def __init__(
        self,
        positive_indices: List[int],
        negative_indices: List[int],
        samples_per_epoch: int = 512,
        seed: int = 42,
    ):
        super().__init__(None)
        self.pos_idx = positive_indices
        self.neg_idx = negative_indices
        self.samples_per_epoch = samples_per_epoch
        self.half_samples = samples_per_epoch // 2
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000)
        sampled_pos = rng.sample(self.pos_idx, min(self.half_samples, len(self.pos_idx)))
        sampled_neg = rng.sample(self.neg_idx, min(self.half_samples, len(self.neg_idx)))
        combined = sampled_pos + sampled_neg
        rng.shuffle(combined)
        return iter(combined)

    def __len__(self) -> int:
        return self.samples_per_epoch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Runs one training epoch and returns mean batch loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for t1, t2, target in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(t1, t2)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_val_loss_and_f1(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
) -> Tuple[float, float]:
    """Computes validation loss and micro-averaged F1 score over validation patches."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    cm = ConfusionMatrix()

    for t1, t2, target in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        target = target.to(device)

        logits = model(t1, t2)
        loss = loss_fn(logits, target)
        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long().squeeze(1).cpu().numpy()
        targets = target.long().squeeze(1).cpu().numpy()

        for b in range(preds.shape[0]):
            cm.update(preds[b], targets[b])

    metrics = ChangeDetectionEvaluator.evaluate(cm)
    mean_loss = total_loss / max(num_batches, 1)
    return mean_loss, metrics["f1"]


def optimize_validation_threshold(
    model: SNUNet,
    val_dataset: LEVIRDataset,
    patch_extractor: PatchExtractor,
    device: torch.device,
    candidate_thresholds: List[float],
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates validation scenes across candidate thresholds to select optimal tau*."""
    detector = SNUNetDetector(model=model, device=str(device))
    print(f"\n[Threshold Optimization] Evaluating {len(val_dataset)} validation scenes across {len(candidate_thresholds)} candidates...")

    scene_prob_gt: List[Tuple[np.ndarray, np.ndarray]] = []
    t0 = time.perf_counter()

    for idx in range(len(val_dataset)):
        t1_t, t2_t, lbl_t, meta = val_dataset[idx]
        img1 = (t1_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img2 = (t2_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt = lbl_t.squeeze(0).numpy().astype(np.uint8)

        stem = meta["sample_id"] if isinstance(meta, dict) else getattr(meta, "sample_id", f"scene_{idx}")
        patches = patch_extractor.extract_from_pair(img1, img2, gt, parent_stem=stem)
        diff_patches = []
        for p1, p2, _, p_meta in patches:
            p_prob = detector.compute_difference_map(p1, p2)
            diff_patches.append((p_prob, p_meta))

        full_prob = patch_extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
        scene_prob_gt.append((full_prob, gt))

    t_eval = time.perf_counter() - t0
    print(f"[Threshold Optimization] Computed {len(val_dataset)} validation probability maps in {t_eval:.2f}s")

    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics: Dict[str, Any] = {}
    curve_records = []

    for tau in candidate_thresholds:
        cm = ConfusionMatrix()
        for full_prob, gt in scene_prob_gt:
            pred_bin = (full_prob >= tau).astype(np.uint8)
            cm.update(pred_bin, gt)

        m = ChangeDetectionEvaluator.evaluate(cm)
        curve_records.append({
            "threshold": tau,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "iou": m["iou"],
        })

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = tau
            best_metrics = m

    print(f"[Threshold Optimization] Selected optimal threshold: tau* = {best_thresh:.4f} (Val F1: {best_f1:.4f}, IoU: {best_metrics['iou']:.4f})")
    return best_thresh, {
        "best_threshold": best_thresh,
        "best_metrics": best_metrics,
        "curve": curve_records,
    }


def evaluate_test_set(
    model: SNUNet,
    test_dataset: LEVIRDataset,
    patch_extractor: PatchExtractor,
    frozen_threshold: float,
    device: torch.device,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates official held-out test split (128 scenes) using strictly frozen validation threshold."""
    detector = SNUNetDetector(model=model, device=str(device))
    print(f"\n[Test Evaluation] Evaluating {len(test_dataset)} test scenes with frozen threshold tau* = {frozen_threshold:.4f}...")

    cm_total = ConfusionMatrix()
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
        patches = patch_extractor.extract_from_pair(img1, img2, gt, parent_stem=stem)
        diff_patches = []
        for p1, p2, _, p_meta in patches:
            p_prob = detector.compute_difference_map(p1, p2)
            diff_patches.append((p_prob, p_meta))

        full_prob = patch_extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
        pred_bin = (full_prob >= frozen_threshold).astype(np.uint8)

        pair_time_ms = (time.perf_counter() - t0_pair) * 1000.0
        latencies_ms.append(pair_time_ms)

        # Track scene-level confusion & metrics
        cm_pair = ConfusionMatrix()
        cm_pair.update(pred_bin, gt)
        p_m = ChangeDetectionEvaluator.evaluate(cm_pair)

        per_image_rows.append({
            "sample_id": stem,
            "threshold": frozen_threshold,
            "precision": p_m["precision"],
            "recall": p_m["recall"],
            "f1": p_m["f1"],
            "iou": p_m["iou"],
            "accuracy": p_m["accuracy"],
            "tp": p_m["tp"],
            "fp": p_m["fp"],
            "fn": p_m["fn"],
            "tn": p_m["tn"],
            "latency_ms": pair_time_ms,
        })

        cm_total.update(pred_bin, gt)

        # Track RSS
        curr_rss = process.memory_info().rss / (1024 * 1024)
        if curr_rss > peak_rss_mb:
            peak_rss_mb = curr_rss

    total_time_s = time.perf_counter() - t_start
    overall_metrics = ChangeDetectionEvaluator.evaluate(cm_total)

    overall_metrics.update({
        "method": "M8_SNUNet",
        "threshold_source": "Validation F1 (Frozen)",
        "threshold_value": frozen_threshold,
        "mean_pair_latency_ms": float(np.mean(latencies_ms)),
        "p50_pair_latency_ms": float(np.median(latencies_ms)),
        "p95_pair_latency_ms": float(np.percentile(latencies_ms, 95)),
        "total_runtime_s": total_time_s,
        "peak_rss_mb": peak_rss_mb,
    })

    per_image_df = pd.DataFrame(per_image_rows)
    print(f"[Test Evaluation Complete] Overall F1: {overall_metrics['f1']:.4f}, IoU: {overall_metrics['iou']:.4f}, Precision: {overall_metrics['precision']:.4f}, Recall: {overall_metrics['recall']:.4f}")

    return overall_metrics, per_image_df


def run_m8_pipeline(
    config_path: Optional[Path | str] = None,
    epochs_override: Optional[int] = None,
    samples_per_epoch_override: Optional[int] = None,
    skip_training: bool = False,
) -> Dict[str, Any]:
    """Executes end-to-end M8 training, threshold calibration, and test benchmark."""
    root = get_project_root()
    config = load_m8_config() if config_path is None else load_m8_config()

    seed = config["experiment"].get("seed", 42)
    device_str = config["experiment"].get("device", "cpu")
    device = torch.device(device_str)
    num_threads = config["experiment"].get("num_threads", 4)
    torch.set_num_threads(num_threads)
    set_seed(seed)

    raw_dir = root / config["dataset"]["raw_dir"]
    ckpt_dir = root / config["paths"]["checkpoint_dir"]
    results_dir = root / config["paths"]["results_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = epochs_override or config["training"].get("epochs", 5)
    samples_per_epoch = samples_per_epoch_override or config["training"].get("samples_per_epoch", 512)
    batch_size = config["training"].get("batch_size", 16)
    lr = config["training"].get("learning_rate", 0.001)
    base_ch = config["model"].get("base_channels", 16)

    print("=" * 80)
    print("MILESTONE M8: SNUNet-CD TRAINING & ADVANCED BENCHMARK")
    print(f"Device: {device_str} | Threads: {num_threads} | Seed: {seed}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Samples/Epoch: {samples_per_epoch} | LR: {lr}")
    print(f"Raw Dataset Dir: {raw_dir}")
    print("=" * 80)

    # 1. Initialize Datasets
    print("\n[Step 1/5] Initializing LEVIR-CD train and validation patch datasets...")
    train_patch_ds = LEVIRPatchDataset(root_dir=raw_dir, split="train", patch_size=256)
    val_patch_ds = LEVIRPatchDataset(root_dir=raw_dir, split="val", patch_size=256)
    val_raw_ds = LEVIRDataset(root_dir=raw_dir, split="val")
    test_raw_ds = LEVIRDataset(root_dir=raw_dir, split="test")

    print(f"Train scenes: {len(train_patch_ds.levir_ds.pairs)} ({len(train_patch_ds):,} total 256x256 patches)")
    print(f"Val scenes: {len(val_raw_ds)} ({len(val_patch_ds):,} total 256x256 patches)")
    print(f"Test scenes: {len(test_raw_ds)} ({len(test_raw_ds)*16:,} total 256x256 patches)")

    # Balanced Patch Mining on Train Split
    print("Scanning train patches for change-class occurrence...")
    pos_indices = []
    neg_indices = []
    for idx in range(len(train_patch_ds)):
        _, _, lbl = train_patch_ds[idx]
        if lbl.sum() > 0:
            pos_indices.append(idx)
        else:
            neg_indices.append(idx)

    print(f"Discovered: {len(pos_indices)} positive change patches ({len(pos_indices)/len(train_patch_ds)*100:.1f}%), {len(neg_indices)} negative background patches")

    balanced_sampler = BalancedPatchSampler(
        positive_indices=pos_indices,
        negative_indices=neg_indices,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )

    train_loader = DataLoader(
        train_patch_ds,
        batch_size=batch_size,
        sampler=balanced_sampler,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_patch_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 2. Initialize Model, Loss, Optimizer
    print("\n[Step 2/5] Initializing SNUNet-CD architecture and BCEDiceLoss...")
    model = SNUNet(in_channels=3, out_channels=1, base_channels=base_ch).to(device)
    num_params = model.count_trainable_parameters()
    print(f"Model Parameters: {num_params:,} (<=2,000,000 ceiling: {num_params <= 2_000_000})")

    loss_fn = BCEDiceLoss(
        bce_weight=config["training"]["loss"].get("bce_weight", 1.0),
        dice_weight=config["training"]["loss"].get("dice_weight", 1.0),
        smooth=config["training"]["loss"].get("smooth", 1.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 3. Model Training / Checkpoint Loading
    print("\n[Step 3/5] Starting model training...")
    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = ckpt_dir / "snunet_best.pt"
    history = []

    if skip_training and best_ckpt_path.exists():
        print(f"\n[Step 3/5] Loading existing trained checkpoint from {best_ckpt_path}...")
        best_model, meta = load_checkpoint(best_ckpt_path, device=device_str)
        best_epoch = meta.get("best_epoch", epochs)
        best_val_f1 = meta.get("best_val_f1", 0.0)
        history_path = results_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        print(f"[Training Restored] Best Checkpoint loaded: Epoch {best_epoch} with Val F1 = {best_val_f1:.4f}")
    else:
        for epoch in range(1, epochs + 1):
            t0_ep = time.perf_counter()
            balanced_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            scheduler.step()

            val_loss, val_f1 = evaluate_val_loss_and_f1(model, val_loader, loss_fn, device)
            ep_duration = time.perf_counter() - t0_ep

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:4.1f}s) | "
                f"Train Loss: {train_loss:6.4f} | Val Loss: {val_loss:6.4f} | Val F1: {val_f1:6.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "lr": scheduler.get_last_lr()[0],
                "epoch_duration_s": ep_duration,
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                save_checkpoint(
                    model=model,
                    path=best_ckpt_path,
                    metadata={
                        "best_epoch": epoch,
                        "epoch": epoch,
                        "best_val_f1": val_f1,
                        "seed": seed,
                        "base_channels": base_ch,
                        "epochs": epochs,
                        "samples_per_epoch": samples_per_epoch,
                    },
                )
                print(f"  --> Saved new best checkpoint at epoch {epoch} (Val F1: {val_f1:.4f})")

        print(f"\n[Training Complete] Best Checkpoint selected at Epoch {best_epoch} with Val F1 = {best_val_f1:.4f}")

    # Reload best model for evaluation
    best_model, meta = load_checkpoint(best_ckpt_path, device=device_str)
    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")

    # 4. Validation Threshold Optimization
    print("\n[Step 4/5] Optimizing validation decision threshold...")
    num_candidates = config["thresholding"].get("num_candidates", 19)
    min_t = config["thresholding"].get("min_threshold", 0.05)
    max_t = config["thresholding"].get("max_threshold", 0.95)
    candidate_taus = [round(t, 2) for t in np.linspace(min_t, max_t, num_candidates).tolist()]

    frozen_tau, thresh_data = optimize_validation_threshold(
        model=best_model,
        val_dataset=val_raw_ds,
        patch_extractor=extractor,
        device=device,
        candidate_thresholds=candidate_taus,
    )

    # Save threshold metadata
    with open(results_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresh_data, f, indent=2)

    # Update checkpoint with chosen frozen threshold
    meta["frozen_threshold"] = frozen_tau
    save_checkpoint(model=best_model, path=best_ckpt_path, metadata=meta)

    # 5. Held-Out Test Split Benchmark
    print("\n[Step 5/5] Executing official test split benchmark...")
    test_metrics, per_image_df = evaluate_test_set(
        model=best_model,
        test_dataset=test_raw_ds,
        patch_extractor=extractor,
        frozen_threshold=frozen_tau,
        device=device,
    )

    # Save M8 Results
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump([test_metrics], f, indent=2)

    summary_df = pd.DataFrame([test_metrics])
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    per_image_df.to_csv(results_dir / "per_image_metrics.csv", index=False)

    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save Checkpoint Metadata
    ckpt_meta = {
        "model_architecture": "SNUNet-CD",
        "parameters": num_params,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "frozen_threshold": frozen_tau,
        "test_f1": test_metrics["f1"],
        "test_iou": test_metrics["iou"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "seed": seed,
        "device": device_str,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(ckpt_meta, f, indent=2)

    # 6. Generate Comprehensive Comparison (B1, B2, B3, M7, M8)
    comparison_records = []

    # Historical M5 baselines
    m5_summary_path = root / "experiments" / "results" / "m5" / "summary.json"
    if m5_summary_path.exists():
        with open(m5_summary_path, "r", encoding="utf-8") as f:
            m5_records = json.load(f)
        for r in m5_records:
            if "Frozen" in r.get("threshold_source", ""):
                comparison_records.append(r)

    # M7 FC-Siam-diff
    m7_summary_path = root / "experiments" / "results" / "m7" / "summary.json"
    if m7_summary_path.exists():
        with open(m7_summary_path, "r", encoding="utf-8") as f:
            m7_records = json.load(f)
        for r in m7_records:
            comparison_records.append(r)

    # Add M8
    comparison_records.append(test_metrics)

    with open(results_dir / "comparison_m5_m7_m8.json", "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv(results_dir / "comparison_m5_m7_m8.csv", index=False)

    print("\n[Comparison with All Baselines]")
    for r in comparison_records:
        tau_val = r.get("threshold_value", 0.0)
        print(
            f"  {r['method']:18s} | Threshold: {tau_val:.4f} | "
            f"F1: {r['f1']:.4f} | IoU: {r['iou']:.4f} | "
            f"Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f}"
        )

    print("\n" + "=" * 80)
    print("M8 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Checkpoint saved: {best_ckpt_path}")
    print(f"Results saved: {results_dir / 'summary.json'}")
    print("=" * 80)

    return {
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "frozen_threshold": frozen_tau,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Milestone M8 SNUNet Training & Evaluation Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to M8 configuration YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--samples-per-epoch", type=int, default=None, help="Override samples per epoch")
    parser.add_argument("--eval-only", action="store_true", help="Run validation threshold search and test evaluation using existing trained checkpoint")
    args = parser.parse_args()

    run_m8_pipeline(
        config_path=args.config,
        epochs_override=args.epochs,
        samples_per_epoch_override=args.samples_per_epoch,
        skip_training=args.eval_only,
    )


if __name__ == "__main__":
    main()
