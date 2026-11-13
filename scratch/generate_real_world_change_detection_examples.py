import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

from src.change_detection.bit_cd import BITChangeDetector, load_checkpoint
from ui.image_viewer import create_error_overlay

figures_dir = "/Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/experiments/figures/m11"
artifact_dir = "/Users/rahulmanchanda/.gemini/antigravity-ide/brain/b2512c2b-3bd9-4485-b200-368394226612"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# Load M9 detector
ckpt_path = "experiments/checkpoints/m9/bit_cd_best.pt"
model, meta = load_checkpoint(ckpt_path, device="cpu")
detector = BITChangeDetector(model=model, device="cpu", name="M9_BIT_CD")
tau = float(meta.get("frozen_threshold", 0.45))

# 3 Representative Real-World Test Scenes
scenes = [
    {
        "id": "test_102.png",
        "label": "Case 1: test_102\nSuburban Rooftops\nF1: 0.8893 • IoU: 80.1%",
    },
    {
        "id": "test_10.png",
        "label": "Case 2: test_10\nDense Subdivision\nF1: 0.8300 • IoU: 70.9%",
    },
    {
        "id": "test_100.png",
        "label": "Case 3: test_100\nPlanned Community\nF1: 0.7743 • IoU: 63.2%",
    },
]

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
})

fig = plt.figure(figsize=(20, 13.0), facecolor="white")
gs = GridSpec(3, 5, hspace=0.24, wspace=0.08, left=0.09, right=0.97, top=0.86, bottom=0.04)

for row_idx, sc in enumerate(scenes):
    name = sc["id"]
    t1 = cv2.cvtColor(cv2.imread(f"data/demo_samples/A/{name}"), cv2.COLOR_BGR2RGB)
    t2 = cv2.cvtColor(cv2.imread(f"data/demo_samples/B/{name}"), cv2.COLOR_BGR2RGB)
    gt = (cv2.imread(f"data/demo_samples/label/{name}", cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

    probs = detector.compute_difference_map(t1, t2)
    pred = (probs >= tau).astype(np.uint8)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    err_overlay = create_error_overlay(pred, gt)

    # Col 1: T1
    ax1 = fig.add_subplot(gs[row_idx, 0])
    ax1.imshow(t1)
    ax1.set_ylabel(sc["label"], fontsize=9.5, fontweight="bold", color="#0F172A", labelpad=12)
    if row_idx == 0:
        ax1.set_title("1. Pre-Change T1 Optical", pad=8, color="#1E293B")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Col 2: T2
    ax2 = fig.add_subplot(gs[row_idx, 1])
    ax2.imshow(t2)
    if row_idx == 0:
        ax2.set_title("2. Post-Change T2 Optical", pad=8, color="#1E293B")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Col 3: Ground Truth Mask
    ax3 = fig.add_subplot(gs[row_idx, 2])
    ax3.imshow(gt, cmap="gray")
    gt_pct = np.mean(gt) * 100
    ax3.text(512, 980, f"Ground Truth ({gt_pct:.2f}% Changed • {np.sum(gt):,} px)",
             ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#0F172A", alpha=0.85))
    if row_idx == 0:
        ax3.set_title("3. Ground Truth Change Mask", pad=8, color="#1E293B")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Col 4: M9 BIT-CD Predicted Probability Map
    ax4 = fig.add_subplot(gs[row_idx, 3])
    im_prob = ax4.imshow(probs, cmap="magma", vmin=0.0, vmax=1.0)
    pred_pct = np.mean(pred) * 100
    ax4.text(512, 980, f"M9 Output (τ*={tau:.2f} • {pred_pct:.2f}% Changed)",
             ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#0F172A", alpha=0.85))
    if row_idx == 0:
        ax4.set_title("4. M9 BIT-CD Probability Map", pad=8, color="#1E293B")
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Col 5: Error Classification Overlay
    ax5 = fig.add_subplot(gs[row_idx, 4])
    ax5.imshow(err_overlay)
    ax5.text(512, 980, f"Prec: {prec*100:.1f}% | Rec: {rec*100:.1f}%\nF1: {f1:.4f} | IoU: {iou*100:.1f}%",
             ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#0F172A", alpha=0.90))
    if row_idx == 0:
        ax5.set_title("5. Error Overlay (Confusion)", pad=8, color="#1E293B")
    ax5.set_xticks([])
    ax5.set_yticks([])

# Legend at top right of figure
legend_elements = [
    patches.Patch(facecolor="#22C55E", edgecolor="none", label="True Positive (TP) — Correctly Detected Change"),
    patches.Patch(facecolor="#EF4444", edgecolor="none", label="False Positive (FP) — Spurious False Alarm"),
    patches.Patch(facecolor="#3B82F6", edgecolor="none", label="False Negative (FN) — Missed Change"),
    patches.Patch(facecolor="#0F172A", edgecolor="none", label="True Negative (TN) — Invariant Terrain"),
]
fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.97, 0.905), ncol=4, framealpha=0.95, fontsize=8.8)

# Figure Master Title
fig.suptitle(
    "Real-World Temporal Change Detection Benchmarks on LEVIR-CD (1024×1024 Optical Scenes)\n"
    "Bitemporal Transformer (M9 BIT-CD) Evaluation Across Suburban Construction & Master-Planned Community Development",
    fontsize=13.5, fontweight="bold", color="#0F172A", y=0.965, x=0.04, ha="left"
)

dest_1 = os.path.join(figures_dir, "real_world_change_detection_examples.png")
dest_2 = os.path.join(artifact_dir, "real_world_change_detection_examples.png")
plt.savefig(dest_1, dpi=300, bbox_inches="tight")
plt.savefig(dest_2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved real world change detection figures to:\n  {dest_1}\n  {dest_2}")
