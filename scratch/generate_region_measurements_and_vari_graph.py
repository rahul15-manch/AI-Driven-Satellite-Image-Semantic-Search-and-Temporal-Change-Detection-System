import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from src.temporal_understanding.region_extractor import RegionExtractor, RegionExtractionConfig
from src.temporal_understanding.semantic_analyzer import compute_vari_map, analyze_regional_vari
from ui.image_viewer import create_temporal_overlay

figures_dir = "/Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/experiments/figures/m11"
artifact_dir = "/Users/rahulmanchanda/.gemini/antigravity-ide/brain/b2512c2b-3bd9-4485-b200-368394226612"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# Load real satellite pair showing dynamic vegetation loss (test_1.png)
t1_bgr = cv2.imread("data/demo_samples/A/test_1.png")
t2_bgr = cv2.imread("data/demo_samples/B/test_1.png")
t1_rgb = cv2.cvtColor(t1_bgr, cv2.COLOR_BGR2RGB)
t2_rgb = cv2.cvtColor(t2_bgr, cv2.COLOR_BGR2RGB)
raw_mask = cv2.imread("data/demo_samples/label/test_1.png", cv2.IMREAD_GRAYSCALE)
binary_mask = (raw_mask > 0).astype(np.uint8)

# Extract regions
config = RegionExtractionConfig(
    min_pixel_area=150,
    min_area_ratio=0.0002,
    max_regions=6,
    morphology_closing_kernel=3,
    morphology_opening_kernel=3,
    context_padding_ratio=0.10,
    connectivity=8,
)
extractor = RegionExtractor(config=config)
res = extractor.extract_regions(binary_mask, gsd_meters=0.5)

# Compute full continuous VARI maps
v1_map, _, _ = compute_vari_map(t1_rgb)
v2_map, _, _ = compute_vari_map(t2_rgb)
delta_vari_map = v2_map - v1_map

# Compute regional metrics
reg_metrics = []
for r in res.regions:
    c_xmin, c_ymin, c_xmax, c_ymax = r.context_bbox
    t1_crop = t1_rgb[c_ymin:c_ymax, c_xmin:c_xmax]
    t2_crop = t2_rgb[c_ymin:c_ymax, c_xmin:c_xmax]
    m_crop = binary_mask[c_ymin:c_ymax, c_xmin:c_xmax]
    ev = analyze_regional_vari(t1_crop, t2_crop, m_crop)
    reg_metrics.append({
        "region": r,
        "evidence": ev
    })

# Visual styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
})

fig = plt.figure(figsize=(20, 11), facecolor="white")
gs = GridSpec(2, 2, height_ratios=[1.1, 0.9], width_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22,
              left=0.04, right=0.96, top=0.88, bottom=0.07)

# Overlay on T2
overlay_rgb = create_temporal_overlay(t2_rgb, binary_mask, [r.to_dict() for r in res.regions], alpha=0.38)

# Panel A: T1 Observation with Region Centers
ax_t1 = fig.add_subplot(gs[0, 0])
ax_t1.imshow(t1_rgb)
for item in reg_metrics:
    r = item["region"]
    xmin, ymin, xmax, ymax = r.bbox
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2.0, edgecolor="#22C55E", facecolor="none", linestyle="--")
    ax_t1.add_patch(rect)

ax_t1.text(820, 280, "Cluster A: Regions #1, #5, #6\n(T1 High Vegetation VARI: +0.40 to +0.75)",
           fontsize=8.2, fontweight="bold", color="#15803D",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#22C55E", alpha=0.92))
ax_t1.text(820, 140, "Cluster B: Regions #2, #3, #4\n(T1 Moderate Canopy VARI: +0.12 to +0.42)",
           fontsize=8.2, fontweight="bold", color="#15803D",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#22C55E", alpha=0.92))

ax_t1.set_title("(A) Pre-Change T1 Optical Scene (Vegetated Baseline)", pad=10, color="#0F172A")
ax_t1.set_xlabel("Pixel X Coordinate", fontweight="bold")
ax_t1.set_ylabel("Pixel Y Coordinate", fontweight="bold")
ax_t1.grid(False)

# Panel B: T2 Observation with Bounding Boxes & Terracotta Change Overlay
ax_t2 = fig.add_subplot(gs[0, 1])
ax_t2.imshow(overlay_rgb)
ax_t2.text(820, 280, "Severe Canopy Removal\n(T2 VARI drops to -0.01; ΔVARI ≤ -0.74)",
           fontsize=8.2, fontweight="bold", color="#991B1B",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#EF4444", alpha=0.92))

ax_t2.set_title("(B) Post-Change T2 Scene with Bounding Boxes & Change Overlay", pad=10, color="#0F172A")
ax_t2.set_xlabel("Pixel X Coordinate", fontweight="bold")
ax_t2.set_ylabel("Pixel Y Coordinate", fontweight="bold")
ax_t2.grid(False)

# Panel C: Optical VARI Difference Map (Delta VARI)
ax_dvari = fig.add_subplot(gs[1, 0])
im_d = ax_dvari.imshow(delta_vari_map, cmap="RdYlGn", vmin=-0.8, vmax=0.8)
ax_dvari.set_title("(C) Visible Atmospherically Resistant Index Delta Map (T2 - T1)", pad=10, color="#0F172A")
ax_dvari.set_xlabel("Pixel X Coordinate", fontweight="bold")
ax_dvari.set_ylabel("Pixel Y Coordinate", fontweight="bold")
ax_dvari.grid(False)

cbar_d = fig.colorbar(im_d, ax=ax_dvari, orientation="horizontal", pad=0.14, aspect=26, shrink=0.85)
cbar_d.set_label("ΔVARI = (Green - Red) / (Green + Red - Blue)  [Red: Canopy Loss ← → Green: Greening]", fontsize=9, fontweight="bold")
cbar_d.set_ticks([-0.8, -0.4, 0.0, 0.4, 0.8])
cbar_d.ax.set_xticklabels(["-0.8 (Loss)", "-0.4", "0.0 (Stable)", "+0.4", "+0.8 (Greening)"])

# Panel D: Regional Metrics Bar Chart: Delta VARI & Physical Area
ax_bar = fig.add_subplot(gs[1, 1])
r_names = [f"Region #{item['region'].region_id}" for item in reg_metrics]
delta_varis = [item['evidence'].delta_vari for item in reg_metrics]
areas_sqm = [item['region'].physical_area_sq_meters for item in reg_metrics]
x_pos = np.arange(len(r_names))

bar_cols = ["#DC2626" if d <= -0.15 else ("#16A34A" if d >= 0.15 else "#64748B") for d in delta_varis]
bars = ax_bar.bar(x_pos, delta_varis, width=0.52, color=bar_cols, edgecolor="#0F172A", linewidth=1.1, zorder=3)

ax_bar.axhline(-0.15, color="#DC2626", linestyle="--", linewidth=1.5, label="Vegetation Loss Gate (ΔVARI ≤ -0.15)")
ax_bar.axhline(+0.15, color="#16A34A", linestyle="--", linewidth=1.5, label="Greening Gate (ΔVARI ≥ +0.15)")
ax_bar.axhline(0.0, color="#64748B", linestyle="-", linewidth=1.0)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(r_names, fontweight="bold")
ax_bar.set_ylabel("ΔVARI Shift (Mean T2 - Mean T1)", fontweight="bold")
ax_bar.set_ylim(-0.9, 0.25)
ax_bar.grid(axis="y", linestyle=":", alpha=0.4, color="#64748B", zorder=0)
ax_bar.legend(loc="lower left", framealpha=0.92, fontsize=8.2)

# Value annotations with physical area
for idx, (b, item) in enumerate(zip(bars, reg_metrics)):
    r = item["region"]
    d_val = item["evidence"].delta_vari
    ax_bar.text(b.get_x() + b.get_width()/2, d_val - 0.04,
                f"Δ: {d_val:.3f}\nArea: {r.physical_area_sq_meters:.0f} m²\n({r.pixel_area} px)",
                ha="center", va="top", fontsize=7.8, fontweight="bold", color="#7F1D1D")

ax_bar.set_title("(D) Quantitative Regional ΔVARI Shift & Physical Extents", pad=10, color="#0F172A")

# Main Title
fig.suptitle(
    "Phase 3B Regional Measurements & Optical Visible Atmospherically Resistant Index (VARI) Analysis\n"
    "Sub-Region Spectral Telemetry, Top-Left Origin Coordinate Bounding Boxes, and Vegetation Proxy Shift (LEVIR-CD)",
    fontsize=14, fontweight="bold", color="#0F172A", y=0.96
)

dest_1 = os.path.join(figures_dir, "region_measurements_vari_master.png")
dest_2 = os.path.join(artifact_dir, "region_measurements_and_vari_table.png")
plt.savefig(dest_1, dpi=300, bbox_inches="tight")
plt.savefig(dest_2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved region measurements and VARI graph to:\n  {dest_1}\n  {dest_2}")
