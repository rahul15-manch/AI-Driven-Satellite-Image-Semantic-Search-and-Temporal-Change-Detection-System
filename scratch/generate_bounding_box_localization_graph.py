import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image

from src.temporal_understanding.region_extractor import RegionExtractor, RegionExtractionConfig
from ui.image_viewer import create_temporal_overlay

# Output paths
figures_dir = "/Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/experiments/figures/m11"
artifact_dir = "/Users/rahulmanchanda/.gemini/antigravity-ide/brain/b2512c2b-3bd9-4485-b200-368394226612"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# 1. Load Real Demo Satellite Pair and Mask
t2_bgr = cv2.imread("data/demo_samples/B/test_100.png")
t2_rgb = cv2.cvtColor(t2_bgr, cv2.COLOR_BGR2RGB)
raw_mask = cv2.imread("data/demo_samples/label/test_100.png", cv2.IMREAD_GRAYSCALE)
binary_mask = (raw_mask > 0).astype(np.uint8)

# 2. Run RegionExtractor
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
result = extractor.extract_regions(binary_mask, gsd_meters=0.5)

# 3. Create Full Temporal Overlay
overlay_rgb = create_temporal_overlay(t2_rgb, binary_mask, [r.to_dict() for r in result.regions], alpha=0.35)

# Styling Configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.titlesize": 14.5,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
})

fig = plt.figure(figsize=(20, 11.0), facecolor="white")
gs = GridSpec(2, 2, height_ratios=[1.1, 0.9], width_ratios=[1.0, 1.05], hspace=0.36, wspace=0.22,
              left=0.04, right=0.96, top=0.86, bottom=0.07)

# =============================================================================
# PANEL A: Full Satellite Scene with Spatial Bounding Box Overlay
# =============================================================================
ax_overlay = fig.add_subplot(gs[0, 0])
ax_overlay.imshow(overlay_rgb)
ax_overlay.set_title("(A) Spatial Bounding Box Localization on Satellite T2 (1024×1024)", pad=10, color="#0F172A")
ax_overlay.set_xlabel("Pixel X Coordinate (Origin: Top-Left)", fontweight="bold")
ax_overlay.set_ylabel("Pixel Y Coordinate (Origin: Top-Left)", fontweight="bold")
ax_overlay.grid(False)

# Add legend / key for overlay
legend_elements = [
    patches.Patch(facecolor="#C86B4D", edgecolor="#9C442A", label="Terracotta Change Mask (τ*=0.4500)"),
    patches.Patch(facecolor="none", edgecolor="#E1A95F", linewidth=2.0, label="Yellow Ochre Bounding Box [xmin, ymin, xmax, ymax]"),
    patches.Patch(facecolor="#4A2E2B", edgecolor="none", label="Rank Badge Pill (#1 to #6)"),
]
ax_overlay.legend(handles=legend_elements, loc="upper left", framealpha=0.92, fontsize=8.5)

# =============================================================================
# PANEL B: Zoomed Crop of Region #1 Illustrating Strict BBox vs Contextual Margin
# =============================================================================
ax_zoom = fig.add_subplot(gs[0, 1])

# Extract Region 1 crop
r1 = result.regions[0]
xmin, ymin, xmax, ymax = r1.bbox
c_xmin, c_ymin, c_xmax, c_ymax = r1.context_bbox

# Expand view slightly beyond context bbox for clear illustration
margin = 40
v_xmin = max(0, c_xmin - margin)
v_ymin = max(0, c_ymin - margin)
v_xmax = min(1024, c_xmax + margin)
v_ymax = min(1024, c_ymax + margin)

crop_t2 = t2_rgb[v_ymin:v_ymax, v_xmin:v_xmax].copy()
crop_mask = binary_mask[v_ymin:v_ymax, v_xmin:v_xmax]

# Apply Terracotta tint over changed pixels in zoom crop
crop_vis = crop_t2.copy()
color_tint = np.array([200, 107, 77], dtype=np.uint8)
m_crop = crop_mask > 0
crop_vis[m_crop] = cv2.addWeighted(crop_vis[m_crop], 0.65, np.tile(color_tint, (np.sum(m_crop), 1)), 0.35, 0)

ax_zoom.imshow(crop_vis, extent=[v_xmin, v_xmax, v_ymax, v_ymin])

# Draw Strict Bounding Box (Yellow Ochre)
rect_strict = patches.Rectangle(
    (xmin, ymin), xmax - xmin, ymax - ymin,
    linewidth=2.5, edgecolor="#E1A95F", facecolor="none", linestyle="-", label="Strict Bounding Box (87×59 px)"
)
ax_zoom.add_patch(rect_strict)

# Draw Contextual Padded Bounding Box (+10% margin, Cyan/Teal dashed)
rect_context = patches.Rectangle(
    (c_xmin, c_ymin), c_xmax - c_xmin, c_ymax - c_ymin,
    linewidth=2.0, edgecolor="#06B6D4", facecolor="none", linestyle="--", label="Contextual Bounding Box (+10% Margin)"
)
ax_zoom.add_patch(rect_context)

# Centroid Marker
ax_zoom.scatter([r1.centroid[0]], [r1.centroid[1]], color="#EF4444", s=90, marker="x", linewidths=2.5,
                zorder=5, label=f"Centroid ({r1.centroid[0]:.1f}, {r1.centroid[1]:.1f})")

# Dimension Annotations
ax_zoom.text(xmin + (xmax - xmin)/2, ymin - 8, f"W = {r1.bbox_width} px",
             ha="center", va="bottom", fontsize=9, fontweight="bold", color="#B45309",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#E1A95F", alpha=0.95))
ax_zoom.text(xmax + 8, ymin + (ymax - ymin)/2, f"H = {r1.bbox_height} px",
             ha="left", va="center", fontsize=9, fontweight="bold", color="#B45309",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#E1A95F", alpha=0.95))

# Top callout
ax_zoom.text((v_xmin + v_xmax)/2, v_ymin + 15,
             f"Region #1: {r1.pixel_area} pixels • {r1.physical_area_sq_meters:.1f} m² (GSD = 0.5m)",
             ha="center", va="top", fontsize=9.5, fontweight="bold", color="#0F172A",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7", edgecolor="#D97706", alpha=0.95))

ax_zoom.set_title("(B) Sub-Region Geometric Geometry & Contextual Padding (+10%)", pad=10, color="#0F172A")
ax_zoom.set_xlabel("Pixel X Coordinate", fontweight="bold")
ax_zoom.set_ylabel("Pixel Y Coordinate", fontweight="bold")
ax_zoom.legend(loc="lower left", framealpha=0.92, fontsize=8.8)
ax_zoom.grid(True, linestyle=":", alpha=0.35, color="#64748B")

# =============================================================================
# PANEL C: Ranked Change Region Area Spectrum (Pixels & Physical m²)
# =============================================================================
ax_bar = fig.add_subplot(gs[1, 0])

region_labels = [f"Region #{r.region_id}" for r in result.regions]
pixel_areas = [r.pixel_area for r in result.regions]
sqm_areas = [r.physical_area_sq_meters for r in result.regions]
x_indices = np.arange(len(region_labels))

bars = ax_bar.bar(x_indices, pixel_areas, width=0.55, color="#C86B4D", edgecolor="#8B3A2B", linewidth=1.2, zorder=3)
ax_bar.set_ylabel("Pixel Area (pixels)", fontweight="bold", color="#8B3A2B")
ax_bar.set_xticks(x_indices)
ax_bar.set_xticklabels(region_labels, fontweight="bold")
ax_bar.set_ylim(0, 3100)
ax_bar.grid(axis="y", linestyle="--", alpha=0.4, color="#64748B", zorder=0)

# Secondary axis for physical area in m²
ax_sqm = ax_bar.twinx()
ax_sqm.set_ylim(0, 3100 * (0.5**2))  # GSD = 0.5m -> 1 px = 0.25 m2
ax_sqm.set_ylabel("Physical Area (m² at 0.5m GSD)", fontweight="bold", color="#1E40AF")
ax_sqm.tick_params(axis="y", labelcolor="#1E40AF")

# Value Labels on bars
for idx, (bar, r) in enumerate(zip(bars, result.regions)):
    h_val = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h_val + 70,
                f"{r.pixel_area} px\n({r.physical_area_sq_meters:.1f} m²)",
                ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#4A2E2B")
    # Subtitle with dimensions
    ax_bar.text(bar.get_x() + bar.get_width()/2, h_val / 2,
                f"{r.bbox_width}×{r.bbox_height} px",
                ha="center", va="center", fontsize=8.0, fontweight="bold", color="white")

ax_bar.set_title("(C) Ranked Top-6 Change Clusters: Geometry & Area Spectrum", pad=10, color="#0F172A")

# =============================================================================
# PANEL D: Connected-Component Region Extraction Architecture & Data Flow
# =============================================================================
ax_flow = fig.add_subplot(gs[1, 1])
ax_flow.set_facecolor("#FCFDFD")
ax_flow.set_xlim(0, 100)
ax_flow.set_ylim(0, 100)
ax_flow.axis("off")
ax_flow.set_title("(D) Phase 3A Region Extraction & Filtering Data Pipeline", pad=10, color="#0F172A")

def draw_block(ax, x, y, w, h, bg, border, title, desc, title_col="#0F172A"):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.8",
                                facecolor=bg, edgecolor=border, linewidth=1.4, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.64, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=title_col, zorder=3)
    ax.text(x + w/2, y + h*0.28, desc, ha="center", va="center", fontsize=7.4, color="#475569", zorder=3)

def draw_conn(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.5, mutation_scale=10), zorder=1)
    if text:
        ax.text((x1+x2)/2 + 2, (y1+y2)/2, text, fontsize=7.2, color="#64748B", fontweight="bold", va="center")

# Flowchart Nodes
draw_block(ax_flow, 4, 80, 92, 16, "#F1F5F9", "#CBD5E1",
           "Input: Dense Binary Change Mask {0, 1}^(H×W)",
           "From BIT-CD (M9) / QAT-CD (M10) at optimal threshold τ* = 0.4500 (1024×1024 px)")
draw_conn(ax_flow, 50, 80, 50, 72)

draw_block(ax_flow, 4, 56, 92, 16, "#EFF6FF", "#3B82F6",
           "Step 1: Morphological Cleanup (Dual Kernel 3×3)",
           "Closing (3×3) bridges fractured boundaries • Opening (3×3) removes single-pixel noise", title_col="#1D4ED8")
draw_conn(ax_flow, 50, 56, 50, 48)

draw_block(ax_flow, 4, 32, 92, 16, "#FEF3C7", "#D97706",
           "Step 2: 8-Connectivity Connected-Component Labeling (CCL) & Gating",
           "cv2.connectedComponentsWithStats • Dual Gate: Area ≥ 150 px AND Area Ratio ≥ 0.02%", title_col="#B45309")
draw_conn(ax_flow, 50, 32, 50, 24)

draw_block(ax_flow, 4, 0, 92, 24, "#F0FDF4", "#22C55E",
           "Step 3: Descending Area Ranking & Contextual Bounding-Box Extraction",
           "Rank Top-K (K=6) • Strict BBox [xmin, ymin, xmax, ymax] • Centroid (cx, cy)\n"
           "Context BBox: +10% margin [c_xmin, c_ymin, c_xmax, c_ymax] for Stage A VLM Crops\n"
           "Physical Area: Area × GSD² (m²) • JSON Entity Serialization via Pydantic Schema", title_col="#15803D")

# Master Title
fig.suptitle(
    "Phase 3A Bounding-Box Region Localization & Spatial Clustering Engine\n"
    "Morphological Connected-Component Extraction, 10% Contextual Padding, and Physical Area Computation (LEVIR-CD)",
    fontsize=14.5, fontweight="bold", color="#0F172A", y=0.95
)

# Save figure
dest_1 = os.path.join(figures_dir, "bounding_box_localization_master.png")
dest_2 = os.path.join(artifact_dir, "bounding_box_region_localization.png")
plt.savefig(dest_1, dpi=300, bbox_inches="tight")
plt.savefig(dest_2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved bounding box localization figures to:\n  {dest_1}\n  {dest_2}")
