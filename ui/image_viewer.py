"""Side-by-side image comparison viewers and retrieval card grid components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import streamlit as st


def create_error_overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Creates RGB visual audit overlay:
    - True Positives (TP): Green [0, 220, 0]
    - False Positives (FP): Red [230, 20, 20]
    - False Negatives (FN): Blue [30, 100, 255]
    - True Negatives (TN): Dark background [20, 20, 20]
    """
    h, w = pred.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    p = (pred > 0)
    g = (gt > 0)
    
    # TN
    overlay[~p & ~g] = [30, 35, 45]
    # TP (Green)
    overlay[p & g] = [34, 197, 94]
    # FP (Red)
    overlay[p & ~g] = [239, 68, 68]
    # FN (Blue/Cyan)
    overlay[~p & g] = [59, 130, 246]
    
    return overlay


def render_retrieval_card_grid(results: List[Dict[str, Any]], cols_per_row: int = 4):
    """Renders ranked semantic retrieval matches in an academic card layout.

    Each result dict contains:
    - 'rank': int
    - 'image_id': str
    - 'image_path': Path or str
    - 'score': float (cosine similarity or BM25 score)
    - 'category': str
    """
    if not results:
        st.info("No retrieval results to display.")
        return

    for i in range(0, len(results), cols_per_row):
        batch = results[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_idx, item in enumerate(batch):
            with cols[col_idx]:
                rank = item.get("rank", i + col_idx + 1)
                score = item.get("score", None)
                cat = item.get("category", "Unknown")
                img_path = Path(item["image_path"])

                st.markdown(
                    f"""
                    <div style="background: #E3DAC9; border: 1px solid #BFAF9E; border-radius: 8px; padding: 0.65rem; margin-bottom: 0.8rem; box-shadow: 0 1px 3px rgba(62,59,57,0.06);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <span style="background: #4A2E2B; color: #F1EBE4; font-weight: 700; font-size: 0.75rem; padding: 0.18rem 0.55rem; border-radius: 4px;">
                                Rank #{rank}
                            </span>
                            <span style="font-size: 0.75rem; color: #3E3B39; font-weight: 600; text-transform: capitalize;">
                                {cat}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if img_path.exists():
                    img = Image.open(img_path)
                    st.image(img, width="stretch")
                else:
                    st.warning(f"Image not found: {img_path.name}")

                score_type = item.get("score_type", "Score")
                if score is not None:
                    score_label = f"{score_type}: {score:.4f}"
                else:
                    score_label = f"{score_type}: N/A"
                st.caption(f"**{item.get('image_id', img_path.name)}** · {score_label}")


def render_change_detection_panel(
    img_t1: Image.Image | np.ndarray,
    img_t2: Image.Image | np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    diff_map: Optional[np.ndarray] = None,
    show_error_overlay: bool = True,
):
    """Renders standard 4-column change detection comparison layout:
    Col 1: T1 Satellite Image
    Col 2: T2 Satellite Image
    Col 3: Ground Truth Binary Mask
    Col 4: Predicted Change Map (or Error Overlay)
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**1. T1 Pre-Change**")
        st.image(img_t1, caption="T1 Optical (LEVIR-CD)", width="stretch")

    with col2:
        st.markdown("**2. T2 Post-Change**")
        st.image(img_t2, caption="T2 Optical (LEVIR-CD)", width="stretch")

    with col3:
        st.markdown("**3. Ground Truth**")
        gt_disp = (gt_mask * 255).astype(np.uint8) if gt_mask.max() <= 1 else gt_mask.astype(np.uint8)
        st.image(gt_disp, caption=f"Ground Truth ({np.mean(gt_mask > 0)*100:.2f}% changed)", width="stretch")

    with col4:
        st.markdown("**4. Predicted Map**")
        pred_disp = (pred_mask * 255).astype(np.uint8) if pred_mask.max() <= 1 else pred_mask.astype(np.uint8)
        st.image(pred_disp, caption=f"Prediction ({np.mean(pred_mask > 0)*100:.2f}% changed)", width="stretch")

    # Optional detailed diagnostics
    if diff_map is not None or show_error_overlay:
        st.markdown("---")
        diag_cols = st.columns(2)
        if diff_map is not None:
            with diag_cols[0]:
                st.markdown("**Continuous Difference Score Map**")
                # Normalize difference map to 0-255 for display
                d_norm = ((diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8) * 255).astype(np.uint8)
                st.image(d_norm, caption="Continuous Difference Map (0.0 to 1.0)", width="stretch")

        if show_error_overlay:
            with diag_cols[1]:
                st.markdown("**Classification Error Overlay**")
                overlay = create_error_overlay(pred_mask, gt_mask)
                st.image(overlay, caption="Green=TP, Red=FP, Blue=FN, Dark=TN", width="stretch")


def create_temporal_overlay(
    image_rgb: np.ndarray | Image.Image,
    mask: np.ndarray,
    regions: Optional[List[Dict[str, Any]]] = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Generates an academic spatial overlay combining change detection and region bounds.

    - Changed area mask highlighted in Terracotta (#C86B4D, RGB: 200, 107, 77)
    - Ranked discrete change regions outlined in Yellow Ochre (#E1A95F, RGB: 225, 169, 95)
    - Region labels drawn with top-left origin coordinates.
    """
    import cv2

    if isinstance(image_rgb, Image.Image):
        base = np.array(image_rgb.convert("RGB"), dtype=np.uint8)
    else:
        base = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    h, w = base.shape[:2]
    vis = base.copy()

    # 1. Overlay change mask
    m = (mask > 0)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    if np.any(m):
        color_mask = np.zeros_like(vis)
        color_mask[m] = [200, 107, 77]  # Terracotta accent
        vis = np.where(m[:, :, None], cv2.addWeighted(vis, 1.0 - alpha, color_mask, alpha, 0), vis)

    # 2. Draw ranked region bounding boxes and badges
    if regions:
        for r in regions:
            bbox = r.get("bbox") if isinstance(r, dict) else getattr(r, "bbox", None)
            rid = r.get("region_id") if isinstance(r, dict) else getattr(r, "region_id", None)
            cat = r.get("change_category") if isinstance(r, dict) else getattr(r, "change_category", None)
            if not bbox or len(bbox) != 4:
                continue

            xmin, ymin, xmax, ymax = [int(v) for v in bbox]
            xmin = max(0, min(w - 1, xmin))
            xmax = max(0, min(w - 1, xmax))
            ymin = max(0, min(h - 1, ymin))
            ymax = max(0, min(h - 1, ymax))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Draw crisp bounding box
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (225, 169, 95), 2)  # Yellow Ochre

            # Label badge
            label_text = f"#{rid}" if rid is not None else "Region"
            if cat and cat != "Uncertain":
                label_text += f": {cat}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, ymin - th - 6)
            badge_y2 = ymin
            badge_x2 = min(w, xmin + tw + 8)

            cv2.rectangle(vis, (xmin, badge_y1), (badge_x2, badge_y2), (74, 46, 43), -1)  # Deep Bark
            cv2.putText(
                vis,
                label_text,
                (xmin + 4, ymin - 4),
                font,
                font_scale,
                (241, 235, 228),  # Alabaster
                thickness,
                lineType=cv2.LINE_AA,
            )

    return vis


def render_temporal_understanding_panel(
    img_t1: Image.Image | np.ndarray,
    img_t2_aligned: Image.Image | np.ndarray,
    pred_mask: np.ndarray,
    spatial_overlay: np.ndarray,
    caption_t1: str = "T1 Optical Scene",
    caption_t2: str = "T2 Registered Scene",
    caption_mask: str = "Predicted Binary Mask",
    caption_overlay: str = "Spatial Change Overlay",
):
    """Renders 4-way balanced visualization panel for temporal change understanding."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**1. Observation T1**")
        st.image(img_t1, caption=caption_t1, width="stretch")

    with col2:
        st.markdown("**2. Observation T2 (Aligned)**")
        st.image(img_t2_aligned, caption=caption_t2, width="stretch")

    with col3:
        st.markdown("**3. Binary Change Mask**")
        mask_disp = (pred_mask * 255).astype(np.uint8) if pred_mask.max() <= 1 else pred_mask.astype(np.uint8)
        st.image(mask_disp, caption=caption_mask, width="stretch")

    with col4:
        st.markdown("**4. Spatial Overlay & Regions**")
        st.image(spatial_overlay, caption=caption_overlay, width="stretch")

