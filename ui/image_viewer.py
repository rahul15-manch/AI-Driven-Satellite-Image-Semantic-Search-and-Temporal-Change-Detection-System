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
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem; margin-bottom: 0.8rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <span style="background: #1e293b; color: #ffffff; font-weight: 700; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 4px;">
                                Rank #{rank}
                            </span>
                            <span style="font-size: 0.75rem; color: #475569; font-weight: 600; text-transform: capitalize;">
                                {cat}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if img_path.exists():
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                else:
                    st.warning(f"Image not found: {img_path.name}")

                score_label = f"Score: {score:.4f}" if score is not None else "Score: N/A"
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
        st.image(img_t1, caption="T1 Optical (LEVIR-CD)", use_container_width=True)

    with col2:
        st.markdown("**2. T2 Post-Change**")
        st.image(img_t2, caption="T2 Optical (LEVIR-CD)", use_container_width=True)

    with col3:
        st.markdown("**3. Ground Truth**")
        gt_disp = (gt_mask * 255).astype(np.uint8) if gt_mask.max() <= 1 else gt_mask.astype(np.uint8)
        st.image(gt_disp, caption=f"Ground Truth ({np.mean(gt_mask > 0)*100:.2f}% changed)", use_container_width=True)

    with col4:
        st.markdown("**4. Predicted Map**")
        pred_disp = (pred_mask * 255).astype(np.uint8) if pred_mask.max() <= 1 else pred_mask.astype(np.uint8)
        st.image(pred_disp, caption=f"Prediction ({np.mean(pred_mask > 0)*100:.2f}% changed)", use_container_width=True)

    # Optional detailed diagnostics
    if diff_map is not None or show_error_overlay:
        st.markdown("---")
        diag_cols = st.columns(2)
        if diff_map is not None:
            with diag_cols[0]:
                st.markdown("**Continuous Difference Score Map**")
                # Normalize difference map to 0-255 for display
                d_norm = ((diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8) * 255).astype(np.uint8)
                st.image(d_norm, caption="Continuous Difference Map (0.0 to 1.0)", use_container_width=True)

        if show_error_overlay:
            with diag_cols[1]:
                st.markdown("**Classification Error Overlay**")
                overlay = create_error_overlay(pred_mask, gt_mask)
                st.image(overlay, caption="Green=TP, Red=FP, Blue=FN, Dark=TN", use_container_width=True)
