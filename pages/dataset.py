"""Page 2: Dataset & Preprocessing Pipeline (RSICD & LEVIR-CD)."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import streamlit as st

from ui.components import render_header, render_metric_card
from utils.result_loader import (
    load_rsicd_splits,
    load_levir_splits,
    load_rsicd_validation_report,
    get_available_rsicd_images,
    get_available_levir_test_pairs,
)
from utils.config_loader import get_project_root

ROOT = get_project_root()


def render_dataset_page():
    """Renders the Dataset & Preprocessing exploration page."""
    render_header(
        title="Dataset Acquisition, Verification & Data Pipeline",
        subtitle="Audited Remote Sensing Benchmarks: RSICD Cross-Modal Retrieval & LEVIR-CD Bi-Temporal Change Detection",
        badge_text="Milestone 2 Implemented",
    )

    tabs = st.tabs(["RSICD (Semantic Retrieval)", "LEVIR-CD (Change Detection)", "Preprocessing Architecture"])

    # =========================================================================
    # TAB 1: RSICD
    # =========================================================================
    with tabs[0]:
        st.markdown("### RSICD Benchmark (Remote Sensing Image Captioning Dataset)")
        rsicd_splits = load_rsicd_splits()

        if rsicd_splits:
            counts = rsicd_splits.get("counts", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Total Images", counts.get("total", "10,921"))
            with c2:
                render_metric_card("Train Split", counts.get("train", "8,734"))
            with c3:
                render_metric_card("Validation Split", counts.get("val", "1,094"))
            with c4:
                render_metric_card("Test Split (Gallery)", counts.get("test", "1,093"))
        else:
            st.warning("RSICD split manifest could not be loaded from data/splits/rsicd/rsicd_splits.json.")

        st.markdown(
            """
            - **Image Dimensions:** 224 &times; 224 pixels (RGB, 3 channels)
            - **Captions per Image:** Exactly 5 natural-language descriptive captions (54,605 total captions in full corpus)
            - **Evaluation Protocol (M4):** 5,465 evaluation queries evaluated against the 1,093 test gallery images
            - **Categories (30 Semantic Classes):** Airport, Bareland, Baseball field, Beach, Bridge, Center, Church, Commercial, Dense residential, Desert, Farmland, Forest, Industrial, Meadow, Medium residential, Mountain, Park, Parking, Playground, Pond, Port, Railway station, Resort, River, School, Sparse residential, Square, Stadium, Storage tank, Viaduct.
            """
        )

        st.markdown("#### Sample RSICD Image Gallery")
        rsicd_imgs = get_available_rsicd_images()
        if rsicd_imgs:
            sample_options = rsicd_imgs[:60]
            selected_img_name = st.selectbox(
                "Select a sample RSICD image to preview:",
                sample_options,
                index=0,
                key="rsicd_sample_select",
            )
            img_path = ROOT / "data" / "raw" / "rsicd" / "images" / selected_img_name
            if img_path.exists():
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.image(Image.open(img_path), caption=selected_img_name, width=224)
                with col_info:
                    st.markdown(f"**Filename:** `{selected_img_name}`")
                    st.markdown(f"**Derived Category:** `{selected_img_name.rsplit('_', 1)[0]}`")
                    st.markdown(f"**Resolution:** `224 x 224 px`")
                    st.markdown(f"**Storage Path:** `data/raw/rsicd/images/{selected_img_name}`")
        else:
            st.info("RSICD raw images directory not found at data/raw/rsicd/images/.")

    # =========================================================================
    # TAB 2: LEVIR-CD
    # =========================================================================
    with tabs[1]:
        st.markdown("### LEVIR-CD Benchmark (Bi-Temporal Building Change Detection)")
        levir_splits = load_levir_splits()

        if levir_splits:
            l_counts = levir_splits.get("counts", {})
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                render_metric_card("Total Bi-Temporal Pairs", l_counts.get("total", "637"))
            with lc2:
                render_metric_card("Train Pairs", l_counts.get("train", "445"))
            with lc3:
                render_metric_card("Validation Pairs", l_counts.get("val", "64"))
            with lc4:
                render_metric_card("Test Pairs (Evaluation)", l_counts.get("test", "128"))
        else:
            st.warning("LEVIR-CD split manifest could not be loaded from data/splits/levir_cd/levir_splits.json.")

        st.markdown(
            """
            - **Spatial Resolution:** 0.5 meters Ground Sample Distance (GSD)
            - **Native Image Pair Dimensions:** 1024 &times; 1024 pixels (RGB optical bi-temporal imagery)
            - **Temporal Span:** 5 to 14 years between acquisition times $T_1$ and $T_2$
            - **Change Annotations:** Pixel-level binary masks focusing on significant building construction and expansion
            - **Class Imbalance in Test Split:** Changed pixels constitute only **5.094%** of the 134,217,728 test pixels (severe negative skew)
            """
        )

        st.markdown("#### LEVIR-CD Test Pair Browser")
        levir_pairs = get_available_levir_test_pairs()
        if levir_pairs:
            # Pick representative examples
            sample_choice = st.selectbox(
                "Select a LEVIR-CD test pair to inspect:",
                levir_pairs[:30],
                index=0,
                key="levir_sample_select",
            )

            p_t1 = ROOT / "data" / "raw" / "levir_cd" / "test" / "A" / f"{sample_choice}.png"
            p_t2 = ROOT / "data" / "raw" / "levir_cd" / "test" / "B" / f"{sample_choice}.png"
            p_gt = ROOT / "data" / "raw" / "levir_cd" / "test" / "label" / f"{sample_choice}.png"

            if p_t1.exists() and p_t2.exists() and p_gt.exists():
                c_a, c_b, c_gt = st.columns(3)
                with c_a:
                    st.image(Image.open(p_t1), caption=f"{sample_choice} - T1 (Pre-Change)", use_container_width=True)
                with c_b:
                    st.image(Image.open(p_t2), caption=f"{sample_choice} - T2 (Post-Change)", use_container_width=True)
                with c_gt:
                    st.image(Image.open(p_gt), caption=f"{sample_choice} - Ground Truth Mask", use_container_width=True)
            else:
                st.warning(f"Images for {sample_choice} are incomplete on disk.")
        else:
            st.info("LEVIR-CD test images not found at data/raw/levir_cd/test/.")

    # =========================================================================
    # TAB 3: PREPROCESSING ARCHITECTURE
    # =========================================================================
    with tabs[2]:
        st.markdown("### Preprocessing Pipeline & Patch Extraction")
        st.markdown(
            """
            To respect strict CPU memory budgets and avoid memory allocation spikes, 
            high-resolution $1024 \\times 1024$ image pairs are processed via deterministic patching:
            """
        )

        st.markdown(
            """
            ```
            [Input 1024×1024 T1/T2 Pair]
                       ↓
            [Integrity Verification & Shape Check]
                       ↓
            [RGB Float32 Normalization: [0, 255] → [0.0, 1.0]]
                       ↓
            [Deterministic Patch Extraction (PatchExtractor)]
                       ↓
            [16 Non-Overlapping 256×256 Patches (Stride=256, Padding=0)]
                       ↓
            [Detector Forward Pass on Patches (B1, B2, B3)]
                       ↓
            [Seamless Mosaic Reconstruction → 1024×1024 Continuous Map]
                       ↓
            [Frozen Validation Threshold Applied (τ*)]
                       ↓
            [1024×1024 Binary Change Map Output]
            ```
            """
        )

        st.markdown("#### Mathematical Guarantee of Non-Overlapping Reconstruction")
        st.markdown(
            """
            With patch size $P=256$ and stride $S=256$ across an image of dimension $W=H=1024$:
            $$\\text{Patches per Axis} = \\frac{1024}{256} = 4 \\implies 4 \\times 4 = 16 \\text{ patches}$$
            $$\\text{Total Evaluation Pixels per Pair} = 16 \\times (256 \\times 256) = 1,048,576 \\text{ pixels (Exact}$$
            Zero border artifacts, zero overlapping blend ambiguities, and zero pixel omissions.
            """
        )


if __name__ == "__main__":
    render_dataset_page()
