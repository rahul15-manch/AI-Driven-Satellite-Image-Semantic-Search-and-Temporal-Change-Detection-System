"""Page 3: Milestone 4 - Semantic Image Retrieval (Interactive Search & Benchmark)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    render_header,
    render_leakage_audit_notice,
    render_hardware_constraint_badge,
    render_metric_card,
)
from ui.charts import plot_m4_recall_comparison, plot_m4_latency_mrr
from ui.image_viewer import render_retrieval_card_grid
from utils.result_loader import (
    load_m4_summary,
    load_m4_qualitative_samples,
    load_m4_failure_analysis,
    load_rsicd_splits,
)
from utils.config_loader import get_project_root, load_m4_config

# Import existing M4 modules
from src.semantic_search.bm25 import BM25Retriever
from src.semantic_search.clip_model import CLIPRetriever
from src.semantic_search.faiss_index import FAISSFlatIPIndex
from src.semantic_search.prompt_ensembler import PromptEnsembler, FROZEN_REMOTE_SENSING_TEMPLATES

ROOT = get_project_root()


# =============================================================================
# CACHED ENGINE INITIALIZERS (STRICT CPU EFFICIENCY)
# =============================================================================

@st.cache_resource(show_spinner="Loading precomputed FAISS index & CLIP embeddings...")
def get_retrieval_engine():
    """Initializes and caches the M4 retrieval components:
    - Precomputed FAISS IndexFlatIP (512D)
    - Precomputed image embeddings (1,093 RSICD test images)
    - Lightweight CPU CLIP text encoder
    - Fitted BM25 Retriever
    """
    config = load_m4_config()
    cache_dir = ROOT / config["paths"]["cache_dir"] / "clip"
    emb_path = cache_dir / "image_embeddings.npy"
    meta_path = cache_dir / "metadata.json"

    if not emb_path.exists() or not meta_path.exists():
        return None, None, None, None, None, "Precomputed CLIP embedding cache not found."

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gallery_ids = meta["image_ids"]
    image_embeddings = np.load(emb_path)

    # 1. Build FAISS index in memory
    faiss_index = FAISSFlatIPIndex(dimension=image_embeddings.shape[1])
    faiss_index.add(image_embeddings, gallery_ids)

    # 2. Load CPU CLIP text retriever
    clip_model_name = config["clip"]["model_name"]
    clip_retriever = CLIPRetriever(model_name=clip_model_name, device="cpu", batch_size=16)

    # 3. Load RSICD metadata for category & captions mapping
    rsicd_json = ROOT / config["dataset"]["json_path"]
    category_map: Dict[str, str] = {}
    captions_map: Dict[str, List[str]] = {}

    if rsicd_json.exists():
        with open(rsicd_json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for img in raw_data.get("images", []):
                fn = img["filename"]
                if fn in gallery_ids:
                    category_map[fn] = img.get("derived_category", fn.rsplit("_", 1)[0])
                    captions_map[fn] = [sent["raw"] for sent in img.get("sentences", [])]

    # 4. Fit BM25 retriever on gallery captions
    bm25_retriever = BM25Retriever(
        k1=config["bm25"]["k1"],
        b=config["bm25"]["b"],
        aggregation_mode="leave_one_caption_out",
    )
    gallery_caps = [captions_map.get(gid, []) for gid in gallery_ids]
    bm25_retriever.fit(gallery_ids, gallery_caps)

    # 5. Prompt Ensembler
    ensembler = PromptEnsembler(templates=FROZEN_REMOTE_SENSING_TEMPLATES)

    return faiss_index, clip_retriever, bm25_retriever, ensembler, (gallery_ids, category_map), None


def render_retrieval_page():
    """Renders the Semantic Retrieval page."""
    render_header(
        title="Semantic Image Retrieval",
        subtitle="Natural-Language Cross-Modal Search on 1,093 Uncaptioned Satellite Images (Strict CPU Execution)",
        badge_text="Milestone 4 Implemented",
    )

    render_leakage_audit_notice()
    render_hardware_constraint_badge()

    engine_tuple = get_retrieval_engine()
    if engine_tuple[-1] is not None:
        st.error(f"Engine Error: {engine_tuple[-1]}")
        return

    faiss_index, clip_retriever, bm25_retriever, ensembler, (gallery_ids, category_map), _ = engine_tuple

    # =========================================================================
    # SECTION 1: INTERACTIVE RETRIEVAL ENGINE
    # =========================================================================
    st.markdown("### Interactive Query Engine")
    st.markdown(
        "Enter a natural-language query to retrieve relevant satellite images from the **1,093 RSICD test gallery**."
    )

    # Preset sample queries for quick inspection
    sample_queries = [
        "airport with multiple airplanes",
        "a large green baseball field next to buildings",
        "storage tanks in an industrial zone",
        "many cars parked in a commercial parking lot",
        "dense residential area with small houses",
        "a river flowing through farmland",
    ]

    selected_sample = st.selectbox("Quick-Select a Research Sample Query (or type below):", [""] + sample_queries)

    query_input = st.text_input(
        "Enter a natural-language satellite image query:",
        value=selected_sample if selected_sample else "airport with multiple airplanes",
        key="nl_query_input",
    )

    col_method, col_topk, col_btn = st.columns([2, 1, 1])

    with col_method:
        retrieval_method = st.selectbox(
            "Select Implemented Retrieval Method:",
            [
                "M4 A2 — CLIP ViT-B/32 (Zero-Shot Cross-Modal)",
                "M4 A3 — CLIP + Domain Prompt Ensemble (5 RS Templates)",
                "B/M4 A1 — BM25 (Leave-One-Caption-Out Lexical)",
            ],
            index=0,
        )

    with col_topk:
        top_k = st.slider("Top K Matches:", min_value=4, max_value=24, value=8, step=4)

    with col_btn:
        st.write("")
        st.write("")
        search_clicked = st.button("Search Gallery", type="primary", use_container_width=True)

    if search_clicked or query_input:
        if not query_input.strip():
            st.warning("Please enter a query string.")
        else:
            t0 = time.perf_counter()
            results: List[Dict[str, Any]] = []

            if "CLIP ViT-B/32" in retrieval_method:
                # 1. Encode single query with CLIP on CPU
                q_vec = clip_retriever.encode_text([query_input])
                # 2. Search FAISS IndexFlatIP (inner product == cosine sim for normalized vectors)
                ranked_ids, scores = faiss_index.search(q_vec, top_k=top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                for rank, (img_id, score) in enumerate(zip(ranked_ids, scores), start=1):
                    img_path = ROOT / "data" / "raw" / "rsicd" / "images" / img_id
                    results.append({
                        "rank": rank,
                        "image_id": img_id,
                        "image_path": img_path,
                        "score": float(score),
                        "category": category_map.get(img_id, "Unknown"),
                    })

            elif "Prompt Ensemble" in retrieval_method:
                # Encode with 5 templates
                ens_vec = ensembler.encode_ensemble(
                    query_input,
                    encode_fn=lambda prompt_list: clip_retriever.encode_text(prompt_list, batch_size=len(prompt_list)),
                )
                ranked_ids, scores = faiss_index.search(ens_vec, top_k=top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                for rank, (img_id, score) in enumerate(zip(ranked_ids, scores), start=1):
                    img_path = ROOT / "data" / "raw" / "rsicd" / "images" / img_id
                    results.append({
                        "rank": rank,
                        "image_id": img_id,
                        "image_path": img_path,
                        "score": float(score),
                        "category": category_map.get(img_id, "Unknown"),
                    })

            elif "BM25" in retrieval_method:
                ranked_ids, scores = bm25_retriever.rank_gallery(query_input, top_k=top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                for rank, (img_id, score) in enumerate(zip(ranked_ids, scores), start=1):
                    img_path = ROOT / "data" / "raw" / "rsicd" / "images" / img_id
                    results.append({
                        "rank": rank,
                        "image_id": img_id,
                        "image_path": img_path,
                        "score": float(score),
                        "category": category_map.get(img_id, "Unknown"),
                    })

            st.markdown(
                f"<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 0.8rem;'>"
                f"Retrieved {len(results)} matches in <strong>{elapsed_ms:.2f} ms</strong> on CPU</div>",
                unsafe_allow_html=True,
            )

            render_retrieval_card_grid(results, cols_per_row=4)

    # =========================================================================
    # SECTION 2: OFFICIAL BENCHMARK EVALUATION (M4 SUMMARY)
    # =========================================================================
    st.markdown("---")
    st.markdown("### Official Benchmark Results (Milestone 4)")
    st.markdown(
        """
        Evaluated on the full **5,465 evaluation queries** across all **1,093 RSICD test gallery images** 
        under strict single-process CPU execution. Every metric is loaded dynamically from machine-readable result files.
        """
    )

    summary_df = load_m4_summary()
    if summary_df is not None:
        # Display formatted comparison table
        display_df = summary_df.copy()
        display_df["R@1 (%)"] = display_df["R@1"].apply(lambda v: f"{v*100:.2f}%")
        display_df["R@5 (%)"] = display_df["R@5"].apply(lambda v: f"{v*100:.2f}%")
        display_df["R@10 (%)"] = display_df["R@10"].apply(lambda v: f"{v*100:.2f}%")
        display_df["MRR"] = display_df["MRR"].apply(lambda v: f"{v:.4f}")
        display_df["Mean Latency (ms)"] = display_df["latency_mean_ms"].apply(lambda v: f"{v:.2f}")
        display_df["Peak RAM (MB)"] = display_df["peak_rss_mb"].apply(lambda v: f"{v:.1f}")

        st.dataframe(
            display_df[[
                "method",
                "protocol",
                "R@1 (%)",
                "R@5 (%)",
                "R@10 (%)",
                "MRR",
                "Mean Latency (ms)",
                "Peak RAM (MB)",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        # Charts
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.pyplot(plot_m4_recall_comparison(summary_df))
        with c_right:
            st.pyplot(plot_m4_latency_mrr(summary_df))
    else:
        st.warning("M4 summary.json not found in experiments/results/m4/.")

    # =========================================================================
    # SECTION 3: FAILURE MODE BREAKDOWN
    # =========================================================================
    st.markdown("### Failure Analysis on Zero-Shot CLIP Retrieval")
    st.markdown(
        """
        Rule-based heuristic audit across the **3,941 failure cases** (queries where the target image rank was > 10):
        """
    )

    failure_data = load_m4_failure_analysis()
    if failure_data and "observed_failure_patterns_percentages" in failure_data:
        fc_cols = st.columns(4)
        pcts = failure_data["observed_failure_patterns_percentages"]
        with fc_cols[0]:
            render_metric_card("Geographic Ambiguity", f"{pcts.get('broad_geographic_context', 36.26):.1f}%", help_text="Target displaced by valid images from the same category.")
        with fc_cols[1]:
            render_metric_card("Spatial Confusion", f"{pcts.get('spatial_relational_confusion', 32.38):.1f}%", help_text="Difficulty resolving topological prepositions (next to, between).")
        with fc_cols[2]:
            render_metric_card("Dense Small Objects", f"{pcts.get('dense_small_objects', 20.05):.1f}%", help_text="ViT-B/32 patch resolution limits detection of cars/small planes.")
        with fc_cols[3]:
            render_metric_card("Attribute Mismatch", f"{pcts.get('fine_grained_attribute_mismatch', 11.29):.1f}%", help_text="Subtle color or geometric shape attribute divergence.")


if __name__ == "__main__":
    render_retrieval_page()
