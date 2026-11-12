"""Service layer for semantic search, change detection inference, and benchmark access."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import logging
import uuid

from src.change_detection.base import BaseChangeDetector
from src.change_detection.bit_cd import BITChangeDetector, load_checkpoint as load_m9_ckpt
from src.change_detection.cva_detector import CVADetector
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from src.change_detection.fc_siam_diff import FCSiamDiffDetector, load_checkpoint as load_m7_ckpt
from src.change_detection.pixel_diff import PixelDiffDetector
from src.change_detection.qat_cd import QualityAwareChangeDetector
from src.change_detection.snunet import SNUNetDetector, load_checkpoint as load_m8_ckpt
from src.change_detection.ssim_detector import SSIMDetector
from src.data.levir_loader import LEVIRDataset
from src.data.patch_extractor import PatchExtractor
from src.semantic_search.bm25 import BM25Retriever
from src.semantic_search.clip_model import CLIPRetriever
from src.semantic_search.faiss_index import FAISSFlatIPIndex
from src.semantic_search.prompt_ensembler import PromptEnsembler
from src.temporal_understanding import (
    ImageCoRegistrator,
    RegistrationPolicy,
    RegistrationStatus,
    RegionExtractor,
    RegionExtractionConfig,
    SemanticChangeAnalyzer,
    VisionAnalyzerConfig,
    TemporalExplanationEngine,
    TemporalExplanationConfig,
)
from utils.config_loader import get_project_root, load_m4_config, load_m11_config

logger = logging.getLogger(__name__)

ROOT = get_project_root()


def array_to_png_data_uri(arr: np.ndarray, is_mask: bool = False, is_heatmap: bool = False) -> str:
    """Encodes a 2D or 3D numpy array to a base64 PNG data URI."""
    if is_mask:
        # Binary mask {0, 1} -> black and white
        mask_uint8 = (arr > 0).astype(np.uint8) * 255
        img = Image.fromarray(mask_uint8, mode="L")
    elif is_heatmap:
        # Continuous diff map in [0, 1] -> Inferno/Turbo colormap
        norm = np.clip(arr, 0.0, 1.0)
        cmap = plt.get_cmap("inferno")
        rgba = (cmap(norm) * 255).astype(np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
    else:
        # Standard RGB uint8 array
        img = Image.fromarray(arr.astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_error_overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Generates an RGB diagnostic error overlay.
    - True Positive (TP): Emerald Green [34, 197, 94]
    - False Positive (FP): Crimson Red [239, 68, 68]
    - False Negative (FN): Royal Blue [59, 130, 246]
    - True Negative (TN): Dark Gray [30, 41, 59]
    """
    p = pred.astype(bool)
    g = gt.astype(bool)

    h, w = pred.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:] = [30, 41, 59]  # Dark background
    overlay[p & g] = [34, 197, 94]     # TP: Green
    overlay[p & ~g] = [239, 68, 68]    # FP: Red
    overlay[~p & g] = [59, 130, 246]   # FN: Blue

    return overlay


class RetrievalService:
    """Thread-safe, cached service orchestrating M4 semantic retrieval."""

    def __init__(self):
        self._initialized = False
        self.faiss_index: Optional[FAISSFlatIPIndex] = None
        self.clip_retriever: Optional[CLIPRetriever] = None
        self.prompt_ensembler: Optional[PromptEnsembler] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.gallery_ids: List[str] = []
        self.category_map: Dict[str, str] = {}
        self.captions_map: Dict[str, List[str]] = {}

    def initialize(self) -> None:
        """Loads FAISS index, gallery embeddings, and model weights lazily."""
        if self._initialized:
            return

        config = load_m4_config()
        raw_cache = Path(config["paths"]["cache_dir"])
        if raw_cache.name == "clip":
            cache_dir = ROOT / raw_cache
        elif (ROOT / raw_cache / "clip").exists():
            cache_dir = ROOT / raw_cache / "clip"
        else:
            cache_dir = ROOT / raw_cache

        emb_path = cache_dir / "image_embeddings.npy"
        meta_path = cache_dir / "metadata.json"

        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"CLIP embedding cache not found at {cache_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.gallery_ids = meta["image_ids"]
        image_embeddings = np.load(emb_path)

        # 1. Build FAISS index
        self.faiss_index = FAISSFlatIPIndex(dimension=image_embeddings.shape[1])
        self.faiss_index.add(image_embeddings, self.gallery_ids)

        # 2. Load CPU CLIP text encoder
        clip_model_name = config["clip"]["model_name"]
        self.clip_retriever = CLIPRetriever(model_name=clip_model_name, device="cpu", batch_size=16)

        # 3. Prompt Ensembler
        self.prompt_ensembler = PromptEnsembler()

        # 4. Load RSICD metadata
        rsicd_json = ROOT / config["dataset"]["json_path"]
        if rsicd_json.exists():
            with open(rsicd_json, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                for img in raw_data.get("images", []):
                    fn = img["filename"]
                    if fn in self.gallery_ids:
                        self.category_map[fn] = img.get("derived_category", fn.rsplit("_", 1)[0])
                        self.captions_map[fn] = [sent["raw"] for sent in img.get("sentences", [])]
        else:
            # Serverless fallback: derive categories and semantic templates from gallery filenames
            for fn in self.gallery_ids:
                cat = fn.rsplit("_", 1)[0].replace("_", " ")
                self.category_map[fn] = cat
                self.captions_map[fn] = [
                    f"an aerial photograph of {cat}",
                    f"a remote sensing image showing {cat}",
                    f"satellite view of {cat}",
                ]

        # 5. Fit BM25 retriever
        self.bm25_retriever = BM25Retriever(
            k1=config["bm25"]["k1"],
            b=config["bm25"]["b"],
            aggregation_mode="leave_one_caption_out",
        )
        self.bm25_retriever.fit(self.captions_map, self.gallery_ids)

        self._initialized = True

    def search(self, query: str, method: str = "clip", top_k: int = 8) -> Dict[str, Any]:
        """Executes ranked retrieval matching query text."""
        self.initialize()
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []

        method_key = method.lower()
        if method_key == "clip":
            method_display = "M4 A2 — CLIP ViT-B/32 (Zero-Shot Cross-Modal)"
            score_type = "Cosine Similarity"
            q_vec = self.clip_retriever.encode_text([query])
            ranked_ids, scores = self.faiss_index.search(q_vec, top_k=top_k)
        elif method_key == "prompt_ensemble":
            method_display = "M4 A3 — CLIP + Domain Prompt Ensemble (5 RS Templates)"
            score_type = "Ensemble Cosine Similarity"
            ens_vec = self.prompt_ensembler.encode_ensemble(
                query,
                encode_fn=lambda prompt_list: self.clip_retriever.encode_text(prompt_list, batch_size=len(prompt_list)),
            )
            ranked_ids, scores = self.faiss_index.search(ens_vec, top_k=top_k)
        elif method_key == "bm25":
            method_display = "B/M4 A1 — BM25 (Leave-One-Caption-Out Lexical)"
            score_type = "BM25 Score"
            from src.semantic_search.bm25 import tokenize
            query_tokens = tokenize(query)
            all_scores = self.bm25_retriever.score_query_combined(query_tokens)
            ranked_indices = sorted(
                range(self.bm25_retriever.num_docs),
                key=lambda idx: (all_scores[idx], -idx),
                reverse=True,
            )[:top_k]
            ranked_ids = [self.bm25_retriever.image_ids[idx] for idx in ranked_indices]
            scores = [float(all_scores[idx]) for idx in ranked_indices]
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        for rank, (img_id, score) in enumerate(zip(ranked_ids, scores), start=1):
            results.append({
                "rank": rank,
                "image_id": img_id,
                "filename": img_id,
                "image_url": f"/api/image/rsicd/{img_id}",
                "score": round(float(score), 4),
                "similarity_score": round(float(score), 4),
                "score_type": score_type,
                "category": self.category_map.get(img_id, "Unknown"),
            })

        return {
            "query": query,
            "method": method_key,
            "method_display": method_display,
            "total_results": len(results),
            "latency_ms": round(elapsed_ms, 2),
            "results": results,
        }


class ChangeDetectionService:
    """Service orchestrating bi-temporal change detection inference on CPU."""

    def __init__(self):
        self.extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
        self.detectors: Dict[str, BaseChangeDetector] = {}
        self.thresholds: Dict[str, float] = {
            "B1_Pixel_Diff": 0.4100,
            "B2_SSIM": 0.9000,
            "B3_CVA": 0.4050,
            "M7_FC_Siam_diff": 0.5500,
            "M8_SNUNet": 0.5000,
            "M9_BIT_CD": 0.4500,
            "M10_QAT_CD": 0.4500,
        }
        self.model_displays: Dict[str, str] = {
            "B1_Pixel_Diff": "B1 Pixel Difference (L1 Radiometric Baseline)",
            "B2_SSIM": "B2 SSIM Dissimilarity (Structural Baseline)",
            "B3_CVA": "B3 Change Vector Analysis (Spectral Baseline)",
            "M7_FC_Siam_diff": "M7 FC-Siam-diff (Fully Convolutional Siamese U-Net)",
            "M8_SNUNet": "M8 SNUNet-CD (Dense Siamese + ECAM Attention)",
            "M9_BIT_CD": "M9 BIT-CD (Bitemporal Image Transformer)",
            "M10_QAT_CD": "M10 QAT-CD (Quality-Aware Temporal Change Detection)",
        }
        self._init_classical()

    def _init_classical(self) -> None:
        self.detectors["B1_Pixel_Diff"] = PixelDiffDetector()
        self.detectors["B2_SSIM"] = SSIMDetector()
        self.detectors["B3_CVA"] = CVADetector()

    def get_detector(self, model_id: str, gamma: Optional[float] = None) -> Tuple[BaseChangeDetector, float]:
        """Loads and returns detector instance and frozen threshold."""
        clean_id = model_id.strip()
        if clean_id not in self.model_displays:
            raise ValueError(f"Unsupported model identifier '{model_id}'. Supported models: {list(self.model_displays.keys())}")

        tau = self.thresholds[clean_id]

        if clean_id in self.detectors:
            return self.detectors[clean_id], tau

        # Lazy load learned models
        if clean_id == "M7_FC_Siam_diff":
            ckpt = ROOT / "experiments" / "checkpoints" / "m7" / "fc_siam_diff_best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"M7 checkpoint not found at {ckpt}")
            m, _ = load_m7_ckpt(ckpt, device="cpu")
            self.detectors[clean_id] = FCSiamDiffDetector(model=m, device="cpu")

        elif clean_id == "M8_SNUNet":
            ckpt = ROOT / "experiments" / "checkpoints" / "m8" / "snunet_best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"M8 checkpoint not found at {ckpt}")
            m, _ = load_m8_ckpt(ckpt, device="cpu")
            self.detectors[clean_id] = SNUNetDetector(model=m, device="cpu")

        elif clean_id == "M9_BIT_CD":
            ckpt = ROOT / "experiments" / "checkpoints" / "m9" / "bit_cd_best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"M9 checkpoint not found at {ckpt}")
            m, _ = load_m9_ckpt(ckpt, device="cpu")
            self.detectors[clean_id] = BITChangeDetector(model=m, device="cpu")

        elif clean_id == "M10_QAT_CD":
            bit_det, _ = self.get_detector("M9_BIT_CD")
            g = 0.0 if gamma is None else float(gamma)
            self.detectors[clean_id] = QualityAwareChangeDetector(
                base_detector=bit_det,
                gamma=g,
                weights=(0.34, 0.33, 0.33),
            )

        return self.detectors[clean_id], tau

    def list_available_pairs(self) -> List[Dict[str, Any]]:
        """Returns verified LEVIR-CD test scene pairs (raw or curated demo pack)."""
        test_dir = ROOT / "data" / "raw" / "levir_cd" / "test"
        is_raw = test_dir.exists()
        if not is_raw:
            test_dir = ROOT / "data" / "demo_samples"

        dir_a = test_dir / "A"
        dir_b = test_dir / "B"
        dir_lbl = test_dir / "label"

        if not dir_a.exists():
            return []

        pairs = []
        for img_p in sorted(dir_a.glob("*.png")):
            stem = img_p.stem
            has_b = (dir_b / f"{stem}.png").exists()
            has_gt = (dir_lbl / f"{stem}.png").exists()
            if has_b:
                pairs.append({
                    "pair_id": stem,
                    "dataset": "LEVIR-CD",
                    "image_t1_url": f"/api/image/levir_cd/test/A/{stem}.png" if is_raw else f"/api/image/levir_cd/A/{stem}.png",
                    "image_t2_url": f"/api/image/levir_cd/test/B/{stem}.png" if is_raw else f"/api/image/levir_cd/B/{stem}.png",
                    "has_ground_truth": has_gt,
                })
        return pairs

    def detect_change(
        self,
        pair_id: Optional[str] = None,
        model_id: str = "M9_BIT_CD",
        threshold_override: Optional[float] = None,
        gamma: Optional[float] = None,
        img_a: Optional[np.ndarray] = None,
        img_b: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Performs full change detection inference pipeline."""
        detector, frozen_tau = self.get_detector(model_id, gamma=gamma)
        tau = threshold_override if threshold_override is not None else frozen_tau
        thresh_source = "User Override" if threshold_override is not None else "Validation F1-Optimal (Frozen)"

        gt_mask: Optional[np.ndarray] = None
        img_t1_url: Optional[str] = None
        img_t2_url: Optional[str] = None

        if pair_id:
            p_t1 = ROOT / "data" / "raw" / "levir_cd" / "test" / "A" / f"{pair_id}.png"
            p_t2 = ROOT / "data" / "raw" / "levir_cd" / "test" / "B" / f"{pair_id}.png"
            p_gt = ROOT / "data" / "raw" / "levir_cd" / "test" / "label" / f"{pair_id}.png"

            if not (p_t1.exists() and p_t2.exists()):
                p_t1 = ROOT / "data" / "demo_samples" / "A" / f"{pair_id}.png"
                p_t2 = ROOT / "data" / "demo_samples" / "B" / f"{pair_id}.png"
                p_gt = ROOT / "data" / "demo_samples" / "label" / f"{pair_id}.png"

            if not (p_t1.exists() and p_t2.exists()):
                raise FileNotFoundError(f"LEVIR-CD test pair '{pair_id}' not found on disk.")

            img_a = np.array(Image.open(p_t1).convert("RGB"), dtype=np.uint8)
            img_b = np.array(Image.open(p_t2).convert("RGB"), dtype=np.uint8)
            if p_gt.exists():
                gt_mask = (np.array(Image.open(p_gt).convert("L"), dtype=np.uint8) > 128).astype(np.uint8)

            img_t1_url = f"/api/image/levir_cd/test/A/{pair_id}.png" if (ROOT / "data" / "raw" / "levir_cd" / "test").exists() else f"/api/image/levir_cd/A/{pair_id}.png"
            img_t2_url = f"/api/image/levir_cd/test/B/{pair_id}.png" if (ROOT / "data" / "raw" / "levir_cd" / "test").exists() else f"/api/image/levir_cd/B/{pair_id}.png"

        elif img_a is None or img_b is None:
            raise ValueError("Must provide either a pair_id or both img_a and img_b arrays.")

        # Inference timing
        t0 = time.perf_counter()

        # Patch extraction and mosaic reconstruction
        patches = self.extractor.extract_from_pair(img_a, img_b, parent_stem=pair_id or "uploaded")
        diff_patches = []
        for p_a, p_b, _, p_meta in patches:
            if model_id == "M10_QAT_CD":
                # Base probability
                d_p = detector.base_detector.compute_difference_map(p_a, p_b)
            else:
                d_p = detector.compute_difference_map(p_a, p_b)
            diff_patches.append((d_p, p_meta))

        base_diff = self.extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)

        diagnostics_data = None
        if model_id == "M10_QAT_CD":
            diag = detector.compute_quality_diagnostics(img_a, img_b, base_prob=base_diff)
            full_diff = diag["gated_prob"]
            if return_diagnostics:
                diagnostics_data = {
                    "gamma_attenuation": detector.gamma,
                    "mean_registration_discrepancy": round(float(diag["reg_map"].mean()), 4),
                    "mean_illumination_gradient": round(float(diag["illum_map"].mean()), 4),
                    "mean_sharpness_asymmetry": round(float(diag["sharp_map"].mean()), 4),
                    "mean_margin_uncertainty": round(float(diag["uncertainty_map"].mean()), 4),
                }
        else:
            full_diff = base_diff

        pred_mask = (full_diff >= tau).astype(np.uint8)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Encode visualizations to Data URIs
        heatmap_uri = array_to_png_data_uri(full_diff, is_heatmap=True)
        mask_uri = array_to_png_data_uri(pred_mask, is_mask=True)

        metrics_obj = None
        error_overlay_uri = None
        if gt_mask is not None:
            cm = ConfusionMatrix()
            cm.update(pred_mask, gt_mask)
            eval_dict = ChangeDetectionEvaluator.evaluate(cm)
            metrics_obj = {
                "precision": round(float(eval_dict["precision"]), 4),
                "recall": round(float(eval_dict["recall"]), 4),
                "f1": round(float(eval_dict["f1"]), 4),
                "iou": round(float(eval_dict["iou"]), 4),
                "accuracy": round(float(eval_dict["accuracy"]), 4),
                "tp": int(eval_dict["tp"]),
                "fp": int(eval_dict["fp"]),
                "fn": int(eval_dict["fn"]),
                "tn": int(eval_dict["tn"]),
                "total_pixels": int(cm.total),
            }
            overlay = create_error_overlay(pred_mask, gt_mask)
            error_overlay_uri = array_to_png_data_uri(overlay)

        return {
            "pair_id": pair_id,
            "model_id": model_id,
            "model": model_id,
            "model_display": self.model_displays.get(model_id, model_id),
            "threshold_used": round(float(tau), 4),
            "threshold": round(float(tau), 4),
            "threshold_source": thresh_source,
            "latency_ms": round(elapsed_ms, 2),
            "image_t1_url": img_t1_url,
            "image_t2_url": img_t2_url,
            "t1_url": img_t1_url,
            "t2_url": img_t2_url,
            "heatmap_data_uri": heatmap_uri,
            "mask_data_uri": mask_uri,
            "error_overlay_data_uri": error_overlay_uri,
            "overlay_data_uri": error_overlay_uri,
            "metrics": metrics_obj,
            "diagnostics": diagnostics_data,
        }


class BenchmarkService:
    """Service serving verified Milestone M11 benchmarking and statistical outputs."""

    @staticmethod
    def get_benchmark_payload() -> Dict[str, Any]:
        """Loads and formats all M11 results and figures."""
        res_dir = ROOT / "experiments" / "results" / "m11"
        fig_dir = ROOT / "experiments" / "figures" / "m11"

        summary_p = res_dir / "benchmark_summary.json"
        if not summary_p.exists():
            raise FileNotFoundError(f"M11 benchmark summary not found at {summary_p}")

        with open(summary_p, "r", encoding="utf-8") as f:
            models_summary = json.load(f)

        def read_json_safe(filename: str) -> Any:
            p = res_dir / filename
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        boot_ci = read_json_safe("statistical_bootstrap_ci.json")
        wilcoxon = read_json_safe("paired_statistical_tests.json")
        ablation = read_json_safe("qat_ablation_analysis.json")

        boot_list = []
        if isinstance(boot_ci, dict):
            for m_k, m_v in boot_ci.items():
                if isinstance(m_v, dict):
                    boot_list.append({
                        "model": m_k,
                        "precision": m_v.get("precision", {}),
                        "recall": m_v.get("recall", {}),
                        "f1": m_v.get("f1", {}),
                        "iou": m_v.get("iou", {}),
                    })

        wilcoxon_list = []
        if isinstance(wilcoxon, list):
            for t in wilcoxon:
                wilcoxon_list.append({
                    "comparison": f"{t.get('model_a')} vs {t.get('model_b')}",
                    "delta_f1_mean": t.get("f1_delta_mean", 0.0),
                    "statistic_f1": t.get("wilcoxon_stat_f1", 0.0),
                    "p_value_f1": t.get("wilcoxon_pval_f1", 1.0),
                    "significant_f1": t.get("significant_at_05_f1", False),
                    **t,
                })

        ablation_list = []
        if isinstance(ablation, dict) and "ablation_conditions" in ablation:
            for cond_name, c_data in ablation.get("ablation_conditions", {}).items():
                m = c_data.get("metrics", {})
                ablation_list.append({
                    "condition": cond_name.replace("Ablation_", "").replace("_", " "),
                    "precision": m.get("precision", 0.0),
                    "recall": m.get("recall", 0.0),
                    "f1": m.get("f1", 0.0),
                    "false_positives": m.get("fp", 0),
                    "fp_reduction_pct": c_data.get("fp_reduction_percent", 0.0),
                    "precision_delta": c_data.get("precision_change_pp", 0.0) / 100.0,
                    "recall_delta": c_data.get("recall_change_pp", 0.0) / 100.0,
                    "exchange_ratio": c_data.get("operational_benefit_ratio", 0.0),
                })

        # Curated catalog of publication research figures across milestones
        catalog = [
            # --- Milestone M13 End-to-End ---
            {
                "name": "m13_cd_pareto_frontier.png",
                "milestone": "m13",
                "category": "Milestone M13 End-to-End",
                "category_key": "m13",
                "title": "M13 Change Detection Pareto Frontier (F1 vs CPU Inference Latency)",
                "description": "Multi-detector Pareto efficiency frontier relating F1 score to CPU inference latency (ms) across all 8 evaluated change detectors on LEVIR-CD.",
            },
            {
                "name": "m13_retrieval_tradeoff.png",
                "milestone": "m13",
                "category": "Milestone M13 End-to-End",
                "category_key": "m13",
                "title": "M13 Semantic Retrieval Trade-Off (Recall vs CPU Latency)",
                "description": "Comparative trade-off between CLIP ViT-B/32 dense semantic retrieval and BM25 sparse lexical search across RSICD test queries.",
            },
            {
                "name": "m13_qat_ablation_waterfall.png",
                "milestone": "m13",
                "category": "Milestone M13 End-to-End",
                "category_key": "m13",
                "title": "M13 QAT-CD Component Ablation Waterfall Analysis",
                "description": "Stepwise contribution of quality attenuation, difference threshold gating, and spatial filtering to overall F1 score and false-positive reduction.",
            },
            {
                "name": "m13_gamma_sensitivity_curve.png",
                "milestone": "m13",
                "category": "Milestone M13 End-to-End",
                "category_key": "m13",
                "title": "M13 Quality Attenuation Parameter Sensitivity Curve",
                "description": "Continuous sweep of attenuation factor gamma in [0.0, 1.0], displaying false-positive pixel count suppression and test F1 preservation.",
            },
            # --- Milestone M11 Benchmark ---
            {
                "name": "precision_recall_tradeoff.png",
                "milestone": "m11",
                "category": "Milestone M11 Benchmark",
                "category_key": "m11",
                "title": "M11 Precision-Recall Frontier with Iso-F1 Contours",
                "description": "Scatter plot of Precision vs Recall across all 8 evaluated change detectors, overlaid with iso-F1 contours on the 128 LEVIR-CD test scenes.",
            },
            {
                "name": "f1_iou_benchmark_bars.png",
                "milestone": "m11",
                "category": "Milestone M11 Benchmark",
                "category_key": "m11",
                "title": "M11 F1 Score & IoU Benchmark with 95% Bootstrap CI",
                "description": "Clustered bar chart of F1 score and IoU with 95% confidence intervals derived from 1,000 bootstrap resamples on the 128 test scenes.",
            },
            {
                "name": "false_positive_reduction_log.png",
                "milestone": "m11",
                "category": "Milestone M11 Benchmark",
                "category_key": "m11",
                "title": "M11 False Positive Pixel Reduction (Log Scale)",
                "description": "Total false-positive pixels across 128 test scenes (134.2M pixels), highlighting dramatic false-alarm suppression in QAT-CD.",
            },
            {
                "name": "cpu_latency_vs_f1.png",
                "milestone": "m11",
                "category": "Milestone M11 Benchmark",
                "category_key": "m11",
                "title": "M11 CPU Latency vs F1 Score Operational Efficiency Frontier",
                "description": "Efficiency frontier relating CPU latency (ms) to F1 score, where bubble area is proportional to model parameter count.",
            },
            {
                "name": "per_scene_f1_distribution.png",
                "milestone": "m11",
                "category": "Milestone M11 Benchmark",
                "category_key": "m11",
                "title": "M11 Per-Scene F1 Distribution (Box Plots)",
                "description": "Box plot showing the variance, median dispersion, and outlier distribution of scene-level F1 scores across 128 evaluation scenes.",
            },
            {
                "name": "qat_ablation_tradeoff.png",
                "milestone": "m11",
                "category": "Milestone M11 Benchmark",
                "category_key": "m11",
                "title": "M11 Quality-Aware Attenuation Signal Ablation Trade-off",
                "description": "Multi-bar trade-off chart illustrating false-positive reduction percentage, precision gain, and recall drop across QAT-CD conditions.",
            },
            {
                "name": "cross_model_robustness.png",
                "milestone": "m11",
                "category": "Milestone M11 Benchmark",
                "category_key": "m11",
                "title": "M11 Cross-Model Environmental Stress Robustness",
                "description": "Line plot tracing relative F1 degradation under illumination shifts, gaussian blur, and geometric misregistration perturbations.",
            },
            # --- Milestone M10 QAT-CD ---
            {
                "name": "detector_comparison_m5_to_m10.png",
                "milestone": "m10",
                "category": "Milestone M10 QAT-CD",
                "category_key": "m10",
                "title": "M10 Detector Benchmark Comparison (M5 to M10)",
                "description": "Progression bar chart tracing F1 and IoU metric advancements from classical baselines to Quality-Aware Temporal CD.",
            },
            {
                "name": "validation_calibration_surface.png",
                "milestone": "m10",
                "category": "Milestone M10 QAT-CD",
                "category_key": "m10",
                "title": "M10 Validation Calibration Surface (Gamma x Tau Grid)",
                "description": "Validation F1 response surface mapping optimal quality attenuation factor gamma and binarization threshold tau.",
            },
            {
                "name": "ablation_f1_comparison.png",
                "milestone": "m10",
                "category": "Milestone M10 QAT-CD",
                "category_key": "m10",
                "title": "M10 Gating Component Ablation Comparison",
                "description": "Test set F1 score comparison across baseline, attenuation-only, gating-only, and full QAT-CD calibrated configurations.",
            },
            # --- Milestone M6 & M5 Baselines ---
            {
                "name": "detector_perturbation_heatmap.png",
                "milestone": "m6",
                "category": "Milestone M6 & M5 Baselines",
                "category_key": "m6-m5",
                "title": "M6 Environmental Perturbation Sensitivity Heatmap",
                "description": "Heatmap matrix documenting detector resilience across environmental stress conditions and severity levels.",
            },
            {
                "name": "f1_vs_severity.png",
                "milestone": "m6",
                "category": "Milestone M6 & M5 Baselines",
                "category_key": "m6-m5",
                "title": "M6 F1 Degradation vs Perturbation Severity",
                "description": "Parametric performance decay trajectories as blur, illumination, and registration noise escalate.",
            },
            {
                "name": "validation_threshold_curves.png",
                "milestone": "m5",
                "category": "Milestone M6 & M5 Baselines",
                "category_key": "m6-m5",
                "title": "M5 Classical Baselines Validation Threshold Calibration",
                "description": "Validation sweep curves across B1 Pixel Diff, B2 SSIM Dissimilarity, and B3 CVA establishing frozen optimal thresholds.",
            },
        ]

        figures = []
        seen_names = set()
        for item in catalog:
            fig_p = ROOT / "experiments" / "figures" / item["milestone"] / item["name"]
            if fig_p.exists() and fig_p.is_file():
                figures.append({
                    "name": item["name"],
                    "url": f"/api/figures/{item['name']}",
                    "title": item["title"],
                    "category": item["category"],
                    "category_key": item["category_key"],
                    "description": item["description"],
                })
                seen_names.add(item["name"])

        # Dynamically append any remaining uncataloged publication PNGs
        for md in ["m13", "m11", "m10", "m6", "m5"]:
            d = ROOT / "experiments" / "figures" / md
            if d.exists():
                for p in sorted(d.glob("*.png")):
                    if p.name not in seen_names and not p.name.startswith(("qualitative_test", "qualitative_comparison")):
                        figures.append({
                            "name": p.name,
                            "url": f"/api/figures/{p.name}",
                            "title": p.stem.replace("_", " ").title(),
                            "category": f"Milestone {md.upper()}",
                            "category_key": md,
                            "description": f"Precomputed research visualization artifact from {md.upper()}.",
                        })
                        seen_names.add(p.name)


        return {
            "models": models_summary,
            "statistical_bootstrap_ci": boot_ci,
            "bootstrap_ci": boot_list,
            "paired_statistical_tests": wilcoxon,
            "wilcoxon_tests": wilcoxon_list,
            "qat_ablation_analysis": ablation,
            "ablation": ablation_list,
            "cpu_performance_benchmark": read_json_safe("cpu_performance_benchmark.json"),
            "threshold_provenance_audit": read_json_safe("threshold_provenance_audit.json"),
            "figures": figures,
        }


# Singleton service instances
retrieval_service = RetrievalService()
change_detection_service = ChangeDetectionService()
benchmark_service = BenchmarkService()
from src.api.llm_service import llm_explanation_service, LLMExplanationService


# ==============================================================================
# PHASE 3C: TEMPORAL UNDERSTANDING SERVICE
# ==============================================================================

class TemporalUnderstandingService:
    """Orchestrates end-to-end temporal satellite image understanding.

    Full pipeline workflow:
    1. Upload & Ingestion: In-memory byte streaming, safety validation, optical RGB normalization.
    2. Phase 3A Co-Registration: ORB keypoint matching + fine ECC alignment with policy gating.
    3. Change Detection: Dynamic patch-based inference with M9 BIT-CD / M10 QAT-CD.
    4. Phase 3A Region Extraction: Connected-component analysis and Top-K ranking.
    5. Phase 3B Optical VARI: Visible Atmospherically Resistant Index greenness delta.
    6. Phase 3B Multimodal Vision: Groq Qwen 3.8 Multimodal analysis over 8-class taxonomy.
    7. Phase 3B Explanation Synthesis: Groq Qwen 27B academic report synthesis.
    """

    def __init__(
        self,
        change_detector_service: Optional[ChangeDetectionService] = None,
        max_upload_size: int = 25 * 1024 * 1024,  # 25 MB
        max_dimension: int = 4096,
    ):
        self.cd_service = change_detector_service or change_detection_service
        self.registrator = ImageCoRegistrator(RegistrationPolicy(allow_multichannel_tiff=True))
        self.extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="pad")
        self.max_upload_size = max_upload_size
        self.max_dimension = max_dimension
        self.vision_analyzer = SemanticChangeAnalyzer(VisionAnalyzerConfig())
        self.explanation_engine = TemporalExplanationEngine(TemporalExplanationConfig())

    def validate_and_decode_upload(
        self,
        file_bytes: bytes,
        filename: Optional[str] = None,
        name: str = "T1",
    ) -> np.ndarray:
        """Validates upload size, extension, decompression safety, and converts to optical RGB uint8 array."""
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Uploaded file {name} is empty.")

        if len(file_bytes) > self.max_upload_size:
            raise ValueError(
                f"Uploaded file {name} exceeds maximum allowed size ({self.max_upload_size / (1024*1024):.1f} MB)."
            )

        if filename:
            clean_fn = Path(filename).name.lower()
            suffix = Path(clean_fn).suffix
            allowed_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            if suffix not in allowed_suffixes:
                raise ValueError(
                    f"Unsupported image file extension '{suffix}' for {name}. Allowed formats: PNG, JPEG, TIFF."
                )

        # Decompression bomb safety check
        Image.MAX_IMAGE_PIXELS = 50_000_000

        try:
            arr = self.registrator.load_and_validate_image(file_bytes, name=name)
        except Exception as e:
            raise ValueError(f"Failed to decode or validate optical image {name}: {str(e)}")

        # Ensure 3-channel optical RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]

        h, w = arr.shape[:2]
        if h > self.max_dimension or w > self.max_dimension:
            raise ValueError(
                f"{name} image dimension ({w}x{h}) exceeds maximum supported limit {self.max_dimension}x{self.max_dimension}."
            )

        return arr

    def analyze_pair(
        self,
        t1_bytes: bytes,
        t2_bytes: bytes,
        filename_t1: Optional[str] = None,
        filename_t2: Optional[str] = None,
        observation_date_t1: Optional[str] = None,
        observation_date_t2: Optional[str] = None,
        gsd_meters: Optional[float] = None,
        description: Optional[str] = None,
        detector_model: str = "M9_BIT_CD",
        max_regions: int = 6,
    ) -> Dict[str, Any]:
        """Executes full end-to-end temporal understanding analysis."""
        req_id = f"temp_{uuid.uuid4().hex[:12]}"

        # 1. Decode and validate uploads
        img_t1 = self.validate_and_decode_upload(t1_bytes, filename=filename_t1, name="T1")
        img_t2 = self.validate_and_decode_upload(t2_bytes, filename=filename_t2, name="T2")

        image_metadata = {
            "width": int(img_t1.shape[1]),
            "height": int(img_t1.shape[0]),
            "channels": 3,
            "observation_date_t1": observation_date_t1,
            "observation_date_t2": observation_date_t2,
            "gsd_meters_per_pixel": float(gsd_meters) if gsd_meters is not None and gsd_meters > 0 else None,
            "description": description,
        }

        # 2. Image Co-Registration
        reg_result = self.registrator.register_pair(img_t1, img_t2)
        diag = reg_result.diagnostics

        if diag.registration_status == RegistrationStatus.REJECTED or reg_result.aligned_t2 is None:
            logger.warning("Co-registration rejected for request %s (confidence: %.3f)", req_id, diag.confidence_score)
            return {
                "request_id": req_id,
                "status": "registration_rejected",
                "image_metadata": image_metadata,
                "registration_diagnostics": diag.to_dict(),
                "detector_metadata": None,
                "change_mask": None,
                "regions": [],
                "vari_observations": None,
                "semantic_findings": [],
                "temporal_explanation": None,
                "warnings": diag.warnings + ["Image pair alignment failed quality policy; registration was rejected."],
                "errors": ["Registration quality rejected: insufficient inliers or excessive reprojection error."],
            }

        img_t1_aligned = img_t1
        img_t2_aligned = reg_result.aligned_t2
        self.last_aligned_t1 = img_t1_aligned
        self.last_aligned_t2 = img_t2_aligned
        h_aligned, w_aligned = img_t1_aligned.shape[:2]

        image_metadata["width"] = w_aligned
        image_metadata["height"] = h_aligned

        # 3. Change Detection Inference
        valid_model = detector_model.strip() if detector_model in ("M9_BIT_CD", "M10_QAT_CD") else "M9_BIT_CD"
        detector, tau = self.cd_service.get_detector(valid_model)

        t_det0 = time.perf_counter()
        patches = self.extractor.extract_from_pair(img_t1_aligned, img_t2_aligned, parent_stem=req_id)
        diff_patches = []
        for p_a, p_b, _, p_meta in patches:
            if valid_model == "M10_QAT_CD":
                d_p = detector.base_detector.compute_difference_map(p_a, p_b)
            else:
                d_p = detector.compute_difference_map(p_a, p_b)
            diff_patches.append((d_p, p_meta))

        base_diff = self.extractor.reconstruct_image(
            diff_patches,
            original_height=h_aligned,
            original_width=w_aligned,
            dtype=np.float32,
        )

        qat_metrics: Optional[Dict[str, Any]] = None
        if valid_model == "M10_QAT_CD":
            diag_qat = detector.compute_quality_diagnostics(img_t1_aligned, img_t2_aligned, base_prob=base_diff)
            full_diff = diag_qat["gated_prob"]
            qat_metrics = {
                "gamma_attenuation": detector.gamma,
                "mean_registration_discrepancy": round(float(diag_qat["reg_map"].mean()), 4),
                "mean_illumination_gradient": round(float(diag_qat["illum_map"].mean()), 4),
                "mean_sharpness_asymmetry": round(float(diag_qat["sharp_map"].mean()), 4),
                "mean_margin_uncertainty": round(float(diag_qat["uncertainty_map"].mean()), 4),
            }
        else:
            full_diff = base_diff

        pred_mask = (full_diff >= tau).astype(np.uint8)
        det_latency_ms = (time.perf_counter() - t_det0) * 1000.0

        ch_pixels = int(np.count_nonzero(pred_mask > 0))
        ch_pct = round(float(ch_pixels / (h_aligned * w_aligned) * 100.0), 4)

        mask_data_uri = array_to_png_data_uri(pred_mask, is_mask=True)
        mask_rep = {
            "format": "png_data_uri",
            "mask_data_uri": mask_data_uri,
            "width": w_aligned,
            "height": h_aligned,
            "total_changed_pixels": ch_pixels,
            "coordinate_frame": "top_left_origin_pixels",
        }

        detector_meta = {
            "model_id": valid_model,
            "model_display": self.cd_service.model_displays.get(valid_model, valid_model),
            "threshold_used": round(float(tau), 4),
            "total_changed_pixels": ch_pixels,
            "total_changed_area_pct": ch_pct,
            "latency_ms": round(det_latency_ms, 2),
            "qat_diagnostics": qat_metrics,
        }

        # 4. Region Extraction
        extractor = RegionExtractor(RegionExtractionConfig(max_regions=max_regions))
        region_result = extractor.extract_regions(
            binary_mask=pred_mask,
            probability_map=full_diff,
            gsd_meters=gsd_meters if gsd_meters and gsd_meters > 0 else None,
        )

        region_items = []
        for r in region_result.regions:
            region_items.append({
                "region_id": r.region_id,
                "bbox": list(r.bbox),
                "context_bbox": list(r.context_bbox),
                "pixel_area": r.pixel_area,
                "raw_pixel_area": r.raw_pixel_area,
                "area_ratio": round(float(r.area_ratio), 6),
                "centroid": [round(float(r.centroid[0]), 2), round(float(r.centroid[1]), 2)],
                "bbox_width": r.bbox_width,
                "bbox_height": r.bbox_height,
                "mean_probability": round(float(r.mean_probability), 4) if r.mean_probability is not None else None,
                "physical_area_sqm": (
                    round(float(r.physical_area_sq_meters), 2)
                    if r.physical_area_sq_meters is not None
                    else None
                ),
            })

        # 5. Semantic Analysis (VARI & Multimodal Vision)
        semantic_findings_raw = self.vision_analyzer.analyze_regions(
            t1_image=img_t1_aligned,
            t2_image=img_t2_aligned,
            regions=region_result.regions,
            binary_mask=region_result.cleaned_mask,
        )

        semantic_findings_list = [f.to_dict() for f in semantic_findings_raw]
        vari_obs = [f.vari_evidence.to_dict() for f in semantic_findings_raw if f.vari_evidence]

        # 6. Explanation Synthesis
        report = self.explanation_engine.generate_explanation(
            detector_telemetry=detector_meta,
            regions=region_result.regions,
            semantic_findings=semantic_findings_raw,
            registration_diagnostics=diag,
            metadata={
                "gsd_meters": gsd_meters if gsd_meters and gsd_meters > 0 else None,
                "observation_date_t1": observation_date_t1,
                "observation_date_t2": observation_date_t2,
                "description": description,
            },
        )

        return {
            "request_id": req_id,
            "status": "success",
            "image_metadata": image_metadata,
            "registration_diagnostics": diag.to_dict(),
            "detector_metadata": detector_meta,
            "change_mask": mask_rep,
            "regions": region_items,
            "vari_observations": vari_obs if vari_obs else None,
            "semantic_findings": semantic_findings_list,
            "temporal_explanation": report.to_dict(),
            "warnings": diag.warnings,
            "errors": [],
        }


temporal_understanding_service = TemporalUnderstandingService()

