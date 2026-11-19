# Evidence Inventory — Stage 1 Application Audit

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Document Location:** `docs/stage1_application_audit/evidence_inventory.md`  
**Related Workbook:** [`docs/stage1_application_audit/Stage1_Application_Audit_Completed.xlsx`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/docs/stage1_application_audit/Stage1_Application_Audit_Completed.xlsx)  
**Date:** November 13, 2026  

---

## 1. Overview

This evidence inventory provides an exhaustive, clickable index linking every audit item, technical capability, mathematical formulation, model checkpoint, and experimental result in the Stage 1 Application Audit to its exact source code file, function, class, configuration key, or data artifact within the repository.

---

## 2. Source Code & Algorithmic Modules

### A. Semantic Search & Cross-Modal Retrieval
- **CLIP ViT-B/32 Dual Encoder:**
  - File: [`src/semantic_search/clip_model.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/semantic_search/clip_model.py)
  - Class: `CLIPFeatureExtractor`
  - Methods: `encode_text(query: str)`, `encode_image(image: np.ndarray)`
  - Key Logic: Forces PyTorch CPU execution (`torch.device('cpu')`), 512-D float32 embeddings, strict L2-normalization.
- **Prompt Ensembling (5 RS Templates):**
  - File: [`src/semantic_search/prompt_ensembler.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/semantic_search/prompt_ensembler.py)
  - Class: `PromptEnsembler`
  - Method: `encode_ensemble(query: str)`
  - Frozen Templates:
    1. `"{query}"`
    2. `"a satellite image of {query}"`
    3. `"a remote sensing image of {query}"`
    4. `"an overhead image showing {query}"`
    5. `"an aerial photograph of {query}"`
- **FAISS Exact Cosine Search:**
  - File: [`src/semantic_search/faiss_index.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/semantic_search/faiss_index.py)
  - Class: `FAISSIndex`
  - Underlying Index: `faiss.IndexFlatIP(512)`
  - Logic: Inner product on L2-normalized vectors mathematically equals cosine similarity.
- **BM25 Lexical Baselines:**
  - File: [`src/semantic_search/bm25.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/semantic_search/bm25.py)
  - Class: `BM25Retriever`
  - Methods: `search(query, top_k)`
  - Protocols: Leave-One-Caption-Out (`bm25_loco`) and Category Metadata (`bm25_metadata`).

---

### B. Bi-Temporal Change Detection Models
- **Abstract Base Detector:**
  - File: [`src/change_detection/base.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/base.py)
  - Class: `BaseChangeDetector`
  - Contract: `predict(t1: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
- **B1 Pixel Difference:**
  - File: [`src/change_detection/pixel_diff.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/pixel_diff.py)
  - Class: `PixelDiffDetector`
  - Threshold: $	au^* = 0.4100$
- **B2 SSIM Dissimilarity:**
  - File: [`src/change_detection/ssim_detector.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/ssim_detector.py)
  - Class: `SSIMDetector`
  - Formulation: $D(x, y) = 1.0 - 	ext{SSIM}(x, y) \in [0.0, 1.0]$
  - Threshold: $	au^* = 0.9000$
- **B3 Change Vector Analysis (CVA):**
  - File: [`src/change_detection/cva.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/cva.py)
  - Class: `CVADetector`
  - Formulation: Normalized Euclidean magnitude of spectral difference vector
  - Threshold: $	au^* = 0.4050$
- **M7 FC-Siam-diff:**
  - File: [`src/change_detection/fc_siam_diff.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/fc_siam_diff.py)
  - Class: `FCSiamDiffDetector`
  - Parameters: 1,472,130 (1.47M)
  - Checkpoint: `experiments/checkpoints/m7/fc_siam_diff_best.pt` (5.94 MB)
  - Threshold: $	au^* = 0.5500$
- **M8 SNUNet-CD:**
  - File: [`src/change_detection/snunet.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/snunet.py)
  - Class: `SNUNetDetector`
  - Parameters: 563,202 (563k)
  - Checkpoint: `experiments/checkpoints/m8/snunet_best.pt` (2.31 MB)
  - Threshold: $	au^* = 0.5000$
- **M9 BIT-CD (Transformer):**
  - File: [`src/change_detection/bit_cd.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/bit_cd.py)
  - Class: `BITChangeDetector`
  - Parameters: 552,066 (552k)
  - Checkpoint: `experiments/checkpoints/m9/bit_cd_best.pt` (2.26 MB)
  - Threshold: $	au^* = 0.4500$
- **M10 QAT-CD (Quality-Aware Gating):**
  - File: [`src/change_detection/qat_cd.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/qat_cd.py)
  - Class: `QualityAwareChangeDetector`
  - Gating Formula: $P_{	ext{QAT}} = P_0 \cdot (1 - \gamma \cdot U \cdot Q_{	ext{defect}})$
  - Operating Points: $\gamma = 0.0$ (calibrated baseline), $\gamma = 0.30$ (active gating)

---

### C. Real-World Temporal Satellite Understanding Modules
- **Two-Stage Co-Registration Engine:**
  - File: [`src/temporal_understanding/registration.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/registration.py)
  - Class: `ImageCoRegistrator`
  - Policy: `RegistrationPolicy`
    - ORB Max Features: 5,000
    - Lowe's Ratio Test: 0.80
    - Min Inliers: 10
    - Max Reprojection Error: 4.0 pixels
    - Min Spatial Coverage: 0.05 (5%)
    - ECC Correlation Threshold: 0.50
    - Rejection Confidence: $< 0.40 ightarrow 	ext{REJECTED}$
- **Region Extraction & Geometric Clustering:**
  - File: [`src/temporal_understanding/region_extractor.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/region_extractor.py)
  - Class: `RegionExtractor`
  - Configuration: `RegionExtractionConfig`
    - Morphology Kernel: $3 	imes 3$ Close followed by Open
    - Connectivity: 8-way
    - Min Area: 150 pixels ($0.02\%$ canvas)
    - Context Margin: $10\%$ padding
    - Bounding Box Format: `[ymin, xmin, ymax, xmax]` in top-left pixel space
- **Visible Atmospherically Resistant Index (VARI):**
  - File: [`src/temporal_understanding/semantic_analyzer.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/semantic_analyzer.py)
  - Function: `compute_vari_map(image: np.ndarray)`
  - Formula: $	ext{VARI} = rac{	ext{Green} - 	ext{Red}}{	ext{Green} + 	ext{Red} - 	ext{Blue}}$ clamped to $[-1.0, 1.0]$
  - Verbatim Caveat: Lines 139–141 (warns VARI lacks NIR and is not multispectral NDVI).
- **8-Class Change Taxonomy & Multimodal Vision Analysis:**
  - File: [`src/temporal_understanding/semantic_analyzer.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/semantic_analyzer.py)
  - Class: `SemanticChangeAnalyzer`
  - Method: `_call_groq_vision(composite_crop_b64, prompt)`
  - Model: Groq `qwen/qwen3.8-27b`
  - Minimum Crop Dimension: $32 	imes 32$ pixels
- **Dual-Stage Report Synthesis Explainer:**
  - File: [`src/temporal_understanding/explanation_engine.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/explanation_engine.py)
  - Class: `TemporalExplanationEngine`
  - Method: `generate_explanation(...) -> TemporalUnderstandingReport`
  - Model: Groq `qwen/qwen3.8-27b` (Text API)
  - Anti-Hallucination Guard: Strictly receives structured telemetry; zero raw pixels.

---

### D. API Serving & Web User Interfaces
- **FastAPI Endpoints:**
  - Files: [`src/api/routes.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/routes.py), [`src/api/app.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/app.py)
  - Endpoints:
    - `GET /api/health`: System health and loaded checkpoints
    - `POST /api/v1/search`: Cross-modal semantic retrieval
    - `POST /api/v1/benchmark/evaluate`: Benchmark change detection
    - `POST /api/v1/temporal/analyze`: End-to-end temporal understanding
    - `POST /api/v1/explain/change_detection`: LLM explanation generation
- **Streamlit Research Application:**
  - Entry: [`app.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/app.py)
  - Pages:
    - [`pages/retrieval.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/retrieval.py): Semantic Search UI
    - [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py): Dual-mode Change Detection UI
    - [`pages/research_results.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/research_results.py): Interactive Research Results UI
    - [`pages/methodology.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/methodology.py): Methodology, Limitations & Profiling UI
- **Web SPA:**
  - HTML Entry: [`frontend/index.html`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/frontend/index.html)
  - Logic: `frontend/js/app.js`, `api.js`, `retrieval.js`, `change_detection.js`, `results.js`
  - Styles: `frontend/css/styles.css`

---

## 3. Preserved Experimental Results Artifacts

| Artifact Path | Format | Description & Contents |
|---|---|---|
| `experiments/results/m11/benchmark_summary.json` | JSON | Complete metrics (P, R, F1, IoU, Acc, FP, Latency, RSS) for all 7 detectors across 128 scenes. |
| `experiments/results/m11/benchmark_summary.csv` | CSV | Tabular CSV export of comparative benchmark metrics. |
| `experiments/results/m11/statistical_bootstrap_ci.json` | JSON | Empirical non-parametric bootstrap ($B=1,000$) 95% confidence intervals for P, R, F1, IoU. |
| `experiments/results/m11/paired_statistical_tests.json` | JSON | Paired Wilcoxon signed-rank test statistics ($W$) and p-values across all 128 test scenes. |
| `experiments/results/m11/qat_ablation_analysis.json` | JSON | 6-condition multi-signal quality gating ablation metrics. |
| `experiments/results/m11/cpu_performance_benchmark.json` | JSON | CPU latency (mean, P50, P95), memory profiling, and RSS peak metrics on Apple Silicon. |
| `experiments/results/m11/per_scene_all_models.csv` | CSV | Per-scene confusion matrices and evaluation metrics for all 128 scenes ($102,256	ext{ bytes}$). |
| `experiments/results/m13/semantic_retrieval_ablation.json` | JSON | Comparative retrieval metrics for BM25 LOCO, BM25 Metadata, CLIP ViT-B/32, and Prompt Ensemble. |
| `experiments/results/m13/qat_cd_controlled_ablation.json` | JSON | Cross-system controlled ablation evaluating $\gamma$ sensitivity and FP exchange ratios. |
| `experiments/results/m5/m5_thresholds.json` | JSON | Validation calibration threshold optimization curves for classical baselines B1, B2, B3. |

---

## 4. Research Figures Artifacts

The repository contains 46 publication-ready figures located in `experiments/figures/`:
- `experiments/figures/m11/f1_iou_benchmark_bars.png`: Comparative F1 and IoU performance across all 7 detectors.
- `experiments/figures/m11/precision_recall_tradeoff.png`: Precision vs Recall cross-detector scatter plot.
- `experiments/figures/m11/qat_ablation_master_comparison.png`: Comprehensive QAT-CD gating ablation bars.
- `experiments/figures/m11/false_positive_reduction_log.png`: Logarithmic false-positive pixel reduction graph.
- `experiments/figures/m11/cpu_latency_vs_f1.png`: CPU inference latency vs F1 score trade-off curve.
- `experiments/figures/m11/cpu_latency_graph.png`: Detailed latency distributions across detectors.
- `experiments/figures/m11/memory_profiling_graph.png`: Peak RAM consumption and memory footprints.
- `experiments/figures/m11/registration_diagnostics_master.png`: ORB feature matches, inliers, and ECC correlation curves.
- `experiments/figures/m11/bounding_box_localization_master.png`: Spatial cluster localization and bounding-box extractions.
- `experiments/figures/m11/region_measurements_vari_master.png`: Regional VARI greenness distributions.
- `experiments/figures/m11/vari_vegetation_example.png`: Detailed case study of vegetation clearing and regrowth.
- `experiments/figures/m11/all_change_detection_comparison.png`: 7-detector visual side-by-side comparison on test scenes.
- `experiments/figures/m13/retrieval_model_comparison.png`: Bar chart comparison of retrieval protocols.
- `experiments/figures/m13/m13_retrieval_tradeoff.png`: Retrieval latency vs Recall@1 trade-off frontier.
- `experiments/figures/m13/m13_qat_ablation_waterfall.png`: Stepwise contribution of registration, illumination, and blur gating.
- `experiments/figures/m6/detector_perturbation_heatmap.png`: Perturbation sensitivity heatmap under misregistration, blur, and shadows.
- `experiments/figures/phase3e/registration_stress_test_examples.png`: Real-world registration stress testing under extreme geometric shears.

---

## 5. Automated Regression Test Suite

The automated test suite in `tests/` contains **419 collected pytest tests** verifying regression boundaries:
- `tests/test_clip_model.py`: CLIP encoder initialization, CPU device forcing, and embedding shapes.
- `tests/test_prompt_ensembler.py`: 5 domain prompt templates, string substitution, and vector averaging.
- `tests/test_faiss_index.py`: IndexFlatIP construction, normalization, and top-K search.
- `tests/test_bm25.py`: Okapi BM25 tokenization, inverted indexing, and score ordering.
- `tests/test_classical_detectors.py`: B1, B2, B3 inference, thresholding, and continuous maps.
- `tests/test_deep_detectors.py`: M7, M8, M9 deep model forward passes and weight loading.
- `tests/test_qat_cd.py`: Gating formula, defect index fusion, and margin uncertainty computation.
- `tests/test_temporal_registration.py`: Two-stage ORB+ECC alignment, inlier thresholds, and rejection guards.
- `tests/test_temporal_region_extractor.py`: Morphological operations, 8-way connectivity, and context padding.
- `tests/test_temporal_semantic_analyzer.py`: VARI zero-division guards, RGB channel validation, and verbatim caveats.
- `tests/test_temporal_explanation_engine.py`: Groq Qwen payload formation, schema parsing, and fallback handling.
- `tests/test_temporal_api.py`: FastAPI end-to-end integration across valid/invalid uploads, missing GSD, and latency.
- `tests/test_ui_smoke.py`: Streamlit multi-page startup, result loading, and visual artifact integrity.
