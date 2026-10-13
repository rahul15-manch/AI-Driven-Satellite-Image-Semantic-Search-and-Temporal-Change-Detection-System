# Milestone 12: Streamlit Parity & Verification Report

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Milestone:** M12 — Web Interface Parity Implementation & Verification  
**Date:** October 13, 2026  
**Auditor / Engineer:** Senior Research & QA Systems Engineer  
**Status:** **COMPLETE** (277/277 tests passing, 0 regressions)

---

## 1. Executive Summary

Following the forensic Milestone 12 audit, this report documents the complete implementation and empirical verification of **Streamlit Parity**. The existing Streamlit dashboard (`app.py`, `pages/`) now provides full access to all research artifacts and capabilities established across Milestones M1 through M11, with 100% feature parity with the FastAPI frontend, zero degradation to CPU constraints, and zero regression across the existing test suite.

Key milestones delivered in this implementation:
1. **P0 BM25 Retrieval Defect Resolved:** Repaired the search failure in [`pages/retrieval.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/retrieval.py) where calling `rank_gallery(query, top_k=top_k)` resulted in `TypeError: BM25Retriever.rank_gallery() got an unexpected keyword argument 'top_k'`. Implemented the supported query-tokenization and score-sorting pipeline from [`src/api/service.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/service.py).
2. **Method-Aware Score Interpretation:** Updated [`ui/image_viewer.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/ui/image_viewer.py) and [`pages/retrieval.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/retrieval.py) to render explicit, method-aware score labels (e.g., `Cosine Similarity: 0.8520`, `BM25 Score: 12.3400`), avoiding confusion and never misrepresenting BM25 scores as percentages.
3. **QAT-CD Gating Configuration Control:** Implemented an interactive attenuation selector in [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py) exposing Calibrated ($\gamma = 0.0$) versus Exploratory Active Gating ($\gamma = 0.30$) with verified runtime propagation into model inference.
4. **Manual Decision Threshold Override:** Added an optional slider $\tau \in [0.0, 1.0]$ in [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py) with real-time feedback distinguishing user overrides from validation-calibrated frozen thresholds $\tau^*$.
5. **Milestone M11 Research Evidence Integration:** Extended [`utils/result_loader.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/utils/result_loader.py) and added 5 research tabs to [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py) presenting the full comparative benchmark, scene-level bootstrap 95% confidence intervals, paired Wilcoxon signed-rank hypothesis tests, QAT-CD ablation analysis, and 7 publication-grade research figures.
6. **Mandatory Scientific Disclosures:** Surfaced all three required scientific disclosures (Test-Set Reuse, Exploratory Active Gating trade-offs, and SNUNet-CD False Alarms) in both [`pages/overview.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/overview.py) and [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py).
7. **Complete Regression Verification:** Created 20 new automated parity tests in [`tests/test_m12_streamlit_parity.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/tests/test_m12_streamlit_parity.py) (20/20 passing) and executed the full repository test suite (**277/277 passed in 50.54s**).

---

## 2. Defects Fixed

### 2.1 P0 BM25 Retrieval Keyword Argument Crash

- **Defect Symptom:** Selecting "BM25 (Leave-One-Caption-Out)" on [`pages/retrieval.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/retrieval.py) crashed with:
  ```
  TypeError: BM25Retriever.rank_gallery() got an unexpected keyword argument 'top_k'
  ```
- **Root Cause:** In [`src/semantic_search/bm25.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/semantic_search/bm25.py), `BM25Retriever.rank_gallery(query_text, target_image_id=None, query_caption_idx=None)` accepts only three arguments and returns an unranked or fully ranked list of string IDs without slicing top-K. Passing `top_k=top_k` was invalid.
- **Resolution:** Replaced the invocation with the supported, verified scoring interface:
  ```python
  q_tokens = tokenize(query_input)
  scores_arr = bm25_retriever.score_query_combined(q_tokens)
  ranked_indices = np.argsort(scores_arr)[::-1][:top_k]
  ```
  Scores and image identifiers are mapped 1-to-1 with metadata, properly preserving descending rank and top-K bounds.
- **Verification:** Verified via `test_bm25_rank_gallery_unsupported_kwarg_reproduced_and_isolated`, `test_bm25_supported_combined_scoring_and_top_k`, and `test_retrieval_page_bm25_apptest`.

---

## 3. Files Modified

| File | Type | Changes Made |
| :--- | :--- | :--- |
| [`pages/retrieval.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/retrieval.py) | Python (Page) | Fixed BM25 scoring call, added top-K argsort slice, injected `score_type` into result dictionaries. |
| [`ui/image_viewer.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/ui/image_viewer.py) | Python (UI) | Updated `render_retrieval_card_grid` to format method-aware score labels (`Cosine Similarity: 0.8520` or `BM25 Score: 12.3400`). |
| [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py) | Python (Page) | Added QAT-CD $\gamma$ selectbox ($\gamma = 0.0$ vs $\gamma = 0.30$), manual threshold override $\tau \in [0.0, 1.0]$, M11 5-tab research evidence section, and mandatory disclosures expander. |
| [`utils/result_loader.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/utils/result_loader.py) | Python (Util) | Added loaders: `load_m11_summary()`, `load_m11_bootstrap_ci()`, `load_m11_paired_tests()`, `load_m11_ablation()`, and `get_m11_figures()`. |
| [`app.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/app.py) | Python (Entry) | Updated application title and sidebar telemetry to "Milestones M1–M12 Verified" while strictly complying with architectural boundary governance. |
| [`pages/overview.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/overview.py) | Python (Page) | Updated research pipeline diagram (M1–M12), authoritative milestone matrix, and styled callout for the 3 mandatory research disclosures. |
| [`ui/components.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/ui/components.py) | Python (UI) | Updated milestone badge text default to "Milestones M1–M12 Implemented". |
| [`tests/test_m12_streamlit_parity.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/tests/test_m12_streamlit_parity.py) | Python (Test) | New test suite covering BM25 ranking, score labels, QAT-CD $\gamma$, threshold overrides, M11 loaders, disclosures, and AppTest runs. |

---

## 4. Features Newly Exposed in Streamlit

### 4.1 Method-Aware Score Labels in Image Viewer
Retrieval results in the Streamlit grid now display caller-provided score semantics:
- **CLIP ViT-B/32 & Prompt Ensemble:** `Cosine Similarity: 0.8520`
- **BM25 LOCO & Metadata:** `BM25 Score: 12.3400`
- **Generic Fallback:** `Score: 0.5000`
BM25 scores are strictly kept as unbounded real numbers and never presented as percentages.

### 4.2 QAT-CD Interactive Configuration & Gating
When selecting "M10: QAT-CD (Quality-Aware Temporal Change Detection)":
- **Calibrated ($\gamma = 0.0$):** Operates at the validation-calibrated operating point for optimal unweighted F1 (0.5927).
- **Exploratory Active Gating ($\gamma = 0.30$):** Attenuates change probabilities using registration, illumination, and blur discrepancy signals.
- **Verified Propagation:** Setting `m10_det.gamma = selected_gamma` directly controls the gating formula $P_{\text{QAT}} = P_0 \cdot (1 - \gamma \cdot U_{\text{det}} \cdot Q_{\text{defect}})$. Confirmed via unit tests that $\gamma = 0.30$ dynamically alters inference masks.
- **Exploratory Callout:** Explicitly discloses that active gating reduces false positives by ~29.19% relative to M9 BIT-CD, but sacrifices recall and unweighted F1.

### 4.3 Manual Decision Threshold Override
- Available for all 7 detectors (classical B1–B3, M7 FC-Siam, M8 SNUNet, M9 BIT-CD, M10 QAT-CD).
- Slider range: $[0.0, 1.0]$, step: $0.01$, default: frozen validation threshold $\tau^*$.
- Applied directly to the continuous difference/probability map ($P \ge \tau$) prior to binary evaluation.
- Displays a clear visual warning distinguishing user overrides from validation-calibrated benchmarks without mutating saved artifacts.

### 4.4 Comprehensive Milestone M11 Research Evidence
Organized into 5 dedicated tabs in [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py):
1. **Benchmark Comparison:** Dynamic table of all 7 evaluated models on the 128 LEVIR-CD test pairs (134.2M pixels), displaying Precision, Recall, F1, IoU, Accuracy, CPU Latency, and Peak RAM alongside a grouped comparison bar chart.
2. **Bootstrap Confidence Intervals:** Table of empirical 95% confidence intervals from $B = 1,000$ scene-level resamples across the 128 test scenes.
3. **Paired Statistical Tests:** Exact paired Wilcoxon signed-rank tests ($N = 128$) reporting mean $\Delta \text{F1}$, test statistic $W$, exact p-values (e.g. $9.29 \times 10^{-20}$), and significance interpretation at $\alpha = 0.05$.
4. **QAT-CD Ablation Analysis:** Detailed breakdown of all 6 ablation conditions (`QAT_CD_Full`, `Ablation_NoQuality`, `Ablation_ActiveAll`, `Ablation_RegOnly`, `Ablation_IllumOnly`, `Ablation_SharpOnly`) displaying FP reduction pixels, FP reduction %, TP loss, and Operational Benefit Ratio (1.71x).
5. **Publication Figures Gallery:** Interactive gallery with high-resolution previews and academic captions for all 7 publication figures:
   - `f1_iou_benchmark_bars.png`
   - `cpu_latency_vs_f1.png`
   - `precision_recall_tradeoff.png`
   - `false_positive_reduction_log.png`
   - `per_scene_f1_distribution.png`
   - `qat_ablation_tradeoff.png`
   - `cross_model_robustness.png`

### 4.5 Mandatory Scientific Disclosures
Visible in [`pages/overview.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/overview.py) and in the benchmark header of [`pages/change_detection.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py):
1. **Test-Set Reuse Disclosure:** Notes that the LEVIR-CD test split was referenced during earlier milestones; reported metrics reflect standard reproducible benchmarking rather than untouched out-of-distribution evaluation.
2. **Exploratory Active Gating Disclosure:** Notes that active gating ($\gamma = 0.30$) eliminates 573,553 false-positive pixels (-29.19%) at the cost of 336,213 true-positive pixels (-9.07%), trading recall (49.30%) and unweighted F1 (0.5812) for precision (70.78%).
3. **SNUNet-CD False Alarms Disclosure:** Records that SNUNet-CD generated **53,485,389 false-positive pixels** (a 39.85% false alarm rate) under standardized evaluation.

---

## 5. Verification & Test Execution Evidence

### 5.1 Targeted Parity Test Results (`tests/test_m12_streamlit_parity.py`)
```
tests/test_m12_streamlit_parity.py::TestBM25RetrievalParity::test_bm25_rank_gallery_unsupported_kwarg_reproduced_and_isolated PASSED [  5%]
tests/test_m12_streamlit_parity.py::TestBM25RetrievalParity::test_bm25_supported_combined_scoring_and_top_k PASSED [ 10%]
tests/test_m12_streamlit_parity.py::TestBM25RetrievalParity::test_bm25_empty_query_and_empty_gallery PASSED [ 15%]
tests/test_m12_streamlit_parity.py::TestBM25RetrievalParity::test_retrieval_page_bm25_apptest PASSED [ 20%]
tests/test_m12_streamlit_parity.py::TestMethodAwareScoreLabels::test_score_label_distinction PASSED [ 25%]
tests/test_m12_streamlit_parity.py::TestQATCDConfigurationAndThresholdOverride::test_qat_cd_gamma_propagation_changes_output PASSED [ 30%]
tests/test_m12_streamlit_parity.py::TestQATCDConfigurationAndThresholdOverride::test_frozen_thresholds_remain_unmodified PASSED [ 35%]
tests/test_m12_streamlit_parity.py::TestQATCDConfigurationAndThresholdOverride::test_manual_threshold_override_logic PASSED [ 40%]
tests/test_m12_streamlit_parity.py::TestM11ArtifactLoaders::test_load_m11_summary PASSED [ 45%]
tests/test_m12_streamlit_parity.py::TestM11ArtifactLoaders::test_load_m11_bootstrap_ci PASSED [ 50%]
tests/test_m12_streamlit_parity.py::TestM11ArtifactLoaders::test_load_m11_paired_tests PASSED [ 55%]
tests/test_m12_streamlit_parity.py::TestM11ArtifactLoaders::test_load_m11_ablation PASSED [ 60%]
tests/test_m12_streamlit_parity.py::TestM11ArtifactLoaders::test_get_m11_figures PASSED [ 65%]
tests/test_m12_streamlit_parity.py::TestM11ArtifactLoaders::test_missing_artifact_graceful_handling PASSED [ 70%]
tests/test_m12_streamlit_parity.py::TestScientificDisclosures::test_disclosures_in_overview PASSED [ 75%]
tests/test_m12_streamlit_parity.py::TestScientificDisclosures::test_disclosures_in_change_detection PASSED [ 80%]
tests/test_m12_streamlit_parity.py::TestStreamlitAppTestIntegration::test_app_main PASSED [ 85%]
tests/test_m12_streamlit_parity.py::TestStreamlitAppTestIntegration::test_overview_page PASSED [ 90%]
tests/test_m12_streamlit_parity.py::TestStreamlitAppTestIntegration::test_change_detection_page PASSED [ 95%]
tests/test_m12_streamlit_parity.py::TestStreamlitAppTestIntegration::test_false_alarm_page PASSED [100%]

============================== 20 passed in 8.48s ==============================
```

### 5.2 UI Smoke & Boundary Governance Suite (`tests/test_ui_smoke.py`)
```
============================= 28 passed in 14.08s ==============================
- test_no_unimplemented_models_in_pages: PASSED
- test_all_pages_apptest: PASSED
- test_retrieval_page: PASSED
- test_change_detection_page: PASSED
```

### 5.3 M12 FastAPI & Web Frontend Suite (`tests/test_m12_api.py`, `tests/test_m12_frontend.py`)
```
======================== 21 passed, 1 warning in 5.91s =========================
- test_health_endpoint: PASSED
- test_search_clip_success: PASSED
- test_search_bm25_success: PASSED
- test_change_detection_classical_b1: PASSED
- test_change_detection_threshold_override: PASSED
- test_benchmark_endpoint: PASSED
- test_frontend_directory_structure: PASSED
- test_javascript_modules_export_syntax: PASSED
```

### 5.4 Full Repository Regression Suite
```
======================= 277 passed, 1 warning in 50.54s ========================
```
- **Total Tests:** 277
- **Passing:** 277 (100%)
- **Failures:** 0
- **Regressions:** 0
- **Duration:** 50.54s on Apple Silicon CPU

---

## 6. Dual-Interface Feature Parity Matrix

| Feature / Capability | FastAPI Web Frontend (`frontend/`) | Streamlit Dashboard (`app.py`, `pages/`) | Parity Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **CLIP ViT-B/32 Search** | Yes (`POST /api/search`) | Yes (`pages/retrieval.py`) | **PARITY** | Precomputed embeddings, FAISS IndexFlatIP |
| **BM25 Lexical Retrieval** | Yes (`POST /api/search`) | Yes (`pages/retrieval.py`) | **PARITY** | Fixed P0 TypeError; top-K ranking aligned |
| **Method-Aware Score Labels** | Yes (`Cosine Similarity` / `BM25 Score`) | Yes (`ui/image_viewer.py`) | **PARITY** | Distinguishes cosine [0,1] from BM25 [0, +inf) |
| **Classical CD (B1, B2, B3)** | Yes (`POST /api/change-detect`) | Yes (`pages/change_detection.py`) | **PARITY** | Patch extraction & 1024x1024 mosaic |
| **Learned CD (M7, M8, M9)** | Yes (`POST /api/change-detect`) | Yes (`pages/change_detection.py`) | **PARITY** | FC-Siam-diff, SNUNet-CD, BIT-CD |
| **QAT-CD Gating Control** | Yes (`gamma` parameter) | Yes ($\gamma=0.0$ vs $\gamma=0.30$) | **PARITY** | Dynamic inference attenuation verified |
| **Decision Threshold Override** | Yes (`threshold` float) | Yes (slider $\tau \in [0.0, 1.0]$) | **PARITY** | Applied at continuous difference map |
| **Interactive Error Overlay** | Yes (TP/FP/FN/TN color) | Yes (TP/FP/FN/TN overlay) | **PARITY** | Identical RGB color mapping |
| **M11 Benchmark Summary** | Yes (`GET /api/benchmarks`) | Yes (M11 Tab 1 + Chart) | **PARITY** | Direct load from `benchmark_summary.json` |
| **Bootstrap Confidence Intervals** | Readme / Docs | Yes (M11 Tab 2) | **PARITY+** | 95% CIs across 1,000 bootstrap resamples |
| **Paired Wilcoxon Tests** | Readme / Docs | Yes (M11 Tab 3) | **PARITY+** | Non-parametric hypothesis tests ($N=128$) |
| **QAT-CD Ablation Analysis** | Readme / Docs | Yes (M11 Tab 4) | **PARITY+** | Full breakdown of quality signals & trade-offs |
| **Publication Figures Gallery** | Static files | Yes (M11 Tab 5) | **PARITY+** | Previews and captions for all 7 M11 figures |
| **Scientific Disclosures** | In HTML Methodology | In Overview & CD Header | **PARITY** | Visible callouts on test-set, gating, & SNUNet |

---

## 7. Remaining Architectural Differences

While user-facing research features are now in full parity, the underlying technical delivery mechanisms differ intentionally by design:
1. **Serving Architecture:**
   - **FastAPI Frontend:** Decoupled single-page application (HTML5/CSS3/Vanilla JS) backed by REST API JSON endpoints (`src/api/service.py`).
   - **Streamlit Application:** Python server-rendered reactive multi-page dashboard utilizing Streamlit component state.
2. **State Management:**
   - FastAPI utilizes client-side `fetch` calls and DOM manipulation.
   - Streamlit utilizes `@st.cache_resource` for model weights and server-side session state for widget parameters.
3. **Hardware Footprint:**
   - Both interfaces strictly respect CPU-only execution constraints (`torch.set_num_threads(4)`, RSS $\le$ 8 GB). Live process RSS telemetry is actively displayed in the Streamlit sidebar (typically 300–600 MB RSS).

---

## 8. Final Status

- **Status:** **COMPLETE**
- **Blockers:** None
- **Unresolved Warnings:** Single third-party deprecation warning from Starlette test client (`httpx` import), harmless to execution.
- **Repository Health:** 277 passed out of 277 tests. Zero regressions.
