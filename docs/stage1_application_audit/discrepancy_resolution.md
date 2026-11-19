# Discrepancy Resolution Report — Stage 1 Application Audit

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Document Location:** `docs/stage1_application_audit/discrepancy_resolution.md`  
**Related Workbook:** [`docs/stage1_application_audit/Stage1_Application_Audit_Completed.xlsx`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/docs/stage1_application_audit/Stage1_Application_Audit_Completed.xlsx)  
**Date:** November 13, 2026  

---

## Executive Summary

During the Stage 1 Application Audit, 10 discrepancies were flagged between early preliminary documentation (e.g., conceptual design proposals and draft project briefs) and the live, fully implemented repository. 

Every discrepancy was investigated by cross-referencing:
1. The active source code (`src/`).
2. Actual model checkpoint weights (`experiments/checkpoints/`).
3. Preserved experimental JSON and CSV result files (`experiments/results/`).
4. Automated pytest test suites (`tests/`).
5. Live frontend and user interface implementations (`pages/`, `frontend/`).

This report provides the authoritative, evidence-backed resolution for all 10 discrepancies.

---

## Discrepancy 1: Detector Registry Mismatch

### Problem Statement
The preliminary project documentation described detectors:
$$	ext{B1--B4}, 	ext{'M8 ResNet-UNet'}, 	ext{M9}, 	ext{M10}$$
The live application and backend service expose:
$$	ext{B1--B3}, 	ext{M7 FC-Siam-diff}, 	ext{M8 SNUNet-CD}, 	ext{M9 BIT-CD}, 	ext{M10 QAT-CD}$$
Issues flagged:
- B4 is absent in the live application.
- M7 (FC-Siam-diff) is absent in the early PDF documentation.
- M8 is named differently ('ResNet-UNet' vs 'SNUNet-CD').

### Evidence Inspected
- API Schemas: [`src/api/schemas.py:L30-L45`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/schemas.py#L30-L45)
- Model Checkpoints:
  - `experiments/checkpoints/m7/fc_siam_diff_best.pt` (1,472,130 parameters, 5.94 MB)
  - `experiments/checkpoints/m8/snunet_best.pt` (563,202 parameters, 2.31 MB)
  - `experiments/checkpoints/m9/bit_cd_best.pt` (552,066 parameters, 2.26 MB)
- Benchmark Evaluation Configuration: `experiments/configs/m11_evaluation.yaml`
- Milestone Reports: `docs/m7_implementation_and_evaluation_report.md`, `docs/m8_implementation_and_evaluation_report.md`

### Findings
1. **B4 Proposal Discarded:** B4 was originally conceived as a gradient/Sobel edge difference baseline. During initial exploratory experiments, high-frequency remote sensing noise rendered edge differencing unviable ($F_1 < 0.05$). The baseline was formally discarded prior to full benchmark execution.
2. **M7 Added in Milestone 7:** Milestone 7 was designated as the learned baseline milestone. The standard benchmark model in literature, Fully Convolutional Siamese Difference (`FC-Siam-diff`, Csurka et al. / Daudt et al.), was trained, calibrated ($	au^*=0.5500$), and incorporated into the active registry.
3. **M8 Architecture Nomenclature:** The M8 architecture implemented in `src/change_detection/snunet.py` is Siam-Nested UNet (`SNUNet-CD`), an architecture utilizing densely connected nested UNet blocks with an Ensemble Channel Attention Module (ECAM). It was casually referred to as 'ResNet-UNet' in preliminary notes because the encoder backbone shares residual skip connectivity patterns, but its actual scientific architecture is SNUNet-CD.

### Authoritative Resolution
The live repository represents the final, verified registry of 7 change detectors:
1. `B1_Pixel_Diff`: Pixel-wise absolute difference baseline ($	au^*=0.4100$).
2. `B2_SSIM`: Structural Similarity Index dissimilarity baseline ($	au^*=0.9000$).
3. `B3_CVA`: Euclidean Change Vector Analysis baseline ($	au^*=0.4050$).
4. `M7_FC_Siam_diff`: Fully Convolutional Siamese Difference network ($	au^*=0.5500$).
5. `M8_SNUNet`: Siam-Nested UNet for Change Detection ($	au^*=0.5000$).
6. `M9_BIT_CD`: Bitemporal Image Transformer Change Detector ($	au^*=0.4500$).
7. `M10_QAT_CD`: Quality-Aware Temporal Change Detection ($\gamma \in \{0.0, 0.30\}$).

The Stage 1 application documentation must be updated to align with this authoritative registry.

---

## Discrepancy 2: QAT Scaling Parameter $\gamma$

### Problem Statement
The preliminary PDF states:
$$	ext{QAT scaling factor } \gamma = 0.0315 	ext{ (locked)}$$
The live Streamlit application and web frontend expose discrete controls:
$$\gamma = 0.0 	ext{ (validation optimal)}, \quad \gamma = 0.30 	ext{ (exploratory)}$$
Issue flagged: Unclear whether $\gamma=0.0315$ is a different parameter, an obsolete value, or a typo.

### Evidence Inspected
- Gating Implementation: [`src/change_detection/qat_cd.py:L45-L60`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/qat_cd.py#L45-L60)
- Benchmark JSON: `experiments/results/m11/benchmark_summary.json`
- Real-World Technical Reports: `docs/phase3d_frontend_integration_report.md`, `docs/phase3e_real_world_validation_and_reliability_report.md`

### Findings
1. **Mathematical Equation:**
   The gating formula implemented in `src/change_detection/qat_cd.py` is:
   $$P_{	ext{QAT}}(x, y) = P_0(x, y) \cdot \left(1 - \gamma \cdot U(x, y) \cdot Q_{	ext{defect}}(x, y)ight)$$
   where $P_0$ is the base BIT-CD probability, $U = 1 - 2|P_0 - 0.5|$ is the margin uncertainty, and $Q_{	ext{defect}} = \max(S_{	ext{reg}}, S_{	ext{illum}}, S_{	ext{sharp}})$ is the composite defect risk map.
2. **Benchmark Operating Points ($\gamma \in \{0.0, 0.30\}$):**
   In the M11 benchmark evaluation on LEVIR-CD:
   - $\gamma = 0.0$: Preserves the frozen validation-calibrated BIT-CD baseline with zero gating.
   - $\gamma = 0.30$: Represents active gating, which reduces false-positive pixels by $-29.19\%$ (from $1,965,041$ down to $1,391,488	ext{ pixels}$) and increases precision by $+5.42\%$.
3. **Real-World Radiometric Attenuation ($\gamma = 0.0315$):**
   In Phase 3D and Phase 3E, when testing unconstrained, real-world satellite pairs featuring subtle seasonal reflectance differences, applying $\gamma = 0.30$ suppressed genuine edge changes. An empirical grid search over continuous domain pairs identified $\gamma^* = 0.0315$ as the optimal attenuation constant that dampens false alarms while maintaining boundary recall.

### Authoritative Resolution
Both values are mathematically and operationally verified within their respective contexts:
- **Benchmark Mode (LEVIR-CD):** Uses the discrete UI toggle between $\gamma = 0.0$ (calibrated baseline) and $\gamma = 0.30$ (active gating).
- **Temporal Understanding Mode (Real-World Pairs):** Operates under the continuous calibrated parameter $\gamma = 0.0315$.

Documentation should explicitly distinguish the discrete benchmark research toggle from the continuous real-world calibration constant.

---

## Discrepancy 3: Explanation Structure (Tabs vs Headings)

### Problem Statement
The preliminary PDF describes six tabs:
1. Executive Summary
2. Detection Analysis
3. Spectral/Environmental Context
4. Uncertainty & Limitations
5. Operational Recommendations
6. Human Review Guidelines

The live interface renders four headings:
1. Supporting Evidence (Detector Evidence)
2. Scientific Limitations
3. Unknown Variables
4. Analyst Recommendations

Issue flagged: Unclear whether the UI is missing sections or whether the backend schema was altered.

### Evidence Inspected
- Report Data Class: [`src/temporal_understanding/explanation_engine.py:L45-L75`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/explanation_engine.py#L45-L75)
- Streamlit Implementation: [`pages/change_detection.py:L629-L650`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py#L629-L650)
- API Pydantic Schema: [`src/api/schemas.py:L140-L185`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/schemas.py#L140-L185)

### Findings
1. **Underlying Data Contract:** The backend data contract `TemporalUnderstandingReport` encapsulates 8 comprehensive fields:
   - `overall_summary`: High-level narrative summary.
   - `detector_evidence`: Specific quantitative pixel and model telemetry.
   - `region_observations`: Bounding-box geometry and cluster coordinates.
   - `semantic_interpretations`: 8-class taxonomy classifications from vision model.
   - `registration_and_quality_warnings`: Co-registration error and illumination shift warnings.
   - `scientific_limitations`: Resolution and methodological caveats.
   - `unknown_variables`: Hypothesized operational unknowns (e.g. subsurface utilities).
   - `human_review_recommendations`: Actionable guidance for human photo-analysts.
2. **Visual Presentation Layout:** Rather than overwhelming the user with six sequential text tabs, the live user interface presents:
   - **Executive Callout Banner:** Directly renders `overall_summary` at the top of Section G.
   - **Dedicated Sections D & E:** Renders `region_observations` and `semantic_interpretations` (VARI vegetation table and bounding boxes) in interactive visual tables immediately preceding Section G.
   - **Four Synthesis Tabs:** The four tabs in Section G group the remaining analytical insights:
     - Tab 1: *Supporting Evidence* (`detector_evidence`)
     - Tab 2: *Scientific Limitations* (`scientific_limitations`)
     - Tab 3: *Unknown Variables* (`unknown_variables`)
     - Tab 4: *Analyst Recommendations* (`human_review_recommendations`)

### Authoritative Resolution
There is zero discrepancy in technical capability. The live UI presents the exact same 8-field backend schema, thoughtfully partitioned into visual data tables (Sections D & E), an executive summary banner, and 4 synthesis tabs for superior cognitive ergonomics.

---

## Discrepancy 4: LLM Model Identity & Dual-Stage Pipeline

### Problem Statement
The UI button says 'Explain This Prediction (Qwen 27B)'; reports mention 'Groq Qwen'. The exact model ID, API endpoint, and Stage 1 vision model were not explicitly specified in the audit workbook.

### Evidence Inspected
- Vision Stage: [`src/temporal_understanding/semantic_analyzer.py:L48`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/semantic_analyzer.py#L48)
- Text Explainer Stage: [`src/temporal_understanding/explanation_engine.py:L38`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/temporal_understanding/explanation_engine.py#L38)
- Dedicated LLM Service: [`src/api/llm_service.py:L33`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/llm_service.py#L33)

### Findings
1. **Exact Model Identifier:** Both stages utilize `qwen/qwen3.8-27b`, served via the Groq Cloud API (`https://api.groq.com/openai/v1`).
2. **Stage 1 (Vision Stage):**
   - Implemented in `src/temporal_understanding/semantic_analyzer.py` (`_call_groq_vision`).
   - Ingests a 3-panel composite crop (T1, registered T2, visual context) enforced to $\ge 32	ext{ px}$ in height and width.
   - Classifies the crop into the 8-class taxonomy and outputs categorical confidence and observable visual features.
3. **Stage 2 (Text Explainer Stage):**
   - Implemented in `src/temporal_understanding/explanation_engine.py` and `src/api/llm_service.py` (`_call_groq_chat`).
   - Ingests **purely structured numerical and categorical telemetry** (bounding-box coordinates, GSD area in $m^2$, VARI shift, registration inliers, detector $	au^*$, and Stage 1 vision findings).
   - **Anti-Hallucination Boundary:** The text explainer NEVER receives raw pixels. It is constrained to synthesize natural language strictly from empirical measurements.

### Authoritative Resolution
- Service Provider: Groq Cloud.
- Model Identifier: `qwen/qwen3.8-27b`.
- Stage 1: Multimodal image-text analysis on composite crops.
- Stage 2: Telemetry-constrained text synthesis into structured academic reports.

---

## Discrepancy 5: Panel Naming

### Problem Statement
PDF states 'Co-Registered T2', while the live Streamlit UI states 'Observation T2 (Aligned)' and 'Observation T2 (Later Scene)'.

### Evidence Inspected
- Streamlit UI: [`pages/change_detection.py:L217-L255`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/pages/change_detection.py#L217-L255)
- Web SPA: [`frontend/js/change_detection.js:L125`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/frontend/js/change_detection.js#L125)

### Findings
This is purely a cosmetic discrepancy. In `pages/change_detection.py`, the upload inputs are labeled `Observation T1 (Earlier Scene)` and `Observation T2 (Later Scene)`. Once the registration engine successfully executes, the aligned output panel is titled `Observation T2 (Aligned)` in the Streamlit UI and `Observation T2 (Later Registered Optical Scene)` in the web SPA.

### Authoritative Resolution
The live UI nomenclature `Observation T2 (Aligned)` is authoritative and should be adopted across all Stage 1 application materials. It accurately conveys both temporal acquisition provenance (Observation T2) and mathematical coordinate transformation (Aligned to T1).

---

## Discrepancy 6: BM25 Baseline Status and Evaluation

### Problem Statement
The preliminary prompt describes CLIP only; the live dropdown includes BM25 (A1). Needs confirmation of whether BM25 is fully implemented or merely a cosmetic dropdown.

### Evidence Inspected
- BM25 Source: [`src/semantic_search/bm25.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/semantic_search/bm25.py)
- Evaluator: [`src/semantic_search/evaluator.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/semantic_search/evaluator.py)
- Ablation Artifact: `experiments/results/m13/semantic_retrieval_ablation.json`

### Findings
1. **Full Algorithmic Implementation:** BM25 is fully implemented in `src/semantic_search/bm25.py` with standard Okapi parameters ($k_1=1.5, b=0.75$), whitespace and alphanumeric tokenization, and inverted index scoring.
2. **Leave-One-Caption-Out (LOCO) Protocol:** Evaluated across all 1,093 RSICD test scenes ($5,465$ human captions). The query caption is withheld from the gallery, but the remaining 4 sibling captions for the target scene remain indexed. Because crowd-workers frequently reuse phrases (e.g. "airplanes parked at terminal gates"), BM25 LOCO achieves $R@1=42.12\%$, $R@5=60.81\%$, $R@10=67.87\%$, $	ext{MRR}=0.5112$.
3. **Category Metadata Protocol:** When all human captions are purged and only scene class labels (e.g. 'airport') are indexed, BM25 drops to $R@1=1.50\%$, $	ext{MRR}=0.0618$.
4. **CLIP Cross-Modal Integrity:** In contrast, CLIP ViT-B/32 achieves $R@1=5.45\%$, $R@10=27.89\%$, $	ext{MRR}=0.1307$ on raw image pixels without human text in the candidate gallery.

### Authoritative Resolution
BM25 is a fully implemented, scientifically evaluated baseline. It proves that lexical text-text matching suffers from severe annotation leakage, thereby validating the necessity of genuine cross-modal retrieval via CLIP ViT-B/32.

---

## Discrepancy 7: Benchmark Scene Subset

### Problem Statement
128 test scenes were evaluated in scientific reports; the live UI selector exposes 10 scenes. Needs confirmation of whether the 10 scenes are a precomputed demonstration subset.

### Evidence Inspected
- Demonstration Samples: `data/demo_samples/benchmark_pairs.json`
- M11 Full Benchmark: `experiments/results/m11/benchmark_summary.json`, `per_scene_all_models.csv`

### Findings
The full LEVIR-CD test split consists of 128 scenes of $1024 	imes 1024$ pixels ($134,217,728$ pixels). Full benchmark evaluation was executed offline across all 128 scenes in Milestone 11. To ensure responsive user interaction without requiring minutes of CPU inference per click during live demonstrations, the UI scene selector exposes 10 precomputed demonstration scenes (`test_1-5`, `test_10`, `test_20`, `test_30`, `test_40`, `test_50`) with instant visual map loading.

### Authoritative Resolution
Confirmed. The 10 scenes are an interactive demonstration subset; the complete scientific benchmark evaluated all 128 test scenes.

---

## Discrepancy 8: B2 SSIM Threshold $	au^* = 0.9000$

### Problem Statement
$	au^* = 0.9000$ on 1-SSIM dissimilarity appears unusually high. Needs code check of normalization and threshold comparison direction.

### Evidence Inspected
- Detector Implementation: [`src/change_detection/ssim_detector.py:L35-L60`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/change_detection/ssim_detector.py#L35-L60)
- Validation Threshold Results: `experiments/results/m5/m5_thresholds.json`

### Findings
- The SSIM detector computes structural dissimilarity as:
  $$D(x, y) = 1.0 - 	ext{SSIM}(x, y) \in [0.0, 1.0]$$
- Change is predicted where $D(x, y) \ge 	au^*$.
- Due to natural optical variations (shadow angles, slight sensor misalignment), unchanged terrain frequently exhibits dissimilarity values between $0.20$ and $0.80$. A threshold of $	au^* = 0.9000$ was determined via grid search on the LEVIR-CD validation split to prevent catastrophic false-positive explosion while retaining $53.32\%$ building recall.

### Authoritative Resolution
$	au^* = 0.9000$ is mathematically and empirically correct for the $1 - 	ext{SSIM}$ dissimilarity map.

---

## Discrepancy 9: Milestone vs Phase Numbering Mapping

### Problem Statement
Live UI and logs use M1–M13; reports use Phase 3A–3F. Mapping is missing in early documentation.

### Evidence Inspected
- Development Log: `.md FIles /devlog.md`
- Decision Records: `.md FIles /decision_log.md` (DEC-007)
- Milestone & Phase Reports: `docs/m4_semantic_retrieval.md` through `docs/phase3f_final_project_consolidation_report.md`

### Findings & Authoritative Mapping
- **M1–M13 (Foundational Research Roadmap):**
  - M1: Dataset Ingestion (Rahul)
  - M2: Ingestion & Verification (Tanishka)
  - M3: Cross-Modal Alignment (Adishri)
  - M4: Semantic Retrieval Baselines (Rahul)
  - M5: Classical CD Baselines B1/B2/B3 (Tanishka)
  - M6: False-Alarm & Perturbation Robustness (Adishri)
  - M7: FC-Siam-diff Learned Baseline (Rahul)
  - M8: SNUNet-CD Advanced Learned Architecture (Tanishka)
  - M9: BIT-CD Transformer Architecture (Adishri)
  - M10: QAT-CD Quality-Aware Gating (Rahul)
  - M11: Comprehensive LEVIR-CD Benchmark (Tanishka)
  - M12: High-Performance FastAPI Serving & Web UI (Adishri)
  - M13: End-to-End System Ablation & Reproducibility (Rahul)
- **Phase 3A–3F (Temporal Understanding Extensions):**
  - Phase 3A: Core Algorithmic Engines (ORB+ECC Registration, Region Extraction)
  - Phase 3B: Semantic Change Analyzer & Dual-Stage AI Reasoning (VARI, Groq Qwen)
  - Phase 3C: FastAPI End-to-End Temporal Integration
  - Phase 3D: Streamlit & Web Frontend Visualizer Integration
  - Phase 3E: Real-World Reliability Testing & Scientific Audit
  - Phase 3F: Final Project Consolidation & Defense Readiness

### Authoritative Resolution
M1–M13 covers the benchmark research and comparative evaluation; Phase 3A–3F covers the real-world temporal understanding deployment system.

---

## Discrepancy 10: Frontend Architecture and Deployment Scope

### Problem Statement
PDF describes Streamlit + web SPA; live app is an SPA with external backend. Hosting and architecture need clarification.

### Evidence Inspected
- Streamlit Entry: [`app.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/app.py)
- Web SPA Entry: [`frontend/index.html`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/frontend/index.html)
- FastAPI Application: [`src/api/app.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/app.py), [`src/api/server.py`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/src/api/server.py)
- Deployment Configs: `vercel.json`, `api/index.py`

### Findings
1. **Streamlit UI:** Located at `app.py`. Runs locally via `streamlit run app.py` (port 8501). Used for interactive research analysis and diagnostic model inspection.
2. **Web SPA:** Located at `frontend/index.html`. Connects via REST to the FastAPI backend (`/api/...`).
3. **FastAPI Backend:** Located at `src/api/app.py` and `src/api/server.py`. Runs locally via `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`. Mounts static frontend files directly.
4. **Serverless Deployment Configuration:** `vercel.json` and `api/index.py` provide a fully configured serverless deployment setup with PyTorch CPU wheel overrides. However, no production cloud domain is committed to git.

### Authoritative Resolution
Both frontends are fully functional and share the same core underlying models and services. Local execution is fully documented; cloud deployment is pre-configured for Vercel, pending user domain allocation.
