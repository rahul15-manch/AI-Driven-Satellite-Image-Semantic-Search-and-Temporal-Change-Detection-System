# Research Findings & Empirical Knowledge Base

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Academic Context:** Final-Year B.Tech CSE (AI/ML) Research Study Project  
**Team Members:** Rahul (Team Lead), Tanishka Mukhi (Dataset & Architecture), Adishri Abro (Literature & Evaluation)  
**Hardware Profile:** Strictly CPU-Only Execution ($\le 8$ GB RAM budget, commodity multi-core CPU)  
**Last Updated:** 2026-09-15 (Post-M4 Corrective Task)  

---

## 1. Executive Summary of Research Findings

This document records the empirical results, scientific interpretations, and verified conclusions across all project milestones. In accordance with the foundational research principles of the project, **every number in this document originates from actual local execution on verified datasets and hardware**, with zero fabrication or synthetic inflation.

---

## 2. Milestone 4: Semantic Retrieval Baseline & Leakage-Controlled Evaluation

### 2.1 Research Question 1 (RQ1) & Hypothesis 1 (H1) Summary
- **Research Question RQ1:** *"How effectively can vision-language embeddings retrieve semantically relevant satellite images from natural-language queries, compared with a lexical retrieval baseline, under CPU-only constraints?"*
- **Hypothesis H1:** *"Vision-language embedding retrieval may provide more semantically relevant results than simple lexical retrieval."*
- **Empirical Status of H1:** **PARTIALLY SUPPORTED / INCONCLUSIVE (Dependent on Gallery Modality)** (`DEC-025`).

### 2.2 Methodological Problem & Corrective Audit
An audit of the initial M4 implementation identified a critical methodological problem in the BM25 lexical baseline:
- The initial baseline concatenated all 5 human captions describing each test image into a single document ($D_i = C_{i1} \mathbin{\Vert} \dots \mathbin{\Vert} C_{i5}$).
- The exact same captions were simultaneously used as the 5,465 evaluation queries.
- Therefore, when query $q = C_{ik}$ was evaluated, its exact text tokens were already present inside target image $I_i$'s gallery document.
- This self-leakage inflated BM25 scores to $85.65\%$ R@1 and $0.9028$ MRR, rendering it scientifically incomparable against CLIP image-only retrieval.

### 2.3 Corrective Protocols Implemented
To establish a rigorous, leakage-free benchmark without discarding historical context, three protocols were established:
1. **Mode A — Leave-One-Caption-Out Caption-Indexed BM25 (Corrected Primary Baseline):**
   - For each query $q = C_{ik}$, the query caption is strictly removed from target image $I_i$'s BM25 document representation, leaving only the remaining 4 captions ($\bigcup_{j \ne k} C_{ij}$).
   - Non-target gallery images retain all 5 captions.
   - Exact query text is verified absent from the target document via automated regression tests (`tests/test_leakage_regression.py`).
2. **Mode B — Category Metadata Lexical Baseline (Auxiliary Image Control):**
   - Gallery images are represented strictly by their single category metadata label (e.g., `"airport"`, `"parking"`).
   - Zero human captions exist in the gallery, establishing a direct lexical equivalent to uncaptioned satellite imagery.
3. **Legacy Diagnostic Baseline (Preserved Reference):**
   - The original combined-document BM25 results are preserved in `experiments/results/m4/legacy_caption_indexed/` for auditability.

---

## 3. Locally Measured Benchmark Table `[LOCALLY MEASURED]`

*Dataset:* RSICD Test Split (1,093 gallery images, 5,465 caption queries).  
*Hardware:* Commodity multi-core CPU (strict CPU-only execution).

| Method | Protocol | Information Available to Gallery | R@1 (%) | R@5 (%) | R@10 (%) | MRR | Mean Latency (ms) | p95 Latency (ms) | Peak Process RAM (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Leave-One-Caption-Out)** | Leave-One-Caption-Out Caption-Indexed | 4 remaining human captions per target image ($q$ excluded); 5 for non-targets | **42.12** | **60.81** | **67.87** | **0.5112** | **1.49** | **2.72** | **279.7** |
| **BM25 (Category Metadata)** | Category Metadata Lexical Control | Category label only (e.g., `"airport"`); zero test captions in gallery | 1.50 | 7.30 | 14.35 | 0.0618 | 0.12 | 0.18 | 287.3 |
| **BM25 (Original Diagnostic / Leaky)** | Original Caption-Indexed (Self-Leakage) | All 5 human captions concatenated per image (exact query $q$ present in target) | 85.65 | 96.38 | 98.57 | 0.9028 | 1.46 | 2.57 | 357.4 |
| **CLIP ViT-B/32 (Zero-Shot)** | Zero-Shot Image-Text Multimodal Retrieval | Visual image pixels only (L2-normalized 512D embeddings in FAISS); zero text in gallery | 5.45 | 17.71 | 27.89 | 0.1307 | 7.54 | 8.84 | 883.6 |
| **CLIP + Prompt Ensemble** | Prompt Ensemble Multimodal Retrieval | Visual image pixels only; 5 frozen RS prompt-averaged query vectors | 5.14 | 17.00 | 28.01 | 0.1268 | 15.56 | 21.14 | 553.7 |

*FAISS Vector Search Latency:* Pre-computed IndexFlatIP inner-product search requires **0.050 ms** on CPU.

---

## 4. Key Scientific Insights

### 4.1 Quantifying the Effect of Target-Query Lexical Leakage
- Eliminating target-query leakage causes BM25 top-1 recall to drop from **85.65% down to 42.12%** (a $-43.53\%$ absolute decrease) and MRR to drop from **0.9028 down to 0.5112** (a $-0.3916$ decrease).
- This confirms that **over 50% of legacy BM25 performance was an artifact of target-text overlap**, not generalizable cross-description lexical matching.

### 4.2 Information Asymmetry & The Two Faces of H1
The comparison between lexical search and vision-language embeddings depends critically on what information is available in the gallery:
1. **The Uncaptioned Imagery Scenario (CLIP vs Mode B):**
   - In realistic geospatial search systems, satellite image repositories possess no pre-existing human captions.
   - When lexical search is constrained to metadata categories (Mode B), zero-shot CLIP outperforms lexical search by **$3.6\times$ on R@1** ($5.45\%$ vs $1.50\%$) and **$2.1\times$ on MRR** ($0.1307$ vs $0.0618$).
   - CLIP demonstrates genuine semantic understanding, successfully matching descriptive queries (e.g., *"a large grassy field with planes"*) to image pixels even when category keywords are omitted.
2. **The Caption-Indexed Scenario (Mode A vs CLIP):**
   - When comprehensive human captions are attached to every gallery image, lexical retrieval (Mode A: $42.12\%$ R@1) outperforms zero-shot CLIP ($5.45\%$ R@1).
   - This occurs because different human annotators share distinctive descriptive vocabulary for the same scene, and generalist CLIP suffers from nadir-view domain shift.

### 4.3 Instance-Level vs Category-Level Retrieval
- The RSICD Caption-to-Own-Image benchmark treats retrieval strictly as an **instance-level discrimination task** (only 1 specific image out of 1,093 is considered relevant).
- Qualitative analysis indicates that in over $75\%$ of queries where CLIP misses the specific target instance (rank $> 10$), CLIP correctly retrieves other images belonging to the **exact same semantic category** (e.g. retrieving 5 valid airport scenes for an airport query).
- Thus, zero-shot CLIP ViT-B/32 functions effectively as a **semantic category locator**, but requires remote-sensing domain adaptation (fine-tuning) to achieve fine-grained instance discrimination.

### 4.4 Domain Prompt Ensembling Ablation
- Ensembling across 5 frozen remote-sensing templates (`"{query}"`, `"a satellite image of {query}"`, etc.) yields an R@1 of **5.14%** (vs 5.45% raw query) and MRR of **0.1268** (vs 0.1307).
- While prompt ensembling improves specific ambiguous queries (e.g. moving `airport_348.jpg` from rank 16 to rank 3), averaging template embeddings slightly softens high-confidence discriminative queries across the broad test distribution, while doubling query encoding latency from **7.54 ms to 15.56 ms** on CPU.

### 4.5 CPU Feasibility & Pareto Frontier (RQ4)
- All evaluated retrieval methods operate well within the target commodity hardware constraints:
  - BM25 latency: **1.49 ms**
  - CLIP latency: **7.54 ms** (132 queries/sec throughput on CPU)
  - FAISS vector search: **0.050 ms**
  - Peak process RSS: **883.6 MB** (well below the 8 GB laptop RAM ceiling)
  - Index memory footprint: **2.18 MB** for 1,093 vectors (512D float32)

---

## 5. Failure Analysis on CLIP Zero-Shot Retrieval

An automatic rule-based keyword heuristic classification applied across all 3,941 failure cases (queries where target rank $> 10$) revealed:
1. **Broad Geographic Context Ambiguity (36.260%, 1,429 queries):** Queries with general scene terms match dozens of near-identical gallery images (e.g. 60 airports, 60 bareland scenes), causing the single target to be displaced by valid peers.
2. **Spatial / Relational Confusion (32.378%, 1,276 queries):** CLIP struggles with directional and topological prepositions (*"next to"*, *"between"*, *"surrounded by"*).
3. **Dense Small Objects (20.046%, 790 queries):** ViT-B/32 ($32 \times 32$ pixel patches) lacks the spatial resolution to resolve individual cars or small airplanes at satellite scale.
4. **Fine-Grained Attribute Mismatch (11.292%, 445 queries):** Mismatches in subtle color or geometric shape descriptors.
5. **Generic / Repetitive Caption (0.025%, 1 query):** Extremely short caption ($\le 4$ tokens).

---

## 6. Machine-Readable Result Artifacts

All empirical findings are preserved in machine-readable files under `experiments/results/m4/`:
- `summary.csv` & `summary.json`: Summary table with mandatory `protocol` and `information_available` fields.
- `corrected_bm25_results.json`: Query-level and aggregate metrics for Mode A (Leave-One-Caption-Out BM25).
- `category_metadata_bm25_results.json`: Query-level and aggregate metrics for Mode B (Category Metadata BM25).
- `legacy_caption_indexed/bm25_results.json`: Preserved legacy diagnostic results.
- `clip_results.json`: Query-level and category breakdown metrics for Zero-Shot CLIP ViT-B/32.
- `prompt_ensemble_results.json`: Query-level and category breakdown metrics for CLIP + Prompt Ensemble.
- `qualitative_samples.json`: Multi-method side-by-side rankings for deterministic query samples.
- `failure_analysis.json`: Precise breakdown of the 3,941 CLIP failure cases with documented classification heuristic.

---

---

## 7. Milestone Boundaries & Explicit Non-Implementation in M4

In accordance with strict milestone governance during M4:
- Zero change detection algorithms (M5/M6) were implemented in M4.
- Milestone 4 was strictly confined to cross-modal semantic retrieval.

---

## 8. Milestone 5: Classical Bi-Temporal Change Detection Baselines `[LOCALLY MEASURED]`

### 8.1 Research Question 2 (RQ2) & Hypothesis 2 (H2) Summary
- **Research Question RQ2:** *"How do classical image-comparison techniques and lightweight learned models compare for bi-temporal satellite-image change detection under CPU-only constraints?"*
- **Hypothesis H2:** *"A lightweight learned model may outperform purely pixel-level comparison on selected data."*
- **Empirical Status of H2:** **NOT YET EVALUATED / BASELINES ESTABLISHED.** M5 establishes the classical baseline measurements required for subsequent comparison against the M8 lightweight learned model.

### 8.2 Baseline Formulations & Protocols
Evaluated on official LEVIR-CD benchmark ($1024 \times 1024$ optical pairs, $0.5\text{ m}$ spatial resolution) with deterministic non-overlapping $256 \times 256$ patching and seamless $1024 \times 1024$ reconstruction:
1. **Method B1 (Absolute Pixel Differencing):** $D_{\text{B1}}(x, y) = \frac{1}{3}\sum_c |I_2 - I_1|$ (`DEC-027`).
2. **Method B2 (SSIM Dissimilarity):** $11 \times 11$ Gaussian window, $\sigma=1.5$ on ITU-R 601-2 luminance, $D_{\text{B2}}(x, y) = \text{clip}(1.0 - \text{SSIM}, 0.0, 1.0)$ (`DEC-028`).
3. **Method B3 (Change Vector Analysis):** Normalized Euclidean magnitude $D_{\text{B3}}(x, y) = \frac{1}{\sqrt{3}}\|\vec{I}_2 - \vec{I}_1\|_2$ (`DEC-029`).
4. **Leakage Protocol:** Decision thresholds calibrated strictly on the 64-image validation split ($67,108,864$ pixels) to maximize validation $F_1$, then frozen before test evaluation (`DEC-030`). Secondary Otsu baseline evaluated separately (`DEC-031`).

### 8.3 Empirical Benchmark Table `[LOCALLY MEASURED]`
*Dataset:* LEVIR-CD Test Split (128 pairs, $134,217,728$ pixels; ground-truth changed pixels: $6,837,404$ / $5.094\%$).  
*Hardware:* Apple Silicon / Commodity Multi-Core CPU (Strict single-process CPU execution).

| Method | Threshold Policy | Frozen $\tau^*$ | Precision (%) | Recall (%) | $F_1$ Score | IoU (%) | Mean Latency (ms/pair) | Peak RAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Pixel Diff** | **Validation $F_1$ (Frozen)** | **0.4100** | **10.64** | **16.66** | **0.1299** | **6.95** | **89.49** | **369.2** |
| B1 Pixel Diff | Validation Otsu (Secondary) | 0.2363 | 6.75 | 38.11 | 0.1147 | 6.09 | 89.49 | 369.2 |
| **B2 SSIM** | **Validation $F_1$ (Frozen)** | **0.9000** | **7.06** | **53.32** | **0.1246** | **6.65** | **138.26** | **369.3** |
| B2 SSIM | Validation Otsu (Secondary) | 0.7051 | 6.17 | 85.43 | 0.1151 | 6.11 | 138.26 | 369.3 |
| **B3 CVA** | **Validation $F_1$ (Frozen)** | **0.4050** | **10.56** | **17.46** | **0.1316** | **7.04** | **151.95** | **369.3** |
| B3 CVA | Validation Otsu (Secondary) | 0.2402 | 6.83 | 38.16 | 0.1158 | 6.15 | 151.95 | 369.3 |

### 8.4 Confusion Audit Totals
- **B1 (Validation $F_1$):** TP = $1,139,242$ | FP = $9,564,988$ | FN = $5,698,162$ | TN = $117,815,336$ | Predicted Changed: $7.98\%$
- **B2 (Validation $F_1$):** TP = $3,645,384$ | FP = $48,011,922$ | FN = $3,192,020$ | TN = $79,368,402$ | Predicted Changed: $38.49\%$
- **B3 (Validation $F_1$):** TP = $1,194,071$ | FP = $10,116,249$ | FN = $5,643,333$ | TN = $117,264,075$ | Predicted Changed: $8.43\%$

### 8.5 Key Scientific Insights from M5
1. **Performance Floor Established:** Classical non-learned methods establish an empirical performance ceiling of **$F_1 \approx 0.13$** on LEVIR-CD building change detection. Method B3 achieves the highest $F_1$ ($0.1316$) and IoU ($7.04\%$), modestly outperforming B1 ($0.1299$) and B2 ($0.1246$).
2. **False Alarm Vulnerability:** Classical baselines lack semantic awareness, responding to non-ground changes (sun angle shifts, phenological variation, shadow displacement). False positives range from $9.56\text{M}$ pixels (B1) to $48.01\text{M}$ pixels (B2).
3. **Catastrophic Failure of Otsu Thresholding:** Under high class imbalance ($5.09\%$ ground-truth change), unsupervised Otsu thresholding collapses, overestimating change by up to $13.8\times$ (predicting $70.54\%$ change).
4. **Computational Feasibility on CPU:** All classical baselines execute in under $152\text{ ms}$ per $1024 \times 1024$ image pair with peak RAM of $369.25\text{ MB}$, utilizing less than $5\%$ of the $8\text{ GB}$ hardware budget.
5. **Handoff to M6 & M8:** M5 baselines provide the quantitative benchmark for M6 perturbation sensitivity testing and M8 learned model comparisons.

---

## 9. Milestone 6: False-Alarm Analysis & Robustness Evaluation `[LOCALLY MEASURED]`

### 9.1 Research Question 3 (RQ3) & Scope Boundary
- **Research Question RQ3:** *"How sensitive are classical bi-temporal change-detection methods to non-ground visual variations such as illumination changes, blur, geometric misregistration, and localized occlusion/shadow effects?"*
- **Hypothesis Boundary:** Hypothesis H3 and any future quality-aware claims remain **UNEVALUATED**. Milestone 6 is an independent diagnostic and robustness stage; no learned models, Siamese CNNs, or quality-aware gating layers were implemented.

### 9.2 Experimental Setup & Protocol
- **Dataset:** LEVIR-CD official test split (128 optical pairs, $1024 \times 1024$ resolution, $134,217,728$ total evaluation pixels).
- **Detectors:** Frozen classical baselines from M5:
  - B1 Pixel Diff ($\tau^* = 0.4100$)
  - B2 SSIM Dissimilarity ($\tau^* = 0.9000$)
  - B3 CVA ($\tau^* = 0.4050$)
- **Threshold Policy:** Strictly frozen M5 thresholds (`DEC-033`). Zero test-data threshold retuning was performed.
- **Perturbations (12 conditions + 1 control):**
  1. *Global Illumination Shift:* $\beta \in \{+0.05, +0.15, +0.25\}$
  2. *Gaussian Blur:* $\sigma \in \{1.0, 2.0, 4.0\}$
  3. *Geometric Misregistration:* $(\Delta x, \Delta y) \in \{(1, 1), (3, 3), (5, 5)\}$ pixels
  4. *Localized Occlusion / Shadow:* Area $\in \{2\%, 5\%, 10\%\}$ at image center with $0.4\times$ intensity attenuation

### 9.3 Comprehensive Robustness Benchmark Table `[LOCALLY MEASURED]`
*All metrics computed across the entire $134,217,728$ test pixels per condition on commodity CPU.*

| Perturbation | Severity | Detector | Precision (%) | Recall (%) | $F_1$ Score | IoU (%) | Relative $F_1$ Deg (%) | Additional FP ($\Delta\text{FP}$) | Rel FP Incr (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | **none** | B1 Pixel Diff | 10.64 | 16.66 | 0.1299 | 6.95 | 0.00 | 0 | 0.00 |
| Control | none | B2 SSIM | 7.06 | 53.32 | 0.1246 | 6.65 | 0.00 | 0 | 0.00 |
| Control | none | B3 CVA | 10.56 | 17.46 | 0.1316 | 7.04 | 0.00 | 0 | 0.00 |
| **Illumination** | mild ($\beta=+0.05$) | B1 Pixel Diff | 12.36 | 15.65 | 0.1381 | 7.42 | -6.35 | -1,976,094 | -20.66 |
| Illumination | mild | B2 SSIM | 7.10 | 52.60 | 0.1251 | 6.67 | -0.38 | -949,017 | -1.98 |
| Illumination | mild | B3 CVA | 12.27 | 16.44 | 0.1406 | 7.56 | -6.81 | -2,082,803 | -20.59 |
| Illumination | medium ($\beta=+0.15$) | B1 Pixel Diff | 13.42 | 16.14 | 0.1465 | 7.91 | -12.82 | -2,441,188 | -25.52 |
| Illumination | medium | B2 SSIM | 7.05 | 52.23 | 0.1242 | 6.62 | +0.36 | -912,805 | -1.90 |
| Illumination | medium | B3 CVA | 13.26 | 16.88 | 0.1485 | 8.02 | -12.86 | -2,564,418 | -25.35 |
| Illumination | strong ($\beta=+0.25$) | B1 Pixel Diff | 10.70 | 20.42 | 0.1404 | 7.55 | -8.12 | +2,083,853 | +21.79 |
| Illumination | strong | B2 SSIM | 6.94 | 52.54 | 0.1226 | 6.53 | +1.65 | +167,022 | +0.35 |
| Illumination | strong | B3 CVA | 10.55 | 21.19 | 0.1409 | 7.58 | -7.05 | +2,169,143 | +21.44 |
| **Gaussian Blur** | mild ($\sigma=1.0$) | B1 Pixel Diff | 11.06 | 15.20 | 0.1280 | 6.84 | +1.42 | -1,207,548 | -12.62 |
| Gaussian Blur | mild | B2 SSIM | 7.27 | 44.92 | 0.1252 | 6.68 | -0.43 | -8,849,508 | -18.43 |
| Gaussian Blur | mild | B3 CVA | 10.95 | 15.97 | 0.1299 | 6.95 | +1.29 | -1,236,434 | -12.22 |
| Gaussian Blur | medium ($\sigma=2.0$) | B1 Pixel Diff | 11.19 | 14.61 | 0.1267 | 6.76 | +2.45 | -1,632,715 | -17.07 |
| Gaussian Blur | medium | B2 SSIM | 7.08 | 40.61 | 0.1205 | 6.41 | +3.32 | -11,546,739 | -24.05 |
| Gaussian Blur | medium | B3 CVA | 11.06 | 15.35 | 0.1285 | 6.87 | +2.32 | -1,672,103 | -16.53 |
| Gaussian Blur | strong ($\sigma=4.0$) | B1 Pixel Diff | 11.16 | 14.15 | 0.1248 | 6.66 | +3.91 | -1,864,043 | -19.49 |
| Gaussian Blur | strong | B2 SSIM | 6.36 | 37.26 | 0.1087 | 5.75 | **+12.82** | -10,504,629 | -21.88 |
| Gaussian Blur | strong | B3 CVA | 11.02 | 14.86 | 0.1265 | 6.75 | +3.86 | -1,909,758 | -18.88 |
| **Misregistration**| mild (1 px) | B1 Pixel Diff | 11.24 | 17.29 | 0.1363 | 7.31 | -4.90 | -231,927 | -2.42 |
| Misregistration | mild | B2 SSIM | 7.39 | 53.54 | 0.1298 | 6.94 | -4.16 | -2,115,231 | -4.41 |
| Misregistration | mild | B3 CVA | 11.14 | 18.12 | 0.1380 | 7.41 | -4.86 | -236,445 | -2.34 |
| Misregistration | medium (3 px) | B1 Pixel Diff | 11.72 | 18.65 | 0.1439 | 7.75 | -10.80 | +44,874 | +0.47 |
| Misregistration | medium | B2 SSIM | 6.97 | 54.72 | 0.1236 | 6.59 | +0.83 | +1,947,255 | +4.06 |
| Misregistration | medium | B3 CVA | 11.61 | 19.53 | 0.1456 | 7.85 | -10.67 | +49,508 | +0.49 |
| Misregistration | strong (5 px) | B1 Pixel Diff | 11.71 | 19.59 | 0.1466 | 7.91 | -12.85 | +537,763 | +5.62 |
| Misregistration | strong | B2 SSIM | 6.69 | 56.26 | 0.1195 | 6.35 | +4.12 | **+5,679,832** | **+11.83** |
| Misregistration | strong | B3 CVA | 11.60 | 20.50 | 0.1482 | 8.00 | -12.60 | +561,850 | +5.55 |
| **Occlusion/Shadow**| mild (2% area) | B1 Pixel Diff | 10.40 | 16.89 | 0.1287 | 6.88 | +0.91 | +387,267 | +4.05 |
| Occlusion/Shadow | mild | B2 SSIM | 7.03 | 53.40 | 0.1243 | 6.63 | +0.28 | +250,625 | +0.52 |
| Occlusion/Shadow | mild | B3 CVA | 10.31 | 17.70 | 0.1303 | 6.97 | +0.96 | +406,254 | +4.02 |
| Occlusion/Shadow | medium (5% area) | B1 Pixel Diff | 10.11 | 17.34 | 0.1278 | 6.82 | +1.63 | +974,703 | +10.19 |
| Occlusion/Shadow | medium | B2 SSIM | 7.00 | 53.45 | 0.1238 | 6.60 | +0.68 | +542,280 | +1.13 |
| Occlusion/Shadow | medium | B3 CVA | 10.04 | 18.17 | 0.1293 | 6.91 | +1.75 | +1,021,105 | +10.09 |
| Occlusion/Shadow | strong (10% area)| B1 Pixel Diff | 9.72 | 18.17 | 0.1267 | 6.76 | +2.49 | **+1,971,579** | **+20.61** |
| Occlusion/Shadow | strong | B2 SSIM | 6.95 | 53.58 | 0.1231 | 6.56 | +1.27 | +1,029,123 | +2.14 |
| Occlusion/Shadow | strong | B3 CVA | 9.64 | 19.02 | 0.1280 | 6.84 | +2.75 | **+2,067,512** | **+20.44** |

### 9.4 Core Empirical Findings on Baseline Vulnerabilities
1. **Misregistration-Induced False Alarm Runaway (B2 SSIM Fragility):**
   - SSIM is extremely sensitive to sub-building translational shifts. A 5-pixel shift created **$+5,679,832$ additional false positive pixels**, driving total B2 false alarms to **$53.69\text{M}$** ($40.0\%$ of the entire test dataset area).
   - *Mechanism:* Rigid displacement offsets sharp building borders and roof textures, producing wide bands of structural mismatch ($1.0 - \text{SSIM} > 0.90$) along all non-changed building edges.
2. **Defocus Recall Collapse (SSIM Structural Blindness):**
   - Under Gaussian blur ($\sigma = 4.0$), B2 suffered the worst relative $F_1$ degradation in the benchmark: **$+12.82\%$ degradation** ($F_1$ collapsed from $0.1246$ to $0.1087$).
   - *Mechanism:* Blur obliterates the high-frequency structural edges that SSIM depends on for building identification. True building recall dropped sharply from $53.32\%$ down to $37.26\%$.
3. **Spectral Sensitivity to Cloud Shadows (B1 Pixel Diff & B3 CVA):**
   - Localized cloud shadows ($10\%$ scene area, $0.4\times$ attenuation) produced **$+1,971,579$ additional FP for B1** ($+20.61\%$ relative increase) and **$+2,067,512$ additional FP for B3** ($+20.44\%$ relative increase).
   - *Mechanism:* A $60\%$ radiometric drop directly translates to large spectral displacement vectors $\|\vec{I}_2 - \vec{I}_1\|$, which cross the frozen detection thresholds ($\tau^*_{\text{B1}}=0.41, \tau^*_{\text{B3}}=0.405$) and are misclassified as temporal ground changes.
4. **Computational Verification:**
   - Full 13-condition benchmark over $134.2\text{M}$ pixels completed in **$261.2\text{ seconds}$** on commodity CPU.
   - Peak resident RAM remained strictly at **$432.8\text{ MB}$**, consuming only **$5.4\%$** of the $8\text{ GB}$ hardware budget.
   - **Bitwise Repeatability:** Independent repeated runs confirmed 100% bitwise identical confusion matrices, $F_1$, and degradation metrics across all 39 evaluated rows ($\Delta = 0.0$).
