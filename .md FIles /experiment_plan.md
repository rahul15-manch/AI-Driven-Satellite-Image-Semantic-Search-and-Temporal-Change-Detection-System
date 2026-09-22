# Empirical Experiment & Evaluation Plan

**Milestone:** M3 — Research Definition, Literature Review & Baseline Formalization  
**Author:** Adishri Abro (Literature & Evaluation Lead)  
**Collaborators:** Rahul (Team Lead), Tanishka Mukhi (Dataset & Architecture Lead)  
**Document Version:** 2.0.0 (Grounded in Verified Literature, Formal Metrics, and M2 Dataset Partitions)

---

## 1. Experimental Philosophy & Standards

To ensure that the project maintains uncompromising scientific integrity:
1. **Null Hypothesis Stance:** We assume zero inherent superiority for learned models or proposed heuristics over classical baselines. Any claim of advantage must be demonstrated through reproducible empirical evidence under identical conditions.
2. **Strict Test Set Isolation:** No hyperparameter tuning, model checkpoint selection, or threshold calibration may be performed on the test set. All classical decision thresholds $\tau^*$ must be determined strictly on validation splits or via unsupervised algorithms, then frozen for test evaluation `[PROJECT DECISION: DEC-017]`.
3. **No Metric Fabrication:** No accuracy values, F1-scores, or latencies may be documented until benchmark scripts execute and generate empirical log files `[PROJECT DECISION: DEC-005]`.
4. **Deprecation of Misleading Metrics:** Overall Accuracy is strictly excluded as a model selection metric for change detection due to LEVIR-CD's 95.35% non-change class dominance `[PROJECT DECISION: DEC-016]`.
5. **Standardized Hardware Profiling:** All timing and memory benchmarks must be recorded on standard CPU hardware under quiet operating conditions (minimal background tasks), documenting CPU architecture, thread count, and peak RSS RAM `[PROJECT DECISION: DEC-003]`.

---

## 2. Experiment Suite A: Cross-Modal Semantic Retrieval

Grounding: Radford et al. (ICML 2021); Lu et al. (IEEE TGRS 2018); Liu et al. (IEEE TGRS 2024); `literature_review.md` Section 2.

```
                    ┌─────────────────────────┐
                    │ Natural Language Query  │
                    └───────────┬─────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
      [Method A1: BM25]                 [Method A2: CLIP ViT-B/32]
     Inverted Token Index              Normalized Dot Product
               │                                 │
               ▼                                 ▼
        Ranked Gallery                    Ranked Gallery
               │                                 │
               └────────────────┬────────────────┘
                                ▼
                   Comparative Evaluation:
                   Recall@1, Recall@5, MRR, Latency
```

### EXP-RET-01: Classical Text-to-Image Lexical Baseline (Method A1 — Leakage-Controlled)
- **Status:** **Completed & Corrected `[LOCALLY MEASURED]`**
- **Research Question:** RQ1
- **Method:** Okapi BM25 lexical retrieval evaluated across two leakage-controlled protocols plus legacy reference:
  1. *Mode A (Primary):* Leave-One-Caption-Out Caption-Indexed BM25 (query caption strictly excluded from target document).
  2. *Mode B (Auxiliary Control):* Category Metadata Lexical Baseline (gallery indexed by category labels only, zero test captions).
  3. *Legacy Reference (Diagnostic):* Original combined-document BM25 (contains target-query text overlap leakage).
- **Baseline Role:** Non-neural lexical lower bounds under distinct information modalities.
- **Dataset:** RSICD `[LOCALLY MEASURED: data/raw/rsicd]`.
- **Split:** Test split (1,093 images, 5,465 query captions) evaluated against the test gallery.
- **Input:** Raw text query string $q$.
- **Output:** Permutation ranking of the 1,093 test gallery images $\pi_q$.
- **Metric:** Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR), Query Latency (ms).
- **Hardware:** Commodity CPU, single-thread execution.
- **Controls:** Fixed RSICD test split; identical tokenization pipeline; strict regression tests preventing query re-entry.
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Mode A (Leave-One-Caption-Out, Primary Corrected):**
    - $R@1$: **42.12%**, $R@5$: **60.81%**, $R@10$: **67.87%**, MRR: **0.5112**
    - Latency: Mean **1.49 ms** (p95: **2.72 ms**) | Peak Process RAM: **279.7 MB**
  - **Mode B (Category Metadata Control, No Gallery Captions):**
    - $R@1$: **1.50%**, $R@5$: **7.30%**, $R@10$: **14.35%**, MRR: **0.0618**
    - Latency: Mean **0.12 ms** (p95: **0.18 ms**) | Peak Process RAM: **287.3 MB**
  - **Legacy Diagnostic (Leakage-Affected Reference):**
    - $R@1$: **85.65%**, $R@5$: **96.38%**, $R@10$: **98.57%**, MRR: **0.9028**
    - Latency: Mean **1.46 ms** (p95: **2.57 ms**) | Peak Process RAM: **357.4 MB**
- **Scientific Interpretation:** Removing target-query leakage drops caption-indexed BM25 R@1 from 85.65% to 42.12%. However, when comparing against zero-shot CLIP (5.45% R@1), Mode A retains an information advantage (having 4 human descriptive captions per image). When evaluated fairly with zero gallery captions (Mode B), zero-shot CLIP dramatically outperforms lexical retrieval (5.45% vs 1.50% R@1; 0.1307 vs 0.0618 MRR).


### EXP-RET-02: Zero-Shot Dual-Encoder VLM Retrieval (Method A2)
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ1, RQ4
- **Method:** Pretrained OpenAI CLIP (ViT-B/32 backbone, 512-dimensional embeddings) with FAISS `IndexFlatIP` exact inner-product search.
- **Baseline Role:** Primary vision-language embedding baseline.
- **Dataset:** RSICD.
- **Split:** Test split (1,093 candidate image embeddings, 5,465 query captions).
- **Input:** Natural language query $q$ and pre-computed test image gallery embeddings $\mathbf{V} \in \mathbb{R}^{1093 \times 512}$.
- **Output:** Sorted gallery ranking $\pi_q$ by descending cosine similarity.
- **Metric:** Recall@1, Recall@5, Recall@10, MRR, Query Latency (ms), Index RAM (MB).
- **Hardware:** Commodity CPU, default PyTorch thread pool, $\le 8$ GB RAM budget.
- **Independent Variables:** Cross-modal embedding alignment; query complexity tiers.
- **Dependent Variables:** $R@1, R@5, R@10$, MRR, Query Latency.
- **Controls:** Fixed RSICD test split; identical embedding normalization ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0$).
- **Measured Empirical Outcome `[LOCALLY MEASURED]`:**
  - $R@1$: **5.45%**, $R@5$: **17.71%**, $R@10$: **27.89%**, MRR: **0.1307**
  - Latency: Mean **7.18 ms** (p95: **7.74 ms**), FAISS search only: **0.053 ms** | Peak Process RAM: **1,260.6 MB**
- **Scientific Interpretation:** Under Caption-to-Own-Image, zero-shot CLIP experiences severe overhead domain shift. While it retrieves visually consistent semantic categories, it cannot discriminate specific target image instances from visually similar category peers.

### EXP-RET-03: Prompt Template & Ensembling Evaluation (Method A3 / Ablation ABL-4)
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ1
- **Method:** Method A2 with context-guided prompt engineering and multi-prompt ensembling across 5 frozen templates.
- **Baseline Role:** Prompt ablation on top of Method A2.
- **Dataset:** RSICD.
- **Split:** Test split (1,093 images, 5,465 queries).
- **Input:** Query $q$ wrapped in 5 frozen templates (`"a satellite image of {q}"`, `"a remote sensing image of {q}"`, etc.).
- **Output:** Similarity ranking from ensembled query embeddings.
- **Metric:** $\Delta R@1, \Delta R@5, \Delta \text{MRR}$ relative to bare query prompt.
- **Hardware:** Commodity CPU.
- **Independent Variables:** Prompt template structure and ensemble depth ($M = 5$).
- **Dependent Variables:** Relative recall change.
- **Controls:** Same image embeddings $\mathbf{V}$; same test split.
- **Measured Empirical Outcome `[LOCALLY MEASURED]`:**
  - $R@1$: **5.14%**, $R@5$: **17.00%**, $R@10$: **28.01%**, MRR: **0.1268**
  - Latency: Mean **13.97 ms** (p95: **17.82 ms**) | Peak Process RAM: **669.2 MB**
- **Scientific Interpretation:** Domain prompt ensembling slightly improves deep ranks for select ambiguous categories (e.g. airport rank improved from 16 to 3 in qualitative examples), but overall global instance-level R@1 remains essentially flat ($5.14\%$ vs $5.45\%$) while doubling query latency on CPU.

---

## 3. Experiment Suite B: Bi-Temporal Change Detection

Grounding: Singh (1989); Wang et al. (2004); Malila (1980); Daudt et al. (IEEE ICIP 2018); Chen & Shi (Remote Sensing 2020); `literature_review.md` Sections 4 & 5.

```
                      ┌────────────────────────────┐
                      │ Bi-Temporal Images (T1, T2)│
                      └─────────────┬──────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
[Method B1: Diff]            [Method B2: SSIM]            [Method B4: FC-Siam]
Absolute Pixel Dist          Structural Dissim           Learned Conv Features
       │                            │                            │
       ▼                            ▼                            ▼
Difference Map D             Dissimilarity Map D          Probability Map P
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
                     [Leakage-Safe Thresholding]
                     Calibrated strictly on Val Split (64 scenes)
                     Frozen threshold tau* applied to Test (128 scenes)
                                    │
                                    ▼
                     Comparative Evaluation:
                     F1-Score, IoU, Precision, Recall, Latency
```

### EXP-CD-01: Absolute Pixel Differencing Baseline (Method B1)
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ2
- **Method:** Deterministic channel-averaged absolute difference $D_{\text{B1}}(x, y) = \frac{1}{3}\sum_{c=1}^3 |I_2(x, y, c) - I_1(x, y, c)|$ followed by frozen validation-calibrated thresholding $\tau^*$.
- **Baseline Role:** Simplest deterministic change detection baseline.
- **Dataset:** LEVIR-CD `[LOCALLY MEASURED: data/raw/levir_cd]`.
- **Split:** Calibrated on Validation split (64 scenes, 1,024 patches); evaluated on Test split (128 scenes, 2,048 patches).
- **Input:** Aligned RGB image pairs $I_1, I_2 \in [0, 1]^{256 \times 256 \times 3}$ reconstructed to $1024 \times 1024$.
- **Output:** Binary change mask $\hat{Y} \in \{0, 1\}^{1024 \times 1024}$.
- **Metric:** Precision, Recall, F1-score (Changed Class), IoU, Confusion Matrix (TP, FP, FN, TN), Per-Pair Latency (ms).
- **Hardware:** Commodity CPU (single-process CPU execution).
- **Controls:** Normalization to $[0, 1]$; fixed threshold $\tau^*$ derived exclusively from validation split.
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Validation-F1 Frozen Threshold ($\tau^* = 0.4100$):**
    - Precision: **10.64%** | Recall: **16.66%** | $F_1$: **0.1299** | IoU: **6.95%** | Accuracy: **88.63%**
    - Pixel Totals: TP = **1,139,242** | FP = **9,564,988** | FN = **5,698,162** | TN = **117,815,336**
    - Predicted Changed Ratio: **7.98%** (Ground Truth: **5.09%**)
    - Latency: Mean **89.49 ms** / pair (p50: **88.30 ms**, p95: **93.82 ms**) | Peak Process RAM: **369.2 MB**
  - **Validation-Otsu Secondary Threshold ($\tau = 0.2363$):**
    - Precision: **6.75%** | Recall: **38.11%** | $F_1$: **0.1147** | IoU: **6.09%** | Accuracy: **70.04%**
    - Pixel Totals: TP = **2,605,459** | FP = **35,976,899** | FN = **4,231,945** | TN = **91,403,425**
    - Predicted Changed Ratio: **28.75%**
- **Scientific Interpretation:** Confirms the performance floor. B1 suffers from low precision (10.64%) due to illumination variations across bare terrain and shadows, generating over 9.5M false positive pixels. Otsu fails severely due to class imbalance, over-predicting changes by nearly 6x.

### EXP-CD-02: SSIM Dissimilarity Baseline (Method B2)
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ2
- **Method:** Structural Similarity Index Measure dissimilarity $D_{\text{SSIM}}(x, y) = \text{clip}(1.0 - \text{SSIM}(I_1, I_2), 0.0, 1.0)$ using an $11 \times 11$ Gaussian sliding window ($\sigma = 1.5$) on ITU-R 601-2 luminance, followed by frozen validation thresholding $\tau^*$.
- **Baseline Role:** Structural classical baseline resilient to uniform luminance shift.
- **Dataset:** LEVIR-CD.
- **Split:** Validation split (64 scenes) for $\tau^*$ calibration; Test split (128 scenes) for evaluation.
- **Input:** Aligned RGB image pairs $I_1, I_2$ reconstructed to $1024 \times 1024$.
- **Output:** Binary change mask $\hat{Y} \in \{0, 1\}^{1024 \times 1024}$.
- **Metric:** Precision, Recall, F1-score, IoU, Confusion Totals, Per-Pair Latency (ms).
- **Hardware:** Commodity CPU, vectorized skimage/NumPy.
- **Controls:** Fixed SSIM constants ($11 \times 11, \sigma = 1.5$); validation-frozen threshold $\tau^*$.
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Validation-F1 Frozen Threshold ($\tau^* = 0.9000$):**
    - Precision: **7.06%** | Recall: **53.32%** | $F_1$: **0.1246** | IoU: **6.65%** | Accuracy: **61.85%**
    - Pixel Totals: TP = **3,645,384** | FP = **48,011,922** | FN = **3,192,020** | TN = **79,368,402**
    - Predicted Changed Ratio: **38.49%** (Ground Truth: **5.09%**)
    - Latency: Mean **138.26 ms** / pair (p50: **136.93 ms**, p95: **145.96 ms**) | Peak Process RAM: **369.3 MB**
  - **Validation-Otsu Secondary Threshold ($\tau = 0.7051$):**
    - Precision: **6.17%** | Recall: **85.43%** | $F_1$: **0.1151** | IoU: **6.11%** | Accuracy: **33.07%**
    - Pixel Totals: TP = **5,841,170** | FP = **88,835,150** | FN = **996,234** | TN = **38,545,174**
    - Predicted Changed Ratio: **70.54%**
- **Scientific Interpretation:** While SSIM captures building structural shifts effectively (achieving the highest recall at 53.32%), it is excessively sensitive to natural surface texture variations (foliage, plowed earth, seasonal grass shifts), generating 48.0M false positive pixels and yielding the lowest precision (7.06%) among validation-F1 baselines.

### EXP-CD-03: Change Vector Analysis Baseline (Method B3)
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ2
- **Method:** Spectral vector displacement magnitude $D_{\text{B3}}(x, y) = \frac{1}{\sqrt{3}}\|\mathbf{I}_2(x, y) - \mathbf{I}_1(x, y)\|_2$ with validation-calibrated thresholding $\tau^*$.
- **Baseline Role:** Foundational remote-sensing multispectral baseline.
- **Dataset:** LEVIR-CD.
- **Split:** Validation split (64 scenes); Test split (128 scenes).
- **Input:** Aligned RGB image pairs $I_1, I_2$ reconstructed to $1024 \times 1024$.
- **Output:** Continuous magnitude map and binary change mask $\hat{Y}$.
- **Metric:** Precision, Recall, F1-score, IoU, Confusion Totals, Per-Pair Latency (ms).
- **Hardware:** Commodity CPU.
- **Controls:** Identical color space preprocessing; validation-frozen threshold.
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Validation-F1 Frozen Threshold ($\tau^* = 0.4050$):**
    - Precision: **10.56%** | Recall: **17.46%** | $F_1$: **0.1316** | IoU: **7.04%** | Accuracy: **88.26%**
    - Pixel Totals: TP = **1,194,071** | FP = **10,116,249** | FN = **5,643,333** | TN = **117,264,075**
    - Predicted Changed Ratio: **8.43%** (Ground Truth: **5.09%**)
    - Latency: Mean **151.95 ms** / pair (p50: **150.35 ms**, p95: **161.11 ms**) | Peak Process RAM: **369.3 MB**
  - **Validation-Otsu Secondary Threshold ($\tau = 0.2402$):**
    - Precision: **6.83%** | Recall: **38.16%** | $F_1$: **0.1158** | IoU: **6.15%** | Accuracy: **70.33%**
    - Pixel Totals: TP = **2,608,860** | FP = **35,599,612** | FN = **4,228,544** | TN = **91,780,712**
    - Predicted Changed Ratio: **28.47%**
- **Scientific Interpretation:** B3 achieves the highest $F_1$ (0.1316) and IoU (7.04%) among all classical baselines. The 3D Euclidean magnitude provides slightly better signal integration than channel averaging (B1), but still lacks semantic discrimination, suffering from 10.1M false positive pixels on non-building land-cover transitions.


### EXP-CD-04: Fully Convolutional Siamese Difference Network (Method B4)
- **Research Question:** RQ2, RQ4
- **Method:** `FC-Siam-diff` architecture (Daudt et al., 2018): weight-sharing twin convolutional encoders with multi-scale feature differencing $|\mathbf{F}_1^{(l)} - \mathbf{F}_2^{(l)}|$ routed to a convolutional decoder with skip connections. Total parameters $\le 2$M.
- **Baseline Role:** Primary lightweight learned deep baseline.
- **Dataset:** LEVIR-CD.
- **Split:** Train split (445 scenes / 7,120 patches) for training; Validation split (64 scenes) for model checkpoint selection; Test split (128 scenes) for evaluation.
- **Input:** Tiled image patches $I_1, I_2 \in \mathbb{R}^{256 \times 256 \times 3}$.
- **Output:** Probability heatmap $P(\text{Change}) \in [0, 1]^{256 \times 256}$ and binary mask $\hat{Y}$ at threshold $\tau = 0.5$.
- **Metric:** Precision, Recall, F1-score, IoU, CPU Inference Latency (ms/tile), Peak RAM (MB).
- **Hardware:** Commodity CPU inference, $\le 8$ GB RAM.
- **Independent Variables:** Hierarchical convolutional feature extraction vs. pixel-level metrics.
- **Dependent Variables:** Predictive accuracy ($F_1$, IoU) and computational consumption.
- **Controls:** Identical $256 \times 256$ patching; identical data augmentation protocol.
- **Expected Interpretation:** Tests Hypothesis H2. Quantifies the exact margin of performance gain achieved by learned structural representations over classical baselines, alongside the corresponding CPU computational cost.

### EXP-CD-05: Diagnostic Quality-Gated Change Detection (Method B5)
- **Research Question:** RQ3
- **Method:** Method B4 coupled with a lightweight diagnostic gating layer that attenuates change confidence in regions exhibiting high illumination disparity or registration jitter.
- **Baseline Role:** Proposed adaptation/combination pipeline.
- **Dataset:** LEVIR-CD.
- **Input:** Image pairs $I_1, I_2$ and learned change probability map $P$.
- **Output:** Gated change map $\hat{Y}_{\text{gated}}$ and diagnostic confidence surface.
- **Metric:** Precision, Recall, F1-score, IoU, False Positive Amplification Factor (FPAF).
- **Hardware:** Commodity CPU.
- **Independent Variables:** Gating layer activation (Enabled B5 vs. Disabled B4).
- **Dependent Variables:** False positive pixel count, FPAF, F1-score.
- **Controls:** Identical underlying trained backbone weights (Method B4).
- **Expected Interpretation:** Tests Hypothesis H3. Demonstrates whether diagnostic quality gating suppresses false alarms on clean test scenes without harming true building recall.

---

## 4. Experiment Suite C: Controlled Perturbation Robustness Testing (Milestone 6)

Grounding: Hall et al. (1991); Bovolo & Bruzzone (2007); `literature_review.md` Section 6; `metrics.md` Section 5; `decision_log.md` DEC-032 to DEC-035.

```
                                  [Test Pair (T1, T2)]
                                           │
         ┌───────────────────┬─────────────┴───────┬───────────────────┐
         ▼                   ▼                     ▼                   ▼
    Perturbation 1      Perturbation 2        Perturbation 3      Perturbation 4
 [Global Illumination]  [Gaussian Blur]     [Misregistration]   [Occlusion/Shadow]
  beta in {0.05,0.15,0.25} sigma in {1,2,4}  dx,dy in {1,3,5}px  area {2%,5%,10%}, 0.4x
         │                   │                     │                   │
         └───────────────────┴─────────────┬───────┴───────────────────┘
                                           │
                                           ▼
                            Evaluate B1, B2, B3 (Frozen M5 Tau)
                                           │
                                           ▼
                           Measure Delta F1, Rel Deg %, Delta FP
```

### EXP-PERT-01: Global Illumination Shift Stress Test
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ3
- **Perturbation Formulation:** Apply additive radiometric shift to $I_2$: $I_2'(x, y) = \text{clip}(I_2(x, y) + \beta, 0.0, 1.0)$ with $\beta \in \{+0.05, +0.15, +0.25\}$.
- **Evaluated Methods:** B1 (Diff, $\tau^*=0.4100$), B2 (SSIM, $\tau^*=0.9000$), B3 (CVA, $\tau^*=0.4050$).
- **Dataset & Split:** LEVIR-CD Test Split (128 scenes, 134,217,728 pixels).
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Mild ($\beta = +0.05$):**
    - B1: $F_1 = 0.1381$ (deg: $-6.35\%$), FP = $7,588,894$ ($\Delta\text{FP} = -1,976,094$)
    - B2: $F_1 = 0.1251$ (deg: $-0.38\%$), FP = $47,062,905$ ($\Delta\text{FP} = -949,017$)
    - B3: $F_1 = 0.1406$ (deg: $-6.81\%$), FP = $8,033,446$ ($\Delta\text{FP} = -2,082,803$)
  - **Medium ($\beta = +0.15$):**
    - B1: $F_1 = 0.1465$ (deg: $-12.82\%$), FP = $7,123,800$ ($\Delta\text{FP} = -2,441,188$)
    - B2: $F_1 = 0.1242$ (deg: $+0.36\%$), FP = $47,099,117$ ($\Delta\text{FP} = -912,805$)
    - B3: $F_1 = 0.1485$ (deg: $-12.86\%$), FP = $7,551,831$ ($\Delta\text{FP} = -2,564,418$)
  - **Strong ($\beta = +0.25$):**
    - B1: $F_1 = 0.1404$ (deg: $-8.12\%$), FP = $11,648,841$ ($\Delta\text{FP} = +2,083,853$, $+21.79\%$ increase)
    - B2: $F_1 = 0.1226$ (deg: $+1.65\%$), FP = $48,178,944$ ($\Delta\text{FP} = +167,022$, $+0.35\%$ increase)
    - B3: $F_1 = 0.1409$ (deg: $-7.05\%$), FP = $12,285,392$ ($\Delta\text{FP} = +2,169,143$, $+21.44\%$ increase)
- **Scientific Interpretation:** Mild/medium positive shifts paradoxically suppress background false alarms in dark terrain for B1/B3 while shifting true changes closer to the decision boundary. However, under strong shift ($\beta=+0.25$), false alarms explode by over $+2.08\text{M}$ pixels for B1 and $+2.17\text{M}$ pixels for B3. SSIM (B2) demonstrates high radiometric invariance, maintaining false alarms within $\pm 2\%$ across all severities.

### EXP-PERT-02: Gaussian Blur Defocus Stress Test
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ3
- **Perturbation Formulation:** Convolve $I_2$ with an isotropic Gaussian filter of standard deviation $\sigma \in \{1.0, 2.0, 4.0\}$ and truncation $4.0$.
- **Evaluated Methods:** B1, B2, B3 under frozen M5 thresholds.
- **Dataset & Split:** LEVIR-CD Test Split (128 scenes).
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Mild ($\sigma = 1.0$):**
    - B1: $F_1 = 0.1280$ (deg: $+1.42\%$), FP = $8,357,440$ ($\Delta\text{FP} = -1,207,548$)
    - B2: $F_1 = 0.1252$ (deg: $-0.43\%$), FP = $39,162,414$ ($\Delta\text{FP} = -8,849,508$)
    - B3: $F_1 = 0.1299$ (deg: $+1.29\%$), FP = $8,879,815$ ($\Delta\text{FP} = -1,236,434$)
  - **Medium ($\sigma = 2.0$):**
    - B1: $F_1 = 0.1267$ (deg: $+2.45\%$), FP = $7,932,273$ ($\Delta\text{FP} = -1,632,715$)
    - B2: $F_1 = 0.1205$ (deg: $+3.32\%$), FP = $36,465,183$ ($\Delta\text{FP} = -11,546,739$)
    - B3: $F_1 = 0.1285$ (deg: $+2.32\%$), FP = $8,444,146$ ($\Delta\text{FP} = -1,672,103$)
  - **Strong ($\sigma = 4.0$):**
    - B1: $F_1 = 0.1248$ (deg: $+3.91\%$), FP = $7,700,945$ ($\Delta\text{FP} = -1,864,043$)
    - B2: $F_1 = 0.1087$ (deg: $+12.82\%$), FP = $37,507,293$ ($\Delta\text{FP} = -10,504,629$)
    - B3: $F_1 = 0.1265$ (deg: $+3.86\%$), FP = $8,206,491$ ($\Delta\text{FP} = -1,909,758$)
- **Scientific Interpretation:** Blur smooths high-frequency texture, reducing noise false positives across all detectors. However, B2 suffers severe structural collapse: true building recall drops from $53.32\%$ down to $37.26\%$, causing a $+12.82\%$ relative $F_1$ degradation due to loss of edge definition.

### EXP-PERT-03: Geometric Misregistration Jitter Stress Test
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ3
- **Perturbation Formulation:** Apply deterministic 2D rigid spatial translation to $I_2$: $(\Delta x, \Delta y) \in \{(1, 1), (3, 3), (5, 5)\}$ pixels with nearest-neighbor padding.
- **Evaluated Methods:** B1, B2, B3 under frozen M5 thresholds.
- **Dataset & Split:** LEVIR-CD Test Split (128 scenes).
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Mild (1 px shift):**
    - B1: $F_1 = 0.1363$ (deg: $-4.90\%$), FP = $9,333,061$ ($\Delta\text{FP} = -231,927$)
    - B2: $F_1 = 0.1298$ (deg: $-4.16\%$), FP = $45,896,691$ ($\Delta\text{FP} = -2,115,231$)
    - B3: $F_1 = 0.1380$ (deg: $-4.86\%$), FP = $9,879,804$ ($\Delta\text{FP} = -236,445$)
  - **Medium (3 px shift):**
    - B1: $F_1 = 0.1439$ (deg: $-10.80\%$), FP = $9,609,862$ ($\Delta\text{FP} = +44,874$)
    - B2: $F_1 = 0.1236$ (deg: $+0.83\%$), FP = $49,959,177$ ($\Delta\text{FP} = +1,947,255$)
    - B3: $F_1 = 0.1456$ (deg: $-10.67\%$), FP = $10,165,757$ ($\Delta\text{FP} = +49,508$)
  - **Strong (5 px shift):**
    - B1: $F_1 = 0.1466$ (deg: $-12.85\%$), FP = $10,102,751$ ($\Delta\text{FP} = +537,763$)
    - B2: $F_1 = 0.1195$ (deg: $+4.12\%$), FP = $53,691,754$ ($\Delta\text{FP} = +5,679,832$, $+11.83\%$ increase)
    - B3: $F_1 = 0.1482$ (deg: $-12.60\%$), FP = $10,678,099$ ($\Delta\text{FP} = +561,850$)
- **Scientific Interpretation:** Misregistration exposes the severe boundary fragility of SSIM (B2). At 5 pixels of displacement, B2 false alarms surge by **$+5,679,832$ pixels** (reaching $53.69\text{M}$ total FP), as displaced building and terrain edges create complete structural dissimilarity rings.

### EXP-PERT-04: Localized Occlusion and Cloud Shadow Stress Test
- **Status:** **Completed `[LOCALLY MEASURED]`**
- **Research Question:** RQ3
- **Perturbation Formulation:** Deterministic centered square patch covering $\{2\%, 5\%, 10\%\}$ scene area in $I_2$ attenuated by $0.4\times$ intensity scaling ($60\%$ darkening).
- **Evaluated Methods:** B1, B2, B3 under frozen M5 thresholds.
- **Dataset & Split:** LEVIR-CD Test Split (128 scenes).
- **Measured Empirical Outcomes `[LOCALLY MEASURED]`:**
  - **Mild (2% area):**
    - B1: $F_1 = 0.1287$ (deg: $+0.91\%$), FP = $9,952,255$ ($\Delta\text{FP} = +387,267$)
    - B2: $F_1 = 0.1243$ (deg: $+0.28\%$), FP = $48,262,547$ ($\Delta\text{FP} = +250,625$)
    - B3: $F_1 = 0.1303$ (deg: $+0.96\%$), FP = $10,522,503$ ($\Delta\text{FP} = +406,254$)
  - **Medium (5% area):**
    - B1: $F_1 = 0.1278$ (deg: $+1.63\%$), FP = $10,539,691$ ($\Delta\text{FP} = +974,703$)
    - B2: $F_1 = 0.1238$ (deg: $+0.68\%$), FP = $48,554,202$ ($\Delta\text{FP} = +542,280$)
    - B3: $F_1 = 0.1293$ (deg: $+1.75\%$), FP = $11,137,354$ ($\Delta\text{FP} = +1,021,105$)
  - **Strong (10% area):**
    - B1: $F_1 = 0.1267$ (deg: $+2.49\%$), FP = $11,536,567$ ($\Delta\text{FP} = +1,971,579$, $+20.61\%$ increase)
    - B2: $F_1 = 0.1231$ (deg: $+1.27\%$), FP = $49,041,045$ ($\Delta\text{FP} = +1,029,123$, $+2.14\%$ increase)
    - B3: $F_1 = 0.1280$ (deg: $+2.75\%$), FP = $12,183,761$ ($\Delta\text{FP} = +2,067,512$, $+20.44\%$ increase)
- **Scientific Interpretation:** Localized cloud shadows produce severe false alarms for spectral detectors (B1 and B3), generating ~2.0M additional false-positive pixels under strong occlusion. Because the shadow abruptly reduces reflectance across all bands, pixel differencing and CVA mistake the shadow boundary and interior for actual temporal ground changes. SSIM produces ~1.0M additional false positives along shadow boundaries where local contrast drops.

---

## 5. Experiment Suite D: Systematic Ablation Studies

Grounding: `literature_review.md` Sections 7 & 12; `research_questions.md` Section 3.

| Ablation ID | Target Pipeline | Configuration Tested | Hypothesis Being Tested |
| :--- | :--- | :--- | :--- |
| **EXP-ABL-01** | Method B5 (Full Quality-Gated) | Complete pipeline: B4 + Radiometric Gating + Structural Check | Full proposed configuration reference |
| **EXP-ABL-02** | Method B5 (Ablation) | Remove Radiometric Gating (Evaluate B4 + Structural Check only) | Isolates exact false-alarm reduction attributable to radiometric gating |
| **EXP-ABL-03** | Method B5 (Ablation) | Remove Structural Consistency (Evaluate B4 + Radiometric Gating only) | Isolates contribution of local structural checks |
| **EXP-ABL-04** | Method A2 (Retrieval) | Raw query vs. Domain-specific prompt templates (`"a satellite image of..."`) | Quantifies retrieval gain from zero-shot prompt engineering |

---

## 6. Experiment Suite E: Computational Cost & Pareto Profiling

Grounding: `metrics.md` Section 4; `decision_log.md` DEC-003.

### EXP-HW-01: Full-Pipeline CPU Hardware Benchmarking
- **Research Question:** RQ4
- **Evaluated Methods:** All retrieval methods (A1, A2) and change detection methods (B1, B2, B3, B4, B5).
- **Input:** Standardized batch of 100 retrieval queries and 100 LEVIR-CD test patches ($256 \times 256$).
- **Profiling Metrics:**
  - Per-sample wall-clock latency: Mean, Median ($p50$), 95th percentile ($p95$).
  - Peak resident memory (RSS in MB) via `psutil`.
  - Peak heap allocation (MB) via `tracemalloc`.
  - Total model parameter count ($\sum \theta$).
  - Storage footprint of model checkpoints on disk (MB).
- **Hardware:** Standard student laptop CPU (documenting exact CPU model and thread allocation).
- **Expected Interpretation:** Tests Hypothesis H4. Produces the empirical Pareto frontier plotting Task Performance ($R@1$ or $F_1$) against CPU Latency and Peak RAM.
