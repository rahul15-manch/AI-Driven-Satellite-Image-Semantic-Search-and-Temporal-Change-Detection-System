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

### EXP-RET-01: Classical Text-to-Image Inverted Index Baseline (Method A1)
- **Research Question:** RQ1
- **Method:** Inverted Token Index / BM25 lexical retrieval over image captions.
- **Baseline Role:** Non-neural lower bound.
- **Dataset:** RSICD `[LOCALLY MEASURED: data/raw/rsicd]`.
- **Split:** Test split (1,093 images, 5,465 query captions) evaluated against the test gallery.
- **Input:** Raw text query string $q$.
- **Output:** Permutation ranking of the 1,093 test gallery images $\pi_q$.
- **Metric:** Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR), Query Latency (ms).
- **Hardware:** Commodity CPU, single-thread execution.
- **Independent Variables:** Query length, token overlap.
- **Dependent Variables:** Retrieval accuracy ($R@K$, MRR) and query latency.
- **Controls:** Fixed RSICD Karpathy test split; identical vocabulary tokenization pipeline.
- **Expected Interpretation:** Establishes the exact degree to which literal keyword matching fails when queries use synonymous expressions or describe visual compositions not literally matched in annotation text.

### EXP-RET-02: Zero-Shot Dual-Encoder VLM Retrieval (Method A2)
- **Research Question:** RQ1, RQ4
- **Method:** Pretrained OpenAI CLIP (ViT-B/32 backbone, 512-dimensional embeddings) with FAISS `IndexFlatIP` exact inner-product search.
- **Baseline Role:** Primary vision-language embedding baseline.
- **Dataset:** RSICD.
- **Split:** Test split (1,093 candidate image embeddings, 5,465 query captions).
- **Input:** Natural language query $q$ and pre-computed test image gallery embeddings $\mathbf{V} \in \mathbb{R}^{1093 \times 512}$.
- **Output:** Sorted gallery ranking $\pi_q$ by descending cosine similarity.
- **Metric:** Recall@1, Recall@5, Recall@10, MRR, Query Latency (ms), Index RAM (MB).
- **Hardware:** Commodity CPU, default PyTorch thread pool, $\le 8$ GB RAM budget.
- **Independent Variables:** Cross-modal embedding alignment; query complexity tiers (Tier 1: Direct Category, Tier 2: Attribute + Scene, Tier 3: Spatial & Relational).
- **Dependent Variables:** $R@1, R@5, R@10$, MRR, Query Latency.
- **Controls:** Fixed RSICD test split; identical embedding normalization ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0$).
- **Expected Interpretation:** Tests Hypothesis H1. Quantifies whether zero-shot generalist vision-language representations transfer effectively to overhead satellite imagery without domain-specific fine-tuning.

### EXP-RET-03: Prompt Template & Ensembling Evaluation (Method A3 / Ablation ABL-4)
- **Research Question:** RQ1
- **Method:** Method A2 with context-guided prompt engineering and multi-prompt ensembling ($\mathbf{u}_{\text{ens}} = \frac{1}{M}\sum \mathbf{u}_m$).
- **Baseline Role:** Prompt ablation on top of Method A2.
- **Dataset:** RSICD.
- **Split:** Test split (1,093 images, 5,465 queries).
- **Input:** Query $q$ wrapped in templates (`"a satellite image of {q}"`, `"aerial overhead view of {q}"`, `"remote sensing photo of {q}"`).
- **Output:** Similarity ranking from ensembled query embeddings.
- **Metric:** $\Delta R@1, \Delta R@5, \Delta \text{MRR}$ relative to bare query prompt.
- **Hardware:** Commodity CPU.
- **Independent Variables:** Prompt template structure and ensemble depth ($M \in \{1, 3, 5\}$).
- **Dependent Variables:** Relative recall change.
- **Controls:** Same image embeddings $\mathbf{V}$; same test split.
- **Expected Interpretation:** Demonstrates whether simple prompt adaptation mitigates the domain shift of general-domain CLIP without adding model parameters or retraining.

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
- **Research Question:** RQ2
- **Method:** Euclidean channel difference $D_{\text{diff}}(x, y) = \sqrt{\frac{1}{3}\sum_{c=1}^3 (I_1(x, y, c) - I_2(x, y, c))^2}$ followed by frozen validation-calibrated thresholding $\tau^*$.
- **Baseline Role:** Simplest deterministic change detection baseline.
- **Dataset:** LEVIR-CD `[LOCALLY MEASURED: data/raw/levir_cd]`.
- **Split:** Calibrated on Validation split (64 scenes, 1,024 patches); evaluated on Test split (128 scenes, 2,048 patches).
- **Input:** Aligned RGB image pairs $I_1, I_2 \in [0, 1]^{256 \times 256 \times 3}$.
- **Output:** Binary change mask $\hat{Y} \in \{0, 1\}^{256 \times 256}$.
- **Metric:** Precision, Recall, F1-score (Changed Class), IoU, Per-Tile Latency (ms).
- **Hardware:** Commodity CPU, single-thread NumPy execution.
- **Independent Variables:** Radiometric pixel difference magnitude.
- **Dependent Variables:** Pixel classification metrics ($F_1$, IoU).
- **Controls:** Normalization to $[0, 1]$; fixed threshold $\tau^*$ derived from validation set.
- **Expected Interpretation:** Establishes the performance floor. Shows how vulnerable raw pixel comparison is to minor lighting differences and natural ground texture variation.

### EXP-CD-02: SSIM Dissimilarity Baseline (Method B2)
- **Research Question:** RQ2
- **Method:** Structural Similarity Index Measure dissimilarity $D_{\text{SSIM}}(x, y) = 1.0 - \text{SSIM}(I_1, I_2)$ using an $11 \times 11$ Gaussian sliding window ($\sigma = 1.5$), followed by frozen validation thresholding $\tau^*$.
- **Baseline Role:** Structural classical baseline resilient to uniform luminance shift.
- **Dataset:** LEVIR-CD.
- **Split:** Validation split (64 scenes) for $\tau^*$ calibration; Test split (128 scenes) for evaluation.
- **Input:** Aligned RGB image pairs $I_1, I_2$.
- **Output:** Binary change mask $\hat{Y}$.
- **Metric:** Precision, Recall, F1-score, IoU, Per-Tile Latency (ms).
- **Hardware:** Commodity CPU, vectorized OpenCV/NumPy.
- **Independent Variables:** Local structural degradation.
- **Dependent Variables:** $F_1$, IoU, Latency.
- **Controls:** Fixed SSIM constants ($C_1 = 0.0001, C_2 = 0.0009$); validation-frozen threshold $\tau^*$.
- **Expected Interpretation:** Evaluates whether separating local luminance and contrast from structural correlations reduces false alarms caused by uniform lighting shifts compared to Method B1.

### EXP-CD-03: Change Vector Analysis Baseline (Method B3)
- **Research Question:** RQ2
- **Method:** Spectral vector displacement magnitude $\rho(x, y) = \|\mathbf{I}_2(x, y) - \mathbf{I}_1(x, y)\|_2$ with validation-calibrated thresholding $\tau^*$.
- **Baseline Role:** Foundational remote-sensing multispectral baseline.
- **Dataset:** LEVIR-CD.
- **Split:** Validation split (64 scenes); Test split (128 scenes).
- **Input:** Aligned RGB image pairs $I_1, I_2$.
- **Output:** Continuous magnitude map $\rho$ and binary change mask $\hat{Y}$.
- **Metric:** Precision, Recall, F1-score, IoU, Per-Tile Latency (ms).
- **Hardware:** Commodity CPU.
- **Independent Variables:** Multi-spectral radiometric displacement.
- **Dependent Variables:** $F_1$, IoU, Latency.
- **Controls:** Identical color space preprocessing; validation-frozen threshold.
- **Expected Interpretation:** Provides direct theoretical comparison against classical remote sensing CVA formulation in optical RGB space.

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
- **Split:** Validation split (64 scenes); Test split (128 scenes).
- **Input:** Image pairs $I_1, I_2$ and learned change probability map $P$.
- **Output:** Gated change map $\hat{Y}_{\text{gated}}$ and diagnostic confidence surface.
- **Metric:** Precision, Recall, F1-score, IoU, False Positive Amplification Factor (FPAF).
- **Hardware:** Commodity CPU.
- **Independent Variables:** Gating layer activation (Enabled B5 vs. Disabled B4).
- **Dependent Variables:** False positive pixel count, FPAF, F1-score.
- **Controls:** Identical underlying trained backbone weights (Method B4).
- **Expected Interpretation:** Tests Hypothesis H3. Demonstrates whether diagnostic quality gating suppresses false alarms on clean test scenes without harming true building recall.

---

## 4. Experiment Suite C: Controlled Perturbation Robustness Testing

Grounding: Hall et al. (1991); Bovolo & Bruzzone (2007); `literature_review.md` Section 6; `metrics.md` Section 5.

```
                                  [Test Pair (T1, T2)]
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   Perturbation 1                    Perturbation 2                    Perturbation 3
[Global Illumination]               [Spatial Blur]                 [Misregistration Jitter]
I2' = alpha * I2 + beta            Gaussian Kernel sigma            dx, dy in {0.5, 1, 2} px
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           │
                                           ▼
                            Evaluate B1, B2, B3, B4, B5
                                           │
                                           ▼
                          Measure FPAF and Delta F1
```

### EXP-PERT-01: Illumination Disparity Stress Test
- **Research Question:** RQ3
- **Perturbation Formulation:** Apply synthetic radiometric gain and offset to $I_2$: $I_2'(x, y) = \text{clip}(\alpha \cdot I_2(x, y) + \beta, 0, 1)$, with $\alpha \in \{0.7, 0.85, 1.15, 1.30\}$ and $\beta \in \{-0.1, 0.1\}$.
- **Evaluated Methods:** B1 (Diff), B2 (SSIM), B3 (CVA), B4 (Siamese CNN), B5 (Quality-Gated).
- **Dataset & Split:** LEVIR-CD Test Split (128 scenes).
- **Metric:** False Positive Amplification Factor ($\text{FPAF} = \text{FP}_{\text{perturbed}} / \text{FP}_{\text{clean}}$) and Relative F1 Degradation ($\Delta F_1$).
- **Hardware:** Commodity CPU.
- **Expected Interpretation:** Demonstrates whether SSIM (B2) and learned features (B4) exhibit higher intrinsic illumination invariance than raw differencing (B1), and whether B5 prevents false alarm runaway.

### EXP-PERT-02: Atmospheric Blur & Defocus Stress Test
- **Research Question:** RQ3
- **Perturbation Formulation:** Convolve $I_2$ with a 2D Gaussian kernel of size $k \times k$ with standard deviation $\sigma \in \{1.0, 1.5, 2.0\}$ to simulate atmospheric turbulence or sensor optical defocus.
- **Evaluated Methods:** B1, B2, B3, B4, B5.
- **Dataset & Split:** LEVIR-CD Test Split (128 scenes).
- **Metric:** FPAF, $\Delta F_1$, Precision degradation.
- **Hardware:** Commodity CPU.
- **Expected Interpretation:** Quantifies detector degradation under resolution and focus mismatches between acquisition dates.

### EXP-PERT-03: Geometric Misregistration Jitter Stress Test
- **Research Question:** RQ3
- **Perturbation Formulation:** Apply 2D rigid affine translations to $I_2$ relative to $I_1$: $\Delta x, \Delta y \in \{0.5, 1.0, 2.0\}$ pixels using bilinear interpolation.
- **Evaluated Methods:** B1, B2, B3, B4, B5.
- **Dataset & Split:** LEVIR-CD Test Split (128 scenes).
- **Metric:** Edge False Positive Rate, FPAF, $\Delta F_1$.
- **Hardware:** Commodity CPU.
- **Expected Interpretation:** Tests how quickly boundary false alarms multiply as a function of co-registration error, establishing whether learned pooling in B4 or diagnostic gating in B5 provides misregistration tolerance.

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
