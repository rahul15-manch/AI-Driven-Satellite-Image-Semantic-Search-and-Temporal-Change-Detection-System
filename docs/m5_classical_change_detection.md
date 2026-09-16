# Milestone 5: Classical Bi-Temporal Change Detection Baselines

**Research Milestone:** M5 — Classical Change Detection Baselines  
**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Owner:** Tanishka Mukhi (Dataset & Preprocessing)  
**Project Lead:** Rahul  
**Collaborator:** Adishri Abro (Literature & Evaluation)  
**Date:** 2026-09-16  
**Hardware Environment:** Apple Silicon / Multi-Core Commodity CPU, $\le 8$ GB RAM Budget, Strict CPU-Only Execution  
**Benchmark Dataset:** LEVIR-CD (Official Splits: Train 445, Val 64, Test 128 pairs, $1024 \times 1024$ optical imagery)  

---

## 1. Executive Summary & Objective

The primary research objective of Milestone 5 is to establish rigorous, empirically measured, and leakage-controlled classical change detection baselines on the official LEVIR-CD benchmark:
1. **Method B1: Absolute Pixel Differencing** (deterministic channel-averaged pixel disparity)
2. **Method B2: SSIM Dissimilarity** (structural similarity comparison with an $11 \times 11$ Gaussian window, $\sigma=1.5$)
3. **Method B3: Change Vector Analysis (CVA)** (Euclidean change magnitude across RGB spectral bands)

Milestone 5 contributes empirical baseline evidence toward **Research Question 2 (RQ2)**:
> *"How do classical image-comparison techniques and lightweight learned models compare for bi-temporal satellite-image change detection under CPU-only constraints?"*

Milestone 5 deliberately establishes only the classical baseline side of this comparison. In accordance with strict scientific boundaries:
- **No novelty is claimed** for B1, B2, or B3 (they are established classical remote sensing baselines).
- **Hypothesis H2** (*"A lightweight learned model may outperform purely pixel-level comparison on selected data"*) **MUST NOT be declared supported or rejected in M5**, as testing H2 strictly requires comparison against later learned models (such as the M8 lightweight architecture).
- **No M6 perturbation framework** (synthetic illumination, Gaussian blur stress tests, FPAF, quality gating) is implemented here.
- **No M8/M9 learned models** (Siamese CNNs, FC-Siam-diff, QAT-CD) are implemented here.

---

## 2. Mathematical Formulations & Methods

All input optical images $I_1$ (time $T_1$) and $I_2$ (time $T_2$) are preprocessed into normalized float32 arrays in $[0.0, 1.0]$.

### 2.1 Method B1: Absolute Pixel Differencing
Measures the per-pixel absolute difference between co-registered temporal acquisitions and averages deterministically across color channels:
$$D_{\text{B1}}(x, y) = \frac{1}{3} \sum_{c \in \{R, G, B\}} |I_2(x, y, c) - I_1(x, y, c)|$$
Where $D_{\text{B1}}(x, y) \in [0.0, 1.0]$. A secondary variant supporting maximum channel difference ($\max_c |I_2 - I_1|$) is also implemented in the codebase.

### 2.2 Method B2: Structural Similarity (SSIM) Dissimilarity
Assesses structural degradation and pattern shifts rather than pure luminance differences. In accordance with literature parameters established in M3:
- Local window size: $11 \times 11$
- Gaussian weighting standard deviation: $\sigma = 1.5$
- Dynamic range: $\Delta R = 1.0$
- Channel conversion: Standard ITU-R 601-2 luminance transformation:
$$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$
Local structural similarity is computed via:
$$\text{SSIM}(x, y) = \frac{(2\mu_1 \mu_2 + C_1)(2\sigma_{12} + C_2)}{(\mu_1^2 + \mu_2^2 + C_1)(\sigma_1^2 + \sigma_2^2 + C_2)}$$
Where continuous dissimilarity is defined as:
$$D_{\text{B2}}(x, y) = \text{clip}(1.0 - \text{SSIM}(x, y), 0.0, 1.0)$$

### 2.3 Method B3: Change Vector Analysis (CVA)
Computes the Euclidean magnitude of change vectors in the 3D RGB spectral feature space:
$$\Delta \vec{v}(x, y) = \vec{I}_2(x, y) - \vec{I}_1(x, y) = \begin{bmatrix} R_2 - R_1 \\ G_2 - G_1 \\ B_2 - B_1 \end{bmatrix}$$
$$D_{\text{B3}}(x, y) = \frac{\|\Delta \vec{v}(x, y)\|_2}{\sqrt{3}} = \frac{1}{\sqrt{3}} \sqrt{(R_2 - R_1)^2 + (G_2 - G_1)^2 + (B_2 - B_1)^2}$$
Dividing by $\sqrt{3}$ ensures continuous normalization within $[0.0, 1.0]$.

---

## 3. Patching Strategy & Full-Image Reconstruction

High-resolution $1024 \times 1024$ satellite imagery requires significant memory when processed monolithically. To guarantee execution under $\le 8$ GB commodity CPU constraints:
- Each $1024 \times 1024$ image pair is deterministically partitioned into **$16$ non-overlapping $256 \times 256$ patches** ($\text{stride} = 256$, $\text{padding\_mode} = \text{"drop"}$).
- Test split: $128 \text{ pairs} \times 16 = \mathbf{2,048} \text{ test patches}$.
- Validation split: $64 \text{ pairs} \times 16 = \mathbf{1,024} \text{ validation patches}$.
- Training split: $445 \text{ pairs} \times 16 = \mathbf{7,120} \text{ training patches}$.

### Critical Leakage-Control & Spatial Boundary Rule
Split boundaries are defined strictly at the **ORIGINAL IMAGE-PAIR LEVEL**. All 16 patches derived from a given parent pair belong exclusively to the parent's assigned split (Train, Val, or Test). Patches from the same geographic scene are never split across subsets.

### Dataset-Level Metric Reconstruction
Predictions and difference maps are computed patch-by-patch, then seamlessly assembled via `PatchExtractor.reconstruct_image` into complete $1024 \times 1024$ prediction maps. All reported test metrics are aggregated over the complete test split ($134,217,728$ pixels) at the reconstructed full-image level, avoiding patch-averaging distortion.

---

## 4. Threshold Selection Protocol

> [!IMPORTANT]
> **Strict Validation-Only Calibration**: Decision thresholds ($\tau^*$) are determined **EXCLUSIVELY on the 64-image validation split** ($67,108,864$ pixels).
> The objective function is:
> $$\tau^* = \arg\max_{\tau \in \mathcal{T}} F_1(\tau; \mathcal{D}_{\text{val}})$$
> Where candidate thresholds $\mathcal{T} = \{0.005, 0.010, \dots, 0.995\}$ (199 candidates, step size 0.005). Tie-breaking selects the smallest threshold among equal-F1 candidates.
> 
> Once calibrated, **$\tau^*$ is permanently FROZEN**. Test images are evaluated against this frozen threshold without any adjustment.

### Secondary Otsu Baseline
As specified in M3, Otsu's global histogram thresholding was evaluated as a secondary reference. Otsu's threshold $\tau_{\text{otsu}}$ was computed over the validation difference distribution and frozen before test evaluation.

---

## 5. Measured Empirical Results `[LOCALLY MEASURED]`

All values below were empirically measured across the complete LEVIR-CD test set ($128$ pairs, $134,217,728$ pixels) under strict CPU execution. **Zero numbers are fabricated or estimated.**

### 5.1 Primary Research Comparison Table

| Method | Threshold Source | Frozen Threshold $\tau^*$ | Precision (%) | Recall (%) | $F_1$ Score | IoU (%) | Mean Pair Latency (ms) | Peak RAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Pixel Diff** | **Validation $F_1$ (Primary)** | **0.4100** | **10.64** | **16.66** | **0.1299** | **6.95** | **89.49** | **369.2** |
| B1 Pixel Diff | Validation Otsu (Secondary) | 0.2363 | 6.75 | 38.11 | 0.1147 | 6.09 | 89.49 | 369.2 |
| **B2 SSIM** | **Validation $F_1$ (Primary)** | **0.9000** | **7.06** | **53.32** | **0.1246** | **6.65** | **138.26** | **369.3** |
| B2 SSIM | Validation Otsu (Secondary) | 0.7051 | 6.17 | 85.43 | 0.1151 | 6.11 | 138.26 | 369.3 |
| **B3 CVA** | **Validation $F_1$ (Primary)** | **0.4050** | **10.56** | **17.46** | **0.1316** | **7.04** | **151.95** | **369.3** |
| B3 CVA | Validation Otsu (Secondary) | 0.2402 | 6.83 | 38.16 | 0.1158 | 6.15 | 151.95 | 369.3 |

### 5.2 Pixel Confusion Audit Totals

The total number of ground-truth changed pixels in the LEVIR-CD test split is **$6,837,404$** out of $134,217,728$ pixels (**$5.094\%$**).

| Method | Threshold Policy | True Positive (TP) | False Positive (FP) | False Negative (FN) | True Negative (TN) | Predicted Changed % |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| **B1 Pixel Diff** | Validation $F_1$ | 1,139,242 | 9,564,988 | 5,698,162 | 117,815,336 | 7.98% |
| B1 Pixel Diff | Validation Otsu | 2,605,459 | 35,976,899 | 4,231,945 | 91,403,425 | 28.75% |
| **B2 SSIM** | Validation $F_1$ | 3,645,384 | 48,011,922 | 3,192,020 | 79,368,402 | 38.49% |
| B2 SSIM | Validation Otsu | 5,841,170 | 88,835,150 | 996,234 | 38,545,174 | 70.54% |
| **B3 CVA** | Validation $F_1$ | 1,194,071 | 10,116,249 | 5,643,333 | 117,264,075 | 8.43% |
| B3 CVA | Validation Otsu | 2,608,860 | 35,599,612 | 4,228,544 | 91,780,712 | 28.47% |

---

## 6. CPU Hardware & Profiling Analysis

Benchmarking was conducted on an Apple Silicon multi-core processor operating in strict single-process CPU mode:
- **Throughput & Latency:**
  - **B1 Pixel Diff** is the fastest baseline: **$89.49 \text{ ms}$** per $1024 \times 1024$ image pair (p50: $88.30 \text{ ms}$, p95: $93.82 \text{ ms}$).
  - **B2 SSIM** requires **$138.26 \text{ ms}$** per pair (p50: $136.93 \text{ ms}$, p95: $145.96 \text{ ms}$), dominated by Gaussian filter convolutions.
  - **B3 CVA** requires **$151.95 \text{ ms}$** per pair (p50: $150.35 \text{ ms}$, p95: $161.11 \text{ ms}$), with slight overhead from vector norm computations.
  - End-to-end evaluation of all 128 test image pairs (2,048 patches) took only **$19.45 \text{ seconds}$**.
- **Memory Consumption:**
  - Peak Resident Set Size (RSS) across all three baseline runs was **$369.25 \text{ MB}$**.
  - This utilizes **less than $5\%$ of the $8 \text{ GB}$ commodity hardware ceiling**, demonstrating that patch-level streaming and reconstructive aggregation is exceptionally memory-efficient.

---

## 7. Qualitative Analysis & Failure Mode Taxonomy

Qualitative comparison figures were generated using a deterministic selection policy (`test_1`, `test_20`, `test_50`, `test_100` saved in `experiments/figures/m5/`).

### 7.1 Key Observational Differences Across Baselines
1. **Method B1 (Absolute Pixel Differencing):**
   - *Behavior:* Captures high-contrast changes (such as white building roofs constructed on dark vegetation), but produces broken, noisy masks around edges.
   - *Failure Mode:* Highly sensitive to global illumination shifts. In scenes where sun angle or haze differs between T1 and T2, bare terrain generates false alarms ($9.56\text{M}$ FP pixels).
2. **Method B2 (SSIM Dissimilarity):**
   - *Behavior:* Captures structural boundaries very effectively, achieving the highest recall among validation-F1 baselines ($53.32\%$).
   - *Failure Mode:* Severe false-alarm inflation. Structural dissimilarity responds aggressively to seasonal tree foliage changes, plowed soil textures, shadow displacement, and field boundaries. It predicts $38.49\%$ of test pixels as changed (when true change is $5.09\%$), yielding $48.01\text{M}$ false positives and low precision ($7.06\%$).
3. **Method B3 (Change Vector Analysis):**
   - *Behavior:* Demonstrates the highest overall $F_1$ ($0.1316$) and IoU ($7.04\%$) among all baselines. Combining 3-band spectral differences into a single Euclidean magnitude provides slightly better signal-to-noise ratio than channel-averaging B1.
   - *Failure Mode:* Like B1, B3 is fundamentally non-semantic. Natural phenological variations (green grass to dried brown pasture) produce large spectral vectors indistinguishable from new building construction.
4. **Otsu Thresholding Failure:**
   - Otsu's method assumes a bimodal distribution of change versus non-change. Because building changes occupy only $5\%$ of pixels in LEVIR-CD, the distribution is overwhelmingly unimodal. Consequently, Otsu selects thresholds that are far too low ($\tau_{\text{otsu}} \approx 0.24$ for B1/B3 and $0.70$ for B2), causing catastrophic false alarm rates ($28.5\%$ to $70.5\%$ predicted change).

---

## 8. Research Conclusions & Boundary Adherence

1. **RQ2 Baseline Evidence:**
   - Classical pixel-level and structural baselines achieve modest $F_1$ scores between **$0.1246$ and $0.1316$** on LEVIR-CD building change detection.
   - Under strict validation-only thresholding, **Method B3 (CVA)** achieved the highest measured $F_1$ ($0.1316$), closely followed by **Method B1** ($0.1299$) and **Method B2** ($0.1246$).
   - However, all classical methods suffer from extremely high false-alarm rates ($9.5\text{M}$ to $48.0\text{M}$ false positive pixels), driven by illumination differences, agricultural phenology, and shadow movements.
2. **Hypothesis H2 Status:**
   - **H2 remains unevaluated.** M5 establishes the necessary, leakage-safe classical baseline measurements. A valid test of H2 will occur only after training and evaluating the lightweight learned model in M8.
3. **Handoff to Subsequent Milestones:**
   - **M5 $\to$ M6:** The classical baseline implementations and identified failure modes will serve as the reference for M6's controlled perturbation experiments (illumination shifts, registration errors, and false-alarm characterization).
   - **M5 $\to$ M8:** The measured $F_1 = 0.1316$ (B3), $0.1299$ (B1), and $0.1246$ (B2) provide the exact quantitative benchmark against which the M8 Siamese CNN will be compared.
