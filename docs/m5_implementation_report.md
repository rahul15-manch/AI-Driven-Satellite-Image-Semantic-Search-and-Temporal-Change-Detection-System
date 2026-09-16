# Milestone 5 Implementation Report: Classical Bi-Temporal Change Detection Baselines

**Project Title:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Milestone:** M5 — Classical Bi-Temporal Change Detection Baselines  
**Milestone Owner:** Tanishka Mukhi (Dataset & Preprocessing)  
**Project Lead:** Rahul  
**Collaborator:** Adishri Abro (Literature Review & Evaluation)  
**Date of Completion:** 2026-09-16  
**Repository Branch / Head Commit:** `main` (clean working directory)  
**Evaluation Hardware Profile:** Apple Silicon / Multi-Core Commodity CPU, Strict CPU-Only Execution, 8 GB RAM Budget  

---

## 1. Executive Summary

Milestone 5 has successfully implemented, unit-tested, and empirically benchmarked three classical bi-temporal change detection baselines on the official LEVIR-CD benchmark:
1. **Method B1 (Absolute Pixel Differencing):** Channel-averaged absolute pixel disparity.
2. **Method B2 (SSIM Dissimilarity):** Structural similarity comparison using an $11 \times 11$ Gaussian kernel ($\sigma=1.5$).
3. **Method B3 (Change Vector Analysis - CVA):** Euclidean change magnitude across RGB spectral dimensions.

To enforce strict data leakage control, all decision thresholds were calibrated **exclusively on the 64-image validation split** to maximize validation $F_1$, and subsequently **permanently frozen** prior to evaluation on the 128-image test split. Primary metrics were computed on full $1024 \times 1024$ reconstructed prediction maps aggregated across all $134,217,728$ test pixels.

**Key Measured Findings:**
- Under the frozen validation threshold protocol, **Method B3 (CVA)** achieved the highest test performance (**$F_1 = 0.1316$**, **IoU = $7.04\%$**, Precision = $10.56\%$, Recall = $17.46\%$), closely followed by **Method B1 (Pixel Diff)** (**$F_1 = 0.1299$**, **IoU = $6.95\%$**) and **Method B2 (SSIM)** (**$F_1 = 0.1246$**, **IoU = $6.65\%$**).
- Classical methods are dominated by false alarms ($9.56\text{M}$ to $48.01\text{M}$ false positive pixels), driven by illumination shifts, phenology, and shadow movements.
- Secondary Otsu thresholding failed catastrophically due to severe class imbalance ($5.09\%$ ground-truth change), over-predicting changes up to $70.54\%$ of total test pixels.
- Average execution latency on CPU is **$89.49\text{ ms}$ (B1)**, **$138.26\text{ ms}$ (B2)**, and **$151.95\text{ ms}$ (B3)** per image pair. Peak memory footprint is **$369.25\text{ MB}$** (<5% of 8 GB ceiling).
- All 106 unit, regression, leakage, and integration tests pass with 100% green coverage. Results are 100% bitwise reproducible across repeated runs.

---

## 2. Objective

The objective of M5 is to establish reproducible, empirical classical baselines for bi-temporal satellite image change detection on LEVIR-CD under strict CPU constraints, contributing empirical evidence toward **Research Question 2 (RQ2)**. M5 is a **RESEARCH BASELINE MILESTONE**, not a production system. No algorithmic novelty is claimed.

---

## 3. Dataset

- **Dataset Name:** LEVIR-CD (Large-scale Visible Change Detection Dataset).
- **Format:** RGB optical bi-temporal image pairs ($1024 \times 1024$ resolution, $0.5\text{ m/pixel}$ spatial resolution) with binary building change masks.
- **Source Integrity:** Verified user-provided dataset validated in M2. No external downloads or dataset substitutions were permitted.
- **Binary Mask Convention:** Standard LEVIR-CD convention where pixel value $> 127$ indicates class `1` (Changed / Building Construction) and $\le 127$ indicates class `0` (Unchanged).

---

## 4. Split Management & Leakage Boundaries

The official LEVIR-CD split manifest established in M2 was strictly enforced:
- **Training Set:** 445 pairs ($7,120$ patches) — reserved for M8 learned models.
- **Validation Set:** 64 pairs ($1,024$ patches / $67,108,864$ pixels) — exclusively used for threshold calibration.
- **Test Set:** 128 pairs ($2,048$ patches / $134,217,728$ pixels) — used exclusively for frozen baseline evaluation.
- **Leakage Boundary Rule:** Split partitioning is enforced at the parent image-pair level. All patches from any parent pair remain quarantined in that parent's assigned split. Pair IDs across train, val, and test are mutually disjoint.

---

## 5. Patching Strategy

LEVIR-CD full scenes are $1024 \times 1024$. To satisfy strict commodity CPU execution without memory exhaustion:
- Patch Size: $256 \times 256$ pixels.
- Stride: $256$ pixels (non-overlapping tiling, `padding_mode="drop"`).
- Tiling Grid: Exactly $4 \times 4 = 16$ patches per image pair.
- Reconstruction: Difference maps and binary masks are generated patch-by-patch and assembled back into full $1024 \times 1024$ continuous spatial arrays using `PatchExtractor.reconstruct_image`.

---

## 6. Preprocessing

- Inputs $I_1$ (pre-change) and $I_2$ (post-change) are converted to float32 arrays and normalized to $[0.0, 1.0]$:
  $$I_{\text{norm}} = \frac{I}{255.0}$$
- Preprocessing is strictly deterministic. No random cropping, random flipping, color jitter, or learned data augmentations were applied.

---

## 7. Method B1 Formulation (Absolute Pixel Differencing)

Calculates the absolute per-pixel difference averaged across the 3 RGB color channels:
$$D_{\text{B1}}(x, y) = \frac{1}{3} \sum_{c \in \{R, G, B\}} |I_2(x, y, c) - I_1(x, y, c)|$$
Values are normalized to $[0.0, 1.0]$. A secondary maximum-channel mode ($\max_c |I_2 - I_1|$) is also implemented.

---

## 8. Method B2 Formulation (SSIM Dissimilarity)

Calculates structural similarity over localized Gaussian windows using M3 literature parameters:
- Local window size: $11 \times 11$
- Gaussian sigma: $\sigma = 1.5$
- Dynamic range: $1.0$
- Channel handling: Standard ITU-R 601-2 luminance conversion:
  $$Y = 0.299 R + 0.587 G + 0.114 B$$
Continuous dissimilarity map:
$$D_{\text{B2}}(x, y) = \text{clip}(1.0 - \text{SSIM}(x, y), 0.0, 1.0)$$

---

## 9. Method B3 Formulation (Change Vector Analysis)

Calculates the Euclidean magnitude of change vectors in the 3D RGB spectral space:
$$D_{\text{B3}}(x, y) = \frac{1}{\sqrt{3}} \sqrt{(R_2 - R_1)^2 + (G_2 - G_1)^2 + (B_2 - B_1)^2}$$
Normalized by $\frac{1}{\sqrt{3}}$ to enforce range in $[0.0, 1.0]$.

---

## 10. Threshold Selection Procedure

1. **Validation Search Space:** Candidate threshold grid $\mathcal{T} = \{0.005, 0.010, \dots, 0.995\}$ (199 candidate thresholds, step size $0.005$).
2. **Optimization Objective:** Select threshold $\tau^*$ maximizing validation $F_1$ across all 64 validation pairs ($67,108,864$ pixels):
   $$\tau^* = \arg\max_{\tau \in \mathcal{T}} F_1(\tau; \mathcal{D}_{\text{val}})$$
3. **Tie-Breaking Rule:** In the event of identical maximum $F_1$ scores, select the smallest threshold candidate.
4. **Frozen Application:** The selected $\tau^*$ is permanently frozen and saved to `thresholds.json`. No test split labels or predictions are ever accessible to the threshold optimizer.

---

## 11. Secondary Otsu Experiment

Otsu's global variance minimization threshold was computed on the aggregated validation difference map distributions:
- B1 Otsu threshold: $\tau_{\text{otsu}} = 0.2363$ (vs $\tau^* = 0.4100$)
- B2 Otsu threshold: $\tau_{\text{otsu}} = 0.7051$ (vs $\tau^* = 0.9000$)
- B3 Otsu threshold: $\tau_{\text{otsu}} = 0.2402$ (vs $\tau^* = 0.4050$)

Otsu selected thresholds that were drastically lower than the optimal $F_1$ thresholds because change events represent only $5\%$ of the total pixels. This produced severe over-detection on test imagery.

---

## 12. Primary Test Metrics `[LOCALLY MEASURED]`

All metrics below were computed over the complete test set ($128$ pairs, $134,217,728$ pixels) for the **CHANGED** class ($1$).

| Baseline Method | Threshold Policy | Threshold $\tau$ | Precision | Recall | $F_1$ Score | IoU | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Pixel Diff** | **Validation $F_1$ (Frozen)** | **0.4100** | **0.1064** | **0.1666** | **0.1299** | **0.0695** | **0.8863** |
| B1 Pixel Diff | Validation Otsu (Secondary) | 0.2363 | 0.0675 | 0.3811 | 0.1147 | 0.0609 | 0.7004 |
| **B2 SSIM** | **Validation $F_1$ (Frozen)** | **0.9000** | **0.0706** | **0.5332** | **0.1246** | **0.0665** | **0.6185** |
| B2 SSIM | Validation Otsu (Secondary) | 0.7051 | 0.0617 | 0.8543 | 0.1151 | 0.0611 | 0.3307 |
| **B3 CVA** | **Validation $F_1$ (Frozen)** | **0.4050** | **0.1056** | **0.1746** | **0.1316** | **0.0704** | **0.8826** |
| B3 CVA | Validation Otsu (Secondary) | 0.2402 | 0.0683 | 0.3816 | 0.1158 | 0.0615 | 0.7033 |

---

## 13. Pixel Confusion Totals for Auditing

Total ground-truth changed pixels: **$6,837,404$** ($5.094\%$ of all $134,217,728$ test pixels).

| Method | Threshold Policy | True Positive (TP) | False Positive (FP) | False Negative (FN) | True Negative (TN) | Predicted Changed % |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| **B1 Pixel Diff** | Validation $F_1$ | 1,139,242 | 9,564,988 | 5,698,162 | 117,815,336 | 7.98% |
| B1 Pixel Diff | Validation Otsu | 2,605,459 | 35,976,899 | 4,231,945 | 91,403,425 | 28.75% |
| **B2 SSIM** | Validation $F_1$ | 3,645,384 | 48,011,922 | 3,192,020 | 79,368,402 | 38.49% |
| B2 SSIM | Validation Otsu | 5,841,170 | 88,835,150 | 996,234 | 38,545,174 | 70.54% |
| **B3 CVA** | Validation $F_1$ | 1,194,071 | 10,116,249 | 5,643,333 | 117,264,075 | 8.43% |
| B3 CVA | Validation Otsu | 2,608,860 | 35,599,612 | 4,228,544 | 91,780,712 | 28.47% |

---

## 14. CPU Runtime & Latency Profile

Measured using `time.perf_counter()` under strict CPU execution:
- **Total Test Split Runtime (128 pairs / 2,048 patches):** $19.45\text{ s}$.
- **Mean Latency per $1024 \times 1024$ Pair:**
  - **B1 Pixel Diff:** $89.49\text{ ms}$ (p50: $88.30\text{ ms}$, p95: $93.82\text{ ms}$)
  - **B2 SSIM:** $138.26\text{ ms}$ (p50: $136.93\text{ ms}$, p95: $145.96\text{ ms}$)
  - **B3 CVA:** $151.95\text{ ms}$ (p50: $150.35\text{ ms}$, p95: $161.11\text{ ms}$)

---

## 15. Memory Profile

Measured using `resource.getrusage`:
- **Peak Resident Set Size (RSS):** **$369.25\text{ MB}$** (macOS).
- **RAM Budget Ceiling:** $\le 8,192\text{ MB}$ ($8\text{ GB}$).
- **Headroom:** Utilizes only **$4.51\%$** of available memory ceiling.

---

## 16. Qualitative Examples

Deterministic representative test samples were saved to `experiments/figures/m5/`:
- `qualitative_comparison_test_1.png`
- `qualitative_comparison_test_20.png`
- `qualitative_comparison_test_50.png`
- `qualitative_comparison_test_100.png`
- `validation_threshold_curves.png`

---

## 17. Failure Analysis

1. **Illumination Differences:** Solar angle shifts and atmospheric haze create diffuse, high-magnitude differences across unchanged soil and open ground.
2. **Phenological & Agricultural Variation:** Color shifts in non-building vegetation produce large change vectors in B1 and B3.
3. **Shadow Movements:** Sun azimuth changes displace building and tree shadows, which SSIM and CVA register as high-contrast structural alterations.
4. **Texture & Edge Sensitivity:** SSIM responds aggressively to micro-textures (plowed fields, tree canopies), generating $48\text{M}$ false positive pixels.
5. **Class Imbalance Sensitivity:** Unsupervised thresholding techniques like Otsu fail when change represents $<10\%$ of pixels.

---

## 18. Reproducibility & Repeatability

The M5 benchmark runner was executed across two independent runs. Metrics, thresholds, and confusion totals matched bitwise:
- Run 1 vs Run 2 Thresholds: Bitwise identical ($\tau^*_{\text{B1}} = 0.410000$, $\tau^*_{\text{B2}} = 0.900000$, $\tau^*_{\text{B3}} = 0.405000$).
- Run 1 vs Run 2 Test $F_1$: Bitwise identical ($0.129890$, $0.124640$, $0.131595$).
- Numerical Tolerance: Deviation $\Delta = 0.0$.

---

## 19. Unit Tests

Unit test suites created and passing:
- `tests/test_patch_reconstruction.py` (3 tests)
- `tests/test_b1_pixel_diff.py` (5 tests)
- `tests/test_b2_ssim.py` (4 tests)
- `tests/test_b3_cva.py` (4 tests)
- `tests/test_m5_evaluator.py` (4 tests)

---

## 20. Leakage & Integration Tests

- `tests/test_m5_leakage.py` (6 tests: Tests A through F verifying zero test-set exposure, immutability of calibrated thresholds, disjoint split IDs, and invariance under adversarial test label permutations).
- `tests/integration/test_m5_change_detection.py` (End-to-end integration test from LEVIR loader through patch reconstruction and metric evaluation).
- **Existing Regression:** All 79 prior M1–M4 tests continue to pass cleanly (Total active tests: **106**).

---

## 21. Research Observations

- Classical baselines establish a clear performance ceiling of **$F_1 \approx 0.13$** on LEVIR-CD building change detection.
- Method B3 (CVA) slightly outperforms B1 and B2 because Euclidean combination across 3 spectral bands reduces single-channel noise.
- None of the classical methods possess semantic awareness: they cannot distinguish between building construction, vegetation growth, soil tillage, or shadow elongation.

---

## 22. Limitations

1. High sensitivity to lighting and atmospheric variations.
2. Inability to distinguish semantic ground change from natural environmental shifts.
3. SSIM is prone to severe false alarms on textured natural surfaces.
4. Classical baselines rely on global static thresholds that fail under heterogeneous scene lighting.

---

## 23. Key Assumptions

1. LEVIR-CD optical image pairs are sufficiently orthorectified and co-registered.
2. Normalized RGB spectral differences provide a legitimate classical baseline signal.
3. $11 \times 11$ Gaussian SSIM provides an appropriate structural baseline representation.

---

## 24. Decision Log

Decisions recorded in `decision_log.md`:
- `DEC-026`: Deterministic LEVIR-CD $256 \times 256$ Patching and Seamless Reconstruction.
- `DEC-027`: Method B1 Absolute Pixel Differencing Formulation.
- `DEC-028`: Method B2 SSIM Dissimilarity Configuration.
- `DEC-029`: Method B3 Change Vector Analysis (CVA) Formulation.
- `DEC-030`: Strict Validation-Only Threshold Selection Policy.
- `DEC-031`: Secondary Otsu Thresholding Evaluation.

---

## 25. What Was NOT Implemented

In strict adherence to milestone boundaries:
- **No M6 perturbation framework** (synthetic illumination, blur, misregistration, FPAF, quality gating).
- **No M8 Siamese CNN** (FC-Siam-diff, lightweight deep change detectors).
- **No M9 quality-aware model** (QAT-CD, temporal confidence models).

---

## 26. Hypothesis H2 Status

> **Hypothesis H2:** *"A lightweight learned model may outperform purely pixel-level comparison on selected data."*

**Status: NOT YET EVALUATED / BASELINES ESTABLISHED.**  
M5 establishes the classical baseline measurements ($F_1 = 0.1316$ for B3, $0.1299$ for B1, $0.1246$ for B2). Testing H2 strictly requires comparison against the lightweight learned model to be trained in Milestone 8.

---

## 27. Handoff to M6 and M8

- **M5 $\to$ M6 Handoff:** The classical baseline implementations (`PixelDiffDetector`, `SSIMDetector`, `CVADetector`) and their calibrated thresholds are ready for M6's synthetic perturbation experiments to quantify degradation under illumination, blur, and misregistration.
- **M5 $\to$ M8 Handoff:** The quantitative benchmark metrics ($F_1 = 0.1316$, Precision = $10.56\%$, Recall = $17.46\%$, Latency = $89\text{--}152\text{ ms}$, RAM = $369\text{ MB}$) provide the exact targets for evaluating the M8 Siamese CNN.
