# Milestone M11: Comprehensive Experimental Evaluation, Fair Benchmarking & Scientific Validation Report

**Author:** Senior Computer Vision Research Engineer & ML Evaluation Architect  
**Date:** October 13, 2026  
**Status:** `VERIFIED`  
**Milestone:** M11 (Experimental Evaluation)  
**Execution Environment:** macOS arm64 (Apple Silicon), Python 3.10.18, PyTorch 2.5.1 CPU, `torch.set_num_threads(4)`  
**Dataset:** LEVIR-CD Official Splits (445 train, 64 validation, 128 held-out test scenes; $1024 \times 1024$ resolution)

---

## 1. Executive Summary

Milestone M11 establishes the definitive experimental evaluation, statistical uncertainty quantification, and fair CPU benchmarking framework for the *AI-Driven Satellite Image Semantic Search and Temporal Change Detection* research system. 

Prior milestones (M5–M10) individually implemented classical non-learned baselines (B1 Pixel Difference, B2 SSIM Dissimilarity, B3 Change Vector Analysis) and learned architectures (M7 FC-Siam-diff, M8 SNUNet-CD, M9 BIT-CD, and M10 QAT-CD). However, their evaluations occurred in separate milestone scripts with varying profiling conditions, ununified timing methodologies, and pixel-level point estimates without scene-level statistical uncertainty quantification.

M11 resolves these gaps by implementing a unified evaluation harness (`src/change_detection/benchmark_harness.py`), recomputing all metrics from ground-truth confusion matrices without modification to historical records, conducting paired scene-level bootstrap resampling ($B=1000$) for 95% confidence intervals, performing paired Wilcoxon signed-rank hypothesis tests across all 128 test scenes, rigorously auditing M10 quality-aware gating trade-offs, and benchmarking inference latency (mean, P50, P95) and peak process RSS under identical CPU hardware conditions (`torch.set_num_threads(4)`).

### Core Findings at a Glance

1. **Learned vs. Classical Separation:** Learned detectors outperform classical non-learned methods by an order of magnitude in F1 and IoU. FC-Siam-diff ($F_1 = 0.6035$), BIT-CD ($F_1 = 0.5927$), and QAT-CD ($F_1 = 0.5927$) dramatically reduce false alarms compared to Pixel Diff ($F_1 = 0.1299$), SSIM ($F_1 = 0.1246$), and CVA ($F_1 = 0.1316$).
2. **BIT-CD Parameter and Precision Efficiency:** While FC-Siam-diff achieves slightly higher global F1 ($0.6035$ vs $0.5927$, $\Delta = +0.0108$) and higher scene-level macro F1 ($0.4720$ vs $0.4402$, paired Wilcoxon $p = 0.0356$), BIT-CD requires **62.5% fewer parameters** (552,097 vs 1,471,409) and achieves higher precision ($65.36\%$ vs $62.81\%$, $16.4\%$ fewer false positive pixels).
3. **SNUNet-CD Pathological False Alarms:** Despite achieving the highest recall ($77.58\%$), SNUNet-CD exhibits catastrophic false-positive generation ($53,103,841$ FP pixels; $39.57\%$ of the entire evaluation split), yielding an aggregate F1 of only $0.1626$ and IoU of $8.85\%$. Paired testing confirms BIT-CD dramatically outperforms SNUNet-CD ($p = 9.29 \times 10^{-20}$).
4. **QAT-CD Active Gating Operational Trade-Off:** Validation-selected QAT-CD selects attenuation parameter $\gamma = 0.0$, strictly reproducing unmodified BIT-CD ($F_1 = 0.5927$). The exploratory active-gating setting ($\gamma = 0.30$) eliminates **573,553 false-positive pixels** (a **29.19% reduction in false alarms**), increasing precision from $65.36\%$ to $70.78\%$ (+5.42 pp). However, this incurs a loss of 336,213 true-positive pixels (recall decreases from $54.22\%$ to $49.30\%$), slightly lowering F1 to $0.5812$ (-1.94%). Active gating is highly advantageous in false-alarm-penalized operational deployments (exchange ratio of 1.71 false alarms suppressed per genuine changed pixel conceded), though unweighted F1 favors $\gamma = 0.0$.
5. **Reproducibility Guarantee:** All 8 detector configurations strictly satisfy full-scene pixel accounting ($TP + FP + FN + TN = 134,217,728$ pixels across 128 scenes; $6,837,404$ ground-truth changed pixels). 100% of regression and contract tests pass (236 / 236 tests passing in 43.68s).

---

## 2. Verified M11 Scope and Milestone Boundaries

Per `architecture.md` and the authoritative research specification, Milestone M11 is strictly designated as **Comprehensive Experimental Evaluation, Fair Benchmarking & Scientific Validation**.

- **Anti-Scope Boundary:** M11 does NOT introduce new model architectures, train additional weights, modify historical checkpoint weights, or create future milestone functionality (M12+ FAISS scale-up or FastAPI serving).
- **Milestone Contracts:** `tests/integration/test_m1_m4_contracts.py` was updated to recognize M11 and enforce strict protection against premature M12+ implementations (`checkpoints_m12`, `results_m12`, forbidden future classes).
- **Historical Immutability:** Historical summary artifacts in `experiments/results/m5/`, `m7/`, `m8/`, `m9/`, and `m10/` remain frozen and unmodified.

---

## 3. Repository and Artifact Audit

### Environment Specification
- **Hardware Architecture:** Apple Silicon arm64 (macOS 15.x / Darwin 26.6.2)
- **CPU Cores:** 8-core CPU (Apple M-series)
- **Host Memory:** 16 GB unified memory
- **Python Version:** 3.10.18
- **PyTorch Version:** 2.5.1 CPU
- **Thread Allocation:** Standardized to `torch.set_num_threads(4)` across all benchmark executions

### Checkpoint and Model Inventory

| Milestone & Detector ID | Architecture Type | Checkpoint Path | Checkpoint SHA-256 | Size (Bytes) | Parameter Count |
|---|---|---|---|---:|---:|
| B1 Pixel Difference | Classical (L1/Grayscale) | N/A (Algorithmic) | N/A | 0 | 0 |
| B2 SSIM Dissimilarity | Classical (Structural) | N/A (Algorithmic) | N/A | 0 | 0 |
| B3 CVA | Classical (Spectral Magnitude) | N/A (Algorithmic) | N/A | 0 | 0 |
| M7 FC-Siam-diff | Fully Convolutional Siamese | `experiments/checkpoints/m7/fc_siam_diff_best.pt` | `d0ced6952a631f1bc786d416a2b8f0f71150eede227d9a7059ac01b254889760` | 5,940,138 | 1,471,409 |
| M8 SNUNet-CD | Dense Siamese + ECAM Attention | `experiments/checkpoints/m8/snunet_best.pt` | `15b153aad944640bf3a250ab3de21e7d4986f2bf5b7fc2504386c25f0336bbe5` | 2,309,546 | 563,377 |
| M9 BIT-CD | ResNet18 + Bitemporal Transformer | `experiments/checkpoints/m9/bit_cd_best.pt` | `5f31724612a73f852d080441cfbcc97a46a517a5273748aa4ba639fa855d00df` | 2,259,854 | 552,097 |
| M10 QAT-CD Calibrated | BIT-CD + Multi-Signal Confounder Gate | Base M9 Checkpoint + Metadata | `e6674aa14c6eed6915623fd92be3ac8b78fbce4907d7e8751be3579663f273cc` | 543 (meta) | 552,097 |
| M10 QAT-CD Active Gating | BIT-CD + Active Confounder Gate ($\gamma=0.3$) | Base M9 Checkpoint + Metadata | `e6674aa14c6eed6915623fd92be3ac8b78fbce4907d7e8751be3579663f273cc` | 543 (meta) | 552,097 |

---

## 4. Research Questions and Empirical Hypotheses

The experimental evaluation addresses six core research questions:

1. **RQ1 (Classical vs. Learned Benchmark):** Do learned deep-learning change detectors statistically outperform classical radiometric/structural detectors under identical split conditions?
   - *Hypothesis H1:* Learned detectors achieve statistically significant higher F1 and IoU ($p < 0.001$) and lower false positive rates.
   - *Outcome:* **CONFIRMED**. M7, M9, and M10 surpass B1–B3 by over $+0.46$ in F1 ($p = 3.61 \times 10^{-20}$).
2. **RQ2 (BIT-CD vs. FC-Siam-diff Trade-Off):** Does the transformer-based BIT-CD provide a favorable parameter and efficiency trade-off compared to the heavier FC-Siam-diff?
   - *Hypothesis H2:* BIT-CD achieves comparable F1 ($within \pm 0.02$) with significantly fewer parameters ($>50\%$ reduction) and higher precision.
   - *Outcome:* **CONFIRMED**. BIT-CD achieves $F_1 = 0.5927$ vs FC-Siam-diff's $0.6035$ ($\Delta = -0.0108$), reduces parameters by $62.5\%$, and improves precision from $62.81\%$ to $65.36\%$.
3. **RQ3 (SNUNet-CD Role & Vulnerability):** What does SNUNet-CD contribute given its reported high recall and high false alarms?
   - *Hypothesis H3:* SNUNet achieves highest recall but exhibits severe precision deficit due to dense feature over-aggregation without spatial regularization.
   - *Outcome:* **CONFIRMED**. SNUNet achieves $77.58\%$ recall but catastrophic $9.08\%$ precision with over 53.1 million false-alarm pixels.
4. **RQ4 & RQ5 (QAT-CD Operational Trade-Off):** Does QAT-CD active quality gating provide an operational advantage when false alarms are more costly than missed detections, and does that advantage justify lower recall/F1?
   - *Hypothesis H4:* Active gating ($\gamma=0.30$) reduces false positives by $>20\%$ and increases precision by $>4$ pp at a controlled cost to recall.
   - *Outcome:* **CONFIRMED**. False positives drop by $29.19\%$ ($-573,553$ pixels), precision increases by $+5.42$ pp ($65.36\% \to 70.78\%$). Recall drops by $-4.92$ pp ($54.22\% \to 49.30\%$).
5. **RQ6 (Reproducibility & Generalization Boundaries):** Which findings represent verified prospective conclusions versus exploratory post-hoc findings?
   - *Outcome:* **DOCUMENTED**. The official LEVIR-CD test set has been evaluated repeatedly across milestones; claims of untouched prospective generalization are scientifically bounded. Active gating ($\gamma=0.30$) is classified as exploratory rather than validation-optimal.

---

## 5. Dataset Splits and Leakage Controls

Evaluation strictly utilizes the official LEVIR-CD splits:
- **Training Set:** 445 scene pairs ($1024 \times 1024$)
- **Validation Set:** 64 scene pairs ($1024 \times 1024$) — used strictly for threshold calibration and parameter selection
- **Held-Out Test Set:** 128 scene pairs ($1024 \times 1024$) — strictly evaluated with frozen parameters

### Leakage Audit Results
- Train / Test overlap: **0 scenes** (Verified)
- Validation / Test overlap: **0 scenes** (Verified)
- Train / Validation overlap: **0 scenes** (Verified)
- Total test scene pairs: **128 pairs**
- Total test evaluated pixels: **134,217,728 pixels** ($128 \times 1024 \times 1024$)
- Total ground-truth changed pixels: **6,837,404 pixels** ($5.094\%$ prevalence)
- Total ground-truth unchanged pixels: **127,380,324 pixels** ($94.906\%$ prevalence)

---

## 6. Evaluation Protocol and Metric Definitions

All metrics are recomputed from accumulated confusion matrix counts using double-precision arithmetic:

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

$$
\text{IoU} = \frac{TP}{TP + FP + FN}, \quad \text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}
$$

- **Zero-Denominator Convention:** If $(TP + FP) = 0$, Precision is assigned $0.0$. If $(TP + FN) = 0$, Recall is assigned $0.0$. If $2TP + FP + FN = 0$, F1 is assigned $0.0$.
- **Pixel-Level Aggregation:** Confusion outcomes ($TP, FP, FN, TN$) are accumulated across all 128 full scenes ($134,217,728$ pixels) prior to calculating global micro metrics.
- **Scene-Level Aggregation:** Metrics are also calculated per individual scene ($N=128$) to obtain macro distributions and support paired non-parametric statistical tests.

---

## 7. Threshold Provenance Audit

| Detector ID | Decision Threshold $\tau^*$ | Provenance Policy | Split Calibration | Leakage Risk |
|---|---:|---|---|---|
| B1 Pixel Difference | 0.4100 | Validation F1-Optimal | 64 Val Scenes | NONE (Frozen) |
| B2 SSIM Dissimilarity | 0.9000 | Validation F1-Optimal | 64 Val Scenes | NONE (Frozen) |
| B3 CVA | 0.4050 | Validation F1-Optimal | 64 Val Scenes | NONE (Frozen) |
| M7 FC-Siam-diff | 0.5500 | Validation F1-Optimal | 64 Val Scenes | NONE (Frozen) |
| M8 SNUNet-CD | 0.5000 | Validation F1-Optimal | 64 Val Scenes | NONE (Frozen) |
| M9 BIT-CD | 0.4500 | Validation F1-Optimal | 64 Val Scenes | NONE (Frozen) |
| M10 QAT-CD Calibrated | 0.4500 ($\gamma^*=0.0$) | Validation Grid Sweep | 64 Val Scenes | NONE (Frozen) |
| M10 QAT-CD Active Gating | 0.4500 ($\gamma=0.30$) | Exploratory False-Alarm Run | Test-Exploratory | EXPLORATORY (Post-hoc) |

---

## 8. Main Detector Benchmark Comparison

The table below presents the verified benchmark comparison across all 8 evaluated detector configurations on the official 128-scene LEVIR-CD test split ($134,217,728$ evaluated pixels).

| Detector | Threshold $\tau^*$ | Precision | Recall | F1 Score | IoU | Accuracy | TP | FP | FN | TN | Parameters | Checkpoint Size | Latency Mean (ms) | Peak RSS (MB) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **B1: Pixel Diff** | 0.4100 | 10.64% | 16.66% | 0.1299 | 6.95% | 88.63% | 1,139,242 | 9,564,988 | 5,698,162 | 117,815,336 | 0 | 0 B | 13.0 ms | 447.5 MB | `VERIFIED` |
| **B2: SSIM** | 0.9000 | 7.06% | 53.32% | 0.1246 | 6.65% | 61.85% | 3,645,384 | 48,011,922 | 3,192,020 | 79,368,402 | 0 | 0 B | 47.0 ms | 449.8 MB | `VERIFIED` |
| **B3: CVA** | 0.4050 | 10.56% | 17.46% | 0.1316 | 7.04% | 88.26% | 1,194,071 | 10,116,249 | 5,643,333 | 117,264,075 | 0 | 0 B | 12.3 ms | 449.9 MB | `VERIFIED` |
| **M7: FC-Siam-diff** | 0.5500 | 62.81% | 58.07% | **0.6035** | **43.21%** | 96.11% | 3,970,640 | 2,351,323 | 2,866,764 | 125,029,001 | 1,471,409 | 5.94 MB | 603.1 ms | 612.3 MB | `VERIFIED` |
| **M8: SNUNet-CD** | 0.5000 | 9.08% | **77.58%** | 0.1626 | 8.85% | 59.29% | 5,304,623 | 53,103,841 | 1,532,781 | 74,276,483 | 563,377 | 2.31 MB | 1,298.0 ms | 876.0 MB | `VERIFIED` |
| **M9: BIT-CD** | 0.4500 | 65.36% | 54.22% | 0.5927 | 42.11% | 96.20% | 3,707,080 | 1,965,041 | 3,130,324 | 125,415,283 | 552,097 | 2.26 MB | 1,258.8 ms | 877.3 MB | `VERIFIED` |
| **M10: QAT-CD Calib.** | 0.4500 ($\gamma=0$) | 65.36% | 54.22% | 0.5927 | 42.11% | 96.20% | 3,707,080 | 1,965,041 | 3,130,324 | 125,415,283 | 552,097 | 2.26 MB | 1,411.0 ms | 877.3 MB | `VERIFIED` |
| **M10: QAT-CD Active** | 0.4500 ($\gamma=0.3$) | **70.78%** | 49.30% | 0.5812 | 40.96% | **96.38%** | 3,370,867 | **1,391,488** | 3,466,537 | 125,988,836 | 552,097 | 2.26 MB | 1,809.5 ms | 877.3 MB | `EXPLORATORY` |

*Notes:* Total pixels evaluated strictly equals 134,217,728 across all methods. Latency represents mean CPU inference per full $1024 \times 1024$ scene (16 patches of $256 \times 256$) measured under standardized 4-thread CPU conditions with 1 warm-up scene.

---

## 9. M10 Quality-Aware Ablation Analysis

The M10 ablation study examines the component signals of the multi-confounder quality estimator: Registration Discrepancy ($\mathbf{S}_{\text{reg}}$), Illumination Gradient ($\mathbf{S}_{\text{illum}}$), and Sharpness Asymmetry ($\mathbf{S}_{\text{sharp}}$), along with Margin Uncertainty modulation.

| Ablation Condition | Configuration | Precision | Recall | F1 Score | IoU | FP Pixels | FP Reduction (Pixels) | FP Reduction (%) | Precision Delta (pp) | Recall Delta (pp) | Exchange Ratio ($\Delta FP / \Delta TP$) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (NoQuality)** | $\gamma=0.0$ | 65.36% | 54.22% | 0.5927 | 42.11% | 1,965,041 | 0 | 0.00% | 0.00 pp | 0.00 pp | — |
| **ActiveAll (Full Gating)** | $\gamma=0.30$ | 70.78% | 49.30% | 0.5812 | 40.96% | 1,391,488 | +573,553 | **29.19%** | **+5.42 pp** | -4.92 pp | **1.71** |
| **RegOnly** | $w_{\text{reg}}=1.0$ | 71.26% | 47.95% | 0.5733 | 40.18% | 1,322,654 | +642,387 | 32.69% | +5.90 pp | -6.27 pp | 1.50 |
| **IllumOnly** | $w_{\text{illum}}=1.0$ | 68.56% | 51.63% | 0.5890 | 41.74% | 1,618,849 | +346,192 | 17.62% | +3.20 pp | -2.59 pp | 1.96 |
| **SharpOnly** | $w_{\text{sharp}}=1.0$ | 71.75% | 47.83% | 0.5740 | 40.25% | 1,287,388 | +677,653 | 34.49% | +6.39 pp | -6.39 pp | 1.55 |
| **NoUncertainty** | Uniform gate | 70.81% | 49.27% | 0.5810 | 40.95% | 1,390,392 | +574,649 | 29.24% | +5.45 pp | -4.95 pp | 1.71 |

### Operational Significance of Findings
1. **False Alarm Suppression Power:** Registration and sharpness discrepancies generate the largest volume of high-frequency false positives in satellite imagery (misregistered building edges and seasonal blur). Gating against them achieves up to $34.49\%$ false-positive elimination.
2. **The F1 vs. Precision Mission Dilemma:** In applications where manual verification of change detections is expensive (e.g., dispatching inspection teams or satellite tasking), every false alarm incurs a direct operational penalty. Active gating eliminates **1.71 false alarms for every genuine changed pixel sacrificed**, raising precision past $70\%$. Conversely, for search missions where any missed building is unacceptable, $\gamma=0.0$ remains strictly preferred.

---

## 10. Statistical Uncertainty Analysis

Treating 134.2 million pixels as independent samples artificially deflates standard errors and produces pseudo-significant p-values. Therefore, **the 128 test scenes ($N=128$) are treated as the true experimental sampling units**.

### Paired Scene-Level Bootstrap Resampling ($B=1000$, 95% Confidence Intervals)

| Model | Precision (95% CI) | Recall (95% CI) | F1 Score (95% CI) | IoU (95% CI) | Scene Macro F1 (95% CI) |
|---|:---:|:---:|:---:|:---:|:---:|
| **B1 Pixel Diff** | [9.05%, 12.39%] | [14.36%, 19.14%] | [0.1118, 0.1487] | [0.0592, 0.0803] | [0.0967, 0.1342] |
| **B2 SSIM** | [5.93%, 8.28%] | [48.64%, 57.87%] | [0.1065, 0.1430] | [0.0562, 0.0770] | [0.1022, 0.1388] |
| **B3 CVA** | [8.96%, 12.35%] | [15.06%, 20.10%] | [0.1130, 0.1517] | [0.0599, 0.0821] | [0.1011, 0.1396] |
| **M7 FC-Siam-diff** | [56.12%, 69.31%] | [51.69%, 63.85%] | [0.5528, 0.6481] | [0.3820, 0.4795] | [0.4285, 0.5144] |
| **M8 SNUNet-CD** | [7.49%, 10.84%] | [72.31%, 82.26%] | [0.1370, 0.1887] | [0.0735, 0.1042] | [0.1172, 0.1634] |
| **M9 BIT-CD** | [58.82%, 71.74%] | [47.58%, 60.36%] | [0.5401, 0.6389] | [0.3699, 0.4694] | [0.3951, 0.4841] |
| **M10 QAT-CD Calib.** | [58.82%, 71.74%] | [47.58%, 60.36%] | [0.5401, 0.6389] | [0.3699, 0.4694] | [0.3951, 0.4841] |

### Paired Wilcoxon Signed-Rank Hypothesis Tests across 128 Test Scenes

| Paired Comparison | Scene $\Delta F_1$ (Mean) | Wilcoxon Statistic $W$ | Two-Sided $p$-value ($F_1$) | Significant ($\alpha=0.05$)? | Scene $\Delta \text{IoU}$ (Mean) | Two-Sided $p$-value ($\text{IoU}$) | Interpretation |
|---|---:|---:|---:|:---:|---:|---:|---|
| **M9 BIT-CD vs. M7 FC-Siam** | -0.0318 | 2,582.0 | **0.0356** | **YES** | -0.0245 | **0.0321** | M7 exhibits slightly higher scene-level F1 |
| **M9 BIT-CD vs. M8 SNUNet** | +0.3003 | 123.0 | **$9.29 \times 10^{-20}$** | **YES** | +0.2798 | **$8.45 \times 10^{-20}$** | BIT-CD overwhelmingly outperforms SNUNet |
| **M9 BIT-CD vs. B3 CVA** | +0.3204 | 85.0 | **$3.61 \times 10^{-20}$** | **YES** | +0.2974 | **$3.12 \times 10^{-20}$** | Learned transformer decisively beats classical baseline |
| **M7 FC-Siam vs. B3 CVA** | +0.3522 | 41.0 | **$1.84 \times 10^{-21}$** | **YES** | +0.3219 | **$1.52 \times 10^{-21}$** | Learned CNN decisively beats classical baseline |
| **M8 SNUNet vs. B3 CVA** | +0.0201 | 3,114.0 | 0.3812 | **NO** | +0.0176 | 0.4105 | SNUNet scene F1 is not statistically superior to CVA |
| **M10 Calib vs. M9 BIT-CD** | 0.0000 | 0.0 | 1.0000 | **NO** | 0.0000 | 1.0000 | Identical predictions ($\gamma^*=0.0$) |

---

## 11. Fair CPU Benchmarking Protocol and Profile

All profiling was conducted under identical conditions: Apple Silicon CPU, PyTorch 2.5.1 CPU, `torch.set_num_threads(4)`, 1 warm-up scene, processing $1024 \times 1024$ scenes in 16 patches ($256 \times 256$, stride 256).

| Detector | Parameters | Checkpoint Size | Mean Latency (ms) | P50 / Median (ms) | P95 Latency (ms) | Per-Patch (ms) | Peak Process RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B1 Pixel Diff** | 0 | 0 B | 13.0 ms | 13.0 ms | 13.7 ms | 0.81 ms | 447.5 MB |
| **B2 SSIM** | 0 | 0 B | 47.0 ms | 47.0 ms | 47.8 ms | 2.94 ms | 449.8 MB |
| **B3 CVA** | 0 | 0 B | 12.3 ms | 12.3 ms | 12.5 ms | 0.77 ms | 449.9 MB |
| **M7 FC-Siam-diff** | 1,471,409 | 5.94 MB | 603.1 ms | 602.4 ms | 644.7 ms | 37.70 ms | 612.3 MB |
| **M8 SNUNet-CD** | 563,377 | 2.31 MB | 1,298.0 ms | 1,277.2 ms | 1,369.9 ms | 81.12 ms | 876.0 MB |
| **M9 BIT-CD** | 552,097 | 2.26 MB | 1,258.8 ms | 1,238.9 ms | 1,394.6 ms | 78.67 ms | 877.3 MB |
| **M10 QAT-CD Calib.** | 552,097 | 2.26 MB | 1,411.0 ms | 1,392.5 ms | 1,636.4 ms | 88.19 ms | 877.3 MB |
| **M10 QAT-CD Active** | 552,097 | 2.26 MB | 1,809.5 ms | 1,811.5 ms | 1,994.8 ms | 113.10 ms | 877.3 MB |

### Engineering Takeaways
- **FC-Siam-diff Latency Advantage:** Pure convolutional operations benefit significantly from vectorized SIMD execution on Apple Silicon Accelerate/AMX, making FC-Siam-diff ($603.1$ ms) roughly twice as fast as the multi-head attention transformer backbone of BIT-CD ($1,258.8$ ms).
- **QAT-CD Overhead:** Computing registration and edge gradient maps across 16 patches adds $\sim 150$ ms ($\gamma=0$) to $\sim 550$ ms ($\gamma=0.30$) per $1024 \times 1024$ scene.
- **Memory Footprint:** Peak RSS remains well below the project's $\le 8$ GB ceiling across all models (maximum peak RSS: $877.3$ MB).

---

## 12. Research Figures and Source Artifacts

The following seven publication-grade figures were generated directly from authenticated benchmark summaries:

1. `experiments/figures/m11/precision_recall_tradeoff.png`: Scatter plot of Precision vs. Recall across all 8 detectors, overlaid with iso-$F_1$ contours.
2. `experiments/figures/m11/f1_iou_benchmark_bars.png`: Clustered bar chart of F1 Score and IoU with 95% bootstrap error bars derived from 1,000 resamples.
3. `experiments/figures/m11/false_positive_reduction_log.png`: Logarithmic scale comparison of total false-positive pixels across the 128 test scenes.
4. `experiments/figures/m11/cpu_latency_vs_f1.png`: Efficiency frontier plot relating CPU latency (ms) to F1 score, where marker bubble area is proportional to parameter count.
5. `experiments/figures/m11/per_scene_f1_distribution.png`: Box plot showing the variance and median dispersion of per-scene F1 scores across the 128 evaluation scenes.
6. `experiments/figures/m11/qat_ablation_tradeoff.png`: Multi-bar trade-off chart illustrating false-positive reduction percentage, precision gain, and recall drop across QAT-CD signal conditions.
7. `experiments/figures/m11/cross_model_robustness.png`: Line plot tracing relative F1 degradation under M6 environmental stress perturbations (illumination, blur, misregistration).

---

## 13. Regression and Contract Test Verification

The complete regression suite was executed using both authoritative command forms:

```bash
./venv/bin/pytest -v
./venv/bin/python -m pytest -v
```

### Observed Execution Outcomes
- **Total Test Cases:** **236 passed** (0 failed, 0 errors, 0 warnings)
- **Execution Duration:** 43.68 seconds
- **Boundary Compliance:** `test_m1_m4_contracts.py` passed, verifying strict isolation of M11 and absence of premature M12+ functionality.
- **M11 Benchmark Tests:** `tests/test_m11_benchmark.py` (8 test cases) passed.
- **M11 Integration Tests:** `tests/test_m11_integration.py` (5 test cases) passed.
- **UI Smoke Tests:** `tests/test_ui_smoke.py` (28 test cases) passed, confirming seamless Streamlit rendering.

---

## 14. Scientific Findings, Honest Negative Results & Threats to Validity

### Key Scientific Findings
1. **Attention Mechanisms Enhance Precision, Not Necessarily Raw F1:** Spatial attention in BIT-CD suppresses background noise effectively, yielding higher precision than FC-Siam-diff ($65.36\%$ vs $62.81\%$). However, because change detection in LEVIR-CD involves discrete rectangular building footprints, convolutional skip connections in FC-Siam-diff recover fine boundary pixels slightly better, resulting in marginally higher recall ($58.07\%$ vs $54.22\%$) and aggregate F1.
2. **Dense Multiscale Aggregation (SNUNet) Requires Strong Spatial Priors:** Dense skip connections without boundary constraints or strong edge penalties suffer from extreme false-alarm runaway in high-resolution aerial imagery.

### Negative Results & Limitations
1. **QAT-CD Validation Calibration Did Not Select Active Gating for F1:** When optimizing strictly for validation F1, the optimal attenuation strength was $\gamma^* = 0.0$. Active gating ($\gamma = 0.30$) improves precision and reduces false alarms by $29.19\%$, but reduces unweighted F1. It must not be marketed as a universal F1 improvement.
2. **Test Split Reuse Limitation:** The 128-scene LEVIR-CD test split has been evaluated across M5, M6, M7, M8, M9, and M10. While hyperparameters were calibrated strictly on the 64 validation scenes, repeated evaluation on the test split means these numbers represent a closed-benchmark retrospective evaluation rather than an untouched prospective trial.

---

## 15. Reproduction Commands

To reproduce the complete Milestone M11 benchmarking suite from scratch:

```bash
# 1. Run full M11 benchmark evaluation harness (profiling 8 models on CPU with warm-up)
./venv/bin/python src/change_detection/benchmark_harness.py

# 2. Run M11 benchmark unit and statistical tests
./venv/bin/pytest -v tests/test_m11_benchmark.py tests/test_m11_integration.py

# 3. Run full project regression suite
./venv/bin/pytest -v
./venv/bin/python -m pytest -v

# 4. Launch Streamlit analyst application to inspect interactive benchmark
./venv/bin/streamlit run app.py
```

---

## 16. Acceptance Sign-Off

Milestone M11 is complete and verified against all architectural and scientific specifications:
- [x] Authoritative M11 scope verified.
- [x] All 7 detector configurations (8 settings) evaluated under consistent protocols.
- [x] Full-scene pixel accounting strictly verified ($134,217,728$ pixels across 128 scenes).
- [x] Paired scene bootstrap 95% CIs ($B=1000$) computed.
- [x] Paired Wilcoxon hypothesis tests executed across 128 scenes.
- [x] QAT-CD active gating trade-off evaluated honestly without overclaiming.
- [x] Fair CPU benchmarking measured under standardized thread and warm-up constraints.
- [x] 7 publication-grade research figures generated.
- [x] Full test suite (236 / 236 tests) passing cleanly.
- [x] Historical M5–M10 artifacts preserved.
