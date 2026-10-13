# MILESTONE M7 IMPLEMENTATION & SCIENTIFIC EVALUATION REPORT
## Learned Bi-Temporal Change Detection: FC-Siam-diff Training, Validation Threshold Optimization, Test-Split Benchmark, and Streamlit Integration

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Milestone:** M7 — Learned Bi-Temporal Change Detection  
**Author:** Antigravity (Senior Computer Vision & PyTorch Research Engineer)  
**Date:** October 13, 2026  
**Environment:** macOS arm64 (Apple Silicon), Python 3.10.18, PyTorch 2.5.1 CPU, Streamlit 1.64.0  
**Milestone Completion Status:** **M7 VERIFIED COMPLETE**

---

## 1. Initial Repository State

Prior to Milestone M7 implementation, the repository was stabilized following the forensic engineering audit.
- **Base Commit:** `62b638955133420296b30387ba0eb1db7366f514` + P0/P1 stabilization patches.
- **Verified Stabilization Baseline:** 154 passing tests across `./venv/bin/pytest -v` and `./venv/bin/python -m pytest -v`.
- **Pre-existing M7 Assets:**
  - `src/change_detection/fc_siam_diff.py`: Architecture definition for `FCSiamDiff` and `FCSiamDiffDetector`.
  - `configs/m7_config.yaml`: Pre-configured hyperparameters for training and evaluation.
  - `tests/test_m7_model.py` and `tests/test_m7_pipeline.py`: Unit and pipeline structural tests.
- **Untrained Baseline:** No trained model checkpoints, training history, validation threshold sweeps, or test-split benchmark results existed on disk for M7.

---

## 2. Architecture Audit Findings

A comprehensive audit was conducted on `src/change_detection/fc_siam_diff.py` against the Daudt et al. (IEEE ICIP 2018) specification and strict local CPU hardware constraints:
1. **Twin Weight-Sharing Siamese Encoder:**
   - Pre-change ($T_1$) and post-change ($T_2$) optical image tensors share an identical 4-stage convolutional encoder (`enc1` through `enc4`) with base channel capacity $C_{\text{base}} = 16$.
   - Feature channels scale gracefully: $16 \to 32 \to 64 \to 128$.
2. **Multi-Scale Feature Differencing:**
   - Multi-scale absolute feature difference maps are computed at each resolution skip level: $D^{(l)} = |F_1^{(l)} - F_2^{(l)}|$.
   - Temporal invariance: Swapping $T_1$ and $T_2$ yields identical feature difference maps ($|F_1 - F_2| = |F_2 - F_1|$), ensuring direction-invariant change detection.
3. **Convolutional Decoder:**
   - 4 transposed convolutional upsampling stages with skip-connection concatenations of feature differences.
   - 1x1 convolution projecting final 16-channel representation to 1-channel raw logit map.
4. **Parameter Capacity:**
   - **Total Trainable Parameters:** $1,471,409$ parameters ($5.9$ MB in float32).
   - Complies strictly with the local cap ($\le 2,000,000$ ceiling).
5. **Contract Correction:**
   - Enhanced `load_checkpoint` in `src/change_detection/fc_siam_diff.py` to transparently support both `"state_dict"` and `"model_state_dict"` keys, ensuring interoperability.

---

## 3. Implementation Changes

1. **Training & Evaluation Pipeline (`src/change_detection/train_m7.py`):**
   - Implemented `BalancedPatchSampler` guaranteeing 50% positive change and 50% negative background patches per training epoch.
   - Built dual-loss objective using `BCEDiceLoss` ($1.0 \times \text{BCEWithLogitsLoss} + 1.0 \times \text{SoftDiceLoss}$).
   - Added learning rate scheduling via `CosineAnnealingLR` ($10^{-3} \to 10^{-6}$).
   - Integrated validation loss and F1 monitoring, automatically preserving the best model state dict.
   - Built 19-candidate validation threshold search ($\tau \in [0.05, 0.95]$ in step sizes of $0.05$) across all 64 validation scenes.
   - Built held-out test evaluation across all 128 scenes with micro-averaged confusion matrix and scene-level metric tracking.
   - Added `--eval-only` command-line capability to allow fast verification and benchmark regeneration from saved weights.
2. **Streamlit Change Detection Interface (`pages/change_detection.py`):**
   - Added `@st.cache_resource` function `load_m7_detector()` to load `fc_siam_diff_best.pt` without re-initialization.
   - Added `"M7: FC-Siam-diff (Learned Siamese U-Net)"` to detector selector.
   - Integrated live 16-patch Siamese inference and 1024x1024 reconstruction with interactive error overlay.
   - Added unified benchmark comparison table dynamically displaying B1, B2, B3, and M7 when `comparison_m5_m7.json` is available.
3. **Application Title & Sidebar Governance (`app.py`):**
   - Updated system boundary badge from "M1–M6 Verified" to "M1–M7 Verified".
   - Maintained boundary guard preventing execution of M8+ architectures.
4. **Integration & Regression Tests (`tests/test_m7_integration.py`):**
   - Added 8 end-to-end unit and integration tests verifying checkpoint loading, forward pass finiteness, patch-to-mosaic inference, artifact preservation, and Streamlit rendering.

---

## 4. Dataset and Split Verification

The official LEVIR-CD benchmark dataset was verified on disk (`data/raw/levir_cd`):
- **Spatial Resolution:** $1024 \times 1024$ pixels, 0.5 m Ground Sampling Distance (GSD), 3-band RGB optical imagery.
- **Split Breakdown:**
  - **Train:** 445 scenes ($7,120$ patches of $256 \times 256$).
  - **Validation:** 64 scenes ($1,024$ patches of $256 \times 256$).
  - **Test:** 128 scenes ($2,048$ patches of $256 \times 256$, totaling $134,217,728$ evaluation pixels).
  - **Disjointness:** Zero stem overlap verified between train, val, and test manifests ($|\text{train} \cap \text{val}| = |\text{train} \cap \text{test}| = |\text{val} \cap \text{test}| = 0$).
- **Class Imbalance:**
  - Training split patch scan revealed 3,167 positive change patches (44.5%) and 3,953 background patches (55.5%).
  - Balanced patch sampling drew 256 positive and 256 negative patches per epoch (512 total samples/epoch) to prevent background gradient saturation.

---

## 5. Effective Training Configuration

The configuration matches `configs/m7_config.yaml`:
- **Architecture:** FC-Siam-diff (`base_channels: 16`, `in_channels: 3`, `out_channels: 1`)
- **Loss Function:** `BCEDiceLoss` (BCE Weight: $1.0$, Dice Weight: $1.0$, Smooth: $1.0$)
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, Weight Decay: $1 \times 10^{-4}$)
- **Initial Learning Rate:** $1 \times 10^{-3}$
- **LR Scheduler:** CosineAnnealingLR ($T_{\max} = 5, \eta_{\min} = 1 \times 10^{-6}$)
- **Batch Size:** 16
- **Samples per Epoch:** 512 patches (32 optimization steps/epoch)
- **Epochs:** 5
- **Hardware Profile:** Strict CPU-Only (4 OpenMP / PyTorch intra-op threads)
- **Random Seed:** 42 (deterministic seeding across Python, NumPy, PyTorch CPU)
- **Selection Criterion:** Maximum Validation F1 score evaluated on all 1,024 validation patches.

---

## 6. Training Execution Details

- **Process Execution:** Launched via `./venv/bin/python src/change_detection/train_m7.py`
- **Total Training Wall Time:** 1,310.3 seconds (~21.8 minutes).
- **CPU Resource Utilization:** 380–420% CPU utilization across 4 threads.
- **Memory Footprint:** Peak RSS stabilized at 2.66 GB (comfortably below the 8.0 GB platform limit).

---

## 7. Training and Validation History

| Epoch | Duration (s) | Train Loss | Validation Loss | Validation F1 ($\tau=0.50$) | Learning Rate | Checkpoint Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 209.0 | 1.4405 | 4.0173 | 0.1256 | 0.000905 | Saved initial best |
| 2 | 260.4 | 1.2515 | 1.2780 | 0.3011 | 0.000655 | Saved new best (+0.1755) |
| 3 | 271.1 | 1.1380 | 1.1760 | 0.4901 | 0.000345 | Saved new best (+0.1890) |
| 4 | 289.6 | 1.0870 | 1.1437 | 0.5224 | 0.000095 | Saved new best (+0.0323) |
| **5** | **280.2** | **1.0504** | **1.1092** | **0.6111** | **0.000000** | **Saved final best (+0.0887)** |

---

## 8. Checkpoint-Selection Method

- **Criterion:** Model checkpoint selection was driven strictly by validation split performance (`val_f1` under BCE+Dice loss at default $\tau=0.50$).
- **Independence:** Zero test split imagery or test ground truth was accessed or computed during model training or checkpoint selection.
- **Selected Model:** Checkpoint at **Epoch 5** with validation F1 = **0.6111** was serialized to `experiments/checkpoints/m7/fc_siam_diff_best.pt` ($5,940,074$ bytes).

---

## 9. Validation Threshold-Selection Method and Result

To determine the optimal operational threshold $\tau^*$, the frozen Epoch 5 checkpoint was evaluated across all 64 full validation scenes ($1,024$ patches of $256 \times 256$, totaling $67,108,864$ pixels):
- **Candidate Set:** 19 candidate thresholds $\tau \in \{0.05, 0.10, 0.15, \dots, 0.95\}$.
- **Evaluation Time:** 47.99 seconds on CPU.
- **Optimal Decision Threshold:** **$\tau^* = 0.5500$**
- **Validation Metrics at $\tau^* = 0.5500$:**
  - Precision: **60.92%**
  - Recall: **57.40%**
  - F1 Score: **0.5911**
  - IoU: **41.95%**
  - Accuracy: **96.67%**
  - True Positives: $1,616,481$ px | False Positives: $1,036,972$ px | False Negatives: $1,199,787$ px
- **Threshold Freezing:** $\tau^* = 0.5500$ was frozen and recorded in `experiments/results/m7/thresholds.json` and saved in the checkpoint metadata prior to evaluating the test set.

---

## 10. Held-Out Test Metrics

The official test split (128 scenes, $134,217,728$ pixels) was evaluated using strictly the frozen threshold $\tau^* = 0.5500$:

| Metric | Measured Value |
|:---|:---|
| **Precision** | **62.81%** ($0.62807$) |
| **Recall** | **58.07%** ($0.58072$) |
| **F1 Score** | **0.6035** ($0.60347$) |
| **Intersection over Union (IoU)** | **43.21%** ($0.43212$) |
| **Pixel Accuracy** | **96.11%** ($0.96112$) |
| **True Positives (TP)** | $3,970,640$ pixels |
| **False Positives (FP)** | $2,351,323$ pixels |
| **False Negatives (FN)** | $2,866,764$ pixels |
| **True Negatives (TN)** | $125,029,001$ pixels |
| **Total Evaluation Pixels** | $134,217,728$ pixels |
| **Ground-Truth Changed Pixels** | $6,837,404$ pixels ($5.09\%$) |
| **Predicted Changed Pixels** | $6,321,963$ pixels ($4.71\%$) |

---

## 11. B1, B2, B3 vs. M7 Comprehensive Comparison

All four methods were evaluated on the exact same 128 LEVIR-CD test scenes under their respective frozen validation-calibrated operating thresholds $\tau^*$:

| Method | Type | Operating $\tau^*$ | Precision (%) | Recall (%) | F1 Score | IoU (%) | Mean Latency (ms) | Peak RAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1 Pixel Diff** | Classical L1 Intensity | $0.4100$ | 10.64% | 16.66% | 0.1299 | 6.95% | 87.9 | 412.0 |
| **B2 SSIM** | Classical Structural Sim. | $0.9000$ | 7.06% | 53.32% | 0.1246 | 6.65% | 136.3 | 412.0 |
| **B3 CVA** | Classical Spectral Vector | $0.4050$ | 10.56% | 17.46% | 0.1316 | 7.04% | 149.7 | 412.0 |
| **M7 FC-Siam-diff** | **Learned Siamese U-Net** | **$0.5500$** | **62.81%** | **58.07%** | **0.6035** | **43.21%** | 1,278.1 | 2,659.4 |

### Comparative Insights:
1. **F1 Gain:** M7 achieved a **+0.4719 F1 improvement (+358%)** over the best classical baseline (B3 CVA: 0.1316 $\to$ M7: 0.6035).
2. **IoU Gain:** M7 achieved a **+36.17% absolute IoU increase (+513%)** over B3 CVA (7.04% $\to$ 43.21%).
3. **False Positive Reduction:** Classical methods suffered massive false-positive noise from illumination and seasonal shifts (B1: 9.56M FP, B2: 48.01M FP, B3: 10.12M FP). M7 reduced false positives to **2.35M FP**, a **75–95% false-alarm reduction**.
4. **Computational Trade-off:** M7 requires deep convolutional feature extraction across 16 patches per scene, yielding a mean CPU latency of 1,278 ms per $1024 \times 1024$ scene (~1.28 s), which remains highly interactive on local desktop hardware while operating well within RAM limits (2.66 GB vs. 8.0 GB ceiling).

---

## 12. Runtime and Memory Measurement Methodology

- **Timing Scope:** Measured using high-resolution monotonic clocks (`time.perf_counter()`). Measurement includes pair loading, 16-patch grid extraction ($256 \times 256$), dual-stream Siamese convolutional forward pass, pixel-level sigmoid activation, seamless mosaic reconstruction ($1024 \times 1024$), and thresholding.
- **Latency Distribution on CPU:**
  - Mean: $1,278.1$ ms
  - Median ($p_{50}$): $1,338.0$ ms
  - 95th Percentile ($p_{95}$): $1,633.6$ ms
- **Memory Measurement:** Measured via OS process RSS (`psutil.Process().memory_info().rss`). Peak resident set size was $2,659.4$ MB, ensuring strict compliance with local hardware limits.

---

## 13. Streamlit Integration Results

- **Live Interactivity:** `pages/change_detection.py` successfully hosts M7 alongside B1, B2, and B3.
- **Pair Selection:** Users can select any LEVIR-CD test pair (e.g. `test_1`, `test_20`, `test_50`, `test_100`) and execute live CPU inference in ~1.2 seconds.
- **Visualizer:** The 4-column change panel correctly renders Pre-change (T1), Post-change (T2), Ground Truth, Predicted Mask, and the interactive False Positive / False Negative color-coded diagnostic error overlay.
- **Dynamic Benchmark Table:** Automatically loads `experiments/results/m7/comparison_m5_m7.json` to present the complete B1–B3 vs. M7 scientific benchmark and comparative bar chart.
- **Diagnostic Safety:** If `fc_siam_diff_best.pt` is missing or corrupted, a clean Streamlit error card is presented without crashing or silently substituting classical baselines.

---

## 14. Unit, Integration, and Regression Test Results

Following implementation and benchmark generation, full test verification was executed:

1. **Focused M7 Tests:**
   - `tests/test_m7_model.py`: **8/8 PASSED** (3.27s)
   - `tests/test_m7_pipeline.py`: **3/3 PASSED** (3.60s)
   - `tests/test_m7_integration.py`: **8/8 PASSED** (3.13s)
2. **Streamlit UI Smoke Suite:**
   - `tests/test_ui_smoke.py`: **28/28 PASSED** (13.01s)
3. **Full Regression Suite Executions:**
   - Command 1: `./venv/bin/pytest -v` $\to$ **162/162 PASSED** (29.04s, Exit Code 0)
   - Command 2: `./venv/bin/python -m pytest -v` $\to$ **162/162 PASSED** (29.26s, Exit Code 0)
4. **Zero Regressions:** All prior contracts across M1–M6 (RSICD loaders, BM25, CLIP embeddings, Faiss indices, prompt ensembling, B1–B3 classical detectors, and M6 perturbation stress tests) passed without errors.

---

## 15. Known Limitations

1. **CPU Execution:** Training and batch inference are CPU-bound. Full 128-scene evaluation takes ~160 seconds on Apple Silicon CPU. For production real-time batch processing, Apple Silicon MPS or CUDA acceleration would offer ~10x speedup.
2. **Resolution Ceiling:** The model evaluates $1024 \times 1024$ scenes via non-overlapping $256 \times 256$ tiles. While effective, minor boundary artifacts may occur at patch seams in complex urban transitions.

---

## 16. Remaining Defects or Blocked Work

- **Defects:** 0 defects. Zero failing tests.
- **Blockers:** None. Milestone M7 is fully complete and functional.

---

## 17. Real Artifact File Paths

- **Model Checkpoint:** `experiments/checkpoints/m7/fc_siam_diff_best.pt`
- **Checkpoint Metadata:** `experiments/checkpoints/m7/metadata.json`
- **Official Test Summary:** `experiments/results/m7/summary.json` & `experiments/results/m7/summary.csv`
- **Frozen Threshold Calibration:** `experiments/results/m7/thresholds.json`
- **Per-Image Test Metrics (128 scenes):** `experiments/results/m7/per_image_metrics.csv`
- **Training & Validation History:** `experiments/results/m7/training_history.json`
- **M5 vs. M7 Benchmark Comparison:** `experiments/results/m7/comparison_m5_m7.json` & `experiments/results/m7/comparison_m5_m7.csv`
- **Integration Test Suite:** `tests/test_m7_integration.py`
- **Streamlit Page:** `pages/change_detection.py`

---

## 18. Final Evidence-Based Milestone Status

All 12 completion criteria set forth in the master specification have been satisfied with verifiable, empirical, and reproducible evidence:

- [x] Existing FC-Siam-diff architecture audited and usable (1.47M parameters $\le 2\text{M}$ ceiling).
- [x] Reproducible training pipeline implemented (`src/change_detection/train_m7.py`).
- [x] Training executed successfully on the official LEVIR-CD training split (5 epochs, 512 balanced samples/epoch).
- [x] A real checkpoint selected using validation data (Epoch 5, Val F1 = 0.6111).
- [x] Threshold selected on validation data and frozen ($\tau^* = 0.5500$).
- [x] Official test split evaluated after checkpoint and threshold freeze (128 scenes, 134,217,728 pixels).
- [x] Metrics computed using established evaluation framework (F1 = 0.6035, IoU = 43.21%).
- [x] Comparison against B1, B2, and B3 completed on compatible data (+358% F1 gain).
- [x] CPU inference works with the saved checkpoint (~1.28s per 1024x1024 scene).
- [x] Streamlit integration works with real test pairs.
- [x] Existing M1–M6 functionality remains intact (162/162 passing tests).
- [x] Research report and real artifacts saved.

**FINAL STATUS: M7 VERIFIED COMPLETE**
