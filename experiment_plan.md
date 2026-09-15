# Empirical Experiment & Evaluation Plan

**Document Version:** 1.0.0 (Milestone 1 — Baseline Specification)  
**Author:** Rahul (Team Lead)  
**Research Focus:** Rigorous, reproducible comparative study under CPU constraints  

---

## 1. Experimental Philosophy & Standards

To ensure that the project remains a scientifically rigorous research endeavor:
1. **Null Hypothesis Stance:** We assume no prior superiority for any proposed technique over classical baselines. All performance claims must be demonstrated through reproducible empirical evidence.
2. **Identical Evaluation Protocols:** All competing algorithms within a capability suite must be evaluated on the exact same dataset splits, test instances, and evaluation metrics.
3. **Hardware Standardization:** All timing and memory benchmarks must be recorded on standard CPU hardware under quiet operating conditions (minimal background tasks), documenting the CPU model, core allocation, and RAM.
4. **No Metric Fabrication:** No accuracy values, F1-scores, or latencies will be documented until benchmark scripts execute and generate empirical log files.

---

## 2. Experiment Suite A: Semantic Satellite Image Retrieval

### 2.1 Candidate Configurations
| ID | Method Class | Candidate Architecture / Strategy | Embedding Source / Mechanism | Status |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | Simple Baseline | Keyword / Bag-of-Words / Metadata Matching | Inverted index on RSICD captions or categorical land-cover tags | Candidate Baseline |
| **A2** | Vision-Language Embedding | Dual-Encoder (e.g., Pretrained CLIP ViT-B/32 or ResNet50) + FAISS-CPU | Zero-shot image and text projections into shared Euclidean space | Candidate Primary |
| **A3** | Improved / Reranked Retrieval | Prompt Engineering + Cross-Modal Score Reranking (or domain-adapted weights if verified feasible) | Text prompt ensemble + secondary feature similarity fusion | Proposed / Conditional |

### 2.2 Evaluation Protocol
- **Benchmark Dataset:** RSICD (Proposed / To Be Verified).
- **Test Set:** A fixed, held-out test split containing image-caption pairs not observed during any fine-tuning or threshold calibration.
- **Query Set:** A curated test bank of unconstrained natural-language queries spanning three difficulty tiers:
  - *Tier 1 (Direct Category):* Single land-use terms (e.g., *"airport runway"*, *"dense residential"*).
  - *Tier 2 (Attribute + Scene):* Compositional queries (e.g., *"circular green storage tanks near highway"*).
  - *Tier 3 (Spatial & Relational):* Complex contextual queries (e.g., *"areas with newly built-up zones near rivers"*).
- **Target Metrics:**
  - **Precision@K ($K \in \{1, 5, 10\}$):** Proportion of top-$K$ retrieved images that are semantically relevant to the query.
  - **Recall@K ($K \in \{1, 5, 10\}$):** Fraction of all ground-truth relevant images retrieved in the top-$K$.
  - **Mean Reciprocal Rank (MRR):** $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$, measuring the rank of the first relevant result.
  - **Query Latency (ms):** Mean and standard deviation of search time per query over 100 iterations on CPU.
  - **Index RAM Footprint (MB):** Memory allocated by the FAISS index structure.

---

## 3. Experiment Suite B: Bi-Temporal Change Detection

### 3.1 Candidate Configurations
| ID | Method Class | Candidate Model / Algorithm | Feature Comparison Mechanism | Status |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | Classical Deterministic | Absolute Pixel Differencing (Grayscale / RGB) | $|X_{T2} - X_{T1}|$ with automated Otsu thresholding | Candidate Baseline 1 |
| **B2** | Classical Structural | Structural Similarity Index Measure (SSIM) | Local luminance, contrast, and structural variance degradation | Candidate Baseline 2 |
| **B3** | Classical Spectral | Change Vector Analysis (CVA) | Spectral magnitude $|\Delta \vec{\rho}|$ and direction analysis | Candidate Baseline 3 (Conditional) |
| **B4** | Lightweight Learned | Lightweight Siamese CNN (e.g., Tiny-UNet backbone) | Siamese feature extraction followed by $L_1$/cosine difference layers | Candidate Learned |
| **B5** | Hybrid / Quality-Aware | Method B4 + Diagnostic Gating Layer | Learned change map weighted by radiometric consistency confidence | Proposed / Conditional |

### 3.2 Evaluation Protocol
- **Benchmark Dataset:** LEVIR-CD (Proposed / To Be Verified, tiled to $256 \times 256$ sub-crops).
- **Test Set:** Standardized test split containing labeled binary ground-truth change masks ($Y \in \{0, 1\}^{H \times W}$, where $1 = \text{Change}, 0 = \text{No-Change}$).
- **Target Metrics:**
  - **Pixel-Level Precision:** $\frac{TP}{TP + FP}$
  - **Pixel-Level Recall:** $\frac{TP}{TP + FN}$
  - **F1-Score (Change Class):** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - **Intersection-over-Union (IoU):** $\frac{TP}{TP + FP + FN}$
  - **Overall Accuracy (OA):** $\frac{TP + TN}{TP + TN + FP + FN}$ (monitored with caution due to class imbalance where non-change dominates).
  - **Inference Latency (seconds/tile):** Per-tile wall-clock execution time on CPU.

---

## 4. Experiment Suite C: False-Alarm & Robustness Perturbation Analysis

To address the specific challenge outlined in the project synopsis regarding false positives, we design a controlled perturbation test suite applied to non-changing and changing test pairs.

### 4.1 Confounder Perturbations
1. **Global Illumination Disparity:**
   - *Simulation:* Uniform scaling of image brightness/gamma on $T_2$: $X'_{T2} = \alpha X_{T2} + \beta$, where $\alpha \in [0.7, 1.3], \beta \in [-30, 30]$.
   - *Measurement:* Increase in False Positive Rate (FPR) on static scenes as a function of $\Delta \text{luminance}$.
2. **Simulated Cloud & Shadow Artifacts:**
   - *Simulation:* Synthetic localized high-reflectance white patches (cloud masks) and localized negative luminance shifts (shadow masks).
   - *Measurement:* Localized false-positive clustering in non-ground change zones.
3. **Geometric Misregistration Jitter:**
   - *Simulation:* Rigid sub-pixel and integer translations of $T_2$ relative to $T_1$ ($\Delta x, \Delta y \in \{0.5, 1.0, 2.0, 3.0\}\text{ pixels}$).
   - *Measurement:* Edge-boundary false-positive dilation rate comparing classical pixel differencing vs. learned deep representations.
4. **Seasonal Phenology Shift:**
   - *Investigation:* Evaluating pairs exhibiting natural seasonal vegetation color shifts (dry season vs. wet season) to assess whether vegetation color variance is erroneously classified as structural change.

---

## 5. Experiment Suite D: Systematic Ablation Studies

To rigorously evaluate the contribution of individual architectural components in proposed methods:

| Ablation ID | Target Pipeline | Configuration Tested | Hypothesis Being Tested |
| :--- | :--- | :--- | :--- |
| **ABL-1** | Change Detection (B5) | Complete Hybrid Pipeline (Learned + Quality Gating) | Full proposed configuration baseline |
| **ABL-2** | Change Detection (B5) | Remove Quality Gating Layer (Learned model only, B4) | Measures exact false-positive reduction attributable to quality gating |
| **ABL-3** | Preprocessing (P3) | Disable Radiometric Alignment (Evaluate B1, B2, B4 without histogram matching) | Measures whether classical normalization improves or degrades downstream models |
| **ABL-4** | Semantic Retrieval (A2) | Standard Prompts vs. Remote-Sensing Domain-Specific Prompts | Measures sensitivity of zero-shot CLIP retrieval to remote-sensing prompt syntax |

---

## 6. Computational Efficiency & Hardware Profiling Methodology

All experiments will incorporate automated resource monitoring:
- **Timer:** High-resolution wall-clock profiling using `time.perf_counter()` after a warm-up pass.
- **Memory Profiling:** Tracking resident set size (RSS) via `psutil.Process().memory_info().rss` and peak allocation via Python's `tracemalloc`.
- **Model Footprint:** Counting trainable parameters ($\sum \theta$) and serializing weights to disk to measure storage footprint in megabytes (MB).
- **Reporting Format:** Results will be logged directly into structured CSV/JSON run reports and summarized in comparative Markdown tables for academic review.
