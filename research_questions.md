# Research Questions, Hypotheses & Formal Investigation Framework

**Milestone:** M3 — Research Definition, Literature Review & Baseline Formalization  
**Author:** Adishri Abro (Literature & Evaluation Lead)  
**Collaborators:** Rahul (Team Lead), Tanishka Mukhi (Dataset & Architecture Lead)  
**Document Version:** 2.0.0 (Grounded in Verified Academic Literature and M2 Dataset Metrics)

---

## 1. Research Problem Definition

Satellite imagery analysis in operational and exploratory contexts faces two interconnected core challenges:
1. **The Semantic Discovery Gap:** Remote sensing catalogs are predominantly indexed by geographic coordinates, acquisition timestamps, or broad land-use land-cover (LULC) tags. Locating scenes matching rich contextual descriptions (e.g., *"industrial warehouse expansion along riverbanks"*) requires cross-modal visual-semantic reasoning that bridges unstructured human language with aerial visual features `[VERIFIED LITERATURE: Lu et al., 2018; Radford et al., 2021]`.
2. **The Bi-Temporal False-Alarm Confounder:** Detecting real-world surface transformations between bi-temporal optical acquisitions ($T_1, T_2$) is corrupted by transient radiometric and geometric noise—including seasonal vegetation phenology, illumination variations from differing solar zenith angles, cloud shadows, and residual registration jitter `[VERIFIED LITERATURE: Singh, 1989; Bovolo & Bruzzone, 2007; Chen & Shi, 2020]`.

While state-of-the-art literature increasingly prioritizes multi-billion parameter transformer architectures evaluated exclusively on enterprise GPU clusters, real-world academic and institutional deployments frequently operate under strict commodity hardware constraints `[PROJECT DECISION: DEC-003]`. This research study investigates whether **rigorous, lightweight, and explainable models** can achieve meaningful cross-modal retrieval and robust change detection strictly under **CPU-constrained execution environments** ($\le 8$ GB RAM, quad/octa-core CPU, no GPU acceleration).

---

## 2. Formal Research Questions (RQs)

### RQ1 — Cross-Modal Semantic Retrieval
> **How effectively can vision-language embeddings retrieve semantically relevant satellite images from natural-language queries compared to classical text matching under zero-shot conditions?**
- *Focus:* Cross-modal alignment between free-form English captions and remote sensing imagery.
- *Core Investigation:* Evaluating whether a zero-shot dual-encoder model (pretrained CLIP ViT-B/32, Method A2) outperforms classical keyword/inverted-index text retrieval (Method A1) on the standardized RSICD test split under both Caption-to-Own-Image and Category-Level relevance protocols.
- *Literature Grounding:* Radford et al. (ICML 2021); Lu et al. (IEEE TGRS 2018); Liu et al. (IEEE TGRS 2024).

### RQ2 — Classical vs. Lightweight Learned Change Detection
> **How do classical image-comparison techniques and lightweight learned models compare for bi-temporal satellite-image change detection under CPU-only computational constraints?**
- *Focus:* Benchmarking traditional deterministic computer vision methods against lightweight neural architectures.
- *Core Investigation:* Comparing absolute pixel differencing (B1), Structural Similarity (SSIM) dissimilarity (B2), and Change Vector Analysis (CVA) (B3) against a lightweight Fully Convolutional Siamese Difference Network (`FC-Siam-diff`, B4) on LEVIR-CD, evaluated with leakage-safe threshold calibration.
- *Literature Grounding:* Singh (1989); Wang et al. (2004); Malila (1980); Daudt et al. (2018); Chen & Shi (2020).

### RQ3 — False-Alarm Suppression & Diagnostic Gating
> **Can diagnostic image-quality and consistency information be incorporated into change-detection outputs to reduce false-positive detections caused by non-ground environmental confounders?**
- *Focus:* Mitigating false detections induced by non-ground variations (illumination disparities, atmospheric blur, and geometric misregistration).
- *Core Investigation:* Evaluating whether a lightweight diagnostic gating layer (Method B5) suppresses spurious change alarms under controlled synthetic perturbations (illumination shift, blur, misregistration jitter) without severely degrading true change recall, quantified by the False Positive Amplification Factor (FPAF).
- *Literature Grounding:* Hall et al. (1991); Bovolo & Bruzzone (2007); Canty & Nielsen (2008); `literature_review.md` Sections 6 & 7.

### RQ4 — Accuracy-Efficiency Pareto Frontier on Commodity Hardware
> **What is the empirical trade-off between predictive performance and computational cost for the investigated approaches on standard CPU-based student hardware?**
- *Focus:* Rigorous hardware efficiency profiling.
- *Core Investigation:* Quantifying the empirical Pareto frontier spanning model parameter count, peak RAM consumption (MB), per-sample CPU inference latency (ms/tile), and task accuracy ($R@K$, MRR, F1-score, IoU).
- *Literature Grounding:* `metrics.md` Section 4; `decision_log.md` DEC-003.

---

## 3. Formulated Hypotheses & Test Criteria

> [!CAUTION]
> The statements below are formal **hypotheses** formulated for empirical testing. They are **not established facts** and must not be treated as conclusions until verified through reproducible experimentation in Milestones M4, M5, and M6.

### Hypothesis 1 (H1) — Retrieval Modality
*Dual-encoder vision-language embeddings (Method A2) will achieve statistically significant improvements in Caption-to-Own-Image Recall@1, Recall@5, and MRR over a classical BM25/keyword inverted index baseline (Method A1) on the RSICD test split.*
- **Experimental Test:** Experiment Suite A (A1 vs. A2 on 1,093 held-out RSICD test scenes, 5,465 queries).
- **Target Metrics:** $R@1, R@5, R@10$, MRR.
- **Required Evidence:** Measured $R@K(\text{A2}) > R@K(\text{A1})$ with non-overlapping confidence intervals across query tiers.

### Hypothesis 2 (H2) — Change Detection Architecture
*A lightweight Fully Convolutional Siamese Difference Network (`FC-Siam-diff`, B4) will achieve higher F1-score and IoU on the changed class than classical pixel-level (B1), structural (B2), and spectral vector (B3) baselines on LEVIR-CD, even when classical methods use optimal validation-calibrated thresholds.*
- **Experimental Test:** Experiment Suite B (B1, B2, B3, B4 evaluated on 128 held-out LEVIR-CD test scenes).
- **Target Metrics:** F1-score and IoU on the changed class.
- **Required Evidence:** Measured $F_1(\text{B4}) > \max(F_1(\text{B1}), F_1(\text{B2}), F_1(\text{B3}))$ under identical frozen validation thresholds.

### Hypothesis 3 (H3) — False-Alarm Suppression Gating
*Under controlled non-ground perturbations (illumination disparity, blur, misregistration jitter), incorporating a diagnostic gating layer (Method B5) will yield a lower False Positive Amplification Factor ($\text{FPAF} < \text{FPAF}_{\text{ungated}}$) and preserve higher relative F1 ($\Delta F_1(\text{B5}) < \Delta F_1(\text{ungated})$) compared to un-gated baselines.*
- **Experimental Test:** Experiment Suite C (controlled perturbation stress-testing across clean and corrupted test pairs).
- **Target Metrics:** $\text{FPAF} = \frac{\text{FP}_{\text{perturbed}}}{\text{FP}_{\text{clean}}}$ and Relative F1 Degradation $\Delta F_1$.
- **Required Evidence:** Demonstrated reduction in false alarm expansion ($\text{FPAF} \le 1.5$) under $\pm 20\%$ illumination disparity or 1-pixel misregistration.

### Hypothesis 4 (H4) — CPU Computational Feasibility
*A lightweight Siamese CNN backbone ($\le 5$M parameters) and zero-shot CLIP ViT-B/32 will execute within standard student laptop constraints ($\le 8$ GB peak RAM, $\le 2.0$ seconds per $256 \times 256$ tile inference, $\le 50$ ms per retrieval query) while maintaining competitive task accuracy.*
- **Experimental Test:** Experiment Suite E (Hardware profiling across all implemented methods).
- **Target Metrics:** Mean and p95 CPU Latency (ms), Peak RAM RSS (MB), Model Storage (MB).
- **Required Evidence:** Profiling logs confirming peak RAM $< 4$ GB and per-tile latency $< 2000$ ms on standard multi-core CPU.

---

## 4. Experimental Variables Framework

| Investigation Dimension | Independent Variables (Manipulated) | Dependent Variables (Measured) | Controlled Variables (Held Fixed) |
| :--- | :--- | :--- | :--- |
| **Semantic Retrieval (RQ1)** | - Retrieval Algorithm (A1 BM25 vs. A2 CLIP ViT-B/32)<br>- Prompt Template (Raw vs. Domain-Specific)<br>- Query Complexity Tier (Direct, Attribute, Relational) | - Recall@1, Recall@5, Recall@10<br>- Mean Reciprocal Rank (MRR)<br>- Query Search Latency (ms)<br>- Index RAM Size (MB) | - Fixed RSICD test split (1,093 images, 5,465 captions)<br>- Candidate gallery embeddings<br>- Hardware CPU allocation (quiet state) |
| **Change Detection (RQ2)** | - Algorithmic Class (B1 Diff, B2 SSIM, B3 CVA, B4 FC-Siam-diff)<br>- Threshold Policy (Validation-calibrated vs. Otsu)<br>- Input Tile Size ($256 \times 256$) | - Pixel-level Precision & Recall<br>- F1-score (Changed Class)<br>- Intersection-over-Union (IoU)<br>- Inference Latency (s/tile) | - Fixed LEVIR-CD test split (128 scene pairs / 2,048 patches)<br>- Zero-mean unit-variance image normalization<br>- No test-set threshold tuning |
| **False-Alarm Mitigation (RQ3)** | - Diagnostic Gating (Enabled B5 vs. Disabled B4)<br>- Perturbation Type: Illumination ($\alpha \in [0.7, 1.3]$), Blur ($\sigma \in [1, 2]$), Misregistration ($\Delta \in \{0.5, 1, 2\}$ px) | - False Positive Amplification Factor (FPAF)<br>- Relative F1 Degradation ($\Delta F_1$)<br>- Pixel False Positive Count (FP) | - Test scene pairs<br>- Base change probability map<br>- Evaluation script random seed |
| **Hardware Trade-off (RQ4)** | - Model Backbone & Parameter Count<br>- Precision (FP32 vs. Quantized INT8 if tested) | - Wall-clock Latency (Mean, p50, p95)<br>- Peak Heap & RSS Memory (MB)<br>- Throughput (tiles/sec) | - Host CPU architecture<br>- Background process daemon load<br>- Number of PyTorch execution threads |

---

## 5. Epistemic Classification: Facts, Assumptions, and Decisions

All claims across the project maintain strict separation across epistemic categories:

### Tier 1: Grounded Project Facts & Verified Data
1. `[PROJECT FACT]` The project scope encompasses cross-modal semantic retrieval and bi-temporal change detection under CPU constraints (approved B.Tech synopsis).
2. `[LOCALLY MEASURED]` RSICD dataset contains 10,921 valid $224 \times 224$ images and 54,605 captions (Karpathy split: 8,734 train / 1,094 val / 1,093 test).
3. `[LOCALLY MEASURED]` LEVIR-CD dataset contains 637 bi-temporal $1024 \times 1024$ pairs (Chen & Shi split: 445 train / 64 val / 128 test). Changed pixels constitute exactly $4.651\%$ of total pixels ($667,942,912$).
4. `[VERIFIED LITERATURE]` Overall Accuracy is fundamentally flawed for LEVIR-CD evaluation because a null model trivially scores $95.35\%$ OA.
5. `[VERIFIED LITERATURE]` The WHU Building Change Detection dataset is formally cited as *Ji et al., Remote Sensing, 2019, 11(11), 1343*, distinct from the mono-temporal extraction paper (*Ji et al., IEEE TGRS, 2019*).

### Tier 2: Confirmed Project Decisions
1. `[PROJECT DECISION: DEC-014]` Baselines A1 (BM25 keyword index) and A2 (Pretrained CLIP ViT-B/32 zero-shot) selected for retrieval.
2. `[PROJECT DECISION: DEC-015]` Baselines B1 (Pixel Diff), B2 (SSIM), B3 (CVA), B4 (`FC-Siam-diff`), and B5 (Quality-Gated) selected for change detection.
3. `[PROJECT DECISION: DEC-016]` Primary metrics fixed: Caption-to-Own-Image R@K and MRR for retrieval; F1-score and IoU for change detection.
4. `[PROJECT DECISION: DEC-017]` Thresholds must be calibrated strictly on validation data ($\tau^* = \arg\max_{\tau} F_1(\tau; \mathcal{D}_{\text{val}})$) or computed via unsupervised Otsu; test-set tuning is prohibited.
5. `[PROJECT DECISION: DEC-019]` The proposed quality-aware method is classified as an **ADAPTATION & COMBINATION**, with zero unsupported claims of foundational theoretical novelty.

### Tier 3: Hypotheses Under Investigation
1. `[HYPOTHESIS: H1]` Pretrained CLIP embeddings significantly outperform keyword inverted indices for remote sensing image retrieval without fine-tuning.
2. `[HYPOTHESIS: H2]` Lightweight Siamese CNNs outperform validation-tuned classical differencing on high-resolution building change detection.
3. `[HYPOTHESIS: H3]` Heuristic quality gating suppresses non-ground false alarms under controlled illumination and registration perturbations.
4. `[HYPOTHESIS: H4]` Both pipelines can execute within an 8 GB RAM and $<2$ second per tile CPU latency budget.
