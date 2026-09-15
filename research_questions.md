# Research Questions, Hypotheses & Investigation Framework

> [!IMPORTANT]
> **Working Status Declaration:**  
> These research questions, hypotheses, variables, and evaluation criteria are provisional and establish the empirical baseline for Milestone 1. They are subject to refinement and formal sharpening following the comprehensive literature review in Milestone 3.

---

## 1. Research Problem Definition

Satellite imagery analysis in operational and exploratory settings is hindered by two interconnected limitations:
1. **The Semantic Discovery Gap:** Most remote sensing catalogs are indexed by geographic coordinates, timestamps, or coarse land-use land-cover (LULC) tags. Locating scenes matching rich contextual descriptions (e.g., *"industrial warehouse expansion along riverbanks"*) requires cross-modal visual-semantic reasoning that bridges free-form human language with aerial visual features.
2. **The Bi-Temporal False-Alarm Confounder:** Detecting true surface changes between bi-temporal optical acquisitions ($T_1, T_2$) is severely degraded by transient radiometric and geometric noise—including seasonal vegetation color shifts, illumination variations from differing solar zenith angles, cloud shadows, and residual registration errors.

Furthermore, state-of-the-art computer vision models for these tasks increasingly prioritize multi-billion parameter architectures evaluated exclusively on high-end GPUs. This project investigates whether **rigorous, lightweight, and explainable models** can deliver meaningful cross-modal retrieval and robust change detection strictly under **CPU-constrained execution environments** typical of standard student and low-resource institutional hardware.

---

## 2. Working Research Questions (RQs)

### RQ1 — Semantic Retrieval
> **How effectively can vision-language embeddings retrieve semantically relevant satellite images from natural-language queries?**
- *Focus:* Cross-modal alignment between free-form English queries and remote sensing imagery.
- *Core Investigation:* Evaluating whether zero-shot or lightweight pre-trained vision-language representations (e.g., standard CLIP or remote-sensing adapted variants) outperform classical text-tag/keyword matching baselines when searching satellite scenes with complex spatial and semantic composition.

### RQ2 — Change Detection
> **How do classical image-comparison techniques and lightweight learned models compare for bi-temporal satellite-image change detection under CPU-only computational constraints?**
- *Focus:* Benchmarking traditional deterministic computer vision methods against lightweight neural architectures.
- *Core Investigation:* Comparing absolute image differencing, Structural Similarity (SSIM), and Change Vector Analysis (CVA) against compact deep-learning models (e.g., lightweight Siamese CNN) on standard building/land-cover change benchmarks, tracking predictive accuracy versus CPU execution time.

### RQ3 — False-Positive Reduction
> **Can image-quality and temporal information be incorporated into change-detection results to reduce false-positive detections caused by non-ground changes?**
- *Focus:* Mitigating false detections induced by non-ground variations (illumination, seasonal vegetation changes, cloud shadows, and misregistration).
- *Core Investigation:* Evaluating whether heuristic quality indicators, radiometric consistency checks, or confidence-weighting mechanisms can filter spurious change alarms without severely compromising true change recall.

### RQ4 — Computational Trade-off
> **What is the trade-off between predictive performance and computational cost for the investigated approaches on standard CPU-based student hardware?**
- *Focus:* Hardware efficiency profiling.
- *Core Investigation:* Quantifying the Pareto frontier across model footprint (MB), peak RAM consumption, per-sample inference latency (seconds on CPU), and detection performance (F1/IoU/Precision@K).

---

## 3. Working Research Hypotheses (Provisional)

> [!CAUTION]
> The statements below are formal **hypotheses** formulated for empirical testing. They are **not established facts** and must not be treated as conclusions until verified through reproducible experimentation.

- **Hypothesis 1 (H1):**  
  *Vision-language embedding-based retrieval may provide more semantically relevant results for natural-language satellite-image queries than a simple keyword/category-based retrieval baseline.*
  - *Rationale:* Dual-encoder vision-language architectures map visual tokens and linguistic tokens into a continuous geometric manifold, potentially capturing spatial relationships and descriptive nuances absent from discrete tag matching.

- **Hypothesis 2 (H2):**  
  *A lightweight learned change-detection model may achieve better change-detection performance than purely pixel-level comparison methods on the selected evaluation data.*
  - *Rationale:* Learned convolutional filters extract hierarchical structural and contextual representations that can generalize across minor radiometric discrepancies better than rigid pixel-level subtraction.

- **Hypothesis 3 (H3):**  
  *Incorporating image-quality and temporal-consistency information into change-detection outputs may reduce false-positive detections while maintaining acceptable change-detection performance.*
  - *Rationale:* Explicitly modeling confounding variables (e.g., uniform illumination gradients or local shadow distributions) can act as an orthogonal gating mechanism on raw change heatmaps.

- **Hypothesis 4 (H4):**  
  *Some lightweight approaches may achieve a useful balance between predictive performance and computational cost suitable for CPU-only execution.*
  - *Rationale:* Judicious model sizing (e.g., parameter counts $< 5$M, input resolutions tiled at $256 \times 256$, and vector quantization) can preserve high detection accuracy while maintaining reasonable per-tile latency on modern quad-core CPUs.

---

## 4. Experimental Variables & Conceptual Metrics

### 4.1 Variables
| Dimension | Independent Variables (Manipulated) | Dependent Variables (Measured) | Controlled Variables |
| :--- | :--- | :--- | :--- |
| **Semantic Retrieval** | - Query complexity & phrasing<br>- Model backbone (Baseline vs. CLIP)<br>- Embedding dimension<br>- Indexing strategy (Flat vs. IVF) | - Precision@K ($K \in \{1, 5, 10\}$)<br>- Recall@K<br>- Mean Reciprocal Rank (MRR)<br>- Query Latency (ms)<br>- Index Size (MB) | - Dataset split (fixed RSICD test split)<br>- Hardware CPU specification<br>- Background process load |
| **Change Detection** | - Algorithmic class (Differencing, SSIM, CVA, Siamese CNN)<br>- Decision threshold ($\tau$)<br>- Input tile resolution ($256 \times 256$) | - Pixel-level Precision<br>- Pixel-level Recall<br>- F1-score (Change class)<br>- Intersection-over-Union (IoU)<br>- Inference Latency (s/tile) | - Test image pairs (fixed LEVIR-CD split)<br>- Co-registration alignment<br>- Normalization pipeline |
| **False-Alarm Mitigation** | - Quality gating strategy (enabled vs. disabled)<br>- Perturbation type & intensity (illumination shift, blur, misregistration jitter) | - False Positive Rate (FPR)<br>- Area Under ROC (AUROC)<br>- Precision-Recall trade-off curve | - Baseline detector backbone<br>- Test scene geographical bounds |

---

## 5. Epistemic Classification: Facts, Assumptions, Proposals, and Unknowns

To maintain strict scientific integrity, all statements within this project are categorized into four epistemic tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EPISTEMIC TAXONOMY                              │
├──────────────────────┬─────────────────────────────────────────────────┤
│ Tier 1: Synopsis     │ Explicit requirements approved in project       │
│        Facts         │ synopsis (dual capabilities, CPU focus).        │
├──────────────────────┼─────────────────────────────────────────────────┤
│ Tier 2: Verified     │ Peer-reviewed findings verified in academic     │
│        Literature    │ literature (to be expanded in Milestone 3).     │
├──────────────────────┼─────────────────────────────────────────────────┤
│ Tier 3: Working      │ Plausible premises adopted to progress system   │
│        Assumptions   │ design before empirical testing.               │
├──────────────────────┼─────────────────────────────────────────────────┤
│ Tier 4: Proposals    │ Candidate methods, datasets, or architectures   │
│        & Unknowns    │ awaiting formal verification or experimentation.│
└──────────────────────┴─────────────────────────────────────────────────┘
```

### Explicit Synopsis Facts
1. The project must investigate both Semantic Satellite Image Search and Bi-temporal Change Detection.
2. The project must investigate false positives resulting from clouds, shadows, seasonal shifts, illumination, and registration disparities.
3. The computational constraint is standard student laptops with CPU-only execution.
4. The team comprises Rahul (Lead), Tanishka Mukhi, and Adishri Abro across 15 structured milestones.

### Working Assumptions
1. Pretrained vision-language models trained on general web data retain sufficient zero-shot representation capacity for aerial remote sensing images without requiring heavy end-to-end retraining.
2. High-resolution satellite images can be partitioned into smaller overlapping or non-overlapping patches ($256 \times 256$) to execute within laptop memory limits without critical boundary artifact loss.
3. Bi-temporal evaluation datasets (e.g., LEVIR-CD) have adequate co-registration precision in their raw state to permit initial baseline benchmarking.

### Proposed Directions & Current Unknowns (To Be Verified in M2/M3)
1. *Unknown:* Does standard OpenAI CLIP ViT-B/32 or ResNet50 fit comfortably in CPU memory during batch inference, or will a quantized/distilled model (e.g., MobileCLIP) be required?
2. *Unknown:* Does LEVIR-CD contain significant natural seasonal variations, or is it predominantly building-focused with relatively uniform background conditions?
3. *Unknown:* What is the empirical degradation slope of classical pixel differencing when two images experience a 1-pixel or 2-pixel registration shift?
4. *To Be Verified:* Licensing and exact distribution terms of the RSICD and LEVIR-CD benchmark archives.
