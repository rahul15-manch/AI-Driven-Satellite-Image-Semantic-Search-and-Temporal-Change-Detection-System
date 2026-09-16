# Project Development Log (devlog.md)

All engineering activities, architectural milestones, and experimental progress are documented here in reverse-chronological order.

---

## [2026-09-15] — Milestone 4 Corrective Task: Leakage-Controlled Semantic Retrieval Evaluation

**Milestone Identifier:** M4 Corrective  
**Milestone Owner:** Rahul (Team Lead)  
**Collaborators:** Tanishka Mukhi (Dataset & Architecture), Adishri Abro (Literature & Evaluation)  
**Status:** Completed, Corrected, and Fully Re-Evaluated on CPU Hardware  

### 1. Objectives Achieved
- **Leakage Problem Identification:** Diagnosed target-query text overlap in legacy BM25 evaluation ($D_i = \bigcup_{k=1}^5 C_{ik}$), where query text $q = C_{ik}$ was present inside the target document, artificially inflating BM25 to 85.65% R@1.
- **Implementation of Mode A (Leave-One-Caption-Out BM25):** For every query $q = C_{ik}$, the target document $D_i$ dynamically retains only the four remaining captions ($\bigcup_{j \ne k} C_{ij}$) and strictly excludes $q$. Non-target images retain all 5 captions.
- **Implementation of Mode B (Category Metadata BM25):** Added an auxiliary control indexing only category labels (e.g., "airport", "parking") with zero test captions in the gallery.
- **Archival of Legacy Baseline:** Preserved original leaky results under `experiments/results/m4/legacy_caption_indexed/` with clear diagnostic labelling (`DEC-025`).
- **Full Empirical Re-evaluation `[LOCALLY MEASURED]`:**
  - **BM25 (Leave-One-Caption-Out, Mode A):** R@1 = **42.12%**, R@5 = **60.81%**, R@10 = **67.87%**, MRR = **0.5112**, Latency = **1.49 ms**, Peak RAM = **279.7 MB**.
  - **BM25 (Category Metadata, Mode B):** R@1 = **1.50%**, R@5 = **7.30%**, R@10 = **14.35%**, MRR = **0.0618**, Latency = **0.12 ms**, Peak RAM = **287.3 MB**.
  - **BM25 Legacy (Diagnostic / Leaky):** R@1 = **85.65%**, R@5 = **96.38%**, R@10 = **98.57%**, MRR = **0.9028**, Latency = **1.46 ms**, Peak RAM = **357.4 MB**.
  - **CLIP ViT-B/32 (Visual Pixels Only):** R@1 = **5.45%**, R@5 = **17.71%**, R@10 = **27.89%**, MRR = **0.1307**, Latency = **7.54 ms**, Peak RAM = **883.6 MB**.
  - **CLIP + Prompt Ensemble (5 Frozen Templates):** R@1 = **5.14%**, R@5 = **17.00%**, R@10 = **28.01%**, MRR = **0.1268**, Latency = **15.56 ms**, Peak RAM = **553.7 MB**.
- **Hypothesis H1 Reassessment:** Reclassified H1 as **PARTIALLY SUPPORTED / INCONCLUSIVE**. When gallery imagery lacks captions (Mode B), CLIP vision-language embeddings outperform lexical retrieval by **3.6x on R@1** (5.45% vs 1.50%). When human captions are available for gallery imagery (Mode A), lexical retrieval remains superior (42.12% vs 5.45% R@1).
- **Leakage Regression Tests:** Implemented 8 dedicated regression tests in `tests/test_leakage_regression.py`. Repository test suite expanded to 47 passing tests (100% pass rate in 4.93s).

---

## [2026-09-15] — Milestone 4: Semantic Retrieval Baseline & Empirical Evaluation

**Milestone Identifier:** M4  
**Milestone Owner:** Rahul (Team Lead)  
**Collaborators:** Tanishka Mukhi (Dataset & Architecture), Adishri Abro (Literature & Evaluation)  
**Status:** Completed & Empirically Verified on CPU Hardware  

### 1. Milestone Objectives Achieved
- **Method A1 (BM25 Lexical Baseline):** Implemented `src/semantic_search/bm25.py` with standard Okapi BM25 ($k_1 = 1.5, b = 0.75$) and multi-caption combined-document aggregation (`DEC-020`).
- **Method A2 (Zero-Shot CLIP ViT-B/32):** Implemented `src/semantic_search/clip_model.py` loading `openai/clip-vit-base-patch32` strictly on CPU (`DEC-021`), with mandatory $L_2$ feature normalization.
- **Method A3 (Domain Prompt Ensembling Ablation):** Implemented `src/semantic_search/prompt_ensembler.py` with 5 frozen remote-sensing templates (`DEC-023`).
- **FAISS CPU Vector Indexing:** Implemented `src/semantic_search/faiss_index.py` using exact `IndexFlatIP(512)` (`DEC-022`), measuring inner-product search latency of **0.053 ms** on CPU.
- **Persistent Embedding Caching:** Implemented `src/semantic_search/embedding_cache.py` saving $1,093 \times 512$ float32 embeddings with metadata validation.
- **Full Benchmark Execution:** Implemented reproducible CLI runner `src/semantic_search/run_m4.py` consuming `experiments/configs/m4_retrieval.yaml`. Evaluated all 5,465 test queries across 1,093 held-out RSICD gallery images under the standardized Caption-to-Own-Image protocol.
- **Empirical Results `[LOCALLY MEASURED]`:**
  - **BM25:** R@1 = **85.65%**, R@5 = **96.38%**, R@10 = **98.57%**, MRR = **0.9028**, Latency = **1.46 ms**, Peak RAM = **357.4 MB**.
  - **CLIP ViT-B/32:** R@1 = **5.45%**, R@5 = **17.71%**, R@10 = **27.89%**, MRR = **0.1307**, Latency = **7.18 ms**, Peak RAM = **1,260.6 MB**.
  - **CLIP + Prompt Ensemble:** R@1 = **5.14%**, R@5 = **17.00%**, R@10 = **28.01%**, MRR = **0.1268**, Latency = **13.97 ms**, Peak RAM = **669.2 MB**.
- **Hypothesis H1 Assessment:** Formally recorded that Hypothesis H1 is **NOT SUPPORTED / REJECTED** under the Caption-to-Own-Image instance retrieval protocol (`DEC-024`). BM25 outperforms zero-shot CLIP by 15.7x on R@1 due to exact caption vocabulary matching and nadir-view domain shift in general-domain CLIP.
- **Failure Analysis:** Analyzed 3,941 failure cases (rank > 10) in CLIP: Broad Geographic Category Ambiguity (36.3%), Spatial/Relational Confusion (32.4%), Dense Small Objects (20.0%), and Fine-Grained Attribute Mismatch (11.3%).
- **Test Suite Expansion:** Added 13 new unit/integration tests across `tests/test_bm25.py`, `tests/test_faiss_index.py`, `tests/test_embedding_cache.py`, `tests/test_prompt_ensembler.py`, `tests/test_retrieval_evaluator.py`, and `tests/test_clip_model.py`. All 37 tests in the repository pass cleanly in 8.00s.

### 2. Documents Created / Modified
1. `src/semantic_search/__init__.py` `[NEW]` — Module exports.
2. `src/semantic_search/bm25.py` `[NEW]` — Okapi BM25 implementation.
3. `src/semantic_search/clip_model.py` `[NEW]` — CPU-forced CLIP ViT-B/32 loader and encoder.
4. `src/semantic_search/faiss_index.py` `[NEW]` — Exact FAISS IndexFlatIP.
5. `src/semantic_search/embedding_cache.py` `[NEW]` — Persistent embedding caching and validation.
6. `src/semantic_search/prompt_ensembler.py` `[NEW]` — Frozen prompt templates and ensembling.
7. `src/semantic_search/retrieval_evaluator.py` `[NEW]` — Caption-to-Own-Image metrics and diagnostics.
8. `src/semantic_search/profiler.py` `[NEW]` — High-precision latency and memory profiler.
9. `src/semantic_search/run_m4.py` `[NEW]` — Reproducible CLI benchmark runner.
10. `experiments/configs/m4_retrieval.yaml` `[NEW]` — Experiment configuration.
11. `docs/m4_semantic_retrieval.md` `[NEW]` — Detailed milestone report and failure analysis.
12. `tests/test_bm25.py`, `tests/test_faiss_index.py`, `tests/test_embedding_cache.py`, `tests/test_prompt_ensembler.py`, `tests/test_retrieval_evaluator.py`, `tests/test_clip_model.py` `[NEW]` — Comprehensive test suite.
13. `decision_log.md` `[UPDATED]` — Added decisions DEC-020 to DEC-024.
14. `devlog.md` `[UPDATED]` — Added Milestone 4 log entry.

### 3. Work Deliberately NOT Performed in Milestone 4
> [!IMPORTANT]
> **Strict Non-Implementation Declaration:**  
> **No change detection models or training were implemented.**
> Specifically:
> - No pixel differencing, SSIM, or CVA code was executed (reserved for M5).
> - No Siamese CNN training or inference was performed (reserved for M8).
> - No synthetic perturbation generators were implemented (reserved for M6).
> - Milestone 4 was strictly confined to the cross-modal semantic retrieval pipeline.

### 4. Handoff to Subsequent Milestones
- **To Milestone 5 (Classical Change Detection Baselines — Owner: Tanishka Mukhi):**
  - M4 establishes the benchmark standard: clean data loaders, frozen validation calibration, and reproducible CLI runners.
  - Implement B1 (Pixel Diff), B2 (SSIM), and B3 (CVA) on LEVIR-CD.
- **To Milestone 6 (Evaluation Suite & Diagnostics — Owner: Adishri Abro):**
  - Integrate retrieval metric outputs (`experiments/results/m4/summary.csv`) into global reporting dashboards.

---

## [2026-09-15] — Milestone 3: Research Definition, Literature Review & Baseline/Metric Formalization

**Milestone Identifier:** M3  
**Milestone Owner:** Adishri Abro (Literature & Evaluation Lead)  
**Collaborators:** Rahul (Team Lead), Tanishka Mukhi (Dataset & Architecture Lead)  
**Status:** Completed & Grounded in Verified Academic Literature  

### 1. Milestone Objectives Achieved
- **Comprehensive Literature Review:** Authored `literature_review.md` spanning 14 verified peer-reviewed publications across 6 core research domains:
  1. Remote sensing cross-modal image-text retrieval (RSICD, CLIP, RemoteCLIP).
  2. Vision-language foundation models and dual-encoder architectures (Radford et al., 2021).
  3. Classical deterministic change detection (Singh, 1989; Wang et al., 2004; Malila, 1980; Otsu, 1979).
  4. Lightweight deep learning change detection (Daudt et al., 2018; Ronneberger et al., 2015; Chen & Shi, 2020).
  5. False-alarm taxonomy and non-ground confounder handling (Hall et al., 1991; Bovolo & Bruzzone, 2007; Canty & Nielsen, 2008).
  6. Quality-aware, uncertainty-aware, and diagnostic gating prior art (Kendall & Gal, 2017).
- **Critical Citation Audit Executed:**
  - Audited the widely conflated WHU Building Dataset literature.
  - **Audit Finding:** The true bi-temporal change detection paper is *Ji, Shen, Lu, & Zhang (2019), Remote Sensing, 11(11), 1343* (`10.3390/rs11111343`). The frequently cited *Ji, Wei, & Lu (2019), IEEE TGRS, 57(1), 574–586* is strictly a mono-temporal building extraction/segmentation paper. The citation record is now formally corrected.
- **Formal Metrics Specification (`metrics.md`):**
  - Formulated exact mathematical definitions for retrieval: Recall@K ($K \in \{1, 5, 10\}$), MRR, and Precision@K. Formally codified the **Caption-to-Own-Image** protocol over 1,093 held-out RSICD test scenes (5,465 queries) as the primary benchmark.
  - Formulated exact confusion matrix metrics for change detection: Precision, Recall, F1-score, and IoU (Jaccard Index).
  - Mathematically demonstrated the **Overall Accuracy (OA) Trap**: On LEVIR-CD's measured $4.651\%$ change class imbalance, a trivial null predictor scores $95.349\%$ OA with zero F1. Formally deprecated OA for model selection.
  - Formulated CPU computational profiling metrics: wall-clock latency (mean, $p50$, $p95$), peak memory RSS (MB), heap allocation, parameter counts, and index storage.
  - Formulated perturbation robustness metrics: Relative F1 Degradation ($\Delta F_1$) and False Positive Amplification Factor (FPAF).
- **Leakage-Safe Thresholding Protocol:**
  - Established that classical difference thresholds must be determined either via unsupervised Otsu or calibrated strictly on the 64-scene validation split ($\tau^* = \arg\max_{\tau} F_1(\tau; \mathcal{D}_{\text{val}})$) and frozen before test evaluation. Test-set threshold tuning is strictly prohibited.
- **Research-Gap Analysis & Novelty Classification:**
  - Identified 4 evidence-backed research gaps (lack of CPU-constrained trade-off studies, fragility under controlled non-ground perturbations, threshold data leakage, and zero-shot VLM domain shift under nadir view).
  - Conducted a novelty audit on the proposed quality-aware change scoring concept: Classified it objectively as an **ADAPTATION & COMBINATION** rather than foundational novelty, eliminating unsupported claims.
- **Updated Research Questions & Experiment Plan:**
  - Updated `research_questions.md` (RQ1–RQ4, H1–H4, variables, epistemic categories).
  - Updated `experiment_plan.md` with explicit protocols for Suites A, B, C, D, and E, complete with inputs, outputs, controls, variables, and methodological interpretations.
- **Decision Records:**
  - Added DEC-014 through DEC-019 to `decision_log.md`.

### 2. Documents Created / Modified
1. `literature_review.md` `[NEW]` — Comprehensive literature review, evidence table, citation audit, research gaps, novelty classification, and verified references.
2. `metrics.md` `[NEW]` — Formal mathematical and operational specifications for all retrieval, change-detection, computational, and perturbation metrics.
3. `research_questions.md` `[UPDATED]` — Literature-grounded research questions, formal hypotheses H1–H4, independent/dependent variables, and epistemic taxonomy.
4. `experiment_plan.md` `[UPDATED]` — Updated experiment suites (A1–A3, B1–B5, Perturbations C1–C3, Ablations D1–D4, Hardware Suite E).
5. `decision_log.md` `[UPDATED]` — Documented decisions DEC-014 to DEC-019.
6. `devlog.md` `[UPDATED]` — Documented Milestone 3 achievements and handoffs.

### 3. Key Decisions Made
- **Decision DEC-014:** Selected A1 (BM25 keyword index) and A2 (Pretrained CLIP ViT-B/32 zero-shot) as retrieval baselines, with A3 evaluated as an ablation.
- **Decision DEC-015:** Selected B1 (Pixel Diff), B2 (SSIM), B3 (CVA), B4 (`FC-Siam-diff`), and B5 (Quality-Gated) as change detection baselines.
- **Decision DEC-016:** Primary metrics standardized: Caption-to-Own-Image R@K and MRR for retrieval; F1-score and IoU on changed class for change detection. Overall Accuracy formally deprecated.
- **Decision DEC-017:** Enforced leakage-safe validation-calibrated frozen thresholds and unsupervised Otsu thresholding.
- **Decision DEC-018:** Formulated 4 literature-grounded perturbation stress tests (illumination, blur, misregistration jitter, occlusion).
- **Decision DEC-019:** Formally classified the proposed quality-aware change detection method as an **ADAPTATION & COMBINATION** with no unsupported novelty claims.

### 4. Work Deliberately NOT Performed in Milestone 3
> [!IMPORTANT]
> **Strict Non-Implementation Declaration:**  
> **No machine learning models, training scripts, FAISS indexes, or application services were implemented in M3.**
> Specifically:
> - No CLIP model weights were downloaded or instantiated.
> - No FAISS vector search index was built.
> - No classical or learned change detection code (B1–B5) was executed.
> - No training loops or inference scripts were written.
> - No FastAPI routes or frontend components were implemented.
> - Milestone 3 was strictly confined to research design, literature review, citation auditing, mathematical formalization, and metric definition.

### 5. Handoff to Subsequent Milestones
- **To Milestone 4 (Semantic Retrieval Baseline & Implementation — Owner: Rahul):**
  - Implement A1 (BM25 / inverted index) and A2 (Pretrained CLIP ViT-B/32 with FAISS `IndexFlatIP`).
  - Follow the **Caption-to-Own-Image** evaluation protocol and formal $R@K$ / MRR metrics specified in `metrics.md` Section 1.
  - Test on the 1,093 held-out test scenes (5,465 queries) from `data/splits/rsicd/rsicd_splits.json`.
- **To Milestone 5 (Classical Change Detection Baselines — Owner: Tanishka Mukhi):**
  - Implement B1 (Absolute Pixel Differencing), B2 (SSIM Dissimilarity), and B3 (Change Vector Analysis).
  - Strictly adhere to `metrics.md` Section 3: Calibrate optimal threshold $\tau^*$ on the 64 validation scenes, freeze $\tau^*$, and evaluate on the 128 test scenes from `data/splits/levir_cd/levir_splits.json`.
  - Report F1-score and IoU on the changed class. Disregard Overall Accuracy.
- **To Milestone 6 (False-Alarm Suppression & Diagnostic Evaluation — Owner: Adishri Abro):**
  - Implement controlled perturbation generators (`EXP-PERT-01` to `03`) and diagnostic quality gating (`EXP-CD-05`).
  - Benchmark FPAF and relative F1 degradation ($\Delta F_1$) across classical vs. learned models.

---

## [2026-09-15] — Milestone 2: Dataset Acquisition, Verification & Preprocessing Pipeline

**Milestone Identifier:** M2  
**Milestone Owner:** Tanishka Mukhi  
**Collaborators:** Rahul (Team Lead), Adishri Abro (Literature & Evaluation)  
**Status:** Completed & Empirically Verified on Real Datasets  

### 1. Milestone Objectives Achieved
- **User-Provided Dataset Integration:**
  - **RSICD Integration:** Successfully extracted all **10,921 images** and **54,605 captions** from user-provided CSV archives into `data/raw/rsicd/images/` and `data/raw/rsicd/dataset_rsicd.json`. Every single image byte stream was verified with zero corrupted files and zero duplicate hashes.
  - **LEVIR-CD Integration:** Integrated all **637 official bitemporal pairs** ($1024 \times 1024$ optical scenes) into `data/raw/levir_cd/` across `train/` (445), `val/` (64), and `test/` (128) partitions.
- **Corrupted Artifact Purge & Reset:** Safely deleted previous synthetic noise artifacts (e.g., `airport_02.jpg`), ensuring no synthetic noise exists in the data pipeline.
- **Clean Standardized Directory Hierarchy:**
  ```text
  data/
  ├── raw/ (rsicd/, levir_cd/)
  ├── processed/ (rsicd/, levir_cd/)
  ├── metadata/ (rsicd/, levir_cd/)
  └── splits/ (rsicd/, levir_cd/)
  ```
- **Factual Dataset Statistics Computation:**
  - RSICD: 10,921 images, 100% valid $224 \times 224$ RGB, 31 categories, 8,734 train / 1,094 val / 1,093 test split.
  - LEVIR-CD: 637 scenes, 667,942,912 total pixels, exactly **31,066,643 changed pixels (4.651%)** measured across all pairs (train: 4.589%, val: 4.197%, test: 5.094%).
- **Visual Sanity Confirmation:** Rendered 16-sample contact sheet to `data/processed/rsicd/sanity_check_grid.png`. Visual inspection confirmed authentic satellite imagery (airports, residential, coastal, desert, industrial, waterbodies) rather than noise.
- **Metadata & Splits Persistence:**
  - Generated machine-readable manifests: `data/metadata/rsicd/rsicd_metadata.json`, `rsicd_metadata.csv`, `data/metadata/levir_cd/levir_metadata.json`, and `levir_metadata.csv`.
  - Saved deterministic split manifests: `data/splits/rsicd/rsicd_splits.json` and `data/splits/levir_cd/levir_splits.json`.
- **Test Suite Status:** 24 unit and integration tests passing in 0.80s (100% pass rate).

### 2. Datasets Investigated & Verification Summary
- **LEVIR-CD:**
  - *Authors & Source:* Hao Chen and Zhenwei Shi, LEVIR Lab, Beihang University; *Remote Sensing* (MDPI), 2020.
  - *Properties Locally Measured:* 637 bitemporal pairs, $1024 \times 1024$ pixels, 3 channels RGB, 4.651% changed pixel ratio.
  - *Status:* `[LOCALLY-MEASURED & VALIDATED]`.
- **RSICD:**
  - *Authors & Source:* Xiaoqiang Lu, Binqiang Wang, Xiangtao Zheng, and Xuelong Li; *IEEE TGRS*, 2018.
  - *Properties Locally Measured:* 10,921 images, $224 \times 224$ pixels, 3 channels RGB, 54,605 captions, 0 duplicate hashes.
  - *Status:* `[LOCALLY-MEASURED & VALIDATED]`.

### 3. Files Created / Modified
- `data_card.md` — Authoritative data cards for LEVIR-CD and RSICD with explicit epistemic labels.
- `data/README.md` — Dataset placement instructions, verification commands, and raw data protection policy.
- `requirements.txt` — Minimal pinned dependencies.
- `.gitignore` — Excludes `data/raw/*` and `data/processed/*` binaries while preserving directory structure.
- `src/data/dataset_verifier.py` — Image decoding, SHA-256 duplicate hashing, and RSICD directory verification.
- `src/data/verify_rsicd.py` — Dedicated CLI verification tool for user-supplied RSICD images.
- `src/data/inspect_rsicd.py` — Visual sanity check and contact sheet generator.
- `src/data/metadata_builder.py` — Automated metadata manifest generator with SHA-256 hashing.
- `src/data/patch_extractor.py` — Configurable patch extractor with spatial leakage prevention.
- `src/data/split_manager.py` — Leakage-safe train/val/test splitting and manifest persistence.
- `src/data/levir_loader.py` — Bi-temporal dataset loader and class distribution profiler.
- `src/data/rsicd_loader.py` — RSICD loader and category parser.
- `scripts/download_datasets.py` — Provenance guide with zero automated downloads.
- `tests/test_dataset_verifier.py` — 8 unit tests covering corrupted, truncated, zero-byte, and duplicate cases.
- `tests/test_metadata_builder.py` — Unit tests for metadata serialization and directory extraction.
- `tests/test_rsicd_loader.py` — Dynamic tmp_path fixture integration tests for RSICD loading.
- `tests/test_pipeline.py` — End-to-end integration test (raw -> validation -> metadata -> preprocessing).
- `decision_log.md` — Added decisions DEC-008 through DEC-013.
- `devlog.md` — Updated development log entry.
- `tests/test_split_manager.py` — Unit tests for split determinism and leakage detection.
- `tests/test_levir_loader.py` — Integration tests for LEVIR dataset loading.
- `tests/test_rsicd_loader.py` — Integration tests for RSICD dataset loading.
- `tests/test_metadata_builder.py` — Unit tests for metadata serialization.
- `decision_log.md` — Added decisions DEC-008 through DEC-012.
- `devlog.md` — Added this M2 milestone entry.

### 4. Empirical Resource & Throughput Measurements (Locally Measured)
- **Patch Extraction Throughput (CPU):** Extracting sixteen $256 \times 256$ patches from a $1024 \times 1024$ bi-temporal pair ($T_1, T_2, \text{label}$) and writing to disk took **132.06 ms** on standard CPU.
- **Peak Operational Memory:** Peak heap allocation during patch extraction was **9.46 MB**, well below our 8 GB ceiling.
- **Projected Full-Dataset Tiling Time:** Estimated at $\approx 637 \text{ scenes} \times 0.14\text{ s} \approx 89\text{ seconds}$ on quad-core CPU.
- **Storage Profile:** Local test fixtures require $< 5\text{ MB}$; full LEVIR-CD raw requires ~10 GB; processed patches require ~3.5 GB. Host system has 83 GB available.

### 5. Key Decisions Made
- **Decision DEC-008:** Excluded raw and processed image binaries from Git; tracked only manifests and code.
- **Decision DEC-009:** Enforced scene-first partitioning before patch extraction to prevent spatial cross-validation leakage.
- **Decision DEC-010:** Adopted authoritative official splits (Chen & Shi for LEVIR; Karpathy for RSICD).
- **Decision DEC-011:** Enforced strict stem-based temporal pair association.
- **Decision DEC-012:** Implemented deterministic synthetic test fixtures for mock-free offline testing.

### 6. Working Assumptions Introduced
- **Assumption 1:** 256x256 non-overlapping patches (stride 256) provide a balanced spatial context for building changes under CPU memory constraints (configurable via `patch_size`).
- **Assumption 2:** Standard ImageNet channel normalization coefficients are appropriate as the default optical baseline for both datasets.

### 7. Unresolved Questions & Risks for Future Milestones
- **LEVIR-CD Scene Diversity:** LEVIR-CD is heavily focused on building changes with limited seasonal/atmospheric variation; Milestone 9 / Milestone 11 will need synthetic perturbation tests to stress-test false alarms.
- **RSICD Caption Quality:** Some RSICD captions are repetitive; Milestone 4 must examine whether repeated captions impact Precision@K ranking metrics.

### 8. Work Deliberately NOT Performed in Milestone 2
> [!IMPORTANT]
> **Strict Non-Implementation Declaration:**  
> **No machine learning models, change detection algorithms, or retrieval indexing have been implemented.**
> Specifically:
> - No CLIP models or text encoders were downloaded or initialized.
> - No FAISS indices were created.
> - No Siamese CNN or difference thresholding algorithms were implemented.
> - No FastAPI routes or frontend components were implemented.
> - Milestone 2 was strictly confined to dataset verification, loading, patching, integrity checking, and preprocessing pipelines.

### 9. Next Milestones & Handoff
- **To Milestone 3 (Owner: Adishri Abro):** Literature review, baseline formulation, and metric definitions can now reference verified dataset structures and mathematical properties documented in `data_card.md`.
- **To Milestone 4 (Owner: Rahul):** Semantic retrieval baseline can directly consume `RSICDDataset` and `SplitManager` manifests.
- **To Milestone 5 (Owner: Tanishka Mukhi):** Classical change detection baselines can directly consume `LEVIRDataset` and `PatchExtractor`.
- **To Milestone 6 (Owner: Adishri Abro):** Evaluation framework can build upon validated split manifests and class balance calculations.

---

## [2026-09-15] — Milestone 1: Research Definition & System Architecture

**Milestone Identifier:** M1  
**Milestone Owner:** Rahul (Team Lead)  
**Collaborators:** Tanishka Mukhi, Adishri Abro  
**Status:** Completed  

### 1. Milestone Objectives Achieved
- Established formal research problem framing for both Semantic Satellite Image Retrieval and Bi-Temporal Change Detection under strict commodity CPU constraints.
- Formulated four initial research questions (RQ1–RQ4) and four companion hypotheses (H1–H4) with an explicit provisional disclaimer.
- Formalized project boundaries, functional/non-functional requirements, and explicit out-of-scope criteria in `PRD.md`.
- Designed an 8-layer decoupled modular architecture with data flow diagrams, CPU memory/thread execution guidelines, and strict module anti-responsibilities in `architecture.md`.
- Outlined an empirical experiment plan spanning baselines, candidate models, false-alarm perturbation stress-tests, and ablation studies in `experiment_plan.md`.
- Formulated initial foundational decisions (DEC-001 through DEC-007) in `decision_log.md`.
- Defined a 15-milestone roadmap (5 milestones per team member) with explicit dependency justifications.

### 2. Documents Created / Updated
1. `PRD.md` — Product / Research Requirements Document defining system purpose, problem statement, user personas, FRs, NFRs, constraints, and success criteria.
2. `research_questions.md` — Research problem definition, provisional RQs (RQ1–RQ4), provisional hypotheses (H1–H4), variables, and epistemic taxonomy.
3. `architecture.md` — Detailed system architecture, Mermaid data-flow diagrams, module boundary contracts, CPU execution design, and 15-milestone dependency graph.
4. `experiment_plan.md` — Experimental protocols for retrieval (A1–A3), change detection (B1–B5), perturbation tests (illumination, shadow, misregistration), ablations, and CPU profiling.
5. `decision_log.md` — Structured decision records for architectural choices, tech stack, datasets, and milestone structures.
6. `devlog.md` — Initial project development log entry.
7. `README.md` — Updated repository root with research summary, roadmap, and documentation sitemap.

### 3. Key Decisions Made
- **Decision DEC-001:** Scope locked to dual-capability system (Semantic Search + Change Detection).
- **Decision DEC-002:** Primary programming language confirmed as Python 3.10+ with standard scientific libraries (`PyTorch`, `OpenCV`, `NumPy`, `FAISS`, `scikit-learn`).
- **Decision DEC-003:** Strict CPU-only execution constraint adopted for all modeling and evaluation.
- **Decision DEC-004:** RSICD and LEVIR-CD designated as proposed candidate benchmark datasets, pending empirical verification.
- **Decision DEC-005:** Baseline-first evaluation philosophy enforced (classical algorithms benchmarked prior to learned models).
- **Decision DEC-006:** Decoupled 8-layer architecture adopted to prevent tight component coupling.
- **Decision DEC-007:** 15-milestone team distribution confirmed across Rahul, Tanishka, and Adishri.

### 4. Working Assumptions Introduced
- **Assumption 1:** Zero-shot representations from general vision-language backbones (e.g., standard CLIP) will retain sufficient semantic alignment for remote sensing scenes without requiring GPU-intensive end-to-end retraining.
- **Assumption 2:** Large satellite scenes ($1024 \times 1024$) can be partitioned into $256 \times 256$ sub-crops to bound CPU RAM usage below 8 GB without severe boundary detection artifacts.
- **Assumption 3:** Public benchmark datasets exhibit adequate native co-registration quality to evaluate baseline change algorithms without mandatory prior orthorectification.

### 5. Unresolved Questions & Items Requiring Verification
- *Verification Target 1 (Milestone 2):* Exact archive sizes, licensing terms, and disk footprint for RSICD and LEVIR-CD.
- *Verification Target 2 (Milestone 2):* Feasibility and speed of running batch preprocessing/tiling for LEVIR-CD on laptop CPUs.
- *Verification Target 3 (Milestone 3):* Academic literature confirmation on state-of-the-art classical change detection baselines (exact mathematical formulations for SSIM and CVA thresholding).
- *Verification Target 4 (Milestone 3):* Identification of the most lightweight public vision-language checkpoint that runs efficiently on CPU (e.g., CLIP ViT-B/32 vs. MobileCLIP vs. ResNet-50).

### 6. Work Deliberately NOT Performed in Milestone 1
> [!IMPORTANT]
> **Strict Non-Implementation Declaration:**  
> **No model training, application implementation, frontend implementation, or backend implementation has been performed as part of M1.**
> Specifically:
> - No neural network weights or pretrained models were downloaded or initialized.
> - No raw datasets (RSICD, LEVIR-CD) were downloaded.
> - No FastAPI routes, endpoints, or server scripts were written.
> - No HTML/CSS/JavaScript UI files were created.
> - No baseline change detection or retrieval code was executed.
> - Milestone 1 is strictly confined to research definition, architectural contracts, and experimental planning.

### 7. Next Milestones & Handoffs
- **Milestone 2 (Owner: Tanishka Mukhi):** Dataset Acquisition, Verification & Preprocessing Pipeline (download verification, format standardization, patch tiling, metadata generation).
- **Milestone 3 (Owner: Adishri Abro):** Comprehensive Literature Review, Mathematical Baseline Formulation & Metric Formalization.
