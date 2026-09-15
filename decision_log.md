# Project Decision Log

This document records the foundational research, architectural, and engineering decisions established during the project lifecycle. All entries must reflect verified facts, explicit project constraints, or clearly labeled provisional assumptions.

---

## Decision Record Format
- **Decision ID:** Unique identifier (e.g., `DEC-001`)
- **Title:** Brief summary of the architectural or research choice
- **Date:** ISO Date of decision
- **Owner:** Team member responsible
- **Context & Problem Statement:** Why this decision was required
- **Alternatives Considered:** Other options evaluated
- **Decision Made:** The chosen approach
- **Rationale & Trade-offs:** Technical justification and accepted trade-offs
- **Status:** `Confirmed` | `Provisional` | `Under Verification`
- **Evidence / Source:** Source in approved synopsis, verified literature, or empirical data (strictly non-fabricated)

---

### DEC-001: Dual-Capability System Scope
- **Date:** 2026-09-15
- **Owner:** Rahul (Team Lead)
- **Context:** The project requires a cohesive research focus spanning computer vision and remote sensing. We needed to confirm whether the system should address retrieval, change detection, or both.
- **Alternatives Considered:**
  1. Semantic search only.
  2. Temporal change detection only.
  3. Integrated dual-capability research prototype.
- **Decision Made:** Pursue an integrated dual-capability system containing both Semantic Satellite Image Retrieval and Bi-Temporal Change Detection, coupled by a common geospatial representation framework.
- **Rationale & Trade-offs:** This matches the approved project synopsis and addresses real-world analyst workflows (searching for an area of interest, then analyzing temporal evolution). The trade-off is a broader implementation scope across two distinct subfields of computer vision.
- **Status:** Confirmed
- **Evidence / Source:** Approved B.Tech CSE (AI/ML) Project Synopsis.

---

### DEC-002: Primary Programming Language & Core Ecosystem
- **Date:** 2026-09-15
- **Owner:** Rahul (Team Lead)
- **Context:** Selecting a programming language and runtime ecosystem that supports modern machine learning, computer vision, vector search, and local execution.
- **Alternatives Considered:**
  1. C++ (high speed, but steep development curve and limited rapid-prototyping ML libraries).
  2. Python (rich geospatial, scientific, and deep learning ecosystem).
  3. Rust / Go (fast, but lacking mature vision-language and remote sensing tooling).
- **Decision Made:** Select Python 3.10+ as the primary programming language.
- **Rationale & Trade-offs:** Python is the standard in academic AI/ML research with mature, interoperable libraries (`PyTorch`, `OpenCV`, `NumPy`, `FAISS`, `scikit-learn`). The trade-off is higher interpreted overhead, which we will mitigate by using C-accelerated underlying libraries (`OpenCV`, `NumPy`, `faiss-cpu`).
- **Status:** Confirmed
- **Evidence / Source:** Project synopsis technical direction and cross-member software competency.

---

### DEC-003: Strict CPU-Only Execution Constraint
- **Date:** 2026-09-15
- **Owner:** Rahul (Team Lead)
- **Context:** Deciding the target compute platform and hardware execution profile for all modeling and evaluation.
- **Alternatives Considered:**
  1. Cloud GPU infrastructure (AWS EC2, Google Cloud Vertex, Lambda Labs).
  2. Local discrete CUDA GPUs (restricted to specific member laptops).
  3. Strict CPU-only execution on standard student laptops.
- **Decision Made:** Adopt strict CPU-only execution on standard student laptops (quad-core/octa-core CPUs, 8–16 GB RAM).
- **Rationale & Trade-offs:** Eliminates external funding dependencies, GPU driver fragmentation, and cloud costs. Ensures the project is accessible, reproducible, and forces disciplined engineering (lightweight models, patch tiling, compact indices). The trade-off is that training massive foundation models from scratch is precluded, shifting our research focus to lightweight architectures and efficient inference.
- **Status:** Confirmed
- **Evidence / Source:** Computational constraints mandate in project brief.

---

### DEC-004: Proposed Datasets for Initial Investigation (RSICD & LEVIR-CD)
- **Date:** 2026-09-15
- **Owner:** Rahul (Team Lead)
- **Context:** Selecting standard benchmark datasets to ground empirical investigations in semantic retrieval and change detection.
- **Alternatives Considered:**
  1. For Retrieval: UCMerced-Captions, Sydney-Captions, RSICD.
  2. For Change Detection: CDD (Change Detection Dataset), WHU-CD, LEVIR-CD, OSCD.
- **Decision Made:** Designate RSICD as the candidate benchmark for Semantic Retrieval and LEVIR-CD as the candidate benchmark for Bi-Temporal Change Detection, subject to verification in Milestone 2.
- **Rationale & Trade-offs:** RSICD provides diverse natural-language captions across thousands of remote sensing scenes; LEVIR-CD provides high-resolution bi-temporal optical pairs with precise building masks. The trade-off is dataset download size and high spatial resolution ($1024 \times 1024$), requiring tiling into $256 \times 256$ sub-crops for CPU processing.
- **Status:** Provisional / To Be Verified (Pending Milestone 2 data feasibility analysis)
- **Evidence / Source:** Literature convention in remote sensing benchmarks. Licensing and download integrity to be verified.

---

### DEC-005: Empirical Baseline-First Evaluation Strategy
- **Date:** 2026-09-15
- **Owner:** Rahul (Team Lead)
- **Context:** Establishing the scientific methodology for comparing models.
- **Alternatives Considered:**
  1. Implement a single advanced deep neural network directly.
  2. Establish rigorous, transparent classical baselines first before introducing learned models.
- **Decision Made:** Adopt a baseline-first evaluation strategy. Classical deterministic methods (pixel differencing, SSIM, CVA, keyword search) must be benchmarked and evaluated before training or deploying learned models.
- **Rationale & Trade-offs:** A learned model's performance cannot be claimed as superior without a direct, reproducible comparison against well-calibrated baselines on identical data splits.
- **Status:** Confirmed
- **Evidence / Source:** Empirical research methodology principles.

---

### DEC-006: Decoupled, Modular Pipeline Architecture
- **Date:** 2026-09-15
- **Owner:** Rahul (Team Lead)
- **Context:** Structuring codebase modules to enable parallel collaboration across the 3 team members without merge conflicts.
- **Alternatives Considered:**
  1. Monolithic single-script architecture.
  2. Decoupled multi-layer architecture with strict interface contracts (Data, Preprocessing, Retrieval, Change Detection, Diagnostics, Evaluation).
- **Decision Made:** Implement a modular multi-layer architecture with explicit interface boundaries and anti-responsibilities.
- **Rationale & Trade-offs:** Allows Rahul, Tanishka, and Adishri to develop their assigned modules independently while ensuring seamless integration in Milestone 13.
- **Status:** Confirmed
- **Evidence / Source:** Software engineering best practices for multi-contributor research teams.

---

### DEC-007: 15-Milestone Allocation and Cross-Dependency Structure
- **Date:** 2026-09-15
- **Owner:** Rahul (Team Lead)
- **Context:** Organizing team workload across the project timeline into distinct, verifiable deliverables.
- **Alternatives Considered:**
  1. Ad-hoc task assignments.
  2. Structured 15-milestone framework with 5 milestones per member.
- **Decision Made:** Adopt the 15-milestone structure:
  - Rahul: M1, M4, M7, M10, M13
  - Tanishka Mukhi: M2, M5, M8, M11, M14
  - Adishri Abro: M3, M6, M9, M12, M15
- **Rationale & Trade-offs:** Provides equitable workload distribution, accountability, and clear milestone dependency gates.
- **Status:** Confirmed
- **Evidence / Source:** Team project management agreement.

---

### DEC-008: Strict Git Exclusion for Large Raw and Processed Datasets
- **Date:** 2026-09-15
- **Owner:** Tanishka Mukhi
- **Context:** Deciding how to store and track raw imagery (~10 GB for LEVIR-CD, ~1.5 GB for RSICD) and extracted patch crops in version control.
- **Alternatives Considered:**
  1. Commit subsets directly to Git repository.
  2. Use Git LFS (Large File Storage) with cloud hosting.
  3. Strictly exclude `data/raw/*` and `data/processed/*` from Git while tracking directory structure (`.gitkeep`), split manifests (`data/splits/*.json`), and metadata (`data/metadata/*`).
- **Decision Made:** Exclude all raw and processed image binaries from Git via `.gitignore`. Provide automated download/fixture scripts and structured JSON/CSV metadata in version control.
- **Rationale & Trade-offs:** Prevents repository bloat, clone timeouts, and Git LFS quota issues. Guarantees that student laptops can clone the repository instantly while reproducing data preprocessing deterministically.
- **Status:** Confirmed
- **Evidence / Source:** Open-source remote sensing repository management standards.

---

### DEC-009: Configurable Patch Extraction with Leakage-Safe Scene-First Partitioning
- **Date:** 2026-09-15
- **Owner:** Tanishka Mukhi
- **Context:** Managing high-resolution satellite imagery ($1024 \times 1024$) under CPU constraints while preventing data leakage across train, val, and test partitions.
- **Alternatives Considered:**
  1. Tile all raw imagery into patches first, then randomly shuffle and split patches.
  2. Partition parent scene pairs into splits first, then extract patches independently within each partition.
  3. Resize entire $1024 \times 1024$ scenes down to $256 \times 256$ without patching.
- **Decision Made:** Partition parent scene pairs into train/val/test splits FIRST, then extract patches. Support configurable patch size (defaulting to 256 with parameter `patch_size`), stride, and padding modes.
- **Rationale & Trade-offs:** Shuffling patches prior to splitting causes severe spatial data leakage (adjacent patches from the same scene sharing identical geographic context across train and test sets). Resizing destroys critical small-building structural features. Scene-first splitting completely eliminates spatial leakage while keeping memory below 10 MB per patch crop.
- **Status:** Confirmed
- **Evidence / Source:** Standard spatial cross-validation literature in remote sensing (Chen & Shi, 2020; STANet).

---

### DEC-010: Adoption of Authoritative Official Splits for LEVIR-CD and RSICD
- **Date:** 2026-09-15
- **Owner:** Tanishka Mukhi
- **Context:** Choosing whether to invent custom random splits or adopt established literature benchmarks for evaluation.
- **Alternatives Considered:**
  1. Generate random custom splits across all data.
  2. Adopt official benchmark partitions (Chen & Shi 2020: 445 train / 64 val / 128 test for LEVIR-CD; standard Karpathy JSON split for RSICD).
- **Decision Made:** Adopt the authoritative official benchmark splits for primary model comparisons, with `SplitManager` available for secondary k-fold validation if required.
- **Rationale & Trade-offs:** Enables direct, scientifically valid comparisons with published remote sensing papers without claiming non-standard baselines.
- **Status:** Confirmed
- **Evidence / Source:** Chen & Shi (Remote Sensing 2020); Lu et al. (IEEE TGRS 2018).

---

### DEC-011: Strict Stem-Based Temporal Pair Association
- **Date:** 2026-09-15
- **Owner:** Tanishka Mukhi
- **Context:** Pairing pre-change ($T_1$) and post-change ($T_2$) images and change masks in LEVIR-CD.
- **Alternatives Considered:**
  1. Sort directory file listings independently and pair by index.
  2. Strictly associate $T_1$, $T_2$, and label masks by identical filename stems (e.g. `train_0012.png`).
- **Decision Made:** Enforce strict filename stem matching across `A/`, `B/`, and `label/` subfolders; reject any pair where stems do not match or a component is missing.
- **Rationale & Trade-offs:** Independent sorting is fragile to differing OS filesystem collation orders and missing files, which silently corrupts temporal pair alignment. Stem matching guarantees zero pairing corruption.
- **Status:** Confirmed
- **Evidence / Source:** Data integrity verification principles.

---

### DEC-012: Deterministic Synthetic Test Fixtures for Zero-Download CI & Unit Testing
- **Date:** 2026-09-15
- **Owner:** Tanishka Mukhi
- **Context:** Testing data loaders, patch extractors, verifiers, and metadata builders on student laptops without requiring team members to immediately download the full 10 GB archive.
- **Alternatives Considered:**
  1. Require full 10 GB download before tests can be executed.
  2. Mock filesystem objects using `unittest.mock`.
  3. Generate deterministic synthetic image and annotation fixtures (`tests/fixtures/`) matching exact LEVIR-CD and RSICD formats.
- **Decision Made:** Implement `scripts/download_datasets.py --create-fixtures` to generate lightweight, deterministic image and annotation fixtures.
- **Rationale & Trade-offs:** Allows 100% test coverage and hardware profiling locally within seconds without network dependencies, while keeping mock-free real image processing intact.
- **Status:** Confirmed
- **Evidence / Source:** Software testing best practices for scientific computing.

---

### DEC-013: Reset Corrupted RSICD Artifacts and Require Manual Dataset Provisioning
- **Date:** 2026-09-15
- **Owner:** Tanishka Mukhi
- **Context:** The previously available local RSICD image files were corrupted (exhibiting random RGB noise rather than authentic remote sensing imagery). The corruption affected the artifacts broadly, rendering them unusable for research experiments.
- **Alternatives Considered:**
  1. Attempt algorithmic image reconstruction or denoising.
  2. Implement an automatic network fetcher to download an external archive.
  3. Safely purge corrupted artifacts, disable automatic downloads, and establish an objective validation pipeline for researcher-supplied dataset provisioning.
- **Decision Made:** Remove all corrupted RSICD artifacts from the repository, explicitly disable automatic dataset downloading, and establish a strict manual provisioning workflow. The researcher will supply the valid RSICD dataset into `data/raw/rsicd/`, where it will be objectively validated via `DatasetVerifier` before any downstream experiments proceed.
- **Rationale & Trade-offs:** Reconstructing corrupted pixels or training on noise invalidates the empirical integrity of the study. Automatic downloading risks network instability, API breakage, or downloading another unverified archive. Manual provisioning with automated objective validation guarantees that only authentic, researcher-verified satellite imagery enters the experimental pipeline. Raw data remains strictly outside version control.
- **Status:** Confirmed
- **Evidence / Source:** Visual inspection of corrupted artifacts; research data integrity guidelines.

---

### DEC-014: Selection of Semantic Retrieval Baselines (A1, A2, and A3 as Ablation)
- **Date:** 2026-09-15
- **Owner:** Adishri Abro (Literature & Evaluation Lead)
- **Context:** Literature review indicates that comparing neural vision-language models requires both a non-neural baseline and a standardized pretrained dual-encoder baseline. We needed to decide on the exact retrieval baselines to implement in M4.
- **Alternatives Considered:**
  1. Train a cross-modal transformer from scratch on RSICD.
  2. Use only zero-shot CLIP without a non-neural baseline.
  3. Establish A1 (Classical BM25 / Inverted Index on text captions) and A2 (Pretrained CLIP ViT-B/32 zero-shot dual-encoder with FAISS `IndexFlatIP`), with A3 (Domain Prompt Ensembling) evaluated as an ablation study rather than a distinct heavyweight architecture.
- **Decision Made:** Adopt A1 (BM25 / Inverted Index) and A2 (Zero-shot CLIP ViT-B/32 with FAISS inner-product index) as the primary retrieval baselines. Treat A3 (prompt engineering / ensembling) as an ablation test suite.
- **Rationale & Trade-offs:** Training from scratch violates the CPU budget and is scientifically unneeded when investigating transferability. A1 establishes whether semantic vector retrieval genuinely beats exact keyword matching, and A2 directly tests Hypothesis H1 under reproducible conditions.
- **Status:** Confirmed
- **Evidence / Source:** Radford et al. (ICML 2021); Lu et al. (IEEE TGRS 2018); `literature_review.md` Section 2.

---

### DEC-015: Selection of Change Detection Baselines (B1, B2, B3, B4, and B5)
- **Date:** 2026-09-15
- **Owner:** Adishri Abro (Literature & Evaluation Lead)
- **Context:** Selecting an appropriate set of classical and lightweight learned change-detection baselines that are mathematically rigorous, reproducible, and feasible on commodity CPUs.
- **Alternatives Considered:**
  1. Benchmark only heavy vision transformers (e.g., BIT, ChangeFormer).
  2. Implement only classical pixel differencing and skip learned models.
  3. Formulate a 5-tier baseline suite:
     - B1: Absolute Pixel Differencing (Singh, 1989)
     - B2: Structural Similarity Index Measure (SSIM) Differencing (Wang et al., 2004)
     - B3: Change Vector Analysis (CVA) (Malila, 1980)
     - B4: Fully Convolutional Siamese Difference Network (`FC-Siam-diff` / Tiny-UNet, Daudt et al., 2018)
     - B5: Lightweight Diagnostic Quality-Gated Model (Hypothesis H3)
- **Decision Made:** Formally select B1, B2, B3, B4, and B5 as our benchmark suite. Large transformer architectures (BIT, ChangeFormer) will be retained as literature context only, due to excessive CPU latency and memory footprint.
- **Rationale & Trade-offs:** This spans the full spectrum from raw radiometric differencing (B1) to local structural variance (B2), spectral displacement (B3), hierarchical learned convolutions (B4), and false-alarm suppression gating (B5), directly addressing RQ2, RQ3, and RQ4.
- **Status:** Confirmed
- **Evidence / Source:** Singh (1989); Wang et al. (2004); Malila (1980); Daudt et al. (2018); Chen & Shi (2020); `literature_review.md` Sections 4 & 5.

---

### DEC-016: Formalized Standard Evaluation Metrics & Class-Imbalance Deprecation Policy
- **Date:** 2026-09-15
- **Owner:** Adishri Abro (Literature & Evaluation Lead)
- **Context:** Defining the exact, mathematically unambiguous evaluation metrics for cross-modal retrieval and bi-temporal change detection, while addressing severe class imbalance in LEVIR-CD.
- **Alternatives Considered:**
  1. Use Overall Accuracy (OA) as the primary change-detection metric.
  2. Use F1-score and IoU on the changed class as primary metrics, while explicitly deprecating Overall Accuracy for model selection.
  3. For retrieval, mix category-matching and caption-matching without distinction.
- **Decision Made:**
  - For Change Detection: Formally establish F1-score and IoU (Jaccard index) on the changed class as primary metrics. Precision and Recall will be reported alongside them. Overall Accuracy is strictly deprecated as a model selection criterion because non-change pixels represent 95.349% of LEVIR-CD (a dummy zero-change predictor trivially scores 95.35% OA).
  - For Semantic Retrieval: Formally designate Caption-to-Own-Image Recall@K ($K \in \{1, 5, 10\}$) and MRR across the 1,093 held-out test scenes (5,465 queries) as the primary benchmark, with category-level Precision@K reported as secondary diagnostic context.
- **Rationale & Trade-offs:** Prevents deceptive accuracy inflation on imbalanced datasets and aligns directly with established remote-sensing benchmarking standards.
- **Status:** Confirmed
- **Evidence / Source:** Lu et al. (2018); Chen & Shi (2020); `metrics.md` Section 2.3.

---

### DEC-017: Leakage-Safe Threshold Calibration Policy
- **Date:** 2026-09-15
- **Owner:** Adishri Abro (Literature & Evaluation Lead)
- **Context:** Classical change-detection methods (B1, B2, B3) output continuous difference maps that require thresholding. Many literature studies search for the best threshold directly on the test set, causing severe data leakage.
- **Alternatives Considered:**
  1. Grid search for optimal threshold directly on the test set (standard in some informal studies, but invalid).
  2. Use only a fixed arbitrary constant threshold (e.g., $\tau = 0.5$).
  3. Implement two leakage-safe policies:
     - Policy A (Supervised Calibration): Grid search $\tau \in [0.01, 0.99]$ to maximize F1 strictly on the 64 validation scenes; freeze $\tau^*$, then apply to the 128 test scenes.
     - Policy B (Unsupervised): Compute scene-adaptive Otsu thresholding directly from histogram variance.
- **Decision Made:** Adopt Policy A (validation-calibrated frozen threshold) as the primary protocol for all classical comparisons, and Policy B (unsupervised Otsu) as a secondary zero-shot baseline. Test-set threshold tuning is strictly prohibited.
- **Rationale & Trade-offs:** Guarantees absolute isolation of the test split and ensures reproducible, scientifically honest benchmarking.
- **Status:** Confirmed
- **Evidence / Source:** Otsu (1979); standard machine learning evaluation integrity rules; `metrics.md` Section 3.

---

### DEC-018: Controlled Perturbation Robustness Test Suite
- **Date:** 2026-09-15
- **Owner:** Adishri Abro (Literature & Evaluation Lead)
- **Context:** Evaluating false-alarm resilience (RQ3) requires testing models under controlled, reproducible environmental confounders rather than relying solely on clean benchmark imagery.
- **Alternatives Considered:**
  1. Test only on clean benchmark pairs without perturbations.
  2. Apply arbitrary, unscientific image distortions (e.g., cartoon filters, extreme color inversion).
  3. Apply 4 literature-grounded physical confounders:
     - Global illumination scaling: $I_2' = \alpha I_2 + \beta$ ($\alpha \in [0.7, 1.3]$).
     - Spatial blur / atmospheric defocus: Gaussian blur kernel ($k \in \{3, 5, 7\}, \sigma \in [1.0, 2.0]$).
     - Geometric misregistration jitter: Sub-pixel and pixel translations ($\Delta x, \Delta y \in \{0.5, 1.0, 2.0\}$ pixels).
     - Synthetic localized occlusion / shadow artifacts.
- **Decision Made:** Implement the 4 literature-grounded perturbation generators. Measure performance using Relative F1 Degradation ($\Delta F_1$) and False Positive Amplification Factor (FPAF).
- **Rationale & Trade-offs:** Isolates specific environmental failure modes and provides quantitative evidence for whether learned representations or quality-gating layers resist non-ground confounders.
- **Status:** Confirmed
- **Evidence / Source:** Hall et al. (1991); Bovolo & Bruzzone (2007); `literature_review.md` Section 6; `metrics.md` Section 5.

---

### DEC-019: Novelty and Research-Gap Classification of Proposed Quality-Aware Method
- **Date:** 2026-09-15
- **Owner:** Adishri Abro (Literature & Evaluation Lead)
- **Context:** Determining whether the project's proposed quality-aware change-detection architecture is a fundamentally novel theoretical invention or an adaptation of existing principles, to avoid making unsupported academic claims.
- **Alternatives Considered:**
  1. Claim foundational novelty for quality-aware change scoring.
  2. Abandon the quality-aware investigation entirely because prior art exists.
  3. Classify the method as an **ADAPTATION & COMBINATION**: acknowledge that radiometric normalization, uncertainty estimation, and confidence maps are established in remote sensing, while positioning our specific contribution as a lightweight, CPU-constrained diagnostic gating pipeline evaluated under controlled perturbation stress tests.
- **Decision Made:** Formally classify the proposed method as an **ADAPTATION & COMBINATION**. Remove any claims of foundational theoretical novelty from all project documentation. The research contribution is explicitly defined as an empirical investigation of lightweight diagnostic gating for false-alarm suppression under CPU constraints.
- **Rationale & Trade-offs:** Scientific honesty and intellectual rigor are paramount. Acknowledging established prior art strengthens the academic credibility of the research.
- **Status:** Confirmed
- **Evidence / Source:** Comprehensive literature audit (`literature_review.md` Sections 7, 11, and 12).

