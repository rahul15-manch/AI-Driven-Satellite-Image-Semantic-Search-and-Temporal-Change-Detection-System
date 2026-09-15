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
