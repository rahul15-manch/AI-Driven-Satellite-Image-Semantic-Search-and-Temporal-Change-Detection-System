# Project Development Log (devlog.md)

All engineering activities, architectural milestones, and experimental progress are documented here in reverse-chronological order.

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
