# AI-Driven Satellite Image Semantic Search and Temporal Change Detection System

**A Research Study in Artificial Intelligence, Machine Learning, Computer Vision, and Geospatial AI**

[![Project Track](https://img.shields.io/badge/Academic%20Track-Final--Year%20B.Tech%20CSE%20(AI%2FML)-blue)](#)
[![Execution Platform](https://img.shields.io/badge/Compute-Commodity%20CPU%20Only-brightgreen)](#)
[![Status](https://img.shields.io/badge/Milestones-M1,%20M2,%20M3%20Completed-success)](#)

---

## 1. Project Overview

This project is a research-oriented study investigating two fundamental computer vision capabilities applied to remote sensing and Earth observation:

1. **Semantic Satellite Image Retrieval:** Cross-modal search enabling users to discover remote sensing scenes through natural-language queries (e.g., *"areas with newly built-up zones near rivers"*), comparing classical text-tag matching against dual-encoder vision-language embeddings indexed with FAISS.
2. **Bi-Temporal Change Detection:** Pixel-level and patch-level change detection between co-registered optical satellite image pairs ($T_1, T_2$), with special focus on quantifying and mitigating false-alarm detections induced by non-ground changes (illumination disparities, seasonal vegetation phenology, cloud shadows, and geometric misregistration).

---

## 2. Core Principles & Computational Constraints

- **Strict Empirical Research Philosophy:** All conclusions must be supported by reproducible experiments, comparative baseline evaluations, ablation studies, and failure-mode analysis. We make zero assumptions of prior superiority for any proposed model.
- **Strict CPU-Only Execution:** Designed to execute on standard student laptops without requiring discrete CUDA GPUs, cloud compute, or paid proprietary AI APIs.
- **Anti-Fabrication Standards:** Strict separation between verified literature, working assumptions, proposed hypotheses, and experimentally established findings.

---

## 3. Team Structure & 15-Milestone Roadmap

The project is organized into 15 structured milestones, with 5 primary milestones assigned to each team member:

| Member | Role | Assigned Milestones |
| :--- | :--- | :--- |
| **Rahul** | Team Lead | **M1** (Architecture & Definition), **M4** (Retrieval Baseline), **M7** (VLM Integration), **M10** (FAISS Index Tuning), **M13** (System Integration & API) |
| **Tanishka Mukhi** | Research Contributor | **M2** (Dataset Pipeline), **M5** (Classical CD Baselines), **M8** (Learned CD Model), **M11** (Robustness & Perturbations), **M14** (Analyst Interface) |
| **Adishri Abro** | Research Contributor | **M3** (Literature & Baselines), **M6** (Evaluation Suite), **M9** (False-Alarm Diagnostics), **M12** (Comprehensive Profiling), **M15** (Final Research Thesis) |

---

## 4. Research Documentation Index

The research foundation is documented in detail across the following records:

- [`literature_review.md`](./literature_review.md) — Comprehensive literature review, citation audit, research gaps, novelty classification, and verified references.
- [`metrics.md`](./metrics.md) — Formal mathematical and operational specifications for retrieval, change-detection, computational, and perturbation metrics.
- [`research_questions.md`](./research_questions.md) — Formal research problem, research questions (RQ1–RQ4), formal hypotheses (H1–H4), and variables.
- [`experiment_plan.md`](./experiment_plan.md) — Empirical experimental protocols, candidate models, false-alarm perturbation tests, and ablation studies.
- [`data_card.md`](./data_card.md) — Authoritative data cards for LEVIR-CD and RSICD datasets.
- [`PRD.md`](./PRD.md) — Product / Research Requirements Document (problem statement, requirements, constraints, system boundaries).
- [`research_findings.md`](./research_findings.md) — Empirical research findings, verified benchmark metrics, leakage audit, and scientific insights.
- [`docs/m4_semantic_retrieval.md`](./docs/m4_semantic_retrieval.md) — Comprehensive empirical benchmark report for Milestone 4 (Leakage-controlled BM25 vs. Zero-Shot CLIP ViT-B/32 on CPU).
- [`architecture.md`](./architecture.md) — 8-layer decoupled modular architecture, data flows, CPU execution strategies, and milestone dependencies.
- [`decision_log.md`](./decision_log.md) — Formal record of architectural, technical, and research decisions (DEC-001 through DEC-025).
- [`devlog.md`](./devlog.md) — Chronological engineering and research progress log.

---

## 5. Current Project Status

- **Completed:**
  - **Milestone 1:** Research Definition & System Architecture (Owner: Rahul).
  - **Milestone 2:** Dataset Acquisition, Verification & Preprocessing Pipeline (Owner: Tanishka Mukhi).
  - **Milestone 3:** Comprehensive Literature Review, Mathematical Baseline Formulation & Metric Formalization (Owner: Adishri Abro).
  - **Milestone 4 (Corrected):** Semantic Retrieval Baseline & Leakage-Controlled Evaluation (Owner: Rahul).
    - 47 passing tests (100% pass rate in 4.93s), including 8 dedicated leakage regression tests.
    - Verified locally measured metrics on 1,093 RSICD test gallery images across 5,465 queries on CPU.
- **Upcoming:**
  - **Milestone 5:** Classical Change Detection Baselines Implementation (Owner: Tanishka Mukhi).
  - **Milestone 6:** Evaluation Framework & Diagnostic Benchmarking Suite (Owner: Adishri Abro).
- **Current State:** Datasets verified; complete literature foundation and formal metrics codified; leakage-controlled cross-modal retrieval baselines (Leave-One-Caption-Out BM25 vs. Category Metadata BM25 vs. Zero-Shot CLIP ViT-B/32) benchmarked on CPU with persistent caching. Ready for Classical Change Detection in M5.


