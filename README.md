# AI-Driven Satellite Image Semantic Search and Temporal Change Detection System

**A Research Study in Artificial Intelligence, Machine Learning, Computer Vision, and Geospatial AI**

[![Project Track](https://img.shields.io/badge/Academic%20Track-Final--Year%20B.Tech%20CSE%20(AI%2FML)-blue)](#)
[![Execution Platform](https://img.shields.io/badge/Compute-Commodity%20CPU%20Only-brightgreen)](#)
[![Status](https://img.shields.io/badge/Milestone-M1%20Completed-success)](#)

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

## 4. Documentation Index

The foundation of the project is documented in detail in the following research records:

- [`PRD.md`](./PRD.md) — Product / Research Requirements Document (problem statement, requirements, constraints, system boundaries, candidate datasets).
- [`research_questions.md`](./research_questions.md) — Formal research problem, provisional research questions (RQ1–RQ4), provisional hypotheses (H1–H4), and variables.
- [`architecture.md`](./architecture.md) — 8-layer decoupled modular architecture, data flows, CPU execution strategies, and milestone dependencies.
- [`experiment_plan.md`](./experiment_plan.md) — Empirical experimental protocols, candidate models, false-alarm perturbation tests, and ablation studies.
- [`decision_log.md`](./decision_log.md) — Formal record of architectural, technical, and research decisions (DEC-001 through DEC-007).
- [`devlog.md`](./devlog.md) — Chronological engineering and research progress log.

---

## 5. Current Project Status

- **Completed:** Milestone 1 — Research Definition & System Architecture (Owner: Rahul).
- **Upcoming:**
  - **Milestone 2:** Dataset Acquisition, Verification & Preprocessing Pipeline (Owner: Tanishka Mukhi).
  - **Milestone 3:** Comprehensive Literature Review, Mathematical Baseline Formulation & Metric Formalization (Owner: Adishri Abro).
- **Current State:** Research foundation established. No model training, API code, or application code has been implemented yet.
