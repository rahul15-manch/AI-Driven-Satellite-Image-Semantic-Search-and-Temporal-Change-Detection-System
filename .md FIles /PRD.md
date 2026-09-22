# Product / Research Requirements Document (PRD)

**Project Title:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Document Version:** 1.0.0 (Milestone 1 — Baseline Foundation)  
**Date:** 2026-09-15  
**Project Track:** Final-Year B.Tech CSE (AI/ML) Research Study  
**Authors / Team:**
- Rahul (Team Lead) — Milestones M1, M4, M7, M10, M13
- Tanishka Mukhi — Milestones M2, M5, M8, M11, M14
- Adishri Abro — Milestones M3, M6, M9, M12, M15

---

## 1. Executive Summary & Purpose

The **AI-Driven Satellite Image Semantic Search and Temporal Change Detection System** is an academic research and engineering prototype designed to study two foundational challenges in Earth Observation (EO) and Geospatial Computer Vision:
1. **Cross-Modal Semantic Retrieval:** Enabling analysts to query satellite image repositories using unconstrained natural-language queries (e.g., *"areas with newly built-up zones near rivers"* or *"dense commercial harbor with cargo ships"*).
2. **Bi-Temporal Change Detection:** Identifying surface-level geometric and land-cover transformations between co-registered satellite image pairs captured at different timestamps ($T_1, T_2$), while systematically investigating and mitigating false alarms caused by non-ground changes (illumination, shadow shifts, seasonal foliage shifts, atmospheric haze, and registration jitter).

This project is framed strictly as an **empirical research study** rather than a commercial software deployment. Its objective is to benchmark classical algorithms against lightweight machine learning approaches under stringent, accessible hardware constraints (standard student laptops, CPU-only execution).

---

## 2. Problem Statement

Satellite imagery volume has grown exponentially, yet downstream utilization faces two primary bottlenecks:
- **Retrieval Bottleneck:** Traditional search in remote sensing archives relies on metadata tags (sensor ID, timestamp, bounding coordinates, or broad land-use categories). Analysts cannot easily search by semantic, visual, or contextual attributes without manual annotation or expensive post-processing.
- **Change Detection & False-Positive Bottleneck:** Bi-temporal change detection algorithms frequently flag irrelevant radiometric variations as true surface changes. Solar angle variations, seasonal phenology (dry vs. wet grass), transient cloud shadows, sensor calibration disparities, and sub-pixel misregistration yield high false-positive rates, exhausting manual inspection resources.

Existing deep-learning solutions in the literature often rely on compute-heavy vision-language backbones and massive multi-scale transformer models trained on multi-GPU server clusters. There is a lack of systematic evaluation of **lightweight, CPU-executable baselines and hybrid pipelines** designed for standard, resource-constrained research environments.

---

## 3. User Personas & Target Use Cases

| Persona | Role / Domain | Core Use Case | System Interaction |
| :--- | :--- | :--- | :--- |
| **Geospatial Analyst** | Urban Planning & Infrastructure Monitoring | Locating unauthorized construction or rapid urban expansion. | Enters semantic query $\rightarrow$ inspects top candidates $\rightarrow$ runs bi-temporal change detection on selected region pairs. |
| **Environmental Researcher** | Deforestation & Water Resource Management | Tracking water-body shrinkage or forest canopy degradation over seasons. | Queries bi-temporal archives for riparian zones $\rightarrow$ reviews change heatmaps with false-positive confidence flags. |
| **Disaster Response Assessor** | Post-Event Damage Assessment | Identifying structural collapse, flooded roads, or debris accumulation. | Submits pre- and post-event satellite pairs $\rightarrow$ inspects change masks isolated from lighting disparities. |

---

## 4. Core Capabilities

### Capability A: Semantic Satellite Image Retrieval
- Ingests natural-language text queries from the user.
- Projects queries and image candidates into a shared multi-modal embedding space.
- Performs high-throughput nearest-neighbor vector similarity search.
- Returns a ranked list of top-$K$ satellite scenes with associated similarity scores.

### Capability B: Bi-Temporal Change Detection
- Ingests a co-registered pair of satellite images ($T_1$ and $T_2$) covering the same geographical area.
- Preprocesses and normalizes radiometric and geometric characteristics.
- Computes pixel-level or patch-level change likelihoods using comparative algorithms.
- Produces a binary change mask and an intensity change heatmap.

### Capability C: Quality-Aware False-Alarm Diagnostic (Proposed)
- Analyzes potential confounding factors (illumination disparity, shadow shifts, cloud heuristics, registration misalignment).
- Augments the change map with a confidence/reliability index to distinguish probable physical ground changes from acquisition artifacts.

---

## 5. Research Objectives & Success Criteria

### 5.1 Research Objectives
1. **Benchmark Cross-Modal Retrieval:** Measure retrieval effectiveness of vision-language embeddings compared to a classical keyword/metadata retrieval baseline on remote sensing benchmarks.
2. **Empirical Comparison of Change Detection:** Systematically benchmark classical image-differencing/structural techniques against lightweight learned models under identical CPU constraints.
3. **Analyze False Alarms:** Quantitatively evaluate the degradation of change-detection models under simulated or natural non-ground disturbances (illumination, clouds, shadows, misregistration).
4. **Hardware Profiling:** Document wall-clock latency, throughput, peak memory (RAM), and model footprint across all evaluated pipelines on standard CPU hardware.

### 5.2 Success Criteria
- **Empirical Validity:** All quantitative assertions are backed by reproducible benchmark scripts with fixed random seeds and recorded metrics.
- **Baseline Integrity:** Classical methods (pixel differencing, SSIM) are properly implemented, calibrated, and evaluated on identical test splits before evaluating complex models.
- **Ablation & Trade-off Analysis:** Every proposed enhancement (e.g., quality gating or confidence scoring) is ablated to prove whether it adds measurable statistical value.
- **Zero-Fabrication Adherence:** No metric, citation, or benchmark score is claimed without documented code execution or verified external literature.
- **CPU Viability:** Full end-to-end inference for both retrieval ($K=10$ across test index) and change detection (tile size $\ge 256 \times 256$) executes locally within reasonable interactive limits ($< 5$ seconds per query/pair on modern quad-core laptop CPUs).

---

## 6. Functional Requirements (FR)

- **FR-1 (Query Input):** The system shall accept arbitrary ASCII/Unicode natural-language text queries.
- **FR-2 (Text Embedding):** The system shall encode text queries into fixed-dimensional vectors using a designated lightweight vision-language text encoder.
- **FR-3 (Image Indexing & Retrieval):** The system shall maintain an indexed database of satellite scene embeddings (FAISS-cpu) and return the top-$K$ nearest neighbors ranked by cosine similarity or inner product.
- **FR-4 (Pairwise Image Ingestion):** The system shall ingest bi-temporal image pairs ($T_1, T_2$) in standard raster formats (PNG, JPEG, TIFF/GeoTIFF).
- **FR-5 (Baseline Change Detection):** The system shall provide deterministic classical change-detection outputs (e.g., Absolute Differencing, SSIM, and CVA where applicable).
- **FR-6 (Learned Change Detection):** The system shall provide a lightweight learned change-detection pipeline (e.g., lightweight Siamese CNN) runnable on CPU.
- **FR-7 (False-Alarm Diagnostics):** The system shall output diagnostic metadata or confidence maps identifying suspected non-ground artifacts.
- **FR-8 (Evaluation Logging):** The system shall log all test run predictions, ground-truth masks, execution times, and calculated metrics to persistent disk logs.

---

## 7. Non-Functional Requirements (NFR)

- **NFR-1 (Compute Constraint):** All runtime components (data loading, inference, vector indexing, evaluation) must run strictly on commodity CPU hardware without requiring CUDA, ROCm, or external GPU drivers.
- **NFR-2 (Memory Footprint):** Peak operational memory consumption during inference must not exceed 8 GB RAM, ensuring stable execution on laptops with 8 GB to 16 GB total system RAM.
- **NFR-3 (Storage Constraint):** Dataset subsets and local model weights must fit within a modest storage allocation ($< 15$ GB total disk space).
- **NFR-4 (Explainability):** Code must be modular, fully commented, and mathematically transparent, prioritizing clear algorithmic steps over complex external framework wrappers.
- **NFR-5 (Reproducibility):** Preprocessing, data splitting, training, and evaluation scripts must specify fixed seeds (`random_state=42`) and deterministic operations wherever supported.
- **NFR-6 (Zero Paid Services):** No paid third-party APIs (e.g., OpenAI, Google Earth Engine commercial tier) are permitted; all dependencies must be free and open-source.

---

## 8. Proposed Datasets (Status: Proposed / To Be Verified)

> [!WARNING]
> Dataset properties listed below are proposed initial directions based on project synopsis requirements. Their exact sizes, splits, licensing terms, and CPU processing feasibility must be formally verified in Milestone 2 and Milestone 3.

1. **RSICD (Remote Sensing Image Captioning Dataset)**
   - *Intended Purpose:* Cross-modal semantic retrieval experiments (evaluating text-to-image search).
   - *Characteristics to Verify:* Number of aerial/satellite scenes (~10,921 images reported in literature), image resolution ($224 \times 224$), caption diversity (5 captions per image), class distribution, and disk storage requirements.
   - *Status:* Proposed / To Be Verified.
2. **LEVIR-CD (Large-scale Building Change Detection Dataset)**
   - *Intended Purpose:* Bi-temporal change-detection experiments.
   - *Characteristics to Verify:* High-resolution Google Earth bi-temporal pairs ($1024 \times 1024$), building construction/demolition masks, patch tiling requirements ($256 \times 256$ sub-crops for CPU processing), download size (~10 GB), and split verification.
   - *Status:* Proposed / To Be Verified.

---

## 9. Proposed Technology Stack

- **Core Programming Language:** Python 3.10+
- **Deep Learning / Tensor Math:** PyTorch (CPU-only distribution)
- **Computer Vision & Image Processing:** OpenCV (`opencv-python-headless`), Pillow (`PIL`), scikit-image
- **Vector Search & Indexing:** FAISS (`faiss-cpu`)
- **Scientific Computing & Dataframes:** NumPy, Pandas, SciPy, scikit-learn
- **Geospatial Utilities (if needed):** Rasterio (for GeoTIFF metadata handling, if verified)
- **Future Serving / API (Milestone 13+):** FastAPI, Uvicorn
- **Future Lightweight UI (Milestone 14+):** Vanilla HTML5, Modern CSS, Vanilla JavaScript (zero heavy node-framework dependencies)

---

## 10. System Boundaries & Out-of-Scope Items

### In-Scope:
- Local offline dataset preprocessing, tiling, and normalization.
- CPU inference benchmarking of baseline vs. lightweight learned models.
- Controlled synthetic and natural perturbation studies for false-alarm analysis.
- Standalone command-line and programmatic evaluation suites.
- Lightweight local web demonstration interface (in final milestones).

### Explicitly Out-of-Scope:
- Multi-GPU distributed training and high-performance cluster computing.
- Training multi-billion parameter foundation models from scratch.
- Real-time ingestion of live orbital satellite downlink telemetry.
- 3D point-cloud and SAR (Synthetic Aperture Radar) processing (focus is optical/multispectral satellite scenes).
- Commercial production deployment with auto-scaling, Kubernetes, or cloud SaaS infrastructure.
