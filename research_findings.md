# Research Findings & Empirical Knowledge Base

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Academic Context:** Final-Year B.Tech CSE (AI/ML) Research Study Project  
**Team Members:** Rahul (Team Lead), Tanishka Mukhi (Dataset & Architecture), Adishri Abro (Literature & Evaluation)  
**Hardware Profile:** Strictly CPU-Only Execution ($\le 8$ GB RAM budget, commodity multi-core CPU)  
**Last Updated:** 2026-09-15 (Post-M4 Corrective Task)  

---

## 1. Executive Summary of Research Findings

This document records the empirical results, scientific interpretations, and verified conclusions across all project milestones. In accordance with the foundational research principles of the project, **every number in this document originates from actual local execution on verified datasets and hardware**, with zero fabrication or synthetic inflation.

---

## 2. Milestone 4: Semantic Retrieval Baseline & Leakage-Controlled Evaluation

### 2.1 Research Question 1 (RQ1) & Hypothesis 1 (H1) Summary
- **Research Question RQ1:** *"How effectively can vision-language embeddings retrieve semantically relevant satellite images from natural-language queries, compared with a lexical retrieval baseline, under CPU-only constraints?"*
- **Hypothesis H1:** *"Vision-language embedding retrieval may provide more semantically relevant results than simple lexical retrieval."*
- **Empirical Status of H1:** **PARTIALLY SUPPORTED / INCONCLUSIVE (Dependent on Gallery Modality)** (`DEC-025`).

### 2.2 Methodological Problem & Corrective Audit
An audit of the initial M4 implementation identified a critical methodological problem in the BM25 lexical baseline:
- The initial baseline concatenated all 5 human captions describing each test image into a single document ($D_i = C_{i1} \mathbin{\Vert} \dots \mathbin{\Vert} C_{i5}$).
- The exact same captions were simultaneously used as the 5,465 evaluation queries.
- Therefore, when query $q = C_{ik}$ was evaluated, its exact text tokens were already present inside target image $I_i$'s gallery document.
- This self-leakage inflated BM25 scores to $85.65\%$ R@1 and $0.9028$ MRR, rendering it scientifically incomparable against CLIP image-only retrieval.

### 2.3 Corrective Protocols Implemented
To establish a rigorous, leakage-free benchmark without discarding historical context, three protocols were established:
1. **Mode A — Leave-One-Caption-Out Caption-Indexed BM25 (Corrected Primary Baseline):**
   - For each query $q = C_{ik}$, the query caption is strictly removed from target image $I_i$'s BM25 document representation, leaving only the remaining 4 captions ($\bigcup_{j \ne k} C_{ij}$).
   - Non-target gallery images retain all 5 captions.
   - Exact query text is verified absent from the target document via automated regression tests (`tests/test_leakage_regression.py`).
2. **Mode B — Category Metadata Lexical Baseline (Auxiliary Image Control):**
   - Gallery images are represented strictly by their single category metadata label (e.g., `"airport"`, `"parking"`).
   - Zero human captions exist in the gallery, establishing a direct lexical equivalent to uncaptioned satellite imagery.
3. **Legacy Diagnostic Baseline (Preserved Reference):**
   - The original combined-document BM25 results are preserved in `experiments/results/m4/legacy_caption_indexed/` for auditability.

---

## 3. Locally Measured Benchmark Table `[LOCALLY MEASURED]`

*Dataset:* RSICD Test Split (1,093 gallery images, 5,465 caption queries).  
*Hardware:* Commodity multi-core CPU (strict CPU-only execution).

| Method | Protocol | Information Available to Gallery | R@1 (%) | R@5 (%) | R@10 (%) | MRR | Mean Latency (ms) | p95 Latency (ms) | Peak Process RAM (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Leave-One-Caption-Out)** | Leave-One-Caption-Out Caption-Indexed | 4 remaining human captions per target image ($q$ excluded); 5 for non-targets | **42.12** | **60.81** | **67.87** | **0.5112** | **1.49** | **2.72** | **279.7** |
| **BM25 (Category Metadata)** | Category Metadata Lexical Control | Category label only (e.g., `"airport"`); zero test captions in gallery | 1.50 | 7.30 | 14.35 | 0.0618 | 0.12 | 0.18 | 287.3 |
| **BM25 (Original Diagnostic / Leaky)** | Original Caption-Indexed (Self-Leakage) | All 5 human captions concatenated per image (exact query $q$ present in target) | 85.65 | 96.38 | 98.57 | 0.9028 | 1.46 | 2.57 | 357.4 |
| **CLIP ViT-B/32 (Zero-Shot)** | Zero-Shot Image-Text Multimodal Retrieval | Visual image pixels only (L2-normalized 512D embeddings in FAISS); zero text in gallery | 5.45 | 17.71 | 27.89 | 0.1307 | 7.54 | 8.84 | 883.6 |
| **CLIP + Prompt Ensemble** | Prompt Ensemble Multimodal Retrieval | Visual image pixels only; 5 frozen RS prompt-averaged query vectors | 5.14 | 17.00 | 28.01 | 0.1268 | 15.56 | 21.14 | 553.7 |

*FAISS Vector Search Latency:* Pre-computed IndexFlatIP inner-product search requires **0.050 ms** on CPU.

---

## 4. Key Scientific Insights

### 4.1 Quantifying the Effect of Target-Query Lexical Leakage
- Eliminating target-query leakage causes BM25 top-1 recall to drop from **85.65% down to 42.12%** (a $-43.53\%$ absolute decrease) and MRR to drop from **0.9028 down to 0.5112** (a $-0.3916$ decrease).
- This confirms that **over 50% of legacy BM25 performance was an artifact of target-text overlap**, not generalizable cross-description lexical matching.

### 4.2 Information Asymmetry & The Two Faces of H1
The comparison between lexical search and vision-language embeddings depends critically on what information is available in the gallery:
1. **The Uncaptioned Imagery Scenario (CLIP vs Mode B):**
   - In realistic geospatial search systems, satellite image repositories possess no pre-existing human captions.
   - When lexical search is constrained to metadata categories (Mode B), zero-shot CLIP outperforms lexical search by **$3.6\times$ on R@1** ($5.45\%$ vs $1.50\%$) and **$2.1\times$ on MRR** ($0.1307$ vs $0.0618$).
   - CLIP demonstrates genuine semantic understanding, successfully matching descriptive queries (e.g., *"a large grassy field with planes"*) to image pixels even when category keywords are omitted.
2. **The Caption-Indexed Scenario (Mode A vs CLIP):**
   - When comprehensive human captions are attached to every gallery image, lexical retrieval (Mode A: $42.12\%$ R@1) outperforms zero-shot CLIP ($5.45\%$ R@1).
   - This occurs because different human annotators share distinctive descriptive vocabulary for the same scene, and generalist CLIP suffers from nadir-view domain shift.

### 4.3 Instance-Level vs Category-Level Retrieval
- The RSICD Caption-to-Own-Image benchmark treats retrieval strictly as an **instance-level discrimination task** (only 1 specific image out of 1,093 is considered relevant).
- Qualitative analysis indicates that in over $75\%$ of queries where CLIP misses the specific target instance (rank $> 10$), CLIP correctly retrieves other images belonging to the **exact same semantic category** (e.g. retrieving 5 valid airport scenes for an airport query).
- Thus, zero-shot CLIP ViT-B/32 functions effectively as a **semantic category locator**, but requires remote-sensing domain adaptation (fine-tuning) to achieve fine-grained instance discrimination.

### 4.4 Domain Prompt Ensembling Ablation
- Ensembling across 5 frozen remote-sensing templates (`"{query}"`, `"a satellite image of {query}"`, etc.) yields an R@1 of **5.14%** (vs 5.45% raw query) and MRR of **0.1268** (vs 0.1307).
- While prompt ensembling improves specific ambiguous queries (e.g. moving `airport_348.jpg` from rank 16 to rank 3), averaging template embeddings slightly softens high-confidence discriminative queries across the broad test distribution, while doubling query encoding latency from **7.54 ms to 15.56 ms** on CPU.

### 4.5 CPU Feasibility & Pareto Frontier (RQ4)
- All evaluated retrieval methods operate well within the target commodity hardware constraints:
  - BM25 latency: **1.49 ms**
  - CLIP latency: **7.54 ms** (132 queries/sec throughput on CPU)
  - FAISS vector search: **0.050 ms**
  - Peak process RSS: **883.6 MB** (well below the 8 GB laptop RAM ceiling)
  - Index memory footprint: **2.18 MB** for 1,093 vectors (512D float32)

---

## 5. Failure Analysis on CLIP Zero-Shot Retrieval

An automatic rule-based keyword heuristic classification applied across all 3,941 failure cases (queries where target rank $> 10$) revealed:
1. **Broad Geographic Context Ambiguity (36.260%, 1,429 queries):** Queries with general scene terms match dozens of near-identical gallery images (e.g. 60 airports, 60 bareland scenes), causing the single target to be displaced by valid peers.
2. **Spatial / Relational Confusion (32.378%, 1,276 queries):** CLIP struggles with directional and topological prepositions (*"next to"*, *"between"*, *"surrounded by"*).
3. **Dense Small Objects (20.046%, 790 queries):** ViT-B/32 ($32 \times 32$ pixel patches) lacks the spatial resolution to resolve individual cars or small airplanes at satellite scale.
4. **Fine-Grained Attribute Mismatch (11.292%, 445 queries):** Mismatches in subtle color or geometric shape descriptors.
5. **Generic / Repetitive Caption (0.025%, 1 query):** Extremely short caption ($\le 4$ tokens).

---

## 6. Machine-Readable Result Artifacts

All empirical findings are preserved in machine-readable files under `experiments/results/m4/`:
- `summary.csv` & `summary.json`: Summary table with mandatory `protocol` and `information_available` fields.
- `corrected_bm25_results.json`: Query-level and aggregate metrics for Mode A (Leave-One-Caption-Out BM25).
- `category_metadata_bm25_results.json`: Query-level and aggregate metrics for Mode B (Category Metadata BM25).
- `legacy_caption_indexed/bm25_results.json`: Preserved legacy diagnostic results.
- `clip_results.json`: Query-level and category breakdown metrics for Zero-Shot CLIP ViT-B/32.
- `prompt_ensemble_results.json`: Query-level and category breakdown metrics for CLIP + Prompt Ensemble.
- `qualitative_samples.json`: Multi-method side-by-side rankings for deterministic query samples.
- `failure_analysis.json`: Precise breakdown of the 3,941 CLIP failure cases with documented classification heuristic.

---

## 7. Milestone Boundaries & Explicit Non-Implementation

In accordance with strict milestone governance:
- **Zero change detection algorithms (M5/M6) have been implemented:** No pixel differencing, SSIM, CVA, or Siamese neural networks were executed.
- Milestone 4 was strictly confined to cross-modal semantic retrieval.
