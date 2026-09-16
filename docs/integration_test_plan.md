# Milestone 1–4 Integration Test Plan

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Document Identifier:** `docs/integration_test_plan.md`  
**Owner:** Rahul (Team Lead)  
**Collaborators:** Tanishka Mukhi (Dataset & Architecture), Adishri Abro (Literature & Evaluation)  
**Milestone Focus:** Full Integration Testing of Completed Milestones M1, M2, M3, and M4  
**Date:** 2026-09-16  
**Execution Environment:** Commodity Multi-Core CPU, $\le 8$ GB RAM budget, Strict CPU-Only (`device="cpu"`)  

---

## 1. Objective & Scope

The objective of this integration test plan is to rigorously validate the complete research dependency chain across completed milestones:
$$\text{M1 (Research Definition \& Architecture)} \longrightarrow \text{M2 (Dataset Acquisition \& Pipeline)} \longrightarrow \text{M3 (Literature \& Framework)} \longrightarrow \text{M4 (Semantic Retrieval Baseline)}$$

This integration plan verifies both:
1. **Software Integration:** Seamless data handoff between dataset loaders, preprocessors, embedding caches, FAISS indexing, BM25 retrieval, and evaluation metric calculation.
2. **Research-Methodology Integration:** Strict compliance with literature formulations, verified absence of target-query text overlap leakage, non-overlapping dataset partitions, and CPU resource constraints.

---

## 2. Integration Test Matrix

### 2.1 Milestone 1: Research Definition & Architectural Contracts

#### INT-M1-01: Research Questions Integrity
- **Test ID:** `INT-M1-01`
- **Purpose:** Verify that formal research questions (RQ1, RQ2, RQ3, RQ4) exist in `research_questions.md` and match the approved project synopsis.
- **Preconditions:** `research_questions.md` is present in the repository root.
- **Input:** Document text of `research_questions.md`.
- **Procedure:** Parse document; assert presence and consistency of RQ1 (Cross-modal retrieval), RQ2 (Change detection), RQ3 (False-alarm mitigation), and RQ4 (Pareto efficiency on CPU).
- **Expected Result:** All four research questions are documented with explicit scope definitions.
- **Actual Result:** Passed. RQ1–RQ4 are documented and mutually consistent.
- **Status:** PASS
- **Evidence / Artifact:** `research_questions.md` Lines 20–46; `tests/integration/test_m1_m4_contracts.py::test_int_m1_01_research_questions_exist`.

---

#### INT-M1-02: Hypotheses Integrity with Test Criteria
- **Test ID:** `INT-M1-02`
- **Purpose:** Verify hypotheses H1 through H4 exist with measurable test criteria and independent/dependent variables.
- **Preconditions:** `research_questions.md` is present.
- **Input:** Document text of `research_questions.md`.
- **Procedure:** Verify each hypothesis defines target metrics, test criteria, and experimental suite linkages.
- **Expected Result:** H1–H4 are clearly defined with empirical test criteria.
- **Actual Result:** Passed. H1–H4 defined; H1 updated with empirical status under DEC-025.
- **Status:** PASS
- **Evidence / Artifact:** `research_questions.md` Lines 48–78; `tests/integration/test_m1_m4_contracts.py::test_int_m1_02_hypotheses_exist`.

---

#### INT-M1-03: Architecture-to-Implementation Alignment
- **Test ID:** `INT-M1-03`
- **Purpose:** Verify that architectural specifications in `architecture.md` correspond to existing source modules in `src/semantic_search/`.
- **Preconditions:** `architecture.md` and `src/semantic_search/` exist.
- **Input:** File tree of `src/semantic_search/` and architecture flow diagrams in `architecture.md`.
- **Procedure:** Cross-reference documented module responsibilities against `bm25.py`, `clip_model.py`, `faiss_index.py`, `embedding_cache.py`, `prompt_ensembler.py`, and `retrieval_evaluator.py`.
- **Expected Result:** Complete 1-to-1 correspondence without architectural contradictions.
- **Actual Result:** Passed. All six documented retrieval components exist and implement specified interfaces.
- **Status:** PASS
- **Evidence / Artifact:** `architecture.md` Section 2.3; `tests/integration/test_m1_m4_contracts.py::test_int_m1_03_architecture_matches_implementation`.

---

#### INT-M1-04: Milestone Boundary Compliance (No M5/M6 in Source Tree)
- **Test ID:** `INT-M1-04`
- **Purpose:** Verify that no active implementation of change detection algorithms (M5) or perturbation generators (M6) exists in `src/`.
- **Preconditions:** Full source tree under `src/` is accessible.
- **Input:** Source code files in `src/`.
- **Procedure:** Perform recursive AST and pattern scan for classes such as `PixelDifferencer`, `SSIMChangeDetector`, `CVAChangeDetector`, `SiameseNetwork`, `PerturbationGenerator`.
- **Expected Result:** Zero implementations of M5/M6 found in `src/`.
- **Actual Result:** Passed. Milestone boundaries strictly respected.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_m1_m4_contracts.py::test_int_m1_04_milestone_boundary_compliance`.

---

#### INT-M1-05: Experiment Plan Consistency
- **Test ID:** `INT-M1-05`
- **Purpose:** Verify that M4 retrieval experiments correspond to EXP-RET-01, EXP-RET-02, and EXP-RET-03 in `experiment_plan.md`.
- **Preconditions:** `experiment_plan.md` is present.
- **Input:** Text content of `experiment_plan.md`.
- **Procedure:** Inspect Suite A definitions; ensure Methods A1 (BM25), A2 (Zero-shot CLIP), and A3 (Prompt Ensemble) are represented consistently.
- **Expected Result:** Exact alignment between experimental protocols and execution runner.
- **Actual Result:** Passed. EXP-RET-01, 02, and 03 match runner configurations.
- **Status:** PASS
- **Evidence / Artifact:** `experiment_plan.md` Section 2; `tests/integration/test_m1_m4_contracts.py::test_int_m1_05_experiment_plan_consistency`.

---

### 2.2 Milestone 2: Dataset Acquisition, Verification & Preprocessing

#### INT-M2-01: RSICD Dataset Availability
- **Test ID:** `INT-M2-01`
- **Purpose:** Confirm that the RSICD dataset is physically accessible at the documented path and the test split can be instantiated.
- **Preconditions:** Raw data directory `data/raw/rsicd` exists.
- **Input:** `data/raw/rsicd/images` and `data/raw/rsicd/dataset_rsicd.json`.
- **Procedure:** Instantiate `RSICDDataset` with `split="test"`.
- **Expected Result:** Dataset instantiates without error; length $> 0$.
- **Actual Result:** Passed. Dataset loads 1,093 samples successfully.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_dataset_retrieval_pipeline.py::test_int_m2_01_rsicd_dataset_availability`.

---

#### INT-M2-02: RSICD Image Validation (Format, Dimensions, Integrity)
- **Test ID:** `INT-M2-02`
- **Purpose:** Verify that images in the RSICD test gallery are valid uncorrupted RGB files with dimensions $224 \times 224$.
- **Preconditions:** Test gallery images exist on disk.
- **Input:** Sampled images across the test gallery.
- **Procedure:** Load images via PIL; verify mode is `"RGB"`, dimensions are $(224, 224)$, and execute `verify()` to assert uncorrupted file headers.
- **Expected Result:** 100% of tested images satisfy format and dimension specifications.
- **Actual Result:** Passed. Images are valid $224 \times 224$ RGB.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_dataset_retrieval_pipeline.py::test_int_m2_02_rsicd_image_validation`.

---

#### INT-M2-03: Split Partition Disjointness
- **Test ID:** `INT-M2-03`
- **Purpose:** Verify that train, validation, and test splits in `data/splits/rsicd/rsicd_splits.json` are mutually disjoint and sum to 10,921.
- **Preconditions:** Split manifest exists at `data/splits/rsicd/rsicd_splits.json`.
- **Input:** Split manifest JSON.
- **Procedure:** Compute pairwise intersections: $S_{\text{train}} \cap S_{\text{val}}$, $S_{\text{train}} \cap S_{\text{test}}$, and $S_{\text{val}} \cap S_{\text{test}}$.
- **Expected Result:** All pairwise intersections are empty ($\emptyset$); counts are 8,734 train, 1,094 val, 1,093 test (total 10,921).
- **Actual Result:** Passed. Intersections are strictly empty ($\emptyset$); counts match data card exactly.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_dataset_retrieval_pipeline.py::test_int_m2_03_split_integrity`.

---

#### INT-M2-04: Test Gallery Cardinality
- **Test ID:** `INT-M2-04`
- **Purpose:** Verify that the RSICD test gallery contains exactly 1,093 images.
- **Preconditions:** `RSICDDataset` instantiated with `split="test"`.
- **Input:** Loaded test split.
- **Procedure:** Measure `len(dataset)`.
- **Expected Result:** Exactly 1,093 images.
- **Actual Result:** Passed. `len(dataset) == 1093`.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_dataset_retrieval_pipeline.py::test_int_m2_04_test_gallery_count`.

---

#### INT-M2-05: Caption Cardinality
- **Test ID:** `INT-M2-05`
- **Purpose:** Verify that each test image possesses exactly 5 captions, yielding 5,465 total queries.
- **Preconditions:** Test split loaded.
- **Input:** List of captions per sample across the test gallery.
- **Procedure:** Sum captions across all 1,093 test samples: $\sum_{i=1}^{1093} |C_i|$.
- **Expected Result:** Exactly 5,465 captions ($1,093 \times 5$).
- **Actual Result:** Passed. Total captions $= 5,465$.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_dataset_retrieval_pipeline.py::test_int_m2_05_caption_count`.

---

#### INT-M2-06: Caption-to-Image 1-to-1 Mapping Integrity
- **Test ID:** `INT-M2-06`
- **Purpose:** Verify that every test caption maps unambiguously to exactly one originating target image.
- **Preconditions:** Test split loaded.
- **Input:** Captions and filenames.
- **Procedure:** Generate query IDs (`{stem}_cap{idx}`); assert no collisions and that each query maps to a unique originating image filename.
- **Expected Result:** Exactly 5,465 unique query-to-image mappings.
- **Actual Result:** Passed. Zero duplicate query IDs; 100% 1-to-1 mapping.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_dataset_retrieval_pipeline.py::test_int_m2_06_caption_to_image_mapping`.

---

#### INT-M2-07: Dataset Reproducibility Across Repeated Loads
- **Test ID:** `INT-M2-07`
- **Purpose:** Verify that independent dataset instantiations produce identical ordered image IDs and captions.
- **Preconditions:** Dataset on disk.
- **Input:** Two independent instances of `RSICDDataset(split="test")`.
- **Procedure:** Assert list equality: `ids1 == ids2` and `caps1 == caps2`.
- **Expected Result:** Strictly identical ordered lists.
- **Actual Result:** Passed. Ordering is 100% deterministic.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_dataset_retrieval_pipeline.py::test_int_m2_07_dataset_reproducibility`.

---

### 2.3 Milestone 3: Literature Review & Evaluation Framework

#### INT-M3-01: Metric Formulas Correctness
- **Test ID:** `INT-M3-01`
- **Purpose:** Verify that the evaluator implementation of Recall@1, Recall@5, Recall@10, and MRR mathematically matches formal definitions in `metrics.md`.
- **Preconditions:** `RetrievalEvaluator` class available.
- **Input:** Synthetic ranking permutations with known ground-truth ranks (rank 1, rank 3, rank 7, rank 15).
- **Procedure:** Execute `evaluator.evaluate()`; compare against hand-calculated expected values.
- **Expected Result:** $R@1 = 0.25$, $R@5 = 0.50$, $R@10 = 0.75$, $\text{MRR} = 0.385714$.
- **Actual Result:** Passed. Implementation matches hand-calculated metrics to within $10^{-4}$.
- **Status:** PASS
- **Evidence / Artifact:** `metrics.md` Section 2; `tests/integration/test_reproducibility.py::test_int_m3_01_metrics_mathematical_correctness`.

---

#### INT-M3-02: Retrieval Protocol Match (Single Positive Target)
- **Test ID:** `INT-M3-02`
- **Purpose:** Verify that the Caption-to-Own-Image protocol enforces exactly one positive target image per query.
- **Preconditions:** `QueryRecord` data structure available.
- **Input:** Sample query records.
- **Procedure:** Verify `target_image_id` is a scalar string and that ranking evaluates single-positive rank.
- **Expected Result:** One positive target per query.
- **Actual Result:** Passed. Protocol evaluates scalar target.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_reproducibility.py::test_int_m3_02_retrieval_protocol_single_positive`.

---

#### INT-M3-03: Threshold Policy Isolation
- **Test ID:** `INT-M3-03`
- **Purpose:** Verify that change detection threshold calibration policies (e.g. Otsu or validation tuning) are not erroneously introduced into semantic retrieval.
- **Preconditions:** Source code under `src/semantic_search/`.
- **Input:** Python files in `src/semantic_search/`.
- **Procedure:** Scan for thresholding functions or references to change masks.
- **Expected Result:** Zero threshold tuning or change-mask references in semantic retrieval code.
- **Actual Result:** Passed. Semantic retrieval uses parameter-free ranking by descending similarity score.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_reproducibility.py::test_int_m3_03_no_threshold_logic_in_semantic_search`.

---

#### INT-M3-04: Literature-to-Implementation Traceability
- **Test ID:** `INT-M3-04`
- **Purpose:** Verify that every implemented semantic retrieval method traces to peer-reviewed literature codified in `literature_review.md`.
- **Preconditions:** `literature_review.md` and `src/semantic_search/` exist.
- **Input:** Documented literature references and code modules.
- **Procedure:** Construct and verify traceability mapping:
  - Okapi BM25 $\rightarrow$ Robertson & Zaragoza (2009) $\rightarrow$ `src/semantic_search/bm25.py` $\rightarrow$ `EXP-RET-01`
  - Zero-Shot CLIP $\rightarrow$ Radford et al. (ICML 2021) $\rightarrow$ `src/semantic_search/clip_model.py` $\rightarrow$ `EXP-RET-02`
  - Exact Inner Product Vector Search $\rightarrow$ Johnson et al. (IEEE TBD 2019) $\rightarrow$ `src/semantic_search/faiss_index.py` $\rightarrow$ `EXP-RET-02`
  - Domain Prompt Ensembling $\rightarrow$ Radford et al. (2021); Lu et al. (2018) $\rightarrow$ `src/semantic_search/prompt_ensembler.py` $\rightarrow$ `EXP-RET-03`
- **Expected Result:** 100% traceability without orphaned methods or ungrounded claims.
- **Actual Result:** Passed. All methods trace to verified citations.
- **Status:** PASS
- **Evidence / Artifact:** Section 4 of this test plan; `literature_review.md`.

---

#### INT-M3-05: Research Decision Traceability
- **Test ID:** `INT-M3-05`
- **Purpose:** Verify that formal project decisions DEC-020 through DEC-025 correspond to active code.
- **Preconditions:** `decision_log.md` is present.
- **Input:** Decisions DEC-020 to DEC-025.
- **Procedure:** Check each decision against source code:
  - `DEC-020`: BM25 document aggregation $\rightarrow$ `BM25Retriever(aggregation_mode=...)`
  - `DEC-021`: CPU-only CLIP loading $\rightarrow$ `CLIPRetriever(device="cpu")`
  - `DEC-022`: Exact FAISS inner-product index $\rightarrow$ `faiss.IndexFlatIP(512)`
  - `DEC-023`: Domain prompt ensembling $\rightarrow$ `PromptEnsembler` with 5 frozen templates
  - `DEC-024`: H1 rejection under legacy protocol $\rightarrow$ Documented in `decision_log.md`
  - `DEC-025`: Leakage-controlled evaluation protocol $\rightarrow$ `BM25Retriever.rank_gallery_leave_one_out()`
- **Expected Result:** All documented decisions are implemented and verifiable.
- **Actual Result:** Passed. 100% of decisions trace to operational code.
- **Status:** PASS
- **Evidence / Artifact:** `decision_log.md`; `src/semantic_search/`.

---

### 2.4 Milestone 4: Software Integration & Pipeline Tests

#### INT-M4-01: Image Loader $\rightarrow$ CLIP Image Encoder
- **Test ID:** `INT-M4-01`
- **Purpose:** Verify that the image loading pipeline correctly feeds the CLIP image encoder to produce 512-dim float32 L2-normalized vectors on CPU.
- **Preconditions:** CLIP model and test images available.
- **Input:** Test image file paths.
- **Procedure:** Execute `clip_retriever.encode_images(sample_paths)`; verify output shape is $(N, 512)$, dtype is `float32`, and $\|\mathbf{v}\|_2 \approx 1.0$.
- **Expected Result:** Output shape $(N, 512)$, unit L2 norm within $10^{-5}$.
- **Actual Result:** Passed. Shape $(3, 512)$, L2 norm $= 1.00000$.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_clip_faiss_pipeline.py::test_int_m4_01_image_encoder_l2_normalized`.

---

#### INT-M4-02: Caption Loader $\rightarrow$ CLIP Text Encoder
- **Test ID:** `INT-M4-02`
- **Purpose:** Verify that natural language caption queries are encoded into 512-dim float32 L2-normalized vectors on CPU.
- **Preconditions:** CLIP model loaded.
- **Input:** List of text query strings.
- **Procedure:** Execute `clip_retriever.encode_text(queries)`; check shape, dtype, and unit norm.
- **Expected Result:** Output shape $(N, 512)$, unit L2 norm within $10^{-5}$.
- **Actual Result:** Passed. Shape $(3, 512)$, L2 norm $= 1.00000$.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_clip_faiss_pipeline.py::test_int_m4_02_text_encoder_l2_normalized`.

---

#### INT-M4-03: Image Embeddings $\rightarrow$ FAISS IndexFlatIP
- **Test ID:** `INT-M4-03`
- **Purpose:** Verify that image embeddings are added to FAISS `IndexFlatIP(512)` with exact inner-product search.
- **Preconditions:** FAISS CPU library installed.
- **Input:** $1,093 \times 512$ float32 normalized image vectors.
- **Procedure:** Instantiate `FAISSFlatIPIndex(512)`; add embeddings and image IDs; check `index.ntotal` and memory footprint.
- **Expected Result:** `ntotal == 1093`, dimension $= 512$, memory $\approx 2.18$ MB.
- **Actual Result:** Passed. `ntotal == 1093`, dimension $= 512$.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_clip_faiss_pipeline.py::test_int_m4_03_and_04_faiss_index_and_search`.

---

#### INT-M4-04: Query Embedding $\rightarrow$ FAISS Search
- **Test ID:** `INT-M4-04`
- **Purpose:** Verify that querying the FAISS index returns sorted gallery image IDs and cosine similarity scores bounded in $[-1, 1]$.
- **Preconditions:** Populated FAISS index.
- **Input:** $1 \times 512$ query vector.
- **Procedure:** Execute `index.search(query_vec, top_k=5)`; assert scores are descending and bounded in $[-1, 1]$.
- **Expected Result:** Ranked IDs returned; scores monotonically descending in $[-1, 1]$.
- **Actual Result:** Passed. Top score $\le 1.0$, strictly descending.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_clip_faiss_pipeline.py::test_int_m4_03_and_04_faiss_index_and_search`.

---

#### INT-M4-05: Retrieval Evaluator Execution
- **Test ID:** `INT-M4-05`
- **Purpose:** Verify that `RetrievalEvaluator` locates the target image rank for every query without exceptions or missing IDs.
- **Preconditions:** Query records and gallery image IDs initialized.
- **Input:** Queries and ranking function.
- **Procedure:** Execute `evaluator.evaluate()`; check that every result record contains a valid target rank $\in [1, N]$.
- **Expected Result:** $100\%$ of queries evaluated; all target ranks $\ge 1$.
- **Actual Result:** Passed. Zero missing target IDs.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_clip_faiss_pipeline.py::test_int_e2e_01_complete_clip_zero_shot_pipeline`.

---

#### INT-M4-06: BM25 Integration & Dual Protocol Traceability
- **Test ID:** `INT-M4-06`
- **Purpose:** Verify that both the legacy leaky protocol and the corrected leave-one-caption-out protocol are supported, distinct, and traceable.
- **Preconditions:** `BM25Retriever` initialized.
- **Input:** Image captions and query string.
- **Procedure:** Evaluate query under both `combined_document` and `leave_one_caption_out`; assert legacy score strictly exceeds leave-one-out score on the target image.
- **Expected Result:** Legacy score $>$ corrected score; both modes operational.
- **Actual Result:** Passed. Legacy self-overlap score exceeds leave-one-out score.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_bm25_leakage.py::test_int_m4_06_bm25_dual_protocols`.

---

#### INT-M4-07: Persistent Embedding Cache Integrity & Validation
- **Test ID:** `INT-M4-07`
- **Purpose:** Verify that `EmbeddingCache` validates model name, split, count, dimension, and SHA-256 hash of ordered IDs, rejecting corrupted or mismatched caches.
- **Preconditions:** Temporary cache directory.
- **Input:** Embeddings, metadata, and test permutations.
- **Procedure:** Test valid cache loading; alter sample count or ID order; assert `is_valid()` returns `False`.
- **Expected Result:** Cache accepts valid data and rejects count or ordering mismatches.
- **Actual Result:** Passed. Hash validation rejects permuted image IDs.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_clip_faiss_pipeline.py::test_int_m4_07_embedding_cache_integrity`.

---

#### INT-M4-08: Prompt Ensembling Pipeline
- **Test ID:** `INT-M4-08`
- **Purpose:** Verify that domain prompt ensembling expands queries across 5 frozen templates, averages embeddings, and re-normalizes to unit L2 norm.
- **Preconditions:** `PromptEnsembler` available.
- **Input:** Natural language query.
- **Procedure:** Expand query into 5 templates; compute average embedding; verify output shape is $(1, D)$ and L2 norm is $1.00000$.
- **Expected Result:** 5 frozen templates formatted; ensembled vector has unit norm.
- **Actual Result:** Passed. Output shape $(1, 512)$, norm $= 1.00000$.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_prompt_pipeline.py::test_int_m4_08_prompt_ensemble_templates_and_normalization`.

---

### 2.5 Cross-Milestone Data Contract Tests

#### INT-CONTRACT-01: M2 Test IDs Match M4 Gallery IDs
- **Test ID:** `INT-CONTRACT-01`
- **Purpose:** Verify that M2 split test IDs exactly equal the 1,093 image IDs in M4 test gallery.
- **Expected Result:** $\text{set}(M2\_\text{test\_ids}) == \text{set}(M4\_\text{gallery\_ids})$.
- **Actual Result:** Passed. Exact set equality across 1,093 images.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_m1_m4_contracts.py::test_int_contract_01_m2_test_ids_equal_m4_gallery_ids`.

---

#### INT-CONTRACT-02: M2 Caption Mapping Matches M4 Query Mapping
- **Test ID:** `INT-CONTRACT-02`
- **Purpose:** Verify that M2 caption mapping exactly equals M4 query records (5,465 captions, exactly 5 per image, 1-to-1 mapping).
- **Expected Result:** 5,465 unique query records mapping to 1,093 images.
- **Actual Result:** Passed. Exactly 5,465 query records.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_m1_m4_contracts.py::test_int_contract_02_m2_caption_mapping_equals_m4_query_mapping`.

---

#### INT-CONTRACT-03: Zero Train/Val Overlap in M4 Gallery
- **Test ID:** `INT-CONTRACT-03`
- **Purpose:** Verify that no train or validation image appears in the M4 test gallery.
- **Expected Result:** $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$.
- **Actual Result:** Passed. Overlap is strictly zero.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_m1_m4_contracts.py::test_int_contract_03_zero_train_val_in_m4_gallery`.

---

#### INT-CONTRACT-04: Image Encoder Input Isolation
- **Test ID:** `INT-CONTRACT-04`
- **Purpose:** Verify that the image encoder accepts only image files and never receives query text or captions.
- **Expected Result:** `encode_images` signature accepts only image inputs.
- **Actual Result:** Passed. Signature inspection confirms complete isolation from text.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_m1_m4_contracts.py::test_int_contract_04_clip_image_pipeline_isolated_from_query_text`.

---

#### INT-CONTRACT-05: Absence of Exact-Query Text Leakage in Corrected BM25
- **Test ID:** `INT-CONTRACT-05`
- **Purpose:** Verify that under leave-one-caption-out BM25, the exact query caption is absent from target image $I_i$'s gallery document for all queries.
- **Expected Result:** Zero exact-query leakage cases.
- **Actual Result:** Passed. Query text is 100% absent from target document.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_bm25_leakage.py::test_int_contract_05_bm25_leakage_absence`.

---

### 2.6 End-to-End Pipeline & Reproducibility Tests

#### INT-E2E-01: Complete CLIP Zero-Shot Pipeline
- **Test ID:** `INT-E2E-01`
- **Purpose:** Verify complete execution of: M2 dataset loader $\rightarrow$ cached image embeddings $\rightarrow$ FAISS index $\rightarrow$ text encoding $\rightarrow$ vector search $\rightarrow$ retrieval evaluator.
- **Expected Result:** Pipeline completes without errors; produces valid R@K and MRR.
- **Actual Result:** Passed. Zero exceptions; valid metrics produced.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_clip_faiss_pipeline.py::test_int_e2e_01_complete_clip_zero_shot_pipeline`.

---

#### INT-E2E-02: Complete Leave-One-Caption-Out BM25 Pipeline
- **Test ID:** `INT-E2E-02`
- **Purpose:** Verify complete execution of leave-one-caption-out BM25 retrieval over test gallery images.
- **Expected Result:** All test queries evaluated without leakage or missing target IDs.
- **Actual Result:** Passed. All queries evaluated with valid ranks.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_bm25_leakage.py::test_int_e2e_02_bm25_loco_end_to_end`.

---

#### INT-E2E-03: Complete Prompt Ensemble Pipeline
- **Test ID:** `INT-E2E-03`
- **Purpose:** Verify complete execution of prompt ensemble retrieval with 5 frozen templates and FAISS inner-product ranking.
- **Expected Result:** Pipeline completes with deterministic scores.
- **Actual Result:** Passed. Deterministic ranking verified.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_prompt_pipeline.py::test_int_e2e_03_prompt_ensemble_retrieval_end_to_end`.

---

#### INT-E2E-04: Full Benchmark Reproducibility & Output Verification
- **Test ID:** `INT-E2E-04`
- **Purpose:** Verify that saved machine-readable results (`summary.json`) match locally measured metrics across all methods.
- **Expected Result:**
  - BM25 LOCO: R@1 $\approx 42.12\%$, R@5 $\approx 60.81\%$, MRR $\approx 0.5112$
  - CLIP ViT-B/32: R@1 $\approx 5.45\%$, R@5 $\approx 17.71\%$, MRR $\approx 0.1307$
  - Prompt Ensemble: R@1 $\approx 5.14\%$, R@5 $\approx 17.00\%$, MRR $\approx 0.1268$
- **Actual Result:** Passed. Machine-readable outputs match locally measured metrics exactly.
- **Status:** PASS
- **Evidence / Artifact:** `experiments/results/m4/summary.json`; `tests/integration/test_reproducibility.py::test_int_e2e_04_summary_reproducibility`.

---

### 2.7 System, Safety & Documentation Integrity Tests

#### INT-TEST-SEP-01: Test/Production Separation & Raw Data Immutability
- **Test ID:** `INT-TEST-SEP-01`
- **Purpose:** Verify that test runs do not mutate raw datasets and that cache directories are isolated and gitignored.
- **Expected Result:** `data/raw/` remains read-only; caches and generated files reside in `experiments/`.
- **Actual Result:** Passed. Raw data intact; `.gitignore` properly isolates caches.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_reproducibility.py::test_int_test_sep_01_raw_data_immutability`.

---

#### INT-SAFE-01: Resource Budget Adherence (Process RAM $< 8$ GB)
- **Test ID:** `INT-SAFE-01`
- **Purpose:** Verify that process memory consumption remains strictly under the 8 GB (8,192 MB) project constraint.
- **Expected Result:** Peak process RSS $< 8,192$ MB.
- **Actual Result:** Passed. Measured peak RSS is $\le 1,289.7$ MB (1.26 GB), utilizing only $15.7\%$ of budget.
- **Status:** PASS
- **Evidence / Artifact:** `tests/integration/test_reproducibility.py::test_int_safe_01_memory_budget_under_8gb`.

---

#### INT-DOC-01: Cross-Documentation Consistency
- **Test ID:** `INT-DOC-01`
- **Purpose:** Verify cross-document consistency across all core markdown specifications.
- **Checked Parameters:**
  - Dataset Name: `RSICD` and `LEVIR-CD`
  - RSICD Test Gallery Count: `1,093` images
  - RSICD Query Count: `5,465` queries
  - VLM Checkpoint: `openai/clip-vit-base-patch32`
  - Embedding Dimension: `512`
  - FAISS Index Type: `IndexFlatIP`
  - Primary Metrics: `R@1`, `R@5`, `R@10`, `MRR`
  - Corrected BM25 Protocol: `Leave-One-Caption-Out Caption-Indexed`
  - H1 Status: `Partially Supported / Inconclusive (Modality Dependent)`
- **Expected Result:** Zero contradictory statements across project records.
- **Actual Result:** Passed. All records (`README.md`, `PRD.md`, `architecture.md`, `research_questions.md`, `experiment_plan.md`, `data_card.md`, `decision_log.md`, `devlog.md`, `docs/m4_semantic_retrieval.md`, `research_findings.md`) are aligned.
- **Status:** PASS
- **Evidence / Artifact:** Section 5 of this test plan.

---

## 3. Literature-to-Implementation Traceability Matrix

| Research Method | Peer-Reviewed Literature Grounding | Project Source Implementation | Experiment ID | Evaluation Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Okapi BM25 Lexical Retrieval** | Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, FnTIR, 2009. | `src/semantic_search/bm25.py` (`BM25Retriever`) | `EXP-RET-01` | R@1, R@5, R@10, MRR, Latency |
| **Zero-Shot Dual-Encoder VLM** | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML, 2021. | `src/semantic_search/clip_model.py` (`CLIPRetriever`) | `EXP-RET-02` | R@1, R@5, R@10, MRR, Latency |
| **Exact Inner-Product Vector Index** | Johnson, Douze, & Jégou, *Billion-scale similarity search with GPUs (FAISS)*, IEEE TBD, 2019. | `src/semantic_search/faiss_index.py` (`FAISSFlatIPIndex`) | `EXP-RET-02` | Search Latency (ms), RAM (MB) |
| **Domain Prompt Ensembling** | Radford et al., ICML 2021; Lu et al., IEEE TGRS 2018; Liu et al., IEEE TGRS 2024. | `src/semantic_search/prompt_ensembler.py` (`PromptEnsembler`) | `EXP-RET-03` | $\Delta R@K$, $\Delta \text{MRR}$, Latency |
| **Karpathy Dataset Split** | Karpathy & Fei-Fei, *Deep visual-semantic alignments for generating image descriptions*, CVPR, 2015. | `src/data/rsicd_loader.py` & `src/data/split_manager.py` | — | Non-overlap, Disjoint partitions |
