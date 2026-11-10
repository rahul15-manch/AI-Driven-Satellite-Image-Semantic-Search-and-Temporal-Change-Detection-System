# Phase 3A: Core Algorithmic Engines — Implementation & Verification Report

> **Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
> **Target Document:** `docs/phase3a_core_algorithmic_engines_report.md`  
> **Milestone / Phase:** Phase 3A — Core Algorithmic Engines (Registration & Region Extraction)  
> **Status:** Implemented, Tested, and Verified  
> **Execution Profile:** Strict CPU-first execution ($\le 8\text{ GB}$ RAM constraint)  
> **Audit & Regression Integrity:** All existing checkpoints, datasets, and benchmark results preserved.

---

## 1. Summary of Files Created and Modified

### Newly Created Core Engine Modules
1. **`src/temporal_understanding/__init__.py`**
   - Package initialization exposing primary abstractions, dataclasses, quality policies, and exception hierarchies.
2. **`src/temporal_understanding/registration.py`**
   - Implements `ImageCoRegistrator`, `RegistrationPolicy`, `RegistrationDiagnostics`, `RegistrationResult`, and `RegistrationStatus`.
   - Handles multi-format validation (PNG, JPEG, optical TIFF including multi-channel extraction), controlled scaling for oversized images ($\le 2048\text{ px}$), Stage A coarse ORB + RANSAC homography, Stage B fine ECC optimization, and multi-metric quality gates.
3. **`src/temporal_understanding/region_extractor.py`**
   - Implements `RegionExtractor`, `RegionExtractionConfig`, `ChangedRegion`, and `RegionExtractionResult`.
   - Handles morphological closing/opening with odd-sized kernels ($3 \times 3$), connected-component labeling ($cv2.connectedComponentsWithStats$), area filtering (absolute pixel threshold + relative canvas ratio), contextual bounding-box padding ($10\%$) with boundary clamping, and top-$K$ area ranking.

### Newly Created Test Suites
4. **`tests/test_temporal_registration.py`**
   - 12 comprehensive synthetic tests covering identical images, synthetic translation recovery, affine rotation recovery, insufficient matches, blank image rejection, corrupt/unsupported formats, differing dimensions, ECC divergence, quality-gate rejection, oversized image coordinate mapping, and array immutability.
5. **`tests/test_temporal_region_extractor.py`**
   - 12 comprehensive synthetic tests covering empty masks, full masks, single components, multiple components, small-noise filtering, relative area thresholds, morphological bridging/cleaning, padding/clipping, top-$K$ ranking, invalid dtypes/shapes, probability map aggregation, and raw mask preservation.

### Modified Existing Files
- **None.** In strict accordance with Phase 3A instructions, zero existing source files, test files, configuration files, checkpoints, datasets, or experimental results were modified.

---

## 2. Registration Architecture & Quality-Gate Explanation

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TWO-STAGE CO-REGISTRATION PIPELINE                              │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                 [Input Pair: Optical T1 and T2 (Path, Bytes, Array)]
                                         │
                                         ▼
            [Validation: Dimensions, Channels, Dtype, Variance, Formats]
                                         │
                                         ▼
                 [Create Grayscale Processing Copy (Max Dim <= 2048)]
                 • Scale Factors: s_x = W_proc / W_orig, s_y = H_proc / H_orig
                                         │
                                         ▼
                 [Stage A: Coarse ORB Feature Detection (Max 2,000)]
                 • BFMatcher (NORM_HAMMING) + Lowe's Ratio Test (tau = 0.75)
                 • RANSAC Homography Estimation (Reprojection Threshold = 3.5 px)
                 • Inlier Count, Inlier Ratio, Reprojection Error, Spatial Coverage
                                         │
                                         ▼
                 [Stage B: Fine ECC Alignment (Enhanced Correlation)]
                 • Constrained Affine Refinement (Max 50 Iterations, eps = 1e-4)
                 • Graceful Fallback if Gradient Singularity / Divergence
                                         │
                                         ▼
                 [Coordinate Frame Rescaling to Native Resolution]
                 • H_native = S_1^-1 * H_proc * S_2
                                         │
                                         ▼
                 [Multi-Metric Confidence Score Calculation]
                 • Confidence = (0.35 * InlierRatio + 0.35 * SpatialCoverage
                                 + 0.30 * max(0, 1 - ReprojErr / 5.0)) * min(1, Inliers / 40)
                                         │
                      ┌──────────────────┴──────────────────┐
                      ▼                                     ▼
           [Confidence >= 0.70 & Inliers >= 10]    [Confidence < 0.70]
                      │                                     │
                      ▼                                     ▼
          {Status: ALIGNED_HOMOGRAPHY}             [Affine Fallback Allowed?]
          Warp Original T2 onto T1                 • If Inliers >= 6 & Conf >= 0.35:
          using H_native via Perspective             {Status: ALIGNED_AFFINE}
                                                   • Else:
                                                     {Status: REJECTED}
                                                     (No Warp; Flag Alert)
```

### Explicit Quality Gates and Metrics
Unlike arbitrary confidence heuristics, the engine evaluates independently measurable physical diagnostics:
1. `match_count`: Count of candidate feature pairs surviving Lowe's ratio test. Minimum threshold: $15$.
2. `inlier_count`: Number of correspondences geometrically consistent with the homography/affine model under RANSAC. Minimum threshold: $10$.
3. `inlier_ratio`: Proportion of valid matches retained ($\text{inliers} / \text{matches}$). Minimum threshold: $0.40$.
4. `reprojection_error`: Mean Euclidean distance in pixels between transformed $T_2$ inliers and ground $T_1$ inliers. Maximum tolerance: $3.5\text{ px}$.
5. `spatial_coverage`: Convex hull area of inliers divided by total canvas area. Minimum threshold: $0.10$ ($10\%$ canvas spread) to prevent clustering in a single corner.
6. `ecc_correlation`: Normalized correlation coefficient from fine ECC optimization ($0.0$ to $1.0$).
7. `confidence_score`: Bounded composite in $[0.0, 1.0]$.
   - $\text{Confidence} \ge 0.70 \rightarrow$ High quality (`ALIGNED_HOMOGRAPHY`).
   - $0.35 \le \text{Confidence} < 0.70 \rightarrow$ Moderate quality (`ALIGNED_AFFINE`).
   - $\text{Confidence} < 0.35 \rightarrow$ Low quality (`REJECTED`). Unwarped fallback with explicit user warning.

---

## 3. Region Extraction & Clustering Algorithm Explanation

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        REGION EXTRACTION & SPATIAL CLUSTERING                          │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                       [Binary Change Mask (H, W), uint8 {0, 1}]
                                         │
                                         ▼
               [Input Validation: 2D, Non-Empty, Binary Value Range]
               • Original raw mask preserved unmodified for telemetry
                                         │
                                         ▼
               [Morphological Closing: 3x3 Rectangular Structuring Element]
               • Bridges small 1-pixel fracture lines within building roofs
                                         │
                                         ▼
               [Morphological Opening: 3x3 Rectangular Structuring Element]
               • Eliminates isolated single-pixel salt noise and speckle artifacts
                                         │
                                         ▼
               [Connected-Component Labeling with Statistics (cv2.CCL)]
               • 8-way connectivity extraction of [left, top, width, height, area]
                                         │
                                         ▼
               [Dual-Threshold Filtering Gate]
               • Absolute Threshold: pixel_area >= 150 px
               • Relative Threshold: pixel_area / (H * W) >= 0.0002 (0.02% of scene)
                                         │
                                         ▼
               [Bounding Box & Contextual Margin Geometry]
               • Strict bbox: [xmin, ymin, xmax, ymax]
               • Context bbox: add 10% margin on all 4 sides; clamp to [0, W] and [0, H]
               • Centroid calculation: (cx, cy)
               • Aggregate mean probability if continuous confidence map supplied
                                         │
                                         ▼
               [Area Ranking & Top-K Truncation]
               • Sort candidates by pixel_area descending
               • Select top-K (default K = 6)
               • Reassign 1-indexed region_id (Rank #1 = largest change site)
```

---

## 4. Unit-Test Coverage Summary

### Test Suite 1: `tests/test_temporal_registration.py` (12 Tests)
| Test Function | Target Behavior Verified | Result |
|---|---|---|
| `test_registration_identical_images` | Identity transform, confidence $= 1.0$, status `IDENTITY_ALIGNED` | **PASSED** |
| `test_registration_synthetic_translation` | Accurate recovery of known synthetic translation ($dx=6, dy=-4$) | **PASSED** |
| `test_registration_synthetic_affine` | Accurate recovery of synthetic rotation ($1.5^\circ$) | **PASSED** |
| `test_registration_insufficient_matches` | Smooth featureless gradients rejected safely | **PASSED** |
| `test_validation_blank_image_rejected` | All-zero or flat image raises `ImageValidationError` | **PASSED** |
| `test_validation_corrupt_or_unsupported` | Corrupt bytes, 1D, 4D, NaN arrays raise `ImageValidationError` | **PASSED** |
| `test_registration_differing_dimensions` | Warps differing-size $T_2$ ($450\times 450$) into $T_1$'s canvas ($384\times 384$) | **PASSED** |
| `test_ecc_failure_graceful_handling` | ECC divergence or max-iteration cutoff does not crash | **PASSED** |
| `test_unreliable_registration_rejection` | Completely uncorrelated scenes rejected with `REJECTED` status | **PASSED** |
| `test_rejection_raises_when_configured` | Policy with `raise_on_rejection=True` raises `RegistrationRejectedError` | **PASSED** |
| `test_oversized_image_scaling` | Downsamples processing copy and maps back to native dimensions ($512\times 512$) | **PASSED** |
| `test_preservation_of_original_arrays` | Input arrays remain immutable; no in-place mutation | **PASSED** |

### Test Suite 2: `tests/test_temporal_region_extractor.py` (12 Tests)
| Test Function | Target Behavior Verified | Result |
|---|---|---|
| `test_extract_empty_mask` | All-zero mask returns 0 regions cleanly | **PASSED** |
| `test_extract_full_mask` | All-ones mask returns 1 comprehensive region ($100\%$ area) | **PASSED** |
| `test_extract_single_component` | Exact bounding box, pixel area, and centroid extraction | **PASSED** |
| `test_extract_multiple_components` | Isolates multiple blocks and ranks them by area descending | **PASSED** |
| `test_small_noise_filtering` | Components $< 200\text{ px}$ discarded; large components retained | **PASSED** |
| `test_relative_area_threshold` | Components below relative area threshold ($0.001$) filtered | **PASSED** |
| `test_morphological_closing_and_opening` | Closing bridges 1-px gap; opening removes 1-px spur | **PASSED** |
| `test_padding_and_boundary_clipping` | Context bbox expands by $20\%$ and clamps strictly to image edges | **PASSED** |
| `test_top_k_selection` | When 5 candidates exist, top 3 largest are returned with IDs $1, 2, 3$ | **PASSED** |
| `test_invalid_mask_dimensions_and_dtypes` | 3D, 1D, non-numpy, or 0-dim inputs raise `RegionExtractionError` | **PASSED** |
| `test_probability_map_aggregation` | Accurately calculates mean probability within component mask ($0.85$) | **PASSED** |
| `test_preservation_of_raw_mask` | Raw mask is preserved alongside cleaned mask | **PASSED** |

---

## 5. Actual Test Execution Results

All tests were executed under Python 3.10.18 in the project virtual environment (`./venv/bin/pytest`).

### 1. Phase 3A New Test Execution
```bash
$ ./venv/bin/pytest tests/test_temporal_registration.py tests/test_temporal_region_extractor.py -v
============================= test session starts ==============================
platform darwin -- Python 3.10.18, pytest-8.3.4, pluggy-1.6.0
rootdir: /Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System
configfile: pytest.ini
plugins: cov-6.0.0, anyio-4.15.1
collected 24 items

tests/test_temporal_registration.py::test_registration_identical_images PASSED [  4%]
tests/test_temporal_registration.py::test_registration_synthetic_translation PASSED [  8%]
tests/test_temporal_registration.py::test_registration_synthetic_affine PASSED [ 12%]
tests/test_temporal_registration.py::test_registration_insufficient_matches PASSED [ 16%]
tests/test_temporal_registration.py::test_validation_blank_image_rejected PASSED [ 20%]
tests/test_temporal_registration.py::test_validation_corrupt_or_unsupported PASSED [ 25%]
tests/test_temporal_registration.py::test_registration_differing_dimensions PASSED [ 29%]
tests/test_temporal_registration.py::test_ecc_failure_graceful_handling PASSED [ 33%]
tests/test_temporal_registration.py::test_unreliable_registration_rejection PASSED [ 37%]
tests/test_temporal_registration.py::test_rejection_raises_when_configured PASSED [ 41%]
tests/test_temporal_registration.py::test_oversized_image_scaling PASSED [ 45%]
tests/test_temporal_registration.py::test_preservation_of_original_arrays PASSED [ 50%]
tests/test_temporal_region_extractor.py::test_extract_empty_mask PASSED  [ 54%]
tests/test_temporal_region_extractor.py::test_extract_full_mask PASSED   [ 58%]
tests/test_temporal_region_extractor.py::test_extract_single_component PASSED [ 62%]
tests/test_temporal_region_extractor.py::test_extract_multiple_components PASSED [ 66%]
tests/test_temporal_region_extractor.py::test_small_noise_filtering PASSED [ 70%]
tests/test_temporal_region_extractor.py::test_relative_area_threshold PASSED [ 75%]
tests/test_temporal_region_extractor.py::test_morphological_closing_and_opening PASSED [ 79%]
tests/test_temporal_region_extractor.py::test_padding_and_boundary_clipping PASSED [ 83%]
tests/test_temporal_region_extractor.py::test_top_k_selection PASSED     [ 87%]
tests/test_temporal_region_extractor.py::test_invalid_mask_dimensions_and_dtypes PASSED [ 91%]
tests/test_temporal_region_extractor.py::test_probability_map_aggregation PASSED [ 95%]
tests/test_temporal_region_extractor.py::test_preservation_of_raw_mask PASSED [100%]

============================== 24 passed in 0.48s ==============================
```

### 2. Full Regression Suite Execution (Existing Milestones)
```bash
$ ./venv/bin/pytest tests/test_b1_pixel_diff.py tests/test_b2_ssim.py tests/test_b3_cva.py tests/test_m7_model.py tests/test_m8_model.py tests/test_m9_model.py tests/test_m10_model.py tests/test_m12_api.py tests/test_llm_explainability.py
============================== 76 passed in 11.37s =============================
```

**Grand Total: 100 passing tests across the entire repository. Zero regressions.**

---

## 6. Known Limitations of Phase 3A Implementation

1. **Extreme Perspective / Tilt Disparity:**
   Planar homography and affine models assume the ground surface is approximately planar or viewed from near-nadir angles. High-oblique satellite imagery with severe relief displacement from tall buildings or mountainous terrain cannot be fully corrected by 2D homography alone without a Digital Elevation Model (DEM).
2. **Extreme Seasonal Radiometric Shifts:**
   If $T_1$ is taken in deep winter with complete snow cover and $T_2$ is taken in summer with dense green foliage, ORB feature match density may drop below $15$. In such cases, the quality gate correctly rejects registration rather than warping the image erroneously.
3. **Physical Area Reporting Requires Verified GSD:**
   In compliance with academic integrity guidelines, `ChangedRegion.physical_area_sq_meters` is calculated *only* if the user or metadata explicitly supplies a valid Ground Sample Distance (GSD). For ordinary uploaded RGB imagery without georeferencing, the engine strictly outputs pixel area and area percentage.

---

## 7. Confirmation of Checkpoint & Artifact Integrity

- **Checkpoints Unchanged:**
  - `experiments/checkpoints/m7/fc_siam_diff_best.pt` (Unmodified)
  - `experiments/checkpoints/m8/snunet_best.pt` (Unmodified)
  - `experiments/checkpoints/m9/bit_cd_best.pt` (Unmodified)
  - `experiments/checkpoints/m10/qat_cd_metadata.json` (Unmodified)
- **Datasets & Splits Unchanged:**
  - `data/raw/levir_cd/` (Unmodified)
  - `data/splits/levir_cd/` (Unmodified)
  - `data/metadata/levir_cd/` (Unmodified)
- **Benchmark Results & Figures Unchanged:**
  - `experiments/results/` (Unmodified)
  - `experiments/figures/` (Unmodified)

---

## 8. Suggested Next Step for Phase 3B

With Phase 3A core engines complete and verified, the suggested next step is **Phase 3B: Semantic Change Analyzer & Dual-Stage AI Reasoning**:
1. Implement optical vegetation index shifts (Visible Atmospherically Resistant Index - VARI) in `src/temporal_understanding/semantic_analyzer.py`.
2. Implement Stage A Vision-Language Model caller (Groq Llama 3.2 Vision) to inspect the top-$K$ cropped regions.
3. Update `PatchExtractor` in `src/data/patch_extractor.py` to support dynamic canvas reconstruction for arbitrary image dimensions.
4. Implement Stage B Qwen 27B explanation synthesis in `src/temporal_understanding/explanation_engine.py`, binding detector evidence with semantic findings.
