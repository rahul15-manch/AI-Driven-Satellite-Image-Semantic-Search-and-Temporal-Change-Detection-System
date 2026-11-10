# Temporal Satellite Image Understanding: Existing System Audit & Capability Assessment

> **Milestone / Phase:** Phase 1 — Comprehensive Pre-Implementation Repository Audit  
> **Target Document:** `docs/temporal_satellite_understanding_existing_system_audit.md`  
> **Status:** Completed Read-Only System Audit  
> **Audit Scope:** Complete repository inspection (FastAPI backend, Streamlit dashboard, static web interface, 7 change detection models, semantic search engine, preprocessing & patch pipeline, Groq LLM integration, and test suite).

---

## Executive Summary

This audit evaluates the codebase of the **AI-Driven Satellite Image Semantic Search and Temporal Change Detection System** to establish a baseline before designing or implementing the **Temporal Satellite Image Understanding** feature. 

The planned feature requires users to upload two arbitrary satellite images (Image $T_1$: earlier date, Image $T_2$: later date), detect spatial differences, categorize the semantic nature of those changes (e.g., new building construction, vegetation loss/gain, infrastructure expansion), and generate a detailed, evidence-grounded natural-language explanation using a vision-capable AI model.

### Key Audit Findings
1. **Change-Detection Models Are Production-Ready for Binary Differences:** The repository possesses 7 fully verified change-detection models (3 classical baselines: $B_1$ Pixel Diff, $B_2$ SSIM, $B_3$ CVA; and 4 learned models: $M_7$ FC-Siam-diff, $M_8$ SNUNet-CD, $M_9$ BIT-CD, $M_{10}$ QAT-CD). All model checkpoints and calibration thresholds are intact and operate under strict CPU-first constraints.
2. **Current Models Are Binary and Building-Centric:** All learned detectors were trained on LEVIR-CD, which defines change strictly as building construction/demolition. **No semantic change-type classification exists** (e.g., separating vegetation loss, water emergence, or road development).
3. **Image Registration Is Completely Absent:** The repository contains zero spatial alignment, feature matching, homography estimation, or warping routines. The entire system strictly assumes that $T_1$ and $T_2$ are already perfectly co-registered orthorectified image pairs of identical pixel dimensions.
4. **Arbitrary Image Upload Is Only Partially Implemented in Backend and Completely Missing in UI:** The FastAPI `/api/change-detection` endpoint accepts base64-encoded strings for `image_t1_base64` and `image_t2_base64`. However, `PatchExtractor.reconstruct_image` hardcodes output dimensions to $1024 \times 1024$, which breaks for arbitrary image resolutions. Neither the Streamlit dashboard nor the FastAPI web interface includes an upload widget or drag-and-drop file interface.
5. **Existing Groq Qwen Integration Is Text-Only:** The configured LLM explanation service (`src/api/llm_service.py`) utilizes `qwen/qwen3.8-27b` via Groq. **This model is strictly text-based and cannot ingest images.** It interprets only structured numerical and categorical telemetry. Generating natural-language descriptions of arbitrary optical image changes will require either integrating a vision-language model (VLM) or extracting rich structured visual/semantic change descriptors to supply to the LLM.

---

## 1. Project Architecture Audit

The repository is organized into a modular structure separating machine learning cores, data management, API serving, and dual frontend interfaces.

```
AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/
├── app.py                         # Streamlit multi-page application entrypoint
├── pages/                         # Streamlit page modules (Overview, Dataset, Retrieval, Change Detection, False-Alarm)
├── ui/                            # Streamlit UI reusable components, charts, and image viewers
├── frontend/                      # Static web frontend (HTML/CSS/JS) served via FastAPI
│   ├── index.html                 # Single-page interface with 4 navigation panels
│   ├── css/styles.css             # Vanilla CSS styling with design system variables
│   └── js/                        # Modular ES JavaScript (api.js, app.js, change_detection.js, search.js, results.js)
├── src/
│   ├── api/                       # FastAPI application, routing, schemas, and service layer
│   │   ├── app.py                 # FastAPI application factory and middleware configuration
│   │   ├── routes.py              # REST API endpoint definitions
│   │   ├── schemas.py             # Pydantic v2 data models and request/response contracts
│   │   ├── service.py             # Service layer orchestrating search, change detection, and benchmarks
│   │   └── llm_service.py         # Dedicated Groq LLM explanation client (Qwen 27B)
│   ├── change_detection/          # Bi-temporal change detection algorithms and training pipelines
│   │   ├── base.py                # Abstract BaseChangeDetector interface
│   │   ├── pixel_diff.py          # B1 Absolute Pixel Differencing baseline
│   │   ├── ssim_detector.py       # B2 Structural Similarity Dissimilarity baseline
│   │   ├── cva_detector.py        # B3 Change Vector Analysis baseline
│   │   ├── fc_siam_diff.py        # M7 Fully Convolutional Siamese Difference Network
│   │   ├── snunet.py              # M8 Densely Connected Siamese Network + ECAM
│   │   ├── bit_cd.py              # M9 Bitemporal Image Transformer
│   │   ├── qat_cd.py              # M10 Quality-Aware Gated Change Detector
│   │   ├── thresholding.py        # Validation threshold sweep optimization
│   │   ├── evaluator.py           # ConfusionMatrix and ChangeDetectionEvaluator metrics
│   │   └── patch_dataset.py       # PyTorch Dataset for synchronized 256x256 patch training
│   ├── semantic_search/           # Cross-modal retrieval engine (RSICD dataset)
│   │   ├── clip_model.py          # OpenAI CLIP ViT-B/32 CPU wrapper
│   │   ├── prompt_ensembler.py    # Remote-sensing domain prompt ensembling (5 templates)
│   │   ├── bm25.py                # BM25 lexical search baseline
│   │   └── faiss_index.py         # FAISS FlatIP normalized inner product vector index
│   ├── data/                      # Dataset loaders, verifiers, and patch extraction
│   │   ├── patch_extractor.py     # Deterministic patch tiling and canvas reconstruction
│   │   ├── levir_loader.py        # LEVIR-CD dataset loading and verification
│   │   ├── rsicd_loader.py        # RSICD dataset loading
│   │   ├── split_manager.py       # Train/Val/Test split manager with leakage prevention
│   │   └── dataset_verifier.py    # Hash and integrity verification
│   └── false_alarm/               # Environmental robustness perturbations (M6 benchmark)
│       ├── base.py                # Base perturbation abstraction
│       ├── illumination.py        # Gamma & linear brightness shifts
│       ├── blur.py                # Gaussian defocus blur
│       └── misregistration.py     # Sub-pixel translation shifts (1-2 px)
├── experiments/
│   ├── checkpoints/               # Trained model weights (m7, m8, m9, m10)
│   ├── figures/                   # Publication-grade figures (m5, m6, m10, m11, m13)
│   └── results/                   # Benchmark metric JSON files and statistical artifacts
└── tests/                         # Pytest test suite (49+ unit & integration test modules)
```

### Component Responsibilities & File Paths

| Architectural Component | File Path | Primary Responsibility |
|---|---|---|
| **FastAPI Backend Application** | `src/api/app.py` | Configures FastAPI app, CORS middleware, global error handling, router inclusion, and static file mount for `frontend/`. |
| **API Endpoints** | `src/api/routes.py` | Exposes REST endpoints for `/api/search`, `/api/change-detection`, `/api/pairs`, `/api/benchmark`, `/api/image/...`, `/api/figures/...`, `/api/explain/...`, `/api/llm/status`. |
| **Data Contracts** | `src/api/schemas.py` | Defines Pydantic validation schemas for all requests and responses. |
| **Backend Service Layer** | `src/api/service.py` | Encapsulates singleton services: `RetrievalService`, `ChangeDetectionService`, and `BenchmarkService`. |
| **LLM Explanation Service** | `src/api/llm_service.py` | Interacts with Groq `/chat/completions` using Qwen 27B to generate structured text explanations. |
| **Streamlit Entrypoint** | `app.py` | Multi-page Streamlit application config, sidebar telemetry, and router navigation. |
| **Streamlit Change Detection** | `pages/change_detection.py` | Interactive dashboard for LEVIR-CD test pair selection, live CPU inference, error overlay display, and M11 benchmark evaluation. |
| **Streamlit Visualizer** | `ui/image_viewer.py` | 4-column visualizer ($T_1$, $T_2$, Ground Truth, Prediction) and error confusion overlays. |
| **Static Web Frontend** | `frontend/index.html` | Dark-mode interface with tabs for Semantic Search, Change Detection, Benchmark Results, and Methodology. |
| **Static Web CD Controller** | `frontend/js/change_detection.js` | Manages change detection form submission, pair selection, image rendering, and calls LLM explanation endpoint. |
| **Patch Engine** | `src/data/patch_extractor.py` | Cuts $1024 \times 1024$ images into $256 \times 256$ patches and stitches prediction patch grids back together. |
| **Model Weights & Metadata** | `experiments/checkpoints/` | Stores serialized PyTorch models (`.pt`) and validation metadata (`.json`) for M7, M8, M9, and M10. |

---

## 2. Existing Change-Detection Models Inventory

The repository contains 7 change-detection models spanning classical baselines, Siamese CNNs, attention networks, vision transformers, and quality-gated pipelines.

```
                                  ┌─────────────────────────────┐
                                  │   Change Detection Models   │
                                  └──────────────┬──────────────┘
                 ┌───────────────────────────────┴───────────────────────────────┐
                 ▼                                                               ▼
   ┌───────────────────────────┐                                   ┌───────────────────────────┐
   │    Classical Baselines    │                                   │     Learned / Neural      │
   ├───────────────────────────┤                                   ├───────────────────────────┤
   │ B1: Pixel Difference      │                                   │ M7: FC-Siam-diff (CNN)    │
   │ B2: SSIM Dissimilarity    │                                   │ M8: SNUNet-CD (Dense)     │
   │ B3: Change Vector (CVA)   │                                   │ M9: BIT-CD (Transformer)  │
   └───────────────────────────┘                                   │ M10: QAT-CD (Quality Gate)│
                                                                   └───────────────────────────┘
```

### Detailed Model Audits

#### 1. Method B1: Absolute Pixel Differencing (`B1_Pixel_Diff`)
1. **Architecture & Family:** Classical radiometric baseline. Pixel-wise $L_1$ color difference: $D(x, y) = \frac{1}{3} \sum_{c \in \{R,G,B\}} |I_2(x,y,c) - I_1(x,y,c)|$.
2. **Input Format:** Two numpy arrays `(H, W, 3)` or `(H, W)` in `uint8` $[0, 255]$ or `float32` $[0.0, 1.0]$.
3. **Accepts Two Images Directly:** Yes (`compute_difference_map(img1, img2)`).
4. **Required Dimensions:** Any matching dimensions $(H, W)$. Does not require $256 \times 256$ or $1024 \times 1024$.
5. **Preprocessing & Normalization:** `BaseChangeDetector._ensure_float_rgb` normalizes values to $[0.0, 1.0]$.
6. **Checkpoint Location:** None (parameter-free algorithm).
7. **Inference Function & Path:** `PixelDiffDetector.compute_difference_map` in `src/change_detection/pixel_diff.py:L32`.
8. **Output Format:** 2D numpy array `(H, W)`, `float32` in range $[0.0, 1.0]$.
9. **Output Nature:** Continuous difference map. Method `predict()` applies threshold $\tau^* = 0.4100$ to yield binary mask $\{0, 1\}$.
10. **Arbitrary Upload Suitability:** Yes. Can execute directly on any matching optical image pair without patch extraction or training.

#### 2. Method B2: Structural Similarity Dissimilarity (`B2_SSIM`)
1. **Architecture & Family:** Classical structural baseline using localized Gaussian windowing (window size $11 \times 11$, $\sigma = 1.5$). Dissimilarity formula: $D_{\text{SSIM}} = \text{clip}(1.0 - \text{SSIM}, 0.0, 1.0)$.
2. **Input Format:** Two numpy arrays `(H, W, 3)` or `(H, W)`, normalized or converted via ITU-R 601-2 luma.
3. **Accepts Two Images Directly:** Yes (`compute_difference_map(img1, img2)`).
4. **Required Dimensions:** Any matching dimensions with $\min(H, W) \ge 11$ pixels.
5. **Preprocessing & Normalization:** Normalized to $[0.0, 1.0]$, converted to ITU-R 601-2 grayscale luminance.
6. **Checkpoint Location:** None (parameter-free algorithm).
7. **Inference Function & Path:** `SSIMDetector.compute_difference_map` in `src/change_detection/ssim_detector.py:L57`.
8. **Output Format:** 2D numpy array `(H, W)`, `float32` in range $[0.0, 1.0]$.
9. **Output Nature:** Continuous dissimilarity map. Thresholded at $\tau^* = 0.9000$ to yield binary mask.
10. **Arbitrary Upload Suitability:** Yes. Operates on arbitrary dimensions $\ge 11 \times 11$.

#### 3. Method B3: Change Vector Analysis (`B3_CVA`)
1. **Architecture & Family:** Classical spectral Euclidean distance in color space: $D_{\text{CVA}}(x,y) = \frac{\|I_2(x,y) - I_1(x,y)\|_2}{\sqrt{3}}$.
2. **Input Format:** Two numpy arrays `(H, W, 3)`, `float32` in $[0.0, 1.0]$.
3. **Accepts Two Images Directly:** Yes (`compute_difference_map(img1, img2)`).
4. **Required Dimensions:** Any matching dimensions $(H, W)$.
5. **Preprocessing & Normalization:** `_ensure_float_rgb` normalizes to $[0.0, 1.0]$.
6. **Checkpoint Location:** None (parameter-free algorithm).
7. **Inference Function & Path:** `CVADetector.compute_difference_map` in `src/change_detection/cva_detector.py:L31`.
8. **Output Format:** 2D numpy array `(H, W)`, `float32` in range $[0.0, 1.0]$.
9. **Output Nature:** Continuous spectral distance map. Thresholded at $\tau^* = 0.4050$ to produce binary mask.
10. **Arbitrary Upload Suitability:** Yes. Fully compatible with arbitrary image pairs.

#### 4. Method M7: Fully Convolutional Siamese Difference Network (`M7_FC_Siam_diff`)
1. **Architecture & Family:** Daudt et al. (ICIP 2018). Twin weight-sharing 4-stage convolutional encoder, multi-scale skip-level absolute differencing $D^{(l)} = |F_1^{(l)} - F_2^{(l)}|$, and convolutional upsampling decoder. Parameter count: 1,471,409.
2. **Input Format:** PyTorch tensors `(B, 3, H, W)` or numpy arrays `(H, W, 3)` in $[0.0, 1.0]$.
3. **Accepts Two Images Directly:** Yes (`forward(t1, t2)` in PyTorch, `compute_difference_map(img1, img2)` in detector wrapper).
4. **Required Dimensions:** Because of 4 pooling layers (stride 2 each), $(H, W)$ must be divisible by 16. Trained on $256 \times 256$ patches.
5. **Preprocessing & Normalization:** Scale to $[0.0, 1.0]$, convert to FloatTensor, permute to `(1, 3, H, W)`.
6. **Checkpoint Location:** `experiments/checkpoints/m7/fc_siam_diff_best.pt` (state dict and metadata: frozen threshold $\tau^* = 0.5500$, validation F1: $0.5401$).
7. **Inference Function & Path:** `FCSiamDiffDetector.compute_difference_map` in `src/change_detection/fc_siam_diff.py:L178`.
8. **Output Format:** 2D numpy array `(H, W)`, `float32` in range $[0.0, 1.0]$ representing change probability (via `torch.sigmoid`).
9. **Output Nature:** Continuous change probability map. Thresholded at $\tau^* = 0.5500$ for binary mask.
10. **Arbitrary Upload Suitability:** Reusable for arbitrary uploads provided $(H, W)$ are multiples of 16 (or images are tiled via `PatchExtractor`). Trained specifically on building changes.

#### 5. Method M8: Densely Connected Siamese Network (`M8_SNUNet`)
1. **Architecture & Family:** Fang et al. (IEEE GRSL 2021). Siamese encoder with dense nested skip connections (UNet++ style) and Ensemble Channel Attention Module (ECAM). Parameter count: 563,377.
2. **Input Format:** PyTorch tensors `(B, 3, H, W)` or numpy arrays `(H, W, 3)` in $[0.0, 1.0]$.
3. **Accepts Two Images Directly:** Yes (`forward(t1, t2)` and `compute_difference_map(img1, img2)`).
4. **Required Dimensions:** $(H, W)$ must be divisible by 16. Trained on $256 \times 256$ patches.
5. **Preprocessing & Normalization:** Scale to $[0.0, 1.0]$, FloatTensor conversion, channel permutation `(1, 3, H, W)`.
6. **Checkpoint Location:** `experiments/checkpoints/m8/snunet_best.pt` (state dict and metadata: frozen threshold $\tau^* = 0.5000$, validation F1: $0.5583$).
7. **Inference Function & Path:** `SNUNetDetector.compute_difference_map` in `src/change_detection/snunet.py:L221`.
8. **Output Format:** 2D numpy array `(H, W)`, `float32` in range $[0.0, 1.0]$ (via `torch.sigmoid`).
9. **Output Nature:** Continuous change probability map. Thresholded at $\tau^* = 0.5000$.
10. **Arbitrary Upload Suitability:** Reusable. *Caveat:* Documented in Milestone M11 to suffer from severe false alarms (39.85% false positive rate on out-of-domain evaluation).

#### 6. Method M9: Bitemporal Image Transformer (`M9_BIT_CD`)
1. **Architecture & Family:** Chen et al. (IEEE TGRS 2021). Siamese CNN backbone (3 stages), Spatial Attention Tokenizer ($L=16$ semantic tokens), Transformer Contextualizer (intra-temporal Self-Attention + inter-temporal Cross-Attention), Spatial Token Projector, and upsampling prediction head. Parameter count: 552,097.
2. **Input Format:** PyTorch tensors `(B, 3, H, W)` or numpy arrays `(H, W, 3)`.
3. **Accepts Two Images Directly:** Yes (`forward(t1, t2)` and `compute_difference_map(img1, img2)`).
4. **Required Dimensions:** $(H, W)$ must be divisible by 4 (due to two stride-2 downsampling layers). Trained on $256 \times 256$ patches.
5. **Preprocessing & Normalization:** Scale to $[0.0, 1.0]$, FloatTensor conversion, `(1, 3, H, W)`.
6. **Checkpoint Location:** `experiments/checkpoints/m9/bit_cd_best.pt` (state dict and metadata: frozen threshold $\tau^* = 0.4500$, validation F1: $0.5578$, test F1: $0.5927$).
7. **Inference Function & Path:** `BITChangeDetector.compute_difference_map` in `src/change_detection/bit_cd.py:L290`.
8. **Output Format:** 2D numpy array `(H, W)`, `float32` in range $[0.0, 1.0]$.
9. **Output Nature:** Continuous change probability map. Thresholded at $\tau^* = 0.4500$.
10. **Arbitrary Upload Suitability:** Highly recommended for reuse. State-of-the-art detector in the repository. Runs on arbitrary pairs with $(H, W)$ divisible by 4.

#### 7. Method M10: Quality-Aware Temporal Change Detection (`M10_QAT_CD`)
1. **Architecture & Family:** Quality-aware gated framework. Wraps M9 BIT-CD with deterministic quality discrepancy estimators:
   $$P_{\text{QAT}}(x, y) = \text{clip}\left(P_0(x, y) \cdot \left[1 - \gamma \cdot U_{\text{det}}(x, y) \cdot Q_{\text{defect}}(x, y)\right], 0.0, 1.0\right)$$
   Where $Q_{\text{defect}} = 0.34 S_{\text{reg}} + 0.33 S_{\text{illum}} + 0.33 S_{\text{sharp}}$ and $U_{\text{det}} = 4 P_0 (1 - P_0)$.
2. **Input Format:** Two numpy arrays `(H, W, 3)`.
3. **Accepts Two Images Directly:** Yes (`compute_difference_map(img1, img2)` and `compute_quality_diagnostics(img1, img2)`).
4. **Required Dimensions:** Determined by base detector (divisible by 4).
5. **Preprocessing & Normalization:** Base detector normalization + SciPy Sobel/uniform/Laplacian filtering.
6. **Checkpoint Location:** `experiments/checkpoints/m10/qat_cd_metadata.json` (links to `m9/bit_cd_best.pt`, calibrated $\gamma = 0.0$, calibrated threshold $\tau^* = 0.4500$).
7. **Inference Function & Path:** `QualityAwareChangeDetector.compute_difference_map` in `src/change_detection/qat_cd.py:L329`.
8. **Output Format:** 2D numpy array `(H, W)`, `float32` in range $[0.0, 1.0]$.
9. **Output Nature:** Continuous quality-attenuated change probability map and diagnostic discrepancy maps ($S_{\text{reg}}, S_{\text{illum}}, S_{\text{sharp}}, U_{\text{det}}, Q_{\text{defect}}$).
10. **Arbitrary Upload Suitability:** Highly reusable. Provides built-in telemetry regarding illumination disparity and edge misregistration.

### Model Inventory Summary Matrix

| Model Identifier | Model Class | Parameters | Best Val Threshold $\tau^*$ | Test F1 Score | Test IoU | Weight Checkpoint Path | Native Arbitrary Upload Support |
|---|---|---|---|---|---|---|---|
| **B1_Pixel_Diff** | `PixelDiffDetector` | 0 | 0.4100 | 0.0617 | 0.0318 | *None (Param-free)* | **Full** (any matching $H, W$) |
| **B2_SSIM** | `SSIMDetector` | 0 | 0.9000 | 0.1587 | 0.0861 | *None (Param-free)* | **Full** ($H, W \ge 11$) |
| **B3_CVA** | `CVADetector` | 0 | 0.4050 | 0.0616 | 0.0318 | *None (Param-free)* | **Full** (any matching $H, W$) |
| **M7_FC_Siam_diff** | `FCSiamDiffDetector` | 1,471,409 | 0.5500 | 0.4852 | 0.3203 | `checkpoints/m7/fc_siam_diff_best.pt` | **Via Patching** ($H, W \pmod{16}=0$) |
| **M8_SNUNet** | `SNUNetDetector` | 563,377 | 0.5000 | 0.2018 | 0.1122 | `checkpoints/m8/snunet_best.pt` | **Via Patching** ($H, W \pmod{16}=0$) |
| **M9_BIT_CD** | `BITChangeDetector` | 552,097 | 0.4500 | 0.5927 | 0.4211 | `checkpoints/m9/bit_cd_best.pt` | **Via Patching** ($H, W \pmod{4}=0$) |
| **M10_QAT_CD** | `QualityAwareChangeDetector`| 552,097 | 0.4500 | 0.5927 | 0.4211 | `checkpoints/m10/qat_cd_metadata.json` | **Via Patching** ($H, W \pmod{4}=0$) |

---

## 3. Existing Image Upload Pipeline Audit

### Image Ingestion Audit Findings

1. **Does the application already support image uploads?**
   - **Backend:** Partially. In `src/api/routes.py:L82-98`, `POST /api/change-detection` accepts base64-encoded strings: `image_t1_base64` and `image_t2_base64`.
   - **Streamlit Frontend:** **No.** `pages/change_detection.py` only provides a dropdown selection of verified LEVIR-CD test pairs (`get_available_levir_test_pairs()`). There is no `st.file_uploader` widget.
   - **Static Web Frontend:** **No.** `frontend/index.html` only provides a `<select id="change-pair-select">` loaded via `/api/pairs`. No file inputs or drop zones exist.
2. **Are uploads supported for one image or image pairs?**
   - The backend schema allows uploading two images as a pair. Uploading a single image is not supported; if only one base64 string is provided, the backend falls back to `pair_id`.
3. **Are there file-type and file-size restrictions?**
   - **File types:** No explicit MIME or extension filtering. PIL simply attempts `Image.open(io.BytesIO(data)).convert("RGB")`.
   - **File sizes:** No maximum payload size constraint is enforced in FastAPI middleware or routes. Unbounded base64 strings could lead to out-of-memory errors on large TIFF images.
4. **Are images validated before inference?**
   - The only validation is a strict equality check on dimensions:
     ```python
     if img_a.shape != img_b.shape:
         raise HTTPException(status_code=400, detail=f"Image shape mismatch: T1 is {img_a.shape}, T2 is {img_b.shape}")
     ```
   - No checks exist for minimum patch dimensions ($\ge 256 \times 256$), non-zero image area, channel validity before conversion, or corrupted byte payloads.
5. **Are images resized or cropped into patches?**
   - In `ChangeDetectionService.detect_change` (`src/api/service.py:L356`):
     ```python
     patches = self.extractor.extract_from_pair(img_a, img_b, parent_stem=pair_id or "uploaded")
     ```
   - `PatchExtractor(patch_size=256, stride=256, padding_mode="drop")` crops images into non-overlapping $256 \times 256$ tiles. Any marginal pixels not fitting an exact multiple of 256 are discarded (`padding_mode="drop"`).
   - Images are **not** automatically resized.
6. **Is image alignment or registration performed?**
   - **No.** Absolutely no spatial alignment or registration is performed during upload or before inference.
7. **Does the system handle different image resolutions?**
   - **No.** If $T_1$ and $T_2$ differ in width, height, or aspect ratio, the endpoint immediately returns HTTP 400.
   - If both $T_1$ and $T_2$ have matching dimensions other than $1024 \times 1024$, `src/api/service.py:L366` contains a critical bug:
     ```python
     base_diff = self.extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
     ```
     The canvas size $(1024, 1024)$ is hardcoded instead of using $(img\_a.shape[0], img\_a.shape[1])$, which corrupts or crashes the output for non-$1024 \times 1024$ image pairs.
8. **Are formats such as JPG, JPEG, PNG, and TIFF supported?**
   - Base64 payload decoding passes through `Image.open()`. Standard JPEG and PNG work.
   - Uncompressed TIFF will decode if PIL supports the format, but multi-band satellite GeoTIFFs (16-bit or multispectral > 3 channels) will either lose dynamic range or fail during `.convert("RGB")`.
9. **Are uploaded images stored temporarily or permanently?**
   - Images are held entirely in memory as Python `bytes` and `numpy.ndarray` objects during the lifecycle of the HTTP request. They are **never saved to disk** (neither temporarily in `/tmp` nor permanently in `data/`).
10. **Is uploaded image data cleaned up after processing?**
    - Yes, memory is reclaimed by Python garbage collection once the request scope ends. No disk cleanup is required because no files are written.

---

## 4. Image Registration and Preprocessing Audit

Bi-temporal satellite image understanding requires accurate spatial alignment, as sub-pixel misregistration between $T_1$ and $T_2$ creates false change boundaries.

### Preprocessing & Alignment Inspection

```
                              ┌──────────────────────────────────┐
                              │  Spatial Alignment Verification  │
                              └─────────────────┬────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
   ┌───────────────────────────┐                                 ┌───────────────────────────┐
   │ Current Pipeline Reality  │                                 │   Required for Feature    │
   ├───────────────────────────┤                                 ├───────────────────────────┤
   │ Spatial Alignment: NONE   │                                 │ Feature-based Registration│
   │ Homography: NONE          │                                 │ Resampling to Equal GSD   │
   │ Image Warping: NONE       │                                 │ Resizing / Canvas Padding │
   │ Shape Equality Check Only │                                 │ Geo-Referencing Handling  │
   └───────────────────────────┘                                 └───────────────────────────┘
```

1. **Spatial Alignment & Registration:**
   - **Verification:** An exhaustive search across the entire repository revealed **zero image registration or alignment algorithms**. There are no calls to OpenCV feature detectors (`cv2.SIFT`, `cv2.ORB`), no homography estimation (`findHomography`), and no image warping (`warpPerspective`, `warpAffine`).
   - **Assumption:** The entire existing system implicitly assumes that $T_1$ and $T_2$ are already orthorectified and co-registered on a shared coordinate grid.
2. **Handling Different Image Dimensions:**
   - The current code strictly asserts $T_1.shape == T_2.shape$. If two arbitrary uploaded images have different bounding boxes or resolutions, the system rejects them.
3. **Handling Different Ground Sample Distances (Resolutions):**
   - The system possesses no mechanism to resample images of differing pixel scale (e.g., $0.5\,\text{m/pixel}$ vs $1.0\,\text{m/pixel}$) to a common spatial scale.
4. **Cropping & Patch Extraction:**
   - `PatchExtractor` (`src/data/patch_extractor.py`) supports non-overlapping ($stride = 256$) and overlapping ($stride < 256$) tiling. It provides `padding_mode="drop"` (default in API) and `padding_mode="pad"`. If `padding_mode="drop"` is active, non-multiple-of-256 boundaries are lost.
5. **Channel Validation & Normalization:**
   - All input arrays are forced to 3-channel RGB via PIL `.convert("RGB")`. Pixel intensities are normalized from $[0, 255]$ to $[0.0, 1.0]$.
6. **Illumination Differences:**
   - Classical detectors ($B_1$, $B_3$) lack illumination robustness.
   - $M_{10}$ QAT-CD computes a local illumination discrepancy map $S_{\text{illum}}$ via uniform box filtering and attenuates uncertainty near decision boundaries, but does not perform radiometric normalization (such as histogram matching or wall-to-wall color balancing).
7. **Misregistration Sensitivity (Evidence from Milestone M6):**
   - Milestone M6 benchmarks (`src/false_alarm/misregistration.py`) proved that a synthetic translation of just **1 to 2 pixels** degrades model F1 scores by **15% to 42%** across classical detectors.
   - While QAT-CD detects gradient discrepancy ($S_{\text{reg}}$) to flag misregistration, it **cannot fix or align the imagery**.

> [!IMPORTANT]
> **Definitive Limitation:** The repository does not possess any image alignment capability. Two images uploaded from different sources, dates, or view angles will produce extensive false positive artifacts unless an automated registration module is introduced.

---

## 5. Existing LLM Integration Audit

The repository contains an LLM explanation service implemented in `src/api/llm_service.py` connected to Groq.

### LLM Implementation Details

- **Class Name:** `LLMExplanationService` (`src/api/llm_service.py:L31`).
- **Hosting Provider:** Groq Cloud API (`https://api.groq.com/openai/v1`).
- **Configured Model:** `qwen/qwen3.8-27b` (read from `GROQ_MODEL` in `.env` or defaulting to `qwen/qwen3.8-27b`).
- **Authentication:** `GROQ_API_KEY` stored in `.env`.
- **Request Parameters:** `temperature: 0.1`, `max_tokens: 700`, `timeout: 60.0s`.
- **Response Format:** Strictly enforced JSON structure:
  ```json
  {
    "prediction_summary": "string",
    "supporting_evidence": ["fact 1", "fact 2"],
    "limitations": ["limitation 1"],
    "unknowns": ["unknown 1"]
  }
  ```

### Verification of Multimodal vs. Text-Only Capabilities

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CRITICAL ARCHITECTURAL VERIFICATION                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Model Identifier: 'qwen/qwen3.8-27b' on Groq is a PURELY TEXTUAL Large Language     │
│    Model. It does NOT possess a vision encoder and cannot process pixel inputs.         │
│ 2. API Message Structure: The client sends standard text messages:                    │
│      {"role": "system", "content": system_prompt}                                      │
│      {"role": "user", "content": user_prompt}                                          │
│    No 'image_url' or base64 image objects are included in the Groq payload.           │
│ 3. Explicit System Prompt Instruction: 'src/api/llm_service.py:L235' states:           │
│      "You have NOT inspected raw image pixels yourself. You interpret only the        │
│       provided numerical and structural evidence."                                     │
│ 4. Conclusion: The existing Groq Qwen integration CANNOT visually inspect satellite   │
│    imagery. A vision-capable model (VLM) or structured visual extraction pipeline      │
│    is mandatory for natural-language interpretation of temporal satellite images.      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Can the Existing LLM Service Be Extended?

Yes. The existing service architecture is well-structured:
1. **Evidence-Grounded Prompting:** It already enforces a strict schema avoiding hallucinations. If the change-detection pipeline is upgraded to extract structured semantic evidence (e.g., connected-component bounding boxes, change area in square meters, spectral vegetation index shifts $\Delta\text{NDVI}$, pre/post surface classification), the existing text-based LLM can generate high-quality grounded explanations.
2. **Vision Model Compatibility:** If Groq's multimodal endpoints (e.g., `llama-3.2-11b-vision-preview` or `llama-3.2-90b-vision-preview`) or an external VLM provider are configured, the service structure (`_call_groq_chat`, caching, schema validation) can be adapted to send image inputs.

---

## 6. Existing API Endpoints and Frontend Interfaces

### REST API Endpoints Inventory

| Endpoint | HTTP Method | Input Contract | Output Contract | Status / Reusability for New Feature |
|---|---|---|---|---|
| `/api/health` | `GET` | None | `HealthResponse` | **Reusable as-is.** |
| `/api/search` | `POST` | `SearchRequest` | `SearchResponse` | Reusable for semantic search. |
| `/api/pairs` | `GET` | None | `List[PairInfo]` | Reusable for pre-indexed test scenes. |
| `/api/change-detection` | `POST` | `ChangeDetectionRequest` | `ChangeDetectionResponse` | **Directly reusable for pair inference.** Requires fixing hardcoded $1024\times 1024$ canvas in service layer. |
| `/api/explain/change-detection` | `POST` | `ExplainChangeDetectionRequest` | `ExplanationResponse` | **Reusable.** Currently accepts numerical telemetry. Needs extension for semantic evidence. |
| `/api/explain/semantic-search` | `POST` | `ExplainSearchRequest` | `ExplanationResponse` | Reusable as-is for retrieval tasks. |
| `/api/llm/status` | `GET` | `check: bool = False` | `LLMStatusResponse` | **Reusable as-is.** Checks Groq readiness. |
| `/api/benchmark` | `GET` | None | `BenchmarkResponse` | Reusable for publication results. |
| `/api/image/{dataset}/{path:path}` | `GET` | Path params | `FileResponse` (PNG/JPEG) | Reusable for pre-indexed images. |
| `/api/figures/{figure_name}` | `GET` | Path params | `FileResponse` | Reusable for research charts. |

### Endpoints Needed for the New Feature
1. **Multipart Upload Endpoint:** `POST /api/upload/pair` accepting multipart `FormData` (files `file_t1` and `file_t2`) to avoid inefficient base64 JSON serialization.
2. **Semantic Temporal Explanation Endpoint:** `POST /api/explain/temporal-understanding` accepting change masks, extracted region descriptors, and optional image thumbnails to return change-type classification and natural language reasoning.

---

### Frontend Interface Comparison & Placement Analysis

The system currently features two complete user interfaces:

#### 1. Streamlit Application (`app.py`, `pages/`, `ui/`)
- **Structure:** Multi-page layout with 5 navigation sections:
  1. *Overview* (`pages/overview.py`)
  2. *Dataset & Preprocessing* (`pages/dataset.py`)
  3. *Semantic Retrieval* (`pages/retrieval.py`)
  4. *Change Detection* (`pages/change_detection.py`)
  5. *False-Alarm Analysis* (`pages/false_alarm.py`)
- **Change Detection Page Structure:**
  - *Section 1: Live Change Detection on LEVIR-CD Scene Pairs* (Pair selector dropdown, detector selector, gamma selector, manual threshold slider, 4-column viewer, confusion metrics, and QAT diagnostics).
  - *Section 2: Milestone M11 Research Evidence & Benchmarks* (5 tabs: Benchmark Comparison, Bootstrap CIs, Paired Wilcoxon Tests, QAT Ablation, Figure Gallery).
- **Natural Placement for New Upload Feature:**
  - **Do NOT create a 6th navigation page.**
  - In `pages/change_detection.py`, add an input mode toggle at the top of Section 1:
    - Option A: *Pre-indexed LEVIR-CD Test Scene* (existing dropdown).
    - Option B: *Upload Custom Bitemporal Pair ($T_1 / T_2$)* (two `st.file_uploader` widgets with instant preview).

#### 2. Static Web Frontend (`frontend/index.html`, `frontend/js/`)
- **Structure:** Single Page Application (SPA) with 4 tabs:
  1. *Semantic Search* (`#panel-search`)
  2. *Change Detection* (`#panel-change`)
  3. *Research Results* (`#panel-results`)
  4. *Methodology & Limitations* (`#panel-methodology`)
- **Natural Placement for New Upload Feature:**
  - Inside `#panel-change`, add a segmented tab control:
    - *Benchmark Scene Selection*
    - *Custom $T_1 / T_2$ Upload Zone* (drag-and-drop file inputs for $T_1$ and $T_2$).

---

## 7. Capability Comparison: Existing vs. Required

| Capability | Existing Implementation Status | Required for New Feature | Analysis & Verification Notes |
|---|---|---|---|
| **Upload $T_1$ Image** | `Partially Implemented` | **Confirm** | Backend supports base64 in `ChangeDetectionRequest`. No UI upload widget exists in Streamlit or Web frontend. |
| **Upload $T_2$ Image** | `Partially Implemented` | **Confirm** | Supported in backend base64 schema simultaneously with $T_1$. Missing frontend upload inputs. |
| **Image Validation** | `Partially Implemented` | **Improve** | Only validates that `shape(T1) == shape(T2)`. Needs dimension, format, corruption, and minimum size checks. |
| **Image Registration** | `Not Implemented` | **Required** | Zero alignment exists. Pipeline assumes pre-registered inputs. Must implement feature matching / warping. |
| **Change Detection Core** | `Implemented` | **Reuse** | 7 models ($B_1$–$B_3$, $M_7$–$M_{10}$) fully implemented and calibrated. $M_9$ BIT-CD and $M_{10}$ QAT-CD are optimal. |
| **Binary Change Mask** | `Implemented` | **Reuse** | `predict()` returns binary mask based on calibrated thresholds ($\tau^* = 0.4500$ for BIT-CD). |
| **Changed-Region Localization** | `Partially Implemented` | **Improve** | Generates pixel mask; does not currently extract discrete connected components, bounding boxes, or polygons. |
| **Change-Type Classification** | `Not Implemented` | **New Capability** | Models only detect binary change. No classification of building vs vegetation vs water vs road. |
| **Vegetation Change Identification** | `Not Implemented` | **New Capability** | No spectral vegetation indices (e.g., NDVI/VARI) or green-cover segmentation currently implemented. |
| **New Building Identification** | `Partially Implemented` | **Reuse / Extend** | LEVIR-CD learned models detect building changes implicitly, but do not differentiate construction vs demolition. |
| **Natural-Language Explanation** | `Implemented (Text-Only)` | **Extend** | Groq Qwen 27B service exists but only accepts numerical metadata. Must extend with visual/semantic evidence or VLM. |
| **Region-Specific Explanation** | `Not Implemented` | **Required** | Current explanation explains the entire scene globally; does not pinpoint individual change clusters. |
| **Confidence & Evidence Reporting** | `Implemented` | **Improve** | Metrics ($F_1$, precision, recall, IoU) and QAT quality signals are tracked; needs confidence scoring for unlabelled uploads. |
| **Before/After Visualization** | `Implemented` | **Reuse / Extend** | 4-column viewer and confusion overlay already exist in both frontends. Reusable for uploaded imagery. |

---

## 8. Reusable Files and Services

The following existing components should be preserved and reused directly:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               REUSABLE ASSET SHORTLIST                                 │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Component File           │ Role & Reuse Contribution                                   │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ src/change_detection/    │ Best-performing model (~552k params, F1=0.5927).            │
│ bit_cd.py                │ Direct reuse on arbitrary patches without retraining.       │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ src/change_detection/    │ Quality gating and defect diagnostics (illumination, blur,  │
│ qat_cd.py                │ edge shifts). Direct reuse over BIT-CD.                     │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ src/change_detection/    │ Standard BaseChangeDetector abstraction ensuring all        │
│ base.py                  │ detectors conform to `compute_difference_map` & `predict`.  │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ src/api/llm_service.py   │ Robust Groq API wrapper with timeout handling, caching,     │
│                          │ and JSON schema enforcement. Extend with semantic evidence. │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ src/data/                │ Synchronized tiling engine. Reuse patch extraction logic    │
│ patch_extractor.py       │ after fixing dynamic canvas reconstruction.                 │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ ui/image_viewer.py       │ Side-by-side $T_1$, $T_2$, difference heatmap, and mask     │
│                          │ rendering. Directly reusable in Streamlit upload flow.      │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ src/api/service.py       │ `array_to_png_data_uri` and detector caching logic.         │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### Detailed Component Analysis

1. **`src/change_detection/bit_cd.py` (`BITChangeDetector`)**
   - *Current Functionality:* Computes bitemporal transformer attention maps between $T_1$ and $T_2$.
   - *Contribution:* Serves as the primary core inference engine for custom uploaded image pairs.
   - *Modifications Needed:* None to the model weights or architecture. Wrap inference to ensure input dimensions are divisible by 4.
   - *Existing Test Coverage:* Covered by `tests/test_m9_model.py` and `tests/test_m9_integration.py` (parameter ceiling, temporal symmetry, forward dimensions).
2. **`src/change_detection/qat_cd.py` (`QualityAwareChangeDetector`)**
   - *Current Functionality:* Attenuates false positives from misregistration and illumination differences.
   - *Contribution:* Generates diagnostic maps for custom uploads to explain whether changes are genuine or sensor-induced artifacts.
   - *Modifications Needed:* None.
   - *Existing Test Coverage:* Covered by `tests/test_m10_model.py` and `tests/test_m10_integration.py` (mathematical bounds, invariants, zero-gamma identity).
3. **`src/data/patch_extractor.py` (`PatchExtractor`)**
   - *Current Functionality:* Deconstructs large scenes into $256 \times 256$ patches and re-stitches difference maps.
   - *Contribution:* Allows arbitrary high-resolution satellite imagery (e.g., $2048 \times 2048$ or non-square) to be processed under CPU memory limits.
   - *Modifications Needed:* Ensure `reconstruct_image()` dynamically uses original image height and width rather than hardcoded 1024.
   - *Existing Test Coverage:* Covered by `tests/test_patch_extractor.py` and `tests/test_patch_reconstruction.py`.
4. **`src/api/llm_service.py` (`LLMExplanationService`)**
   - *Current Functionality:* Calls Groq `qwen/qwen3.8-27b` with retry and fallback parsing.
   - *Contribution:* Base explanation pipeline. Can be extended to accept structured visual evidence or route to a multimodal vision model.
   - *Modifications Needed:* Add explanation prompt builder for semantic change categories and regional detections.
   - *Existing Test Coverage:* Covered by `tests/test_llm_explainability.py` (mocked provider, auth error handling, JSON schema compliance).

---

## 9. Missing Capabilities

To implement the target feature, the following technical capabilities must be built:

1. **Frontend Upload Widgets:**
   - Streamlit: File upload components for $T_1$ and $T_2$ with format validation and instant side-by-side preview.
   - Static Web: Drag-and-drop upload zone with preview thumbnails and progress feedback.
2. **Robust Image Preprocessing & Sanitization:**
   - Multi-format loader supporting PNG, JPG, and single/multi-band TIFFs.
   - Dynamic canvas resizing / padding ensuring compatibility with patch extraction.
3. **Automated Image Registration & Co-Alignment:**
   - Feature matching (e.g., ORB, SIFT, or ECC alignment via OpenCV) to detect affine/homography transforms between $T_1$ and $T_2$ and warp $T_2$ onto $T_1$'s coordinate system before change detection.
4. **Dynamic Canvas Patch Reconstruction:**
   - Update `ChangeDetectionService.detect_change` to reconstruct difference maps matching the exact $(H, W)$ of uploaded inputs rather than $(1024, 1024)$.
5. **Semantic Change-Type Classification:**
   - Connected-component clustering to isolate discrete change regions.
   - Spectral/color metric analysis (e.g., Visible Atmospherically Resistant Index [VARI] or Green Leaf Index [GLI] on RGB imagery) to determine vegetation loss vs. new construction vs. soil clearing.
6. **Region-of-Interest (ROI) Cropping & Bounding Boxes:**
   - Extract top-K largest changed clusters with bounding boxes and patch crops.
7. **Multimodal / Structured Vision Explanation Pipeline:**
   - Multi-modal VLM integration (e.g., Groq Llama 3.2 Vision) or enhanced structured evidence prompting that feeds localized change attributes into the LLM.

---

## 10. Technical Risks and Dependencies

1. **CPU Memory & Latency Constraints:**
   - *Risk:* Processing arbitrary high-resolution images (e.g., $4000 \times 4000$) on CPU could exceed the 8 GB RSS budget or take minutes.
   - *Mitigation:* Downsample large uploads to a standard processing resolution (e.g., max 1024 or 2048) or enforce a maximum file resolution during upload.
2. **Misregistration False Positives on Arbitrary Pairs:**
   - *Risk:* If uploaded images are taken from different camera angles, perspective distortion cannot be corrected by simple affine warping, causing boundary false positives.
   - *Mitigation:* Apply QAT-CD quality gating ($\gamma = 0.30$) to suppress edge discrepancy false alarms, and warn the user when feature matching confidence is low.
3. **Groq Model Vision Limitations:**
   - *Risk:* The currently configured `qwen/qwen3.8-27b` model cannot see images. Groq's multimodal vision models (e.g., Llama 3.2 Vision) may have different rate limits or require updated API schemas.
   - *Mitigation:* Build a two-stage explanation design: Stage 1 extracts deterministic numerical and semantic facts (area, bounding boxes, color shift); Stage 2 generates the explanation using the LLM. If a vision model is available, thumbnails of changed regions can be attached.
4. **Ground Truth Absence:**
   - *Risk:* Arbitrary uploaded images will not have ground truth masks. Evaluation metrics ($F_1$, IoU) cannot be computed.
   - *Mitigation:* The UI must gracefully switch from "Benchmark Evaluation Mode" (showing precision/recall/F1/confusion overlays) to "Operational Inference Mode" (showing change area percentage, change cluster count, and confidence scores).

---

## 11. Preliminary Recommendation for the New Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED TEMPORAL UNDERSTANDING ARCHITECTURE                        │
└────────────────────────────────────────────────────────────────────────────────────────┘

    [ User Uploads Image T1 & Image T2 ]
                     │
                     ▼
    ┌───────────────────────────────────┐
    │  1. Ingestion & Sanitization      │  --> Validate format, depth, dimensions
    └─────────────────┬─────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────┐
    │  2. Automated Co-Registration     │  --> Feature matching (ORB/ECC) & Warping
    └─────────────────┬─────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────┐
    │  3. Change Detection Core         │  --> M9 BIT-CD + M10 QAT-CD Quality Gating
    └─────────────────┬─────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────┐
    │  4. Connected Components & Clust. │  --> Bounding boxes, change area (m²/px)
    └─────────────────┬─────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────┐
    │  5. Semantic Change Interpreter   │  --> Spectral index shift (Vegetation vs Bare)
    └─────────────────┬─────────────────┘      + Texture / Building classifier
                     │
                     ▼
    ┌───────────────────────────────────┐
    │  6. AI Natural Language Explainer │  --> Structured Evidence Prompting / VLM
    └─────────────────┬─────────────────┘
                     │
                     ▼
    [ Interactive Dual-UI Display (Streamlit & Static Web) ]
```

### Architectural Principles
1. **Preserve Milestone Integrity:** Do not modify existing benchmark datasets, checkpoints, or M1–M13 results.
2. **Seamless UI Integration:** In Streamlit, integrate custom pair upload as a toggle inside `pages/change_detection.py:Section 1` rather than cluttering navigation with a new tab. In the static web frontend, add a toggle in the Change Detection panel.
3. **CPU-First Execution:** Maintain strict memory-efficient processing with patch-based execution and image downscaling for overly large uploads.
4. **Evidence-Grounded Explanations:** Continue the established design principle: explanations must be derived from verified algorithmic telemetry, preventing LLM hallucinations.

---

*Report prepared and verified in accordance with Milestone M1–M13 academic integrity standards.*
