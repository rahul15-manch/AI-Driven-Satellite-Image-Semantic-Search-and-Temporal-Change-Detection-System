# Literature Review: Semantic Retrieval, Change Detection, and Quality-Aware False-Alarm Suppression in Remote Sensing

**Milestone:** M3 — Research Definition, Literature Review & Baseline Formalization  
**Author:** Adishri Abro (Literature & Evaluation Lead)  
**Collaborators:** Rahul (Team Lead), Tanishka Mukhi (Dataset & Architecture Lead)  
**Epistemic Taxonomy:** All claims in this document are explicitly categorized into:  
- `[VERIFIED LITERATURE]` — Directly supported by verified peer-reviewed publications.  
- `[LOCALLY MEASURED]` — Quantified on our local verified datasets (RSICD, LEVIR-CD).  
- `[PROJECT DECISION]` — Explicit engineering and architectural choices made by the project team.  
- `[HYPOTHESIS]` — Scientifically framed proposition requiring empirical verification.  
- `[INFERENCE]` — Logical deduction derived from verified literature or empirical constraints.  
- `[UNKNOWN]` — Question or domain behavior not yet definitively resolved by literature or experiments.

---

## 1. Overview and Problem Context

Earth observation (EO) systems and commercial satellite constellations acquire millions of square kilometers of multispectral and high-resolution optical imagery daily. Exploiting this massive volume of data requires two core capabilities:
1. **Semantic Image Retrieval:** Enabling users to locate relevant geographic scenes across vast unstructured image repositories using natural-language descriptions rather than rigid spatial coordinates or categorical tags `[VERIFIED LITERATURE: Lu et al., 2018]`.
2. **Bi-Temporal Change Detection (CD):** Identifying meaningful, real-world surface transformations (e.g., new building construction, infrastructure expansion, deforestation) between two co-registered optical scenes ($T_1, T_2$) captured over the same geographic footprint at different dates `[VERIFIED LITERATURE: Chen & Shi, 2020; Daudt et al., 2018]`.

### Operational Context and Computational Bottlenecks
While recent computer vision literature increasingly explores massive transformer backbones (e.g., ViT-Large, Swin-Base) and multi-billion-parameter foundation models evaluated on enterprise GPU clusters, real-world academic and institutional deployments frequently operate under strict computational bounds `[PROJECT DECISION: DEC-003]`. Our project investigates the Pareto frontier between predictive performance and computational feasibility on standard commodity CPU student laptops (quad/octa-core, $\le 8$ GB RAM, no GPU acceleration).

Furthermore, in bi-temporal change detection, real-world satellite imagery is corrupted by nuisance factors: solar angle differences, cloud cover, cast shadows, atmospheric haze, seasonal vegetation phenology, and residual registration jitter `[VERIFIED LITERATURE: Malila, 1980; Singh, 1989; Bovolo & Bruzzone, 2007]`. These factors produce spurious radiometric differences that classical and naive deep models often mistake for true structural changes, leading to elevated False Alarm Rates (FAR) `[VERIFIED LITERATURE: Chen & Shi, 2020]`.

---

## 2. Remote Sensing Image-Text Retrieval

### 2.1 Evolution of Cross-Modal Retrieval in Remote Sensing
Image retrieval in remote sensing originally relied on Content-Based Image Retrieval (CBIR) using handcrafted spectral, textural, and morphological descriptors (e.g., SIFT, GIST, color histograms) `[VERIFIED LITERATURE: Lu et al., 2018]`. With the advent of deep convolutional networks, feature embeddings extracted from ImageNet-pretrained CNNs (e.g., ResNet, VGG) provided improved visual discriminability.

However, CBIR systems require query images. Natural-Language Text-to-Image Retrieval (TIR) enables users to formulate compositional semantic queries (e.g., *"commercial airplane parked near a terminal building"* or *"dense residential area adjacent to a river"*). The foundational benchmark for this task was established by Lu et al. (2018) with the creation of the Remote Sensing Image Captioning Dataset (RSICD) `[VERIFIED LITERATURE: Lu et al., 2018]`.

### 2.2 Dual-Encoder Framework and Vision-Language Embeddings
Modern cross-modal retrieval is dominated by dual-encoder architectures pioneered in general computer vision by Radford et al. (2021) in Contrastive Language-Image Pre-training (CLIP) `[VERIFIED LITERATURE: Radford et al., 2021]`. 

```
               ┌───────────────────────┐
Text Query  ──►│ Text Encoder (E_text) ├──► u = E_text(Text) / ||E_text|| ──┐
               └───────────────────────┘                                    │  Cosine Similarity
                                                                            ├──► S(u, v) = u^T v
               ┌────────────────────────┐                                   │
Image Scene ──►│ Image Encoder (E_img)  ├──► v = E_img(Image) / ||E_img|| ──┘
               └────────────────────────┘
```

The dual-encoder projects image $I$ and text $T$ into a shared $D$-dimensional metric latent space $\mathbb{R}^D$:
$$\mathbf{v} = \frac{E_{\text{img}}(I)}{\|E_{\text{img}}(I)\|_2}, \quad \mathbf{u} = \frac{E_{\text{text}}(T)}{\|E_{\text{text}}(T)\|_2}$$

Because both vectors are $L_2$-normalized, their dot product is equivalent to cosine similarity:
$$S(I, T) = \langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} \in [-1, 1]$$

### 2.3 Domain Shift and Limitations of General-Domain Models
While general-domain models like OpenAI CLIP (trained on 400 million internet image-text pairs WIT) exhibit strong zero-shot transfer on everyday photographic concepts, remote sensing imagery presents severe domain shift `[VERIFIED LITERATURE: Radford et al., 2021; Liu et al., 2024]`:
1. **Nadir and Overhead Viewing Angle:** Satellite images are captured top-down (zenith perspective), lacking canonical perspective cues (e.g., horizon, frontal views of objects).
2. **Multi-Scale and Dense Tiny Objects:** A single $224 \times 224$ satellite tile may contain tens of small buildings, roads, or vehicles without a single salient foreground object.
3. **Domain Vocabulary Gap:** Specialized geographic terminology (e.g., *"viaduct"*, *"storage tanks"*, *"farmland"*, *"bare land"*) has lower representation in internet photographic captions.

Recent domain-adapted foundation models such as RemoteCLIP `[VERIFIED LITERATURE: Liu et al., 2024]` and GeoRSCLIP `[VERIFIED LITERATURE: Zhang et al., 2024]` demonstrated that contrastive continual pretraining on curated remote sensing pairs improves retrieval metrics (e.g., Recall@1) over vanilla CLIP. However, for a CPU-constrained baseline study, vanilla CLIP (ViT-B/32 or ResNet-50) serves as an indispensable, zero-shot reference point `[INFERENCE]`.

### 2.4 Vector Similarity Search and CPU Feasibility
Retrieval over an image catalog involves indexing image embeddings $\{\mathbf{v}_i\}_{i=1}^N$. For $N = 1,093$ (the RSICD test split) or even $N = 10,921$ (the full RSICD dataset), exact $k$-nearest neighbor search using an exhaustive flat inner-product index (`IndexFlatIP` in FAISS) requires computing $N$ dot products per query:
$$\text{FLOPs} = N \times D \times 2$$
For $D = 512$ and $N = 10,921$, this is $\approx 1.12 \times 10^7$ FLOPs, which executes in under 2 milliseconds on a single modern CPU thread `[INFERENCE]`. Therefore, approximate nearest neighbor (ANN) quantization methods (e.g., IVF, HNSW) are unnecessary for datasets of this scale and would only introduce quantization distortion.

---

## 3. Vision-Language Models for Remote Sensing

### 3.1 Architecture Selection: ViT-B/32 vs. ResNet-50 vs. Lightweight Alternatives
CLIP models commonly use either a Vision Transformer (ViT) or a Modified ResNet backbone `[VERIFIED LITERATURE: Radford et al., 2021]`:
- **ViT-B/32:** Splits a $224 \times 224$ image into $7 \times 7 = 49$ non-overlapping patches of size $32 \times 32$. Embedding dimension $D = 512$. Parameter count: $\approx 87.8$M (image encoder + text encoder).
- **ResNet-50:** Classical 4-stage convolutional backbone with attention pooling. Parameter count: $\approx 102$M.
- **MobileCLIP / TinyCLIP:** Knowledge-distilled models offering sub-30M parameters, optimized for low-latency mobile/edge inference.

Under CPU inference, ViT-B/32 exhibits lower FLOPs per $224 \times 224$ image than standard ResNet-50 and native patch tokenization aligned with transformer attention `[INFERENCE]`.

### 3.2 Prompt Formulation and Prompt Engineering
Zero-shot transfer performance in CLIP is highly sensitive to the query prompt template `[VERIFIED LITERATURE: Radford et al., 2021]`. Using a bare prompt (e.g., `"airport"`) often underperforms compared to context-guided prompt templates such as:
- `"a satellite image of a {category}"`
- `"an aerial photograph of {query}"`
- `"overhead view of {query}"`
Ensembling multiple prompt templates via average embedding projection $\mathbf{u}_{\text{ens}} = \frac{1}{M} \sum_{m=1}^M \mathbf{u}_m$ is documented in Radford et al. (2021) to consistently improve retrieval accuracy without requiring retraining `[VERIFIED LITERATURE: Radford et al., 2021]`.

---

## 4. Classical Change Detection Methods

Classical change detection operates without learned neural parameters, relying on deterministic mathematical comparisons of aligned bi-temporal spectral or structural values `[VERIFIED LITERATURE: Malila, 1980; Singh, 1989; Bovolo & Bruzzone, 2007]`.

### 4.1 Pixel Differencing (B1)
The simplest change indicator computes the point-wise absolute difference between co-registered optical images $I_1$ and $I_2$ `[VERIFIED LITERATURE: Singh, 1989]`:
- For single-channel grayscale $I_1, I_2 \in [0, 1]^{H \times W}$:
  $$D_{\text{diff}}(x, y) = |I_1(x, y) - I_2(x, y)|$$
- For multi-channel RGB imagery $I_1, I_2 \in [0, 1]^{H \times W \times C}$:
  $$D_{\text{diff}}(x, y) = \frac{1}{\sqrt{C}} \|\mathbf{I}_1(x, y) - \mathbf{I}_2(x, y)\|_2 = \sqrt{\frac{1}{C} \sum_{c=1}^C (I_1(x, y, c) - I_2(x, y, c))^2}$$
A binary change mask $M \in \{0, 1\}^{H \times W}$ is obtained via thresholding:
$$M(x, y) = \mathbb{I}[D_{\text{diff}}(x, y) > \tau]$$

### 4.2 Structural Similarity Index Measure (SSIM) Differencing (B2)
Wang et al. (2004) introduced SSIM to quantify visual degradation based on three conceptually distinct components: luminance $l(\mathbf{x}, \mathbf{y})$, contrast $c(\mathbf{x}, \mathbf{y})$, and structure $s(\mathbf{x}, \mathbf{y})$ computed over local sliding windows (typically $11 \times 11$ Gaussian kernels with $\sigma = 1.5$) `[VERIFIED LITERATURE: Wang et al., 2004]`:
$$\text{SSIM}(\mathbf{x}, \mathbf{y}) = [l(\mathbf{x}, \mathbf{y})]^\alpha \cdot [c(\mathbf{x}, \mathbf{y})]^\beta \cdot [s(\mathbf{x}, \mathbf{y})]^\gamma$$
With $\alpha = \beta = \gamma = 1$, this simplifies to:
$$\text{SSIM}(\mathbf{x}, \mathbf{y}) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
Where:
- $\mu_x, \mu_y$ are the local sample means.
- $\sigma_x^2, \sigma_y^2$ are local sample variances.
- $\sigma_{xy}$ is the local sample cross-covariance.
- $C_1 = (K_1 L)^2, C_2 = (K_2 L)^2$ are stabilizing constants ($K_1 = 0.01, K_2 = 0.03$, dynamic range $L = 1.0$).

In change detection, the structural change map is defined as the structural dissimilarity `[VERIFIED LITERATURE: Wang et al., 2004; Bovolo & Bruzzone, 2007]`:
$$D_{\text{SSIM}}(x, y) = 1.0 - \text{SSIM}(x, y) \in [0, 2]$$
Because SSIM normalizes for local mean luminance differences via $\mu_x$ and $\mu_y$, it is inherently more resilient to uniform illumination shifts than raw pixel differencing `[INFERENCE]`.

### 4.3 Change Vector Analysis (CVA) (B3)
Originating in multispectral remote sensing by Malila (1980), Change Vector Analysis models the multi-band spectral change as a vector displacement in $C$-dimensional radiometric space `[VERIFIED LITERATURE: Malila, 1980; Bovolo & Bruzzone, 2007]`:
$$\Delta \mathbf{R}(x, y) = \mathbf{I}_2(x, y) - \mathbf{I}_1(x, y) = \begin{bmatrix} I_2(x, y, 1) - I_1(x, y, 1) \\ \vdots \\ I_2(x, y, C) - I_1(x, y, C) \end{bmatrix}$$
The magnitude of change $\rho(x, y)$ represents the total spectral displacement:
$$\rho(x, y) = \|\Delta \mathbf{R}(x, y)\|_2 = \sqrt{\sum_{c=1}^C (I_2(x, y, c) - I_1(x, y, c))^2}$$
The direction of the change vector in spectral space indicates the nature or type of land-cover transition. For binary change detection on optical RGB imagery, CVA magnitude is proportional to Euclidean channel difference, but provides the theoretical basis for multi-thresholding and spectral angle mapping (SAM) `[VERIFIED LITERATURE: Bovolo & Bruzzone, 2007]`.

### 4.4 Automated Thresholding via Otsu's Method
Classical methods generate a continuous difference surface $D(x, y) \in [0, 1]$. Binarizing $D$ into change/no-change requires selecting a threshold $\tau$. Otsu (1979) established the canonical non-parametric, unsupervised thresholding method that maximizes between-class variance $\sigma_B^2(\tau)$ across gray-level histogram bins $t \in [0, L-1]$ `[VERIFIED LITERATURE: Otsu, 1979]`:
$$\sigma_B^2(\tau) = \omega_0(\tau) (\mu_0(\tau) - \mu_T)^2 + \omega_1(\tau) (\mu_1(\tau) - \mu_T)^2 = \omega_0(\tau) \omega_1(\tau) (\mu_0(\tau) - \mu_1(\tau))^2$$
Where $\omega_0(\tau), \omega_1(\tau)$ are probabilities of the background and changed classes, and $\mu_0(\tau), \mu_1(\tau)$ are their mean levels.

---

## 5. Deep Learning Change Detection Methods

### 5.1 Fully Convolutional Siamese Networks (Daudt et al., 2018)
Daudt, Le Saux, and Boulch (2018) introduced fully convolutional Siamese architectures that became the foundational modern baselines for high-resolution optical change detection `[VERIFIED LITERATURE: Daudt et al., 2018]`:
1. **FC-EF (Early Fusion):** Concatenates $I_1$ and $I_2$ along the channel dimension into a $2C$-channel input ($6$ channels for RGB) and passes it through a standard U-Net encoder-decoder `[VERIFIED LITERATURE: Ronneberger et al., 2015; Daudt et al., 2018]`.
2. **FC-Siam-diff (Siamese Differencing):** Employs twin weight-sharing convolutional encoders for $I_1$ and $I_2$. At each hierarchical skip level $l \in \{1, \dots, L\}$, the feature maps are differenced:
   $$\mathbf{F}_{\text{diff}}^{(l)} = |\mathbf{F}_1^{(l)} - \mathbf{F}_2^{(l)}|$$
   These difference maps are passed via skip connections into the expanding decoder.
3. **FC-Siam-conc (Siamese Concatenation):** Identical twin encoders, but features are concatenated along the channel dimension $[\mathbf{F}_1^{(l)}, \mathbf{F}_2^{(l)}]$ before decoding.

```
       I_1 ──► [Encoder E] ──► F_1^(l) ──┐
                   ▲                     ├──► |F_1^(l) - F_2^(l)| ──► [Decoder D] ──► Change Map
             (shared weights)            │                                 ▲
                   ▼                     ├──► Skip connections ────────────┘
       I_2 ──► [Encoder E] ──► F_2^(l) ──┘
```

Daudt et al. (2018) demonstrated that `FC-Siam-diff` and `FC-Siam-conc` outperform early fusion by enforcing symmetric, invariant feature extraction prior to comparison `[VERIFIED LITERATURE: Daudt et al., 2018]`. With a compact backbone (e.g., 4 encoder stages with initial channel width 16 or 32), `FC-Siam-diff` has only $\approx 1.35$M parameters, making it highly feasible for CPU inference `[INFERENCE]`.

### 5.2 Attention-Based and Transformer Models (STANet, BIT, ChangeFormer)
Chen and Shi (2020) proposed STANet (Spatial-Temporal Attention Network) concurrently with the introduction of LEVIR-CD `[VERIFIED LITERATURE: Chen & Shi, 2020]`. STANet incorporates self-attention modules to capture spatio-temporal dependencies at multiple scales. Subsequently, Bitemporal Image Transformer (BIT) (Chen et al., 2021) and ChangeFormer (Bandara & Patel, 2022) utilized Vision Transformers to model long-range context.

**Operational Implication:** While BIT and ChangeFormer represent state-of-the-art accuracy on GPU benchmarks (achieving F1 $> 90\%$ on LEVIR-CD), their multi-head self-attention mechanisms require tens of GFLOPs and excessive memory overhead during full $1024 \times 1024$ image inference. On CPU-only hardware, executing full transformer backbones results in latency of tens of seconds per scene and exceeds 8 GB RAM if un-tiled `[INFERENCE]`. Therefore, transformer models are reviewed as literature context, while lightweight Siamese CNNs (`FC-Siam-diff` / Tiny-UNet) serve as the appropriate CPU-executable learned baselines `[PROJECT DECISION: DEC-015]`.

---

## 6. False-Alarm and Confounder Handling Literature

Bi-temporal satellite observations taken months or years apart are corrupted by non-ground environmental factors `[VERIFIED LITERATURE: Singh, 1989; Bovolo & Bruzzone, 2007; Chen & Shi, 2020]`.

### 6.1 Confounder Taxonomy
To establish scientific clarity, image discrepancies must be separated into three mutually exclusive categories `[VERIFIED LITERATURE; PROJECT DECISION]`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          IMAGE PAIR DISCREPANCIES                           │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│  1. Ground-Truth Changes │  2. Apparent Non-Ground  │  3. Preprocessing /     │
│                         │        Variations       │    Geometric Artifacts  │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • New building erection │ • Solar zenith angle &  │ • Sub-pixel or multi-   │
│ • Demolition/excavation │   illumination shifts   │   pixel misregistration │
│ • Road construction     │ • Cast building/cloud   │ • Sensor optical blur   │
│ • Land use conversion   │   shadows               │ • JPEG compression      │
│                         │ • Seasonal phenology &  │   quantization noise    │
│                         │   vegetation browning   │ • Resolution mismatch   │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 6.2 Existing Confounder Mitigation Strategies
In remote sensing literature, four distinct approaches exist to mitigate non-ground false alarms:
1. **Relative Radiometric Normalization (RRN):** Techniques such as Pseudo-Invariant Features (PIF) and Iteratively Reweighted Multivariate Alteration Detection (IR-MAD) adjust the histogram or linear radiometric gain of $I_2$ to match $I_1$ prior to comparison `[VERIFIED LITERATURE: Hall et al., 1991; Canty & Nielsen, 2008]`.
2. **Deep Semantic Feature Invariance:** Deep neural networks trained on diverse augmentations learn representations invariant to illumination and seasonal shifts while sensitive to geometric boundaries `[VERIFIED LITERATURE: Daudt et al., 2018; Chen & Shi, 2020]`.
3. **Multi-Temporal Consistency Constraints:** Exploiting sequences of $\ge 3$ images to confirm that a change is monotonic (e.g., land does not flip from grass to building and back to grass in consecutive weeks) `[VERIFIED LITERATURE: Bovolo & Bruzzone, 2007]`.
4. **Post-Processing Morphological Filtering:** Area-opening filters to eliminate single-pixel noise clusters below a minimum building footprint `[VERIFIED LITERATURE: Ji et al., 2019]`.

---

## 7. Quality and Uncertainty-Aware Methods

### 7.1 Existing Prior Art in Quality Assessment & Uncertainty
In computer vision and remote sensing, quality and uncertainty mechanisms are established across several subfields:
- **No-Reference Image Quality Assessment (NR-IQA):** Algorithms such as BRISQUE (Mittal et al., 2012) quantify natural scene statistics and blur/noise degradation without reference images `[VERIFIED LITERATURE]`.
- **Bayesian Deep Learning & Monte Carlo Dropout:** Kendall and Gal (2017) formalized epistemic (model) and aleatoric (data-dependent) uncertainty for semantic segmentation `[VERIFIED LITERATURE: Kendall & Gal, 2017]`.
- **Confidence Maps in Change Detection:** Several remote sensing papers generate fuzzy change maps or evidential confidence intervals (e.g., Dempster-Shafer theory) to flag pixels where change assignment is ambiguous `[VERIFIED LITERATURE: Bovolo & Bruzzone, 2007]`.

### 7.2 Investigation of the Proposed Quality-Aware Approach
Our project proposed investigating a quality-aware change score conceptually represented as:
$$\text{Change Map}_{\text{final}} = f(\text{Raw Change Evidence}, \text{Image Quality}, \text{Registration Confidence}, \text{Temporal Consistency})$$

> [!IMPORTANT]
> **Novelty & Literature Audit Verdict on Quality-Aware Change Detection:**  
> A rigorous audit of external literature demonstrates that:
> 1. *Gating change scores using radiometric disparity or quality indicators is NOT a fundamentally novel paradigm.* Radiometric confidence weighting, co-registration error modeling, and uncertainty-aware filtering have been investigated in various formulations across remote sensing literature since the 1990s `[VERIFIED LITERATURE: Hall et al., 1991; Canty & Nielsen, 2008; Bovolo & Bruzzone, 2007]`.
> 2. *Therefore, our proposed approach cannot be claimed as a novel foundational theory.*
> 3. *Appropriate Scientific Classification:* It represents a **LIGHTWEIGHT ADAPTATION & COMBINATION** `[PROJECT DECISION: DEC-019]`.
> 4. *Legitimate Research Contribution:* Formulating a deterministic, computationally lightweight diagnostic gating layer that operates strictly under CPU constraints to suppress false positives under controlled perturbations (illumination disparity, blur, misregistration), compared fairly against un-gated baselines on verified benchmarks.

---

## 8. Dataset Audit and Citation Verification

### 8.1 Critical Citation Audit: The WHU Building Dataset
The initial project documentation and conventional literature frequently conflate two distinct papers published in 2019 by the research group of Prof. Shunping Ji at Wuhan University:

| Attribute | Paper A: Building Extraction | Paper B: Building Change Detection (The Actual CD Dataset) |
| :--- | :--- | :--- |
| **Full Title** | *Fully Convolutional Networks for Multisource Building Extraction From An Open Aerial and Satellite Imagery Dataset* | *Building Instance Change Detection from Large-Scale Aerial Images using Convolutional Neural Networks and Simulated Samples* |
| **Authors** | Shunping Ji, Shisheng Wei, Meng Lu | Shunping Ji, Shiqing Shen, Meng Lu, Yongjun Zhang |
| **Year & Venue** | 2019, *IEEE Transactions on Geoscience and Remote Sensing* (TGRS), Vol. 57, No. 1, pp. 574–586 | 2019, *Remote Sensing*, Vol. 11, No. 11, Article 1343 |
| **DOI** | `10.1109/TGRS.2018.2858824` | `10.3390/rs11111343` |
| **Core Task** | Mono-temporal building semantic segmentation / extraction | **Bi-temporal building change detection** (Christchurch aerial imagery) |
| **Audit Action** | **Correction:** Cited in many synopses as the change detection paper, but it only covers mono-temporal segmentation. | **Identified as Authoritative Citation** for WHU Building Change Detection. |

### 8.2 Benchmark Datasets Formally Adopted for This Project

#### 1. RSICD (Remote Sensing Image Captioning Dataset)
- **Authoritative Citation:** Lu, X., Wang, B., Zheng, X., & Li, X. (2018). *High-Resolution Remote Sensing Image Captioning*. IEEE Transactions on Geoscience and Remote Sensing, 56(4), 2198–2207. DOI: `10.1109/TGRS.2017.2776321`.
- **Local Validation Status `[LOCALLY MEASURED]`:** 10,921 images verified, $224 \times 224$ RGB JPEGs, 54,605 total captions (exactly 5 captions per image).
- **Split Manifest `[PROJECT DECISION: DEC-010]`:** Fixed Karpathy split:
  - Train: 8,734 images (43,670 captions)
  - Validation: 1,094 images (5,470 captions)
  - Test: 1,093 images (5,465 captions)

#### 2. LEVIR-CD (Large-Scale Building Change Detection Dataset)
- **Authoritative Citation:** Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing, 12(10), Article 1662. DOI: `10.3390/rs12101662`.
- **Local Validation Status `[LOCALLY MEASURED]`:** 637 bi-temporal image pairs verified, $1024 \times 1024$ RGB optical imagery, $0.5$ m/pixel ground resolution.
- **Split Manifest `[PROJECT DECISION: DEC-010]`:**
  - Train: 445 scene pairs ($7,120$ patches of size $256 \times 256$)
  - Validation: 64 scene pairs ($1,024$ patches of size $256 \times 256$)
  - Test: 128 scene pairs ($2,048$ patches of size $256 \times 256$)
- **Class Imbalance Quantification `[LOCALLY MEASURED]`:**
  - Total cumulative pixels across all 637 scenes: $667,942,912$
  - Changed pixels (binary 1): $31,066,643$ ($4.651\%$)
  - Non-changed pixels (binary 0): $636,876,269$ ($95.349\%$)
  - Imbalance ratio: $\approx 20.5 : 1$ non-change to change.

---

## 9. Evaluation Practices and Metric Formalization

### 9.1 Semantic Retrieval Evaluation Protocol
In remote sensing cross-modal retrieval, an evaluation query is matched against the candidate gallery of test images `[VERIFIED LITERATURE: Lu et al., 2018]`.

#### Relevance Definition
Each image $I_i$ in RSICD possesses 5 human-annotated captions $\{c_{i, 1}, \dots, c_{i, 5}\}$. In TIR literature, two distinct evaluation protocols exist:
1. **Caption-to-Own-Image (Instance Retrieval):** A query caption $q$ is considered relevant *strictly* to the specific physical image $I_i$ from which it was written. All other images are non-relevant, even if they share the same broad scene class (e.g., another airport) `[VERIFIED LITERATURE: Lu et al., 2018; Radford et al., 2021]`.
2. **Category Relevance (Semantic Class Retrieval):** Any image belonging to the same semantic class (e.g., any of the 30 classes such as `bridge`, `stadium`, `farmland`) is considered relevant.

`[PROJECT DECISION: DEC-016]` For our primary evaluation, we implement **Caption-to-Own-Image** retrieval over the 1,093 held-out test images (5,465 queries), directly adhering to the standardized protocol of Lu et al. (2018) and RemoteCLIP (Liu et al., 2024). We also report category-level Precision@K for qualitative user queries.

#### Formal Retrieval Metrics
1. **Recall@K ($K \in \{1, 5, 10\}$):**
   $$R@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}[\text{rank}(I_q) \le K]$$
   Where $\text{rank}(I_q)$ is the rank position of the ground-truth image $I_q$ among all candidate images sorted by descending cosine similarity.
2. **Mean Reciprocal Rank (MRR):**
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(I_q)}$$
3. **Mean Average Precision (mAP):**
   Evaluates ranked retrieval precision across all relevant items for multi-instance retrieval tasks.

### 9.2 Change Detection Metrics and the Class-Imbalance Trap
Change detection evaluates pixel-level binary classification:
- **True Positive (TP):** Changed pixel correctly predicted as changed ($Y = 1, \hat{Y} = 1$).
- **False Positive (FP):** Non-changed pixel incorrectly predicted as changed ($Y = 0, \hat{Y} = 1$) [False Alarm].
- **False Negative (FN):** Changed pixel missed by the detector ($Y = 1, \hat{Y} = 0$) [Missed Detection].
- **True Negative (TN):** Non-changed pixel correctly identified as non-changed ($Y = 0, \hat{Y} = 0$).

#### Derived Metrics
- **Precision:** $P = \frac{TP}{TP + FP}$
- **Recall:** $R = \frac{TP}{TP + FN}$
- **F1-Score (Change Class):** $F_1 = \frac{2 \cdot P \cdot R}{P + R} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$
- **Intersection-over-Union (IoU):** $\text{IoU} = \frac{TP}{TP + FP + FN}$

#### The Overall Accuracy (OA) Trap
Overall Accuracy is defined as:
$$\text{OA} = \frac{TP + TN}{TP + TN + FP + FN}$$
`[LOCALLY MEASURED & VERIFIED LITERATURE]` In LEVIR-CD, changed pixels comprise only $4.651\%$ of the dataset. A degenerate "dummy" model that predicts zero change everywhere ($\hat{Y}(x, y) = 0, \forall x, y$) trivially achieves:
$$\text{OA}_{\text{dummy}} = \frac{0 + 636,876,269}{667,942,912} = 95.349\%$$
Yet its F1-score and IoU are exactly $0.00\%$. **Therefore, Overall Accuracy must NEVER be used as the primary selection criterion for change detection.** F1-score and IoU on the changed class are the only scientifically defensible primary metrics `[PROJECT DECISION: DEC-016]`.

### 9.3 Computational and Efficiency Metrics
To respect the CPU constraint `[PROJECT DECISION: DEC-003]`, every algorithm must be profiled for:
1. **Per-Sample Inference Latency (ms):** Wall-clock time per image or tile crop, measured after 5 warmup runs, reporting mean and 95th percentile ($p95$).
2. **Peak RAM Consumption (MB):** Measured using Python `tracemalloc` and resident set size (`psutil.Process().memory_info().rss`).
3. **Model Parameter Count (M):** Total trainable and non-trainable floating-point parameters.
4. **Index Memory Footprint (MB):** Memory required to store the FAISS retrieval index.

---

## 10. Comparative Analysis & Literature Evidence Table

The following table summarizes verified, peer-reviewed literature directly grounding our experimental design:

| Paper Citation | Year | Method / Model | Dataset Evaluated | Task | Key Finding | Critical Limitations | Project Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Radford et al.** (ICML) | 2021 | Contrastive Language-Image Pre-training (CLIP) | WebImageText (400M), ImageNet | Zero-shot image-text retrieval | Contrastive pretraining enables strong zero-shot transfer across general photographic concepts. | Severe domain gap on remote sensing; zenith angle and dense small objects unrepresented. | Establishes candidate VLM architecture (ViT-B/32) and retrieval baseline A2. |
| **Lu et al.** (IEEE TGRS) | 2018 | Multimodal Deep Learning (CNN + RNN) | RSICD, UCM, Sydney | Remote sensing captioning & retrieval | Standardized RSICD benchmark; established caption-to-image retrieval protocol and metrics. | Early RNN captioning; limited vocabulary; no contrastive dual-encoder pretraining. | Direct source of RSICD dataset and baseline retrieval evaluation protocols. |
| **Liu et al.** (IEEE TGRS) | 2024 | RemoteCLIP | RSICD, RSITD, UCM | RS image-text retrieval | Domain-specific continual pretraining significantly boosts R@1 on RSICD over vanilla CLIP. | Requires significant GPU resources to pretrain; weights larger than lightweight CNNs. | Literature context demonstrating the domain shift gap of zero-shot CLIP. |
| **Malila** (LARS) | 1980 | Change Vector Analysis (CVA) | Landsat MSS | Multispectral change detection | Spectral difference vector magnitude indicates change; direction indicates transition type. | Sensitive to atmospheric and seasonal variations; produces high false alarm rate. | Theoretical and mathematical foundation for classical baseline B3. |
| **Wang et al.** (IEEE TIP) | 2004 | Structural Similarity (SSIM) | TID2008, LIVE | Image Quality Assessment | Separates luminance, contrast, and structure; resilient to uniform illumination shifts. | Handcrafted sliding window; cannot distinguish semantic structure from transient texture. | Foundational formulation for classical structural baseline B2. |
| **Otsu** (IEEE TSMC) | 1979 | Otsu's Thresholding | Gray-level imagery | Unsupervised threshold selection | Maximizes between-class variance; eliminates subjective manual thresholding. | Fails under severe class imbalance or multimodal background distributions. | Unsupervised threshold selection method for classical difference maps. |
| **Daudt et al.** (IEEE ICIP) | 2018 | FC-EF, FC-Siam-diff, FC-Siam-conc | Onera Satellite Change Dataset (OSCD) | Bi-temporal change detection | Siamese difference (FC-Siam-diff) achieves superior boundary localization over early fusion. | Evaluated on lower-resolution multispectral scenes; requires tiled training. | Primary lightweight deep baseline B4 for CPU change detection. |
| **Ronneberger et al.** (MICCAI) | 2015 | U-Net | ISBI Cell Tracking | Biomedical segmentation | U-shaped encoder-decoder with skip connections preserves high-frequency spatial details. | Not intrinsically bi-temporal; requires concatenation or differencing adaptation. | Architectural backbone for both FC-Siam-diff and lightweight decoder baselines. |
| **Chen & Shi** (Remote Sensing) | 2020 | STANet | LEVIR-CD | Building change detection | Spatial-temporal attention captures multi-scale building changes; created LEVIR-CD benchmark. | Self-attention modules are computationally heavy on CPU; high memory footprint. | Authoritative benchmark paper for LEVIR-CD dataset and standard splits. |
| **Ji et al.** (Remote Sensing) | 2019 | CNN Instance Change Detection | WHU Building Change Dataset | Aerial building change detection | Combining simulated building samples with CNN reduces false alarms in complex urban zones. | High false positive rate under seasonal shadow displacement. | Authoritative citation for WHU Change Detection (audited against Ji et al. TGRS extraction). |
| **Hall et al.** (RSE) | 1991 | Relative Radiometric Normalization (RRN) | Landsat MSS | Radiometric correction | Linear regression on pseudo-invariant features eliminates solar/atmospheric differences. | Requires identifying invariant targets; sensitive to registration misalignment. | Theoretical basis for false-alarm suppression via radiometric alignment. |

---

## 11. Research Gap Analysis

Based on the verified literature and empirical constraints, we identify four concrete, evidence-backed research gaps:

### Gap 1: Lack of Systematic CPU-Constrained Trade-off Profiling Across Classical vs. Learned Methods
Existing remote sensing change detection literature focuses almost exclusively on maximizing benchmark F1-scores using GPU-accelerated transformer architectures `[VERIFIED LITERATURE: Chen & Shi, 2020; Chen et al., 2021]`. There is an absence of systematic, reproducible studies that quantify the exact Pareto trade-off between predictive accuracy (F1/IoU) and CPU hardware consumption (latency, peak RAM) comparing calibrated classical methods (differencing, SSIM, CVA) against lightweight Siamese CNNs on modern high-resolution benchmarks.

### Gap 2: Fragility of Change Detectors Under Controlled Radiometric & Registration Perturbations
Published models demonstrate high test performance on clean, identically preprocessed benchmark datasets. However, literature provides limited comparative evidence regarding how classical vs. learned models degrade when subjected to systematic, controlled confounders:
- Global illumination scaling ($\Delta \text{luminance}$).
- Optical blur / spatial defocus.
- Sub-pixel and pixel-level geometric misregistration jitter.
Existing studies rarely measure the False Alarm Amplification Factor under isolated synthetic stress conditions.

### Gap 3: Uncalibrated Threshold Selection and Severe Class Imbalance Leakage
In classical change detection literature, researchers frequently report "optimal" F1-scores by evaluating hundreds of threshold candidates directly on the test set, creating severe data leakage `[INFERENCE]`. A rigorous, leakage-safe threshold selection protocol—where thresholds are either unsupervised (Otsu) or calibrated strictly on the validation set and then frozen for test evaluation—is rarely formalized or compared.

### Gap 4: Domain Shift vs. Computational Cost in Zero-Shot Satellite TIR
While recent domain-adapted foundation models (e.g., RemoteCLIP) improve retrieval over OpenAI CLIP, they introduce significant parameter and runtime overhead. The baseline capability of standard zero-shot CLIP under systematic prompt templates on RSICD, compared against classical text retrieval (BM25 / inverted index), has not been established as a benchmark specifically for low-resource CPU environments.

---

## 12. Novelty and Classification of Proposed Project Ideas

To prevent unsupported claims, every candidate method and concept investigated in this project is classified according to our epistemic taxonomy:

| Project Concept / Idea | Methodological Classification | Literature Evidence & Justification |
| :--- | :--- | :--- |
| **A1: Keyword / Inverted Index Retrieval** | **ESTABLISHED** | Classical information retrieval (Salton, BM25); standard baseline. |
| **A2: Zero-Shot CLIP Semantic Retrieval** | **ESTABLISHED / ADAPTATION** | Radford et al. (2021); applying standard CLIP to RSICD is an adaptation to remote sensing. |
| **A3: Prompt Ensembling for Satellite TIR** | **ADAPTATION** | Domain-specific prompt templates applied to satellite retrieval; established in NLP/CV. |
| **B1: Absolute Pixel Differencing** | **ESTABLISHED** | Singh (1989); fundamental image processing baseline. |
| **B2: SSIM Dissimilarity Change Detection** | **ESTABLISHED** | Wang et al. (2004); Bovolo & Bruzzone (2007). |
| **B3: Change Vector Analysis (CVA)** | **ESTABLISHED** | Malila (1980); standard multispectral difference technique. |
| **B4: Fully Convolutional Siamese CNN (FC-Siam-diff)** | **ESTABLISHED** | Daudt, Le Saux, & Boulch (2018); established lightweight deep baseline. |
| **B5: Quality-Aware Diagnostic Gating for False Alarms** | **ADAPTATION & COMBINATION** | **Not fundamentally novel.** Combines radiometric quality metrics, structural consistency, and learned change heatmaps to gate spurious detections. |
| **Controlled Perturbation Stress Testing** | **EXPERIMENTAL VARIATION** | Systematic stress-testing of CD models under controlled illumination, blur, and registration jitter. |
| **CPU Pareto Frontier Profiling** | **EXPERIMENTAL VARIATION** | Rigorous quantification of accuracy vs. latency/RAM under strict CPU-only constraints. |

---

## 13. Implications for Subsequent Milestones

1. **For M4 (Semantic Retrieval Baseline & Implementation):**
   - Implement A1 (BM25 / keyword inverted index) and A2 (Pretrained CLIP ViT-B/32 zero-shot retrieval with FAISS `IndexFlatIP`).
   - Primary metric: Recall@K ($K \in \{1, 5, 10\}$) under **Caption-to-Own-Image** protocol on the 1,093 held-out test images.
   - Profile CPU latency per query and index memory footprint.

2. **For M5 (Classical Change Detection Implementation):**
   - Implement B1 (Absolute Pixel Differencing), B2 (SSIM Dissimilarity), and B3 (Change Vector Analysis).
   - Implement strict **leakage-safe thresholding**: calibrate optimal threshold $\tau^*$ on the 64 validation scenes (or apply unsupervised Otsu), freeze $\tau^*$, and evaluate on the 128 test scenes.
   - Primary metrics: F1-score and IoU on the changed class. Disregard Overall Accuracy as a discriminator.

3. **For M6 (False-Alarm Suppression & Diagnostic Evaluation):**
   - Implement the controlled perturbation test suite (illumination shifts, Gaussian blur, misregistration jitter).
   - Evaluate false-alarm amplification across classical baselines vs. learned models.
   - Benchmark the lightweight quality-gated combination (B5) against un-gated baselines to test Hypothesis H3.

---

## 14. Verified Academic References

1. **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I.** (2021). *Learning Transferable Visual Models From Natural Language Supervision*. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, pp. 8748–8763.
2. **Lu, X., Wang, B., Zheng, X., & Li, X.** (2018). *High-Resolution Remote Sensing Image Captioning*. IEEE Transactions on Geoscience and Remote Sensing, 56(4), pp. 2198–2207. DOI: `10.1109/TGRS.2017.2776321`.
3. **Liu, F., Chen, D., Guan, Z., Zhou, X., Zhu, J., & Shen, Q.** (2024). *RemoteCLIP: A Vision Language Foundation Model for Remote Sensing*. IEEE Transactions on Geoscience and Remote Sensing, 62, Article 4402616. DOI: `10.1109/TGRS.2024.3370047`.
4. **Chen, H., & Shi, Z.** (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing, 12(10), Article 1662. DOI: `10.3390/rs12101662`.
5. **Daudt, R. C., Le Saux, B., & Boulch, A.** (2018). *Fully Convolutional Siamese Networks for Change Detection*. Proceedings of the 25th IEEE International Conference on Image Processing (ICIP), pp. 4063–4067. DOI: `10.1109/ICIP.2018.8451652`.
6. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer LNCS 9351, pp. 234–241. DOI: `10.1007/978-3-319-24574-4_28`.
7. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE Transactions on Image Processing, 13(4), pp. 600–612. DOI: `10.1109/TIP.2003.819861`.
8. **Malila, W. A.** (1980). *Change Vector Analysis: An Approach for Detecting Forest Changes with Landsat*. Proceedings of the LARS Symposia, Laboratory for Applications of Remote Sensing, Purdue University, Paper 385, pp. 326–335.
9. **Otsu, N.** (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-9(1), pp. 62–66. DOI: `10.1109/TSMC.1979.4310076`.
10. **Ji, S., Shen, S., Lu, M., & Zhang, Y.** (2019). *Building Instance Change Detection from Large-Scale Aerial Images using Convolutional Neural Networks and Simulated Samples*. Remote Sensing, 11(11), Article 1343. DOI: `10.3390/rs11111343`.
11. **Ji, S., Wei, S., & Lu, M.** (2019). *Fully Convolutional Networks for Multisource Building Extraction From An Open Aerial and Satellite Imagery Dataset*. IEEE Transactions on Geoscience and Remote Sensing, 57(1), pp. 574–586. DOI: `10.1109/TGRS.2018.2858824`. [Audited Extraction Citation].
12. **Singh, A.** (1989). *Review Article: Digital Change Detection Techniques Using Remotely-Sensed Data*. International Journal of Remote Sensing, 10(6), pp. 989–1003. DOI: `10.1080/01431168908903939`.
13. **Bovolo, F., & Bruzzone, L.** (2007). *A Theoretical Framework for Unsupervised Change Detection Based on Change Vector Analysis in the Polar Domain*. IEEE Transactions on Geoscience and Remote Sensing, 45(1), pp. 218–236. DOI: `10.1109/TGRS.2006.885408`.
14. **Canty, M. J., & Nielsen, A. A.** (2008). *Automatic Radiometric Normalization of Multispectral Imagery with the Iteratively Reweighted MAD Transformation*. Remote Sensing of Environment, 112(3), pp. 1025–1036. DOI: `10.1016/j.rse.2007.07.013`.
15. **Kendall, A., & Gal, Y.** (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* Advances in Neural Information Processing Systems (NeurIPS 2017), Vol. 30, pp. 5574–5584.
