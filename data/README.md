# Dataset Directory & Management Guide

This directory manages the storage, verification, splitting, and preprocessing of remote sensing datasets used in this research project:
1. **RSICD** (Remote Sensing Image Captioning Dataset) — for cross-modal semantic image retrieval.
2. **LEVIR-CD** (Large-scale Building Change Detection Dataset) — for bi-temporal change detection.

---

## 1. Directory Structure

```text
data/
├── raw/
│   ├── rsicd/                  # User-provided raw RSICD imagery & annotations
│   │   ├── dataset_rsicd.json  # (Optional) Karpathy-format annotation file
│   │   └── images/             # 224x224 RGB image files (~10,921 images)
│   └── levir_cd/               # User-provided raw LEVIR-CD imagery
│       ├── train/              # A/, B/, label/ (1024x1024 PNG)
│       ├── val/                # A/, B/, label/
│       └── test/               # A/, B/, label/
│
├── processed/
│   ├── rsicd/                  # Standardized/normalized tensors (if cached)
│   └── levir_cd/               # Tiled 256x256 patches (train, val, test)
│
├── metadata/                   # Machine-readable JSON/CSV sample manifests
│   ├── rsicd/
│   │   ├── rsicd_metadata.json
│   │   ├── rsicd_metadata.csv
│   │   └── validation_report.json
│   └── levir_cd/
│       ├── levir_metadata.json
│       └── levir_metadata.csv
│
├── splits/                     # Deterministic train/val/test ID manifests
│   ├── rsicd/
│   │   └── rsicd_splits.json
│   └── levir_cd/
│       └── levir_splits.json
│
└── README.md                   # This management guide
```

---

## 2. Git Exclusion Policy

To comply with research repository best practices and prevent large binary commits:
- All raw image files inside `data/raw/*` are **strictly excluded from Git** via `.gitignore`.
- All processed patch files inside `data/processed/*` are **strictly excluded from Git** via `.gitignore`.
- Only directory structure markers (`.gitkeep`), split manifests (`data/splits/*`), and lightweight metadata (`data/metadata/*`) are tracked in version control.

---

## 3. RSICD: User-Provided Dataset Workflow

> [!IMPORTANT]
> **Dataset Source & Acquisition Policy:**  
> The RSICD dataset is **supplied manually by the researcher**. The pipeline validates the supplied dataset rather than downloading it automatically.

### Step 1: Placement
Place your verified RSICD image files (and optional `dataset_rsicd.json`) into:
```text
data/raw/rsicd/images/
```
*(Or place the images directly inside `data/raw/rsicd/`).*

### Step 2: Objective Integrity Validation
Run the objective image validation tool:
```bash
python3 -m src.data.verify_rsicd
```
You can also point to a custom directory using `--data-dir`:
```bash
python3 -m src.data.verify_rsicd --data-dir /path/to/custom/rsicd/images
```
This utility:
- Performs full pixel-level decoding via Pillow to catch truncated/corrupted byte streams.
- Rejects zero-byte files, corrupted headers, and unsupported extensions.
- Checks spatial dimensions ($224 \times 224$) and channels (3-channel RGB).
- Computes SHA-256 hashes to identify exact duplicate files.
- Generates a full report in `data/metadata/rsicd/validation_report.json`.

### Step 3: Visual Sanity Check
Inspect a representative sample of supplied images and generate a visual contact sheet:
```bash
python3 -m src.data.inspect_rsicd
```
This saves a sample contact sheet grid to `data/processed/rsicd/sanity_check_grid.png` so the researcher can confirm that authentic remote sensing scenes are present rather than noise or artifacts.

### Step 4: Metadata Generation
Once validation passes, generate structured JSON and CSV metadata manifests:
```bash
python3 -c "from src.data.metadata_builder import MetadataBuilder; b = MetadataBuilder.build_for_rsicd_directory('data/raw/rsicd/images', 'data/raw/rsicd/dataset_rsicd.json'); b.save_json('data/metadata/rsicd/rsicd_metadata.json'); b.save_csv('data/metadata/rsicd/rsicd_metadata.csv'); print('Metadata generated successfully.')"
```

### Step 5: Preprocessing
The preprocessing pipeline ingests valid raw images, applies optical standardization, and outputs to `data/processed/rsicd/`.

> [!CAUTION]
> **Raw Data Protection Rule:**  
> **Never modify or overwrite files inside `data/raw/`.** Preprocessing operates strictly read-only on raw imagery and writes output artifacts to `data/processed/`.

---

## 4. LEVIR-CD Workflow
Place the official LEVIR-CD directory structure into `data/raw/levir_cd/`:
- `data/raw/levir_cd/train/` (`A/`, `B/`, `label/`)
- `data/raw/levir_cd/val/` (`A/`, `B/`, `label/`)
- `data/raw/levir_cd/test/` (`A/`, `B/`, `label/`)

Verify the split integrity:
```bash
python3 -m src.data.inspect_dataset --dataset levir --data-dir data/raw/levir_cd/train
```

---

## 5. Leakage Prevention Protocol
1. **Scene-First Partitioning:** Always split parent scene pairs before patch extraction.
2. **Atomic Temporal Pairing:** Pre-change ($T_1$) and post-change ($T_2$) scenes from the same region remain assigned to the exact same partition.
3. **Disjoint Partitions:** `SplitManager` guarantees $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, and $\text{val} \cap \text{test} = \emptyset$.
