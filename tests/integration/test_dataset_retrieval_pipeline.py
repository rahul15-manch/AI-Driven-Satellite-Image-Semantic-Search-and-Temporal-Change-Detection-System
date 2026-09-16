"""Integration Test: Dataset Acquisition, Verification, and Preprocessing (M2 Integration).

Covers:
- INT-M2-01: RSICD Dataset Availability and Split Loading
- INT-M2-02: RSICD Image Validation (RGB, 224x224, Integrity)
- INT-M2-03: Split Disjointness and Exact Sums
- INT-M2-04: Test Gallery Count (Exactly 1,093 images)
- INT-M2-05: Caption Count (Exactly 5 captions per image -> 5,465 total)
- INT-M2-06: Caption-to-Image 1-to-1 Mapping Integrity
- INT-M2-07: Dataset Reproducibility across Repeated Loads
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

from src.data.rsicd_loader import RSICDDataset


@pytest.mark.integration
@pytest.mark.dataset
def test_int_m2_01_rsicd_dataset_availability():
    """INT-M2-01: Verify RSICD dataset files exist and test split loads."""
    raw_dir = Path("data/raw/rsicd")
    assert raw_dir.exists(), "data/raw/rsicd directory must exist."
    json_path = raw_dir / "dataset_rsicd.json"
    assert json_path.exists(), "dataset_rsicd.json must exist."
    img_dir = raw_dir / "images"
    assert img_dir.exists(), "images directory must exist."

    dataset = RSICDDataset(image_dir=img_dir, json_path=json_path, split="test")
    assert len(dataset) > 0, "Test split must not be empty."


@pytest.mark.integration
@pytest.mark.dataset
def test_int_m2_02_rsicd_image_validation():
    """INT-M2-02: Verify test images are valid RGB, 224x224, with uncorrupted headers."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    # Validate sample of 50 images spread evenly across the test gallery
    indices_to_test = list(range(0, len(dataset), len(dataset) // 50))
    for idx in indices_to_test:
        sample = dataset.samples[idx]
        img_path = sample.image_path
        assert img_path.exists(), f"Image file {img_path} must exist."
        with Image.open(img_path) as img:
            assert img.mode == "RGB", f"Image {img_path.name} mode must be RGB, got {img.mode}."
            assert img.size == (224, 224), f"Image {img_path.name} size must be 224x224, got {img.size}."
            img.verify()


@pytest.mark.integration
def test_int_m2_03_split_integrity():
    """INT-M2-03: Verify train, validation, and test splits are mutually disjoint and sum to 10,921."""
    splits_path = Path("data/splits/rsicd/rsicd_splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train = set(splits.get("train_ids", splits.get("train", [])))
    val = set(splits.get("val_ids", splits.get("val", [])))
    test = set(splits.get("test_ids", splits.get("test", [])))

    # Disjointness
    assert train.isdisjoint(val), "Train and Validation splits must be disjoint."
    assert train.isdisjoint(test), "Train and Test splits must be disjoint."
    assert val.isdisjoint(test), "Validation and Test splits must be disjoint."

    # Exact split counts from data_card.md
    assert len(train) == 8734, f"Expected 8,734 train images, found {len(train)}."
    assert len(val) == 1094, f"Expected 1,094 val images, found {len(val)}."
    assert len(test) == 1093, f"Expected 1,093 test images, found {len(test)}."
    assert len(train) + len(val) + len(test) == 10921, "Total splits must sum to 10,921."


@pytest.mark.integration
@pytest.mark.dataset
def test_int_m2_04_test_gallery_count():
    """INT-M2-04: Test gallery must contain exactly 1,093 images."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    assert len(dataset) == 1093, f"Expected exactly 1,093 test images, found {len(dataset)}."


@pytest.mark.integration
@pytest.mark.dataset
def test_int_m2_05_caption_count():
    """INT-M2-05: Exactly 5 captions per image -> 5,465 total test queries."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    total_captions = sum(len(sample.captions) for sample in dataset.samples)
    assert total_captions == 5465, f"Expected exactly 5,465 captions, got {total_captions}."


@pytest.mark.integration
@pytest.mark.dataset
def test_int_m2_06_caption_to_image_mapping():
    """INT-M2-06: Every test caption must map to exactly one originating target image."""
    dataset = RSICDDataset(
        image_dir="data/raw/rsicd/images",
        json_path="data/raw/rsicd/dataset_rsicd.json",
        split="test",
    )
    query_target_map = {}
    for sample in dataset.samples:
        for c_idx, caption in enumerate(sample.captions):
            q_id = f"{Path(sample.filename).stem}_cap{c_idx}"
            assert q_id not in query_target_map, f"Duplicate query ID detected: {q_id}"
            query_target_map[q_id] = sample.filename

    assert len(query_target_map) == 5465, f"Expected 5,465 query-to-image mappings, got {len(query_target_map)}."


@pytest.mark.integration
@pytest.mark.dataset
def test_int_m2_07_dataset_reproducibility():
    """INT-M2-07: Loading the same split twice must yield identical ordered image IDs and captions."""
    ds1 = RSICDDataset(image_dir="data/raw/rsicd/images", json_path="data/raw/rsicd/dataset_rsicd.json", split="test")
    ds2 = RSICDDataset(image_dir="data/raw/rsicd/images", json_path="data/raw/rsicd/dataset_rsicd.json", split="test")

    ids1 = [s.filename for s in ds1.samples]
    ids2 = [s.filename for s in ds2.samples]
    assert ids1 == ids2, "Ordered image IDs must be strictly identical across independent loads."

    caps1 = [s.captions for s in ds1.samples]
    caps2 = [s.captions for s in ds2.samples]
    assert caps1 == caps2, "Captions must be strictly identical across independent loads."
