"""End-to-end integration test: raw dataset -> validation -> metadata -> preprocessing."""

from pathlib import Path
import json
import torch
from PIL import Image
import numpy as np
import pytest

from src.data.dataset_verifier import DatasetVerifier
from src.data.metadata_builder import MetadataBuilder
from src.data.rsicd_loader import RSICDDataset


def test_rsicd_end_to_end_pipeline(tmp_path: Path):
    """Verifies complete pipeline workflow:

    1. User places images in raw/rsicd/images
    2. DatasetVerifier validates integrity
    3. MetadataBuilder extracts structured metadata
    4. Data loader ingests and processes images into tensors
    5. Raw data remains strictly untouched
    """
    raw_dir = tmp_path / "data" / "raw" / "rsicd" / "images"
    raw_dir.mkdir(parents=True)
    meta_dir = tmp_path / "data" / "metadata" / "rsicd"
    meta_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed" / "rsicd"
    processed_dir.mkdir(parents=True)

    # Step 1: Simulate user placing valid images
    for i in range(3):
        img = Image.new("RGB", (224, 224), color=(100 + i * 20, 120, 140))
        img.save(raw_dir / f"airport_{i+1:02d}.jpg", format="JPEG")

    # Record original file modification time and hash to verify untouched raw data
    p1 = raw_dir / "airport_01.jpg"
    mtime_before = p1.stat().st_mtime
    bytes_before = p1.read_bytes()

    # Step 2: DatasetVerifier validates directory
    report = DatasetVerifier.verify_rsicd_directory(raw_dir, expected_shape=(224, 224), expected_channels=3)
    assert report.total_checked == 3
    assert report.valid_count == 3
    assert report.invalid_count == 0
    assert report.is_clean is True

    # Step 3: Metadata generation
    builder = MetadataBuilder.build_for_rsicd_directory(image_dir=raw_dir, json_path=None, base_dir=tmp_path)
    json_path = meta_dir / "rsicd_metadata.json"
    csv_path = meta_dir / "rsicd_metadata.csv"
    builder.save_json(json_path)
    builder.save_csv(csv_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert len(builder.records) == 3
    assert all(r.valid for r in builder.records)
    assert all(r.sha256 is not None for r in builder.records)

    # Step 4: Data loading & preprocessing into model-ready tensors
    ds = RSICDDataset(image_dir=raw_dir, apply_default_norm=True)
    assert len(ds) == 3

    tensor, captions, category, meta = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert category == "airport"

    # Step 5: Assert that raw files remained strictly untouched
    mtime_after = p1.stat().st_mtime
    bytes_after = p1.read_bytes()
    assert mtime_before == mtime_after
    assert bytes_before == bytes_after
