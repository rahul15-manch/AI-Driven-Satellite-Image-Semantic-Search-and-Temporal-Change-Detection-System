"""Unit tests for MetadataBuilder."""

from pathlib import Path
import json
import pandas as pd
from PIL import Image
import pytest

from src.data.metadata_builder import MetadataBuilder, SampleMetadataRecord


def test_metadata_builder_manual_records(tmp_path: Path):
    """Tests building and saving metadata records manually."""
    builder = MetadataBuilder(dataset_name="Test-Dataset", version="1.0.0")

    rec = SampleMetadataRecord(
        sample_id="scene_001",
        source_dataset="RSICD",
        filename="scene_001.jpg",
        relative_path="images/scene_001.jpg",
        split="train",
        t1_path="/path/to/scene_001.jpg",
        width=224,
        height=224,
        channels=3,
        mode="RGB",
        category="airport",
        caption_count=1,
        captions=["An airplane is parked."],
        sha256="abc123def456",
        valid=True,
    )
    builder.add_record(rec)

    json_out = tmp_path / "metadata.json"
    csv_out = tmp_path / "metadata.csv"

    builder.save_json(json_out)
    builder.save_csv(csv_out)

    assert json_out.exists()
    assert csv_out.exists()

    with open(json_out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_name"] == "Test-Dataset"
    assert data["record_count"] == 1
    assert data["samples"][0]["sha256"] == "abc123def456"

    df = pd.read_csv(csv_out)
    assert len(df) == 1
    assert df.iloc[0]["category"] == "airport"
    assert df.iloc[0]["sha256"] == "abc123def456"


def test_metadata_builder_from_directory_with_and_without_json(tmp_path: Path):
    """Tests automated metadata extraction from a directory of images with optional annotations."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 2 valid images
    img1 = Image.new("RGB", (224, 224), (100, 150, 200))
    img2 = Image.new("RGB", (224, 224), (50, 80, 110))
    img1.save(img_dir / "airport_01.jpg")
    img2.save(img_dir / "forest_01.jpg")

    # 1. Test without JSON annotations (optional metadata absent)
    builder_no_json = MetadataBuilder.build_for_rsicd_directory(
        image_dir=img_dir,
        json_path=None,
        base_dir=tmp_path,
    )
    assert len(builder_no_json.records) == 2
    r1 = next(r for r in builder_no_json.records if r.filename == "airport_01.jpg")
    assert r1.category == "airport"
    assert r1.caption_count == 0
    assert r1.split == "unknown"
    assert r1.valid is True
    assert r1.sha256 is not None

    # 2. Test with JSON annotations present
    json_path = tmp_path / "dataset_rsicd.json"
    ann_data = {
        "images": [
            {
                "filename": "airport_01.jpg",
                "split": "train",
                "sentences": [{"raw": "An airport runway in remote sensing view."}],
            }
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ann_data, f)

    builder_with_json = MetadataBuilder.build_for_rsicd_directory(
        image_dir=img_dir,
        json_path=json_path,
        base_dir=tmp_path,
    )
    r1_json = next(r for r in builder_with_json.records if r.filename == "airport_01.jpg")
    assert r1_json.split == "train"
    assert r1_json.caption_count == 1
    assert "airport runway" in r1_json.captions[0]
