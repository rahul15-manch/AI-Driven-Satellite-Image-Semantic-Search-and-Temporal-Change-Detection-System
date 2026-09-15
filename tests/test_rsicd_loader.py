"""Integration tests for RSICDDataset using isolated temporary test fixtures."""

import json
from pathlib import Path
import torch
from PIL import Image
import pytest

from src.data.rsicd_loader import RSICDDataset


@pytest.fixture
def temp_rsicd_fixture(tmp_path: Path) -> Path:
    """Creates a clean, minimal temporary RSICD fixture for testing data loader functionality."""
    fixture_dir = tmp_path / "rsicd_fixture"
    img_dir = fixture_dir / "images"
    img_dir.mkdir(parents=True)

    categories = ["airport", "forest"]
    json_images = []

    for idx, cat in enumerate(categories):
        for sub_idx in range(2):
            fname = f"{cat}_{sub_idx+1:02d}.jpg"
            img_path = img_dir / fname

            # Generate a solid colored 224x224 RGB test image
            img = Image.new("RGB", (224, 224), color=((idx + 1) * 60, (sub_idx + 1) * 80, 100))
            img.save(img_path, format="JPEG")

            split = "train" if sub_idx == 0 else "test"
            captions = [
                f"An aerial view of a {cat} scene with infrastructure.",
                f"Satellite image depicting a {cat} zone.",
            ]
            sentences = [{"raw": c, "tokens": c.lower().split()} for c in captions]

            json_images.append({
                "imgid": idx * 10 + sub_idx,
                "filename": fname,
                "split": split,
                "category": cat,
                "sentences": sentences,
            })

    rsicd_json = {
        "dataset": "RSICD-Isolated-Test-Fixture",
        "images": json_images,
    }
    with open(fixture_dir / "dataset_rsicd.json", "w", encoding="utf-8") as f:
        json.dump(rsicd_json, f, indent=2)

    return fixture_dir


def test_rsicd_dataset_loading_with_json(temp_rsicd_fixture: Path):
    json_path = temp_rsicd_fixture / "dataset_rsicd.json"
    img_dir = temp_rsicd_fixture / "images"

    ds_train = RSICDDataset(image_dir=img_dir, json_path=json_path, split="train")
    assert len(ds_train) == 2

    img_tensor, captions, category, meta = ds_train[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert len(captions) == 2
    assert category in {"airport", "forest"}
    assert meta["split"] == "train"


def test_rsicd_dataset_loading_directory_fallback(temp_rsicd_fixture: Path):
    img_dir = temp_rsicd_fixture / "images"

    # Load directly without JSON file
    ds = RSICDDataset(image_dir=img_dir, json_path=None)
    assert len(ds) == 4

    img_tensor, captions, category, meta = ds[0]
    assert img_tensor.shape == (3, 224, 224)
    assert category in {"airport", "forest"}
    assert meta["caption_count"] == 0


def test_rsicd_statistics_computation(temp_rsicd_fixture: Path):
    json_path = temp_rsicd_fixture / "dataset_rsicd.json"
    img_dir = temp_rsicd_fixture / "images"

    ds = RSICDDataset(image_dir=img_dir, json_path=json_path)
    assert len(ds) == 4

    stats = ds.compute_dataset_statistics()
    assert stats["total_samples"] == 4
    assert stats["total_captions"] == 8
    assert stats["avg_captions_per_sample"] == 2.0
    assert stats["unique_categories_count"] == 2
