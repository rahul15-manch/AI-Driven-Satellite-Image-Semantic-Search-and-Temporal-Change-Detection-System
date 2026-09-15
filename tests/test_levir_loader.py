"""Integration tests for LEVIRDataset using synthetic fixtures."""

from pathlib import Path
import torch
import pytest

from src.data.levir_loader import LEVIRDataset


@pytest.fixture
def sample_levir_dir() -> Path:
    p = Path("tests/fixtures/sample_levir_cd")
    assert p.exists(), "Fixtures missing! Run 'python3 scripts/download_datasets.py --create-fixtures' first."
    return p


def test_levir_dataset_loading(sample_levir_dir: Path):
    ds = LEVIRDataset(root_dir=sample_levir_dir, split="train")
    assert len(ds) == 2

    t1, t2, target, meta = ds[0]

    assert isinstance(t1, torch.Tensor)
    assert isinstance(t2, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    assert t1.shape == (3, 1024, 1024)
    assert t2.shape == (3, 1024, 1024)
    assert target.shape == (1024, 1024)
    assert target.dtype == torch.long

    # Target mask must only contain binary values {0, 1}
    unique_vals = torch.unique(target).tolist()
    for v in unique_vals:
        assert v in {0, 1}

    assert meta["split"] == "train"
    assert "sample_0001" in meta["sample_id"] or "train_0001" in meta["sample_id"]


def test_levir_class_statistics(sample_levir_dir: Path):
    ds = LEVIRDataset(root_dir=sample_levir_dir, split="train")
    stats = ds.compute_class_statistics()

    assert stats["total_pairs"] == 2
    assert stats["total_pixels"] == 2 * 1024 * 1024
    assert stats["changed_pixels"] > 0
    assert 0.0 < stats["changed_pixel_ratio"] < 1.0
    assert pytest.approx(stats["changed_pixel_ratio"] + stats["non_changed_pixel_ratio"]) == 1.0
