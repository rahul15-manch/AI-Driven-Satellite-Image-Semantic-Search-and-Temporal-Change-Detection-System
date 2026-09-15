"""Unit tests for SplitManager."""

from pathlib import Path
import pytest

from src.data.split_manager import SplitManager, SplitManifest


def test_split_generation_no_leakage():
    sample_ids = [f"sample_{i:04d}" for i in range(100)]
    manifest = SplitManager.create_custom_split(
        sample_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    assert manifest.validate_no_leakage() is True
    assert len(manifest.train_ids) == 70
    assert len(manifest.val_ids) == 15
    assert len(manifest.test_ids) == 15
    assert manifest.total_samples == 100


def test_split_determinism():
    sample_ids = [f"sample_{i:04d}" for i in range(50)]
    m1 = SplitManager.create_custom_split(sample_ids, seed=123)
    m2 = SplitManager.create_custom_split(sample_ids, seed=123)

    assert m1.train_ids == m2.train_ids
    assert m1.val_ids == m2.val_ids
    assert m1.test_ids == m2.test_ids


def test_leakage_detection():
    # Artificially introduce leakage
    leaky_manifest = SplitManifest(
        dataset_name="leaky_dataset",
        split_type="custom",
        seed=42,
        train_ids=["id_1", "id_2", "id_3"],
        val_ids=["id_3", "id_4"],  # id_3 leaked into val!
        test_ids=["id_5"],
    )

    assert leaky_manifest.validate_no_leakage() is False
    summary = leaky_manifest.get_overlap_summary()
    assert summary["train_val_overlap"] == ["id_3"]


def test_save_and_load_manifest(tmp_path: Path):
    sample_ids = [f"pair_{i}" for i in range(20)]
    manifest = SplitManager.create_custom_split(sample_ids, seed=42)

    save_path = tmp_path / "splits.json"
    SplitManager.save_split_manifest(manifest, save_path)
    assert save_path.exists()

    loaded = SplitManager.load_split_manifest(save_path)
    assert loaded.dataset_name == manifest.dataset_name
    assert loaded.train_ids == manifest.train_ids
    assert loaded.validate_no_leakage() is True
