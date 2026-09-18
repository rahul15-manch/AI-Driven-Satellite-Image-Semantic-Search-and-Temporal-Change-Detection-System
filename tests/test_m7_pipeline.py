"""Pipeline and integration tests for M7 learned change detection."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from src.change_detection.fc_siam_diff import FCSiamDiff, FCSiamDiffDetector
from src.change_detection.patch_dataset import LEVIRPatchDataset
from src.data.patch_extractor import PatchExtractor
from src.data.split_manager import SplitManager
from utils.config_loader import get_project_root

ROOT = get_project_root()
RAW_DIR = ROOT / "data" / "raw" / "levir_cd"


class TestM7DataPipeline:
    """Verifies data loading, patch extraction, and strict split partitioning."""

    @pytest.mark.skipif(not RAW_DIR.exists(), reason="LEVIR-CD raw dataset not found")
    def test_val_dataset_patch_dimensions(self):
        val_ds = LEVIRPatchDataset(root_dir=RAW_DIR, split="val", patch_size=256)
        # 64 scenes * 16 patches = 1024 patches
        assert len(val_ds) == 1024, f"Expected 1024 validation patches, got {len(val_ds)}"

        p_a, p_b, p_lbl = val_ds[0]
        assert p_a.shape == (3, 256, 256), f"Expected (3, 256, 256), got {p_a.shape}"
        assert p_b.shape == (3, 256, 256), f"Expected (3, 256, 256), got {p_b.shape}"
        assert p_lbl.shape == (1, 256, 256), f"Expected (1, 256, 256), got {p_lbl.shape}"
        assert 0.0 <= p_a.min() and p_a.max() <= 1.0
        assert set(np.unique(p_lbl.numpy())).issubset({0.0, 1.0})

    @pytest.mark.skipif(not RAW_DIR.exists(), reason="LEVIR-CD raw dataset not found")
    def test_split_disjointness_guarantee(self):
        """Verifies zero overlap between train, validation, and test sample IDs."""
        train_ds = LEVIRPatchDataset(root_dir=RAW_DIR, split="train")
        val_ds = LEVIRPatchDataset(root_dir=RAW_DIR, split="val")
        test_ds = LEVIRPatchDataset(root_dir=RAW_DIR, split="test")

        train_stems = {p.sample_id for p in train_ds.levir_ds.pairs}
        val_stems = {p.sample_id for p in val_ds.levir_ds.pairs}
        test_stems = {p.sample_id for p in test_ds.levir_ds.pairs}

        assert len(train_stems & val_stems) == 0, "Leakage detected: train and val stems overlap!"
        assert len(train_stems & test_stems) == 0, "Leakage detected: train and test stems overlap!"
        assert len(val_stems & test_stems) == 0, "Leakage detected: val and test stems overlap!"


class TestM7ReconstructionAndInference:
    """Verifies that patch-wise detector outputs reconstruct seamlessly to 1024x1024 full scenes."""

    def test_full_scene_patch_reconstruction(self):
        model = FCSiamDiff()
        detector = FCSiamDiffDetector(model=model, device="cpu")
        extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")

        # Synthetic 1024x1024 image pair
        img_a = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        img_b = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        gt_mask = np.zeros((1024, 1024), dtype=np.uint8)
        gt_mask[100:200, 100:200] = 1

        # Extract 16 patches
        patches = extractor.extract_from_pair(img_a, img_b, gt_mask, parent_stem="test_synthetic")
        assert len(patches) == 16, f"Expected 16 patches, got {len(patches)}"

        diff_patches = []
        for p_a, p_b, p_lbl, p_meta in patches:
            d_patch = detector.compute_difference_map(p_a, p_b)
            assert d_patch.shape == (256, 256)
            diff_patches.append((d_patch, p_meta))

        # Reconstruct full 1024x1024 continuous map
        full_diff = extractor.reconstruct_image(
            diff_patches, original_height=1024, original_width=1024, dtype=np.float32
        )

        assert full_diff.shape == (1024, 1024)
        assert full_diff.dtype == np.float32
        assert 0.0 <= full_diff.min() and full_diff.max() <= 1.0

        # Apply threshold to produce binary prediction
        pred_binary = (full_diff >= 0.5).astype(np.uint8)
        assert pred_binary.shape == (1024, 1024)
        assert set(np.unique(pred_binary)).issubset({0, 1})
