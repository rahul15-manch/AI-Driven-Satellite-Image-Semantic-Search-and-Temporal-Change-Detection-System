"""Integration tests for Milestone 5 Change Detection Pipeline.

Verifies complete execution chain:
LEVIR loader -> PatchExtractor (256x256) -> B1 / B2 / B3 Detectors
-> Validation Threshold Selection -> Frozen Threshold Evaluation
-> 1024x1024 Full Image Reconstruction -> Metric Evaluation.
"""

from pathlib import Path
import numpy as np
import pytest

from src.data.levir_loader import LEVIRDataset
from src.data.patch_extractor import PatchExtractor
from src.change_detection.pixel_diff import PixelDiffDetector
from src.change_detection.ssim_detector import SSIMDetector
from src.change_detection.cva_detector import CVADetector
from src.change_detection.thresholding import ValidationThresholdOptimizer
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix


@pytest.fixture
def sample_levir_dir() -> Path:
    p = Path("tests/fixtures/sample_levir_cd")
    assert p.exists(), "Sample fixtures missing!"
    return p


def test_m5_end_to_end_pipeline(sample_levir_dir: Path):
    extractor = PatchExtractor(patch_size=256, stride=256)

    val_dataset = LEVIRDataset(root_dir=sample_levir_dir, split="val")
    test_dataset = LEVIRDataset(root_dir=sample_levir_dir, split="test")

    detectors = {
        "B1": PixelDiffDetector(aggregation_mode="mean"),
        "B2": SSIMDetector(win_size=11, sigma=1.5, channel_mode="grayscale"),
        "B3": CVADetector(normalize=True),
    }

    # Step 1: Validation calibration
    val_opts = {name: ValidationThresholdOptimizer(min_threshold=0.01, max_threshold=0.99, num_candidates=50) for name in detectors}

    for idx in range(len(val_dataset)):
        t1, t2, target, meta = val_dataset[idx]
        img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_mask = target.numpy().astype(np.uint8)

        patches = extractor.extract_from_pair(img_a, img_b, gt_mask)
        for name, det in detectors.items():
            diff_patches = [(det.compute_difference_map(pa, pb), pmeta) for pa, pb, plbl, pmeta in patches]
            full_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
            val_opts[name].update(full_diff, gt_mask)

    frozen_thresholds = {name: val_opts[name].optimize().best_threshold for name in detectors}

    # Step 2: Frozen evaluation on test split
    for name, det in detectors.items():
        tau = frozen_thresholds[name]
        assert 0.0 < tau < 1.0

        cm = ConfusionMatrix()
        for idx in range(len(test_dataset)):
            t1, t2, target, meta = test_dataset[idx]
            img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            gt_mask = target.numpy().astype(np.uint8)

            patches = extractor.extract_from_pair(img_a, img_b, gt_mask)
            diff_patches = [(det.compute_difference_map(pa, pb), pmeta) for pa, pb, plbl, pmeta in patches]
            full_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)

            pred = (full_diff >= tau).astype(np.uint8)
            cm.update(pred, gt_mask)

        metrics = ChangeDetectionEvaluator.evaluate(cm)

        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["iou"] <= 1.0
        assert metrics["total_pixels"] == len(test_dataset) * 1024 * 1024
        assert not np.isnan(metrics["f1"])
