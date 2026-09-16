"""Integration tests for Milestone 6 False-Alarm & Robustness Evaluation pipeline."""

from pathlib import Path
import numpy as np
import pytest

from src.data.levir_loader import LEVIRDataset
from src.data.patch_extractor import PatchExtractor
from src.change_detection.pixel_diff import PixelDiffDetector
from src.change_detection.ssim_detector import SSIMDetector
from src.change_detection.cva_detector import CVADetector
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix

from src.false_alarm.illumination import IlluminationShiftPerturbation
from src.false_alarm.blur import GaussianBlurPerturbation
from src.false_alarm.robustness_metrics import RobustnessMetricsCalculator


@pytest.fixture
def sample_levir_dir() -> Path:
    p = Path("tests/fixtures/sample_levir_cd")
    assert p.exists(), "Sample fixtures missing!"
    return p


def test_m6_end_to_end_pipeline(sample_levir_dir: Path):
    test_dataset = LEVIRDataset(root_dir=sample_levir_dir, split="test")
    extractor = PatchExtractor(patch_size=256, stride=256)

    detectors = {
        "B1": PixelDiffDetector(aggregation_mode="mean"),
        "B2": SSIMDetector(win_size=11, sigma=1.5),
        "B3": CVADetector(normalize=True),
    }

    frozen_tau = {"B1": 0.41, "B2": 0.90, "B3": 0.405}

    # 1. Evaluate Control
    ctrl_cms = {name: ConfusionMatrix() for name in detectors}
    for idx in range(len(test_dataset)):
        t1, t2, target, meta = test_dataset[idx]
        img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_mask = target.numpy().astype(np.uint8)

        patches = extractor.extract_from_pair(img_a, img_b, gt_mask)
        for name, det in detectors.items():
            diff_patches = [(det.compute_difference_map(pa, pb), pmeta) for pa, pb, plbl, pmeta in patches]
            full_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
            pred = (full_diff >= frozen_tau[name]).astype(np.uint8)
            ctrl_cms[name].update(pred, gt_mask)

    ctrl_metrics = {name: ChangeDetectionEvaluator.evaluate(ctrl_cms[name]) for name in detectors}

    # 2. Evaluate Perturbed (Illumination Shift Medium)
    pert = IlluminationShiftPerturbation()
    pert_cms = {name: ConfusionMatrix() for name in detectors}

    for idx in range(len(test_dataset)):
        t1, t2, target, meta = test_dataset[idx]
        img_a = (t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_b = (t2.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_mask = target.numpy().astype(np.uint8)

        p_a, p_b, _ = pert.apply(img_a, img_b, severity="medium", target_image="t2")
        patches = extractor.extract_from_pair(p_a, p_b, gt_mask)

        for name, det in detectors.items():
            diff_patches = [(det.compute_difference_map(pa, pb), pmeta) for pa, pb, plbl, pmeta in patches]
            full_diff = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
            pred = (full_diff >= frozen_tau[name]).astype(np.uint8)
            pert_cms[name].update(pred, gt_mask)

    for name in detectors:
        p_metrics = ChangeDetectionEvaluator.evaluate(pert_cms[name])
        deg = RobustnessMetricsCalculator.compute_degradation(ctrl_metrics[name], p_metrics)

        assert "relative_f1_degradation_pct" in deg
        assert "additional_false_positives" in deg
        assert not np.isnan(deg["relative_f1_degradation_pct"])
