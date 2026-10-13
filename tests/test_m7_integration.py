"""Integration tests for Milestone M7 FC-Siam-diff trained model, evaluation artifacts, and UI integration."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.change_detection.fc_siam_diff import (
    FCSiamDiff,
    FCSiamDiffDetector,
    load_checkpoint,
)
from src.data.patch_extractor import PatchExtractor
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from utils.config_loader import get_project_root


ROOT = get_project_root()
CKPT_PATH = ROOT / "experiments" / "checkpoints" / "m7" / "fc_siam_diff_best.pt"
RESULTS_DIR = ROOT / "experiments" / "results" / "m7"


class TestM7TrainedModelContract:
    """Verifies that the trained M7 model satisfies all architectural constraints and serialization invariants."""

    def test_checkpoint_exists_and_loads(self):
        assert CKPT_PATH.exists(), f"M7 checkpoint missing at {CKPT_PATH}"
        model, metadata = load_checkpoint(CKPT_PATH, device="cpu")
        assert isinstance(model, FCSiamDiff)
        assert metadata.get("best_epoch") == 5 or metadata.get("epoch") == 5
        assert metadata["frozen_threshold"] == 0.55
        assert metadata["best_val_f1"] > 0.60

    def test_parameter_count_within_ceiling(self):
        model, _ = load_checkpoint(CKPT_PATH, device="cpu")
        param_count = model.count_trainable_parameters()
        assert param_count == 1_471_409
        assert param_count <= 2_000_000

    def test_forward_pass_finiteness(self):
        model, _ = load_checkpoint(CKPT_PATH, device="cpu")
        t1 = torch.rand(1, 3, 256, 256)
        t2 = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out = model(t1, t2)
        assert out.shape == (1, 1, 256, 256)
        assert torch.isfinite(out).all()


class TestM7InferenceAdapter:
    """Verifies that FCSiamDiffDetector executes live bi-temporal inference on CPU with patch extraction and mosaic reconstruction."""

    def test_patch_to_mosaic_inference(self):
        model, metadata = load_checkpoint(CKPT_PATH, device="cpu")
        detector = FCSiamDiffDetector(model=model, device="cpu", name="M7_FC_Siam_diff")
        extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")

        img1 = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        gt = np.zeros((1024, 1024), dtype=np.uint8)

        patches = extractor.extract_from_pair(img1, img2, gt, parent_stem="test_synthetic")
        assert len(patches) == 16

        diff_patches = []
        for p1, p2, _, p_meta in patches:
            prob = detector.compute_difference_map(p1, p2)
            assert prob.shape == (256, 256)
            assert 0.0 <= prob.min() and prob.max() <= 1.0
            diff_patches.append((prob, p_meta))

        full_prob = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
        assert full_prob.shape == (1024, 1024)
        assert 0.0 <= full_prob.min() and full_prob.max() <= 1.0

        tau = float(metadata["frozen_threshold"])
        pred_mask = (full_prob >= tau).astype(np.uint8)
        assert pred_mask.shape == (1024, 1024)
        assert set(np.unique(pred_mask)).issubset({0, 1})


class TestM7ArtifactIntegrity:
    """Verifies existence, schema correctness, and consistency of all M7 evaluation artifacts."""

    def test_summary_results_file(self):
        summary_path = RESULTS_DIR / "summary.json"
        assert summary_path.exists()
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        record = data[0]
        assert record["method"] == "M7_FC_Siam_diff"
        assert record["threshold_value"] == 0.55
        assert record["f1"] > 0.60
        assert record["iou"] > 0.40
        assert record["total_pixels"] == 134_217_728
        assert record["gt_changed_pixels"] == 6_837_404

    def test_comparison_preserves_m5_historical_results(self):
        comp_path = RESULTS_DIR / "comparison_m5_m7.json"
        assert comp_path.exists()
        with open(comp_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == 4

        methods = {r["method"]: r for r in records}
        assert "B1_Pixel_Diff" in methods
        assert "B2_SSIM" in methods
        assert "B3_CVA" in methods
        assert "M7_FC_Siam_diff" in methods

        # Strict check that M5 historical results match exactly
        assert abs(methods["B1_Pixel_Diff"]["f1"] - 0.1299) < 1e-3
        assert abs(methods["B2_SSIM"]["f1"] - 0.1246) < 1e-3
        assert abs(methods["B3_CVA"]["f1"] - 0.1316) < 1e-3

    def test_thresholds_metadata(self):
        t_path = RESULTS_DIR / "thresholds.json"
        assert t_path.exists()
        with open(t_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        assert t_data["best_threshold"] == 0.55
        assert len(t_data["curve"]) == 19


class TestM7StreamlitIntegration:
    """Verifies that the Streamlit page loads and interacts with the trained M7 detector."""

    def test_page_renders_with_m7_detector(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "pages" / "change_detection.py"), default_timeout=30).run()
        assert len(at.exception) == 0, f"Exceptions in change_detection.py: {at.exception}"
        assert len(at.error) == 0, f"Rendered errors in change_detection.py: {[e.value for e in at.error]}"
