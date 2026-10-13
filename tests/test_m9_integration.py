"""Integration tests for Milestone M9 BIT-CD trained model, evaluation artifacts, and UI integration."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.change_detection.bit_cd import (
    BIT_CD,
    BITChangeDetector,
    load_checkpoint,
)
from src.data.patch_extractor import PatchExtractor
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from utils.config_loader import get_project_root


ROOT = get_project_root()
CKPT_PATH = ROOT / "experiments" / "checkpoints" / "m9" / "bit_cd_best.pt"
RESULTS_DIR = ROOT / "experiments" / "results" / "m9"


class TestM9TrainedModelContract:
    """Verifies that the trained M9 model satisfies all architectural constraints and serialization invariants."""

    def test_checkpoint_exists_and_loads(self):
        assert CKPT_PATH.exists(), f"M9 checkpoint missing at {CKPT_PATH}"
        model, metadata = load_checkpoint(CKPT_PATH, device="cpu")
        assert isinstance(model, BIT_CD)
        assert metadata.get("epoch", 0) >= 1
        assert "best_val_f1" in metadata
        assert metadata["base_channels"] == 32
        assert metadata["num_tokens"] == 16
        assert metadata["num_layers"] == 2

    def test_parameter_count_within_ceiling(self):
        model, _ = load_checkpoint(CKPT_PATH, device="cpu")
        param_count = model.count_trainable_parameters()
        assert param_count == 552_097
        assert param_count <= 2_000_000

    def test_forward_pass_finiteness(self):
        model, _ = load_checkpoint(CKPT_PATH, device="cpu")
        t1 = torch.rand(1, 3, 256, 256)
        t2 = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out = model(t1, t2)
        assert out.shape == (1, 1, 256, 256)
        assert torch.isfinite(out).all()


class TestM9InferenceAdapter:
    """Verifies that BITChangeDetector executes live bi-temporal inference on CPU with patch extraction and mosaic reconstruction."""

    def test_patch_to_mosaic_inference(self):
        model, metadata = load_checkpoint(CKPT_PATH, device="cpu")
        detector = BITChangeDetector(model=model, device="cpu", name="M9_BIT_CD")
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

        tau = float(metadata.get("frozen_threshold", 0.50))
        pred_mask = (full_prob >= tau).astype(np.uint8)
        assert pred_mask.shape == (1024, 1024)
        assert set(np.unique(pred_mask)).issubset({0, 1})


class TestM9ArtifactIntegrity:
    """Verifies existence, schema correctness, and consistency of all M9 evaluation artifacts."""

    def test_historical_preservation(self):
        # M5, M7, M8 historical files must remain untouched
        m7_summary = ROOT / "experiments" / "results" / "m7" / "summary.json"
        assert m7_summary.exists()
        with open(m7_summary, "r", encoding="utf-8") as f:
            m7_data = json.load(f)[0]
        assert abs(m7_data["f1"] - 0.6034697565619987) < 1e-5
        assert m7_data["threshold_value"] == 0.55

        m8_summary = ROOT / "experiments" / "results" / "m8" / "summary.json"
        assert m8_summary.exists()
        with open(m8_summary, "r", encoding="utf-8") as f:
            m8_data = json.load(f)[0]
        assert abs(m8_data["f1"] - 0.1626041054431217) < 1e-5
        assert m8_data["threshold_value"] == 0.50

    def test_m9_summary_schema(self):
        s_path = RESULTS_DIR / "summary.json"
        assert s_path.exists()
        with open(s_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        rec = data[0]
        assert rec["method"] == "M9_BIT_CD"
        assert 0.0 < rec["threshold_value"] < 1.0
        assert 0.0 <= rec["f1"] <= 1.0
        assert 0.0 <= rec["iou"] <= 1.0
        assert rec["total_pixels"] == 134_217_728
        assert rec["total_scenes"] == 128

    def test_m9_comparison_schema(self):
        comp_path = RESULTS_DIR / "comparison_m5_m7_m8_m9.json"
        assert comp_path.exists()
        with open(comp_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == 6
        methods = {r["method"]: r for r in records}
        assert "B1_Pixel_Diff" in methods
        assert "B2_SSIM" in methods
        assert "B3_CVA" in methods
        assert "M7_FC_Siam_diff" in methods
        assert "M8_SNUNet" in methods
        assert "M9_BIT_CD" in methods

    def test_m9_thresholds_schema(self):
        t_path = RESULTS_DIR / "thresholds.json"
        assert t_path.exists()
        with open(t_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        assert 0.0 < t_data["best_threshold"] < 1.0
        assert len(t_data["curve"]) == 19

    def test_m9_per_image_count(self):
        import pandas as pd
        pi_path = RESULTS_DIR / "per_image_metrics.csv"
        assert pi_path.exists()
        df = pd.read_csv(pi_path)
        assert len(df) == 128

    def test_m9_training_history(self):
        h_path = RESULTS_DIR / "training_history.json"
        assert h_path.exists()
        with open(h_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        assert len(history) == 5


class TestM9StreamlitIntegration:
    """Verifies that the Streamlit page loads and interacts with the trained M9 detector."""

    def test_page_renders_with_m9_detector(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "pages" / "change_detection.py"), default_timeout=30).run()
        assert len(at.exception) == 0, f"Exceptions in change_detection.py: {at.exception}"
        assert len(at.error) == 0, f"Rendered errors in change_detection.py: {[e.value for e in at.error]}"
