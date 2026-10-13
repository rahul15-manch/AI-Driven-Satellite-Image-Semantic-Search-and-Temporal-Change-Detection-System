"""Integration tests for Milestone M10 QAT-CD calibration, evaluation artifacts, and UI integration."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.change_detection.bit_cd import load_checkpoint as load_m9_checkpoint, BITChangeDetector
from src.change_detection.qat_cd import (
    QualityAwareChangeDetector,
    load_qat_metadata,
)
from src.data.patch_extractor import PatchExtractor
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from utils.config_loader import get_project_root, load_m10_config


ROOT = get_project_root()
M9_CKPT_PATH = ROOT / "experiments" / "checkpoints" / "m9" / "bit_cd_best.pt"
M10_META_PATH = ROOT / "experiments" / "checkpoints" / "m10" / "qat_cd_metadata.json"
RESULTS_DIR = ROOT / "experiments" / "results" / "m10"


class TestM10TrainedModelContract:
    """Verifies that M10 satisfies architectural and parameter budget constraints."""

    def test_metadata_exists_and_loads(self):
        assert M10_META_PATH.exists(), f"M10 metadata missing at {M10_META_PATH}"
        meta = load_qat_metadata(M10_META_PATH)
        assert meta["architecture"] == "Quality-Aware Temporal Change Detection (QAT-CD)"
        assert "calibrated_gamma" in meta
        assert "calibrated_threshold" in meta
        assert meta["total_parameters"] == 552_097
        assert meta["quality_parameters"] == 0

    def test_parameter_count_within_ceiling(self):
        meta = load_qat_metadata(M10_META_PATH)
        assert meta["total_parameters"] <= 2_000_000

    def test_qat_detector_initialization(self):
        base_model, _ = load_m9_checkpoint(M9_CKPT_PATH, device="cpu")
        base_detector = BITChangeDetector(model=base_model, device="cpu", name="M9_BIT_CD")
        meta = load_qat_metadata(M10_META_PATH)

        qat_det = QualityAwareChangeDetector(
            base_detector=base_detector,
            gamma=meta["calibrated_gamma"],
            weights=(0.34, 0.33, 0.33),
            name="M10_QAT_CD",
        )
        assert qat_det.name == "M10_QAT_CD"
        assert qat_det.gamma == meta["calibrated_gamma"]


class TestM10InferenceAdapter:
    """Verifies that QAT-CD executes live CPU inference on 1024x1024 scenes."""

    def test_patch_to_mosaic_inference(self):
        base_model, _ = load_m9_checkpoint(M9_CKPT_PATH, device="cpu")
        base_detector = BITChangeDetector(model=base_model, device="cpu", name="M9_BIT_CD")
        meta = load_qat_metadata(M10_META_PATH)

        qat_det = QualityAwareChangeDetector(
            base_detector=base_detector,
            gamma=meta["calibrated_gamma"],
            weights=(0.34, 0.33, 0.33),
            name="M10_QAT_CD",
        )
        extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")

        img1 = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        gt = np.zeros((1024, 1024), dtype=np.uint8)

        patches = extractor.extract_from_pair(img1, img2, gt, parent_stem="test_synthetic")
        assert len(patches) == 16

        diff_patches = []
        for p1, p2, _, p_meta in patches:
            prob = base_detector.compute_difference_map(p1, p2)
            assert prob.shape == (256, 256)
            diff_patches.append((prob, p_meta))

        p0_full = extractor.reconstruct_image(diff_patches, 1024, 1024, dtype=np.float32)
        diag = qat_det.compute_quality_diagnostics(img1, img2, base_prob=p0_full)

        assert diag["gated_prob"].shape == (1024, 1024)
        assert 0.0 <= diag["gated_prob"].min() and diag["gated_prob"].max() <= 1.0

        tau = float(meta["calibrated_threshold"])
        pred_mask = (diag["gated_prob"] >= tau).astype(np.uint8)
        assert pred_mask.shape == (1024, 1024)
        assert set(np.unique(pred_mask)).issubset({0, 1})


class TestM10ArtifactIntegrity:
    """Verifies existence, schema correctness, and consistency of all M10 evaluation artifacts."""

    def test_historical_preservation(self):
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

        m9_summary = ROOT / "experiments" / "results" / "m9" / "summary.json"
        assert m9_summary.exists()
        with open(m9_summary, "r", encoding="utf-8") as f:
            m9_data = json.load(f)[0]
        assert abs(m9_data["f1"] - 0.5926811769431694) < 1e-5
        assert m9_data["threshold_value"] == 0.45

    def test_m10_summary_schema(self):
        s_path = RESULTS_DIR / "summary.json"
        assert s_path.exists()
        with open(s_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        rec = data[0]
        assert rec["method"] == "M10_QAT_CD"
        assert 0.0 <= rec["threshold_value"] <= 1.0
        assert 0.0 <= rec["f1"] <= 1.0
        assert 0.0 <= rec["iou"] <= 1.0
        assert rec["total_pixels"] == 134_217_728
        assert rec["total_scenes"] == 128

    def test_m10_thresholds_schema(self):
        t_path = RESULTS_DIR / "thresholds.json"
        assert t_path.exists()
        with open(t_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "best_gamma" in data
        assert "best_threshold" in data
        assert "grid_results" in data
        assert len(data["grid_results"]) == 49

    def test_m10_ablations_schema(self):
        ab_path = RESULTS_DIR / "ablations.json"
        assert ab_path.exists()
        with open(ab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected_ablations = {
            "QAT_CD_Full",
            "Ablation_NoQuality",
            "Ablation_RegOnly",
            "Ablation_IllumOnly",
            "Ablation_SharpOnly",
            "Ablation_NoUncertainty",
        }
        assert expected_ablations.issubset(data.keys())

    def test_m10_per_image_count(self):
        per_img = RESULTS_DIR / "per_image_metrics.csv"
        assert per_img.exists()
        df = pd.read_csv(per_img)
        assert len(df) == 128
        assert "f1" in df.columns
        assert "iou" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns

    def test_m10_comparison_schema(self):
        c_path = RESULTS_DIR / "comparison_m5_to_m10.json"
        assert c_path.exists()
        with open(c_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        methods = [d["method"] for d in data]
        assert "M7_FC_Siam_diff" in methods
        assert "M8_SNUNet" in methods
        assert "M9_BIT_CD" in methods
        assert "M10_QAT_CD" in methods


class TestM10StreamlitIntegration:
    """Verifies that Streamlit change detection page renders M10 detector without errors."""

    def test_page_renders_with_m10_detector(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "pages" / "change_detection.py"), default_timeout=30).run()
        assert len(at.exception) == 0, f"Exceptions in change_detection.py: {at.exception}"
        assert len(at.error) == 0, f"Rendered errors in change_detection.py: {[e.value for e in at.error]}"

