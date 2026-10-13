"""Unit tests for M9 BIT_CD model architecture, tokenization, losses, and detector wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.change_detection.bit_cd import (
    BIT_CD,
    BITChangeDetector,
    save_checkpoint,
    load_checkpoint,
)
from src.change_detection.losses import BCEDiceLoss, SoftDiceLoss


class TestBITCDArchitecture:
    """Verifies that BIT_CD satisfies architectural and parameter constraints."""

    def test_parameter_count_ceiling(self):
        """Tests that parameter count is strictly <= 2,000,000 for local CPU execution."""
        model = BIT_CD(in_channels=3, out_channels=1, base_channels=32, num_tokens=16, num_layers=2)
        num_params = model.count_trainable_parameters()
        assert num_params <= 2_000_000, f"Expected <= 2M parameters, got {num_params:,}"
        assert num_params > 300_000, f"Expected capacity > 300K, got {num_params:,}"

    def test_forward_output_dimensions(self):
        """Tests that forward pass produces (B, 1, H, W) logits matching input spatial dimensions."""
        model = BIT_CD(in_channels=3, out_channels=1, base_channels=32, num_tokens=16, num_layers=2)
        model.eval()

        t1 = torch.randn(2, 3, 256, 256)
        t2 = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            out = model(t1, t2)

        assert out.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN values"
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_temporal_invariance(self):
        """Tests that swapping T1 and T2 produces identical feature difference logits."""
        model = BIT_CD(in_channels=3, out_channels=1, base_channels=32, num_tokens=16, num_layers=2)
        model.eval()

        t1 = torch.randn(1, 3, 128, 128)
        t2 = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            out_12 = model(t1, t2)
            out_21 = model(t2, t1)

        torch.testing.assert_close(out_12, out_21, atol=1e-5, rtol=1e-5)

    def test_gradient_flow_and_backprop(self):
        """Tests that gradients flow backward to all trainable layers."""
        model = BIT_CD(in_channels=3, out_channels=1, base_channels=32, num_tokens=16, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = BCEDiceLoss()

        t1 = torch.randn(1, 3, 64, 64)
        t2 = torch.randn(1, 3, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        optimizer.zero_grad()
        logits = model(t1, t2)
        loss = loss_fn(logits, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

        optimizer.step()


class TestBITCDDetectorWrapper:
    """Verifies that BITChangeDetector satisfies the BaseChangeDetector contract."""

    def test_compute_difference_map_contract(self):
        model = BIT_CD(base_channels=32, num_tokens=16, num_layers=2)
        detector = BITChangeDetector(model=model, device="cpu")

        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        diff_map = detector.compute_difference_map(img1, img2)
        assert isinstance(diff_map, np.ndarray)
        assert diff_map.shape == (256, 256)
        assert diff_map.dtype == np.float32
        assert 0.0 <= diff_map.min()
        assert diff_map.max() <= 1.0

    def test_predict_thresholding(self):
        model = BIT_CD(base_channels=32, num_tokens=16, num_layers=2)
        detector = BITChangeDetector(model=model, device="cpu")

        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        mask, diff_map = detector.predict(img1, img2, threshold=0.50)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        expected_mask = (diff_map >= 0.50).astype(np.uint8)
        np.testing.assert_array_equal(mask, expected_mask)


class TestCheckpointSerialization:
    """Verifies save and load checkpoint consistency."""

    def test_save_and_reload(self):
        model = BIT_CD(base_channels=32, num_tokens=16, num_layers=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_bit_cd.pt"
            meta = {"val_f1": 0.65, "frozen_threshold": 0.45}
            save_checkpoint(model, ckpt_path, metadata=meta)

            assert ckpt_path.exists()
            assert ckpt_path.stat().st_size > 500_000

            loaded_model, loaded_meta = load_checkpoint(ckpt_path, device="cpu")
            assert loaded_meta["val_f1"] == 0.65
            assert loaded_meta["frozen_threshold"] == 0.45

            model.eval()
            t1 = torch.randn(1, 3, 64, 64)
            t2 = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                orig_out = model(t1, t2)
                loaded_out = loaded_model(t1, t2)
            torch.testing.assert_close(orig_out, loaded_out)
