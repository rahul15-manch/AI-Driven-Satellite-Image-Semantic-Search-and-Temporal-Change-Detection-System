"""Unit tests for M7 FC-Siam-diff model architecture, losses, and detector wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.change_detection.fc_siam_diff import (
    FCSiamDiff,
    FCSiamDiffDetector,
    save_checkpoint,
    load_checkpoint,
)
from src.change_detection.losses import BCEDiceLoss, SoftDiceLoss


class TestFCSiamDiffArchitecture:
    """Verifies that the FC-Siam-diff neural network satisfies architectural and parameter constraints."""

    def test_parameter_count_ceiling(self):
        """Tests that parameter count is strictly <= 2,000,000 (literature specification ~1.35M-1.5M)."""
        model = FCSiamDiff(in_channels=3, out_channels=1, base_channels=16)
        num_params = model.count_trainable_parameters()
        assert num_params <= 2_000_000, f"Expected <= 2M parameters, got {num_params:,}"
        assert num_params > 1_000_000, f"Expected reasonable capacity > 1M, got {num_params:,}"

    def test_forward_output_dimensions(self):
        """Tests that forward pass produces (B, 1, H, W) logits matching input spatial dimensions."""
        model = FCSiamDiff(in_channels=3, out_channels=1, base_channels=16)
        model.eval()

        t1 = torch.randn(2, 3, 256, 256)
        t2 = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            out = model(t1, t2)

        assert out.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_gradient_flow_and_backprop(self):
        """Tests that gradients flow backward to all trainable layers."""
        model = FCSiamDiff(in_channels=3, out_channels=1, base_channels=16)
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


class TestLossFunctions:
    """Verifies mathematical correctness of BCEDiceLoss and SoftDiceLoss."""

    def test_soft_dice_bounds(self):
        loss_fn = SoftDiceLoss(smooth=1.0)

        # Perfect prediction (logits high where target is 1, low where target is 0)
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 10:20, 10:20] = 1.0
        perfect_logits = torch.where(target == 1.0, torch.tensor(10.0), torch.tensor(-10.0))

        loss_perfect = loss_fn(perfect_logits, target)
        assert 0.0 <= loss_perfect.item() < 0.05, f"Perfect prediction should have loss near 0, got {loss_perfect.item()}"

        # Complete mismatch (inverse prediction)
        inverted_logits = torch.where(target == 1.0, torch.tensor(-10.0), torch.tensor(10.0))
        loss_inverted = loss_fn(inverted_logits, target)
        assert loss_inverted.item() > 0.8, f"Inverted prediction should have high loss, got {loss_inverted.item()}"

    def test_bce_dice_combined(self):
        loss_fn = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_fn(logits, targets)
        assert loss.ndim == 0, "Loss must be a scalar"
        assert loss.item() > 0.0, "Loss must be strictly positive"


class TestFCSiamDiffDetectorWrapper:
    """Verifies that FCSiamDiffDetector satisfies the BaseChangeDetector contract."""

    def test_compute_difference_map_contract(self):
        model = FCSiamDiff()
        detector = FCSiamDiffDetector(model=model, device="cpu")

        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        diff_map = detector.compute_difference_map(img1, img2)
        assert isinstance(diff_map, np.ndarray)
        assert diff_map.shape == (256, 256)
        assert diff_map.dtype == np.float32
        assert 0.0 <= diff_map.min()
        assert diff_map.max() <= 1.0

    def test_predict_thresholding(self):
        model = FCSiamDiff()
        detector = FCSiamDiffDetector(model=model, device="cpu")

        img1 = np.ones((128, 128, 3), dtype=np.uint8) * 100
        img2 = np.ones((128, 128, 3), dtype=np.uint8) * 150

        binary_mask, diff_map = detector.predict(img1, img2, threshold=0.5)
        assert binary_mask.shape == (128, 128)
        assert binary_mask.dtype == np.uint8
        assert set(np.unique(binary_mask)).issubset({0, 1})


class TestCheckpointSerialization:
    """Verifies that model state can be saved and loaded with bitwise parity."""

    def test_save_and_reload(self):
        model = FCSiamDiff()
        model.eval()

        t1 = torch.randn(1, 3, 64, 64)
        t2 = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            expected_out = model(t1, t2)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_checkpoint(model, tmp_path, metadata={"best_epoch": 3, "val_f1": 0.54})
            reloaded_model, meta = load_checkpoint(tmp_path, device="cpu")

            assert meta.get("best_epoch") == 3
            assert meta.get("val_f1") == 0.54

            with torch.no_grad():
                reloaded_out = reloaded_model(t1, t2)

            torch.testing.assert_close(expected_out, reloaded_out)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
