"""Fully Convolutional Siamese Difference Network (FC-Siam-diff) for Change Detection.

Implements the classic lightweight Siamese change-detection architecture proposed by
Daudt, Le Saux, & Boulch (IEEE ICIP 2018):
- Shared twin convolutional encoders for temporal invariance.
- Multi-scale absolute feature differencing at each skip resolution.
- U-Net style convolutional decoder producing pixel-level change probabilities.
- Total parameters <= 2M, strictly optimized for CPU execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.change_detection.base import BaseChangeDetector


class ConvBlock(nn.Module):
    """Dual convolutional block with Batch Normalization and ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FCSiamDiff(nn.Module):
    """Fully Convolutional Siamese Difference Network (FC-Siam-diff).

    Paper Reference:
        Daudt, R. C., Le Saux, B., & Boulch, A. (2018).
        Fully convolutional siamese networks for change detection.
        IEEE International Conference on Image Processing (ICIP), pp. 4063-4067.

    Architecture:
        - Twin weight-sharing encoder: 4 resolution stages (16, 32, 64, 128 channels)
        - Bottleneck: 128 channels at H/16 x W/16
        - Feature differencing: D^(l) = |F_1^(l) - F_2^(l)| at all skip levels
        - Decoder: 4 upsampling stages with skip difference concatenations
        - Output: Single channel logit map at native resolution (H, W)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # --- Shared Twin Encoder ---
        self.enc1 = ConvBlock(in_channels, base_channels, num_convs=2)               # Stage 1: 16 ch
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2, num_convs=2)         # Stage 2: 32 ch
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, num_convs=3)     # Stage 3: 64 ch
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, num_convs=3)     # Stage 4: 128 ch
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8, num_convs=2)  # 128 ch

        # --- Decoder with Skip Concatenation of Feature Differences ---
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 8 + base_channels * 8, base_channels * 8, num_convs=2)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4 + base_channels * 4, base_channels * 4, num_convs=2)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2 + base_channels * 2, base_channels * 2, num_convs=2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(base_channels + base_channels, base_channels, num_convs=2),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward_encoder(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes a single temporal image through the shared encoder hierarchy.

        Returns:
            Tuple of feature maps (e1, e2, e3, e4, bottleneck).
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        return e1, e2, e3, e4, b

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Executes the forward Siamese differencing pass.

        Args:
            t1: FloatTensor of shape (B, C, H, W) for pre-change images.
            t2: FloatTensor of shape (B, C, H, W) for post-change images.

        Returns:
            FloatTensor of shape (B, 1, H, W) containing unnormalized logits.
        """
        # Shared feature extraction for both temporal branches
        e1_1, e1_2, e1_3, e1_4, b1 = self.forward_encoder(t1)
        e2_1, e2_2, e2_3, e2_4, b2 = self.forward_encoder(t2)

        # Multi-scale absolute feature differencing
        d_skip1 = torch.abs(e1_1 - e2_1)
        d_skip2 = torch.abs(e1_2 - e2_2)
        d_skip3 = torch.abs(e1_3 - e2_3)
        d_skip4 = torch.abs(e1_4 - e2_4)
        d_bottleneck = torch.abs(b1 - b2)

        # Decoder with skip-difference concatenation
        x = self.up4(d_bottleneck)
        x = self.dec4(torch.cat([x, d_skip4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, d_skip3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, d_skip2], dim=1))

        x = self.up1(x)
        logits = self.dec1(torch.cat([x, d_skip1], dim=1))

        return logits

    def count_trainable_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Returns parameter size in megabytes (float32)."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_bytes + buffer_bytes) / (1024 * 1024)


class FCSiamDiffDetector(BaseChangeDetector):
    """Detector wrapper around FCSiamDiff adhering to BaseChangeDetector interface.

    Allows direct integration with PatchExtractor and the M5/M6 evaluation suite.
    """

    def __init__(
        self,
        model: FCSiamDiff,
        device: str = "cpu",
        name: str = "B4_FC_Siam_diff",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, params=params)
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Computes continuous change probability map in [0.0, 1.0] from image pair.

        Args:
            img1: Pre-change optical image (H, W, 3) or (H, W), uint8 [0, 255] or float [0, 1].
            img2: Post-change optical image (H, W, 3) or (H, W), uint8 [0, 255] or float [0, 1].

        Returns:
            2D numpy array of shape (H, W), float32 with change probabilities in [0.0, 1.0].
        """
        arr1 = self._ensure_float_rgb(img1)
        arr2 = self._ensure_float_rgb(img2)

        # Convert to FloatTensor (1, 3, H, W)
        t1 = torch.from_numpy(arr1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(t1, t2)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

        return probs


def save_checkpoint(
    model: FCSiamDiff,
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves model state dict and associated metadata."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_kwargs": {
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "base_channels": model.base_channels,
        },
        "metadata": metadata or {},
    }
    torch.save(payload, save_path)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> Tuple[FCSiamDiff, Dict[str, Any]]:
    """Loads FCSiamDiff model and metadata from a checkpoint file."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    kwargs = checkpoint.get("model_kwargs", {"in_channels": 3, "out_channels": 1, "base_channels": 16})
    model = FCSiamDiff(**kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint.get("metadata", {})
