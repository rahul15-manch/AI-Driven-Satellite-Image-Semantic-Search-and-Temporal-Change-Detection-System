"""SNUNet-CD: A Densely Connected Siamese Network for Change Detection.

Paper Reference:
    Fang, S., Li, K., Shao, J., & Li, Z. (2021).
    SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images.
    IEEE Geoscience and Remote Sensing Letters (GRSL), 19, 1-5.
    DOI: 10.1109/LGRS.2021.3056416

Key Architectural Properties:
    - Shared twin convolutional encoder for temporal invariance across T1 and T2.
    - Multi-scale absolute feature differencing across all hierarchy levels.
    - Dense nested skip pathways (UNet++ style) preserving shallow and deep representations.
    - Ensemble Channel Attention Module (ECAM) dynamically weighting multi-scale outputs.
    - Parameter budget <= 2M parameters for efficient CPU-only training and inference.
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


class UpBlock(nn.Module):
    """Upsampling layer using ConvTranspose2d."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class ECAM(nn.Module):
    """Ensemble Channel Attention Module (ECAM).

    Gathers high-resolution outputs from multiple dense decoder stages,
    computes global cross-channel interaction via squeeze-and-excitation MLP,
    and weights multi-scale feature maps adaptively.
    """

    def __init__(self, total_channels: int, reduction_ratio: int = 4):
        super().__init__()
        mid_channels = max(total_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(total_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, total_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        pooled = self.avg_pool(x).view(b, c)
        attn = self.mlp(pooled).view(b, c, 1, 1)
        return x * attn


class SNUNet(nn.Module):
    """SNUNet-CD: Densely Connected Siamese Network for Change Detection."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        c = base_channels
        c1, c2, c3, c4 = c, c * 2, c * 4, c * 8

        # --- Shared Twin Encoder Backbone (j=0) ---
        self.conv0_0 = ConvBlock(in_channels, c1)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_0 = ConvBlock(c1, c2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_0 = ConvBlock(c2, c3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_0 = ConvBlock(c3, c4)

        # --- Dense Nested Decoder (j=1) ---
        self.up1_0 = UpBlock(c2, c1)
        self.conv0_1 = ConvBlock(c1 + c1, c1)

        self.up2_0 = UpBlock(c3, c2)
        self.conv1_1 = ConvBlock(c2 + c2, c2)

        self.up3_0 = UpBlock(c4, c3)
        self.conv2_1 = ConvBlock(c3 + c3, c3)

        # --- Dense Nested Decoder (j=2) ---
        self.up1_1 = UpBlock(c2, c1)
        self.conv0_2 = ConvBlock(c1 * 2 + c1, c1)

        self.up2_1 = UpBlock(c3, c2)
        self.conv1_2 = ConvBlock(c2 * 2 + c2, c2)

        # --- Dense Nested Decoder (j=3) ---
        self.up1_2 = UpBlock(c2, c1)
        self.conv0_3 = ConvBlock(c1 * 3 + c1, c1)

        # --- Multi-Scale Fusion Head with ECAM ---
        # Combines dense high-resolution nodes (x0_1, x0_2, x0_3) -> 3 * c1 channels
        fusion_channels = c1 * 3
        self.ecam = ECAM(total_channels=fusion_channels)
        self.head = nn.Sequential(
            ConvBlock(fusion_channels, c1),
            nn.Conv2d(c1, out_channels, kernel_size=1),
        )

    def forward_encoder(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes a single temporal branch through the shared encoder hierarchy."""
        x0 = self.conv0_0(x)
        x1 = self.conv1_0(self.pool0(x0))
        x2 = self.conv2_0(self.pool1(x1))
        x3 = self.conv3_0(self.pool2(x2))
        return x0, x1, x2, x3

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Executes forward pass with multi-scale Siamese differencing and dense routing.

        Args:
            t1: Pre-change optical tensor (B, C, H, W).
            t2: Post-change optical tensor (B, C, H, W).

        Returns:
            Unnormalized change logits tensor (B, 1, H, W).
        """
        # Shared feature extraction for T1 and T2
        t1_x0, t1_x1, t1_x2, t1_x3 = self.forward_encoder(t1)
        t2_x0, t2_x1, t2_x2, t2_x3 = self.forward_encoder(t2)

        # Multi-scale absolute feature differences at node column j=0
        d0_0 = torch.abs(t1_x0 - t2_x0)
        d1_0 = torch.abs(t1_x1 - t2_x1)
        d2_0 = torch.abs(t1_x2 - t2_x2)
        d3_0 = torch.abs(t1_x3 - t2_x3)

        # Column j=1
        x0_1 = self.conv0_1(torch.cat([d0_0, self.up1_0(d1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([d1_0, self.up2_0(d2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([d2_0, self.up3_0(d3_0)], dim=1))

        # Column j=2
        x0_2 = self.conv0_2(torch.cat([d0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([d1_0, x1_1, self.up2_1(x2_1)], dim=1))

        # Column j=3
        x0_3 = self.conv0_3(torch.cat([d0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        # Multi-scale aggregation across dense outputs
        multi_scale_features = torch.cat([x0_1, x0_2, x0_3], dim=1)
        attended_features = self.ecam(multi_scale_features)
        logits = self.head(attended_features)

        return logits

    def count_trainable_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Returns parameter size in megabytes (float32)."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_bytes + buffer_bytes) / (1024 * 1024)


class SNUNetDetector(BaseChangeDetector):
    """Inference adapter for SNUNet adhering to BaseChangeDetector contract."""

    def __init__(
        self,
        model: SNUNet,
        device: str = "cpu",
        name: str = "M8_SNUNet",
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
            img1: Pre-change optical image (H, W, 3), uint8 [0, 255] or float [0, 1].
            img2: Post-change optical image (H, W, 3), uint8 [0, 255] or float [0, 1].

        Returns:
            2D float32 numpy array (H, W) with change probabilities in [0.0, 1.0].
        """
        arr1 = self._ensure_float_rgb(img1)
        arr2 = self._ensure_float_rgb(img2)

        t1 = torch.from_numpy(arr1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(t1, t2)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

        return probs


def save_checkpoint(
    model: SNUNet,
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves SNUNet model state dict and associated metadata."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
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
) -> Tuple[SNUNet, Dict[str, Any]]:
    """Loads SNUNet model and metadata from a checkpoint file."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    kwargs = checkpoint.get("model_kwargs", {"in_channels": 3, "out_channels": 1, "base_channels": 16})
    model = SNUNet(**kwargs)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint.get("metadata", {})
