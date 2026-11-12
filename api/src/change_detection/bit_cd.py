"""Bitemporal Image Transformer (BIT-CD) for Attention-Based Change Detection.

Implements the BIT-CD architecture proposed by Chen, Lyu, and Shi (IEEE TGRS 2021):
1. Shared Siamese CNN backbone extracting compact multi-scale spatial representations.
2. Spatial Attention Tokenizer projecting spatial features into compact semantic visual tokens (L=16).
3. Transformer Contextualizer modeling intra-temporal dependencies (Self-Attention) and
   inter-temporal bitemporal interactions (Cross-Attention).
4. Spatial Token Projector reconstructing refined spatial features via attention affinity mapping.
5. High-resolution multi-scale skip-connected upsampling prediction head.

Parameter ceiling: <= 2,000,000 parameters (actual ~552K parameters).
Execution: CPU-optimized with deterministic temporal symmetry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.change_detection.base import BaseChangeDetector


class ResBlock(nn.Module):
    """Residual convolutional block with batch normalization and ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv(x), inplace=True)


class CNNBackbone(nn.Module):
    """Siamese CNN feature extractor with multi-scale skip stages."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResBlock(base_channels),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResBlock(base_channels * 2),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResBlock(base_channels * 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts hierarchical feature maps at resolutions (H, W), (H/2, W/2), (H/4, W/4)."""
        s1 = self.stage1(x)
        s2 = self.stage2(self.pool1(s1))
        s3 = self.stage3(self.pool2(s2))
        return s1, s2, s3


class SpatialAttentionTokenizer(nn.Module):
    """Bitemporal tokenizer aggregating dense spatial features into compact semantic tokens."""

    def __init__(self, in_channels: int = 64, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.conv_att = nn.Conv2d(in_channels, num_tokens, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms spatial tensor (B, C, H, W) into semantic tokens (B, L, C)."""
        b, c, h, w = x.shape
        att = self.conv_att(x)  # (B, L, H, W)
        att = F.softmax(att.view(b, self.num_tokens, h * w), dim=-1)  # (B, L, HW)
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        tokens = torch.bmm(att, x_flat)  # (B, L, C)
        return tokens


class TransformerBlock(nn.Module):
    """Transformer encoder block with Multi-Head Self-Attention (MHSA)."""

    def __init__(self, dim: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Transformer decoder block with Multi-Head Cross-Attention (MHCA)."""

    def __init__(self, dim: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        norm_q = self.norm1_q(q)
        norm_kv = self.norm1_kv(kv)
        attn_out, _ = self.cross_attn(norm_q, norm_kv, norm_kv)
        x = q + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialTokenProjector(nn.Module):
    """Projects contextualized tokens back into dense spatial feature space."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.refine = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Projects tokens (B, L, C) onto spatial grid (B, C, H, W) via affinity mapping."""
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        affinity = torch.bmm(x_flat, tokens.permute(0, 2, 1)) / (self.dim ** 0.5)  # (B, HW, L)
        affinity = F.softmax(affinity, dim=-1)
        proj = torch.bmm(affinity, tokens)  # (B, HW, C)
        proj = proj.permute(0, 2, 1).view(b, c, h, w)
        return self.refine(torch.cat([x, proj], dim=1))


class BIT_CD(nn.Module):
    """Bitemporal Image Transformer Change Detection Network (Chen et al. 2021)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        num_tokens: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        dim = base_channels * 2  # 64
        self.backbone = CNNBackbone(in_channels=in_channels, base_channels=base_channels)
        self.tokenizer = SpatialAttentionTokenizer(in_channels=dim, num_tokens=num_tokens)

        self.self_attn_layers = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=4) for _ in range(num_layers)
        ])
        # Shared cross-attention weights for temporal symmetry
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(dim=dim, num_heads=4) for _ in range(num_layers)
        ])

        self.projector = SpatialTokenProjector(dim=dim)

        # Upsampling and difference prediction head
        self.head_conv0 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head_conv1 = nn.Sequential(
            nn.Conv2d(dim + dim, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head_conv2 = nn.Sequential(
            nn.Conv2d(base_channels + base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(base_channels // 2, out_channels, kernel_size=1)

    def count_trainable_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Computes change detection logits from bitemporal image tensors.

        Args:
            t1: FloatTensor of shape (B, 3, H, W) representing pre-change imagery.
            t2: FloatTensor of shape (B, 3, H, W) representing post-change imagery.

        Returns:
            FloatTensor of shape (B, 1, H, W) containing unnormalized change logits.
        """
        # Step 1: Siamese CNN feature extraction
        s1_1, s1_2, s1_3 = self.backbone(t1)  # s1_3: (B, 64, H/4, W/4)
        s2_1, s2_2, s2_3 = self.backbone(t2)

        # Step 2: Spatial attention tokenization
        tok1 = self.tokenizer(s1_3)  # (B, L, 64)
        tok2 = self.tokenizer(s2_3)

        # Step 3: Intra-temporal self-attention contextualization
        for layer in self.self_attn_layers:
            tok1 = layer(tok1)
            tok2 = layer(tok2)

        # Step 4: Inter-temporal bitemporal cross-attention interaction
        for layer in self.cross_attn_layers:
            next_tok1 = layer(tok1, tok2)
            next_tok2 = layer(tok2, tok1)
            tok1, tok2 = next_tok1, next_tok2

        # Step 5: Spatial projection of refined tokens
        y1 = self.projector(s1_3, tok1)
        y2 = self.projector(s2_3, tok2)

        # Step 6: Symmetric bitemporal difference fusion
        diff = torch.abs(y1 - y2)
        sum_feat = y1 + y2
        feat = torch.cat([sum_feat, diff], dim=1)  # (B, 2*dim, H/4, W/4)
        h = self.head_conv0(feat)

        # Step 7: Progressive upsampling with multi-scale skip differences
        h = self.up1(h)  # (B, dim, H/2, W/2)
        skip2 = torch.abs(s1_2 - s2_2)
        h = self.head_conv1(torch.cat([h, skip2], dim=1))  # (B, base_channels, H/2, W/2)

        h = self.up2(h)  # (B, base_channels, H, W)
        skip1 = torch.abs(s1_1 - s2_1)
        h = self.head_conv2(torch.cat([h, skip1], dim=1))  # (B, base_channels // 2, H, W)

        logits = self.classifier(h)  # (B, 1, H, W)
        return logits


class BITChangeDetector(BaseChangeDetector):
    """Inference wrapper for BIT_CD adhering to BaseChangeDetector interface."""

    def __init__(
        self,
        model: BIT_CD,
        device: str = "cpu",
        name: str = "M9_BIT_CD",
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
    model: BIT_CD,
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves BIT_CD model state dict and associated metadata."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "base_channels": model.base_channels,
            "num_tokens": model.num_tokens,
            "num_layers": model.num_layers,
        },
        "metadata": metadata or {},
    }
    torch.save(payload, save_path)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> Tuple[BIT_CD, Dict[str, Any]]:
    """Loads BIT_CD model and metadata from a checkpoint file."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    kwargs = checkpoint.get("model_kwargs", {
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 32,
        "num_tokens": 16,
        "num_layers": 2,
    })
    model = BIT_CD(**kwargs)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = checkpoint.get("metadata", {})
    return model, metadata
