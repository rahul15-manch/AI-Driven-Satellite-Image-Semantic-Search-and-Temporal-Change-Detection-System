"""Loss functions for learned change detection on class-imbalanced remote sensing benchmarks.

Implements BCEDiceLoss combining pixel-level Binary Cross-Entropy with Logits and
region-level Soft Dice Loss for binary change detection on optical satellite imagery.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation and change detection.

    Optimizes region overlap between predictions and ground truth:
        L_Dice = 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Soft Dice Loss from raw logits.

        Args:
            logits: FloatTensor of shape (B, 1, H, W) or (B, H, W) with unnormalized logit values.
            targets: FloatTensor of shape matching logits with binary ground truth in {0, 1}.

        Returns:
            Scalar tensor representing the Soft Dice Loss.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Soft Dice Loss.

    Formula:
        L_total = bce_weight * L_BCE + dice_weight * L_Dice

    Rationale:
        LEVIR-CD exhibits severe class imbalance (changed pixels ~5%). BCE provides
        smooth, well-behaved point-wise gradients across the entire image space, while
        Soft Dice Loss directly penalizes boundary and spatial overlap errors on the
        sparse positive change class.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ):
        """Initializes BCEDiceLoss.

        Args:
            bce_weight: Multiplier for BCE loss component.
            dice_weight: Multiplier for Dice loss component.
            pos_weight: Optional positive class weighting for BCEWithLogitsLoss.
            smooth: Smoothing factor for Dice calculation to avoid division by zero.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes combined BCE + Dice loss.

        Args:
            logits: FloatTensor of shape (B, 1, H, W) with change prediction logits.
            targets: FloatTensor of shape (B, 1, H, W) or (B, H, W) with binary targets.

        Returns:
            Scalar tensor with total weighted loss.
        """
        if targets.dim() == 3 and logits.dim() == 4 and logits.shape[1] == 1:
            targets = targets.unsqueeze(1)

        targets = targets.float()
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice
