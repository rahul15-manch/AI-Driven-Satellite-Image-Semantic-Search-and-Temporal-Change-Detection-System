"""Patch-level PyTorch Dataset for LEVIR-CD with LRU Scene Caching and Balanced Sampling.

Extracts deterministic 256x256 patches from high-resolution 1024x1024 bi-temporal scenes
while maintaining temporal synchronization and strict train/val/test partition isolation.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.levir_loader import LEVIRDataset, LEVIRPair


class LEVIRPatchDataset(Dataset):
    """PyTorch Dataset that generates 256x256 patches from LEVIR-CD scenes on the fly.

    Employs an in-memory LRU cache over full 1024x1024 scenes to minimize disk I/O
    and memory footprint, strictly respecting the <=8 GB RAM constraint.
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        patch_size: int = 256,
        cache_size: int = 32,
        active_indices: Optional[List[int]] = None,
    ):
        """Initializes LEVIRPatchDataset.

        Args:
            root_dir: Root dataset path containing train/, val/, test/ subdirectories.
            split: Dataset split ('train', 'val', or 'test').
            patch_size: Square patch spatial dimension (default 256).
            cache_size: Number of 1024x1024 scenes to cache in RAM via LRU.
            active_indices: Optional subset of patch indices to restrict iteration to.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size

        # Underlying LEVIRDataset discovers and verifies all valid pairs in this split
        self.levir_ds = LEVIRDataset(root_dir=root_dir, split=split)
        self.num_scenes = len(self.levir_ds)

        # 1024x1024 divided by 256x256 produces 4x4 = 16 patches per scene
        self.patches_per_axis = 1024 // patch_size
        self.patches_per_scene = self.patches_per_axis * self.patches_per_axis
        self.total_patches = self.num_scenes * self.patches_per_scene

        self.active_indices = active_indices

        # Setup cached scene loader
        @functools.lru_cache(maxsize=cache_size)
        def _load_scene(scene_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            t1_tensor, t2_tensor, label_tensor, _ = self.levir_ds[scene_idx]
            # Convert tensors (C, H, W) to float32 numpy arrays in [0.0, 1.0]
            img_a = t1_tensor.numpy()  # (3, 1024, 1024)
            img_b = t2_tensor.numpy()  # (3, 1024, 1024)
            mask = label_tensor.numpy().astype(np.float32)  # (1024, 1024)
            return img_a, img_b, mask

        self._load_scene = _load_scene

    def __len__(self) -> int:
        if self.active_indices is not None:
            return len(self.active_indices)
        return self.total_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx

        scene_idx = actual_idx // self.patches_per_scene
        patch_idx = actual_idx % self.patches_per_scene

        row = patch_idx // self.patches_per_axis
        col = patch_idx % self.patches_per_axis

        y0 = row * self.patch_size
        y1 = y0 + self.patch_size
        x0 = col * self.patch_size
        x1 = x0 + self.patch_size

        img_a, img_b, mask = self._load_scene(scene_idx)

        # Slice 256x256 crops
        patch_a = torch.from_numpy(img_a[:, y0:y1, x0:x1].copy())
        patch_b = torch.from_numpy(img_b[:, y0:y1, x0:x1].copy())
        patch_mask = torch.from_numpy(mask[y0:y1, x0:x1].copy()).unsqueeze(0)  # (1, 256, 256)

        return patch_a, patch_b, patch_mask

    def scan_patch_statistics(self) -> Tuple[List[int], List[int]]:
        """Scans all patches in the dataset to categorize into positive vs. negative.

        Returns:
            positive_indices: List of patch indices containing at least 1 changed pixel.
            negative_indices: List of patch indices with 0 changed pixels.
        """
        positive_indices = []
        negative_indices = []

        for s_idx in range(self.num_scenes):
            _, _, mask = self._load_scene(s_idx)
            for p_idx in range(self.patches_per_scene):
                global_idx = s_idx * self.patches_per_scene + p_idx
                row = p_idx // self.patches_per_axis
                col = p_idx % self.patches_per_axis
                y0 = row * self.patch_size
                x0 = col * self.patch_size
                patch_lbl = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
                if patch_lbl.sum() > 0:
                    positive_indices.append(global_idx)
                else:
                    negative_indices.append(global_idx)

        return positive_indices, negative_indices

    def create_balanced_subdataset(
        self,
        positive_indices: List[int],
        negative_indices: List[int],
        samples_per_epoch: int = 1024,
        seed: int = 42,
    ) -> LEVIRPatchDataset:
        """Creates a balanced sub-dataset with an equal mixture of positive and negative patches."""
        rng = np.random.RandomState(seed)
        half = samples_per_epoch // 2

        pos_sample = rng.choice(positive_indices, size=min(half, len(positive_indices)), replace=False)
        neg_sample = rng.choice(negative_indices, size=min(half, len(negative_indices)), replace=False)

        balanced_indices = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(balanced_indices)

        return LEVIRPatchDataset(
            root_dir=self.root_dir,
            split=self.split,
            patch_size=self.patch_size,
            active_indices=balanced_indices.tolist(),
        )
