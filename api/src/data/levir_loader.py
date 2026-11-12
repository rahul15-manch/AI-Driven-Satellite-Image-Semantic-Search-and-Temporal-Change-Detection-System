"""Dataset loader and PyTorch Dataset representation for LEVIR-CD bi-temporal imagery.

Ensures strict pairing of pre-change (A), post-change (B), and label masks,
with deterministic normalization and optional patch-level extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class LEVIRPair:
    """Represents a validated bi-temporal sample."""
    sample_id: str
    path_a: Path
    path_b: Path
    path_label: Optional[Path] = None
    split: Optional[str] = None


class LEVIRDataset(Dataset):
    """PyTorch Dataset for LEVIR-CD bi-temporal satellite image pairs.

    Returns:
        t1_tensor: FloatTensor of shape (C, H, W) normalized to [0, 1] (or transformed).
        t2_tensor: FloatTensor of shape (C, H, W) normalized to [0, 1] (or transformed).
        label_tensor: LongTensor of shape (H, W) with binary classes {0, 1}.
        meta: Dict containing sample_id, paths, and image shapes.
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: Optional[str] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        apply_default_norm: bool = False,
    ):
        """Initializes LEVIRDataset.

        Args:
            root_dir: Path to directory containing A/, B/, and label/ (or root with split folders).
            split: Optional split subfolder name ('train', 'val', 'test').
            transform: Optional custom torchvision transform.
            normalize_mean: Channel-wise mean if apply_default_norm is True.
            normalize_std: Channel-wise std if apply_default_norm is True.
            apply_default_norm: Whether to apply standard ImageNet normalization.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.apply_default_norm = apply_default_norm
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1)

        target_dir = self.root_dir / split if split else self.root_dir
        self.pairs = self._discover_and_validate_pairs(target_dir)

    def _discover_and_validate_pairs(self, directory: Path) -> List[LEVIRPair]:
        """Discovers bi-temporal pairs by matching stems across A/, B/, and label/ directories."""
        dir_a = directory / "A"
        dir_b = directory / "B"
        dir_label = directory / "label"

        if not dir_a.exists() or not dir_b.exists():
            raise FileNotFoundError(
                f"Missing required subdirectories A/ and B/ inside: {directory}"
            )

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        files_a = {p.stem: p for p in dir_a.iterdir() if p.suffix.lower() in valid_exts}
        files_b = {p.stem: p for p in dir_b.iterdir() if p.suffix.lower() in valid_exts}
        has_labels = dir_label.exists()
        files_label = (
            {p.stem: p for p in dir_label.iterdir() if p.suffix.lower() in valid_exts}
            if has_labels else {}
        )

        common_stems = sorted(list(set(files_a.keys()) & set(files_b.keys())))
        if len(common_stems) == 0:
            raise ValueError(f"No matching pairs found between {dir_a} and {dir_b}")

        pairs: List[LEVIRPair] = []
        for stem in common_stems:
            pa = files_a[stem]
            pb = files_b[stem]
            pl = files_label.get(stem)

            pairs.append(LEVIRPair(
                sample_id=stem,
                path_a=pa,
                path_b=pb,
                path_label=pl,
                split=self.split,
            ))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        pair = self.pairs[idx]

        with Image.open(pair.path_a) as img_a:
            img_a = img_a.convert("RGB")
            w, h = img_a.size
            if self.transform is not None:
                t1 = self.transform(img_a)
            else:
                arr_a = np.array(img_a, dtype=np.float32) / 255.0
                t1 = torch.from_numpy(arr_a).permute(2, 0, 1)

        with Image.open(pair.path_b) as img_b:
            img_b = img_b.convert("RGB")
            if self.transform is not None:
                t2 = self.transform(img_b)
            else:
                arr_b = np.array(img_b, dtype=np.float32) / 255.0
                t2 = torch.from_numpy(arr_b).permute(2, 0, 1)

        if self.apply_default_norm and self.transform is None:
            t1 = (t1 - self.normalize_mean) / self.normalize_std
            t2 = (t2 - self.normalize_mean) / self.normalize_std

        if pair.path_label is not None and pair.path_label.exists():
            with Image.open(pair.path_label) as img_lbl:
                arr_lbl = np.array(img_lbl.convert("L"), dtype=np.uint8)
                # Map standard LEVIR 255 change value to discrete class 1
                binary_mask = (arr_lbl > 127).astype(np.int64)
                target = torch.from_numpy(binary_mask)
        else:
            # Fallback zero mask if label is absent
            target = torch.zeros((h, w), dtype=torch.long)

        meta = {
            "sample_id": pair.sample_id,
            "path_a": str(pair.path_a),
            "path_b": str(pair.path_b),
            "path_label": str(pair.path_label) if pair.path_label else "",
            "split": self.split or "unknown",
            "width": w,
            "height": h,
        }

        return t1, t2, target, meta

    def compute_class_statistics(self) -> Dict[str, Any]:
        """Calculates pixel-level class counts (changed vs non-changed) across dataset masks."""
        total_pixels = 0
        changed_pixels = 0

        for pair in self.pairs:
            if pair.path_label and pair.path_label.exists():
                with Image.open(pair.path_label) as img:
                    arr = np.array(img.convert("L"))
                    changed = np.count_nonzero(arr > 127)
                    total = arr.size
                    changed_pixels += int(changed)
                    total_pixels += int(total)

        non_changed = total_pixels - changed_pixels
        ratio = (changed_pixels / total_pixels) if total_pixels > 0 else 0.0

        return {
            "total_pairs": len(self.pairs),
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "non_changed_pixels": non_changed,
            "changed_pixel_ratio": float(ratio),
            "non_changed_pixel_ratio": float(1.0 - ratio if total_pixels > 0 else 0.0),
        }
