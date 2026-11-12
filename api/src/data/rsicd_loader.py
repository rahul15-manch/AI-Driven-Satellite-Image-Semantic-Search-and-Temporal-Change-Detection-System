"""Dataset loader and PyTorch Dataset representation for RSICD remote sensing imagery and captions.

Supports parsing standard Karpathy-format dataset_rsicd.json annotations,
category derivation from filename prefixes, and structured batch retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class RSICDSample:
    """Represents an individual RSICD image with associated textual descriptions."""
    sample_id: str
    filename: str
    image_path: Path
    captions: List[str] = field(default_factory=list)
    category: str = "unknown"
    split: str = "unknown"


class RSICDDataset(Dataset):
    """PyTorch Dataset for RSICD (Remote Sensing Image Captioning Dataset).

    Returns:
        image_tensor: FloatTensor of shape (C, H, W) normalized to [0, 1] (or transformed).
        captions: List[str] of natural-language descriptions (typically 5 per image).
        category: String scene category label (e.g., 'airport', 'forest', 'denseresidential').
        meta: Dict containing sample metadata and file paths.
    """

    def __init__(
        self,
        image_dir: Path | str,
        json_path: Optional[Path | str] = None,
        split: Optional[str] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        apply_default_norm: bool = False,
    ):
        """Initializes RSICDDataset.

        Args:
            image_dir: Path to directory containing RSICD image files (224x224 RGB).
            json_path: Optional path to dataset_rsicd.json annotation file.
            split: Optional split filter ('train', 'val', 'test').
            transform: Optional custom torchvision transform.
            normalize_mean: Channel mean if apply_default_norm is True.
            normalize_std: Channel std if apply_default_norm is True.
            apply_default_norm: Whether to apply standard ImageNet normalization.
        """
        self.image_dir = Path(image_dir)
        self.json_path = Path(json_path) if json_path is not None else None
        self.split = split
        self.transform = transform
        self.apply_default_norm = apply_default_norm
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1)

        self.samples = self._load_samples()

    def _infer_category_from_filename(self, filename: str) -> str:
        """Infers class category from RSICD filename conventions (e.g. 'airport_14.jpg' -> 'airport')."""
        stem = Path(filename).stem
        match = re.match(r"^([a-zA-Z]+)(?:_|\d)", stem)
        if match:
            return match.group(1).lower()
        return "unclassified"

    def _load_samples(self) -> List[RSICDSample]:
        """Loads and filters dataset samples from JSON or filesystem."""
        if self.json_path and self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_images = data.get("images", [])
            samples: List[RSICDSample] = []

            for item in raw_images:
                item_split = item.get("split", "unknown")
                if self.split and item_split != self.split:
                    continue

                fname = item.get("filename", "")
                img_path = self.image_dir / fname

                raw_sentences = item.get("sentences", [])
                captions = [s.get("raw", "").strip() for s in raw_sentences if s.get("raw")]

                category = item.get("category") or self._infer_category_from_filename(fname)

                samples.append(RSICDSample(
                    sample_id=str(item.get("imgid", Path(fname).stem)),
                    filename=fname,
                    image_path=img_path,
                    captions=captions,
                    category=category,
                    split=item_split,
                ))

            return samples

        # Fallback: discover images directly from directory if JSON is not present
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        image_files = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in valid_exts])

        samples = []
        for p in image_files:
            cat = self._infer_category_from_filename(p.name)
            samples.append(RSICDSample(
                sample_id=p.stem,
                filename=p.name,
                image_path=p,
                captions=[],
                category=cat,
                split=self.split or "unknown",
            ))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[str], str, Dict[str, Any]]:
        sample = self.samples[idx]

        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                arr = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(arr).permute(2, 0, 1)

        if self.apply_default_norm and self.transform is None:
            img_tensor = (img_tensor - self.normalize_mean) / self.normalize_std

        meta = {
            "sample_id": sample.sample_id,
            "filename": sample.filename,
            "path": str(sample.image_path),
            "category": sample.category,
            "split": sample.split,
            "caption_count": len(sample.captions),
            "width": w,
            "height": h,
        }

        return img_tensor, sample.captions, sample.category, meta

    def compute_dataset_statistics(self) -> Dict[str, Any]:
        """Computes factual category distribution and caption statistics for verified samples."""
        categories: Dict[str, int] = {}
        total_captions = 0
        caption_lengths: List[int] = []

        for sample in self.samples:
            cat = sample.category
            categories[cat] = categories.get(cat, 0) + 1
            total_captions += len(sample.captions)
            for c in sample.captions:
                caption_lengths.append(len(c.split()))

        avg_len = float(np.mean(caption_lengths)) if caption_lengths else 0.0

        return {
            "total_samples": len(self.samples),
            "total_captions": total_captions,
            "avg_captions_per_sample": (total_captions / len(self.samples)) if self.samples else 0.0,
            "avg_caption_word_length": avg_len,
            "unique_categories_count": len(categories),
            "category_distribution": dict(sorted(categories.items())),
        }
