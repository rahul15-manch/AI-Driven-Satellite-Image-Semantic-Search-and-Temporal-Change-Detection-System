"""Configurable patch extraction utility for high-resolution remote sensing imagery.

Supports deterministic sub-patch extraction for bi-temporal image pairs (T1, T2)
and ground-truth masks with strict spatial alignment and leakage-prevention guarantees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


@dataclass
class PatchMetadata:
    """Metadata for an individual extracted patch."""
    parent_stem: str
    patch_index: int
    x: int
    y: int
    patch_width: int
    patch_height: int
    original_width: int
    original_height: int
    is_padded: bool = False
    changed_pixel_count: Optional[int] = None
    changed_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatchExtractor:
    """Extracts synchronized patches from high-resolution remote sensing scenes.

    CRITICAL LEAKAGE-PREVENTION PRINCIPLE:
    This class operates on single image pairs. Split partitioning (Train / Val / Test)
    MUST be performed on the parent image pairs BEFORE patch extraction is invoked.
    Patches derived from the same parent geographic scene must never cross split boundaries.
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        padding_mode: str = "drop",
        pad_value: int = 0,
    ):
        """Initializes the patch extractor with configurable parameters.

        Args:
            patch_size: Width and height of extracted square patches (default 256).
            stride: Step size between consecutive patch origins. If None, defaults to patch_size (non-overlapping).
            padding_mode: 'drop' to discard boundary leftovers smaller than patch_size,
                          or 'pad' to pad boundary edges to patch_size.
            pad_value: Constant fill value when padding_mode='pad'.
        """
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        if padding_mode not in {"drop", "pad"}:
            raise ValueError(f"padding_mode must be 'drop' or 'pad', got {padding_mode}")
        self.padding_mode = padding_mode
        self.pad_value = pad_value

    @property
    def is_overlapping(self) -> bool:
        """Returns True if stride < patch_size resulting in overlapping patches."""
        return self.stride < self.patch_size

    def compute_patch_grid(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Computes top-left (x, y) coordinates for all patches in an image of given dimensions.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            List of (x, y) coordinates.
        """
        if width < self.patch_size or height < self.patch_size:
            if self.padding_mode == "drop":
                return []
            else:
                return [(0, 0)]

        coords: List[Tuple[int, int]] = []

        if self.padding_mode == "drop":
            y_steps = (height - self.patch_size) // self.stride + 1
            x_steps = (width - self.patch_size) // self.stride + 1
            for j in range(y_steps):
                y = j * self.stride
                for i in range(x_steps):
                    x = i * self.stride
                    coords.append((x, y))
        else:  # pad mode
            y_max = math.ceil((height - self.patch_size) / self.stride) if height > self.patch_size else 0
            x_max = math.ceil((width - self.patch_size) / self.stride) if width > self.patch_size else 0
            for j in range(y_max + 1):
                y = j * self.stride
                for i in range(x_max + 1):
                    x = i * self.stride
                    coords.append((x, y))

        return coords

    def extract_patch(
        self,
        arr: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        """Extracts a single patch from a 2D or 3D numpy array at coordinate (x, y).

        Pads with pad_value if the patch exceeds array bounds and padding_mode is 'pad'.
        """
        h, w = arr.shape[:2]
        crop = arr[y : y + self.patch_size, x : x + self.patch_size]

        crop_h, crop_w = crop.shape[:2]
        if crop_h == self.patch_size and crop_w == self.patch_size:
            return crop

        if self.padding_mode == "pad":
            pad_h = self.patch_size - crop_h
            pad_w = self.patch_size - crop_w
            if arr.ndim == 3:
                return np.pad(
                    crop,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=self.pad_value,
                )
            else:
                return np.pad(
                    crop,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=self.pad_value,
                )

        return crop

    def extract_from_pair(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        label: Optional[np.ndarray] = None,
        parent_stem: str = "scene",
    ) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], PatchMetadata]]:
        """Extracts synchronized patches across T1, T2, and an optional label mask.

        Guarantees exact spatial correspondence across all components.

        Args:
            img_a: Pre-change image array (H, W, C).
            img_b: Post-change image array (H, W, C).
            label: Optional binary change mask array (H, W).
            parent_stem: Stem identifier of the source scene.

        Returns:
            List of tuples: (patch_a, patch_b, patch_label, PatchMetadata).
        """
        if img_a.shape[:2] != img_b.shape[:2]:
            raise ValueError(
                f"T1 and T2 shape mismatch: img_a {img_a.shape} vs img_b {img_b.shape}"
            )

        if label is not None and label.shape[:2] != img_a.shape[:2]:
            raise ValueError(
                f"Label shape mismatch: label {label.shape} vs image {img_a.shape}"
            )

        h, w = img_a.shape[:2]
        coords = self.compute_patch_grid(width=w, height=h)

        results = []
        for idx, (x, y) in enumerate(coords):
            patch_a = self.extract_patch(img_a, x, y)
            patch_b = self.extract_patch(img_b, x, y)

            patch_lbl = None
            changed_pixels = None
            changed_ratio = None

            if label is not None:
                patch_lbl = self.extract_patch(label, x, y)
                changed_pixels = int(np.count_nonzero(patch_lbl > 0))
                changed_ratio = float(changed_pixels / (self.patch_size * self.patch_size))

            is_padded = (y + self.patch_size > h) or (x + self.patch_size > w)

            meta = PatchMetadata(
                parent_stem=parent_stem,
                patch_index=idx,
                x=x,
                y=y,
                patch_width=self.patch_size,
                patch_height=self.patch_size,
                original_width=w,
                original_height=h,
                is_padded=is_padded,
                changed_pixel_count=changed_pixels,
                changed_ratio=changed_ratio,
            )

            results.append((patch_a, patch_b, patch_lbl, meta))

        return results

    def save_patches_to_disk(
        self,
        img_a_path: Path | str,
        img_b_path: Path | str,
        label_path: Optional[Path | str],
        out_dir_a: Path | str,
        out_dir_b: Path | str,
        out_dir_label: Optional[Path | str],
        parent_stem: Optional[str] = None,
    ) -> List[PatchMetadata]:
        """Reads parent pair from disk, extracts patches, and saves them to disk.

        Maintains synchronized filenames: {parent_stem}_p{idx:04d}.png across folders.
        """
        pa = Path(img_a_path)
        pb = Path(img_b_path)
        pl = Path(label_path) if label_path is not None else None
        stem = parent_stem or pa.stem

        out_a = Path(out_dir_a)
        out_b = Path(out_dir_b)
        out_a.mkdir(parents=True, exist_ok=True)
        out_b.mkdir(parents=True, exist_ok=True)

        out_lbl = Path(out_dir_label) if out_dir_label is not None else None
        if out_lbl is not None:
            out_lbl.mkdir(parents=True, exist_ok=True)

        with Image.open(pa) as img:
            arr_a = np.array(img.convert("RGB"))
        with Image.open(pb) as img:
            arr_b = np.array(img.convert("RGB"))

        arr_l = None
        if pl is not None and pl.exists():
            with Image.open(pl) as img:
                arr_l = np.array(img.convert("L"))

        extracted = self.extract_from_pair(arr_a, arr_b, arr_l, parent_stem=stem)
        metadata_list: List[PatchMetadata] = []

        for p_a, p_b, p_lbl, meta in extracted:
            fname = f"{stem}_p{meta.patch_index:04d}.png"
            Image.fromarray(p_a).save(out_a / fname)
            Image.fromarray(p_b).save(out_b / fname)
            if p_lbl is not None and out_lbl is not None:
                Image.fromarray(p_lbl).save(out_lbl / fname)
            metadata_list.append(meta)

        return metadata_list
