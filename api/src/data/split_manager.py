"""Leakage-safe split management utility for remote sensing datasets.

Supports reading official split definitions or generating reproducible,
deterministic custom splits with strict non-overlap and temporal-pairing integrity guarantees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class SplitManifest:
    """Summary and contents of a dataset split."""
    dataset_name: str
    split_type: str  # "official" or "custom"
    seed: Optional[int]
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]

    @property
    def total_samples(self) -> int:
        return len(self.train_ids) + len(self.val_ids) + len(self.test_ids)

    def validate_no_leakage(self) -> bool:
        """Verifies that train, val, and test partitions are completely disjoint."""
        s_train = set(self.train_ids)
        s_val = set(self.val_ids)
        s_test = set(self.test_ids)

        if len(s_train & s_val) > 0:
            return False
        if len(s_train & s_test) > 0:
            return False
        if len(s_val & s_test) > 0:
            return False
        return True

    def get_overlap_summary(self) -> Dict[str, List[str]]:
        """Returns any overlapping IDs between splits (empty if no leakage)."""
        s_train = set(self.train_ids)
        s_val = set(self.val_ids)
        s_test = set(self.test_ids)

        return {
            "train_val_overlap": sorted(list(s_train & s_val)),
            "train_test_overlap": sorted(list(s_train & s_test)),
            "val_test_overlap": sorted(list(s_val & s_test)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split_type": self.split_type,
            "seed": self.seed,
            "counts": {
                "train": len(self.train_ids),
                "val": len(self.val_ids),
                "test": len(self.test_ids),
                "total": self.total_samples,
            },
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
        }


class SplitManager:
    """Creates, validates, and persists train/validation/test partitions."""

    @staticmethod
    def create_custom_split(
        sample_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        dataset_name: str = "custom_dataset",
    ) -> SplitManifest:
        """Generates a deterministic custom train/val/test split from a list of unique sample IDs.

        Guarantees:
        - Deterministic reproducibility via explicit random seed.
        - Absolute non-overlap (no data leakage).
        - Temporal pairs share the same sample_id and are thus never split across partitions.
        """
        if not math_is_close(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"
            )

        unique_ids = sorted(list(set(sample_ids)))
        if len(unique_ids) != len(sample_ids):
            raise ValueError("Input sample_ids contains duplicates; sample IDs must be unique.")

        n = len(unique_ids)
        if n == 0:
            return SplitManifest(
                dataset_name=dataset_name,
                split_type="custom",
                seed=seed,
                train_ids=[],
                val_ids=[],
                test_ids=[],
            )

        rng = np.random.RandomState(seed)
        shuffled = unique_ids.copy()
        rng.shuffle(shuffled)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Allocate remaining samples to test to guarantee sum == n
        train_ids = sorted(shuffled[:n_train])
        val_ids = sorted(shuffled[n_train : n_train + n_val])
        test_ids = sorted(shuffled[n_train + n_val :])

        manifest = SplitManifest(
            dataset_name=dataset_name,
            split_type="custom",
            seed=seed,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

        assert manifest.validate_no_leakage(), "Internal error: generated split contains leakage!"
        return manifest

    @staticmethod
    def save_split_manifest(manifest: SplitManifest, output_path: Path | str) -> None:
        """Saves a SplitManifest to a formatted JSON file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    @staticmethod
    def load_split_manifest(input_path: Path | str) -> SplitManifest:
        """Loads a SplitManifest from a JSON file and verifies non-overlap."""
        inp_p = Path(input_path)
        if not inp_p.exists():
            raise FileNotFoundError(f"Split file not found: {inp_p}")

        with open(inp_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = SplitManifest(
            dataset_name=data.get("dataset_name", "unknown"),
            split_type=data.get("split_type", "unknown"),
            seed=data.get("seed"),
            train_ids=data.get("train_ids", []),
            val_ids=data.get("val_ids", []),
            test_ids=data.get("test_ids", []),
        )

        if not manifest.validate_no_leakage():
            overlaps = manifest.get_overlap_summary()
            raise ValueError(f"Loaded split manifest contains data leakage: {overlaps}")

        return manifest


def math_is_close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol
