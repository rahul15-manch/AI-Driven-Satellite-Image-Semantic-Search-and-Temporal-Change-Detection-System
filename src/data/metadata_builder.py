"""Machine-readable metadata generation utility for remote sensing datasets.

Generates structured JSON and CSV manifests capturing sample provenance,
checksums, image geometry, split assignments, and patch statistics for downstream experiments.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from PIL import Image

from src.data.dataset_verifier import DatasetVerifier, compute_file_sha256


@dataclass
class SampleMetadataRecord:
    """Standardized record describing an image sample or patch."""
    sample_id: str
    source_dataset: str  # "RSICD" or "LEVIR-CD"
    filename: str = ""
    relative_path: str = ""
    split: str = "unknown"
    t1_path: str = ""
    t2_path: Optional[str] = None
    label_path: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 3
    mode: str = "RGB"
    category: str = "unknown"
    caption_count: int = 0
    captions: List[str] = field(default_factory=list)
    sha256: Optional[str] = None
    valid: bool = True
    error_message: Optional[str] = None
    # Patch-specific fields (for LEVIR-CD)
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    patch_size: Optional[int] = None
    patch_index: Optional[int] = None
    patch_x: Optional[int] = None
    patch_y: Optional[int] = None
    changed_pixel_count: Optional[int] = None
    changed_ratio: Optional[float] = None
    preprocessing_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetadataBuilder:
    """Builds and serializes structured metadata for datasets and patch collections."""

    def __init__(self, dataset_name: str, version: str = "1.0.0"):
        self.dataset_name = dataset_name
        self.version = version
        self.records: List[SampleMetadataRecord] = []

    def add_record(self, record: SampleMetadataRecord) -> None:
        """Appends a validated sample metadata record."""
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts records into a Pandas DataFrame."""
        rows = [r.to_dict() for r in self.records]
        return pd.DataFrame(rows)

    def save_json(self, output_path: Path | str) -> None:
        """Saves metadata records to a formatted JSON file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dataset_name": self.dataset_name,
            "record_count": len(self.records),
            "metadata_version": self.version,
            "samples": [r.to_dict() for r in self.records],
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, output_path: Path | str) -> None:
        """Saves metadata records to a tabular CSV file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        if "captions" in df.columns:
            df["captions_joined"] = df["captions"].apply(lambda c: "; ".join(c) if isinstance(c, list) else "")
            df = df.drop(columns=["captions"])
        df.to_csv(out_p, index=False)

    @classmethod
    def build_for_rsicd_directory(
        cls,
        image_dir: Path | str,
        json_path: Optional[Path | str] = None,
        base_dir: Optional[Path | str] = None,
    ) -> MetadataBuilder:
        """Builds metadata for a directory of RSICD images, computing SHA-256 and validating properties."""
        img_dir = Path(image_dir)
        b_dir = Path(base_dir) if base_dir else img_dir.parent
        builder = cls(dataset_name="RSICD", version="1.0.0")

        # Load captions from json if available
        captions_by_filename: Dict[str, List[str]] = {}
        splits_by_filename: Dict[str, str] = {}
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                for item in ann_data.get("images", []):
                    fn = item.get("filename", "")
                    sents = [s.get("raw", "").strip() for s in item.get("sentences", []) if s.get("raw")]
                    captions_by_filename[fn] = sents
                    splits_by_filename[fn] = item.get("split", "unknown")
            except Exception:
                pass

        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        if img_dir.exists():
            files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
            for p in files:
                check = DatasetVerifier.check_image(p, compute_hash=True)
                stem = p.stem
                cat = stem.split("_")[0].lower() if "_" in stem else "unknown"

                caps = captions_by_filename.get(p.name, [])
                split = splits_by_filename.get(p.name, "unknown")

                try:
                    rel_path = str(p.relative_to(b_dir))
                except ValueError:
                    rel_path = str(p)

                rec = SampleMetadataRecord(
                    sample_id=stem,
                    source_dataset="RSICD",
                    filename=p.name,
                    relative_path=rel_path,
                    split=split,
                    t1_path=str(p.resolve()),
                    width=check.width or 0,
                    height=check.height or 0,
                    channels=check.channels or 0,
                    mode=check.mode or "unknown",
                    category=cat,
                    caption_count=len(caps),
                    captions=caps,
                    sha256=check.sha256,
                    valid=check.is_valid,
                    error_message=check.error_message,
                )
                builder.add_record(rec)

        return builder
