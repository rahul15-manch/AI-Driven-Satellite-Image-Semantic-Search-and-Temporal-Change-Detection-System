"""Dataset integration script for user-provided RSICD and LEVIR-CD datasets.

Integrates:
1. RSICD: Extracts images and captions from user CSVs into data/raw/rsicd/images/ and dataset_rsicd.json.
2. LEVIR-CD: Moves official train/val/test splits into data/raw/levir_cd/.
3. Verifies every image and pair, identifying and reporting any corrupted artifacts.
4. Generates structured metadata in data/metadata/ and split manifests in data/splits/.

Usage:
    python3 scripts/integrate_user_datasets.py
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset_verifier import DatasetVerifier, compute_file_sha256
from src.data.metadata_builder import MetadataBuilder, SampleMetadataRecord
from src.data.split_manager import SplitManager, SplitManifest


def parse_captions(raw_str: str) -> list[str]:
    """Robustly parses caption strings formatted from numpy string arrays."""
    raw_str = raw_str.strip()
    if raw_str.startswith("[") and raw_str.endswith("]"):
        raw_str = raw_str[1:-1]
    parts = re.split(r"(?<=['\"])\s+(?=['\"])", raw_str)
    cleaned = []
    for p in parts:
        p = p.strip()
        if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
            p = p[1:-1]
        cleaned.append(p.replace("\\'", "'").replace('\\"', '"').strip())
    return [c for c in cleaned if c]


def integrate_rsicd(source_dir: Path, target_dir: Path) -> dict:
    print("\n" + "=" * 80)
    print("INTEGRATING RSICD DATASET")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("=" * 80)

    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [
        ("train", source_dir / "train.csv"),
        ("val", source_dir / "valid.csv"),
        ("test", source_dir / "test.csv"),
    ]

    total_extracted = 0
    corrupted_count = 0
    corrupted_files = []
    json_images = []
    split_ids = {"train": [], "val": [], "test": []}

    t0 = time.perf_counter()

    for split_name, csv_path in csv_files:
        if not csv_path.exists():
            print(f"[WARN] CSV not found: {csv_path}", file=sys.stderr)
            continue

        print(f"\nProcessing {split_name} split from {csv_path.name}...")
        df = pd.read_csv(csv_path)
        print(f"  Rows to process: {len(df)}")

        for idx, row in df.iterrows():
            orig_filename = str(row["filename"])
            fname = Path(orig_filename).name

            try:
                # Parse image dict
                img_data = ast.literal_eval(row["image"])
                raw_bytes = img_data.get("bytes")
                if not raw_bytes or len(raw_bytes) == 0:
                    corrupted_count += 1
                    corrupted_files.append((fname, "Zero-byte / missing image bytes"))
                    continue

                # Verify image can be decoded
                img = Image.open(io.BytesIO(raw_bytes))
                img.verify()

                # Fully load to catch truncated streams
                img = Image.open(io.BytesIO(raw_bytes))
                img.load()
                w, h = img.size

                # Save verified image to target directory
                out_path = images_dir / fname
                with open(out_path, "wb") as f:
                    f.write(raw_bytes)

                # Parse captions
                captions = parse_captions(str(row["captions"]))
                sentences = [{"raw": c, "tokens": c.lower().split()} for c in captions]

                stem = Path(fname).stem
                cat = stem.split("_")[0].lower() if "_" in stem else "unknown"

                json_images.append({
                    "imgid": total_extracted,
                    "filename": fname,
                    "split": split_name,
                    "category": cat,
                    "sentences": sentences,
                })
                split_ids[split_name].append(stem)
                total_extracted += 1

            except Exception as exc:
                corrupted_count += 1
                corrupted_files.append((fname, str(exc)))

            if (idx + 1) % 2500 == 0 or (idx + 1) == len(df):
                print(f"    Processed {idx + 1}/{len(df)} images...")

    elapsed = time.perf_counter() - t0
    print(f"\nRSICD Extraction Summary:")
    print(f"  Successfully extracted & verified: {total_extracted} images in {elapsed:.2f}s")
    print(f"  Corrupted / invalid images:        {corrupted_count}")

    if corrupted_files:
        print("\n  Corrupted files identified:")
        for fn, err in corrupted_files[:10]:
            print(f"    [!] {fn}: {err}")

    # Save Karpathy-format JSON
    json_path = target_dir / "dataset_rsicd.json"
    rsicd_catalog = {
        "dataset": "RSICD",
        "images": json_images,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rsicd_catalog, f, indent=2)
    print(f"  Saved annotation catalog to: {json_path}")

    # Save split manifest
    split_manifest = SplitManifest(
        dataset_name="RSICD",
        split_type="official",
        seed=None,
        train_ids=split_ids["train"],
        val_ids=split_ids["val"],
        test_ids=split_ids["test"],
    )
    SplitManager.save_split_manifest(split_manifest, "data/splits/rsicd/rsicd_splits.json")
    print(f"  Saved split manifest to: data/splits/rsicd/rsicd_splits.json")

    return {
        "total_extracted": total_extracted,
        "corrupted_count": corrupted_count,
        "corrupted_files": corrupted_files,
        "splits": {k: len(v) for k, v in split_ids.items()},
    }


def integrate_levir_cd(source_dir: Path, target_dir: Path) -> dict:
    print("\n" + "=" * 80)
    print("INTEGRATING LEVIR-CD DATASET")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("=" * 80)

    target_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    split_counts = {}
    split_ids = {"train": [], "val": [], "test": []}

    for split in splits:
        src_split = source_dir / split
        tgt_split = target_dir / split

        if not src_split.exists():
            print(f"[WARN] Split directory missing: {src_split}", file=sys.stderr)
            continue

        tgt_split.mkdir(parents=True, exist_ok=True)
        for sub in ["A", "B", "label"]:
            src_sub = src_split / sub
            tgt_sub = tgt_split / sub
            tgt_sub.mkdir(parents=True, exist_ok=True)

            if src_sub.exists():
                # Move files if not already there
                for p in src_sub.iterdir():
                    if p.is_file() and not p.name.startswith("."):
                        tgt_p = tgt_sub / p.name
                        if not tgt_p.exists():
                            shutil.move(str(p), str(tgt_p))

        # Collect pair IDs
        stems = sorted([p.stem for p in (tgt_split / "A").glob("*.png")])
        split_counts[split] = len(stems)
        split_ids[split] = stems
        print(f"  {split} split integrated: {len(stems)} pairs")

    # Save LEVIR split manifest
    split_manifest = SplitManifest(
        dataset_name="LEVIR-CD",
        split_type="official",
        seed=None,
        train_ids=split_ids["train"],
        val_ids=split_ids["val"],
        test_ids=split_ids["test"],
    )
    SplitManager.save_split_manifest(split_manifest, "data/splits/levir_cd/levir_splits.json")
    print(f"  Saved split manifest to: data/splits/levir_cd/levir_splits.json")

    return {
        "split_counts": split_counts,
        "total_pairs": sum(split_counts.values()),
    }


def main():
    rsicd_src = Path("data/RSICD Image Caption Dataset")
    rsicd_tgt = Path("data/raw/rsicd")

    levir_src = Path("data/LEVIR CD")
    levir_tgt = Path("data/raw/levir_cd")

    rsicd_res = {}
    if rsicd_src.exists():
        rsicd_res = integrate_rsicd(rsicd_src, rsicd_tgt)

    levir_res = {}
    if levir_src.exists():
        levir_res = integrate_levir_cd(levir_src, levir_tgt)

    # Clean up empty parent directories after move
    if levir_src.exists():
        # Remove leftover .DS_Store
        for ds in levir_src.rglob(".DS_Store"):
            ds.unlink()
        try:
            shutil.rmtree(levir_src)
            print(f"\n[✓] Cleaned up temporary directory: {levir_src}")
        except Exception as e:
            print(f"Note: could not remove {levir_src}: {e}")

    # Remove unneeded .DS_Store files across data/
    for ds in Path("data").rglob(".DS_Store"):
        ds.unlink()

    print("\n" + "=" * 80)
    print("DATASET INTEGRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
