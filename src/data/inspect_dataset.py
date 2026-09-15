"""Command-line utility to inspect, validate, and compute factual statistics on local datasets.

Usage:
    python3 -m src.data.inspect_dataset --dataset levir --data-dir data/raw/levir_cd/train
    python3 -m src.data.inspect_dataset --dataset rsicd --data-dir data/raw/rsicd/images --json-path data/raw/rsicd/dataset_rsicd.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from src.data.dataset_verifier import DatasetVerifier
from src.data.levir_loader import LEVIRDataset
from src.data.rsicd_loader import RSICDDataset


def inspect_levir(data_dir: Path, split_name: str) -> None:
    print(f"=== Inspecting LEVIR-CD Split: {data_dir} ===")
    report = DatasetVerifier.verify_levir_split(data_dir, split_name=split_name, expected_shape=None)
    print(f"Total pairs checked: {report.total_checked}")
    print(f"Valid pairs:         {report.valid_count}")
    print(f"Invalid pairs:       {report.invalid_count}")

    if report.missing_files:
        print(f"Missing files ({len(report.missing_files)}):")
        for f in report.missing_files[:5]:
            print(f"  - {f}")
    if report.dimension_mismatches:
        print(f"Dimension mismatches ({len(report.dimension_mismatches)}):")
        for f in report.dimension_mismatches[:5]:
            print(f"  - {f}")
    if report.label_value_anomalies:
        print(f"Label value anomalies ({len(report.label_value_anomalies)}):")
        for f in report.label_value_anomalies[:5]:
            print(f"  - {f}")

    if report.valid_count > 0:
        try:
            ds = LEVIRDataset(root_dir=data_dir)
            stats = ds.compute_class_statistics()
            print("\nPixel-level Class Statistics (Locally Measured):")
            print(f"  - Total pixels:        {stats['total_pixels']:,}")
            print(f"  - Changed pixels:      {stats['changed_pixels']:,} ({stats['changed_pixel_ratio']*100:.2f}%)")
            print(f"  - Non-changed pixels:  {stats['non_changed_pixels']:,} ({stats['non_changed_pixel_ratio']*100:.2f}%)")
        except Exception as e:
            print(f"Could not compute pixel statistics: {e}")


def inspect_rsicd(image_dir: Path, json_path: Path | None) -> None:
    print(f"=== Inspecting RSICD Dataset: {image_dir} ===")
    try:
        ds = RSICDDataset(image_dir=image_dir, json_path=json_path)
        stats = ds.compute_dataset_statistics()
        print(f"Total samples loaded:        {stats['total_samples']}")
        print(f"Total captions:              {stats['total_captions']}")
        print(f"Avg captions per sample:     {stats['avg_captions_per_sample']:.2f}")
        print(f"Avg caption length (words):  {stats['avg_caption_word_length']:.2f}")
        print(f"Unique scene categories:     {stats['unique_categories_count']}")
        print("Category distribution sample:")
        for cat, cnt in list(stats['category_distribution'].items())[:10]:
            print(f"  - {cat}: {cnt}")
    except Exception as e:
        print(f"Error loading RSICD: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect remote sensing dataset directory")
    parser.add_argument("--dataset", choices=["levir", "rsicd"], required=True, help="Dataset type")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--json-path", type=str, default=None, help="Optional path to annotation JSON (for RSICD)")
    parser.add_argument("--split-name", type=str, default="custom", help="Split label name")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dataset == "levir":
        inspect_levir(data_dir, args.split_name)
    elif args.dataset == "rsicd":
        json_p = Path(args.json_path) if args.json_path else None
        inspect_rsicd(data_dir, json_p)


if __name__ == "__main__":
    main()
