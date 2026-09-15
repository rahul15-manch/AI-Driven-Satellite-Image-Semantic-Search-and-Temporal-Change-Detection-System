"""Dedicated validation tool for user-supplied RSICD dataset.

Performs objective image decoding, corruption detection, duplicate hashing,
and dimension/mode checks on user-provided RSICD imagery.

Usage:
    python3 -m src.data.verify_rsicd
    python3 -m src.data.verify_rsicd --data-dir /path/to/rsicd/images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.data.dataset_verifier import DatasetVerifier, VerificationReport


def run_rsicd_verification(
    data_dir: Path,
    expected_width: int = 224,
    expected_height: int = 224,
    save_report: bool = True,
    output_report_path: Path | None = None,
) -> VerificationReport:
    """Executes full verification of supplied RSICD directory."""
    print("=" * 80)
    print("RSICD DATASET INTEGRITY VERIFICATION")
    print(f"Target Directory: {data_dir.resolve()}")
    print(f"Expected Properties: {expected_width}x{expected_height} RGB (3 channels)")
    print("=" * 80)

    if not data_dir.exists():
        print(f"\n[ERROR] Provided RSICD directory does not exist: {data_dir}", file=sys.stderr)
        print("\nPlease place the valid RSICD image files inside:")
        print("    data/raw/rsicd/images/   (or directly in data/raw/rsicd/)")
        print("\nThen re-run: python3 -m src.data.verify_rsicd")
        report = VerificationReport(dataset_name="RSICD", root_path=data_dir)
        report.missing_files.append(f"Directory not found: {data_dir}")
        return report

    # Check if images are inside an 'images/' subfolder or directly in data_dir
    target_img_dir = data_dir / "images" if (data_dir / "images").is_dir() else data_dir

    report = DatasetVerifier.verify_rsicd_directory(
        image_dir=target_img_dir,
        expected_shape=(expected_width, expected_height),
        expected_channels=3,
    )

    print("\nRSICD Validation Report")
    print("-" * 50)
    print(f"Total files checked:       {report.total_checked}")
    print(f"Valid files:               {report.valid_count}")
    print(f"Invalid / Corrupted files: {report.invalid_count}")
    print(f"Duplicate files (SHA-256): {report.duplicate_hash_count}")

    print("\nDimensions Distribution:")
    if report.dimensions_distribution:
        for dim, count in sorted(report.dimensions_distribution.items()):
            print(f"  - {dim}: {count}")
    else:
        print("  None detected.")

    print("\nColor Modes Distribution:")
    if report.modes_distribution:
        for mode, count in sorted(report.modes_distribution.items()):
            print(f"  - {mode}: {count}")
    else:
        print("  None detected.")

    print(f"\nUnique Categories Detected: {len(report.categories_distribution)}")
    if report.categories_distribution:
        for cat, count in sorted(report.categories_distribution.items())[:15]:
            print(f"  - {cat}: {count}")
        if len(report.categories_distribution) > 15:
            print(f"  ... and {len(report.categories_distribution) - 15} more categories")

    if report.corrupted_files:
        print(f"\nCorrupted Files ({len(report.corrupted_files)}):")
        for err in report.corrupted_files[:10]:
            print(f"  [!] {err}")
        if len(report.corrupted_files) > 10:
            print(f"  ... and {len(report.corrupted_files) - 10} more corrupted files")

    if report.dimension_mismatches:
        print(f"\nDimension Mismatches ({len(report.dimension_mismatches)}):")
        for err in report.dimension_mismatches[:10]:
            print(f"  [!] {err}")

    if report.channel_mismatches:
        print(f"\nChannel Mismatches ({len(report.channel_mismatches)}):")
        for err in report.channel_mismatches[:10]:
            print(f"  [!] {err}")

    if report.duplicates_by_hash:
        print(f"\nIdentified Exact Duplicates ({len(report.duplicates_by_hash)} hash collisions):")
        for h, fnames in list(report.duplicates_by_hash.items())[:5]:
            print(f"  Hash {h[:12]}... : {fnames}")

    print("\n" + "=" * 80)
    if report.total_checked == 0:
        print("[STATUS] No supported image files found in target directory.")
        print("Please place the unpacked RSICD images in data/raw/rsicd/ and re-run.")
    elif report.is_clean:
        print("[STATUS] Validation PASSED: All supplied images are valid, uncorrupted, and meet specifications.")
    else:
        print(f"[STATUS] Validation FAILED: Found {report.invalid_count} anomalous or corrupted files.")
    print("=" * 80)

    if save_report:
        rep_path = output_report_path or Path("data/metadata/rsicd/validation_report.json")
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "dataset_name": report.dataset_name,
            "directory": str(target_img_dir.resolve()),
            "total_checked": report.total_checked,
            "valid_count": report.valid_count,
            "invalid_count": report.invalid_count,
            "duplicate_hash_count": report.duplicate_hash_count,
            "dimensions_distribution": report.dimensions_distribution,
            "modes_distribution": report.modes_distribution,
            "categories_distribution": report.categories_distribution,
            "corrupted_files": report.corrupted_files,
            "dimension_mismatches": report.dimension_mismatches,
            "channel_mismatches": report.channel_mismatches,
            "duplicates_by_hash": report.duplicates_by_hash,
            "is_clean": report.is_clean,
        }
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nFull validation report saved to: {rep_path}")

    return report


def main():
    default_dir = os.environ.get("RSICD_ROOT", "data/raw/rsicd")
    parser = argparse.ArgumentParser(description="Validate user-supplied RSICD image dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_dir,
        help=f"Path to RSICD root or images directory (default: {default_dir})",
    )
    parser.add_argument("--expected-width", type=int, default=224, help="Expected image width (default: 224)")
    parser.add_argument("--expected-height", type=int, default=224, help="Expected image height (default: 224)")
    parser.add_argument("--report-path", type=str, default=None, help="Optional custom output path for JSON report")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    rep_path = Path(args.report_path) if args.report_path else None

    run_rsicd_verification(
        data_dir=data_dir,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        save_report=True,
        output_report_path=rep_path,
    )


if __name__ == "__main__":
    main()
