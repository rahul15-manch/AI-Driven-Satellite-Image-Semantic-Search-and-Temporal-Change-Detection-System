"""Dataset integrity verification and validation utilities.

Provides deterministic checks for image files, temporal pairs, mask encodings,
SHA-256 duplicate detection, and directory consistency for remote sensing research datasets.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageCheckResult:
    """Result of an individual image file integrity check."""
    path: Path
    is_valid: bool
    filename: str = ""
    extension: str = ""
    file_size_bytes: int = 0
    sha256: Optional[str] = None
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    mode: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.filename and self.path:
            self.filename = self.path.name
        if not self.extension and self.path:
            self.extension = self.path.suffix.lower()


@dataclass
class PairCheckResult:
    """Result of a bi-temporal image pair and ground truth mask verification."""
    stem: str
    is_valid: bool
    path_a: Path
    path_b: Path
    path_label: Optional[Path] = None
    t1_shape: Optional[Tuple[int, int, int]] = None
    t2_shape: Optional[Tuple[int, int, int]] = None
    mask_shape: Optional[Tuple[int, int]] = None
    unique_mask_values: Optional[List[int]] = None
    error_message: Optional[str] = None


@dataclass
class VerificationReport:
    """Summary report of dataset verification."""
    dataset_name: str
    root_path: Path
    total_checked: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    missing_files: List[str] = field(default_factory=list)
    corrupted_files: List[str] = field(default_factory=list)
    dimension_mismatches: List[str] = field(default_factory=list)
    channel_mismatches: List[str] = field(default_factory=list)
    label_value_anomalies: List[str] = field(default_factory=list)
    duplicate_hash_count: int = 0
    duplicates_by_hash: Dict[str, List[str]] = field(default_factory=dict)
    modes_distribution: Dict[str, int] = field(default_factory=dict)
    dimensions_distribution: Dict[str, int] = field(default_factory=dict)
    categories_distribution: Dict[str, int] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Returns True if all checked items are valid and no anomalies were found."""
        return self.invalid_count == 0 and len(self.missing_files) == 0


def compute_file_sha256(path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetVerifier:
    """Provides methods to check remote sensing dataset integrity."""

    @staticmethod
    def check_image(
        path: Path | str,
        expected_channels: Optional[int] = None,
        expected_shape: Optional[Tuple[int, int]] = None,
        compute_hash: bool = True,
    ) -> ImageCheckResult:
        """Verifies if an image file exists, is readable, uncorrupted, and meets criteria.

        Performs complete decoding to detect truncated byte streams or broken headers.

        Args:
            path: Path to the image file.
            expected_channels: Optional expected channel count (e.g., 3 for RGB, 1 for grayscale).
            expected_shape: Optional expected (width, height) tuple.
            compute_hash: Whether to calculate SHA-256 hash.

        Returns:
            ImageCheckResult with validity status and detected properties.
        """
        p = Path(path)
        if not p.exists():
            return ImageCheckResult(
                path=p,
                is_valid=False,
                error_message=f"File does not exist: {p}",
            )

        if not p.is_file():
            return ImageCheckResult(
                path=p,
                is_valid=False,
                error_message=f"Path is not a regular file: {p}",
            )

        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return ImageCheckResult(
                path=p,
                is_valid=False,
                extension=ext,
                error_message=f"Unsupported image extension '{ext}'. Supported: {sorted(list(SUPPORTED_EXTENSIONS))}",
            )

        file_size = p.stat().st_size
        if file_size == 0:
            return ImageCheckResult(
                path=p,
                is_valid=False,
                extension=ext,
                file_size_bytes=0,
                error_message="Zero-byte file (empty image file)",
            )

        sha256_hash = compute_file_sha256(p) if compute_hash else None

        try:
            # First pass: verify container header
            with Image.open(p) as img:
                img.verify()

            # Second pass: fully load and decode pixel bytes
            with Image.open(p) as img:
                width, height = img.size
                mode = img.mode
                img_format = img.format

                if width <= 0 or height <= 0:
                    return ImageCheckResult(
                        path=p,
                        is_valid=False,
                        extension=ext,
                        file_size_bytes=file_size,
                        sha256=sha256_hash,
                        error_message=f"Invalid image dimensions: ({width}, {height})",
                    )

                # Channel count mapping
                channels_map = {
                    "L": 1, "P": 1, "1": 1,
                    "RGB": 3, "YCbCr": 3, "LAB": 3, "HSV": 3,
                    "RGBA": 4, "CMYK": 4, "I": 1, "F": 1
                }
                channels = channels_map.get(mode, len(img.getbands()))

                # Force full pixel decoding into memory to detect truncated streams
                img.load()

                # Check pixel data validity via numpy array conversion
                arr = np.array(img)
                if arr.size == 0:
                    return ImageCheckResult(
                        path=p,
                        is_valid=False,
                        extension=ext,
                        file_size_bytes=file_size,
                        sha256=sha256_hash,
                        error_message="Image pixel array is empty",
                    )

            if expected_shape is not None:
                exp_w, exp_h = expected_shape
                if (width, height) != (exp_w, exp_h):
                    return ImageCheckResult(
                        path=p,
                        is_valid=False,
                        filename=p.name,
                        extension=ext,
                        file_size_bytes=file_size,
                        sha256=sha256_hash,
                        format=img_format,
                        width=width,
                        height=height,
                        channels=channels,
                        mode=mode,
                        error_message=f"Dimension mismatch: expected {expected_shape}, got ({width}, {height})",
                    )

            if expected_channels is not None and channels != expected_channels:
                return ImageCheckResult(
                    path=p,
                    is_valid=False,
                    filename=p.name,
                    extension=ext,
                    file_size_bytes=file_size,
                    sha256=sha256_hash,
                    format=img_format,
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    error_message=f"Channel mismatch: expected {expected_channels}, got {channels} (mode '{mode}')",
                )

            return ImageCheckResult(
                path=p,
                is_valid=True,
                filename=p.name,
                extension=ext,
                file_size_bytes=file_size,
                sha256=sha256_hash,
                format=img_format,
                width=width,
                height=height,
                channels=channels,
                mode=mode,
            )

        except UnidentifiedImageError as exc:
            return ImageCheckResult(
                path=p,
                is_valid=False,
                extension=ext,
                file_size_bytes=file_size,
                sha256=sha256_hash,
                error_message=f"Cannot identify image file (corrupted header or non-image format): {exc}",
            )
        except OSError as exc:
            return ImageCheckResult(
                path=p,
                is_valid=False,
                extension=ext,
                file_size_bytes=file_size,
                sha256=sha256_hash,
                error_message=f"OS/IO error decoding image (truncated or corrupt data stream): {exc}",
            )
        except Exception as exc:
            return ImageCheckResult(
                path=p,
                is_valid=False,
                extension=ext,
                file_size_bytes=file_size,
                sha256=sha256_hash,
                error_message=f"Unexpected error validating image: {exc}",
            )

    @staticmethod
    def check_mask(
        path: Path | str,
        expected_shape: Optional[Tuple[int, int]] = None,
        valid_values: Set[int] | Tuple[int, ...] = (0, 255),
    ) -> ImageCheckResult:
        """Verifies that a ground truth change mask is valid and contains only allowed binary values."""
        base_res = DatasetVerifier.check_image(path, expected_channels=1, expected_shape=expected_shape)
        if not base_res.is_valid:
            return base_res

        try:
            with Image.open(path) as img:
                arr = np.array(img)
                unique_vals = np.unique(arr)
                disallowed = set(unique_vals) - set(valid_values)
                if disallowed:
                    return ImageCheckResult(
                        path=base_res.path,
                        is_valid=False,
                        filename=base_res.filename,
                        extension=base_res.extension,
                        file_size_bytes=base_res.file_size_bytes,
                        sha256=base_res.sha256,
                        format=base_res.format,
                        width=base_res.width,
                        height=base_res.height,
                        channels=base_res.channels,
                        mode=base_res.mode,
                        error_message=(
                            f"Disallowed mask values found: {sorted(list(disallowed))}. "
                            f"Expected only subset of {sorted(list(valid_values))}"
                        ),
                    )

            return base_res

        except Exception as exc:
            return ImageCheckResult(
                path=base_res.path,
                is_valid=False,
                filename=base_res.filename,
                extension=base_res.extension,
                file_size_bytes=base_res.file_size_bytes,
                sha256=base_res.sha256,
                error_message=f"Failed to decode mask array: {exc}",
            )

    @staticmethod
    def check_levir_pair(
        path_a: Path | str,
        path_b: Path | str,
        path_label: Optional[Path | str] = None,
        expected_shape: Optional[Tuple[int, int]] = None,
        valid_mask_values: Set[int] | Tuple[int, ...] = (0, 255),
    ) -> PairCheckResult:
        """Verifies a bi-temporal pair (T1, T2) and its change label mask."""
        pa = Path(path_a)
        pb = Path(path_b)
        pl = Path(path_label) if path_label is not None else None
        stem = pa.stem

        res_a = DatasetVerifier.check_image(pa, expected_channels=3, expected_shape=expected_shape)
        if not res_a.is_valid:
            return PairCheckResult(
                stem=stem,
                is_valid=False,
                path_a=pa,
                path_b=pb,
                path_label=pl,
                error_message=f"T1 image invalid ({res_a.error_message})",
            )

        res_b = DatasetVerifier.check_image(
            pb,
            expected_channels=3,
            expected_shape=(res_a.width, res_a.height) if res_a.width and res_a.height else expected_shape,
        )
        if not res_b.is_valid:
            return PairCheckResult(
                stem=stem,
                is_valid=False,
                path_a=pa,
                path_b=pb,
                path_label=pl,
                error_message=f"T2 image invalid ({res_b.error_message})",
            )

        t1_shape = (res_a.width or 0, res_a.height or 0, res_a.channels or 0)
        t2_shape = (res_b.width or 0, res_b.height or 0, res_b.channels or 0)

        if (res_a.width, res_a.height) != (res_b.width, res_b.height):
            return PairCheckResult(
                stem=stem,
                is_valid=False,
                path_a=pa,
                path_b=pb,
                path_label=pl,
                t1_shape=t1_shape,
                t2_shape=t2_shape,
                error_message=(
                    f"T1/T2 spatial dimension mismatch: T1 is ({res_a.width}, {res_a.height}), "
                    f"T2 is ({res_b.width}, {res_b.height})"
                ),
            )

        mask_shape = None
        unique_mask_values = None
        if pl is not None:
            expected_mask_dims = (res_a.width, res_a.height) if res_a.width and res_a.height else expected_shape
            res_mask = DatasetVerifier.check_mask(pl, expected_shape=expected_mask_dims, valid_values=valid_mask_values)
            if not res_mask.is_valid:
                return PairCheckResult(
                    stem=stem,
                    is_valid=False,
                    path_a=pa,
                    path_b=pb,
                    path_label=pl,
                    t1_shape=t1_shape,
                    t2_shape=t2_shape,
                    error_message=f"Label mask invalid ({res_mask.error_message})",
                )
            mask_shape = (res_mask.width or 0, res_mask.height or 0)
            try:
                with Image.open(pl) as img:
                    arr = np.array(img)
                    unique_mask_values = [int(v) for v in np.unique(arr)]
            except Exception:
                pass

        return PairCheckResult(
            stem=stem,
            is_valid=True,
            path_a=pa,
            path_b=pb,
            path_label=pl,
            t1_shape=t1_shape,
            t2_shape=t2_shape,
            mask_shape=mask_shape,
            unique_mask_values=unique_mask_values,
        )

    @classmethod
    def verify_rsicd_directory(
        cls,
        image_dir: Path | str,
        expected_shape: Optional[Tuple[int, int]] = (224, 224),
        expected_channels: int = 3,
    ) -> VerificationReport:
        """Verifies an RSICD image directory, checking integrity, duplicates, modes, and categories."""
        i_dir = Path(image_dir)
        report = VerificationReport(
            dataset_name="RSICD (Image Directory)",
            root_path=i_dir,
        )

        if not i_dir.exists():
            report.missing_files.append(f"Directory does not exist: {i_dir}")
            report.invalid_count += 1
            return report

        # Find all files with supported image extensions
        image_files = sorted([p for p in i_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
        report.total_checked = len(image_files)

        hashes_seen: Dict[str, List[str]] = {}

        for p in image_files:
            res = cls.check_image(
                p,
                expected_channels=expected_channels,
                expected_shape=expected_shape,
                compute_hash=True,
            )

            if res.is_valid:
                report.valid_count += 1

                # Track mode and dimension distributions
                mode_str = res.mode or "unknown"
                report.modes_distribution[mode_str] = report.modes_distribution.get(mode_str, 0) + 1

                dim_str = f"{res.width}x{res.height}"
                report.dimensions_distribution[dim_str] = report.dimensions_distribution.get(dim_str, 0) + 1

                # Category inference from filename prefix (e.g. airport_01.jpg -> airport)
                stem = p.stem
                cat = stem.split("_")[0].lower() if "_" in stem else "unclassified"
                report.categories_distribution[cat] = report.categories_distribution.get(cat, 0) + 1

                # Track SHA-256 hash for exact duplicate detection
                if res.sha256:
                    hashes_seen.setdefault(res.sha256, []).append(p.name)

            else:
                report.invalid_count += 1
                err_msg = res.error_message or "Unknown error"
                if "dimension mismatch" in err_msg.lower():
                    report.dimension_mismatches.append(f"{p.name}: {err_msg}")
                elif "channel mismatch" in err_msg.lower():
                    report.channel_mismatches.append(f"{p.name}: {err_msg}")
                else:
                    report.corrupted_files.append(f"{p.name}: {err_msg}")

            report.details.append({
                "filename": p.name,
                "valid": res.is_valid,
                "error": res.error_message,
                "width": res.width,
                "height": res.height,
                "channels": res.channels,
                "mode": res.mode,
                "file_size": res.file_size_bytes,
                "sha256": res.sha256,
            })

        # Identify duplicate hashes
        duplicates = {h: names for h, names in hashes_seen.items() if len(names) > 1}
        report.duplicate_hash_count = sum(len(names) - 1 for names in duplicates.values())
        report.duplicates_by_hash = duplicates

        return report

    @classmethod
    def verify_levir_split(
        cls,
        split_dir: Path | str,
        split_name: str = "custom",
        expected_shape: Optional[Tuple[int, int]] = (1024, 1024),
    ) -> VerificationReport:
        """Verifies an entire LEVIR-CD split directory containing A/, B/, and label/ subfolders."""
        s_dir = Path(split_dir)
        report = VerificationReport(
            dataset_name=f"LEVIR-CD ({split_name})",
            root_path=s_dir,
        )

        dir_a = s_dir / "A"
        dir_b = s_dir / "B"
        dir_label = s_dir / "label"

        if not dir_a.exists():
            report.missing_files.append(f"Directory missing: {dir_a}")
        if not dir_b.exists():
            report.missing_files.append(f"Directory missing: {dir_b}")

        has_labels = dir_label.exists()

        if not dir_a.exists() or not dir_b.exists():
            report.invalid_count += 1
            return report

        files_a = {p.name: p for p in dir_a.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS}
        files_b = {p.name: p for p in dir_b.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS}
        files_label = (
            {p.name: p for p in dir_label.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS}
            if has_labels else {}
        )

        all_names = sorted(list(set(files_a.keys()) | set(files_b.keys())))
        report.total_checked = len(all_names)

        for name in all_names:
            pa = files_a.get(name)
            pb = files_b.get(name)
            pl = files_label.get(name)

            if pa is None:
                report.missing_files.append(f"Missing in A/: {name}")
                report.invalid_count += 1
                continue
            if pb is None:
                report.missing_files.append(f"Missing in B/: {name}")
                report.invalid_count += 1
                continue
            if has_labels and pl is None:
                report.missing_files.append(f"Missing in label/: {name}")
                report.invalid_count += 1
                continue

            pair_res = cls.check_levir_pair(pa, pb, pl, expected_shape=expected_shape)
            if pair_res.is_valid:
                report.valid_count += 1
            else:
                report.invalid_count += 1
                if "dimension" in (pair_res.error_message or "").lower():
                    report.dimension_mismatches.append(f"{name}: {pair_res.error_message}")
                elif "disallowed mask values" in (pair_res.error_message or "").lower():
                    report.label_value_anomalies.append(f"{name}: {pair_res.error_message}")
                else:
                    report.corrupted_files.append(f"{name}: {pair_res.error_message}")

            report.details.append({
                "name": name,
                "valid": pair_res.is_valid,
                "error": pair_res.error_message,
                "t1_shape": pair_res.t1_shape,
                "t2_shape": pair_res.t2_shape,
                "unique_mask_values": pair_res.unique_mask_values,
            })

        return report
