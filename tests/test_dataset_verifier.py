"""Comprehensive unit tests for DatasetVerifier covering all corruption and integrity cases."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.data.dataset_verifier import DatasetVerifier, compute_file_sha256


def test_verify_valid_rgb_image(tmp_path: Path):
    """Verifies that a well-formed RGB image is recognized as valid with correct metadata."""
    img_path = tmp_path / "valid_rgb.jpg"
    img = Image.new("RGB", (224, 224), (120, 140, 160))
    img.save(img_path, format="JPEG")

    res = DatasetVerifier.check_image(img_path, expected_channels=3, expected_shape=(224, 224))
    assert res.is_valid is True
    assert res.width == 224
    assert res.height == 224
    assert res.channels == 3
    assert res.mode == "RGB"
    assert res.error_message is None
    assert res.sha256 is not None
    assert len(res.sha256) == 64


def test_verify_zero_byte_file(tmp_path: Path):
    """Verifies that an empty (0-byte) image file is detected as invalid."""
    empty_path = tmp_path / "empty.jpg"
    empty_path.touch()
    assert empty_path.stat().st_size == 0

    res = DatasetVerifier.check_image(empty_path)
    assert res.is_valid is False
    assert "zero-byte" in res.error_message.lower()


def test_verify_unsupported_image_extension(tmp_path: Path):
    """Verifies that files with unsupported extensions are rejected."""
    bad_ext_path = tmp_path / "notes.txt"
    bad_ext_path.write_text("just text")

    res = DatasetVerifier.check_image(bad_ext_path)
    assert res.is_valid is False
    assert "unsupported" in res.error_message.lower()


def test_verify_corrupted_header(tmp_path: Path):
    """Verifies that a file containing non-image junk bytes fails identification."""
    corrupt_path = tmp_path / "broken_header.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_JPEG_HEADER_AT_ALL_JUNK_DATA_1234567890")

    res = DatasetVerifier.check_image(corrupt_path)
    assert res.is_valid is False
    assert "cannot identify" in res.error_message.lower() or "corrupt" in res.error_message.lower()


def test_verify_truncated_image(tmp_path: Path):
    """Verifies that an image truncated midway fails load decoding."""
    valid_path = tmp_path / "temp.png"
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    img.save(valid_path, format="PNG")

    raw_bytes = valid_path.read_bytes()
    truncated_path = tmp_path / "truncated.png"
    # Write only the first 20% of the byte stream
    truncated_path.write_bytes(raw_bytes[: len(raw_bytes) // 5])

    res = DatasetVerifier.check_image(truncated_path)
    assert res.is_valid is False
    assert res.error_message is not None


def test_verify_invalid_dimensions(tmp_path: Path):
    """Verifies that an image with mismatched dimensions is flagged."""
    img_path = tmp_path / "wrong_dim.jpg"
    img = Image.new("RGB", (100, 100), (50, 50, 50))
    img.save(img_path)

    res = DatasetVerifier.check_image(img_path, expected_shape=(224, 224))
    assert res.is_valid is False
    assert "dimension mismatch" in res.error_message.lower()


def test_verify_grayscale_when_rgb_expected(tmp_path: Path):
    """Verifies that a single-channel grayscale image is rejected when 3-channel RGB is required."""
    gray_path = tmp_path / "gray.jpg"
    img = Image.new("L", (224, 224), 128)
    img.save(gray_path)

    res = DatasetVerifier.check_image(gray_path, expected_channels=3)
    assert res.is_valid is False
    assert "channel mismatch" in res.error_message.lower()


def test_verify_duplicate_detection(tmp_path: Path):
    """Verifies that identical files produce matching SHA-256 hashes and are flagged in directory reports."""
    dir_path = tmp_path / "images"
    dir_path.mkdir()

    img = Image.new("RGB", (224, 224), (80, 120, 160))
    p1 = dir_path / "airport_01.jpg"
    p2 = dir_path / "airport_02_copy.jpg"
    img.save(p1)
    img.save(p2)  # Exact duplicate content

    h1 = compute_file_sha256(p1)
    h2 = compute_file_sha256(p2)
    assert h1 == h2

    report = DatasetVerifier.verify_rsicd_directory(dir_path, expected_shape=(224, 224))
    assert report.total_checked == 2
    assert report.valid_count == 2
    assert report.duplicate_hash_count == 1
    assert len(report.duplicates_by_hash) == 1
