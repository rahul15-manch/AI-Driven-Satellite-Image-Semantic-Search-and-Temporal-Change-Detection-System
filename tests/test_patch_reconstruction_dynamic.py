"""Unit tests for dynamic patch extraction and seamless image reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.patch_extractor import (
    PatchExtractor,
    PatchMetadata,
)


def test_reconstruction_1024x1024_regression():
    """Regression test: exactly preserves existing 1024x1024 reconstruction behavior."""
    h, w = 1024, 1024
    original = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
    coords = extractor.compute_patch_grid(w, h)
    patches = [extractor.extract_patch(original, x, y) for x, y in coords]

    # 1024 / 256 = 4 patches along each axis -> 16 patches
    assert len(patches) == 16

    # Reconstruct with explicit target dimensions
    reconstructed = extractor.reconstruct_image(patches, original_height=1024, original_width=1024)
    assert reconstructed.shape == (1024, 1024, 3)
    np.testing.assert_array_equal(original, reconstructed)


def test_reconstruction_non_square_dimensions():
    """Tests reconstruction for non-square arbitrary image dimensions (e.g. 768x512)."""
    h, w = 768, 512
    y_coords, x_coords = np.ogrid[:h, :w]
    original = ((y_coords + x_coords * 2) % 256).astype(np.uint8)
    original = np.stack([original, (original + 50) % 256, (original + 100) % 256], axis=-1)

    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
    coords = extractor.compute_patch_grid(w, h)
    patches = [extractor.extract_patch(original, x, y) for x, y in coords]

    # h: 768 / 256 = 3, w: 512 / 256 = 2 -> 6 patches
    assert len(patches) == 6

    reconstructed = extractor.reconstruct_image(patches, original_height=h, original_width=w)
    assert reconstructed.shape == (h, w, 3)
    np.testing.assert_array_equal(original, reconstructed)


def test_reconstruction_with_metadata_auto_dimensions():
    """Tests automatic canvas dimension inference from PatchMetadata tuples."""
    h, w = 512, 768
    original = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
    # Use extract_from_pair which returns tuples containing PatchMetadata
    paired_results = extractor.extract_from_pair(original, original, parent_stem="test_scene")

    # Each element is (patch_a, patch_b, patch_label, metadata)
    # Pass (patch_a, metadata) tuples to reconstruct_image without explicit original_height/width
    patches_with_meta = [(item[0], item[3]) for item in paired_results]

    reconstructed = extractor.reconstruct_image(patches_with_meta)
    assert reconstructed.shape == (h, w, 3)
    np.testing.assert_array_equal(original, reconstructed)


def test_reconstruction_invalid_patch_count_raises_error():
    """Tests that mismatched patch counts raise ValueError."""
    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
    patches = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(5)]
    # Target 1024x1024 requires 16 patches (4x4)
    with pytest.raises(ValueError, match="Patch count mismatch"):
        extractor.reconstruct_image(patches, original_height=1024, original_width=1024)


def test_reconstruction_overlapping_stride_blending():
    """Tests reconstruction when stride < patch_size (overlapping patches with blending)."""
    h, w = 512, 512
    original = np.ones((h, w, 3), dtype=np.float32) * 128.0

    extractor = PatchExtractor(patch_size=256, stride=128, padding_mode="drop")
    coords = extractor.compute_patch_grid(w, h)
    patches = [extractor.extract_patch(original, x, y) for x, y in coords]

    reconstructed = extractor.reconstruct_image(patches, original_height=h, original_width=w, dtype=np.float32)
    assert reconstructed.shape == (h, w, 3)
    np.testing.assert_allclose(original, reconstructed, atol=1e-3)


def test_reconstruction_partial_edge_patches_pad_mode():
    """Tests reconstruction when padding_mode='pad' on dimensions not divisible by patch_size."""
    h, w = 600, 450
    original = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="pad")
    coords = extractor.compute_patch_grid(w, h)
    patches = [extractor.extract_patch(original, x, y) for x, y in coords]

    # Reconstruct should produce exact (600, 450) shape without out-of-bounds artifact
    reconstructed = extractor.reconstruct_image(patches, original_height=h, original_width=w)
    assert reconstructed.shape == (h, w)
    np.testing.assert_array_equal(original, reconstructed)
