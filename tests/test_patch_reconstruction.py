"""Unit tests for PatchExtractor reconstruction functionality."""

import numpy as np
import pytest

from src.data.patch_extractor import PatchExtractor


def test_reconstruct_image_identity_2d():
    """Verifies that extracting patches and reconstructing reproduces the exact original 2D array."""
    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
    rng = np.random.default_rng(42)
    original = rng.integers(0, 2, size=(1024, 1024), dtype=np.uint8)

    # Extract patches
    coords = extractor.compute_patch_grid(1024, 1024)
    patches = [extractor.extract_patch(original, x, y) for x, y in coords]
    assert len(patches) == 16

    # Reconstruct
    reconstructed = extractor.reconstruct_image(patches, original_height=1024, original_width=1024)

    assert reconstructed.shape == (1024, 1024)
    assert reconstructed.dtype == original.dtype
    np.testing.assert_array_equal(reconstructed, original)


def test_reconstruct_image_identity_3d():
    """Verifies that extracting patches and reconstructing reproduces the exact original 3D array."""
    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
    rng = np.random.default_rng(42)
    original = rng.uniform(0.0, 1.0, size=(1024, 1024, 3)).astype(np.float32)

    coords = extractor.compute_patch_grid(1024, 1024)
    patches = [extractor.extract_patch(original, x, y) for x, y in coords]
    assert len(patches) == 16

    reconstructed = extractor.reconstruct_image(patches, original_height=1024, original_width=1024, dtype=np.float32)

    assert reconstructed.shape == (1024, 1024, 3)
    np.testing.assert_allclose(reconstructed, original, atol=1e-6)


def test_reconstruct_empty_patches_raises():
    extractor = PatchExtractor(patch_size=256)
    with pytest.raises(ValueError, match="empty list"):
        extractor.reconstruct_image([])
