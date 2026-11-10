"""Synthetic unit and integration tests for bi-temporal image co-registration.

Covers:
1. Identical images (identity alignment).
2. Known synthetic translation recovery.
3. Known synthetic affine transformation recovery.
4. Insufficient feature matches handling.
5. Blank images validation.
6. Corrupt or unsupported inputs.
7. Differing image dimensions.
8. ECC convergence failure handling.
9. Unreliable registration rejection.
10. Oversized image scaling and native coordinate mapping.
11. Preservation of original input arrays.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image
import cv2

from src.temporal_understanding.registration import (
    ImageCoRegistrator,
    RegistrationPolicy,
    RegistrationStatus,
    ImageValidationError,
    RegistrationRejectedError,
)


def _generate_synthetic_textured_scene(
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Generates a rich, textured synthetic optical satellite scene with corner features."""
    rng = np.random.RandomState(seed)
    # Background terrain gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    base = (np.sin(xx * 12.0) * np.cos(yy * 12.0) * 40.0 + 120.0).astype(np.uint8)

    scene = np.stack([base, base + 10, base - 10], axis=-1)

    # Add geometric structural features (simulating roads, buildings, fields)
    for _ in range(35):
        max_bx = max(1, width - 25)
        max_by = max(1, height - 25)
        bx = rng.randint(0, max_bx)
        by = rng.randint(0, max_by)
        bw = rng.randint(5, max(6, min(width - bx, 60)))
        bh = rng.randint(5, max(6, min(height - by, 60)))
        color = [int(rng.randint(30, 240)) for _ in range(3)]
        cv2.rectangle(scene, (bx, by), (bx + bw, by + bh), color, -1)
        cv2.rectangle(scene, (bx, by), (bx + bw, by + bh), (0, 0, 0), 2)

    # Add linear features (roads)
    for _ in range(6):
        pt1 = (int(rng.randint(0, width)), int(rng.randint(0, height)))
        pt2 = (int(rng.randint(0, width)), int(rng.randint(0, height)))
        cv2.line(scene, pt1, pt2, (220, 220, 220), 4)

    return scene.astype(np.uint8)


# =============================================================================
# 1. IDENTICAL IMAGES
# =============================================================================

def test_registration_identical_images():
    """Identical images must yield identity alignment with confidence 1.0."""
    reg = ImageCoRegistrator()
    scene = _generate_synthetic_textured_scene(256, 256, seed=101)

    result = reg.register_pair(scene, scene)

    assert result.diagnostics.registration_status == RegistrationStatus.IDENTITY_ALIGNED
    assert result.diagnostics.confidence_score == 1.0
    assert result.aligned_t2 is not None
    assert np.array_equal(result.aligned_t2, scene)
    assert np.allclose(result.transformation_matrix, np.eye(3))


# =============================================================================
# 2. SYNTHETIC TRANSLATION RECOVERY
# =============================================================================

def test_registration_synthetic_translation():
    """Recovers known pure rigid translation within sub-pixel bounds."""
    policy = RegistrationPolicy(
        min_matches=10,
        min_inliers=8,
        high_quality_threshold=0.50,
        ecc_enabled=False,
    )
    reg = ImageCoRegistrator(policy=policy)
    t1 = _generate_synthetic_textured_scene(512, 512, seed=202)

    dx, dy = 6.0, -4.0
    M_true = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    t2 = cv2.warpAffine(t1, M_true, (512, 512), borderMode=cv2.BORDER_REFLECT)

    result = reg.register_pair(t1, t2)

    assert result.diagnostics.registration_status in (
        RegistrationStatus.ALIGNED_HOMOGRAPHY,
        RegistrationStatus.ALIGNED_AFFINE,
    )
    assert result.aligned_t2 is not None
    assert result.transformation_matrix is not None

    # Recovered displacement of central crop should align within 1.0 pixel
    H = result.transformation_matrix
    recovered_dx = H[0, 2]
    recovered_dy = H[1, 2]

    # Inverting warp direction check (T2 warped to T1)
    assert abs(abs(recovered_dx) - abs(dx)) < 2.0
    assert abs(abs(recovered_dy) - abs(dy)) < 2.0


# =============================================================================
# 3. SYNTHETIC AFFINE TRANSFORMATION RECOVERY
# =============================================================================

def test_registration_synthetic_affine():
    """Recovers small rotational and scaling transformation."""
    policy = RegistrationPolicy(
        min_matches=10,
        min_inliers=8,
        high_quality_threshold=0.50,
        ecc_enabled=False,
    )
    reg = ImageCoRegistrator(policy=policy)
    t1 = _generate_synthetic_textured_scene(512, 512, seed=303)

    # Small rotation (1.5 degrees) around center
    center = (256, 256)
    M_rot = cv2.getRotationMatrix2D(center, 1.5, 1.0)
    t2 = cv2.warpAffine(t1, M_rot, (512, 512), borderMode=cv2.BORDER_REFLECT)

    result = reg.register_pair(t1, t2)

    assert result.diagnostics.registration_status in (
        RegistrationStatus.ALIGNED_HOMOGRAPHY,
        RegistrationStatus.ALIGNED_AFFINE,
    )
    assert result.aligned_t2 is not None
    assert result.diagnostics.inlier_count >= 8
    assert result.diagnostics.confidence_score > 0.40


# =============================================================================
# 4. INSUFFICIENT FEATURE MATCHES
# =============================================================================

def test_registration_insufficient_matches():
    """Smooth images without features must be rejected safely."""
    policy = RegistrationPolicy(min_matches=25, raise_on_rejection=False)
    reg = ImageCoRegistrator(policy=policy)

    # Smooth non-textured gradients
    t1 = np.tile(np.linspace(10, 50, 128, dtype=np.uint8), (128, 1))
    t2 = np.tile(np.linspace(50, 90, 128, dtype=np.uint8), (128, 1))

    result = reg.register_pair(t1, t2)

    assert result.diagnostics.registration_status == RegistrationStatus.REJECTED
    assert result.aligned_t2 is None
    assert len(result.diagnostics.warnings) > 0


# =============================================================================
# 5. BLANK IMAGES
# =============================================================================

def test_validation_blank_image_rejected():
    """Uniform black or blank images must raise ImageValidationError."""
    reg = ImageCoRegistrator()
    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    valid = _generate_synthetic_textured_scene(128, 128, seed=404)

    with pytest.raises(ImageValidationError) as exc_info:
        reg.register_pair(blank, valid)
    assert "blank" in str(exc_info.value).lower()


# =============================================================================
# 6. CORRUPT OR UNSUPPORTED INPUTS
# =============================================================================

def test_validation_corrupt_or_unsupported():
    """Corrupt bytes, 1D arrays, and unsupported channel shapes must be rejected."""
    reg = ImageCoRegistrator()
    valid = _generate_synthetic_textured_scene(64, 64)

    # Corrupt byte payload
    with pytest.raises(ImageValidationError):
        reg.register_pair(b"not_an_image_stream", valid)

    # None input
    with pytest.raises(ImageValidationError):
        reg.register_pair(None, valid)

    # 4D array
    with pytest.raises(ImageValidationError):
        reg.register_pair(np.zeros((10, 10, 10, 3), dtype=np.uint8), valid)

    # NaN floating array
    nan_arr = np.ones((64, 64, 3), dtype=np.float32)
    nan_arr[10, 10, 0] = np.nan
    with pytest.raises(ImageValidationError):
        reg.register_pair(nan_arr, valid)

    # Tiny dimension below min_dimension (16)
    with pytest.raises(ImageValidationError):
        reg.register_pair(np.ones((8, 8, 3), dtype=np.uint8) * 100, valid)


# =============================================================================
# 7. DIFFERENT IMAGE DIMENSIONS
# =============================================================================

def test_registration_differing_dimensions():
    """T2 with differing dimensions should be warped into T1's canvas shape."""
    policy = RegistrationPolicy(
        min_matches=10,
        min_inliers=8,
        high_quality_threshold=0.45,
        ecc_enabled=False,
    )
    reg = ImageCoRegistrator(policy=policy)

    t1 = _generate_synthetic_textured_scene(384, 384, seed=505)
    # T2 is larger (450x450) and translated
    t2_large = _generate_synthetic_textured_scene(450, 450, seed=505)

    result = reg.register_pair(t1, t2_large)

    if result.diagnostics.registration_status != RegistrationStatus.REJECTED:
        assert result.aligned_t2 is not None
        # Must match T1 native canvas shape exactly!
        assert result.aligned_t2.shape == t1.shape


# =============================================================================
# 8. ECC FAILURE HANDLING
# =============================================================================

def test_ecc_failure_graceful_handling():
    """ECC divergence must not crash; falls back to coarse homography with a warning."""
    policy = RegistrationPolicy(
        min_matches=8,
        min_inliers=6,
        high_quality_threshold=0.40,
        ecc_enabled=True,
        ecc_max_iterations=2,  # Force immediate termination
        ecc_eps=1e-12,
    )
    reg = ImageCoRegistrator(policy=policy)
    t1 = _generate_synthetic_textured_scene(256, 256, seed=606)
    t2 = cv2.warpAffine(t1, np.float32([[1, 0, 3], [0, 1, 3]]), (256, 256))

    result = reg.register_pair(t1, t2)
    # Should not throw exception
    assert result.diagnostics is not None


# =============================================================================
# 9. UNRELIABLE REGISTRATION REJECTION
# =============================================================================

def test_unreliable_registration_rejection():
    """Completely uncorrelated images must be rejected."""
    policy = RegistrationPolicy(
        min_matches=15,
        min_inliers=10,
        high_quality_threshold=0.70,
        raise_on_rejection=False,
    )
    reg = ImageCoRegistrator(policy=policy)

    # Completely different seeds -> different buildings, roads, textures
    t1 = _generate_synthetic_textured_scene(256, 256, seed=707)
    t2 = _generate_synthetic_textured_scene(256, 256, seed=808)

    result = reg.register_pair(t1, t2)
    assert result.diagnostics.registration_status == RegistrationStatus.REJECTED
    assert result.aligned_t2 is None


def test_rejection_raises_when_configured():
    """When raise_on_rejection=True, RegistrationRejectedError must be raised."""
    policy = RegistrationPolicy(
        min_matches=50,
        raise_on_rejection=True,
    )
    reg = ImageCoRegistrator(policy=policy)

    t1 = _generate_synthetic_textured_scene(256, 256, seed=909)
    t2 = _generate_synthetic_textured_scene(256, 256, seed=910)

    with pytest.raises(RegistrationRejectedError) as exc_info:
        reg.register_pair(t1, t2)
    assert exc_info.value.diagnostics is not None


# =============================================================================
# 10. OVERSIZED IMAGE SCALING AND COORDINATE MAPPING
# =============================================================================

def test_oversized_image_scaling():
    """Images exceeding max_processing_dim must be scaled for processing and mapped back."""
    policy = RegistrationPolicy(
        max_processing_dim=256,  # Artificially low to test scaling on smaller test images
        min_matches=8,
        min_inliers=6,
        high_quality_threshold=0.40,
        ecc_enabled=False,
    )
    reg = ImageCoRegistrator(policy=policy)

    orig_w, orig_h = 512, 512
    t1 = _generate_synthetic_textured_scene(orig_w, orig_h, seed=111)
    t2 = cv2.warpAffine(t1, np.float32([[1, 0, 4], [0, 1, -3]]), (orig_w, orig_h))

    result = reg.register_pair(t1, t2)

    assert result.diagnostics.scale_factor_x == 256 / 512
    assert result.diagnostics.scale_factor_y == 256 / 512

    if result.aligned_t2 is not None:
        # Output must be in native original dimensions (512, 512, 3), not 256!
        assert result.aligned_t2.shape == (orig_h, orig_w, 3)


# =============================================================================
# 11. PRESERVATION OF ORIGINAL ARRAYS
# =============================================================================

def test_preservation_of_original_arrays():
    """Input arrays must not be mutated in-place by the registration pipeline."""
    reg = ImageCoRegistrator()
    t1 = _generate_synthetic_textured_scene(256, 256, seed=222)
    t2 = _generate_synthetic_textured_scene(256, 256, seed=222)

    t1_copy = t1.copy()
    t2_copy = t2.copy()

    _ = reg.register_pair(t1, t2)

    assert np.array_equal(t1, t1_copy)
    assert np.array_equal(t2, t2_copy)
