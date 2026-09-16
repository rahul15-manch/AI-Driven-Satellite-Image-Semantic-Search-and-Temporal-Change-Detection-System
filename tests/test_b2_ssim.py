"""Unit tests for Method B2: SSIM Dissimilarity."""

import numpy as np
import pytest

from src.change_detection.ssim_detector import SSIMDetector


def test_ssim_parameters_and_defaults():
    detector = SSIMDetector()
    assert detector.win_size == 11
    assert detector.sigma == 1.5
    assert detector.channel_mode == "grayscale"

    with pytest.raises(ValueError, match="odd integer"):
        SSIMDetector(win_size=10)

    with pytest.raises(ValueError, match="positive"):
        SSIMDetector(sigma=-1.0)


def test_ssim_identical_images():
    detector = SSIMDetector(win_size=11, sigma=1.5)
    img = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    diff = detector.compute_difference_map(img, img)

    assert diff.shape == (64, 64)
    # Identical images have SSIM = 1.0 -> dissimilarity = 0.0
    assert np.allclose(diff, 0.0, atol=1e-4)


def test_ssim_inverted_structural_pattern():
    detector = SSIMDetector(win_size=11, sigma=1.5)
    # Checkerboard pattern
    pattern1 = np.indices((64, 64)).sum(axis=0) % 2
    pattern1 = np.repeat(pattern1[:, :, None], 3, axis=2).astype(np.float32)
    pattern2 = 1.0 - pattern1

    diff = detector.compute_difference_map(pattern1, pattern2)
    # Highly dissimilar structural patterns should yield large dissimilarity
    assert np.mean(diff) > 0.7
    assert np.all(diff >= 0.0)
    assert np.all(diff <= 1.0)


def test_ssim_small_image_raises():
    detector = SSIMDetector(win_size=11)
    img1 = np.zeros((8, 8, 3), dtype=np.float32)
    img2 = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be at least win_size"):
        detector.compute_difference_map(img1, img2)
