"""Unit tests for Method B3: Change Vector Analysis (CVA)."""

import numpy as np
import pytest

from src.change_detection.cva_detector import CVADetector


def test_cva_identical_images():
    detector = CVADetector()
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    diff = detector.compute_difference_map(img, img)

    assert diff.shape == (64, 64)
    assert np.all(diff == 0.0)


def test_cva_maximum_inversion():
    detector = CVADetector(normalize=True)
    img1 = np.zeros((32, 32, 3), dtype=np.float32)
    img2 = np.ones((32, 32, 3), dtype=np.float32)

    diff = detector.compute_difference_map(img1, img2)
    # sqrt(1^2 + 1^2 + 1^2) / sqrt(3) = 1.0
    assert np.allclose(diff, 1.0, atol=1e-5)


def test_cva_single_channel_step():
    detector = CVADetector(normalize=True)
    img1 = np.zeros((10, 10, 3), dtype=np.float32)
    img2 = np.zeros((10, 10, 3), dtype=np.float32)
    img2[:, :, 0] = 1.0  # R channel step of 1.0

    diff = detector.compute_difference_map(img1, img2)
    # sqrt(1^2 + 0 + 0) / sqrt(3) = 1 / sqrt(3)
    expected = 1.0 / np.sqrt(3.0)
    assert np.allclose(diff, expected, atol=1e-5)


def test_cva_unnormalized():
    detector = CVADetector(normalize=False)
    img1 = np.zeros((10, 10, 3), dtype=np.float32)
    img2 = np.ones((10, 10, 3), dtype=np.float32)

    diff = detector.compute_difference_map(img1, img2)
    # sqrt(3) without 1/sqrt(3) scaling, clipped to 1.0
    assert np.allclose(diff, 1.0)
