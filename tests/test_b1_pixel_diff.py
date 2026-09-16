"""Unit tests for Method B1: Absolute Pixel Differencing."""

import numpy as np
import pytest

from src.change_detection.pixel_diff import PixelDiffDetector


def test_pixel_diff_identical_images():
    detector = PixelDiffDetector(aggregation_mode="mean")
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    diff = detector.compute_difference_map(img, img)

    assert diff.shape == (256, 256)
    assert diff.dtype == np.float32
    assert np.all(diff == 0.0)


def test_pixel_diff_maximum_inversion():
    detector = PixelDiffDetector(aggregation_mode="mean")
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2 = np.ones((256, 256, 3), dtype=np.uint8) * 255
    diff = detector.compute_difference_map(img1, img2)

    assert np.allclose(diff, 1.0, atol=1e-4)


def test_pixel_diff_aggregation_modes():
    detector_mean = PixelDiffDetector(aggregation_mode="mean")
    detector_max = PixelDiffDetector(aggregation_mode="max")

    # Only red channel changes by 1.0 (255)
    img1 = np.zeros((10, 10, 3), dtype=np.float32)
    img2 = np.zeros((10, 10, 3), dtype=np.float32)
    img2[:, :, 0] = 1.0

    diff_mean = detector_mean.compute_difference_map(img1, img2)
    diff_max = detector_max.compute_difference_map(img1, img2)

    assert np.allclose(diff_mean, 1.0 / 3.0, atol=1e-5)
    assert np.allclose(diff_max, 1.0, atol=1e-5)


def test_pixel_diff_predict_binary_mask():
    detector = PixelDiffDetector()
    img1 = np.zeros((10, 10, 3), dtype=np.float32)
    img2 = np.ones((10, 10, 3), dtype=np.float32) * 0.5  # diff is 0.5

    # Threshold 0.4 -> should be 1
    mask, diff = detector.predict(img1, img2, threshold=0.4)
    assert np.all(mask == 1)
    assert np.allclose(diff, 0.5)

    # Threshold 0.6 -> should be 0
    mask, _ = detector.predict(img1, img2, threshold=0.6)
    assert np.all(mask == 0)


def test_pixel_diff_shape_mismatch_raises():
    detector = PixelDiffDetector()
    img1 = np.zeros((10, 10, 3), dtype=np.float32)
    img2 = np.zeros((12, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        detector.compute_difference_map(img1, img2)
