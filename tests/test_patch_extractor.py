"""Unit tests for PatchExtractor."""

import numpy as np
import pytest

from src.data.patch_extractor import PatchExtractor


def test_grid_computation_non_overlapping():
    extractor = PatchExtractor(patch_size=256, stride=256, padding_mode="drop")
    grid = extractor.compute_patch_grid(width=1024, height=1024)
    # In 1024x1024 with 256 stride: 4x4 = 16 patches
    assert len(grid) == 16
    assert grid[0] == (0, 0)
    assert grid[-1] == (768, 768)


def test_grid_computation_overlapping():
    extractor = PatchExtractor(patch_size=256, stride=128, padding_mode="drop")
    grid = extractor.compute_patch_grid(width=1024, height=1024)
    # (1024 - 256) // 128 + 1 = 7 steps each dimension -> 7 x 7 = 49
    assert len(grid) == 49
    assert extractor.is_overlapping is True


def test_pair_patch_synchronization():
    extractor = PatchExtractor(patch_size=256, stride=256)

    img_a = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img_b = np.zeros((1024, 1024, 3), dtype=np.uint8)
    label = np.zeros((1024, 1024), dtype=np.uint8)

    # Place a change in patch (x=256, y=256)
    label[260:300, 260:300] = 255  # 40x40 = 1600 pixels

    patches = extractor.extract_from_pair(img_a, img_b, label, parent_stem="test_scene")
    assert len(patches) == 16

    # Find patch at (x=256, y=256)
    target_patch = next(p for p in patches if p[3].x == 256 and p[3].y == 256)
    meta = target_patch[3]
    assert meta.changed_pixel_count == 1600
    assert meta.changed_ratio == pytest.approx(1600 / (256 * 256))

    # All other patches should have 0 changed pixels
    zero_patches = [p for p in patches if p[3].x != 256 or p[3].y != 256]
    for p in zero_patches:
        assert p[3].changed_pixel_count == 0


def test_padding_mode():
    extractor = PatchExtractor(patch_size=256, padding_mode="pad")
    # 300x300 image: ceil((300-256)/256) = 1 -> 2x2 = 4 patches
    grid = extractor.compute_patch_grid(width=300, height=300)
    assert len(grid) == 4

    arr = np.ones((300, 300, 3), dtype=np.uint8)
    patch = extractor.extract_patch(arr, x=256, y=256)
    assert patch.shape == (256, 256, 3)
