"""Synthetic unit tests for RegionExtractor and spatial clustering.

Covers:
1. Empty mask handling.
2. Full mask handling.
3. Single connected component extraction.
4. Multiple disconnected components.
5. Small-noise filtering (absolute pixel count).
6. Relative area threshold filtering.
7. Morphological cleanup (closing & opening).
8. Contextual bounding-box padding and boundary clipping.
9. Top-K selection and area-descending ranking.
10. Invalid mask dimensions and dtypes.
11. Continuous probability map aggregation.
12. Preservation of raw input mask.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.temporal_understanding.region_extractor import (
    RegionExtractor,
    RegionExtractionConfig,
    RegionExtractionError,
    ChangedRegion,
)


# =============================================================================
# 1. EMPTY MASK
# =============================================================================

def test_extract_empty_mask():
    """Empty mask (all zeros) must return zero regions without errors."""
    extractor = RegionExtractor()
    mask = np.zeros((256, 256), dtype=np.uint8)

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 0
    assert result.total_candidates_found == 0
    assert result.total_regions_retained == 0
    assert result.total_changed_pixels_raw == 0
    assert result.total_changed_pixels_clean == 0
    assert result.mask_width == 256
    assert result.mask_height == 256


# =============================================================================
# 2. FULL MASK
# =============================================================================

def test_extract_full_mask():
    """Full mask (all ones) must produce a single comprehensive region."""
    extractor = RegionExtractor()
    mask = np.ones((100, 100), dtype=np.uint8)

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 1
    reg = result.regions[0]
    assert reg.region_id == 1
    assert reg.pixel_area == 100 * 100
    assert reg.bbox == (0, 0, 100, 100)
    assert reg.context_bbox == (0, 0, 100, 100)
    assert abs(reg.area_ratio - 1.0) < 1e-4


# =============================================================================
# 3. SINGLE CONNECTED COMPONENT
# =============================================================================

def test_extract_single_component():
    """Extracts exact coordinates and area for a single rectangular block."""
    config = RegionExtractionConfig(
        min_pixel_area=50,
        min_area_ratio=0.001,
        morphology_closing_kernel=1,
        morphology_opening_kernel=1,
    )
    extractor = RegionExtractor(config=config)

    mask = np.zeros((200, 200), dtype=np.uint8)
    # Box from y: 30..80 (h=50), x: 40..100 (w=60) -> area = 3000 px
    mask[30:80, 40:100] = 1

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 1
    reg = result.regions[0]
    assert reg.region_id == 1
    assert reg.bbox == (40, 30, 100, 80)
    assert reg.pixel_area == 3000
    assert reg.bbox_width == 60
    assert reg.bbox_height == 50
    assert abs(reg.centroid[0] - 69.5) < 1.0
    assert abs(reg.centroid[1] - 54.5) < 1.0


# =============================================================================
# 4. MULTIPLE DISCONNECTED COMPONENTS
# =============================================================================

def test_extract_multiple_components():
    """Extracts multiple distinct components correctly."""
    config = RegionExtractionConfig(
        min_pixel_area=50,
        min_area_ratio=0.0001,
        morphology_closing_kernel=1,
        morphology_opening_kernel=1,
    )
    extractor = RegionExtractor(config=config)

    mask = np.zeros((300, 300), dtype=np.uint8)
    # Component 1 (Area = 20x20 = 400)
    mask[20:40, 20:40] = 1
    # Component 2 (Area = 40x40 = 1600)
    mask[100:140, 100:140] = 1
    # Component 3 (Area = 30x30 = 900)
    mask[200:230, 200:230] = 1

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 3
    # Check descending area ranking: Component 2 (1600), Component 3 (900), Component 1 (400)
    assert result.regions[0].pixel_area == 1600
    assert result.regions[0].region_id == 1
    assert result.regions[1].pixel_area == 900
    assert result.regions[1].region_id == 2
    assert result.regions[2].pixel_area == 400
    assert result.regions[2].region_id == 3


# =============================================================================
# 5. SMALL-NOISE FILTERING (ABSOLUTE PIXEL THRESHOLD)
# =============================================================================

def test_small_noise_filtering():
    """Components below min_pixel_area must be discarded as noise."""
    config = RegionExtractionConfig(min_pixel_area=200)
    extractor = RegionExtractor(config=config)

    mask = np.zeros((500, 500), dtype=np.uint8)
    # Valid large region: 20x20 = 400 px
    mask[50:70, 50:70] = 1
    # Noise cluster: 8x8 = 64 px (< 200)
    mask[200:208, 200:208] = 1
    # Isolated scattered single pixels
    mask[10, 10] = 1
    mask[20, 30] = 1

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 1
    assert result.regions[0].pixel_area >= 400
    assert result.filtered_small_regions_count >= 1


# =============================================================================
# 6. RELATIVE AREA THRESHOLD
# =============================================================================

def test_relative_area_threshold():
    """Components below min_area_ratio (proportion of image canvas) must be filtered."""
    # 1000x1000 = 1,000,000 pixels. min_area_ratio = 0.001 -> min 1,000 pixels
    config = RegionExtractionConfig(min_pixel_area=10, min_area_ratio=0.001)
    extractor = RegionExtractor(config=config)

    mask = np.zeros((1000, 1000), dtype=np.uint8)
    # 20x20 = 400 px (ratio = 0.0004 < 0.001) -> must be filtered
    mask[100:120, 100:120] = 1
    # 40x40 = 1600 px (ratio = 0.0016 >= 0.001) -> retained
    mask[500:540, 500:540] = 1

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 1
    assert result.regions[0].pixel_area == 1600


# =============================================================================
# 7. MORPHOLOGICAL CLEANUP
# =============================================================================

def test_morphological_closing_and_opening():
    """Morphological closing bridges 1-pixel gap; opening removes 1-pixel spur."""
    config = RegionExtractionConfig(
        min_pixel_area=50,
        morphology_closing_kernel=3,
        morphology_opening_kernel=3,
    )
    extractor = RegionExtractor(config=config)

    mask = np.zeros((100, 100), dtype=np.uint8)
    # Two 10x10 blocks separated by a 1-pixel gap
    mask[20:30, 20:30] = 1
    mask[20:30, 31:41] = 1

    result = extractor.extract_regions(mask)

    # Closing with 3x3 kernel bridges the 1-pixel gap into a single unified region!
    assert len(result.regions) == 1
    assert result.regions[0].bbox_width >= 20


# =============================================================================
# 8. CONTEXTUAL PADDING AND BOUNDARY CLIPPING
# =============================================================================

def test_padding_and_boundary_clipping():
    """Contextual bounding box expands by padding ratio and clips strictly to [0, W] and [0, H]."""
    config = RegionExtractionConfig(
        min_pixel_area=50,
        context_padding_ratio=0.20,  # 20% margin
        morphology_closing_kernel=1,
        morphology_opening_kernel=1,
    )
    extractor = RegionExtractor(config=config)

    mask = np.zeros((200, 200), dtype=np.uint8)
    # Region placed right at the top-left corner: x: 0..50, y: 0..40
    mask[0:40, 0:50] = 1

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 1
    reg = result.regions[0]
    assert reg.bbox == (0, 0, 50, 40)
    # Context bbox must clip at 0, not negative!
    c_xmin, c_ymin, c_xmax, c_ymax = reg.context_bbox
    assert c_xmin == 0
    assert c_ymin == 0
    # Right and bottom padded by 20%
    assert c_xmax == 50 + int(round(50 * 0.20))
    assert c_ymax == 40 + int(round(40 * 0.20))


# =============================================================================
# 9. TOP-K SELECTION AND RANKING
# =============================================================================

def test_top_k_selection():
    """When more candidates exist than max_regions, strictly return top-K largest."""
    config = RegionExtractionConfig(
        min_pixel_area=20,
        max_regions=3,  # Cap at top 3
        morphology_closing_kernel=1,
        morphology_opening_kernel=1,
    )
    extractor = RegionExtractor(config=config)

    mask = np.zeros((400, 400), dtype=np.uint8)
    # Create 5 components with areas: 50, 100, 200, 400, 800
    mask[10:20, 10:15] = 1    # 10x5 = 50
    mask[40:50, 40:50] = 1    # 10x10 = 100
    mask[80:100, 80:90] = 1   # 20x10 = 200
    mask[130:150, 130:150] = 1 # 20x20 = 400
    mask[200:240, 200:220] = 1 # 40x20 = 800

    result = extractor.extract_regions(mask)

    assert len(result.regions) == 3
    assert result.total_candidates_found == 5
    assert result.total_regions_retained == 3
    # Top 3 must be 800, 400, 200
    assert result.regions[0].pixel_area == 800
    assert result.regions[0].region_id == 1
    assert result.regions[1].pixel_area == 400
    assert result.regions[1].region_id == 2
    assert result.regions[2].pixel_area == 200
    assert result.regions[2].region_id == 3


# =============================================================================
# 10. INVALID MASK DIMENSIONS AND DTYPES
# =============================================================================

def test_invalid_mask_dimensions_and_dtypes():
    """Rejects non-2D, empty, or non-numpy arrays with descriptive error."""
    extractor = RegionExtractor()

    # 3D mask
    with pytest.raises(RegionExtractionError):
        extractor.extract_regions(np.zeros((10, 10, 3)))

    # 1D mask
    with pytest.raises(RegionExtractionError):
        extractor.extract_regions(np.zeros((100,)))

    # Non-numpy
    with pytest.raises(RegionExtractionError):
        extractor.extract_regions([[0, 1], [1, 0]])  # type: ignore

    # Zero dimension
    with pytest.raises(RegionExtractionError):
        extractor.extract_regions(np.zeros((0, 100)))


# =============================================================================
# 11. PROBABILITY MAP AGGREGATION
# =============================================================================

def test_probability_map_aggregation():
    """Mean probability inside component must equal average of continuous scores."""
    config = RegionExtractionConfig(
        min_pixel_area=50,
        morphology_closing_kernel=1,
        morphology_opening_kernel=1,
    )
    extractor = RegionExtractor(config=config)

    mask = np.zeros((100, 100), dtype=np.uint8)
    prob = np.zeros((100, 100), dtype=np.float32)

    # 10x10 block = 100 pixels
    mask[20:30, 20:30] = 1
    # Set half to 0.80 and half to 0.90 -> mean = 0.85
    prob[20:25, 20:30] = 0.80
    prob[25:30, 20:30] = 0.90

    result = extractor.extract_regions(mask, probability_map=prob)

    assert len(result.regions) == 1
    reg = result.regions[0]
    assert reg.mean_probability is not None
    assert abs(reg.mean_probability - 0.85) < 1e-4


# =============================================================================
# 12. PRESERVATION OF RAW MASK
# =============================================================================

def test_preservation_of_raw_mask():
    """The raw mask in the result object must exactly equal the input mask."""
    extractor = RegionExtractor()
    mask = np.zeros((150, 150), dtype=np.uint8)
    mask[30:60, 30:60] = 1
    mask_copy = mask.copy()

    result = extractor.extract_regions(mask)

    assert np.array_equal(result.raw_mask, mask_copy)
    assert np.array_equal(mask, mask_copy)
