"""Unit and integration tests for optical VARI and Groq multimodal semantic analyzer."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from src.temporal_understanding.region_extractor import ChangedRegion
from src.temporal_understanding.semantic_analyzer import (
    APPROVED_CATEGORIES,
    RegionalSemanticInterpretation,
    RegionalVARIEvidence,
    SemanticCategory,
    SemanticChangeAnalyzer,
    VisionAnalyzerConfig,
    analyze_regional_vari,
    compute_vari_map,
    prepare_region_composite_image,
)


# =============================================================================
# 1. OPTICAL VARI TESTS
# =============================================================================

def test_vari_pure_green_and_red():
    """Tests VARI on synthetic pure green and pure red inputs."""
    # Pure Green: R=0, G=255, B=0 -> VARI = (1 - 0) / (1 + 0 - 0) = 1.0
    green_img = np.zeros((30, 30, 3), dtype=np.uint8)
    green_img[:, :, 1] = 255
    vari_map, status, limitations = compute_vari_map(green_img)
    assert status == "supported"
    assert vari_map is not None
    assert np.allclose(vari_map, 1.0, atol=1e-3)

    # Pure Red: R=255, G=0, B=0 -> VARI = (0 - 1) / (0 + 1 - 0) = -1.0
    red_img = np.zeros((30, 30, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255
    vari_map_red, status_red, _ = compute_vari_map(red_img)
    assert status_red == "supported"
    assert vari_map_red is not None
    assert np.allclose(vari_map_red, -1.0, atol=1e-3)


def test_vari_zero_and_near_zero_denominator():
    """Tests that zero and near-zero denominators (Green + Red - Blue = 0) are handled safely."""
    zero_denom_img = np.zeros((20, 20, 3), dtype=np.uint8)
    zero_denom_img[:, :, 1] = 100  # G = 100
    zero_denom_img[:, :, 0] = 0    # R = 0
    zero_denom_img[:, :, 2] = 100  # B = 100 -> G + R - B = 0

    vari_map, status, _ = compute_vari_map(zero_denom_img, eps=1e-6)
    assert status == "supported"
    assert vari_map is not None
    assert not np.isnan(vari_map).any()
    assert not np.isinf(vari_map).any()
    assert (vari_map >= -1.0).all() and (vari_map <= 1.0).all()


def test_vari_invalid_values_nan_inf():
    """Tests that NaN and Inf values in input float images are safely rejected."""
    nan_img = np.ones((20, 20, 3), dtype=np.float32) * 0.5
    nan_img[5, 5, 0] = np.nan
    nan_img[10, 10, 1] = np.inf

    vari_map, status, limitations = compute_vari_map(nan_img)
    assert status == "invalid_values"
    assert vari_map is None
    assert any("NaN or infinite values" in l for l in limitations)


def test_vari_missing_or_unsupported_channels():
    """Tests that images lacking exactly 3 RGB channels return an unsupported status."""
    # 2D grayscale image
    gray_img = np.zeros((30, 30), dtype=np.uint8)
    vari_map, status, limitations = compute_vari_map(gray_img)
    assert status == "unsupported_channels"
    assert vari_map is None
    assert any("Expected" in l or "lacks 3-channel RGB" in l for l in limitations)

    # 4-channel image (e.g. RGBA or multispectral)
    rgba_img = np.zeros((30, 30, 4), dtype=np.uint8)
    vari_map, status, limitations = compute_vari_map(rgba_img)
    assert status == "unsupported_channels"
    assert vari_map is None


def test_vari_illumination_and_dynamic_range():
    """Tests that low-dynamic-range / underexposed images flag illumination warnings."""
    # Dark image with non-zero variance but low mean intensity (~15/255)
    rng = np.random.default_rng(42)
    dark_img = rng.integers(10, 25, size=(30, 30, 3), dtype=np.uint8)
    vari_map, status, limitations = compute_vari_map(dark_img)
    assert status == "supported"
    assert any("Low dynamic range or underexposed region" in l for l in limitations)


def test_analyze_regional_vari():
    """Tests regional VARI comparison between T1 (barren) and T2 (vegetated)."""
    h, w = 40, 40
    # T1: Barren soil / red-tinted (R=180, G=100, B=80)
    t1 = np.zeros((h, w, 3), dtype=np.uint8)
    t1[:, :, 0] = 180
    t1[:, :, 1] = 100
    t1[:, :, 2] = 80

    # T2: Green vegetation (R=50, G=200, B=50)
    t2 = np.zeros((h, w, 3), dtype=np.uint8)
    t2[:, :, 0] = 50
    t2[:, :, 1] = 200
    t2[:, :, 2] = 50

    # Mask covering central 20x20
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:30, 10:30] = 255

    evidence = analyze_regional_vari(t1, t2, mask)
    assert evidence.status == "supported"
    assert evidence.t1_mean_vari is not None
    assert evidence.t2_mean_vari is not None
    assert evidence.delta_vari is not None
    assert evidence.delta_vari > 0.2
    assert "Vegetation greenness proxy increase" in evidence.interpretation
    assert any("NOT calibrated multispectral NDVI" in l for l in evidence.limitations)


# =============================================================================
# 2. IMAGE COMPOSITING & PADDING TESTS
# =============================================================================

def test_prepare_region_composite_image_min_dimension_enforcement():
    """Tests that tiny crops (< 32px) are scaled up to >= 32px for Groq Qwen vision compatibility."""
    crop_t1 = np.full((10, 10, 3), 100, dtype=np.uint8)
    crop_t2 = np.full((10, 10, 3), 200, dtype=np.uint8)
    crop_mask = np.full((10, 10), 255, dtype=np.uint8)

    b64_data = prepare_region_composite_image(crop_t1, crop_t2, crop_mask, min_dim=32)
    assert b64_data != ""

    raw_bytes = base64.b64decode(b64_data)
    import cv2
    img_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert img_array is not None
    h, w, _ = img_array.shape
    assert h >= 32
    assert w >= 32 * 3


# =============================================================================
# 3. GROQ MULTIMODAL VISION MODEL INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def dummy_region():
    """Creates a sample ChangedRegion object conforming to RegionExtractor data contract."""
    return ChangedRegion(
        region_id=1,
        bbox=(10, 10, 30, 30),
        context_bbox=(5, 5, 35, 35),
        pixel_area=400,
        raw_pixel_area=400,
        area_ratio=0.01,
        centroid=(20.0, 20.0),
        bbox_width=20,
        bbox_height=20,
        mean_probability=0.92,
        gsd_meters=0.5,
        physical_area_sq_meters=100.0,
    )


def test_parse_vision_response_valid_json():
    """Tests parsing a valid JSON response from Groq multimodal model."""
    analyzer = SemanticChangeAnalyzer(VisionAnalyzerConfig(groq_api_key="mock_key"))
    valid_payload = json.dumps({
        "change_category": "new_construction",
        "visual_observation": "Rectangular concrete foundation and structural frame visible at T2.",
        "evidence_for_category": ["High contrast rectilinear boundaries", "Bright roofing material"],
        "alternative_interpretations": ["Temporary storage depot"],
        "uncertainty_reason": "Low oblique shadow angle limits height estimation",
        "requires_human_review": False
    })

    result = analyzer._parse_vision_response(valid_payload, 1)
    assert result["region_id"] == 1
    assert result["change_category"] == SemanticCategory.NEW_CONSTRUCTION.value
    assert "foundation" in result["visual_observation"]
    assert result["requires_human_review"] is False


def test_parse_vision_response_markdown_fences():
    """Tests parsing markdown ```json fences commonly produced by LLMs."""
    analyzer = SemanticChangeAnalyzer(VisionAnalyzerConfig(groq_api_key="mock_key"))
    fenced_payload = (
        "Here is the evaluation:\n"
        "```json\n"
        "{\n"
        '  "change_category": "road_development",\n'
        '  "visual_observation": "Linear corridor clearing with asphalt paving.",\n'
        '  "evidence_for_category": ["Linear geometry", "Connecting two arterial roads"],\n'
        '  "alternative_interpretations": ["Firebreak clearing"],\n'
        '  "uncertainty_reason": null,\n'
        '  "requires_human_review": false\n'
        "}\n"
        "```"
    )
    result = analyzer._parse_vision_response(fenced_payload, 2)
    assert result["change_category"] == SemanticCategory.ROAD_DEVELOPMENT.value


def test_parse_vision_response_unsupported_category():
    """Tests that an unsupported category is safely mapped to 'uncertain'."""
    analyzer = SemanticChangeAnalyzer(VisionAnalyzerConfig(groq_api_key="mock_key"))
    invalid_category_payload = json.dumps({
        "change_category": "alien_spacecraft_landing",
        "visual_observation": "Unidentified circular pattern.",
        "evidence_for_category": [],
        "alternative_interpretations": [],
        "uncertainty_reason": "Out of domain taxonomy",
        "requires_human_review": True
    })

    result = analyzer._parse_vision_response(invalid_category_payload, 3)
    assert result["change_category"] == SemanticCategory.UNCERTAIN.value
    assert result["requires_human_review"] is True


def test_parse_vision_response_malformed_json():
    """Tests that malformed JSON triggers safe uncertain fallback."""
    analyzer = SemanticChangeAnalyzer(VisionAnalyzerConfig(groq_api_key="mock_key"))
    malformed_payload = "This is definitely not a JSON object { invalid: "

    result = analyzer._parse_vision_response(malformed_payload, 4)
    assert result["change_category"] == SemanticCategory.UNCERTAIN.value
    assert "Model response was empty or did not conform" in (result["uncertainty_reason"] or "")
    assert result["requires_human_review"] is True


def test_analyze_regions_missing_api_key(dummy_region):
    """Tests that an unconfigured API key yields safe fallback without making API calls."""
    analyzer = SemanticChangeAnalyzer(VisionAnalyzerConfig(groq_api_key=""))
    t1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    t2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 255

    results = analyzer.analyze_regions(t1, t2, [dummy_region], mask)
    assert len(results) == 1
    assert results[0].change_category == SemanticCategory.UNCERTAIN.value
    assert "Vision model inference unavailable" in (results[0].uncertainty_reason or "")
    assert results[0].requires_human_review is True
    assert results[0].vari_evidence is not None


def test_analyze_regions_successful_mocked_groq_call(dummy_region):
    """Tests end-to-end regional analysis with mocked Groq multimodal API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "change_category": "vegetation_decrease",
                        "visual_observation": "Tree canopy removed leaving bare soil.",
                        "evidence_for_category": ["Loss of green texture", "Exposed brown subsoil"],
                        "alternative_interpretations": ["Seasonal senescence"],
                        "uncertainty_reason": "No multi-temporal NIR confirmation",
                        "requires_human_review": False
                    })
                }
            }
        ]
    }

    t1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    t2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 255

    with patch("httpx.Client.post", return_value=mock_response):
        analyzer = SemanticChangeAnalyzer(
            VisionAnalyzerConfig(groq_api_key="gsk_mock_valid_key", vision_model="qwen/qwen3.8-27b")
        )
        results = analyzer.analyze_regions(t1, t2, [dummy_region], mask)

        assert len(results) == 1
        res = results[0]
        assert res.change_category == "vegetation_decrease"
        assert "canopy" in res.visual_observation
        assert res.vari_evidence is not None


def test_analyze_regions_rate_limit_429(dummy_region):
    """Tests that HTTP 429 rate limit triggers safe uncertain fallback without crashing."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    t1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    t2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    with patch("httpx.Client.post", return_value=mock_response):
        analyzer = SemanticChangeAnalyzer(
            VisionAnalyzerConfig(groq_api_key="gsk_mock_valid_key", max_retries=1)
        )
        results = analyzer.analyze_regions(t1, t2, [dummy_region], mask)

        assert len(results) == 1
        assert results[0].change_category == SemanticCategory.UNCERTAIN.value
        assert results[0].requires_human_review is True


def test_analyze_regions_timeout(dummy_region):
    """Tests that network timeout returns safe uncertain fallback without crashing."""
    t1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    t2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
        analyzer = SemanticChangeAnalyzer(
            VisionAnalyzerConfig(groq_api_key="gsk_mock_valid_key", max_retries=1)
        )
        results = analyzer.analyze_regions(t1, t2, [dummy_region], mask)

        assert len(results) == 1
        assert results[0].change_category == SemanticCategory.UNCERTAIN.value
        assert results[0].requires_human_review is True


def test_region_count_limit():
    """Tests that regions analyzed by vision model are limited to max_regions_to_analyze (default 6)."""
    regions = []
    for i in range(10):
        regions.append(
            ChangedRegion(
                region_id=i + 1,
                bbox=(0, 0, 32, 32),
                context_bbox=(0, 0, 32, 32),
                pixel_area=1000 - i * 50,
                raw_pixel_area=1000 - i * 50,
                area_ratio=0.01,
                centroid=(16.0, 16.0),
                bbox_width=32,
                bbox_height=32,
                mean_probability=0.9,
                gsd_meters=0.5,
                physical_area_sq_meters=100.0,
            )
        )

    t1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    t2 = np.full((100, 100, 3), 200, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)

    analyzer = SemanticChangeAnalyzer(
        VisionAnalyzerConfig(groq_api_key="", max_regions_to_analyze=6)
    )
    results = analyzer.analyze_regions(t1, t2, regions, mask)
    assert len(results) == 6
    assert [r.region_id for r in results] == [1, 2, 3, 4, 5, 6]
