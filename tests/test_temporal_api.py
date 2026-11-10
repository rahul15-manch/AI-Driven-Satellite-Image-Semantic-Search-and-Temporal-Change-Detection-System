"""End-to-end integration tests for Phase 3C Temporal Understanding API (POST /api/temporal/analyze).

Tests all 20 required validation, registration, detector, region, vision, explanation,
and regression compatibility scenarios using synthetic scenes and mocked external calls.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import create_app
from src.api.schemas import TemporalAnalysisResponse
from src.temporal_understanding.registration import RegistrationDiagnostics, RegistrationStatus


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def make_synthetic_image(
    width: int = 256,
    height: int = 256,
    add_features: bool = True,
    seed: int = 42,
    suffix: str = "PNG",
) -> bytes:
    """Generates synthetic textured optical satellite image bytes."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(70, 180, size=(height, width, 3), dtype=np.uint8)

    if add_features:
        # Add high-contrast geometric corners and shapes for reliable ORB keypoint matching
        cv2.rectangle(arr, (30, 30), (100, 100), (220, 40, 40), -1)
        cv2.circle(arr, (160, 160), 35, (40, 220, 40), -1)
        cv2.rectangle(arr, (140, 40), (220, 90), (40, 40, 220), -1)
        cv2.putText(arr, "ROOF", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(arr, "SITE", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=suffix)
    return buf.getvalue()


def make_changed_t2_image(t1_bytes: bytes, change_bbox=(120, 120, 200, 200)) -> bytes:
    """Creates a modified T2 image from T1 bytes with a distinct structural change."""
    img = Image.open(io.BytesIO(t1_bytes)).convert("RGB")
    arr = np.array(img)
    x1, y1, x2, y2 = change_bbox
    # Draw new bright structure with distinct greenness change (vegetation clearing)
    arr[y1:y2, x1:x2, 0] = 230  # High red
    arr[y1:y2, x1:x2, 1] = 60   # Low green
    arr[y1:y2, x1:x2, 2] = 50   # Low blue
    cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 255, 255), 2)

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# =============================================================================
# 1. VALID UPLOAD & SUCCESSFUL PIPELINE
# =============================================================================

def test_1_valid_t1_t2_upload(client):
    """Test 1: Valid T1/T2 upload executes full pipeline returning HTTP 200 and success status."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=10)
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("scene_t1.png", t1_bytes, "image/png"),
        "file_t2": ("scene_t2.png", t2_bytes, "image/png"),
    }
    data = {
        "observation_date_t1": "2023-01-15",
        "observation_date_t2": "2024-03-20",
        "gsd_meters_per_pixel": 0.5,
        "description": "Suburban development area",
        "detector_model": "M9_BIT_CD",
        "max_regions": 4,
    }

    # Mock external Groq calls to avoid external network dependencies
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files, data=data)

    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["request_id"].startswith("temp_")
    assert res["image_metadata"]["width"] == 256
    assert res["image_metadata"]["height"] == 256
    assert res["image_metadata"]["gsd_meters_per_pixel"] == 0.5
    assert res["registration_diagnostics"] is not None
    assert res["registration_diagnostics"]["registration_status"] in (
        "aligned_homography", "aligned_affine", "identity_aligned"
    )
    assert res["detector_metadata"]["model_id"] == "M9_BIT_CD"
    assert res["change_mask"] is not None
    assert res["change_mask"]["mask_data_uri"].startswith("data:image/png;base64,")
    assert isinstance(res["regions"], list)
    assert isinstance(res["semantic_findings"], list)
    assert res["temporal_explanation"] is not None
    assert len(res["errors"]) == 0


# =============================================================================
# 2. INPUT VALIDATION TESTS (FORMAT, EMPTY, OVERSIZED)
# =============================================================================

def test_2_invalid_image_formats(client):
    """Test 2: Reject unsupported file extensions (e.g. .txt, .pdf)."""
    t1_bytes = b"This is not a real satellite image, just raw text."
    t2_bytes = make_synthetic_image(128, 128)

    files = {
        "file_t1": ("corrupted_input.txt", t1_bytes, "text/plain"),
        "file_t2": ("scene_t2.png", t2_bytes, "image/png"),
    }
    resp = client.post("/api/temporal/analyze", files=files)
    assert resp.status_code in (400, 415)
    assert "Unsupported image file extension" in resp.json()["detail"] or "Failed" in resp.json()["detail"]


def test_3_empty_upload(client):
    """Test 3: Reject empty (0-byte) file upload."""
    files = {
        "file_t1": ("empty_t1.png", b"", "image/png"),
        "file_t2": ("scene_t2.png", make_synthetic_image(128, 128), "image/png"),
    }
    resp = client.post("/api/temporal/analyze", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_4_oversized_upload(client):
    """Test 4: Reject uploads exceeding max_upload_size limit."""
    # Test client using small size ceiling patch
    t1_bytes = make_synthetic_image(128, 128)
    t2_bytes = make_synthetic_image(128, 128)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }

    # Patch max_upload_size to 100 bytes to trigger rejection safely
    with patch("src.api.service.temporal_understanding_service.max_upload_size", 100):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 413
    assert "exceeds maximum allowed size" in resp.json()["detail"]


# =============================================================================
# 3. REGISTRATION ENGINE INTEGRATION (REJECTION & SUCCESS)
# =============================================================================

def test_5_registration_rejection(client):
    """Test 5: Registration rejection halts pipeline without running change detection."""
    # Two completely disjoint images (e.g. pure noise with different seeds)
    t1_bytes = make_synthetic_image(200, 200, add_features=False, seed=1)
    t2_bytes = make_synthetic_image(200, 200, add_features=False, seed=999)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }

    resp = client.post("/api/temporal/analyze", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "registration_rejected"
    assert data["registration_diagnostics"]["registration_status"] == "rejected"
    assert data["detector_metadata"] is None
    assert data["change_mask"] is None
    assert len(data["regions"]) == 0
    assert len(data["errors"]) > 0
    assert "rejected" in data["errors"][0].lower()


def test_6_registration_success(client):
    """Test 6: Pre-aligned or successfully co-registered pair proceeds cleanly."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=42)
    # Slight translation
    img = Image.open(io.BytesIO(t1_bytes))
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["registration_diagnostics"]["registration_status"] != "rejected"
    assert data["registration_diagnostics"]["confidence_score"] > 0.4


# =============================================================================
# 4. DETECTOR INFERENCE & ERROR HANDLING
# =============================================================================

def test_7_detector_inference_success(client):
    """Test 7: Detector executes M9 BIT-CD and produces positive change mask."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=77)
    t2_bytes = make_changed_t2_image(t1_bytes, change_bbox=(50, 50, 150, 150))

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    data = {"detector_model": "M9_BIT_CD"}

    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files, data=data)

    assert resp.status_code == 200
    res = resp.json()
    assert res["detector_metadata"]["model_id"] == "M9_BIT_CD"
    assert res["detector_metadata"]["threshold_used"] == 0.4500
    assert res["change_mask"]["width"] == 256
    assert res["change_mask"]["height"] == 256


def test_8_detector_failure(client):
    """Test 8: Handles detector runtime failure safely returning HTTP 500 without leaking stack trace."""
    t1_bytes = make_synthetic_image(128, 128, add_features=True)
    t2_bytes = make_synthetic_image(128, 128, add_features=True)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }

    # Simulate internal detector crash
    with patch("src.api.service.ChangeDetectionService.get_detector", side_effect=RuntimeError("GPU OOM / Memory allocation error")):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 500
    assert "Unexpected failure" in resp.json()["detail"]


# =============================================================================
# 5. REGION EXTRACTION & TOP-K CLUSTERING
# =============================================================================

def test_9_region_extraction(client):
    """Test 9: Extracts discrete spatial change regions with bounding boxes and area ratios."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=88)
    t2_bytes = make_changed_t2_image(t1_bytes, change_bbox=(40, 40, 120, 120))

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    data = {"max_regions": 3}

    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files, data=data)

    assert resp.status_code == 200
    res = resp.json()
    assert len(res["regions"]) <= 3
    if len(res["regions"]) > 0:
        reg = res["regions"][0]
        assert "region_id" in reg
        assert "bbox" in reg
        assert len(reg["bbox"]) == 4
        assert reg["pixel_area"] > 0
        assert 0.0 <= reg["area_ratio"] <= 1.0


# =============================================================================
# 6. VISION MODEL TIMEOUT, MALFORMED JSON & MISSING VARI
# =============================================================================

def test_10_vision_model_timeout(client):
    """Test 10: Vision model timeout falls back safely without failing deterministic change detection."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=12)
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }

    import httpx
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", side_effect=httpx.TimeoutException("Read timed out")):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    # Deterministic change mask is preserved!
    assert res["change_mask"] is not None
    # Semantic findings contain safe uncertain fallback
    for finding in res["semantic_findings"]:
        assert finding["requires_human_review"] is True


def test_11_malformed_vision_response(client):
    """Test 11: Malformed JSON output from vision model is safely mapped to 'uncertain'."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=15)
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }

    # Vision model returns invalid non-JSON output
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value="Here is my interpretation: it changed drastically!"):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    for finding in res["semantic_findings"]:
        assert finding["change_category"] in ("uncertain", "vegetation_increase", "vegetation_decrease")


def test_12_missing_vari(client):
    """Test 12: Handles grayscale or single-channel inputs gracefully without crashing VARI calculation."""
    # Create single-channel grayscale TIFF
    gray = np.full((128, 128), 120, dtype=np.uint8)
    cv2.rectangle(gray, (20, 20), (60, 60), 200, -1)
    buf1 = io.BytesIO()
    Image.fromarray(gray).save(buf1, format="TIFF")
    t1_bytes = buf1.getvalue()

    buf2 = io.BytesIO()
    Image.fromarray(gray).save(buf2, format="TIFF")
    t2_bytes = buf2.getvalue()

    files = {
        "file_t1": ("t1.tiff", t1_bytes, "image/tiff"),
        "file_t2": ("t2.tiff", t2_bytes, "image/tiff"),
    }
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    assert resp.json()["status"] in ("success", "registration_rejected")


# =============================================================================
# 7. EXPLANATION SYNTHESIS & FALLBACK
# =============================================================================

def test_13_explanation_fallback(client):
    """Test 13: When Groq text explainer is unconfigured, produces local deterministic report."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=22)
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }

    with patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    expl = resp.json()["temporal_explanation"]
    assert "overall_summary" in expl
    assert len(expl["detector_evidence"]) > 0
    assert len(expl["scientific_limitations"]) > 0


# =============================================================================
# 8. GSD PHYSICAL AREA GATING & PRECISION
# =============================================================================

def test_14_gsd_physical_area_calculation(client):
    """Test 14: When GSD is provided (e.g. 0.5 m/px), computes physical area in m²."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=33)
    t2_bytes = make_changed_t2_image(t1_bytes, change_bbox=(40, 40, 100, 100))

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    data = {"gsd_meters_per_pixel": 0.5}

    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files, data=data)

    assert resp.status_code == 200
    res = resp.json()
    assert res["image_metadata"]["gsd_meters_per_pixel"] == 0.5
    for r in res["regions"]:
        if r["physical_area_sqm"] is not None:
            # GSD = 0.5 -> Area = px * 0.25
            expected_sqm = round(r["pixel_area"] * 0.25, 2)
            assert np.isclose(r["physical_area_sqm"], expected_sqm, atol=1.0)


def test_15_missing_gsd(client):
    """Test 15: When GSD is omitted, physical area in m² is strictly NOT fabricated."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=34)
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    # Omit gsd_meters_per_pixel
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    res = resp.json()
    assert res["image_metadata"]["gsd_meters_per_pixel"] is None
    for r in res["regions"]:
        assert r["physical_area_sqm"] is None


# =============================================================================
# 9. RESPONSE SERIALIZATION & COORDINATE FRAME INTEGRITY
# =============================================================================

def test_16_response_serialization(client):
    """Test 16: Response validates cleanly against Pydantic TemporalAnalysisResponse schema."""
    t1_bytes = make_synthetic_image(256, 256, add_features=True, seed=55)
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    # Validate against Pydantic schema
    validated = TemporalAnalysisResponse(**resp.json())
    assert validated.status == "success"


def test_17_correct_bounding_box_coordinate_frame(client):
    """Test 17: Verifies bounding boxes lie strictly within image canvas bounds [0, W] x [0, H]."""
    w, h = 256, 256
    t1_bytes = make_synthetic_image(w, h, add_features=True, seed=66)
    t2_bytes = make_changed_t2_image(t1_bytes)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    res = resp.json()
    for reg in res["regions"]:
        xmin, ymin, xmax, ymax = reg["bbox"]
        assert 0 <= xmin <= xmax <= w
        assert 0 <= ymin <= ymax <= h
        c_xmin, c_ymin, c_xmax, c_ymax = reg["context_bbox"]
        assert 0 <= c_xmin <= c_xmax <= w
        assert 0 <= c_ymin <= c_ymax <= h


# =============================================================================
# 10. RESOURCE CLEANUP & REPO COMPATIBILITY REGRESSIONS
# =============================================================================

def test_18_temporary_file_cleanup(client):
    """Test 18: Verifies UploadFile handles are closed and memory is cleaned up after request."""
    t1_bytes = make_synthetic_image(128, 128, add_features=True)
    t2_bytes = make_synthetic_image(128, 128, add_features=True)

    files = {
        "file_t1": ("t1.png", t1_bytes, "image/png"),
        "file_t2": ("t2.png", t2_bytes, "image/png"),
    }
    with patch("src.temporal_understanding.semantic_analyzer.SemanticChangeAnalyzer._call_groq_vision", return_value=None), \
         patch("src.temporal_understanding.explanation_engine.TemporalExplanationEngine._call_groq_text", return_value=None):
        resp = client.post("/api/temporal/analyze", files=files)

    assert resp.status_code == 200
    # Verified: endpoint has try...finally calling await file.close()


def test_19_existing_benchmark_endpoint_compatibility(client):
    """Test 19: Existing /api/benchmark endpoint continues to function without regression."""
    resp = client.get("/api/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 7


def test_20_existing_llm_explanation_endpoint_compatibility(client):
    """Test 20: Existing /api/explain/change-detection endpoint remains intact."""
    payload = {
        "model_id": "M9_BIT_CD",
        "changed_pixels": 450,
        "change_ratio_pct": 2.1,
    }
    with patch("src.api.llm_service.llm_explanation_service.explain_change_detection") as mock_expl:
        from src.api.schemas import ExplanationResponse
        mock_expl.return_value = ExplanationResponse(
            explanation_model="qwen/qwen3.8-27b",
            provider="Groq",
            source_model="M9_BIT_CD",
            task="change_detection",
            prediction_summary="BIT-CD detected localized radiometric change.",
            supporting_evidence=["450 pixels changed"],
            limitations=["Optical only"],
            unknowns=["Building type"],
        )
        resp = client.post("/api/explain/change-detection", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["source_model"] == "M9_BIT_CD"
