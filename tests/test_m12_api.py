"""
Unit and Integration Tests for Milestone 12 FastAPI Backend.
Covers health checks, semantic search, change detection, benchmark loader,
static file serving, and security constraints (path traversal).
"""

import pytest
import warnings
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")

from src.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["milestone"] == "M12"
    assert data["device"] == "cpu"
    assert data["services"]["retrieval"] is True
    assert data["services"]["change_detection"] is True
    assert data["services"]["benchmark"] is True


def test_search_clip_success(client):
    payload = {
        "query": "airport with commercial airplanes",
        "method": "clip",
        "top_k": 4,
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["method"] == "clip"
    assert len(data["results"]) == 4
    for item in data["results"]:
        assert "rank" in item
        assert "image_id" in item
        assert "filename" in item
        assert "category" in item
        assert "similarity_score" in item
        assert "image_url" in item
        assert item["image_url"].startswith("/api/image/rsicd/")


def test_search_bm25_success(client):
    payload = {
        "query": "residential area with houses",
        "method": "bm25",
        "top_k": 3,
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "bm25"
    assert len(data["results"]) == 3


def test_search_invalid_method(client):
    payload = {
        "query": "harbor with ships",
        "method": "unsupported_method",
        "top_k": 5,
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_search_empty_query(client):
    payload = {
        "query": "",
        "method": "clip",
        "top_k": 5,
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 422


def test_get_pairs_endpoint(client):
    response = client.get("/api/pairs")
    assert response.status_code == 200
    pairs = response.json()
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    test_1 = next((p for p in pairs if p["pair_id"] == "test_1"), None)
    assert test_1 is not None
    assert test_1["has_ground_truth"] is True


def test_change_detection_classical_b1(client):
    payload = {
        "pair_id": "test_1",
        "model": "B1_Pixel_Diff",
    }
    response = client.post("/api/change-detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pair_id"] == "test_1"
    assert data["model"] == "B1_Pixel_Diff"
    assert data["heatmap_data_uri"].startswith("data:image/png;base64,")
    assert data["mask_data_uri"].startswith("data:image/png;base64,")
    assert data["overlay_data_uri"].startswith("data:image/png;base64,")
    assert data["metrics"] is not None
    assert 0.0 <= data["metrics"]["f1"] <= 1.0
    assert data["metrics"]["fp"] >= 0


def test_change_detection_threshold_override(client):
    payload = {
        "pair_id": "test_1",
        "model": "B1_Pixel_Diff",
        "threshold": 0.85,
    }
    response = client.post("/api/change-detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert abs(data["threshold"] - 0.85) < 1e-4


def test_change_detection_unsupported_model(client):
    payload = {
        "pair_id": "test_1",
        "model": "NON_EXISTENT_MODEL",
    }
    response = client.post("/api/change-detection", json=payload)
    assert response.status_code == 400
    assert "Unsupported model identifier" in response.json()["detail"]


def test_change_detection_invalid_pair(client):
    payload = {
        "pair_id": "invalid_pair_99999",
        "model": "B1_Pixel_Diff",
    }
    response = client.post("/api/change-detection", json=payload)
    assert response.status_code == 404


def test_benchmark_endpoint(client):
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "bootstrap_ci" in data
    assert "wilcoxon_tests" in data
    assert "ablation" in data
    assert "figures" in data

    model_ids = [m["model_id"] for m in data["models"]]
    assert "M9_BIT_CD" in model_ids
    assert "M10_QAT_CD_Calibrated" in model_ids
    assert "M7_FC_Siam_diff" in model_ids
    assert "B1_Pixel_Diff" in model_ids

    # Verify exact M11 BIT-CD metrics
    bit_cd = next(m for m in data["models"] if m["model_id"] == "M9_BIT_CD")
    assert abs(bit_cd["f1"] - 0.5927) < 1e-3
    assert abs(bit_cd["iou"] - 0.4211) < 1e-3


def test_image_serving_security_traversal(client):
    # Attempting directory traversal
    response = client.get("/api/image/levir_cd/..%2F..%2Fetc/passwd")
    assert response.status_code in (403, 404)


def test_image_serving_invalid_dataset(client):
    response = client.get("/api/image/secret_dataset/image.png")
    assert response.status_code == 400


def test_static_files_serving(client):
    # Main HTML
    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "<title>Satellite AI" in html_resp.text

    # CSS
    css_resp = client.get("/css/styles.css")
    assert css_resp.status_code == 200
    assert "--bg-base" in css_resp.text

    # Modular JS files
    for js_file in ["api.js", "app.js", "search.js", "change_detection.js", "results.js"]:
        js_resp = client.get(f"/js/{js_file}")
        assert js_resp.status_code == 200
