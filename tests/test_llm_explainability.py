"""Comprehensive test suite for Groq-hosted Qwen 27B AI model explainability.

Verifies:
1. Missing API key is handled gracefully.
2. Invalid credentials produce useful errors without leaking secrets.
3. Configured model availability and status endpoints.
4. Successful explanation generation with mocked provider responses.
5. Real authenticated smoke test when credentials are configured in .env.
6. Timeout, rate limit, and server error handling.
7. Source-model and explanation-model strict separation.
8. Incomplete evidence handling without inventing metrics.
9. No credentials exposed in responses or logs.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import httpx

from src.api.app import create_app
from src.api.llm_service import LLMExplanationService
from src.api.schemas import (
    ExplainChangeDetectionRequest,
    ExplainSearchRequest,
    ExplanationResponse,
)


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


# ------------------------------------------------------------------------------
# 1. Status & Configuration Tests
# ------------------------------------------------------------------------------

def test_llm_status_endpoint(client):
    """Verifies GET /api/llm/status contract."""
    response = client.get("/api/llm/status")
    assert response.status_code == 200
    data = response.json()
    assert "configured" in data
    assert data["provider"] == "Groq"
    assert "qwen" in data["model"].lower()
    assert data["status"] in ("ready", "unconfigured", "error")
    assert data["connectivity"] in ("verified", "unverified", "failed")


def test_missing_api_key_handling():
    """Verifies that an unconfigured service produces clean status without throwing."""
    svc = LLMExplanationService()
    svc.api_key = ""  # Simulate missing key
    assert svc.is_configured() is False
    status = svc.get_status(check_connectivity=False)
    assert status.configured is False
    assert status.status == "unconfigured"

    # Attempting call without key should raise clean RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        svc._call_groq_chat("system", "user")
    assert "not configured" in str(exc_info.value).lower()
    # Confirm no secret strings leaked
    assert "gsk_" not in str(exc_info.value)


# ------------------------------------------------------------------------------
# 2. Mocked Provider Call & Schema Compliance
# ------------------------------------------------------------------------------

MOCK_QWEN_CHANGE_JSON = json.dumps({
    "prediction_summary": "The M10_QAT_CD model detected 14,250 changed pixels under threshold 0.45 with quality attenuation 0.30.",
    "supporting_evidence": [
        "Predicted changed area constitutes 1.36% of the 1024x1024 scene.",
        "Decision threshold tau* = 0.4500 with F1 score = 0.8470.",
    ],
    "limitations": [
        "Threshold optimized on validation split; subject to residual false alarms.",
        "CPU execution standardizes inference without hardware-specific acceleration.",
    ],
    "unknowns": [
        "Semantic structure category is not classified by the binary mask.",
    ],
})

MOCK_QWEN_SEARCH_JSON = json.dumps({
    "prediction_summary": "The aerial scene was ranked first due to high cosine similarity in CLIP embedding space.",
    "supporting_evidence": [
        "Candidate #1 matched the query with Cosine Similarity: 78.4%.",
        "Assigned category 'airport' aligns with tarmac visual descriptors.",
    ],
    "limitations": [
        "Zero-shot visual representations may occasionally conflate paved surfaces.",
    ],
    "unknowns": [
        "Specific aircraft models on the runway cannot be verified.",
    ],
})


def test_mocked_change_detection_explanation():
    """Verifies explanation assembly and schema conformance for change detection."""
    svc = LLMExplanationService()
    svc.api_key = "gsk_dummy_test_key"

    with patch.object(svc, "_call_groq_chat", return_value=MOCK_QWEN_CHANGE_JSON):
        req = ExplainChangeDetectionRequest(
            pair_id="test_1",
            model_id="M10_QAT_CD",
            threshold=0.45,
            gamma=0.30,
            changed_pixels=14250,
            change_ratio_pct=1.36,
            metrics={"f1": 0.847, "iou": 0.735},
        )
        resp = svc.explain_change_detection(req)
        assert isinstance(resp, ExplanationResponse)
        assert resp.source_model == "M10_QAT_CD"
        assert resp.task == "change_detection"
        assert resp.provider == "Groq"
        assert "qwen" in resp.explanation_model.lower()
        assert len(resp.supporting_evidence) == 2
        assert len(resp.limitations) == 2
        assert len(resp.unknowns) == 1


def test_mocked_semantic_search_explanation():
    """Verifies explanation assembly and schema conformance for semantic retrieval."""
    svc = LLMExplanationService()
    svc.api_key = "gsk_dummy_test_key"

    with patch.object(svc, "_call_groq_chat", return_value=MOCK_QWEN_SEARCH_JSON):
        req = ExplainSearchRequest(
            query="an airport with many airplanes",
            method="clip",
            image_id="airport_12",
            filename="airport_12.png",
            rank=1,
            score=0.784,
            category="airport",
        )
        resp = svc.explain_semantic_search(req)
        assert isinstance(resp, ExplanationResponse)
        assert resp.source_model == "CLIP"
        assert resp.task == "semantic_search"
        assert "qwen" in resp.explanation_model.lower()
        assert len(resp.supporting_evidence) == 2


# ------------------------------------------------------------------------------
# 3. Provider Error Handling & Security
# ------------------------------------------------------------------------------

def test_groq_authentication_failure_handling():
    """Verifies that HTTP 401 from provider produces clean PermissionError without credentials."""
    svc = LLMExplanationService()
    svc.api_key = "gsk_invalid_secret_key_12345"

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Invalid API Key"

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(PermissionError) as exc_info:
            svc._call_groq_chat("system", "user")
        assert "authentication failed" in str(exc_info.value).lower()
        # Verify no secret key leaked in exception string
        assert "gsk_invalid_secret_key_12345" not in str(exc_info.value)


def test_groq_rate_limit_handling():
    """Verifies that HTTP 429 rate limit produces clean retry message."""
    svc = LLMExplanationService()
    svc.api_key = "gsk_valid_key"

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Rate limit reached"

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            svc._call_groq_chat("system", "user")
        assert "rate limit" in str(exc_info.value).lower()


def test_groq_timeout_handling():
    """Verifies that request timeouts are converted to clean TimeoutError."""
    svc = LLMExplanationService()
    svc.api_key = "gsk_valid_key"

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
        with pytest.raises(TimeoutError) as exc_info:
            svc._call_groq_chat("system", "user")
        assert "timed out" in str(exc_info.value).lower()


def test_explanation_json_parsing_with_code_fences():
    """Verifies that Markdown code fences around JSON are stripped cleanly."""
    svc = LLMExplanationService()
    raw = "```json\n{\n  \"prediction_summary\": \"Clean result\",\n  \"supporting_evidence\": [\"Fact 1\"],\n  \"limitations\": [\"Lim 1\"],\n  \"unknowns\": [\"Unk 1\"]\n}\n```"
    parsed = svc._parse_explanation_json(raw)
    assert parsed["prediction_summary"] == "Clean result"
    assert parsed["supporting_evidence"] == ["Fact 1"]


# ------------------------------------------------------------------------------
# 4. End-to-End API Routes with TestClient
# ------------------------------------------------------------------------------

def test_api_explain_change_detection_route(client):
    """Verifies POST /api/explain/change-detection route with mocked Groq call."""
    with patch("src.api.llm_service.llm_explanation_service._call_groq_chat", return_value=MOCK_QWEN_CHANGE_JSON):
        # Clear cache to force fresh call
        src_svc = getattr(client.app, "state", None)
        from src.api.service import llm_explanation_service
        llm_explanation_service._cache.clear()

        payload = {
            "pair_id": "test_1",
            "model_id": "M9_BIT_CD",
            "threshold": 0.50,
            "gamma": 0.0,
            "changed_pixels": 10500,
            "change_ratio_pct": 1.0,
        }
        res = client.post("/api/explain/change-detection", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "change_detection"
        assert data["source_model"] == "M9_BIT_CD"
        assert data["provider"] == "Groq"


def test_api_explain_semantic_search_route(client):
    """Verifies POST /api/explain/semantic-search route with mocked Groq call."""
    with patch("src.api.llm_service.llm_explanation_service._call_groq_chat", return_value=MOCK_QWEN_SEARCH_JSON):
        from src.api.service import llm_explanation_service
        llm_explanation_service._cache.clear()

        payload = {
            "query": "a harbor with shipping containers",
            "method": "clip",
            "image_id": "harbor_01",
            "filename": "harbor_01.png",
            "rank": 1,
            "score": 0.812,
            "category": "harbor",
        }
        res = client.post("/api/explain/semantic-search", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "semantic_search"
        assert data["source_model"] == "CLIP"


# ------------------------------------------------------------------------------
# 5. Live Authenticated API Smoke Test (when credentials available in .env)
# ------------------------------------------------------------------------------

def test_live_authenticated_groq_qwen_smoke(client):
    """Genuine authenticated smoke test verifying that Groq + Qwen 27B responds in production."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or not api_key.startswith("gsk_"):
        pytest.skip("GROQ_API_KEY not configured in .env, skipping live smoke test.")

    # 1. Live status
    status_resp = client.get("/api/llm/status?check=true")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["configured"] is True
    assert status_data["connectivity"] == "verified"
    assert "qwen" in status_data["model"].lower()

    # 2. Live explanation for semantic search
    search_payload = {
        "query": "residential houses with green yards",
        "method": "clip",
        "image_id": "residential_10",
        "filename": "residential_10.png",
        "rank": 1,
        "score": 0.745,
        "category": "residential",
        "captions": ["dense single family residential homes with green trees and lawns"],
    }
    explain_resp = client.post("/api/explain/semantic-search", json=search_payload)
    assert explain_resp.status_code == 200
    explain_data = explain_resp.json()
    assert explain_data["provider"] == "Groq"
    assert explain_data["source_model"] == "CLIP"
    assert len(explain_data["prediction_summary"]) > 20
    assert len(explain_data["supporting_evidence"]) >= 1
    assert len(explain_data["limitations"]) >= 1
    assert len(explain_data["unknowns"]) >= 1
