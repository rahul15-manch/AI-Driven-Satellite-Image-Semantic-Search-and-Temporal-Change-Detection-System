"""Verification suite for Vercel deployment configuration and serverless entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


def test_vercel_json_structure():
    """Verify vercel.json is valid JSON with proper rewrites and function timeouts."""
    vercel_path = ROOT / "vercel.json"
    assert vercel_path.exists(), "vercel.json must exist in project root"

    with open(vercel_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "rewrites" in config, "vercel.json must define rewrites"
    rewrites = config["rewrites"]
    sources = [r.get("source") for r in rewrites]

    assert "/api/(.*)" in sources, "API rewrite rule must be defined"
    assert (ROOT / "public" / "index.html").exists(), "public/index.html must exist for edge static hosting"
    assert (ROOT / "public" / "css" / "styles.css").exists(), "public/css/styles.css must exist"

    # Function configuration
    assert "functions" in config
    assert "api/index.py" in config["functions"]
    assert config["functions"]["api/index.py"].get("maxDuration", 0) >= 30


def test_vercelignore_excludes_heavy_assets():
    """Verify .vercelignore excludes bulky local datasets, virtualenvs, and test caches."""
    vignore_path = ROOT / ".vercelignore"
    assert vignore_path.exists(), ".vercelignore must exist in project root"

    content = vignore_path.read_text(encoding="utf-8")
    assert "data/" in content
    assert "venv/" in content
    assert ".git" in content or ".pytest_cache" in content


def test_api_index_serverless_entrypoint():
    """Verify api/index.py imports cleanly and responds to health check."""
    from api.index import app

    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["milestone"] == "M12"
    assert "models_loaded" in data


def test_requirements_includes_cpu_index():
    """Verify requirements.txt directs pip to lightweight CPU wheels for Vercel."""
    req_path = ROOT / "requirements.txt"
    content = req_path.read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in content, (
        "requirements.txt must include CPU-only PyTorch index to prevent exceeding serverless bundle limits"
    )
