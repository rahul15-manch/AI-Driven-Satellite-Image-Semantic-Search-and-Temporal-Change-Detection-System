"""Unit and integration tests for dual-stage temporal explanation engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from src.temporal_understanding.explanation_engine import (
    TemporalExplanationConfig,
    TemporalExplanationEngine,
    TemporalUnderstandingReport,
)
from src.temporal_understanding.registration import (
    RegistrationDiagnostics,
    RegistrationStatus,
)
from src.temporal_understanding.region_extractor import ChangedRegion
from src.temporal_understanding.semantic_analyzer import (
    RegionalSemanticInterpretation,
    RegionalVARIEvidence,
    SemanticCategory,
)


@pytest.fixture
def sample_diagnostics():
    return RegistrationDiagnostics(
        match_count=420,
        inlier_count=185,
        inlier_ratio=0.44,
        reprojection_error=0.9,
        confidence_score=0.91,
        registration_status=RegistrationStatus.ALIGNED_HOMOGRAPHY,
        warnings=["Slight residual parallax along tall structure boundaries."],
    )


@pytest.fixture
def sample_regions():
    r1 = ChangedRegion(
        region_id=1,
        bbox=(10, 20, 40, 50),
        context_bbox=(5, 15, 45, 55),
        pixel_area=1200,
        raw_pixel_area=1250,
        area_ratio=0.034,
        centroid=(25.0, 35.0),
        bbox_width=30,
        bbox_height=30,
        mean_probability=0.88,
        gsd_meters=0.5,
        physical_area_sq_meters=300.0,
    )
    return [r1]


@pytest.fixture
def sample_semantic_findings():
    vari_ev = RegionalVARIEvidence(
        status="supported",
        t1_mean_vari=0.35,
        t2_mean_vari=-0.15,
        delta_vari=-0.50,
        interpretation="Vegetation greenness proxy decrease: delta VARI -0.50.",
        limitations=["Optical proxy only."],
    )
    return [
        RegionalSemanticInterpretation(
            region_id=1,
            change_category=SemanticCategory.SOIL_CLEARING.value,
            visual_observation="Vegetation stripped exposing bare topsoil.",
            evidence_for_category=["Large VARI decline", "Loss of canopy texture"],
            alternative_interpretations=["Demolition prep"],
            uncertainty_reason="No multispectral NIR to confirm non-photosynthetic vegetation",
            requires_human_review=False,
            vari_evidence=vari_ev,
        )
    ]


@pytest.fixture
def sample_telemetry():
    return {
        "model_id": "M9_BIT_CD",
        "threshold": 0.4500,
        "total_changed_pixels": 1200,
        "total_changed_area_pct": 3.40,
        "latency_ms": 120.5,
        "qat_diagnostics": {
            "mean_registration_discrepancy": 0.12,
            "mean_illumination_gradient": 0.05,
        },
    }


def test_no_fabricated_physical_area_when_gsd_missing(sample_telemetry, sample_diagnostics, sample_regions):
    """Tests that when GSD is omitted, physical area in m² is strictly not fabricated."""
    engine = TemporalExplanationEngine(TemporalExplanationConfig(groq_api_key=""))
    report = engine.generate_explanation(
        detector_telemetry=sample_telemetry,
        regions=sample_regions,
        semantic_findings=[],
        registration_diagnostics=sample_diagnostics,
        metadata={"gsd_meters": None},  # No GSD
    )

    # Check limitations note that GSD is absent
    assert any("Ground Sample Distance (GSD) metadata is absent" in lim for lim in report.scientific_limitations)
    # Check region observation area description does not contain m²
    assert "m²" not in report.region_observations[0]["area_description"]


def test_physical_area_reported_when_gsd_provided(sample_telemetry, sample_diagnostics, sample_regions):
    """Tests that physical area in square metres is computed and reported when GSD is provided."""
    engine = TemporalExplanationEngine(TemporalExplanationConfig(groq_api_key=""))
    report = engine.generate_explanation(
        detector_telemetry=sample_telemetry,
        regions=sample_regions,
        semantic_findings=[],
        registration_diagnostics=sample_diagnostics,
        metadata={"gsd_meters": 0.5},  # GSD = 0.5 m/px -> 1200 px * 0.25 = 300 m²
    )

    assert "300.0 m²" in report.region_observations[0]["area_description"]


def test_explanation_engine_missing_api_key(
    sample_telemetry, sample_diagnostics, sample_regions, sample_semantic_findings
):
    """Tests that missing API key falls back to structured deterministic report without failure."""
    engine = TemporalExplanationEngine(TemporalExplanationConfig(groq_api_key=""))
    report = engine.generate_explanation(
        detector_telemetry=sample_telemetry,
        regions=sample_regions,
        semantic_findings=sample_semantic_findings,
        registration_diagnostics=sample_diagnostics,
        metadata={"gsd_meters": 0.5},
    )

    assert isinstance(report, TemporalUnderstandingReport)
    assert "M9_BIT_CD" in report.overall_summary
    assert len(report.detector_evidence) > 0
    assert len(report.region_observations) == 1
    assert len(report.semantic_interpretations) == 1
    assert report.semantic_interpretations[0]["proposed_category"] == "soil_clearing"
    assert len(report.human_review_recommendations) > 0


def test_explanation_engine_missing_vision_results(
    sample_telemetry, sample_diagnostics, sample_regions
):
    """Tests explanation generation when vision inference was skipped or failed."""
    engine = TemporalExplanationEngine(TemporalExplanationConfig(groq_api_key=""))
    report = engine.generate_explanation(
        detector_telemetry=sample_telemetry,
        regions=sample_regions,
        semantic_findings=[],  # Empty vision results
        registration_diagnostics=sample_diagnostics,
        metadata={"gsd_meters": None},
    )

    assert isinstance(report, TemporalUnderstandingReport)
    assert len(report.region_observations) == 1
    assert report.semantic_interpretations[0]["proposed_category"] == "unverified"
    assert "unavailable" in report.semantic_interpretations[0]["visual_observation"].lower()


def test_explanation_engine_successful_mocked_groq_call(
    sample_telemetry, sample_diagnostics, sample_regions, sample_semantic_findings
):
    """Tests synthesis when Groq text model returns a structured JSON academic report."""
    mock_report_json = {
        "overall_summary": "BIT-CD identified localized soil clearing across 1,200 change pixels following accurate sub-pixel co-registration.",
        "detector_evidence": [
            "Core change detection executed using M9_BIT_CD under decision threshold tau* = 0.4500.",
            "Total detected changed area: 1,200 pixels (3.40% of evaluated scene).",
            "Spatial clustering identified 1 primary significant change regions."
        ],
        "scientific_limitations": ["VARI is an optical greenness proxy without NIR validation."],
        "unknown_variables": ["Specific equipment used", "Planned future land use"],
        "human_review_recommendations": ["Field survey recommended to verify soil compaction."]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(mock_report_json)
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        engine = TemporalExplanationEngine(
            TemporalExplanationConfig(groq_api_key="gsk_mock_valid_key", text_model="qwen/qwen3.8-27b")
        )
        report = engine.generate_explanation(
            detector_telemetry=sample_telemetry,
            regions=sample_regions,
            semantic_findings=sample_semantic_findings,
            registration_diagnostics=sample_diagnostics,
            metadata={"gsd_meters": 0.5},
        )

        assert report.overall_summary == mock_report_json["overall_summary"]
        assert len(report.detector_evidence) == 3
        assert report.semantic_interpretations[0]["proposed_category"] == "soil_clearing"
        assert report.metadata["model_id"] == "M9_BIT_CD"
        assert report.metadata["has_vision_findings"] is True


def test_explanation_engine_groq_api_error_fallback(
    sample_telemetry, sample_diagnostics, sample_regions, sample_semantic_findings
):
    """Tests fallback when Groq text model encounters network failure or HTTP errors."""
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
        engine = TemporalExplanationEngine(
            TemporalExplanationConfig(groq_api_key="gsk_mock_valid_key", max_retries=1)
        )
        report = engine.generate_explanation(
            detector_telemetry=sample_telemetry,
            regions=sample_regions,
            semantic_findings=sample_semantic_findings,
            registration_diagnostics=sample_diagnostics,
            metadata={"gsd_meters": 0.5},
        )

        assert isinstance(report, TemporalUnderstandingReport)
        assert "M9_BIT_CD" in report.overall_summary
        assert len(report.detector_evidence) > 0
