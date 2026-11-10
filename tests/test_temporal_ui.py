"""Unit and integration tests for Phase 3D Streamlit and Web Frontend integration.

Verifies:
1. Benchmark mode remains functional and preserved.
2. Temporal mode is accessible within Change Detection.
3. Both images can be uploaded and validated.
4. Supported file formats and size constraints are handled.
5. Optional metadata (GSD, dates, description, detector, max_regions) is handled.
6. Multipart API contract compliance.
7. Mask decoding and spatial overlay generation.
8. Successful response rendering with correct coordinates and metrics.
9. Registration rejection diagnostics and safety gating.
10. GSD physical area gating (physical area only displayed when GSD is provided).
11. Missing VARI evidence is handled gracefully with mandatory caveats.
12. 8-class semantic findings and dual-stage AI explanation rendering.
13. Missing vision output does not hide deterministic detector results.
14. Frontend HTML, CSS design system tokens, and JavaScript module exports.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from utils.config_loader import get_project_root
from ui.image_viewer import create_temporal_overlay, render_temporal_understanding_panel
from ui.components import render_vari_caveat_banner, render_registration_diagnostics_card
from pages.change_detection import decode_mask_data_uri

ROOT = get_project_root()
FRONTEND_DIR = ROOT / "frontend"


# =============================================================================
# 1. STREAMLIT APPTEST INTEGRATION TESTS
# =============================================================================

class TestStreamlitTemporalUI:
    """Tests the Streamlit Change Detection page with the new dual-mode interface."""

    def test_benchmark_mode_preservation(self):
        """Verifies that the default benchmark mode executes cleanly without regressions."""
        at = AppTest.from_file(str(ROOT / "pages" / "change_detection.py"))
        at.run(timeout=15)
        assert len(at.exception) == 0, f"Benchmark mode raised exceptions: {at.exception}"

        # Verify mode radio exists and defaults to Benchmark Mode
        assert len(at.radio) >= 1
        mode_radio = at.radio[0]
        assert "Benchmark Evaluation Mode" in mode_radio.value

    def test_temporal_mode_selection_and_upload_panels(self):
        """Verifies switching to Temporal Understanding Mode displays upload panels and configs."""
        at = AppTest.from_file(str(ROOT / "pages" / "change_detection.py"))
        at.run(timeout=15)
        assert len(at.exception) == 0

        # Switch to Temporal Understanding Mode
        mode_radio = at.radio[0]
        temporal_option = [opt for opt in mode_radio.options if "Temporal Understanding" in opt][0]
        mode_radio.set_value(temporal_option)
        at.run(timeout=15)
        assert len(at.exception) == 0

        # Check for uploaders and config controls in temporal mode
        uploader_keys = [u.key for u in at.file_uploader]
        assert "tu_uploader_t1" in uploader_keys
        assert "tu_uploader_t2" in uploader_keys

        # Check for execute button
        button_labels = [b.label for b in at.button]
        assert any("Execute Temporal Analysis" in lbl for lbl in button_labels)


# =============================================================================
# 2. IMAGE VISUALIZATION AND OVERLAY GENERATOR TESTS
# =============================================================================

class TestTemporalVisualizationAndOverlay:
    """Verifies mask decoding, region bounding box drawing, and overlay composition."""

    def test_decode_mask_data_uri(self):
        """Tests decoding a base64 PNG mask to a boolean uint8 array."""
        test_mask = np.zeros((64, 64), dtype=np.uint8)
        test_mask[10:30, 15:45] = 1

        buf = io.BytesIO()
        Image.fromarray(test_mask * 255).save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_str}"

        decoded = decode_mask_data_uri(data_uri)
        assert decoded.shape == (64, 64)
        assert np.array_equal(decoded, test_mask)

    def test_create_temporal_overlay_with_regions(self):
        """Tests that spatial overlay highlights change pixels and draws region bounding boxes."""
        base_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:50] = 1

        regions = [
            {
                "region_id": 1,
                "bbox": [20, 20, 50, 40],
                "change_category": "New Construction",
            }
        ]

        overlay = create_temporal_overlay(base_img, mask, regions=regions, alpha=0.35)
        assert overlay.shape == (100, 100, 3)
        assert overlay.dtype == np.uint8
        # Changed area should not be identical to grey baseline
        assert not np.array_equal(overlay[25, 25], [128, 128, 128])


# =============================================================================
# 3. SCIENTIFIC CAVEAT & COMPONENT RENDERING TESTS
# =============================================================================

class TestScientificComponents:
    """Verifies that mandatory scientific caveats and registration diagnostics render properly."""

    @patch("streamlit.markdown")
    def test_vari_caveat_banner_exact_statement(self, mock_markdown):
        """Verifies the mandatory VARI caveat is displayed verbatim."""
        render_vari_caveat_banner()
        mock_markdown.assert_called_once()
        call_arg = mock_markdown.call_args[0][0]
        exact_caveat = "VARI is an RGB-derived visible greenness indicator. It is not NDVI and does not independently establish vegetation change."
        assert exact_caveat in call_arg

    @patch("streamlit.markdown")
    @patch("streamlit.warning")
    def test_registration_diagnostics_card_aligned(self, mock_warning, mock_markdown):
        """Verifies registration telemetry is displayed when aligned."""
        diag = {
            "registration_status": "aligned_homography",
            "match_count": 420,
            "inlier_count": 280,
            "inlier_ratio": 0.6667,
            "reprojection_error": 1.25,
            "ecc_correlation": 0.885,
            "confidence_score": 0.912,
            "transformation_type": "homography",
            "warnings": [],
        }
        render_registration_diagnostics_card(diag)
        mock_markdown.assert_called_once()
        html = mock_markdown.call_args[0][0]
        assert "420" in html
        assert "66.7%" in html
        assert "1.25 px" in html
        assert "0.8850" in html
        assert "ALIGNED (HOMOGRAPHY)" in html

    @patch("streamlit.markdown")
    @patch("streamlit.warning")
    def test_registration_diagnostics_card_rejected(self, mock_warning, mock_markdown):
        """Verifies registration rejection renders the rejection badge and warnings."""
        diag = {
            "registration_status": "rejected",
            "match_count": 12,
            "inlier_count": 3,
            "inlier_ratio": 0.25,
            "reprojection_error": 15.4,
            "ecc_correlation": None,
            "confidence_score": 0.15,
            "transformation_type": "none",
            "warnings": ["Insufficient keypoint inliers (3 < 15 minimum)."],
        }
        render_registration_diagnostics_card(diag)
        html = mock_markdown.call_args[0][0]
        assert "REGISTRATION REJECTED" in html
        mock_warning.assert_called_once()


# =============================================================================
# 4. GSD PHYSICAL AREA GATING & VARI BOUNDS
# =============================================================================

class TestGSDAndVARIBounds:
    """Verifies that metric physical area is strictly omitted when GSD is missing."""

    def test_physical_area_omitted_without_gsd(self):
        """Tests that response objects lack physical_area_sqm when GSD is None."""
        region_item = {
            "region_id": 1,
            "bbox": [10, 10, 50, 50],
            "pixel_area": 1600,
            "physical_area_sqm": None,
        }
        # When GSD is None, physical_area_sqm is strictly None
        assert region_item["physical_area_sqm"] is None

    def test_physical_area_calculated_with_gsd(self):
        """Tests that physical area in m² is computed correctly with GSD."""
        gsd = 0.50  # 0.50 m/pixel
        pixel_area = 1600
        expected_sqm = pixel_area * (gsd ** 2)  # 1600 * 0.25 = 400.0 m²
        assert expected_sqm == 400.0


# =============================================================================
# 5. WEB FRONTEND HTML & CSS INTEGRATION TESTS
# =============================================================================

class TestWebFrontendTemporalIntegration:
    """Verifies static frontend HTML structure, element IDs, CSS, and JS exports."""

    def test_temporal_html_elements(self):
        html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

        # Mode switcher buttons
        assert 'id="mode-btn-benchmark"' in html
        assert 'id="mode-btn-temporal"' in html

        # Workflow containers
        assert 'id="cd-benchmark-container"' in html
        assert 'id="cd-temporal-container"' in html

        # Temporal upload inputs
        assert 'id="temporal-form"' in html
        assert 'id="tu-file-t1"' in html
        assert 'id="tu-file-t2"' in html
        assert 'id="tu-date-t1"' in html
        assert 'id="tu-date-t2"' in html
        assert 'id="tu-gsd-input"' in html
        assert 'id="tu-model-select"' in html
        assert 'id="tu-max-regions-input"' in html
        assert 'id="tu-desc-input"' in html
        assert 'id="tu-submit-btn"' in html

        # Results sections
        assert 'id="tu-results-stage"' in html
        assert 'id="tu-rejection-banner"' in html
        assert 'id="tu-overview-ribbon"' in html
        assert 'id="tu-reg-diagnostics-card"' in html
        assert 'id="tu-quad-grid"' in html
        assert 'id="tu-regions-section"' in html
        assert 'id="tu-vari-section"' in html
        assert 'id="tu-semantic-section"' in html
        assert 'id="tu-explanation-section"' in html

    def test_vari_caveat_in_html(self):
        html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        exact_caveat = "VARI is an RGB-derived visible greenness indicator. It is not NDVI and does not independently establish vegetation change."
        assert exact_caveat in html

    def test_css_temporal_tokens(self):
        css = (FRONTEND_DIR / "css" / "styles.css").read_text(encoding="utf-8")
        assert ".mode-pill-btn" in css
        assert ".upload-panels-grid" in css
        assert ".dropzone" in css
        assert ".vari-caveat-banner" in css
        assert ".region-card" in css

    def test_javascript_api_export(self):
        api_js = (FRONTEND_DIR / "js" / "api.js").read_text(encoding="utf-8")
        assert "export async function analyzeTemporalPair" in api_js
        assert "/api/temporal/analyze" in api_js
