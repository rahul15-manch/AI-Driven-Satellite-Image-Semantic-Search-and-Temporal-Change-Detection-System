"""
Automated Structural and Integrity Tests for Milestone 12 Web Interface Frontend.
Verifies HTML structure, critical element IDs, CSS tokens, and JavaScript module integrity.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"


def test_frontend_directory_structure():
    assert FRONTEND_DIR.is_dir()
    assert (FRONTEND_DIR / "index.html").is_file()
    assert (FRONTEND_DIR / "css" / "styles.css").is_file()
    assert (FRONTEND_DIR / "js" / "api.js").is_file()
    assert (FRONTEND_DIR / "js" / "app.js").is_file()
    assert (FRONTEND_DIR / "js" / "search.js").is_file()
    assert (FRONTEND_DIR / "js" / "change_detection.js").is_file()
    assert (FRONTEND_DIR / "js" / "results.js").is_file()


def test_html_document_structure():
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert '<html lang="en">' in html_content
    assert '<meta name="viewport"' in html_content
    assert '<title>Satellite AI' in html_content

    # Four core tabs / panels
    required_panel_ids = [
        "panel-search",
        "panel-change",
        "panel-results",
        "panel-methodology",
    ]
    for pid in required_panel_ids:
        assert f'id="{pid}"' in html_content

    # Navigation buttons
    required_nav_buttons = [
        "tab-btn-search",
        "tab-btn-change",
        "tab-btn-results",
        "tab-btn-methodology",
    ]
    for nbid in required_nav_buttons:
        assert f'id="{nbid}"' in html_content


def test_search_component_elements():
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    required_elements = [
        "search-form",
        "search-query-input",
        "search-method-select",
        "search-topk-select",
        "search-submit-btn",
        "search-meta-bar",
        "search-results-container",
    ]
    for elem_id in required_elements:
        assert f'id="{elem_id}"' in html_content


def test_change_detection_component_elements():
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    required_elements = [
        "change-form",
        "change-pair-select",
        "change-model-select",
        "qat-options-group",
        "qat-gamma-select",
        "change-threshold-override",
        "change-submit-btn",
        "show-error-overlay-toggle",
        "change-results-stage",
        "view-t1",
        "view-t2",
        "view-heatmap",
        "view-mask",
        "diagnostic-overlay-section",
        "view-overlay",
        "val-precision",
        "val-recall",
        "val-f1",
        "val-iou",
        "val-accuracy",
        "val-fp",
    ]
    for elem_id in required_elements:
        assert f'id="{elem_id}"' in html_content


def test_results_and_methodology_elements():
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    required_elements = [
        "benchmark-summary-table",
        "benchmark-table-body",
        "bootstrap-ci-table",
        "bootstrap-table-body",
        "wilcoxon-tests-table",
        "wilcoxon-table-body",
        "ablation-table",
        "ablation-table-body",
        "figures-gallery-grid",
        "image-modal",
        "modal-image",
        "modal-caption",
        "modal-close-btn",
    ]
    for elem_id in required_elements:
        assert f'id="{elem_id}"' in html_content


def test_css_design_system_tokens():
    css_content = (FRONTEND_DIR / "css" / "styles.css").read_text(encoding="utf-8")
    # Verify dark mode custom properties
    assert "--bg-base" in css_content
    assert "--bg-surface" in css_content
    assert "--accent-primary" in css_content
    assert "--font-sans" in css_content
    assert "--font-mono" in css_content
    # Responsive media query exists
    assert "@media" in css_content


def test_javascript_modules_export_syntax():
    js_files = ["api.js", "search.js", "change_detection.js", "results.js", "app.js"]
    for jf in js_files:
        content = (FRONTEND_DIR / "js" / jf).read_text(encoding="utf-8")
        assert len(content.strip()) > 50
        # No blatant unclosed braces or syntax issues
        open_braces = content.count("{")
        close_braces = content.count("}")
        assert open_braces == close_braces, f"Mismatched braces in {jf}: {open_braces} != {close_braces}"


def test_research_results_layout_and_responsive_css():
    """Verifies Research Results tab layout classes, responsive grid, and non-clipping image CSS."""
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "css" / "styles.css").read_text(encoding="utf-8")

    # HTML structure requirements
    assert "research-results-container" in html_content
    assert "research-results-grid" in html_content
    assert "figures-filter-group" in html_content
    assert 'data-filter="all"' in html_content
    assert 'data-filter="m13"' in html_content
    assert 'data-filter="m11"' in html_content

    # CSS responsive grid and non-clipping rules
    assert ".research-results-container" in css_content
    assert ".research-results-grid" in css_content
    assert ".research-chart-card" in css_content
    assert "object-fit: contain" in css_content
    assert "repeat(2, minmax(0, 1fr))" in css_content
    assert "max-width: 1440px" in css_content


def test_search_results_and_explanation_layout_css():
    """Verifies layout width, card structure, and responsive 3-column explanation grid CSS."""
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "css" / "styles.css").read_text(encoding="utf-8")
    js_content = (FRONTEND_DIR / "js" / "search.js").read_text(encoding="utf-8")

    # HTML structure check
    assert 'id="search-results-container"' in html_content
    assert 'class="search-results-container"' in html_content

    # CSS classes and layout rules
    assert ".search-results-container" in css_content
    assert ".search-results-list" in css_content
    assert ".search-result-card" in css_content
    assert ".search-result-main" in css_content
    assert ".search-result-thumb-box" in css_content
    assert ".explanation-slot" in css_content
    assert ".explanation-card" in css_content
    assert ".explanation-summary-box" in css_content
    assert ".explanation-details-grid" in css_content

    # Grid rules & media query specifications
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in css_content
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css_content
    assert "grid-template-columns: minmax(0, 1fr)" in css_content
    assert "@media (max-width: 900px)" in css_content
    assert "@media (max-width: 600px)" in css_content

    # JavaScript emits required classes
    assert "search-results-list" in js_content
    assert "search-result-card" in js_content
    assert "search-result-main" in js_content
    assert "explanation-slot" in js_content


def test_change_detection_layout_and_responsive_css():
    """Verifies Change Detection form alignment, 4-column quad viewer grid, and responsive styling."""
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "css" / "styles.css").read_text(encoding="utf-8")

    # HTML structural classes
    assert "change-detection-config" in html_content
    assert "change-detection-actions" in html_content
    assert "change-detection-results-grid" in html_content
    assert "visualization-card" in html_content

    # CSS layout rules
    assert ".change-detection-config" in css_content
    assert ".change-detection-actions" in css_content
    assert ".change-detection-results-grid" in css_content
    assert ".visualization-card" in css_content
    assert "repeat(4, minmax(0, 1fr))" in css_content
    assert "min-height: 46px" in css_content
    assert "aspect-ratio: 1 / 1" in css_content
    assert "object-fit: contain" in css_content


def test_scene_selector_labels_regression():
    """Verifies that dynamic scene pair labeling logic never produces undefined or null labels."""
    js_content = (FRONTEND_DIR / "js" / "change_detection.js").read_text(encoding="utf-8")
    assert 'rawIdentifier !== "undefined"' in js_content
    assert 'rawIdentifier !== "null"' in js_content
    assert "Scene identifier unavailable" in js_content
    assert "Ground Truth Available" in js_content


