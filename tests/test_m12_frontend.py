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
