"""Semantic change analyzer and multimodal vision interpretation engine.

Computes the optical Visible Atmospherically Resistant Index (VARI) for RGB imagery
and integrates Groq-hosted vision models (Qwen 3.8 Multimodal) to interpret
localized change regions into an approved 8-class taxonomy with rigorous uncertainty handling.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import httpx
import numpy as np
from PIL import Image

from src.temporal_understanding.region_extractor import ChangedRegion

logger = logging.getLogger(__name__)


# =============================================================================
# TAXONOMY ENUM AND CONSTANTS
# =============================================================================

class SemanticCategory(str, Enum):
    """Standardized 8-category taxonomy for remote sensing temporal changes."""
    NEW_CONSTRUCTION = "new_construction"
    DEMOLITION = "demolition"
    VEGETATION_INCREASE = "vegetation_increase"
    VEGETATION_DECREASE = "vegetation_decrease"
    ROAD_DEVELOPMENT = "road_development"
    WATER_BODY_CHANGE = "water_body_change"
    SOIL_CLEARING = "soil_clearing"
    UNCERTAIN = "uncertain"


APPROVED_CATEGORIES = {c.value for c in SemanticCategory}


# =============================================================================
# DATA CONTRACTS
# =============================================================================

@dataclass
class RegionalVARIEvidence:
    """Quantitative optical greenness proxy evidence derived from visible RGB bands."""
    status: str  # "supported", "unsupported_channels", "zero_variance"
    t1_mean_vari: Optional[float]
    t2_mean_vari: Optional[float]
    delta_vari: Optional[float]
    interpretation: str
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "t1_mean_vari": round(float(self.t1_mean_vari), 4) if self.t1_mean_vari is not None else None,
            "t2_mean_vari": round(float(self.t2_mean_vari), 4) if self.t2_mean_vari is not None else None,
            "delta_vari": round(float(self.delta_vari), 4) if self.delta_vari is not None else None,
            "interpretation": self.interpretation,
            "limitations": self.limitations,
        }


@dataclass
class RegionalSemanticInterpretation:
    """Evidence-grounded semantic interpretation for an individual change region."""
    region_id: int
    change_category: str
    visual_observation: str
    evidence_for_category: List[str]
    alternative_interpretations: List[str]
    uncertainty_reason: Optional[str]
    requires_human_review: bool
    vari_evidence: Optional[RegionalVARIEvidence] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "change_category": self.change_category,
            "visual_observation": self.visual_observation,
            "evidence_for_category": self.evidence_for_category,
            "alternative_interpretations": self.alternative_interpretations,
            "uncertainty_reason": self.uncertainty_reason,
            "requires_human_review": self.requires_human_review,
            "vari_evidence": self.vari_evidence.to_dict() if self.vari_evidence is not None else None,
        }


@dataclass
class VisionAnalyzerConfig:
    """Configuration for Groq multimodal vision model analysis."""
    groq_api_base: str = field(
        default_factory=lambda: os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip()
    )
    vision_model: str = field(
        default_factory=lambda: os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()
    )
    timeout_seconds: float = 30.0
    max_retries: int = 2
    max_regions_to_analyze: int = 6
    min_crop_dimension: int = 32  # Groq Qwen vision requires >= 32 pixels in each dimension
    composite_layout: str = "side_by_side"  # "side_by_side" or "multi_image"


# =============================================================================
# 1. OPTICAL VARI IMPLEMENTATION
# =============================================================================

def compute_vari_map(
    rgb_image: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[Optional[np.ndarray], str, List[str]]:
    """Computes the Visible Atmospherically Resistant Index (VARI) for an RGB optical image.

    Formula:
        VARI = (Green - Red) / (Green + Red - Blue)

    Args:
        rgb_image: (H, W, 3) optical image array in [0, 255] uint8 or [0.0, 1.0] float.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Tuple of (vari_map, status_string, limitations_list)
    """
    limitations = [
        "VARI is derived strictly from visible optical bands (Red, Green, Blue) and does NOT contain Near-Infrared (NIR) data.",
        "VARI is an optical greenness proxy, NOT calibrated multispectral NDVI, and cannot measure subsurface chlorophyll absorption.",
        "Variations in illumination angle, sensor gain, or atmospheric haze can skew RGB values and delta indices.",
    ]

    # Validate channel structure (strictly requires 3 RGB channels)
    if not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        return (
            None,
            "unsupported_channels",
            ["Input image lacks 3-channel RGB bands. VARI calculation requires standard Red, Green, and Blue optical channels."],
        )

    # Check for invalid values
    if np.isnan(rgb_image).any() or np.isinf(rgb_image).any():
        return (
            None,
            "invalid_values",
            ["Input image contains NaN or infinite values."],
        )

    arr = rgb_image[:, :, :3].astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0

    # Illumination and exposure checks
    mean_val = float(np.mean(arr))
    if mean_val < 0.12:
        limitations.append("Low dynamic range or underexposed region; optical index may be sensitive to noise.")
    elif mean_val > 0.88:
        limitations.append("High brightness or overexposed region (specular reflection / cloud edge); index may saturate.")

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Check for flat zero images
    if float(np.std(arr)) < 1e-4:
        return (
            np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32),
            "zero_variance",
            limitations + ["Input image has zero variance (completely uniform or black)."],
        )

    numerator = g - r
    denominator = g + r - b

    # Safe numerical handling for near-zero denominators
    near_zero = np.abs(denominator) < eps
    safe_denom = np.where(near_zero, eps, denominator)

    vari = numerator / safe_denom

    # Bounded to [-1.0, 1.0]
    vari_clamped = np.clip(vari, -1.0, 1.0)
    vari_clamped[near_zero] = 0.0

    return vari_clamped.astype(np.float32), "supported", limitations


def analyze_regional_vari(
    t1_crop: np.ndarray,
    t2_crop: np.ndarray,
    mask_crop: Optional[np.ndarray] = None,
) -> RegionalVARIEvidence:
    """Calculates mean VARI and delta shift for a region across T1 and T2 observations."""
    v1_map, status1, lim1 = compute_vari_map(t1_crop)
    v2_map, status2, lim2 = compute_vari_map(t2_crop)

    if status1 != "supported" or status2 != "supported" or v1_map is None or v2_map is None:
        return RegionalVARIEvidence(
            status="unsupported_channels" if (status1 == "unsupported_channels" or status2 == "unsupported_channels") else "invalid",
            t1_mean_vari=None,
            t2_mean_vari=None,
            delta_vari=None,
            interpretation="VARI optical vegetation proxy is unavailable due to incompatible channel structure.",
            limitations=list(dict.fromkeys(lim1 + lim2)),
        )

    # Determine region evaluation pixels
    if mask_crop is not None and mask_crop.shape[:2] == v1_map.shape[:2] and np.count_nonzero(mask_crop > 0) > 0:
        eval_pixels = mask_crop > 0
    else:
        eval_pixels = np.ones_like(v1_map, dtype=bool)

    m1 = float(np.mean(v1_map[eval_pixels]))
    m2 = float(np.mean(v2_map[eval_pixels]))
    delta = m2 - m1

    # Formulate evidence interpretation
    if delta >= 0.15:
        interp = (
            f"Vegetation greenness proxy increase (delta VARI: +{delta:.3f}, from {m1:.3f} to {m2:.3f}). "
            f"Suggestive of visible canopy expansion, agricultural greening, or grass emergence (RGB optical proxy only)."
        )
    elif delta <= -0.15:
        interp = (
            f"Vegetation greenness proxy decrease (delta VARI: {delta:.3f}, from {m1:.3f} to {m2:.3f}). "
            f"Suggestive of vegetation removal, grading, or canopy loss (RGB optical proxy only)."
        )
    else:
        interp = (
            f"Stable optical greenness index (delta VARI: {delta:+.3f}, from {m1:.3f} to {m2:.3f}). "
            f"Visible green-to-red ratio remained relatively constant across observations."
        )

    all_limitations = list(dict.fromkeys(lim1 + lim2))

    return RegionalVARIEvidence(
        status="supported",
        t1_mean_vari=m1,
        t2_mean_vari=m2,
        delta_vari=delta,
        interpretation=interp,
        limitations=all_limitations,
    )


# =============================================================================
# 2. CROP COMPOSITING FOR VISION INGESTION
# =============================================================================

def prepare_region_composite_image(
    t1_crop: np.ndarray,
    t2_crop: np.ndarray,
    mask_crop: Optional[np.ndarray] = None,
    min_dim: int = 32,
) -> str:
    """Prepares a side-by-side composite crop [T1 | T2 | MaskOverlay] in base64 PNG.

    Ensures each crop meets the minimum dimension (32 px) required by Groq Qwen vision.
    Compositing avoids multi-image token explosion and respects Groq's 7,000 ITPM rate limit.
    """
    # Resize crops if smaller than minimum dimension (32 px)
    def ensure_min_size(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        if h < min_dim or w < min_dim:
            new_h = max(h, min_dim)
            new_w = max(w, min_dim)
            return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return arr

    c1 = ensure_min_size(t1_crop)
    c2 = ensure_min_size(t2_crop)

    # Standardize heights
    h1, w1 = c1.shape[:2]
    h2, w2 = c2.shape[:2]
    target_h = max(h1, h2)

    if h1 != target_h:
        c1 = cv2.resize(c1, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        c2 = cv2.resize(c2, (int(w2 * target_h / h2), target_h))

    # Optional 3rd panel: colored change mask overlay
    panels = [c1, c2]
    if mask_crop is not None:
        m = ensure_min_size(mask_crop)
        hm, wm = m.shape[:2]
        if hm != target_h:
            m = cv2.resize(m, (int(wm * target_h / hm), target_h), interpolation=cv2.INTER_NEAREST)
        
        # Color mask: red overlay [239, 68, 68] on dark background
        mask_rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
        mask_rgb[:] = [30, 41, 59]
        mask_rgb[m > 0] = [239, 68, 68]
        panels.append(mask_rgb)

    # Concatenate horizontally
    composite = np.concatenate(panels, axis=1)

    # Encode to base64 PNG
    pil_img = Image.fromarray(composite.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# 3. SEMANTIC CHANGE ANALYZER
# =============================================================================

class SemanticChangeAnalyzer:
    """Interprets localized bitemporal satellite changes using VARI and Groq Qwen Vision."""

    def __init__(self, config: Optional[VisionAnalyzerConfig] = None):
        self.config = config or VisionAnalyzerConfig()

    def is_configured(self) -> bool:
        """Verifies if Groq API key is present."""
        return bool(self.config.groq_api_key and len(self.config.groq_api_key) > 10)

    def _call_groq_vision(self, base64_image: str, prompt_text: str) -> Optional[str]:
        """Executes an authenticated multimodal completion call to Groq."""
        if not self.is_configured():
            logger.warning("Groq API key not configured; skipping vision model call.")
            return None

        headers = {
            "Authorization": f"Bearer {self.config.groq_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    resp = client.post(
                        f"{self.config.groq_api_base}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        return choices[0]["message"].get("content", "")
                    return None
                elif resp.status_code in (429, 503, 504):
                    logger.warning("Groq API transient error %d (attempt %d)", resp.status_code, attempt + 1)
                else:
                    logger.error("Groq Vision API error %d: %s", resp.status_code, resp.text[:200])
                    return None
            except Exception as e:
                logger.warning("Groq Vision connection failed (attempt %d): %s", attempt + 1, type(e).__name__)

        return None

    def _parse_vision_response(self, raw_text: Optional[str], region_id: int) -> Dict[str, Any]:
        """Parses and validates JSON from the vision model response."""
        fallback = {
            "region_id": region_id,
            "change_category": SemanticCategory.UNCERTAIN.value,
            "visual_observation": "Automated vision model response unavailable or could not be verified.",
            "evidence_for_category": ["Vision inference unverified; relying on deterministic detector mask."],
            "alternative_interpretations": ["Unspecified land-cover transition"],
            "uncertainty_reason": "Model response was empty or did not conform to structured JSON schema.",
            "requires_human_review": True,
        }

        if not raw_text:
            return fallback

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if m:
                cleaned = m.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return fallback
            else:
                return fallback

        # Validate category against approved taxonomy
        raw_cat = str(data.get("change_category", "")).strip().lower()
        cat = raw_cat if raw_cat in APPROVED_CATEGORIES else SemanticCategory.UNCERTAIN.value

        obs = str(data.get("visual_observation", "Visible structural differences detected.")).strip()
        evidence = data.get("evidence_for_category", ["Observed optical feature shift."])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        alt = data.get("alternative_interpretations", [])
        if not isinstance(alt, list):
            alt = [str(alt)]

        unc_reason = data.get("uncertainty_reason", None)
        if unc_reason is not None:
            unc_reason = str(unc_reason).strip()

        req_review = bool(data.get("requires_human_review", cat == SemanticCategory.UNCERTAIN.value))

        return {
            "region_id": region_id,
            "change_category": cat,
            "visual_observation": obs,
            "evidence_for_category": evidence,
            "alternative_interpretations": alt,
            "uncertainty_reason": unc_reason,
            "requires_human_review": req_review,
        }

    def analyze_regions(
        self,
        t1_image: np.ndarray,
        t2_image: np.ndarray,
        regions: List[ChangedRegion],
        binary_mask: Optional[np.ndarray] = None,
    ) -> List[RegionalSemanticInterpretation]:
        """Analyzes cropped change regions using VARI indicators and Groq Vision model.

        Limits analysis to at most max_regions_to_analyze (default 6).
        """
        results: List[RegionalSemanticInterpretation] = []
        capped_regions = regions[: self.config.max_regions_to_analyze]

        for reg in capped_regions:
            c_xmin, c_ymin, c_xmax, c_ymax = reg.context_bbox
            t1_crop = t1_image[c_ymin:c_ymax, c_xmin:c_xmax]
            t2_crop = t2_image[c_ymin:c_ymax, c_xmin:c_xmax]
            mask_crop = binary_mask[c_ymin:c_ymax, c_xmin:c_xmax] if binary_mask is not None else None

            # 1. Deterministic VARI evidence
            vari_ev = analyze_regional_vari(t1_crop, t2_crop, mask_crop=mask_crop)

            # 2. Multimodal Vision Model Stage (if configured and crops are valid)
            parsed_data: Optional[Dict[str, Any]] = None
            if self.is_configured() and t1_crop.size > 0 and t2_crop.size > 0:
                composite_b64 = prepare_region_composite_image(
                    t1_crop, t2_crop, mask_crop, min_dim=self.config.min_crop_dimension
                )

                prompt = (
                    "You are a satellite image analyst inspecting a localized bitemporal change region.\n"
                    "The provided image is a horizontal composite containing:\n"
                    "- Left Panel: T1 (earlier optical satellite observation)\n"
                    "- Middle Panel: T2 (later optical satellite observation)\n"
                    "- Right Panel: Binary change mask detected by the detector (red = changed)\n\n"
                    "TASK: Categorize the change type strictly into one of the following approved categories:\n"
                    f"{', '.join(sorted(APPROVED_CATEGORIES))}\n\n"
                    "CRITICAL RULES:\n"
                    "1. Distinguish actual visible observations from hypotheses.\n"
                    "2. If features are ambiguous, set change_category to 'uncertain' and explain why.\n"
                    "3. Return strictly a JSON object with keys:\n"
                    "   - region_id (integer)\n"
                    "   - change_category (string: one of approved taxonomy)\n"
                    "   - visual_observation (string: describe what visible shapes or colors changed)\n"
                    "   - evidence_for_category (list of strings: specific visible cues)\n"
                    "   - alternative_interpretations (list of strings: other possible explanations)\n"
                    "   - uncertainty_reason (string or null)\n"
                    "   - requires_human_review (boolean)"
                )

                try:
                    raw_output = self._call_groq_vision(composite_b64, prompt)
                except Exception as e:
                    logger.warning("Vision model invocation exception: %s", e)
                    raw_output = None
                parsed_data = self._parse_vision_response(raw_output, reg.region_id)

            # Fallback if vision model was unconfigured or skipped
            if parsed_data is None:
                # Deterministic heuristic fallback using VARI
                fallback_cat = SemanticCategory.UNCERTAIN.value
                fallback_ev = ["Vision model unconfigured; classified via deterministic optical variance."]
                fallback_obs = f"Detected spatial change cluster covering {reg.pixel_area} pixels ({reg.area_ratio*100:.2f}% of canvas)."

                if vari_ev.status == "supported" and vari_ev.delta_vari is not None:
                    if vari_ev.delta_vari >= 0.20:
                        fallback_cat = SemanticCategory.VEGETATION_INCREASE.value
                        fallback_ev.append(f"Significant positive greenness shift (+{vari_ev.delta_vari:.2f}).")
                    elif vari_ev.delta_vari <= -0.20:
                        fallback_cat = SemanticCategory.VEGETATION_DECREASE.value
                        fallback_ev.append(f"Significant negative greenness shift ({vari_ev.delta_vari:.2f}).")

                parsed_data = {
                    "region_id": reg.region_id,
                    "change_category": fallback_cat,
                    "visual_observation": fallback_obs,
                    "evidence_for_category": fallback_ev,
                    "alternative_interpretations": ["Construction, grading, or agricultural activity"],
                    "uncertainty_reason": "Vision model inference unavailable; relying on deterministic metrics.",
                    "requires_human_review": True,
                }

            interp_obj = RegionalSemanticInterpretation(
                region_id=reg.region_id,
                change_category=parsed_data["change_category"],
                visual_observation=parsed_data["visual_observation"],
                evidence_for_category=parsed_data["evidence_for_category"],
                alternative_interpretations=parsed_data["alternative_interpretations"],
                uncertainty_reason=parsed_data["uncertainty_reason"],
                requires_human_review=parsed_data["requires_human_review"],
                vari_evidence=vari_ev,
            )
            results.append(interp_obj)

        return results
