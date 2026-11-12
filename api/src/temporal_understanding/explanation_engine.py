"""Dual-stage natural language explanation engine for temporal satellite changes.

Aggregates detector telemetry, registration diagnostics, region geometry,
VARI spectral indicators, and vision-model interpretations to synthesize an
evidence-grounded academic report using Groq Qwen 27B without hallucinating metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from src.temporal_understanding.registration import RegistrationDiagnostics
from src.temporal_understanding.region_extractor import ChangedRegion
from src.temporal_understanding.semantic_analyzer import RegionalSemanticInterpretation

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CONTRACTS
# =============================================================================

@dataclass
class TemporalExplanationConfig:
    """Configuration for Groq Qwen 27B natural language explanation generation."""
    groq_api_base: str = field(
        default_factory=lambda: os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip()
    )
    text_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
    )
    timeout_seconds: float = 45.0
    max_retries: int = 2


@dataclass
class TemporalUnderstandingReport:
    """Evidence-grounded comprehensive academic report for bitemporal satellite analysis."""
    overall_summary: str
    detector_evidence: List[str]
    region_observations: List[Dict[str, Any]]
    semantic_interpretations: List[Dict[str, Any]]
    registration_and_quality_warnings: List[str]
    scientific_limitations: List[str]
    unknown_variables: List[str]
    human_review_recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts report to dictionary representation."""
        return {
            "overall_summary": self.overall_summary,
            "detector_evidence": self.detector_evidence,
            "region_observations": self.region_observations,
            "semantic_interpretations": self.semantic_interpretations,
            "registration_and_quality_warnings": self.registration_and_quality_warnings,
            "scientific_limitations": self.scientific_limitations,
            "unknown_variables": self.unknown_variables,
            "human_review_recommendations": self.human_review_recommendations,
            "metadata": self.metadata,
        }


# =============================================================================
# EXPLANATION ENGINE IMPLEMENTATION
# =============================================================================

class TemporalExplanationEngine:
    """Stage B reasoning engine synthesizing multimodal telemetry into structured explanations."""

    def __init__(self, config: Optional[TemporalExplanationConfig] = None):
        self.config = config or TemporalExplanationConfig()

    def is_configured(self) -> bool:
        """Verifies if Groq API key is present."""
        return bool(self.config.groq_api_key and len(self.config.groq_api_key) > 10)

    def _call_groq_text(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Executes chat completion against Groq Qwen 27B model."""
        if not self.is_configured():
            logger.warning("Groq API key not configured; using deterministic report fallback.")
            return None

        headers = {
            "Authorization": f"Bearer {self.config.groq_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.text_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # Strict factual determinism
            "max_tokens": 1000,
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
                    logger.warning("Groq text API transient error %d (attempt %d)", resp.status_code, attempt + 1)
                else:
                    logger.error("Groq text API error %d: %s", resp.status_code, resp.text[:200])
                    return None
            except Exception as e:
                logger.warning("Groq text connection failed (attempt %d): %s", attempt + 1, type(e).__name__)

        return None

    def _parse_llm_report(self, raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
        """Extracts and parses JSON object from LLM response."""
        if not raw_text:
            return None

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if m:
                cleaned = m.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    def generate_explanation(
        self,
        detector_telemetry: Dict[str, Any],
        regions: List[ChangedRegion],
        semantic_findings: Optional[List[RegionalSemanticInterpretation]] = None,
        registration_diagnostics: Optional[RegistrationDiagnostics] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TemporalUnderstandingReport:
        """Synthesizes structured evidence into a formal academic understanding report.

        Args:
            detector_telemetry: Dict with keys: model_id, threshold, total_changed_pixels,
                                total_changed_area_pct, latency_ms, qat_diagnostics.
            regions: List of extracted ChangedRegion objects.
            semantic_findings: Optional list of RegionalSemanticInterpretation objects.
            registration_diagnostics: Optional RegistrationDiagnostics from Stage 1.
            metadata: Optional user metadata (dates, location, gsd_meters).

        Returns:
            TemporalUnderstandingReport separating evidence, interpretations, and unknowns.
        """
        meta = metadata or {}
        gsd = meta.get("gsd_meters", None)

        # 1. Compile objective detector evidence
        det_model = detector_telemetry.get("model_id", "M9_BIT_CD")
        tau = detector_telemetry.get("threshold", 0.4500)
        ch_pixels = detector_telemetry.get("total_changed_pixels", 0)
        ch_pct = detector_telemetry.get("total_changed_area_pct", 0.0)

        det_evidence = [
            f"Core change detection executed using {det_model} under decision threshold tau* = {tau:.4f}.",
            f"Total detected changed area: {ch_pixels:,} pixels ({ch_pct:.2f}% of evaluated scene).",
            f"Spatial clustering identified {len(regions)} primary significant change regions.",
        ]

        if "qat_diagnostics" in detector_telemetry and detector_telemetry["qat_diagnostics"]:
            qat_diag = detector_telemetry["qat_diagnostics"]
            det_evidence.append(
                f"Quality gating telemetry: Mean registration discrepancy = {qat_diag.get('mean_registration_discrepancy', 0.0):.4f}, "
                f"Illumination gradient = {qat_diag.get('mean_illumination_gradient', 0.0):.4f}."
            )

        # 2. Compile registration warnings
        reg_warnings: List[str] = []
        if registration_diagnostics is not None:
            reg_status = registration_diagnostics.registration_status.value
            conf = registration_diagnostics.confidence_score
            reg_warnings.append(f"Co-registration status: {reg_status} (confidence score: {conf:.3f}).")
            if registration_diagnostics.warnings:
                reg_warnings.extend(registration_diagnostics.warnings)
        else:
            reg_warnings.append("Images assumed pre-registered; automated co-registration was bypassed.")

        # 3. Compile regional observations and semantic interpretations
        region_obs: List[Dict[str, Any]] = []
        semantic_interps: List[Dict[str, Any]] = []

        findings_map = {f.region_id: f for f in (semantic_findings or [])}

        for r in regions:
            # Physical area strictly when GSD is available
            area_str = f"{r.pixel_area:,} pixels ({r.area_ratio*100:.2f}% of canvas)"
            if gsd is not None and gsd > 0:
                area_m2 = r.pixel_area * (gsd ** 2)
                area_str += f" [~{area_m2:,.1f} m² based on GSD={gsd:.2f} m/px]"

            r_dict = {
                "region_id": r.region_id,
                "bbox": list(r.bbox),
                "area_description": area_str,
                "centroid": [round(r.centroid[0], 1), round(r.centroid[1], 1)],
            }
            if r.mean_probability is not None:
                r_dict["mean_detector_activation"] = round(r.mean_probability, 4)

            region_obs.append(r_dict)

            # Link semantic findings
            finding = findings_map.get(r.region_id)
            if finding is not None:
                sem_dict = {
                    "region_id": r.region_id,
                    "proposed_category": finding.change_category,
                    "visual_observation": finding.visual_observation,
                    "supporting_cues": finding.evidence_for_category,
                    "alternative_hypotheses": finding.alternative_interpretations,
                    "requires_human_review": finding.requires_human_review,
                }
                if finding.vari_evidence and finding.vari_evidence.delta_vari is not None:
                    sem_dict["optical_vari_shift"] = {
                        "delta": round(finding.vari_evidence.delta_vari, 4),
                        "t1_mean": round(finding.vari_evidence.t1_mean_vari, 4) if finding.vari_evidence.t1_mean_vari else None,
                        "t2_mean": round(finding.vari_evidence.t2_mean_vari, 4) if finding.vari_evidence.t2_mean_vari else None,
                        "interpretation": finding.vari_evidence.interpretation,
                    }
                semantic_interps.append(sem_dict)
            else:
                semantic_interps.append({
                    "region_id": r.region_id,
                    "proposed_category": "unverified",
                    "visual_observation": "Visual semantic classification unavailable for this region.",
                    "supporting_cues": ["Detector mask confirmed presence of significant radiometric change."],
                    "alternative_hypotheses": ["Building construction, demolition, vegetation change, or grading"],
                    "requires_human_review": True,
                })

        # 4. Mandatory scientific limitations and unknowns
        limitations = [
            "Inference executed under CPU constraints using patch-based bitemporal tiling.",
            "Visual categorizations are probabilistic interpretations, NOT verified ground truth.",
            "Optical VARI is an RGB visible greenness proxy and must NOT be equated with multispectral NDVI.",
        ]
        if gsd is None:
            limitations.append("Ground Sample Distance (GSD) metadata is absent; physical area measurements in square metres cannot be computed.")

        unknowns = [
            "Specific commercial or residential usage of newly identified structural foundations.",
            "Underlying subterranean infrastructure (utilities, drainage, or pipelines).",
            "Exact temporal timestamps of change onset if observation dates are approximate or omitted.",
        ]

        human_reviews = [
            f"Inspect Region #{r.region_id} on high-resolution optical basemaps if ground truth verification is required."
            for r in regions if findings_map.get(r.region_id, None) and findings_map[r.region_id].requires_human_review
        ]
        if not human_reviews and regions:
            human_reviews.append(f"Spot-check primary change site (Region #1, {regions[0].pixel_area:,} pixels).")

        # 5. Synthesis via Groq Qwen 27B if configured
        llm_report_data: Optional[Dict[str, Any]] = None
        if self.is_configured():
            evidence_payload = {
                "source_detector": det_model,
                "detector_evidence": det_evidence,
                "registration_status": reg_warnings,
                "region_observations": region_obs,
                "semantic_interpretations": semantic_interps,
                "temporal_metadata": meta,
            }

            system_prompt = (
                "You are an expert remote sensing scientist synthesizing an academic satellite change report. "
                "CRITICAL GOVERNANCE RULES:\n"
                "1. Strictly separate OBJECTIVE DETECTOR EVIDENCE from PROBABILISTIC INTERPRETATIONS.\n"
                "2. Never claim uncalibrated model scores are statistical probabilities.\n"
                "3. Never invent physical areas in square metres unless GSD is explicitly provided in metadata.\n"
                "4. Explicitly state limitations and unknowns.\n"
                "5. Return strictly a JSON object with keys: overall_summary, detector_evidence, "
                "scientific_limitations, unknown_variables, human_review_recommendations."
            )

            user_prompt = (
                f"Please synthesize a rigorous, factual temporal understanding report from the following verified telemetry:\n\n"
                f"{json.dumps(evidence_payload, indent=2)}\n\n"
                f"Produce your output strictly as a JSON object with keys:\n"
                f"- overall_summary (string: 2-3 sentences summarizing the detected bi-temporal changes)\n"
                f"- detector_evidence (list of strings: factual numbers and model parameters)\n"
                f"- scientific_limitations (list of strings: caveats and bounds)\n"
                f"- unknown_variables (list of strings: what cannot be verified)\n"
                f"- human_review_recommendations (list of strings: actionable guidance)"
            )

            raw_resp = self._call_groq_text(system_prompt, user_prompt)
            llm_report_data = self._parse_llm_report(raw_resp)

        # 6. Assemble final report
        summary_text = (
            llm_report_data.get("overall_summary")
            if llm_report_data and llm_report_data.get("overall_summary")
            else (
                f"Bitemporal change detection identified {len(regions)} significant change regions covering "
                f"{ch_pixels:,} pixels ({ch_pct:.2f}% of canvas) using {det_model} (tau* = {tau:.4f}). "
                f"Localized interpretations suggest "
                f"{', '.join(set(s.get('proposed_category', 'change') for s in semantic_interps))}."
            )
        )

        final_evidence = (
            llm_report_data.get("detector_evidence")
            if llm_report_data and llm_report_data.get("detector_evidence")
            else det_evidence
        )
        final_limits = (
            llm_report_data.get("scientific_limitations")
            if llm_report_data and llm_report_data.get("scientific_limitations")
            else limitations
        )
        final_unknowns = (
            llm_report_data.get("unknown_variables")
            if llm_report_data and llm_report_data.get("unknown_variables")
            else unknowns
        )
        final_reviews = (
            llm_report_data.get("human_review_recommendations")
            if llm_report_data and llm_report_data.get("human_review_recommendations")
            else human_reviews
        )

        return TemporalUnderstandingReport(
            overall_summary=summary_text,
            detector_evidence=final_evidence,
            region_observations=region_obs,
            semantic_interpretations=semantic_interps,
            registration_and_quality_warnings=reg_warnings,
            scientific_limitations=final_limits,
            unknown_variables=final_unknowns,
            human_review_recommendations=final_reviews,
            metadata={
                "model_id": det_model,
                "text_explainer_model": self.config.text_model,
                "regions_analyzed": len(regions),
                "has_gsd": gsd is not None,
                "has_vision_findings": semantic_findings is not None and len(semantic_findings) > 0,
            },
        )
