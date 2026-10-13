"""Dedicated LLM Explanation Service using Groq-hosted Qwen 27B.

Provides evidence-grounded natural language explanations for completed
semantic search and bi-temporal change detection predictions without
altering original model outputs or research artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

from src.api.schemas import (
    ExplainChangeDetectionRequest,
    ExplainSearchRequest,
    ExplanationResponse,
    LLMStatusResponse,
)

logger = logging.getLogger(__name__)

# Load local environment variables if available
load_dotenv()


class LLMExplanationService:
    """Service layer interfacing with Groq API to serve Qwen 27B model explanations."""

    def __init__(self) -> None:
        self.api_base = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
        
        try:
            self.timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))
        except ValueError:
            self.timeout_seconds = 60.0

        # In-memory explanation cache to eliminate duplicate external calls
        self._cache: Dict[str, Dict[str, Any]] = {}

    def is_configured(self) -> bool:
        """Checks if a valid Groq API key is present."""
        return bool(self.api_key and len(self.api_key) > 10)

    def get_status(self, check_connectivity: bool = False) -> LLMStatusResponse:
        """Returns provider configuration and optional live connectivity status."""
        configured = self.is_configured()
        if not configured:
            return LLMStatusResponse(
                configured=False,
                provider="Groq",
                model=self.model,
                status="unconfigured",
                connectivity="unverified",
                message="GROQ_API_KEY is not configured in environment or .env file.",
            )

        connectivity = "unverified"
        status = "ready"
        msg = f"Groq Qwen 27B ({self.model}) explanation service is configured."

        if check_connectivity:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(f"{self.api_base}/models", headers=headers)
                if resp.status_code == 200:
                    connectivity = "verified"
                    status = "ready"
                    msg = f"Groq API connection verified. Model '{self.model}' is available."
                elif resp.status_code == 401:
                    connectivity = "failed"
                    status = "error"
                    msg = "Authentication failed: GROQ_API_KEY is invalid."
                else:
                    connectivity = "failed"
                    status = "error"
                    msg = f"Groq API returned HTTP {resp.status_code}."
            except Exception as e:
                connectivity = "failed"
                status = "error"
                msg = f"Connection error: {type(e).__name__}"

        return LLMStatusResponse(
            configured=True,
            provider="Groq",
            model=self.model,
            status=status,
            connectivity=connectivity,
            message=msg,
        )

    def _call_groq_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Executes an authenticated chat completion call to Groq with strict timeout and error handling."""
        if not self.is_configured():
            raise RuntimeError("GROQ_API_KEY is not configured. Please supply a valid Groq API key in .env.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # Low variance for evidence-grounded explanations
            "max_tokens": 700,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
                raise ValueError("Groq API returned an empty completion response.")

            if resp.status_code == 401:
                raise PermissionError("Groq authentication failed: Invalid or expired API credentials.")
            elif resp.status_code == 429:
                raise RuntimeError("Groq API rate limit or quota exceeded. Please retry shortly.")
            elif resp.status_code == 404:
                raise RuntimeError(f"Configured model '{self.model}' was not found on Groq.")
            else:
                error_msg = resp.text[:200]
                raise RuntimeError(f"Groq API request failed with HTTP {resp.status_code}: {error_msg}")

        except httpx.TimeoutException:
            raise TimeoutError(f"Groq API timed out after {self.timeout_seconds} seconds.")
        except (PermissionError, RuntimeError, ValueError):
            raise
        except Exception as e:
            logger.error("Unexpected error during Groq API call: %s", type(e).__name__)
            raise RuntimeError(f"Failed to communicate with Groq API: {type(e).__name__}")

    def _parse_explanation_json(self, raw_text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from model response, handling Markdown code fences."""
        cleaned = raw_text.strip()
        # Strip markdown json code fences if present
        if cleaned.startswith("```"):
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if fence_match:
                cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Fallback regex extraction of JSON object
        json_obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_obj_match:
            try:
                data = json.loads(json_obj_match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        # Graceful fallback structure if LLM failed JSON formatting
        return {
            "prediction_summary": raw_text.strip()[:400],
            "supporting_evidence": ["Evidence extracted from verified backend telemetry."],
            "limitations": ["Structured response formatting could not be fully parsed from provider output."],
            "unknowns": ["Internal token representation details."],
        }

    def explain_change_detection(self, request: ExplainChangeDetectionRequest) -> ExplanationResponse:
        """Explains a bi-temporal change detection prediction based strictly on structured evidence."""
        cache_key = f"cd_{request.pair_id}_{request.model_id}_{request.threshold}_{request.gamma}_{request.changed_pixels}"
        if cache_key in self._cache:
            return ExplanationResponse(**self._cache[cache_key])

        # Map display properties and detector family characteristics
        model_name = request.model_id
        is_qat = "QAT" in model_name or "qat" in model_name.lower()
        is_bit = "BIT" in model_name or "bit" in model_name.lower()
        is_classical = any(b in model_name for b in ("B1", "B2", "B3", "Pixel_Diff", "SSIM", "CVA"))

        family_desc = (
            "Quality-Aware Temporal Change Detection (attenuates false alarms via feature quality weighting)"
            if is_qat
            else (
                "Bitemporal Image Transformer (dual-branch attention with spatial-temporal token fusion)"
                if is_bit
                else (
                    "Classical Low-Complexity Baseline (unsupervised difference metric with empirical threshold)"
                    if is_classical
                    else "Learned Siamese Feature Difference Architecture"
                )
            )
        )

        thresh_val = request.threshold if request.threshold is not None else 0.50
        changed_px = request.changed_pixels if request.changed_pixels is not None else "Not calculated"
        change_pct = f"{request.change_ratio_pct:.2f}%" if request.change_ratio_pct is not None else "Not calculated"

        evidence_dict = {
            "source_model": model_name,
            "architecture_family": family_desc,
            "test_pair_id": request.pair_id,
            "decision_threshold_tau": thresh_val,
            "quality_attenuation_gamma": request.gamma,
            "predicted_changed_pixels": changed_px,
            "predicted_changed_area_percentage": change_pct,
            "ground_truth_metrics": request.metrics or "No ground truth available for this scene.",
        }

        system_prompt = (
            "You are an AI research scientist and explainability assistant for a satellite image analysis system. "
            "Your objective is to provide a concise, factual, evidence-grounded explanation of a change detection result. "
            "CRITICAL RULES:\n"
            "1. You are the EXPLANATION model (Qwen 27B on Groq). The PREDICTION was generated by the SOURCE MODEL.\n"
            "2. You have NOT inspected raw image pixels yourself. You interpret only the provided numerical and structural evidence.\n"
            "3. Never invent metrics, scores, or findings not present in the evidence.\n"
            "4. Return strictly a JSON object with keys: prediction_summary, supporting_evidence, limitations, unknowns.\n"
            "5. 'supporting_evidence' must list 2-4 factual points from the telemetry.\n"
            "6. 'limitations' must state real operational/algorithmic caveats of the source model.\n"
            "7. 'unknowns' must list what cannot be concluded (e.g. semantic building type, unverified pixel classes)."
        )

        user_prompt = (
            f"Please explain this bi-temporal change detection result based on the following verified telemetry:\n\n"
            f"{json.dumps(evidence_dict, indent=2)}\n\n"
            f"Produce your response strictly as a JSON object with keys:\n"
            f"- prediction_summary (string: 2-3 sentences interpreting the result)\n"
            f"- supporting_evidence (list of strings: specific facts from evidence)\n"
            f"- limitations (list of strings: documented limitations and risks)\n"
            f"- unknowns (list of strings: what remains unverified or missing)"
        )

        raw_output = self._call_groq_chat(system_prompt, user_prompt)
        parsed = self._parse_explanation_json(raw_output)

        resp = ExplanationResponse(
            explanation_model=self.model,
            provider="Groq",
            source_model=model_name,
            task="change_detection",
            prediction_summary=parsed.get("prediction_summary", "Change detection prediction interpreted from evidence."),
            supporting_evidence=parsed.get("supporting_evidence", [
                f"Model {model_name} evaluated using threshold tau* = {thresh_val}.",
                f"Predicted change extent: {change_pct} of evaluated scene.",
            ]),
            limitations=parsed.get("limitations", [
                "Inference executed under standardized CPU constraints.",
                "Threshold calibrated on validation split; subject to false alarms under illumination differences.",
            ]),
            unknowns=parsed.get("unknowns", [
                "Fine-grained semantic category of changed structures.",
                "Ground-truth pixel labels if unlabelled test sample.",
            ]),
        )

        self._cache[cache_key] = resp.model_dump()
        return resp

    def explain_semantic_search(self, request: ExplainSearchRequest) -> ExplanationResponse:
        """Explains a semantic retrieval ranking candidate based strictly on verified evidence."""
        cache_key = f"search_{request.query}_{request.method}_{request.image_id}_{request.rank}"
        if cache_key in self._cache:
            return ExplanationResponse(**self._cache[cache_key])

        method_name = request.method.lower()
        is_clip = "clip" in method_name or "ensemble" in method_name
        method_desc = (
            "OpenAI CLIP ViT-B/32 Cross-Modal Embedding (cosine similarity in 512-dimensional joint text-image space)"
            if "prompt_ensemble" not in method_name and is_clip
            else (
                "CLIP ViT-B/32 + Remote Sensing 5-Template Prompt Ensemble (domain-tailored query embedding)"
                if "prompt_ensemble" in method_name
                else "BM25 Lexical Term Matching Baseline (sparse inverse document frequency word overlap)"
            )
        )

        score_type = request.score_type or ("Cosine Similarity" if is_clip else "BM25 Score")
        score_val = (
            f"{request.score * 100:.1f}%"
            if is_clip and request.score <= 1.0
            else f"{request.score:.3f}"
        )

        evidence_dict = {
            "natural_language_query": request.query,
            "retrieval_method": method_desc,
            "candidate_rank": request.rank,
            "similarity_score": score_val,
            "score_interpretation": score_type,
            "image_identifier": request.image_id,
            "filename": request.filename or request.image_id,
            "scene_category": request.category or "Unspecified remote sensing scene",
            "verified_reference_captions": request.captions or ["No descriptive reference captions recorded."],
        }

        system_prompt = (
            "You are an AI research scientist and explainability assistant for a satellite image analysis system. "
            "Your objective is to provide a concise, factual explanation for why a specific satellite image candidate was retrieved. "
            "CRITICAL RULES:\n"
            "1. You are the EXPLANATION model (Qwen 27B on Groq). The RETRIEVAL was performed by the SOURCE MODEL.\n"
            "2. Clearly explain what the score means (e.g. vector cosine angle for CLIP vs lexical keyword match for BM25).\n"
            "3. Never treat BM25 scores as cosine similarities or vice versa.\n"
            "4. Return strictly a JSON object with keys: prediction_summary, supporting_evidence, limitations, unknowns.\n"
            "5. Do NOT invent image features or scores not supplied in the evidence."
        )

        user_prompt = (
            f"Please explain this semantic search ranking result based on the following verified telemetry:\n\n"
            f"{json.dumps(evidence_dict, indent=2)}\n\n"
            f"Produce your response strictly as a JSON object with keys:\n"
            f"- prediction_summary (string: 2-3 sentences explaining the retrieval match)\n"
            f"- supporting_evidence (list of strings: specific facts from evidence)\n"
            f"- limitations (list of strings: retrieval method trade-offs and risks)\n"
            f"- unknowns (list of strings: missing information or unverified visual details)"
        )

        raw_output = self._call_groq_chat(system_prompt, user_prompt)
        parsed = self._parse_explanation_json(raw_output)

        resp = ExplanationResponse(
            explanation_model=self.model,
            provider="Groq",
            source_model=request.method.upper(),
            task="semantic_search",
            prediction_summary=parsed.get("prediction_summary", f"Image ranked #{request.rank} for query '{request.query}'."),
            supporting_evidence=parsed.get("supporting_evidence", [
                f"Query matched candidate #{request.rank} with {score_type}: {score_val}.",
                f"Assigned scene category: {request.category or 'Aerial scene'}.",
            ]),
            limitations=parsed.get("limitations", [
                "Zero-shot embeddings may occasionally conflate visually similar land-use textures.",
                "BM25 lexical scoring cannot comprehend visual semantics beyond literal token matches.",
            ]),
            unknowns=parsed.get("unknowns", [
                "Sub-object spatial coordinates within the 256x256 aerial patch.",
                "Temporal acquisition conditions (sun angle, season, cloud cover).",
            ]),
        )

        self._cache[cache_key] = resp.model_dump()
        return resp


# Singleton service instance
llm_explanation_service = LLMExplanationService()
