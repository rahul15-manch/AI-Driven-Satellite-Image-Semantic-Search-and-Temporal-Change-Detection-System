"""FastAPI routing endpoints for semantic retrieval, change detection, and benchmarking."""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

from src.api.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    PairInfo,
    BenchmarkResponse,
    LLMStatusResponse,
    ExplainChangeDetectionRequest,
    ExplainSearchRequest,
    ExplanationResponse,
    TemporalAnalysisResponse,
)
from src.api.service import (
    retrieval_service,
    change_detection_service,
    benchmark_service,
    llm_explanation_service,
    temporal_understanding_service,
    ROOT,
)

router = APIRouter(tags=["API"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Service liveness and capability readiness check."""
    return HealthResponse()


@router.post("/search", response_model=SearchResponse)
def execute_search(request: SearchRequest) -> SearchResponse:
    """Executes cross-modal or lexical semantic search across RSICD test gallery."""
    try:
        outcome = retrieval_service.search(
            query=request.query,
            method=request.method,
            top_k=request.top_k,
        )
        return SearchResponse(**outcome)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval indices not found: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )


@router.get("/pairs", response_model=List[PairInfo])
def get_test_pairs() -> List[PairInfo]:
    """Lists verified LEVIR-CD test scene pairs available for change detection."""
    pairs = change_detection_service.list_available_pairs()
    return [PairInfo(**p) for p in pairs]


@router.post("/change-detection", response_model=ChangeDetectionResponse)
def execute_change_detection(request: ChangeDetectionRequest) -> ChangeDetectionResponse:
    """Executes bi-temporal change detection on verified test scenes or uploaded images."""
    try:
        img_a: Optional[np.ndarray] = None
        img_b: Optional[np.ndarray] = None

        # Decode uploaded base64 images if provided
        if request.image_t1_base64 and request.image_t2_base64:
            def decode_b64_img(b64_str: str) -> np.ndarray:
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                data = base64.b64decode(b64_str)
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                return np.array(pil_img, dtype=np.uint8)

            img_a = decode_b64_img(request.image_t1_base64)
            img_b = decode_b64_img(request.image_t2_base64)
            if img_a.shape != img_b.shape:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image shape mismatch: T1 is {img_a.shape}, T2 is {img_b.shape}",
                )

        result = change_detection_service.detect_change(
            pair_id=request.pair_id if not (img_a is not None and img_b is not None) else None,
            model_id=request.model_id,
            threshold_override=request.threshold,
            gamma=request.gamma,
            img_a=img_a,
            img_b=img_b,
            return_diagnostics=request.return_diagnostics,
        )
        return ChangeDetectionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Change detection inference error: {str(e)}",
        )


@router.get("/benchmark", response_model=BenchmarkResponse)
def get_benchmark() -> BenchmarkResponse:
    """Loads standardized Milestone M11 benchmarking results and statistical outputs."""
    try:
        payload = benchmark_service.get_benchmark_payload()
        return BenchmarkResponse(**payload)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"M11 benchmark artifacts missing: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed loading benchmark data: {str(e)}",
        )


@router.get("/image/{dataset}/{path:path}")
def serve_dataset_image(dataset: str, path: str):
    """Safely serves optical satellite images with strict path traversal prevention."""
    dataset_key = dataset.lower().strip()
    if dataset_key not in {"rsicd", "levir_cd"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset '{dataset}'. Must be 'rsicd' or 'levir_cd'.",
        )

    # Base directory for dataset
    if dataset_key == "rsicd":
        base_dir = (ROOT / "data" / "raw" / "rsicd" / "images").resolve()
    else:
        raw_levir = (ROOT / "data" / "raw" / "levir_cd").resolve()
        demo_levir = (ROOT / "data" / "demo_samples").resolve()
        if raw_levir.exists():
            base_dir = raw_levir
        else:
            base_dir = demo_levir
            if path.startswith("test/"):
                path = path[5:]

    target_path = (base_dir / path).resolve()

    # Security check: Ensure target_path starts with base_dir (prevent directory traversal)
    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Path traversal detected.",
        )

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found at path: {path}",
        )

    # Content type
    suffix = target_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return FileResponse(target_path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


@router.get("/figures/{figure_name}")
def serve_research_figure(figure_name: str):
    """Serves generated M13, M11, M10, M6, or M5 research publication figures."""
    clean_name = Path(figure_name).name
    suffix = Path(figure_name).suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".svg"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid figure filename format.",
        )

    search_dirs = [
        ROOT / "experiments" / "figures" / "m13",
        ROOT / "experiments" / "figures" / "m11",
        ROOT / "experiments" / "figures" / "m10",
        ROOT / "experiments" / "figures" / "m6",
        ROOT / "experiments" / "figures" / "m5",
    ]
    figures_base = (ROOT / "experiments" / "figures").resolve()

    for fig_dir in search_dirs:
        target = (fig_dir / clean_name).resolve()
        # Security check: ensure path is strictly within experiments/figures
        if target.is_relative_to(figures_base) and target.exists() and target.is_file():
            media_type = "image/svg+xml" if suffix == ".svg" else ("image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png")
            return FileResponse(target, media_type=media_type, headers={"Cache-Control": "public, max-age=3600"})

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Research figure '{clean_name}' not found.",
    )


# ==============================================================================
# LLM EXPLAINABILITY ENDPOINTS (Groq-hosted Qwen 27B)
# ==============================================================================

@router.get("/llm/status", response_model=LLMStatusResponse)
def get_llm_status(check: bool = Query(default=False, description="Verify live connectivity against Groq API")) -> LLMStatusResponse:
    """Checks Groq Qwen 27B explainability service readiness and connectivity."""
    return llm_explanation_service.get_status(check_connectivity=check)


@router.post("/explain/change-detection", response_model=ExplanationResponse)
def explain_change_detection_result(request: ExplainChangeDetectionRequest) -> ExplanationResponse:
    """Generates an evidence-grounded natural language explanation for a change detection prediction."""
    try:
        return llm_explanation_service.explain_change_detection(request)
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Groq API authentication failed. Please verify GROQ_API_KEY in .env.",
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Groq API explanation timed out: {str(e)}",
        )
    except RuntimeError as e:
        msg = str(e)
        if "rate limit" in msg.lower() or "429" in msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq API rate limit or quota exceeded. Please retry in a few moments.",
            )
        elif "not configured" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Groq LLM explanation service is not configured. Add GROQ_API_KEY to .env.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Explanation generation failed: {msg}",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected failure during explanation generation.",
        )


@router.post("/explain/semantic-search", response_model=ExplanationResponse)
def explain_semantic_search_result(request: ExplainSearchRequest) -> ExplanationResponse:
    """Generates an evidence-grounded natural language explanation for a semantic retrieval result."""
    try:
        return llm_explanation_service.explain_semantic_search(request)
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Groq API authentication failed. Please verify GROQ_API_KEY in .env.",
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Groq API explanation timed out: {str(e)}",
        )
    except RuntimeError as e:
        msg = str(e)
        if "rate limit" in msg.lower() or "429" in msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Groq API rate limit or quota exceeded. Please retry in a few moments.",
            )
        elif "not configured" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Groq LLM explanation service is not configured. Add GROQ_API_KEY to .env.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Explanation generation failed: {msg}",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected failure during explanation generation.",
        )


# ==============================================================================
# PHASE 3C: TEMPORAL UNDERSTANDING ENDPOINTS
# ==============================================================================

@router.post("/temporal/analyze", response_model=TemporalAnalysisResponse)
async def analyze_temporal_pair(
    file_t1: UploadFile = File(..., description="Earlier observation optical satellite image (T1)"),
    file_t2: UploadFile = File(..., description="Later observation optical satellite image (T2)"),
    observation_date_t1: Optional[str] = Form(None, description="Observation timestamp/date for T1"),
    observation_date_t2: Optional[str] = Form(None, description="Observation timestamp/date for T2"),
    gsd_meters_per_pixel: Optional[float] = Form(None, description="Ground Sample Distance in meters per pixel"),
    gsd: Optional[float] = Form(None, description="Alias for gsd_meters_per_pixel"),
    description: Optional[str] = Form(None, description="Optional scene or mission description"),
    detector_model: str = Form("M9_BIT_CD", description="Detector model identifier (M9_BIT_CD or M10_QAT_CD)"),
    max_regions: int = Form(6, ge=1, le=20, description="Maximum number of significant change regions to analyze"),
) -> TemporalAnalysisResponse:
    """Executes full end-to-end temporal satellite image understanding.

    Integrates:
    - Pre-upload validation & optical RGB normalization
    - Phase 3A Two-Stage Co-Registration (ORB + ECC) with policy gating
    - Dynamic patch-based Change Detection (M9 BIT-CD / M10 QAT-CD)
    - Phase 3A Morphological Region Extraction & Ranking
    - Phase 3B Optical VARI greenness proxy analysis
    - Phase 3B Groq Qwen 3.8 Multimodal semantic change classification
    - Phase 3B Groq Qwen 27B dual-stage explanation synthesis
    """
    # 1. Read files into memory with request-scoped cleanup
    try:
        t1_bytes = await file_t1.read()
        t2_bytes = await file_t2.read()
    finally:
        await file_t1.close()
        await file_t2.close()

    # 2. Determine GSD value
    gsd_val = gsd_meters_per_pixel if gsd_meters_per_pixel is not None else gsd

    # 3. Execute analysis pipeline
    try:
        result = temporal_understanding_service.analyze_pair(
            t1_bytes=t1_bytes,
            t2_bytes=t2_bytes,
            filename_t1=file_t1.filename,
            filename_t2=file_t2.filename,
            observation_date_t1=observation_date_t1,
            observation_date_t2=observation_date_t2,
            gsd_meters=gsd_val,
            description=description,
            detector_model=detector_model,
            max_regions=max_regions,
        )
        return TemporalAnalysisResponse(**result)

    except ValueError as e:
        msg = str(e)
        if "empty" in msg.lower():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
        elif "exceeds maximum allowed size" in msg.lower():
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=msg)
        elif "unsupported image file extension" in msg.lower():
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    except Exception as e:
        logger.exception("Unexpected error in /api/temporal/analyze: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected failure during temporal satellite analysis: {str(e)}",
        )


