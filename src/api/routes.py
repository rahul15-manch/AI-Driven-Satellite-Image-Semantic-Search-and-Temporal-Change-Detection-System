"""FastAPI routing endpoints for semantic retrieval, change detection, and benchmarking."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from src.api.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    PairInfo,
    BenchmarkResponse,
)
from src.api.service import (
    retrieval_service,
    change_detection_service,
    benchmark_service,
    ROOT,
)

router = APIRouter(prefix="/api", tags=["API"])


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
        base_dir = (ROOT / "data" / "raw" / "levir_cd").resolve()

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
    """Serves generated M11 or M10 research figures."""
    clean_name = Path(figure_name).name
    if not clean_name.endswith((".png", ".jpg")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid figure filename format.",
        )

    fig_m11 = (ROOT / "experiments" / "figures" / "m11" / clean_name).resolve()
    fig_m10 = (ROOT / "experiments" / "figures" / "m10" / clean_name).resolve()

    if fig_m11.exists() and fig_m11.is_file():
        return FileResponse(fig_m11, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    elif fig_m10.exists() and fig_m10.is_file():
        return FileResponse(fig_m10, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Research figure '{clean_name}' not found.",
    )
