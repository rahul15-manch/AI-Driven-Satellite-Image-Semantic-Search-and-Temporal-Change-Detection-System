"""Pydantic schemas and data contracts for the Milestone 12 Web Interface API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    """Health check and service status contract."""
    status: str = "healthy"
    version: str = "1.0.0"
    milestone: str = "M12"
    device: str = "cpu"
    services: Dict[str, bool] = Field(
        default_factory=lambda: {
            "retrieval": True,
            "change_detection": True,
            "benchmark": True,
            "image_serving": True,
        }
    )


class SearchRequest(BaseModel):
    """Semantic image retrieval query request."""
    query: str = Field(..., min_length=1, max_length=300, description="Natural language search query")
    method: str = Field(
        default="clip",
        description="Retrieval method: 'clip' (CLIP Zero-Shot), 'prompt_ensemble' (5 RS templates), 'bm25' (Lexical)",
    )
    top_k: int = Field(default=8, ge=1, le=48, description="Number of top candidates to return")

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query string cannot be empty or whitespace only.")
        return stripped

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"clip", "prompt_ensemble", "bm25"}
        clean = v.strip().lower()
        if clean not in allowed:
            raise ValueError(f"Unsupported retrieval method '{v}'. Must be one of: {allowed}")
        return clean


class SearchResultItem(BaseModel):
    """Individual ranked satellite image search candidate."""
    rank: int
    image_id: str
    filename: Optional[str] = None
    image_url: str
    score: float
    similarity_score: Optional[float] = None
    score_type: str
    category: str


class SearchResponse(BaseModel):
    """Full semantic search outcome."""
    query: str
    method: str
    method_display: str
    total_results: int
    latency_ms: float
    results: List[SearchResultItem]


class ChangeDetectionRequest(BaseModel):
    """Bi-temporal change detection execution request."""
    pair_id: Optional[str] = Field(default="test_1", description="Identifier of a verified LEVIR-CD test pair")
    model_id: Optional[str] = Field(
        default=None,
        description="Detector model identifier: B1_Pixel_Diff, B2_SSIM, B3_CVA, M7_FC_Siam_diff, M8_SNUNet, M9_BIT_CD, M10_QAT_CD",
    )
    model: Optional[str] = Field(default=None, description="Alias for model_id")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional override decision threshold")
    gamma: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional QAT-CD attenuation parameter (0.0 or 0.30)")
    image_t1_base64: Optional[str] = Field(default=None, description="Optional user-uploaded pre-change image in base64")
    image_t2_base64: Optional[str] = Field(default=None, description="Optional user-uploaded post-change image in base64")
    return_diagnostics: bool = Field(default=False, description="Whether to include QAT-CD confounder diagnostics")

    @model_validator(mode="before")
    @classmethod
    def unify_model(cls, values: Any) -> Any:
        if isinstance(values, dict):
            m = values.get("model_id") or values.get("model") or "M9_BIT_CD"
            values["model_id"] = m
            values["model"] = m
        return values


class ChangeDetectionMetrics(BaseModel):
    """Evaluation metrics against ground truth (when available)."""
    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int
    total_pixels: int


class ChangeDetectionResponse(BaseModel):
    """Change detection inference output."""
    pair_id: Optional[str] = None
    model_id: str
    model: Optional[str] = None
    model_display: str
    threshold_used: float
    threshold: Optional[float] = None
    threshold_source: str
    latency_ms: float
    image_t1_url: Optional[str] = None
    image_t2_url: Optional[str] = None
    t1_url: Optional[str] = None
    t2_url: Optional[str] = None
    heatmap_data_uri: str
    mask_data_uri: str
    error_overlay_data_uri: Optional[str] = None
    overlay_data_uri: Optional[str] = None
    metrics: Optional[ChangeDetectionMetrics] = None
    diagnostics: Optional[Dict[str, Any]] = None


class PairInfo(BaseModel):
    """Metadata record for an available test scene pair."""
    pair_id: str
    name: Optional[str] = None
    dataset: str
    image_t1_url: str
    image_t2_url: str
    has_ground_truth: bool

    @model_validator(mode="before")
    @classmethod
    def set_name_default(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("name"):
            values["name"] = values.get("pair_id")
        return values


class BenchmarkModelSummary(BaseModel):
    """Standardized M11 benchmark record for a detector."""
    model_id: str
    display_name: str
    architecture_type: str
    threshold_value: float
    threshold_source: str
    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int
    parameters: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    peak_rss_mb: float
    execution_status: str
    reproducibility_status: str


class BenchmarkResponse(BaseModel):
    """Comprehensive M11 benchmark evaluation payload."""
    models: List[BenchmarkModelSummary]
    statistical_bootstrap_ci: Dict[str, Any]
    bootstrap_ci: Optional[Any] = None
    paired_statistical_tests: List[Dict[str, Any]]
    wilcoxon_tests: Optional[Any] = None
    qat_ablation_analysis: Dict[str, Any]
    ablation: Optional[Any] = None
    cpu_performance_benchmark: Dict[str, Any]
    threshold_provenance_audit: Dict[str, Any]
    figures: List[Union[str, Dict[str, Any]]]


# ==============================================================================
# LLM EXPLAINABILITY SCHEMAS (Groq-hosted Qwen 27B)
# ==============================================================================

class LLMStatusResponse(BaseModel):
    """Status contract for Groq-hosted LLM explainability."""
    configured: bool
    provider: str = "Groq"
    model: str
    status: str  # "ready", "unconfigured", "error"
    connectivity: str  # "verified", "unverified", "failed"
    message: str


class ExplainChangeDetectionRequest(BaseModel):
    """Request payload to explain a completed bi-temporal change detection run."""
    pair_id: Optional[str] = Field(default="test_1", description="LEVIR-CD pair ID")
    model_id: str = Field(..., description="Detector model identifier")
    threshold: Optional[float] = Field(default=None, description="Decision threshold override if any")
    gamma: Optional[float] = Field(default=0.0, description="QAT-CD quality attenuation parameter")
    changed_pixels: Optional[int] = Field(default=None, description="Optional count of predicted changed pixels")
    change_ratio_pct: Optional[float] = Field(default=None, description="Optional percentage of changed area")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Optional verified ground truth metrics")


class ExplainSearchRequest(BaseModel):
    """Request payload to explain an RSICD semantic search candidate."""
    query: str = Field(..., min_length=1, description="Original natural language query")
    method: str = Field(..., description="Retrieval method (clip, prompt_ensemble, bm25)")
    image_id: str = Field(..., description="Image identifier")
    filename: Optional[str] = Field(default=None, description="Image filename")
    rank: int = Field(..., ge=1, description="Ranking position (1-based)")
    score: float = Field(..., description="Similarity or lexical relevance score")
    score_type: Optional[str] = Field(default=None, description="Score type label")
    category: Optional[str] = Field(default=None, description="Visual scene category")
    captions: Optional[List[str]] = Field(default=None, description="Verified RSICD reference captions")


class ExplanationResponse(BaseModel):
    """Evidence-grounded model explanation output contract."""
    explanation_model: str = Field(..., description="Configured Qwen 27B model identifier on Groq")
    provider: str = Field(default="Groq", description="Hosting API provider")
    source_model: str = Field(..., description="Original research model that generated the prediction")
    task: str = Field(..., description="Task type: change_detection or semantic_search")
    prediction_summary: str = Field(..., description="Clear evidence-grounded interpretation of the prediction")
    supporting_evidence: List[str] = Field(default_factory=list, description="Verified facts and metrics from the results")
    limitations: List[str] = Field(default_factory=list, description="Documented architectural and evaluation constraints")
    unknowns: List[str] = Field(default_factory=list, description="Information missing or outside the system's purview")

