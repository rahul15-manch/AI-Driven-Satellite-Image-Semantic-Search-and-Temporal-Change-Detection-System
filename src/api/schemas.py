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
    models_loaded: List[str] = Field(
        default_factory=lambda: [
            "B1_Pixel_Diff",
            "B2_SSIM",
            "B3_CVA",
            "M7_FC_Siam_diff",
            "M8_SNUNet",
            "M9_BIT_CD",
            "M10_QAT_CD",
            "RSICD_CLIP",
        ]
    )
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


# ==============================================================================
# PHASE 3C: TEMPORAL SATELLITE IMAGE UNDERSTANDING SCHEMAS
# ==============================================================================

class TemporalImageMetadata(BaseModel):
    """Metadata regarding ingested bitemporal satellite scenes."""
    width: int = Field(..., description="Canvas pixel width")
    height: int = Field(..., description="Canvas pixel height")
    channels: int = Field(default=3, description="Channel depth (optical RGB)")
    observation_date_t1: Optional[str] = Field(default=None, description="Timestamp/date of T1 observation")
    observation_date_t2: Optional[str] = Field(default=None, description="Timestamp/date of T2 observation")
    gsd_meters_per_pixel: Optional[float] = Field(default=None, description="Ground Sample Distance in meters per pixel")
    description: Optional[str] = Field(default=None, description="Contextual scene notes or source label")


class TemporalRegistrationDiagnostics(BaseModel):
    """Diagnostics and quality metrics evaluated during image co-registration."""
    registration_status: str = Field(..., description="Registration outcome: aligned_homography, aligned_affine, identity_aligned, rejected")
    match_count: int = Field(..., description="Total ORB keypoint matches identified")
    inlier_count: int = Field(..., description="RANSAC geometric inliers verified")
    inlier_ratio: float = Field(..., description="Inlier ratio (inlier_count / match_count)")
    reprojection_error: float = Field(..., description="Geometric root-mean-square reprojection error")
    ecc_correlation: Optional[float] = Field(default=None, description="Fine ECC correlation coefficient")
    spatial_coverage: float = Field(..., description="Convex hull spatial coverage ratio")
    confidence_score: float = Field(..., description="Composite registration quality score [0.0, 1.0]")
    transformation_type: str = Field(..., description="Transformation model applied (homography, affine, identity, none)")
    warnings: List[str] = Field(default_factory=list, description="Residual parallax, scale, or coverage warnings")


class TemporalDetectorMetadata(BaseModel):
    """Telemetry from the core change detection engine."""
    model_id: str = Field(..., description="Detector model identifier (e.g. M9_BIT_CD)")
    model_display: str = Field(..., description="Human-readable model name")
    threshold_used: float = Field(..., description="Decision threshold tau* applied")
    total_changed_pixels: int = Field(..., description="Count of positive change pixels")
    total_changed_area_pct: float = Field(..., description="Percentage of canvas identified as changed")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    qat_diagnostics: Optional[Dict[str, Any]] = Field(default=None, description="Quality attenuation metrics if M10_QAT_CD")


class TemporalMaskRepresentation(BaseModel):
    """Compact base64 PNG representation of the binary change mask."""
    format: str = Field(default="png_data_uri", description="Encoding format")
    mask_data_uri: str = Field(..., description="Base64 encoded PNG mask (white=change, black=no change)")
    width: int = Field(..., description="Mask width in pixels")
    height: int = Field(..., description="Mask height in pixels")
    total_changed_pixels: int = Field(..., description="Total number of positive pixels")
    coordinate_frame: str = Field(
        default="top_left_origin_pixels",
        description="Reference frame: (0,0) top-left, X rightwards, Y downwards"
    )


class TemporalRegionItem(BaseModel):
    """Discrete ranked spatial change cluster extracted from the cleaned mask."""
    region_id: int = Field(..., description="1-indexed ranked region identifier")
    bbox: List[int] = Field(..., description="[xmin, ymin, xmax, ymax] pixel bounds in aligned canvas")
    context_bbox: List[int] = Field(..., description="[c_xmin, c_ymin, c_xmax, c_ymax] with contextual margin")
    pixel_area: int = Field(..., description="Cleaned mask pixel area")
    raw_pixel_area: Optional[int] = Field(default=None, description="Unfiltered raw mask pixel count")
    area_ratio: float = Field(..., description="Ratio of canvas area covered by this region")
    centroid: List[float] = Field(..., description="[x_center, y_center] centroid coordinates")
    bbox_width: int = Field(..., description="Bounding box pixel width")
    bbox_height: int = Field(..., description="Bounding box pixel height")
    mean_probability: Optional[float] = Field(default=None, description="Mean detector confidence activation")
    physical_area_sqm: Optional[float] = Field(default=None, description="Surface area in m² (strictly when GSD is provided)")


class TemporalVARIEvidenceItem(BaseModel):
    """Quantitative optical greenness proxy evidence derived from visible RGB bands."""
    status: str = Field(..., description="supported, unsupported_channels, zero_variance, invalid_values")
    t1_mean_vari: Optional[float] = Field(default=None, description="Mean VARI across region at T1")
    t2_mean_vari: Optional[float] = Field(default=None, description="Mean VARI across region at T2")
    delta_vari: Optional[float] = Field(default=None, description="Delta VARI (T2 - T1)")
    interpretation: str = Field(..., description="Vegetation greenness proxy interpretation")
    limitations: List[str] = Field(default_factory=list, description="Caveats and non-NDVI scientific bounds")


class TemporalSemanticFindingItem(BaseModel):
    """Evidence-grounded semantic interpretation for an individual change region."""
    region_id: int = Field(..., description="Region identifier")
    change_category: str = Field(..., description="Standardized 8-class remote sensing category")
    visual_observation: str = Field(..., description="Factual visual description of changed optical structures")
    evidence_for_category: List[str] = Field(default_factory=list, description="Specific optical cues supporting hypothesis")
    alternative_interpretations: List[str] = Field(default_factory=list, description="Other plausible explanations")
    uncertainty_reason: Optional[str] = Field(default=None, description="Reason for ambiguity or model uncertainty")
    requires_human_review: bool = Field(..., description="Flag indicating analyst verification is recommended")
    vari_evidence: Optional[TemporalVARIEvidenceItem] = Field(default=None, description="Regional optical greenness proxy data")


class TemporalExplanationReport(BaseModel):
    """Comprehensive academic explanation synthesized by Stage B reasoning engine."""
    overall_summary: str = Field(..., description="Factual narrative summarizing temporal changes")
    detector_evidence: List[str] = Field(default_factory=list, description="Unaltered numerical detector evidence")
    region_observations: List[Dict[str, Any]] = Field(default_factory=list, description="Region geometry and characteristics")
    semantic_interpretations: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence-linked category hypotheses")
    registration_and_quality_warnings: List[str] = Field(default_factory=list, description="Alignment and parallax warnings")
    scientific_limitations: List[str] = Field(default_factory=list, description="Computational and optical methodology bounds")
    unknown_variables: List[str] = Field(default_factory=list, description="Items unverifiable from monocular optical imagery")
    human_review_recommendations: List[str] = Field(default_factory=list, description="Actionable analyst recommendations")


class TemporalAnalysisResponse(BaseModel):
    """Comprehensive outcome contract for the end-to-end temporal understanding API."""
    request_id: str = Field(..., description="Unique analysis request identifier")
    status: str = Field(..., description="Pipeline execution status: success, registration_rejected, error")
    image_metadata: TemporalImageMetadata = Field(..., description="Dimensions and observation metadata")
    registration_diagnostics: Optional[TemporalRegistrationDiagnostics] = Field(
        default=None, description="Co-registration diagnostics and quality metrics"
    )
    detector_metadata: Optional[TemporalDetectorMetadata] = Field(
        default=None, description="Core change detection model telemetry"
    )
    change_mask: Optional[TemporalMaskRepresentation] = Field(
        default=None, description="Compact base64 PNG mask representation"
    )
    regions: List[TemporalRegionItem] = Field(
        default_factory=list, description="Ranked spatial change regions"
    )
    vari_observations: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Summary of optical greenness proxy observations"
    )
    semantic_findings: List[TemporalSemanticFindingItem] = Field(
        default_factory=list, description="Multimodal vision model interpretations per region"
    )
    temporal_explanation: Optional[TemporalExplanationReport] = Field(
        default=None, description="Synthesized academic report separating evidence from hypothesis"
    )
    warnings: List[str] = Field(default_factory=list, description="Pipeline alerts and caveats")
    errors: List[str] = Field(default_factory=list, description="Error messages if execution halted")

