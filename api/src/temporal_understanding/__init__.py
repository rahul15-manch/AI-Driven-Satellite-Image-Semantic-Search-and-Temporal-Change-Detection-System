"""Temporal Satellite Image Understanding Module.

Core algorithmic engines for bi-temporal image pair validation,
co-registration, region extraction, and semantic change understanding.
"""

from src.temporal_understanding.registration import (
    ImageCoRegistrator,
    RegistrationPolicy,
    RegistrationDiagnostics,
    RegistrationResult,
    RegistrationStatus,
    TemporalUnderstandingError,
    ImageValidationError,
    RegistrationError,
    RegistrationRejectedError,
)
from src.temporal_understanding.region_extractor import (
    RegionExtractor,
    RegionExtractionConfig,
    ChangedRegion,
    RegionExtractionResult,
    RegionExtractionError,
)
from src.temporal_understanding.semantic_analyzer import (
    SemanticCategory,
    RegionalVARIEvidence,
    RegionalSemanticInterpretation,
    VisionAnalyzerConfig,
    SemanticChangeAnalyzer,
    compute_vari_map,
    analyze_regional_vari,
    prepare_region_composite_image,
)
from src.temporal_understanding.explanation_engine import (
    TemporalExplanationConfig,
    TemporalUnderstandingReport,
    TemporalExplanationEngine,
)

__all__ = [
    # Registration & Validation
    "ImageCoRegistrator",
    "RegistrationPolicy",
    "RegistrationDiagnostics",
    "RegistrationResult",
    "RegistrationStatus",
    "TemporalUnderstandingError",
    "ImageValidationError",
    "RegistrationError",
    "RegistrationRejectedError",
    # Region Extraction
    "RegionExtractor",
    "RegionExtractionConfig",
    "ChangedRegion",
    "RegionExtractionResult",
    "RegionExtractionError",
    # Semantic Analysis & VARI
    "SemanticCategory",
    "RegionalVARIEvidence",
    "RegionalSemanticInterpretation",
    "VisionAnalyzerConfig",
    "SemanticChangeAnalyzer",
    "compute_vari_map",
    "analyze_regional_vari",
    "prepare_region_composite_image",
    # Dual-Stage Explanation
    "TemporalExplanationConfig",
    "TemporalUnderstandingReport",
    "TemporalExplanationEngine",
]
