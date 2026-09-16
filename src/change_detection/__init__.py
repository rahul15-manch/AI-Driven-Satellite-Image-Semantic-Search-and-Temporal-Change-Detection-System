"""Classical bi-temporal change detection module for remote sensing imagery."""

from src.change_detection.base import BaseChangeDetector
from src.change_detection.pixel_diff import PixelDiffDetector
from src.change_detection.ssim_detector import SSIMDetector
from src.change_detection.cva_detector import CVADetector
from src.change_detection.thresholding import (
    ValidationThresholdOptimizer,
    ThresholdSearchResult,
)
from src.change_detection.evaluator import ChangeDetectionEvaluator, ConfusionMatrix
from src.change_detection.profiler import ChangeDetectionProfiler

__all__ = [
    "BaseChangeDetector",
    "PixelDiffDetector",
    "SSIMDetector",
    "CVADetector",
    "ValidationThresholdOptimizer",
    "ThresholdSearchResult",
    "ChangeDetectionEvaluator",
    "ConfusionMatrix",
    "ChangeDetectionProfiler",
]
