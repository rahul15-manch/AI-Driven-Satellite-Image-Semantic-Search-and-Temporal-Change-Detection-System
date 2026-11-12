"""False-alarm and perturbation robustness analysis module."""

from src.false_alarm.base import BasePerturbation
from src.false_alarm.illumination import IlluminationShiftPerturbation
from src.false_alarm.blur import GaussianBlurPerturbation
from src.false_alarm.misregistration import GeometricMisregistrationPerturbation
from src.false_alarm.occlusion import LocalizedOcclusionShadowPerturbation
from src.false_alarm.robustness_metrics import RobustnessMetricsCalculator

__all__ = [
    "BasePerturbation",
    "IlluminationShiftPerturbation",
    "GaussianBlurPerturbation",
    "GeometricMisregistrationPerturbation",
    "LocalizedOcclusionShadowPerturbation",
    "RobustnessMetricsCalculator",
]
