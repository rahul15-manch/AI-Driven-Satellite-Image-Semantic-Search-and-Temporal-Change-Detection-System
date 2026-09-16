"""Unit tests for fixed-seed reproducibility of perturbations."""

import numpy as np
import pytest

from src.false_alarm.illumination import IlluminationShiftPerturbation
from src.false_alarm.blur import GaussianBlurPerturbation
from src.false_alarm.misregistration import GeometricMisregistrationPerturbation
from src.false_alarm.occlusion import LocalizedOcclusionShadowPerturbation


def test_reproducibility_deterministic_outputs():
    rng = np.random.default_rng(100)
    t1 = rng.uniform(0.0, 1.0, size=(128, 128, 3)).astype(np.float32)
    t2 = rng.uniform(0.0, 1.0, size=(128, 128, 3)).astype(np.float32)

    perturbations = [
        IlluminationShiftPerturbation(),
        GaussianBlurPerturbation(),
        GeometricMisregistrationPerturbation(),
        LocalizedOcclusionShadowPerturbation(),
    ]

    for pert in perturbations:
        # Run 1
        _, out1, _ = pert.apply(t1, t2, severity="medium", seed=42)
        # Run 2
        _, out2, _ = pert.apply(t1, t2, severity="medium", seed=42)

        np.testing.assert_array_equal(out1, out2, err_msg=f"{pert.name} not reproducible!")
