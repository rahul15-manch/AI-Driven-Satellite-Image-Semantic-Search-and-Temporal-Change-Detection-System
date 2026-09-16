"""Unit tests for controlled perturbation transformations."""

import numpy as np
import pytest

from src.false_alarm.illumination import IlluminationShiftPerturbation
from src.false_alarm.blur import GaussianBlurPerturbation
from src.false_alarm.misregistration import GeometricMisregistrationPerturbation
from src.false_alarm.occlusion import LocalizedOcclusionShadowPerturbation


@pytest.fixture
def dummy_pair():
    rng = np.random.default_rng(42)
    t1 = rng.uniform(0.1, 0.9, size=(256, 256, 3)).astype(np.float32)
    t2 = rng.uniform(0.1, 0.9, size=(256, 256, 3)).astype(np.float32)
    return t1, t2


def test_illumination_shift(dummy_pair):
    t1, t2 = dummy_pair
    pert = IlluminationShiftPerturbation()

    p_t1, p_t2, meta = pert.apply(t1, t2, severity="medium", target_image="t2")

    # T1 must be unmodified
    np.testing.assert_array_equal(p_t1, t1)
    # T2 must have shape preserved and values in [0, 1]
    assert p_t2.shape == t2.shape
    assert p_t2.dtype == t2.dtype
    assert np.all(p_t2 >= 0.0) and np.all(p_t2 <= 1.0)
    # Brightness should be increased
    assert np.mean(p_t2) > np.mean(t2)

    # Monotonic severity test
    _, p_mild, _ = pert.apply(t1, t2, severity="mild", target_image="t2")
    _, p_med, _ = pert.apply(t1, t2, severity="medium", target_image="t2")
    _, p_str, _ = pert.apply(t1, t2, severity="strong", target_image="t2")

    assert np.mean(p_mild) < np.mean(p_med) < np.mean(p_str)


def test_gaussian_blur(dummy_pair):
    t1, t2 = dummy_pair
    pert = GaussianBlurPerturbation()

    p_t1, p_t2, meta = pert.apply(t1, t2, severity="medium", target_image="t2")

    assert p_t2.shape == t2.shape
    assert p_t2.dtype == t2.dtype
    assert np.all(p_t2 >= 0.0) and np.all(p_t2 <= 1.0)
    assert meta["sigma"] == 2.0

    # Monotonic severity test on high-frequency noise
    _, p_mild, m_mild = pert.apply(t1, t2, severity="mild", target_image="t2")
    _, p_med, m_med = pert.apply(t1, t2, severity="medium", target_image="t2")
    _, p_str, m_str = pert.apply(t1, t2, severity="strong", target_image="t2")

    assert m_mild["sigma"] < m_med["sigma"] < m_str["sigma"]


def test_geometric_misregistration(dummy_pair):
    t1, t2 = dummy_pair
    pert = GeometricMisregistrationPerturbation()

    p_t1, p_t2, meta = pert.apply(t1, t2, severity="medium", target_image="t2")

    assert p_t2.shape == t2.shape
    assert p_t2.dtype == t2.dtype
    assert meta["shift_dy"] == 3 and meta["shift_dx"] == 3


def test_localized_occlusion_shadow(dummy_pair):
    t1, t2 = dummy_pair
    pert = LocalizedOcclusionShadowPerturbation()

    p_t1, p_t2, meta = pert.apply(t1, t2, severity="medium", target_image="t2")

    assert p_t2.shape == t2.shape
    assert p_t2.dtype == t2.dtype
    assert np.mean(p_t2) < np.mean(t2)  # Shadow darkens pixels

    # Monotonic area test
    _, _, m_mild = pert.apply(t1, t2, severity="mild", target_image="t2")
    _, _, m_med = pert.apply(t1, t2, severity="medium", target_image="t2")
    _, _, m_str = pert.apply(t1, t2, severity="strong", target_image="t2")

    assert m_mild["affected_pixel_count"] < m_med["affected_pixel_count"] < m_str["affected_pixel_count"]


def test_perturbation_functional_purity(dummy_pair):
    t1, t2 = dummy_pair
    t1_orig = np.copy(t1)
    t2_orig = np.copy(t2)

    pert = IlluminationShiftPerturbation()
    pert.apply(t1, t2, severity="strong", target_image="t2")

    # Inputs should not have been mutated
    np.testing.assert_array_equal(t1, t1_orig)
    np.testing.assert_array_equal(t2, t2_orig)
