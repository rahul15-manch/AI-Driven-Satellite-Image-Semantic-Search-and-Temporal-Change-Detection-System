"""Quality-Aware Temporal Change Detection (QAT-CD).

Implements Milestone M10 quality-aware decision mechanisms for bi-temporal
satellite change detection under CPU constraints.

Theoretical formulation:
Given pre-change image T1 and post-change image T2:
1. Base detector outputs continuous prior change probability P_0(x, y) in [0, 1].
2. Modular quality defect indicators are computed:
   - S_reg(x, y): Normalized gradient discrepancy measuring misregistration / geometric shift.
   - S_illum(x, y): Normalized local luminance disparity measuring illumination/shadow shifts.
   - S_sharp(x, y): High-frequency Laplacian energy disparity measuring optical blur differences.
3. Composite defect risk index:
   Q_defect(x, y) = w_reg * S_reg + w_illum * S_illum + w_sharp * S_sharp, bounded in [0, 1].
4. Base detector margin uncertainty:
   U_det(x, y) = 4 * P_0(x, y) * (1 - P_0(x, y)), bounded in [0, 1].
5. Quality-aware attenuation gating rule:
   P_QAT(x, y) = clip(P_0(x, y) * (1 - gamma * U_det(x, y) * Q_defect(x, y)), 0.0, 1.0).

Properties:
- Numerically stable and deterministic.
- Temporal symmetry: If base detector is symmetric under (T1, T2) <-> (T2, T1),
  then P_QAT(T1, T2) == P_QAT(T2, T1).
- Non-destructive to confident genuine changes (as U_det -> 0 when P_0 -> 1).
- Zero-cost identity ablation when gamma = 0.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndi

from src.change_detection.base import BaseChangeDetector


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Converts (H, W, 3) or (H, W) array to 2D float32 grayscale in [0.0, 1.0]."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            # ITU-R BT.601 standard weights
            return 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        return arr.mean(axis=2)
    return arr


def extract_registration_discrepancy(
    img1: np.ndarray,
    img2: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Measures high-frequency gradient disparity indicative of sub-pixel misregistration.

    Args:
        img1: Pre-change image (H, W, 3) or (H, W).
        img2: Post-change image (H, W, 3) or (H, W).
        eps: Small epsilon for numerical stability.

    Returns:
        2D float32 array in [0.0, 1.0].
    """
    y1 = _to_gray_float(img1)
    y2 = _to_gray_float(img2)

    g1_x = ndi.sobel(y1, axis=1)
    g1_y = ndi.sobel(y1, axis=0)
    g1 = np.hypot(g1_x, g1_y)

    g2_x = ndi.sobel(y2, axis=1)
    g2_y = ndi.sobel(y2, axis=0)
    g2 = np.hypot(g2_x, g2_y)

    denom = g1 + g2 + eps
    s_reg = np.abs(g1 - g2) / denom
    return np.clip(s_reg.astype(np.float32), 0.0, 1.0)


def extract_illumination_discrepancy(
    img1: np.ndarray,
    img2: np.ndarray,
    kernel_size: int = 15,
    eps: float = 1e-4,
) -> np.ndarray:
    """Measures local luminance ratio disparity indicative of illumination and shadow shifts.

    Args:
        img1: Pre-change image (H, W, 3) or (H, W).
        img2: Post-change image (H, W, 3) or (H, W).
        kernel_size: Size of local uniform averaging window.
        eps: Small epsilon for numerical stability.

    Returns:
        2D float32 array in [0.0, 1.0].
    """
    y1 = _to_gray_float(img1)
    y2 = _to_gray_float(img2)

    u1 = ndi.uniform_filter(y1, size=kernel_size)
    u2 = ndi.uniform_filter(y2, size=kernel_size)

    denom = u1 + u2 + eps
    s_illum = np.abs(u1 - u2) / denom
    return np.clip(s_illum.astype(np.float32), 0.0, 1.0)


def extract_sharpness_discrepancy(
    img1: np.ndarray,
    img2: np.ndarray,
    kernel_size: int = 15,
    eps: float = 1e-4,
) -> np.ndarray:
    """Measures disparity in high-frequency Laplacian energy (optical blur difference).

    Args:
        img1: Pre-change image (H, W, 3) or (H, W).
        img2: Post-change image (H, W, 3) or (H, W).
        kernel_size: Window size for local energy computation.
        eps: Small epsilon for numerical stability.

    Returns:
        2D float32 array in [0.0, 1.0].
    """
    y1 = _to_gray_float(img1)
    y2 = _to_gray_float(img2)

    l1 = ndi.laplace(y1)
    l2 = ndi.laplace(y2)

    v1 = ndi.uniform_filter(l1**2, size=kernel_size)
    v2 = ndi.uniform_filter(l2**2, size=kernel_size)

    denom = v1 + v2 + eps
    s_sharp = np.abs(v1 - v2) / denom
    return np.clip(s_sharp.astype(np.float32), 0.0, 1.0)


def compute_detector_uncertainty(prob_map: np.ndarray) -> np.ndarray:
    """Computes base detector margin uncertainty around decision boundary.

    Formulation:
        U(x, y) = 4 * P(x, y) * (1 - P(x, y))
        Range: [0.0, 1.0], maximal (1.0) at P=0.5, minimal (0.0) at P=0.0 and P=1.0.

    Args:
        prob_map: 2D probability map in [0.0, 1.0].

    Returns:
        2D float32 array in [0.0, 1.0].
    """
    p = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)
    u = 4.0 * p * (1.0 - p)
    return np.clip(u, 0.0, 1.0)


def compute_composite_defect_index(
    s_reg: Optional[np.ndarray],
    s_illum: Optional[np.ndarray],
    s_sharp: Optional[np.ndarray],
    weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
) -> np.ndarray:
    """Combines active quality defect indicators into a composite risk index in [0, 1].

    Args:
        s_reg: Registration discrepancy map or None.
        s_illum: Illumination discrepancy map or None.
        s_sharp: Sharpness discrepancy map or None.
        weights: Relative weights (w_reg, w_illum, w_sharp).

    Returns:
        2D float32 array in [0.0, 1.0].
    """
    signals = []
    active_weights = []

    if s_reg is not None:
        signals.append(s_reg)
        active_weights.append(float(weights[0]))
    if s_illum is not None:
        signals.append(s_illum)
        active_weights.append(float(weights[1]))
    if s_sharp is not None:
        signals.append(s_sharp)
        active_weights.append(float(weights[2]))

    if not signals:
        # If no quality signals active, defect index is zero everywhere
        ref_shape = (256, 256)
        return np.zeros(ref_shape, dtype=np.float32)

    total_w = sum(active_weights)
    if total_w <= 0:
        total_w = 1.0
    norm_weights = [w / total_w for w in active_weights]

    composite = np.zeros_like(signals[0], dtype=np.float32)
    for sig, w in zip(signals, norm_weights):
        composite += w * sig

    return np.clip(composite, 0.0, 1.0)


def quality_aware_gate(
    base_prob: np.ndarray,
    defect_index: np.ndarray,
    gamma: float = 0.30,
    use_uncertainty: bool = True,
) -> np.ndarray:
    """Applies quality-aware attenuation to base detector probabilities.

    Formulation:
        If use_uncertainty is True:
            P_QAT = P_0 * (1 - gamma * U_det * Q_defect)
        Else:
            P_QAT = P_0 * (1 - gamma * Q_defect)

    Args:
        base_prob: 2D base detector probability map in [0.0, 1.0].
        defect_index: 2D composite defect index in [0.0, 1.0].
        gamma: Attenuation strength in [0.0, 1.0].
        use_uncertainty: Whether to scale attenuation by margin uncertainty.

    Returns:
        2D float32 gated probability map in [0.0, 1.0].
    """
    p0 = np.clip(np.asarray(base_prob, dtype=np.float32), 0.0, 1.0)
    q = np.clip(np.asarray(defect_index, dtype=np.float32), 0.0, 1.0)
    g = float(np.clip(gamma, 0.0, 1.0))

    if use_uncertainty:
        u = compute_detector_uncertainty(p0)
        attenuation = g * u * q
    else:
        attenuation = g * q

    factor = 1.0 - attenuation
    p_gat = p0 * factor
    return np.clip(p_gat, 0.0, 1.0)


class QualityAwareChangeDetector(BaseChangeDetector):
    """Quality-Aware Temporal Change Detector (M10 QAT-CD).

    Wraps a base change detector (e.g. M9 BIT-CD or M7 FC-Siam-diff) and applies
    reproducible, calibration-governed quality gating to suppress false alarms
    originating from misregistration, illumination shifts, and optical blur.
    """

    def __init__(
        self,
        base_detector: BaseChangeDetector,
        gamma: float = 0.30,
        weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
        use_reg: bool = True,
        use_illum: bool = True,
        use_sharp: bool = True,
        use_uncertainty: bool = True,
        name: str = "M10_QAT_CD",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, params=params or {})
        self.base_detector = base_detector
        self.gamma = float(gamma)
        self.weights = weights
        self.use_reg = use_reg
        self.use_illum = use_illum
        self.use_sharp = use_sharp
        self.use_uncertainty = use_uncertainty

    def compute_quality_diagnostics(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        base_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Computes all individual quality signals and intermediate diagnostics.

        Args:
            img1: Pre-change image (H, W, 3) or (H, W).
            img2: Post-change image (H, W, 3) or (H, W).
            base_prob: Precomputed base probability map (optional).

        Returns:
            Dictionary containing:
                - base_prob: Base detector probability in [0, 1].
                - s_reg: Registration discrepancy map in [0, 1].
                - s_illum: Illumination discrepancy map in [0, 1].
                - s_sharp: Sharpness discrepancy map in [0, 1].
                - uncertainty: Detector margin uncertainty in [0, 1].
                - defect_index: Composite defect index in [0, 1].
                - gated_prob: Final quality-aware probability in [0, 1].
        """
        if base_prob is None:
            base_prob = self.base_detector.compute_difference_map(img1, img2)
        base_prob = np.clip(base_prob.astype(np.float32), 0.0, 1.0)

        s_reg = extract_registration_discrepancy(img1, img2) if self.use_reg else None
        s_illum = extract_illumination_discrepancy(img1, img2) if self.use_illum else None
        s_sharp = extract_sharpness_discrepancy(img1, img2) if self.use_sharp else None

        defect_index = compute_composite_defect_index(
            s_reg, s_illum, s_sharp, weights=self.weights
        )

        uncertainty = compute_detector_uncertainty(base_prob)

        gated_prob = quality_aware_gate(
            base_prob=base_prob,
            defect_index=defect_index,
            gamma=self.gamma,
            use_uncertainty=self.use_uncertainty,
        )

        return {
            "base_prob": base_prob,
            "s_reg": s_reg if s_reg is not None else np.zeros_like(base_prob),
            "s_illum": s_illum if s_illum is not None else np.zeros_like(base_prob),
            "s_sharp": s_sharp if s_sharp is not None else np.zeros_like(base_prob),
            "uncertainty": uncertainty,
            "defect_index": defect_index,
            "gated_prob": gated_prob,
        }

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Computes quality-aware change probability map in [0.0, 1.0]."""
        diag = self.compute_quality_diagnostics(img1, img2)
        return diag["gated_prob"]

    def predict(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates binary change mask and quality-aware difference map."""
        diff_map = self.compute_difference_map(img1, img2)
        binary_mask = (diff_map >= threshold).astype(np.uint8)
        return binary_mask, diff_map


# Convenience alias
QATChangeDetector = QualityAwareChangeDetector


def save_qat_metadata(
    path: str | Path,
    metadata: Dict[str, Any],
) -> None:
    """Saves QAT-CD calibration metadata and parameters to JSON."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_qat_metadata(path: str | Path) -> Dict[str, Any]:
    """Loads QAT-CD metadata from JSON."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {load_path}")
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)
