"""Bi-temporal satellite image validation and two-stage co-registration engine.

Provides robust validation, coarse feature-based homography alignment (Stage A),
fine ECC refinement (Stage B), and strict multi-metric quality gating
for remote-sensing optical image pairs.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# EXCEPTIONS
# =============================================================================

class TemporalUnderstandingError(Exception):
    """Base exception for all temporal understanding operations."""
    pass


class ImageValidationError(TemporalUnderstandingError):
    """Raised when an input image fails format, readability, or integrity validation."""
    pass


class RegistrationError(TemporalUnderstandingError):
    """Raised when registration algorithms encounter fatal runtime errors."""
    pass


class RegistrationRejectedError(RegistrationError):
    """Raised when alignment quality fails policy gates and is rejected."""

    def __init__(self, message: str, diagnostics: RegistrationDiagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


# =============================================================================
# ENUMS AND DATA CONTRACTS
# =============================================================================

class RegistrationStatus(str, Enum):
    """Status classification for bi-temporal image registration."""
    ALIGNED_HOMOGRAPHY = "aligned_homography"
    ALIGNED_AFFINE = "aligned_affine"
    IDENTITY_ALIGNED = "identity_aligned"
    REJECTED = "rejected"


@dataclass
class RegistrationPolicy:
    """Configurable quality policy governing bi-temporal registration."""
    # Image constraints
    max_processing_dim: int = 2048
    min_dimension: int = 16
    allow_multichannel_tiff: bool = True

    # Stage A: Coarse ORB alignment parameters
    orb_max_features: int = 2000
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    ratio_test_threshold: float = 0.75
    min_matches: int = 15
    min_inliers: int = 10
    min_inlier_ratio: float = 0.40
    max_reprojection_error: float = 3.5
    min_spatial_coverage: float = 0.10
    high_quality_threshold: float = 0.70

    # Stage B: Fine ECC alignment parameters
    ecc_enabled: bool = True
    ecc_max_iterations: int = 50
    ecc_eps: float = 1e-4
    ecc_min_correlation: float = 0.50

    # Fallback and safety controls
    allow_affine_fallback: bool = True
    raise_on_rejection: bool = False
    warp_border_mode: int = cv2.BORDER_CONSTANT
    warp_border_value: int = 0


@dataclass
class RegistrationDiagnostics:
    """Quantitative diagnostics evaluated during image co-registration."""
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    ecc_correlation: Optional[float] = None
    spatial_coverage: float = 0.0
    confidence_score: float = 0.0
    registration_status: RegistrationStatus = RegistrationStatus.REJECTED
    transformation_type: str = "none"
    scale_factor_x: float = 1.0
    scale_factor_y: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts diagnostics to serializable dictionary."""
        return {
            "match_count": self.match_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "reprojection_error": round(float(self.reprojection_error), 4),
            "ecc_correlation": round(float(self.ecc_correlation), 4) if self.ecc_correlation is not None else None,
            "spatial_coverage": round(float(self.spatial_coverage), 4),
            "confidence_score": round(float(self.confidence_score), 4),
            "registration_status": self.registration_status.value,
            "transformation_type": self.transformation_type,
            "scale_factor_x": round(float(self.scale_factor_x), 4),
            "scale_factor_y": round(float(self.scale_factor_y), 4),
            "warnings": self.warnings,
        }


@dataclass
class RegistrationResult:
    """Full outcome of a registration attempt."""
    aligned_t2: Optional[np.ndarray]
    t1: np.ndarray
    t2: np.ndarray
    transformation_matrix: Optional[np.ndarray]
    diagnostics: RegistrationDiagnostics
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CO-REGISTRATION ENGINE
# =============================================================================

class ImageCoRegistrator:
    """Two-stage image co-registration pipeline designed for CPU-constrained execution."""

    def __init__(self, policy: Optional[RegistrationPolicy] = None):
        self.policy = policy or RegistrationPolicy()

    # -------------------------------------------------------------------------
    # Ingestion & Validation
    # -------------------------------------------------------------------------

    def load_and_validate_image(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
        name: str = "image",
    ) -> np.ndarray:
        """Validates and loads an optical satellite image into a standard numpy array.

        Returns an unmodified copy in RGB (H, W, 3) or Grayscale (H, W), dtype uint8.
        """
        if image_input is None:
            raise ImageValidationError(f"{name} cannot be None.")

        arr: Optional[np.ndarray] = None

        # 1. Path or string
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists() or not p.is_file():
                raise ImageValidationError(f"File not found or unreadable: {p}")
            suffix = p.suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                raise ImageValidationError(f"Unsupported image extension '{suffix}' for {name}. Must be PNG, JPEG, or TIFF.")
            try:
                with Image.open(p) as pil_img:
                    arr = np.array(pil_img)
            except Exception as e:
                raise ImageValidationError(f"Failed to read image file '{p}': {str(e)}")

        # 2. Raw bytes or BytesIO
        elif isinstance(image_input, (bytes, io.BytesIO)):
            raw_bytes = image_input if isinstance(image_input, bytes) else image_input.getvalue()
            if len(raw_bytes) == 0:
                raise ImageValidationError(f"Empty byte stream supplied for {name}.")
            try:
                with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                    arr = np.array(pil_img)
            except Exception as e:
                raise ImageValidationError(f"Failed to decode byte payload for {name}: {str(e)}")

        # 3. PIL Image
        elif isinstance(image_input, Image.Image):
            arr = np.array(image_input)

        # 4. Numpy array
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()

        else:
            raise ImageValidationError(f"Unsupported input type '{type(image_input).__name__}' for {name}.")

        # Dimensional and integrity validation
        if arr is None or arr.size == 0:
            raise ImageValidationError(f"{name} contains no pixel data.")

        if arr.ndim not in (2, 3):
            raise ImageValidationError(f"{name} has invalid dimension {arr.ndim}. Expected 2D or 3D array.")

        h, w = arr.shape[:2]
        if h < self.policy.min_dimension or w < self.policy.min_dimension:
            raise ImageValidationError(
                f"{name} dimensions ({w}x{h}) are smaller than minimum allowed {self.policy.min_dimension}x{self.policy.min_dimension}."
            )

        # Handle channel depths
        if arr.ndim == 3:
            num_channels = arr.shape[2]
            if num_channels == 1:
                arr = arr.squeeze(2)
            elif num_channels == 2:
                raise ImageValidationError(f"{name} has 2 channels, which is unsupported for optical satellite imagery.")
            elif num_channels > 3:
                if not self.policy.allow_multichannel_tiff:
                    raise ImageValidationError(f"{name} has {num_channels} channels. Multichannel TIFFs are disallowed by policy.")
                # Extract first 3 optical bands (typically RGB)
                arr = arr[:, :, :3]

        # Normalization to uint8
        if np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ImageValidationError(f"{name} contains NaN or Infinite values.")
            max_val = float(arr.max())
            if max_val <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype == np.uint16:
            # Scale 16-bit satellite channels (0..65535) down to 8-bit (0..255)
            arr = (arr / 256).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        # Non-blank validation
        if float(np.std(arr)) < 1e-4:
            raise ImageValidationError(f"{name} is completely blank (uniform zero variance).")

        return arr

    # -------------------------------------------------------------------------
    # Spatial Scaling for Processing Copies
    # -------------------------------------------------------------------------

    def create_processing_copy(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Creates a grayscale processing copy scaled to at most max_processing_dim.

        Returns:
            Tuple of:
                - gray_proc: 2D uint8 grayscale array
                - scale_x: width scaling factor (proc_w / orig_w)
                - scale_y: height scaling factor (proc_h / orig_h)
        """
        orig_h, orig_w = img.shape[:2]

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        max_dim = max(orig_h, orig_w)
        if max_dim <= self.policy.max_processing_dim:
            return gray, 1.0, 1.0

        scale = self.policy.max_processing_dim / float(max_dim)
        proc_w = max(int(round(orig_w * scale)), self.policy.min_dimension)
        proc_h = max(int(round(orig_h * scale)), self.policy.min_dimension)

        gray_proc = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        actual_scale_x = proc_w / float(orig_w)
        actual_scale_y = proc_h / float(orig_h)

        return gray_proc, actual_scale_x, actual_scale_y

    # -------------------------------------------------------------------------
    # Stage A: Coarse ORB Alignment
    # -------------------------------------------------------------------------

    def _extract_and_match_orb(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
        """Detects ORB keypoints and computes ratio-filtered matches."""
        orb = cv2.ORB_create(
            nfeatures=self.policy.orb_max_features,
            scaleFactor=self.policy.orb_scale_factor,
            nlevels=self.policy.orb_n_levels,
        )

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return kp1, kp2, []

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(des2, des1, k=2)

        good_matches: List[cv2.DMatch] = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.policy.ratio_test_threshold * n.distance:
                    good_matches.append(m)

        return kp1, kp2, good_matches

    def _compute_spatial_coverage(self, points: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Computes spatial distribution of inliers across the image canvas [0.0, 1.0]."""
        if len(points) < 3:
            return 0.0

        h, w = image_shape[:2]
        total_area = float(h * w)
        if total_area <= 0:
            return 0.0

        try:
            hull = cv2.convexHull(points.astype(np.float32))
            hull_area = float(cv2.contourArea(hull))
            # Ratio of bounding convex hull to image canvas, capped at 1.0
            return float(np.clip(hull_area / total_area, 0.0, 1.0))
        except Exception:
            return 0.0

    def _estimate_coarse_homography(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        shape1: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], int, float, float, float]:
        """Estimates RANSAC homography from feature matches.

        Returns:
            Tuple of (H_matrix, inlier_count, inlier_ratio, reprojection_error, spatial_coverage)
        """
        if len(matches) < 4:
            return None, 0, 0.0, 0.0, 0.0

        pts2 = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        H, inlier_mask = cv2.findHomography(
            pts2,
            pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.policy.max_reprojection_error,
        )

        if H is None or inlier_mask is None:
            return None, 0, 0.0, 0.0, 0.0

        inliers = inlier_mask.ravel() == 1
        inlier_count = int(np.sum(inliers))
        inlier_ratio = float(inlier_count / len(matches))

        if inlier_count < 4:
            return None, inlier_count, inlier_ratio, 0.0, 0.0

        # Calculate reprojection error across inliers
        inlier_pts2 = pts2[inliers]
        inlier_pts1 = pts1[inliers]

        projected = cv2.perspectiveTransform(inlier_pts2.reshape(-1, 1, 2), H).reshape(-1, 2)
        reproj_errors = np.linalg.norm(inlier_pts1 - projected, axis=1)
        mean_reproj_error = float(np.mean(reproj_errors))

        spatial_coverage = self._compute_spatial_coverage(inlier_pts1, shape1)

        # Determinant check to reject degenerate/reflective/singular matrices
        det = np.linalg.det(H[:2, :2])
        if det <= 0.15 or det >= 6.0:
            return None, inlier_count, inlier_ratio, mean_reproj_error, spatial_coverage

        return H, inlier_count, inlier_ratio, mean_reproj_error, spatial_coverage

    # -------------------------------------------------------------------------
    # Stage B: Fine ECC Alignment
    # -------------------------------------------------------------------------

    def _refine_with_ecc(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
        initial_warp: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[float], bool]:
        """Refines residual alignment using Enhanced Correlation Coefficient (ECC) maximization.

        Returns:
            Tuple of (refined_transform, correlation_score, success_bool)
        """
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.policy.ecc_max_iterations,
            self.policy.ecc_eps,
        )

        # Prepare affine warp matrix for ECC
        if initial_warp.shape == (3, 3):
            warp_matrix = initial_warp[:2, :].astype(np.float32)
        else:
            warp_matrix = initial_warp.astype(np.float32)

        try:
            corr, refined_warp = cv2.findTransformECC(
                gray1,
                gray2,
                warp_matrix,
                motionType=cv2.MOTION_AFFINE,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
            # Reconstruct 3x3 homography from affine
            H_refined = np.eye(3, dtype=np.float64)
            H_refined[:2, :] = refined_warp
            return H_refined, float(corr), True
        except cv2.error:
            # ECC failed to converge or encountered gradient singularity
            return initial_warp, None, False

    # -------------------------------------------------------------------------
    # Full Two-Stage Registration Pipeline
    # -------------------------------------------------------------------------

    def register_pair(
        self,
        image_t1: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
        image_t2: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    ) -> RegistrationResult:
        """Executes full validation and two-stage co-registration of image T2 onto T1.

        Returns:
            RegistrationResult containing aligned_t2 warped into T1's coordinate frame,
            unmodified originals, transformation matrix, and comprehensive diagnostics.
        """
        # Step 1: Ingestion & Validation
        t1 = self.load_and_validate_image(image_t1, name="Image T1")
        t2 = self.load_and_validate_image(image_t2, name="Image T2")

        diag = RegistrationDiagnostics()

        # Step 2: Check for identical images
        if t1.shape == t2.shape and np.array_equal(t1, t2):
            diag.match_count = 1000
            diag.inlier_count = 1000
            diag.inlier_ratio = 1.0
            diag.reprojection_error = 0.0
            diag.spatial_coverage = 1.0
            diag.confidence_score = 1.0
            diag.registration_status = RegistrationStatus.IDENTITY_ALIGNED
            diag.transformation_type = "identity"
            return RegistrationResult(
                aligned_t2=t2.copy(),
                t1=t1,
                t2=t2,
                transformation_matrix=np.eye(3, dtype=np.float64),
                diagnostics=diag,
                metadata={"status": "identical_pair"},
            )

        # Step 3: Create processing copies (grayscale, scaled to max 2048)
        gray1_proc, s1_x, s1_y = self.create_processing_copy(t1)
        gray2_proc, s2_x, s2_y = self.create_processing_copy(t2)

        diag.scale_factor_x = s1_x
        diag.scale_factor_y = s1_y

        # Step 4: Stage A - Coarse Alignment
        kp1, kp2, matches = self._extract_and_match_orb(gray1_proc, gray2_proc)
        diag.match_count = len(matches)

        if len(matches) < self.policy.min_matches:
            diag.warnings.append(
                f"Insufficient feature matches ({len(matches)} < {self.policy.min_matches}). Cannot establish geometric correspondence."
            )
            diag.registration_status = RegistrationStatus.REJECTED
            if self.policy.raise_on_rejection:
                raise RegistrationRejectedError("Insufficient feature matches between T1 and T2.", diag)
            return RegistrationResult(
                aligned_t2=None,
                t1=t1,
                t2=t2,
                transformation_matrix=None,
                diagnostics=diag,
            )

        H_proc, inliers, inlier_ratio, reproj_err, coverage = self._estimate_coarse_homography(
            kp1, kp2, matches, gray1_proc.shape
        )
        diag.inlier_count = inliers
        diag.inlier_ratio = inlier_ratio
        diag.reprojection_error = reproj_err
        diag.spatial_coverage = coverage

        if H_proc is None or inliers < self.policy.min_inliers or inlier_ratio < self.policy.min_inlier_ratio:
            # Check affine fallback if homography failed
            if self.policy.allow_affine_fallback and len(matches) >= 6:
                pts2 = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                pts1 = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                affine_mat, aff_inliers = cv2.estimateAffinePartial2D(
                    pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=self.policy.max_reprojection_error
                )
                if affine_mat is not None and np.sum(aff_inliers) >= 6:
                    H_proc = np.eye(3, dtype=np.float64)
                    H_proc[:2, :] = affine_mat
                    diag.transformation_type = "affine"
                    diag.warnings.append("Projective homography was degenerate; constrained to rigid affine fallback.")
                else:
                    H_proc = None

        if H_proc is None:
            diag.warnings.append("RANSAC geometric model estimation failed to converge on stable transformation.")
            diag.registration_status = RegistrationStatus.REJECTED
            if self.policy.raise_on_rejection:
                raise RegistrationRejectedError("RANSAC geometric estimation failed.", diag)
            return RegistrationResult(
                aligned_t2=None,
                t1=t1,
                t2=t2,
                transformation_matrix=None,
                diagnostics=diag,
            )

        if diag.transformation_type != "affine":
            diag.transformation_type = "homography"

        # Step 5: Stage B - Fine ECC Alignment (optional refinement)
        if self.policy.ecc_enabled and diag.transformation_type == "affine":
            H_proc, corr, ecc_ok = self._refine_with_ecc(gray1_proc, gray2_proc, H_proc)
            diag.ecc_correlation = corr
            if ecc_ok and corr is not None and corr >= self.policy.ecc_min_correlation:
                diag.transformation_type = "affine_ecc_refined"
            else:
                diag.warnings.append("Fine ECC refinement did not reach convergence threshold; using RANSAC estimate.")

        # Step 6: Map Processing Transformation Matrix Back to Native Resolution Coordinates
        # Formulation: H_native = S1^-1 * H_proc * S2
        S1 = np.diag([s1_x, s1_y, 1.0])
        S2 = np.diag([s2_x, s2_y, 1.0])
        H_native = np.linalg.inv(S1) @ H_proc @ S2

        # Step 7: Calculate Multi-Metric Quality Confidence Score
        confidence = (
            0.35 * float(np.clip(diag.inlier_ratio, 0.0, 1.0))
            + 0.35 * float(np.clip(diag.spatial_coverage, 0.0, 1.0))
            + 0.30 * float(np.clip(1.0 - (diag.reprojection_error / 5.0), 0.0, 1.0))
        )
        # Scale by count confidence (reaches 1.0 at 40 inliers)
        count_factor = float(np.clip(diag.inlier_count / 40.0, 0.1, 1.0))
        diag.confidence_score = float(np.clip(confidence * count_factor, 0.0, 1.0))

        # Quality Gating Logic
        if diag.confidence_score >= self.policy.high_quality_threshold and diag.inlier_count >= self.policy.min_inliers:
            diag.registration_status = (
                RegistrationStatus.ALIGNED_HOMOGRAPHY
                if "homography" in diag.transformation_type
                else RegistrationStatus.ALIGNED_AFFINE
            )
        elif self.policy.allow_affine_fallback and diag.confidence_score >= 0.35:
            diag.registration_status = RegistrationStatus.ALIGNED_AFFINE
            diag.warnings.append("Registration confidence is moderate. Proceed with quality-aware caution.")
        else:
            diag.warnings.append(
                f"Registration confidence ({diag.confidence_score:.3f}) failed quality threshold ({self.policy.high_quality_threshold}). Alignment rejected."
            )
            diag.registration_status = RegistrationStatus.REJECTED
            if self.policy.raise_on_rejection:
                raise RegistrationRejectedError(f"Alignment confidence {diag.confidence_score:.3f} failed quality gate.", diag)
            return RegistrationResult(
                aligned_t2=None,
                t1=t1,
                t2=t2,
                transformation_matrix=H_native,
                diagnostics=diag,
            )

        # Step 8: Warp Original T2 Image into Native T1 Coordinate Frame
        target_h, target_w = t1.shape[:2]
        aligned_t2 = cv2.warpPerspective(
            t2,
            H_native,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.policy.warp_border_mode,
            borderValue=self.policy.warp_border_value,
        )

        return RegistrationResult(
            aligned_t2=aligned_t2,
            t1=t1,
            t2=t2,
            transformation_matrix=H_native,
            diagnostics=diag,
            metadata={"status": "success"},
        )
