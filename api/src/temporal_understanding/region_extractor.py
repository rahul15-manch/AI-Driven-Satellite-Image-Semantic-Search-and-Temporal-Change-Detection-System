"""Region extraction and spatial clustering engine for bi-temporal change masks.

Converts dense binary change masks into structured discrete change region entities
via morphological cleanup, connected-component analysis, area filtering,
and contextual bounding-box geometry extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# EXCEPTIONS
# =============================================================================

class RegionExtractionError(Exception):
    """Raised when mask validation or region extraction encounters invalid inputs."""
    pass


# =============================================================================
# DATA CONTRACTS
# =============================================================================

@dataclass
class RegionExtractionConfig:
    """Configurable parameters for morphological filtering and region extraction."""
    min_pixel_area: int = 150
    min_area_ratio: float = 0.0002  # 0.02% of total canvas area
    max_regions: int = 6
    morphology_closing_kernel: int = 3  # Odd integer >= 1
    morphology_opening_kernel: int = 3  # Odd integer >= 1
    context_padding_ratio: float = 0.10  # 10% bounding box contextual margin
    connectivity: int = 8  # 4 or 8 connectivity


@dataclass
class ChangedRegion:
    """Structured entity representing an individual spatial change region."""
    region_id: int
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    context_bbox: Tuple[int, int, int, int]  # (c_xmin, c_ymin, c_xmax, c_ymax)
    pixel_area: int  # Area in morphologically cleaned mask
    raw_pixel_area: Optional[int]  # Corresponding area in raw uncleaned mask
    area_ratio: float  # pixel_area / total_image_pixels
    centroid: Tuple[float, float]  # (x_center, y_center)
    bbox_width: int
    bbox_height: int
    mean_probability: Optional[float] = None
    gsd_meters: Optional[float] = None
    physical_area_sq_meters: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts region attributes to dictionary."""
        return {
            "region_id": self.region_id,
            "bbox": list(self.bbox),
            "context_bbox": list(self.context_bbox),
            "pixel_area": self.pixel_area,
            "raw_pixel_area": self.raw_pixel_area,
            "area_ratio": round(float(self.area_ratio), 6),
            "centroid": [round(float(self.centroid[0]), 2), round(float(self.centroid[1]), 2)],
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "mean_probability": round(float(self.mean_probability), 4) if self.mean_probability is not None else None,
            "physical_area_sq_meters": (
                round(float(self.physical_area_sq_meters), 2)
                if self.physical_area_sq_meters is not None
                else None
            ),
        }


@dataclass
class RegionExtractionResult:
    """Full outcome of connected-component region extraction."""
    regions: List[ChangedRegion]
    raw_mask: np.ndarray
    cleaned_mask: np.ndarray
    total_candidates_found: int
    total_regions_retained: int
    filtered_small_regions_count: int
    total_changed_pixels_raw: int
    total_changed_pixels_clean: int
    mask_width: int
    mask_height: int


# =============================================================================
# REGION EXTRACTOR IMPLEMENTATION
# =============================================================================

class RegionExtractor:
    """Extracts discrete, ranked change regions from binary change masks."""

    def __init__(self, config: Optional[RegionExtractionConfig] = None):
        self.config = config or RegionExtractionConfig()

        # Validate odd kernel sizes
        if self.config.morphology_closing_kernel % 2 == 0 or self.config.morphology_closing_kernel < 1:
            raise RegionExtractionError(
                f"morphology_closing_kernel must be an odd positive integer, got {self.config.morphology_closing_kernel}."
            )
        if self.config.morphology_opening_kernel % 2 == 0 or self.config.morphology_opening_kernel < 1:
            raise RegionExtractionError(
                f"morphology_opening_kernel must be an odd positive integer, got {self.config.morphology_opening_kernel}."
            )
        if self.config.connectivity not in (4, 8):
            raise RegionExtractionError(
                f"connectivity must be 4 or 8, got {self.config.connectivity}."
            )

    def _validate_inputs(
        self,
        binary_mask: np.ndarray,
        probability_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validates mask dimensions, dtypes, and value ranges."""
        if not isinstance(binary_mask, np.ndarray):
            raise RegionExtractionError(f"binary_mask must be a numpy.ndarray, got {type(binary_mask).__name__}.")

        if binary_mask.ndim != 2:
            raise RegionExtractionError(f"binary_mask must be 2D, got shape {binary_mask.shape}.")

        h, w = binary_mask.shape
        if h == 0 or w == 0:
            raise RegionExtractionError(f"binary_mask dimensions cannot be zero, got {w}x{h}.")

        # Convert to uint8 binary mask {0, 1}
        if binary_mask.dtype == bool:
            mask_uint8 = binary_mask.astype(np.uint8)
        elif np.issubdtype(binary_mask.dtype, np.integer):
            if binary_mask.max() > 1:
                # If {0, 255} or scaled labels, binarize at median/128
                mask_uint8 = (binary_mask > 0).astype(np.uint8)
            else:
                mask_uint8 = np.clip(binary_mask, 0, 1).astype(np.uint8)
        elif np.issubdtype(binary_mask.dtype, np.floating):
            mask_uint8 = (binary_mask >= 0.5).astype(np.uint8)
        else:
            raise RegionExtractionError(f"Unsupported binary_mask dtype '{binary_mask.dtype}'.")

        # Validate optional probability map
        prob_arr: Optional[np.ndarray] = None
        if probability_map is not None:
            if not isinstance(probability_map, np.ndarray):
                raise RegionExtractionError(f"probability_map must be a numpy.ndarray, got {type(probability_map).__name__}.")
            if probability_map.shape != binary_mask.shape:
                raise RegionExtractionError(
                    f"probability_map shape {probability_map.shape} does not match mask shape {binary_mask.shape}."
                )
            prob_arr = np.clip(probability_map.astype(np.float32), 0.0, 1.0)

        return mask_uint8, prob_arr

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Applies closing to bridge fragments, then opening to eliminate single-pixel noise."""
        k_close = self.config.morphology_closing_kernel
        k_open = self.config.morphology_opening_kernel

        cleaned = mask.copy()

        if k_close > 1:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        if k_open > 1:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

        return cleaned

    def extract_regions(
        self,
        binary_mask: np.ndarray,
        probability_map: Optional[np.ndarray] = None,
        gsd_meters: Optional[float] = None,
    ) -> RegionExtractionResult:
        """Extracts discrete change regions from a binary change mask.

        Args:
            binary_mask: 2D array representing detected changes.
            probability_map: Optional 2D float array in [0.0, 1.0] representing detector confidence.
            gsd_meters: Optional Ground Sample Distance in meters per pixel.
                        If provided, physical area in m² is computed; otherwise omitted.

        Returns:
            RegionExtractionResult containing ranked regions, raw mask, and cleaned mask.
        """
        raw_mask, prob_map = self._validate_inputs(binary_mask, probability_map)
        h, w = raw_mask.shape
        total_canvas_pixels = float(h * w)

        total_changed_raw = int(np.sum(raw_mask))

        # Check for empty mask
        if total_changed_raw == 0:
            return RegionExtractionResult(
                regions=[],
                raw_mask=raw_mask,
                cleaned_mask=np.zeros_like(raw_mask),
                total_candidates_found=0,
                total_regions_retained=0,
                filtered_small_regions_count=0,
                total_changed_pixels_raw=0,
                total_changed_pixels_clean=0,
                mask_width=w,
                mask_height=h,
            )

        # Apply morphological filtering
        cleaned_mask = self._apply_morphology(raw_mask)
        total_changed_clean = int(np.sum(cleaned_mask))

        if total_changed_clean == 0:
            return RegionExtractionResult(
                regions=[],
                raw_mask=raw_mask,
                cleaned_mask=cleaned_mask,
                total_candidates_found=0,
                total_regions_retained=0,
                filtered_small_regions_count=0,
                total_changed_pixels_raw=total_changed_raw,
                total_changed_pixels_clean=0,
                mask_width=w,
                mask_height=h,
            )

        # Connected-Component Labeling (CCL)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=self.config.connectivity
        )

        candidates: List[ChangedRegion] = []
        filtered_count = 0

        # Label 0 is background; evaluate labels 1 to num_labels - 1
        for label_idx in range(1, num_labels):
            left = int(stats[label_idx, cv2.CC_STAT_LEFT])
            top = int(stats[label_idx, cv2.CC_STAT_TOP])
            box_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            box_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            pixel_area = int(stats[label_idx, cv2.CC_STAT_AREA])
            cx, cy = centroids[label_idx]

            area_ratio = pixel_area / total_canvas_pixels

            # Filtering gates: both minimum pixel count and area proportion
            if pixel_area < self.config.min_pixel_area or area_ratio < self.config.min_area_ratio:
                filtered_count += 1
                continue

            # Strict bounding box: [xmin, ymin, xmax, ymax]
            xmin = left
            ymin = top
            xmax = left + box_w
            ymax = top + box_h

            # Contextual padding: add configurable margin (default 10%)
            pad_w = int(round(box_w * self.config.context_padding_ratio))
            pad_h = int(round(box_h * self.config.context_padding_ratio))
            c_xmin = max(0, xmin - pad_w)
            c_ymin = max(0, ymin - pad_h)
            c_xmax = min(w, xmax + pad_w)
            c_ymax = min(h, ymax + pad_h)

            # Measure raw pixel area before morphology within component region
            component_pixels_mask = labels == label_idx
            raw_area = int(np.sum(raw_mask[component_pixels_mask]))

            # Compute mean probability if continuous confidence map is available
            mean_prob: Optional[float] = None
            if prob_map is not None:
                mean_prob = float(np.mean(prob_map[component_pixels_mask]))

            # Optional physical area (only if verified GSD is supplied)
            physical_area: Optional[float] = None
            if gsd_meters is not None and gsd_meters > 0:
                physical_area = float(pixel_area * (gsd_meters ** 2))

            region = ChangedRegion(
                region_id=label_idx,  # Temporary id, reassigned after ranking
                bbox=(xmin, ymin, xmax, ymax),
                context_bbox=(c_xmin, c_ymin, c_xmax, c_ymax),
                pixel_area=pixel_area,
                raw_pixel_area=raw_area,
                area_ratio=area_ratio,
                centroid=(float(cx), float(cy)),
                bbox_width=box_w,
                bbox_height=box_h,
                mean_probability=mean_prob,
                gsd_meters=gsd_meters,
                physical_area_sq_meters=physical_area,
            )
            candidates.append(region)

        # Rank regions by pixel area descending
        candidates.sort(key=lambda r: r.pixel_area, reverse=True)

        # Select top-K regions
        retained = candidates[: self.config.max_regions]

        # Re-assign 1-indexed region IDs matching their area rank
        for rank_idx, reg in enumerate(retained, start=1):
            reg.region_id = rank_idx

        return RegionExtractionResult(
            regions=retained,
            raw_mask=raw_mask,
            cleaned_mask=cleaned_mask,
            total_candidates_found=len(candidates),
            total_regions_retained=len(retained),
            filtered_small_regions_count=filtered_count,
            total_changed_pixels_raw=total_changed_raw,
            total_changed_pixels_clean=total_changed_clean,
            mask_width=w,
            mask_height=h,
        )
