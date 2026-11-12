"""Method B2: Structural Similarity (SSIM) Dissimilarity Change Detector.

Computes structural dissimilarity between pre- and post-change satellite imagery using
localized structural similarity comparison.

Parameters (matching M3 specifications):
    Window size: 11 x 11
    Gaussian sigma: 1.5
    Channel handling: Standard ITU-R 601-2 luminance conversion (default) or multi-channel mean.
    Continuous dissimilarity map: D_SSIM = clip(1.0 - SSIM, 0.0, 1.0)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    from scipy.ndimage import gaussian_filter

    def ssim(
        im1: np.ndarray,
        im2: np.ndarray,
        win_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        gaussian_weights: bool = True,
        full: bool = True,
        channel_axis: Optional[int] = None,
    ):
        if channel_axis is not None:
            maps = []
            for ch in range(im1.shape[channel_axis]):
                if channel_axis == 2:
                    sub1, sub2 = im1[:, :, ch], im2[:, :, ch]
                else:
                    sub1, sub2 = im1[ch], im2[ch]
                _, m = ssim(sub1, sub2, win_size=win_size, sigma=sigma, data_range=data_range, full=True)
                maps.append(m)
            ssim_map = np.stack(maps, axis=channel_axis)
            return float(np.mean(ssim_map)), ssim_map

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        truncate = ((win_size - 1) / 2) / sigma

        ux = gaussian_filter(im1, sigma, mode="reflect", truncate=truncate)
        uy = gaussian_filter(im2, sigma, mode="reflect", truncate=truncate)
        uxx = gaussian_filter(im1 * im1, sigma, mode="reflect", truncate=truncate)
        uyy = gaussian_filter(im2 * im2, sigma, mode="reflect", truncate=truncate)
        uxy = gaussian_filter(im1 * im2, sigma, mode="reflect", truncate=truncate)

        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        ssim_map = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux**2 + uy**2 + C1) * (vx + vy + C2))
        return float(np.mean(ssim_map)), ssim_map

from src.change_detection.base import BaseChangeDetector


class SSIMDetector(BaseChangeDetector):
    """Method B2: SSIM Dissimilarity Detector."""

    def __init__(
        self,
        win_size: int = 11,
        sigma: float = 1.5,
        channel_mode: str = "grayscale",
    ):
        """Initializes SSIMDetector.

        Args:
            win_size: Gaussian window size (must be odd positive integer, default 11).
            sigma: Gaussian filter standard deviation (default 1.5).
            channel_mode: 'grayscale' (ITU-R 601-2 luma, primary) or 'multichannel' (per-channel average).
        """
        if win_size % 2 == 0 or win_size < 3:
            raise ValueError(f"win_size must be an odd integer >= 3, got {win_size}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if channel_mode not in {"grayscale", "multichannel"}:
            raise ValueError(f"channel_mode must be 'grayscale' or 'multichannel', got {channel_mode}")

        super().__init__(
            name="B2_SSIM_Dissimilarity",
            params={
                "win_size": win_size,
                "sigma": sigma,
                "channel_mode": channel_mode,
            },
        )
        self.win_size = win_size
        self.sigma = sigma
        self.channel_mode = channel_mode

    def compute_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Computes localized SSIM dissimilarity map in [0.0, 1.0].

        Args:
            img1: Pre-change optical image (H, W, 3) or (H, W).
            img2: Post-change optical image (H, W, 3) or (H, W).

        Returns:
            2D numpy array (H, W) float32 with continuous values in [0.0, 1.0].
        """
        arr1 = self._ensure_float_rgb(img1)
        arr2 = self._ensure_float_rgb(img2)

        if arr1.shape != arr2.shape:
            raise ValueError(f"Shape mismatch between img1 {arr1.shape} and img2 {arr2.shape}")

        h, w = arr1.shape[:2]
        if min(h, w) < self.win_size:
            raise ValueError(
                f"Image dimensions ({h}, {w}) must be at least win_size ({self.win_size})"
            )

        if arr1.ndim == 3 and arr1.shape[2] == 3:
            if self.channel_mode == "grayscale":
                # ITU-R 601-2 luma transform
                g1 = 0.299 * arr1[:, :, 0] + 0.587 * arr1[:, :, 1] + 0.114 * arr1[:, :, 2]
                g2 = 0.299 * arr2[:, :, 0] + 0.587 * arr2[:, :, 1] + 0.114 * arr2[:, :, 2]
                _, ssim_map = ssim(
                    g1,
                    g2,
                    win_size=self.win_size,
                    gaussian_weights=True,
                    sigma=self.sigma,
                    data_range=1.0,
                    full=True,
                )
            else:
                _, ssim_map = ssim(
                    arr1,
                    arr2,
                    win_size=self.win_size,
                    gaussian_weights=True,
                    sigma=self.sigma,
                    data_range=1.0,
                    channel_axis=2,
                    full=True,
                )
                if ssim_map.ndim == 3:
                    ssim_map = np.mean(ssim_map, axis=2)
        elif arr1.ndim == 3 and arr1.shape[2] == 1:
            g1 = arr1.squeeze(2)
            g2 = arr2.squeeze(2)
            _, ssim_map = ssim(
                g1,
                g2,
                win_size=self.win_size,
                gaussian_weights=True,
                sigma=self.sigma,
                data_range=1.0,
                full=True,
            )
        else:
            _, ssim_map = ssim(
                arr1,
                arr2,
                win_size=self.win_size,
                gaussian_weights=True,
                sigma=self.sigma,
                data_range=1.0,
                full=True,
            )

        dissimilarity = np.clip(1.0 - ssim_map, 0.0, 1.0)
        return dissimilarity.astype(np.float32)
