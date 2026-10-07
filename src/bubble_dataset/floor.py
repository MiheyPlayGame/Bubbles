"""Detect floor boundary and build a crop window between rising/floor lines."""

from __future__ import annotations

import numpy as np


def detect_floor_y(
    gray: np.ndarray,
    search_from_ratio: float = 0.55,
    smooth_ksize: int = 11,
) -> int:
    """
    Estimate the y of the floor start as the strongest downward darkening
    in the lower part of the frame (row-mean brightness profile).
    """
    if gray.ndim != 2:
        raise ValueError("detect_floor_y expects a grayscale image")

    h, _ = gray.shape
    row = gray.mean(axis=1).astype(np.float32)
    k = max(3, int(smooth_ksize) | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(row, kernel, mode="same")

    y0 = int(h * search_from_ratio)
    y0 = min(max(y0, 1), h - 3)
    grad = np.diff(smooth)
    region = grad[y0 : h - 2]
    if region.size == 0:
        return h - 1

    floor_y = y0 + int(np.argmin(region)) + 1
    return int(floor_y)


def default_rising_y(floor_y: int, crop_height: int = 80) -> int:
    return max(0, int(floor_y) - int(crop_height))


def crop_box_from_lines(
    frame_shape: tuple[int, ...],
    floor_y: int,
    rising_y: int,
    side_crop_ratio: float = 0.25,
) -> tuple[int, int, int, int]:
    """
    ROI between rising_y (top, inclusive) and floor_y (bottom, exclusive-ish clip).

    Side margins: side_crop_ratio of full width on each side (default 1/4).
    """
    h, w = frame_shape[:2]
    if not (0.0 <= side_crop_ratio < 0.5):
        raise ValueError("side_crop_ratio must be in [0, 0.5)")

    top = int(np.clip(rising_y, 0, h - 1))
    bottom = int(np.clip(floor_y, top + 1, h))
    height = bottom - top
    if height < 1:
        raise ValueError(f"Invalid lines: rising_y={rising_y}, floor_y={floor_y}")

    side = int(round(w * side_crop_ratio))
    x = side
    width = w - 2 * side
    if width < 1:
        raise ValueError(
            f"side_crop_ratio={side_crop_ratio} leaves no horizontal space for width={w}"
        )
    return x, top, width, height


# Backwards-compatible alias used by older call sites.
def crop_box_from_floor(
    frame_shape: tuple[int, ...],
    floor_y: int,
    crop_height: int = 80,
    side_crop_ratio: float = 0.25,
) -> tuple[int, int, int, int]:
    rising_y = default_rising_y(floor_y, crop_height)
    return crop_box_from_lines(frame_shape, floor_y, rising_y, side_crop_ratio)
