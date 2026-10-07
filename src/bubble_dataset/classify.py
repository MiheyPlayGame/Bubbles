"""Heuristic rising vs sitting classification."""

from __future__ import annotations

from bubble_dataset import CLASS_RISING, CLASS_SITTING
from bubble_dataset.detect import Detection


def classify_detection(
    det: Detection,
    image_height: int,
    bottom_ratio: float = 0.18,
) -> int:
    """
    Class 0 rising: center above the bottom band.
    Class 1 sitting: center inside the bottom band (near the floor).
    """
    band = max(1, int(image_height * bottom_ratio))
    floor_y = image_height - band
    if det.cy >= floor_y:
        return CLASS_SITTING
    return CLASS_RISING
