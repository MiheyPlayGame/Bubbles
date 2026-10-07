"""Shared image enhancement params for preview and export."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class ImageParams:
    contrast: float = 1.0  # alpha
    brightness: float = 0.0  # beta
    gamma: float = 1.0  # <1 lifts shadows
    clahe: bool = False
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    blur: int = 0  # 0/1 = off; odd kernel otherwise
    invert: bool = False
    equalize: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def enhance_image(frame: np.ndarray, params: ImageParams) -> np.ndarray:
    """Apply brightness/contrast/gamma/CLAHE/blur/invert to a gray or BGR frame."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    out = gray
    if params.contrast != 1.0 or params.brightness != 0.0:
        out = cv2.convertScaleAbs(out, alpha=float(params.contrast), beta=float(params.brightness))

    if abs(params.gamma - 1.0) > 1e-6:
        gamma = max(0.05, float(params.gamma))
        table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, table)

    if params.equalize:
        out = cv2.equalizeHist(out)

    if params.clahe:
        clahe = cv2.createCLAHE(
            clipLimit=float(params.clahe_clip),
            tileGridSize=(max(2, int(params.clahe_grid)), max(2, int(params.clahe_grid))),
        )
        out = clahe.apply(out)

    if params.blur and int(params.blur) > 1:
        out = cv2.GaussianBlur(out, (_odd(params.blur), _odd(params.blur)), 0)

    if params.invert:
        out = cv2.bitwise_not(out)

    return out
