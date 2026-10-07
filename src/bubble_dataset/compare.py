"""Compare detection parameter variants on one cropped frame; build a visual palette."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from bubble_dataset.cine_io import estimate_background, iter_frames
from bubble_dataset.classify import classify_detection
from bubble_dataset.detect import DetectConfig, find_bubbles, preprocess
from bubble_dataset.floor import crop_box_from_floor, detect_floor_y
from bubble_dataset.pipeline import apply_roi, draw_debug


@dataclass(frozen=True)
class Variant:
    name: str
    detect: DetectConfig
    bottom_ratio: float = 0.25


def default_variants() -> list[Variant]:
    """
    Parameter sweep for comparison.

    Note: with background absdiff, bubble residuals are usually bright → invert=False.
    Without background, dark bubbles on bright field often need invert=True.
    """
    bg = DetectConfig(use_background=True, invert=False, blur_ksize=5, morph_kernel=3, min_area=40, min_circularity=0.35)
    nobg = DetectConfig(use_background=False, invert=False, blur_ksize=5, morph_kernel=3, min_area=40, min_circularity=0.35)
    nobg_inv = DetectConfig(use_background=False, invert=True, blur_ksize=5, morph_kernel=3, min_area=40, min_circularity=0.35)

    return [
        Variant("01_bg", bg),
        Variant("02_nobg", nobg),
        Variant("03_nobg_invert", nobg_inv),
        Variant("04_bg_blur1", replace(bg, blur_ksize=1)),
        Variant("05_bg_blur9", replace(bg, blur_ksize=9)),
        Variant("06_bg_morph1", replace(bg, morph_kernel=1)),
        Variant("07_bg_morph7", replace(bg, morph_kernel=7)),
        Variant("08_bg_area20", replace(bg, min_area=20)),
        Variant("09_bg_area100", replace(bg, min_area=100)),
        Variant("10_bg_circ0.20", replace(bg, min_circularity=0.20)),
        Variant("11_bg_circ0.55", replace(bg, min_circularity=0.55)),
        Variant("12_bg_strict", replace(bg, min_area=80, min_circularity=0.50, morph_kernel=5, blur_ksize=5)),
        Variant("13_bg_loose", replace(bg, min_area=15, min_circularity=0.15, morph_kernel=1, blur_ksize=3)),
        Variant("14_nobg_inv_blur1", replace(nobg_inv, blur_ksize=1)),
        Variant("15_nobg_inv_area20", replace(nobg_inv, min_area=20, min_circularity=0.20)),
        Variant("16_nobg_inv_strict", replace(nobg_inv, min_area=80, min_circularity=0.50, morph_kernel=5)),
    ]


def _pick_frame(source: Path, stride: int, frame_index: int | None) -> tuple[int, np.ndarray]:
    frames = list(iter_frames(source, stride=stride))
    if not frames:
        raise RuntimeError(f"No frames in {source}")
    if frame_index is None:
        return frames[len(frames) // 2]
    for idx, frame in frames:
        if idx == frame_index:
            return idx, frame
    raise RuntimeError(f"Frame index {frame_index} not found among sampled frames")


def _annotate_title(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Put a title strip ABOVE the image so it does not cover the crop."""
    bar_h = 34
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bar = np.zeros((bar_h, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def run_variant(
    gray_crop: np.ndarray,
    background_crop: np.ndarray | None,
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    bg = background_crop if variant.detect.use_background else None
    gray, binary = preprocess(gray_crop, bg, variant.detect)
    detections = find_bubbles(binary, variant.detect)
    h, _ = gray.shape[:2]
    labeled = []
    rising = sitting = 0
    for det in detections:
        class_id = classify_detection(det, h, variant.bottom_ratio)
        labeled.append((det, class_id))
        if class_id == 0:
            rising += 1
        else:
            sitting += 1
    overlay = draw_debug(gray, labeled)
    band = max(1, int(h * variant.bottom_ratio))
    cv2.line(overlay, (0, h - band), (overlay.shape[1], h - band), (255, 80, 80), 1)
    cv2.line(overlay, (0, h - 1), (overlay.shape[1], h - 1), (80, 80, 255), 2)
    return overlay, binary, len(detections), rising, sitting


def build_palette(
    panels: list[np.ndarray],
    cols: int = 1,
    pad: int = 12,
    bg_color: tuple[int, int, int] = (24, 24, 24),
    v_scale: float = 2.0,
) -> np.ndarray:
    """
    Compose a comparison grid.

    For wide short crops, keep full horizontal resolution (no width downscale)
    and optionally stretch height slightly for readability.
    """
    if not panels:
        raise ValueError("No panels to compose")
    cols = max(1, cols)

    scaled: list[np.ndarray] = []
    for panel in panels:
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        if v_scale != 1.0:
            new_h = max(1, int(round(panel.shape[0] * v_scale)))
            panel = cv2.resize(
                panel,
                (panel.shape[1], new_h),
                interpolation=cv2.INTER_NEAREST,
            )
        scaled.append(panel)

    ph = max(p.shape[0] for p in scaled)
    pw = max(p.shape[1] for p in scaled)
    normalized: list[np.ndarray] = []
    for panel in scaled:
        cell = np.full((ph, pw, 3), bg_color, dtype=np.uint8)
        cell[: panel.shape[0], : panel.shape[1]] = panel
        normalized.append(cell)

    rows = (len(normalized) + cols - 1) // cols
    canvas = np.full(
        (rows * ph + (rows + 1) * pad, cols * pw + (cols + 1) * pad, 3),
        bg_color,
        dtype=np.uint8,
    )
    for i, panel in enumerate(normalized):
        r, c = divmod(i, cols)
        y = pad + r * (ph + pad)
        x = pad + c * (pw + pad)
        canvas[y : y + ph, x : x + pw] = panel
    return canvas


def compare_source(
    source: Path,
    out_dir: Path,
    *,
    stride: int = 500,
    crop_height: int = 80,
    side_crop_ratio: float = 0.25,
    floor_y: int | None = None,
    frame_index: int | None = None,
    cols: int = 1,
) -> Path:
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    masks_dir = out_dir / "masks"
    cells_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    frame_idx, frame = _pick_frame(source, stride, frame_index)
    gray_full = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fy = floor_y if floor_y is not None else detect_floor_y(gray_full)
    roi = crop_box_from_floor(
        gray_full.shape,
        fy,
        crop_height,
        side_crop_ratio=side_crop_ratio,
    )
    gray_crop = apply_roi(gray_full, roi)

    background_crop = None
    try:
        bg_full = estimate_background(source, max_frames=40, sample_stride=max(1, stride // 5 or 1))
        background_crop = apply_roi(bg_full, roi)
    except Exception:  # noqa: BLE001
        background_crop = None

    cv2.imwrite(str(out_dir / "source_crop.png"), gray_crop)

    variants = default_variants()
    panels: list[np.ndarray] = []
    summary_lines = [
        f"source={source}",
        f"frame_idx={frame_idx}",
        f"floor_y={fy}",
        f"roi={roi}",
        f"crop_height={crop_height}",
        f"side_crop_ratio={side_crop_ratio}",
        "",
        "variant\tdets\trising\tsitting",
    ]

    for variant in variants:
        overlay, binary, n_det, rising, sitting = run_variant(gray_crop, background_crop, variant)
        titled = _annotate_title(
            overlay,
            variant.name,
            f"dets={n_det}  rising={rising}  sitting={sitting}",
        )
        panels.append(titled)
        cv2.imwrite(str(cells_dir / f"{variant.name}.jpg"), titled)
        cv2.imwrite(str(masks_dir / f"{variant.name}_mask.png"), binary)
        summary_lines.append(f"{variant.name}\t{n_det}\t{rising}\t{sitting}")
        print(f"{variant.name}: dets={n_det} rising={rising} sitting={sitting}")

    palette = build_palette(panels, cols=cols, v_scale=2.0)
    palette_path = out_dir / "palette.jpg"
    cv2.imwrite(str(palette_path), palette, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # Also save PNG strips stacked at native width (best for pixel-accurate review).
    palette_native = build_palette(panels, cols=1, v_scale=1.0, pad=8)
    native_path = out_dir / "palette_native.png"
    cv2.imwrite(str(native_path), palette_native)
    (out_dir / "summary.tsv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"palette: {palette_path}")
    print(f"native : {native_path}")
    return palette_path
