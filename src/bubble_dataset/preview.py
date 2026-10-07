"""Preview: guides, enhance, and search-preset palette."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bubble_dataset.cine_io import estimate_background, iter_frames
from bubble_dataset.classify import classify_detection
from bubble_dataset.detect import detect_bubbles
from bubble_dataset.floor import crop_box_from_lines, default_rising_y, detect_floor_y
from bubble_dataset.image_ops import ImageParams, enhance_image
from bubble_dataset.pipeline import apply_roi, draw_debug
from bubble_dataset.presets import all_presets, preset_to_dict


def pick_frame_pair(
    source: Path,
    frame_index: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray | None]:
    """Return (idx, frame, previous_frame_or_None) for frame_diff presets."""
    frames = list(iter_frames(source, stride=max(1, 50)))
    if not frames:
        frames = list(iter_frames(source, stride=1))
    if not frames:
        raise RuntimeError(f"No frames in {source}")

    if frame_index is not None:
        exact = list(iter_frames(source, stride=1))
        by_idx = {idx: fr for idx, fr in exact}
        if frame_index not in by_idx:
            raise RuntimeError(f"Frame index {frame_index} not found")
        prev = by_idx.get(frame_index - 1)
        return frame_index, by_idx[frame_index], prev

    mid = len(frames) // 2
    idx, frame = frames[mid]
    prev = frames[mid - 1][1] if mid > 0 else None
    return idx, frame, prev


def draw_guide_lines(
    frame: np.ndarray,
    floor_y: int,
    rising_y: int,
    side_crop_ratio: float = 0.25,
) -> np.ndarray:
    canvas = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    side = int(round(w * side_crop_ratio))

    cv2.line(canvas, (0, rising_y), (w, rising_y), (255, 220, 0), 2)
    cv2.putText(
        canvas,
        f"rising_y={rising_y}",
        (side + 8, max(20, rising_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (0, floor_y), (w, floor_y), (0, 0, 255), 2)
    cv2.putText(
        canvas,
        f"floor_y={floor_y}",
        (side + 8, min(h - 10, floor_y + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (side, 0), (side, h), (0, 200, 0), 1)
    cv2.line(canvas, (w - side, 0), (w - side, h), (0, 200, 0), 1)
    cv2.rectangle(canvas, (side, rising_y), (w - side, floor_y), (0, 255, 255), 1)
    return canvas


def _annotate_title(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bar = np.zeros((34, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def build_palette(
    panels: list[np.ndarray],
    cols: int = 1,
    pad: int = 12,
    bg_color: tuple[int, int, int] = (24, 24, 24),
    v_scale: float = 2.0,
) -> np.ndarray:
    if not panels:
        raise ValueError("No panels")
    cols = max(1, cols)
    scaled: list[np.ndarray] = []
    for panel in panels:
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        if v_scale != 1.0:
            panel = cv2.resize(
                panel,
                (panel.shape[1], max(1, int(round(panel.shape[0] * v_scale)))),
                interpolation=cv2.INTER_NEAREST,
            )
        scaled.append(panel)

    ph = max(p.shape[0] for p in scaled)
    pw = max(p.shape[1] for p in scaled)
    cells = []
    for panel in scaled:
        cell = np.full((ph, pw, 3), bg_color, dtype=np.uint8)
        cell[: panel.shape[0], : panel.shape[1]] = panel
        cells.append(cell)

    rows = (len(cells) + cols - 1) // cols
    canvas = np.full(
        (rows * ph + (rows + 1) * pad, cols * pw + (cols + 1) * pad, 3),
        bg_color,
        dtype=np.uint8,
    )
    for i, panel in enumerate(cells):
        r, c = divmod(i, cols)
        y = pad + r * (ph + pad)
        x = pad + c * (pw + pad)
        canvas[y : y + ph, x : x + pw] = panel
    return canvas


def run_preview(
    source: Path,
    out_dir: Path,
    *,
    frame_index: int | None = None,
    floor_y: int | None = None,
    rising_y: int | None = None,
    side_crop_ratio: float = 0.25,
    default_crop_height: int = 80,
    image_params: ImageParams | None = None,
) -> Path:
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    masks_dir = out_dir / "masks"
    cells_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    params = image_params or ImageParams()
    frame_idx, frame, prev_frame = pick_frame_pair(source, frame_index)
    gray_full = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fy = int(floor_y) if floor_y is not None else detect_floor_y(gray_full)
    ry = int(rising_y) if rising_y is not None else default_rising_y(fy, default_crop_height)
    if ry >= fy:
        raise ValueError(f"rising_y ({ry}) must be < floor_y ({fy})")

    roi = crop_box_from_lines(gray_full.shape, fy, ry, side_crop_ratio=side_crop_ratio)
    gray_crop = apply_roi(gray_full, roi)
    enhanced = enhance_image(gray_crop, params)
    prev_crop = None
    if prev_frame is not None:
        prev_gray = prev_frame if prev_frame.ndim == 2 else cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_crop = enhance_image(apply_roi(prev_gray, roi), params)

    background_crop = None
    try:
        bg_full = estimate_background(source, max_frames=40, sample_stride=20)
        background_crop = enhance_image(apply_roi(bg_full, roi), params)
    except Exception:  # noqa: BLE001
        background_crop = None

    guided = draw_guide_lines(gray_full, fy, ry, side_crop_ratio=side_crop_ratio)
    cv2.imwrite(str(out_dir / "original_guides.jpg"), guided)
    cv2.imwrite(str(out_dir / "source_crop.png"), gray_crop)
    cv2.imwrite(str(out_dir / "enhanced_crop.png"), enhanced)

    panels: list[np.ndarray] = [
        _annotate_title(gray_crop, "00_raw_crop", f"floor_y={fy} rising_y={ry}"),
        _annotate_title(
            enhanced,
            "00_enhanced_crop",
            f"contrast={params.contrast} brightness={params.brightness} clahe={params.clahe}",
        ),
    ]
    summary = {
        "source": str(source.resolve()),
        "frame_index": frame_idx,
        "floor_y": fy,
        "rising_y": ry,
        "side_crop_ratio": side_crop_ratio,
        "roi": list(roi),
        "image_params": params.to_dict(),
        "presets": [],
    }

    for preset in all_presets():
        binary, detections = detect_bubbles(
            enhanced,
            preset.detect,
            background=background_crop,
            prev_frame=prev_crop,
        )
        h, _ = enhanced.shape[:2]
        labeled = []
        rising = sitting = 0
        for det in detections:
            class_id = classify_detection(det, h, preset.bottom_ratio)
            labeled.append((det, class_id))
            if class_id == 0:
                rising += 1
            else:
                sitting += 1
        overlay = draw_debug(enhanced, labeled)
        band = max(1, int(h * preset.bottom_ratio))
        cv2.line(overlay, (0, h - band), (overlay.shape[1], h - band), (255, 80, 80), 1)
        titled = _annotate_title(
            overlay,
            preset.name,
            f"{preset.category}/{preset.detect.method}  dets={len(detections)} r={rising} s={sitting}",
        )
        panels.append(titled)
        cv2.imwrite(str(cells_dir / f"{preset.name}.jpg"), titled)
        cv2.imwrite(str(masks_dir / f"{preset.name}_mask.png"), binary)
        summary["presets"].append(
            {
                **preset_to_dict(preset),
                "detections": len(detections),
                "rising": rising,
                "sitting": sitting,
            }
        )
        print(
            f"{preset.name}: {preset.category}/{preset.detect.method} "
            f"dets={len(detections)} rising={rising} sitting={sitting}"
        )

    palette = build_palette(panels, cols=1, v_scale=2.0)
    native = build_palette(panels, cols=1, v_scale=1.0)
    cv2.imwrite(str(out_dir / "palette.jpg"), palette, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / "palette_native.png"), native)

    meta_path = out_dir / "preview.json"
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"palette: {out_dir / 'palette.jpg'}")
    print(f"meta   : {meta_path}")
    print(f"lines  : floor_y={fy}, rising_y={ry}, roi={roi}")
    return meta_path
