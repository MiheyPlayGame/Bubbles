"""Preview crop guides + enhance on one CINE frame (no classical CV presets)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bubble_yolo.cine_io import iter_frames
from bubble_yolo.floor import crop_box_from_lines, default_rising_y, detect_floor_y
from bubble_yolo.image_ops import ImageParams, enhance_image


def apply_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def pick_frame(source: Path, frame_index: int | None) -> tuple[int, np.ndarray]:
    frames = list(iter_frames(source, stride=max(1, 50)))
    if not frames:
        frames = list(iter_frames(source, stride=1))
    if not frames:
        raise RuntimeError(f"No frames in {source}")
    if frame_index is None:
        return frames[len(frames) // 2]
    exact = list(iter_frames(source, stride=1))
    for idx, frame in exact:
        if idx == frame_index:
            return idx, frame
    raise RuntimeError(f"frame_index={frame_index} not found in {source}")


def draw_guides(
    gray: np.ndarray,
    floor_y: int,
    rising_y: int,
    side_crop_ratio: float,
) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    side = int(round(w * side_crop_ratio))
    cv2.line(canvas, (0, rising_y), (w, rising_y), (255, 220, 0), 2)
    cv2.line(canvas, (0, floor_y), (w, floor_y), (0, 0, 255), 2)
    cv2.line(canvas, (side, 0), (side, h), (0, 255, 255), 1)
    cv2.line(canvas, (w - side, 0), (w - side, h), (0, 255, 255), 1)
    cv2.rectangle(canvas, (side, rising_y), (w - side, floor_y), (0, 255, 255), 1)
    cv2.putText(
        canvas,
        f"rising_y={rising_y}",
        (side + 8, max(20, rising_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"floor_y={floor_y}",
        (side + 8, min(h - 10, floor_y + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def run_preview(
    source: Path,
    out_dir: Path,
    *,
    frame_index: int | None = None,
    floor_y: int | None = None,
    rising_y: int | None = None,
    side_crop_ratio: float = 0.25,
    crop_height: int = 80,
    image_params: ImageParams | None = None,
) -> Path:
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = image_params or ImageParams()

    frame_idx, frame = pick_frame(source, frame_index)
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fy = int(floor_y) if floor_y is not None else detect_floor_y(gray)
    ry = int(rising_y) if rising_y is not None else default_rising_y(fy, crop_height)
    if ry >= fy:
        raise ValueError(f"rising_y ({ry}) must be < floor_y ({fy})")

    roi = crop_box_from_lines(gray.shape, fy, ry, side_crop_ratio=side_crop_ratio)
    crop = apply_roi(gray, roi)
    enhanced = enhance_image(crop, params)
    guides = draw_guides(gray, fy, ry, side_crop_ratio)

    cv2.imwrite(str(out_dir / "original_guides.jpg"), guides)
    cv2.imwrite(str(out_dir / "source_crop.png"), crop)
    cv2.imwrite(str(out_dir / "enhanced_crop.png"), enhanced)

    meta = {
        "source": str(source.resolve()),
        "frame_index": frame_idx,
        "floor_y": fy,
        "rising_y": ry,
        "side_crop_ratio": side_crop_ratio,
        "crop_height": crop_height,
        "roi": list(roi),
        "image_params": params.to_dict(),
    }
    meta_path = out_dir / "preview.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"guides : {out_dir / 'original_guides.jpg'}")
    print(f"crop   : {out_dir / 'enhanced_crop.png'}")
    print(f"meta   : {meta_path}")
    print(f"lines  : floor_y={fy}, rising_y={ry}, roi={roi}")
    return meta_path
