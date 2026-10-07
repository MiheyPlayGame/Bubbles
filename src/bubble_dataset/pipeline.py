"""End-to-end CINE → enhanced frames + YOLO labels via search preset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from bubble_dataset import CLASS_NAMES
from bubble_dataset.cine_io import estimate_background, iter_frames
from bubble_dataset.classify import classify_detection
from bubble_dataset.detect import DetectConfig, Detection, detect_bubbles
from bubble_dataset.floor import crop_box_from_lines, default_rising_y, detect_floor_y
from bubble_dataset.image_ops import ImageParams, enhance_image


@dataclass
class PipelineConfig:
    stride: int = 5
    roi: tuple[int, int, int, int] | None = None
    floor_y: int | None = None
    rising_y: int | None = None
    crop_height: int = 80
    side_crop_ratio: float = 0.25
    bottom_ratio: float = 0.25
    image_params: ImageParams | None = None
    detect: DetectConfig | None = None
    preset_name: str | None = None
    bg_max_frames: int = 50
    bg_sample_stride: int = 10
    save_debug: bool = False


def apply_roi(frame, roi: tuple[int, int, int, int] | None):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def detection_to_yolo(det: Detection, class_id: int, width: int, height: int) -> str:
    xc = min(max(det.cx / width, 0.0), 1.0)
    yc = min(max(det.cy / height, 0.0), 1.0)
    bw = min(max(det.w / width, 0.0), 1.0)
    bh = min(max(det.h / height, 0.0), 1.0)
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def draw_debug(image, labeled: list[tuple[Detection, int]]):
    canvas = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = {0: (0, 220, 0), 1: (0, 140, 255)}
    for det, class_id in labeled:
        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(canvas, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
        cv2.putText(
            canvas,
            CLASS_NAMES[class_id],
            (det.x, max(15, det.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_data_yaml(out_dir: Path) -> None:
    names = ", ".join(f"'{n}'" for n in CLASS_NAMES)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        f"names: [{names}]\n"
        "nc: 2\n",
        encoding="utf-8",
    )


def resolve_geometry(sample_frame, cfg: PipelineConfig):
    if cfg.roi is not None:
        fy = cfg.floor_y if cfg.floor_y is not None else cfg.roi[1] + cfg.roi[3]
        ry = cfg.rising_y if cfg.rising_y is not None else cfg.roi[1]
        return cfg.roi, fy, ry

    gray = sample_frame if sample_frame.ndim == 2 else cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
    floor_y = cfg.floor_y if cfg.floor_y is not None else detect_floor_y(gray)
    rising_y = (
        cfg.rising_y if cfg.rising_y is not None else default_rising_y(floor_y, cfg.crop_height)
    )
    if rising_y >= floor_y:
        raise ValueError(f"rising_y ({rising_y}) must be < floor_y ({floor_y})")
    roi = crop_box_from_lines(
        sample_frame.shape,
        floor_y,
        rising_y,
        side_crop_ratio=cfg.side_crop_ratio,
    )
    return roi, floor_y, rising_y


def run_pipeline(source: Path, out_dir: Path, cfg: PipelineConfig) -> dict[str, int]:
    source = Path(source)
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    debug_dir = out_dir / "debug"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    params = cfg.image_params or ImageParams()
    detect_cfg = cfg.detect or DetectConfig()
    roi = cfg.roi
    floor_y = cfg.floor_y if cfg.floor_y is not None else -1
    rising_y = cfg.rising_y if cfg.rising_y is not None else -1

    background_crop = None
    if detect_cfg.method == "bg_absdiff":
        try:
            bg_full = estimate_background(
                source,
                max_frames=cfg.bg_max_frames,
                sample_stride=cfg.bg_sample_stride,
            )
        except Exception:  # noqa: BLE001
            bg_full = None
    else:
        bg_full = None

    stats = {"frames": 0, "detections": 0, "rising": 0, "sitting": 0, "empty": 0}
    prev_enhanced = None

    for frame_idx, frame in iter_frames(source, stride=cfg.stride):
        if roi is None:
            roi, floor_y, rising_y = resolve_geometry(frame, cfg)
            print(
                f"crop   : roi={roi} floor_y={floor_y} rising_y={rising_y} "
                f"side_crop_ratio={cfg.side_crop_ratio}"
            )
            if bg_full is not None:
                background_crop = enhance_image(apply_roi(bg_full, roi), params)

        cropped = apply_roi(frame, roi)
        enhanced = enhance_image(cropped, params)
        binary, detections = detect_bubbles(
            enhanced,
            detect_cfg,
            background=background_crop,
            prev_frame=prev_enhanced,
        )

        h, w = enhanced.shape[:2]
        labeled = []
        lines = []
        for det in detections:
            class_id = classify_detection(det, h, cfg.bottom_ratio)
            labeled.append((det, class_id))
            lines.append(detection_to_yolo(det, class_id, w, h))
            stats["rising" if class_id == 0 else "sitting"] += 1

        stem = f"frame_{frame_idx:06d}"
        cv2.imwrite(str(images_dir / f"{stem}.png"), enhanced)
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

        if cfg.save_debug:
            overlay = draw_debug(enhanced, labeled)
            band = max(1, int(h * cfg.bottom_ratio))
            cv2.line(overlay, (0, h - band), (w, h - band), (255, 80, 80), 1)
            cv2.imwrite(str(debug_dir / f"{stem}.jpg"), overlay)
            cv2.imwrite(str(debug_dir / f"{stem}_mask.png"), binary)

        stats["frames"] += 1
        stats["detections"] += len(labeled)
        if not labeled:
            stats["empty"] += 1
        prev_enhanced = enhanced

    if roi is None:
        raise RuntimeError(f"No frames found in {source}")

    write_data_yaml(out_dir)
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    export_meta = {
        "source": str(source.resolve()),
        "preset": cfg.preset_name,
        "floor_y": floor_y,
        "rising_y": rising_y,
        "side_crop_ratio": cfg.side_crop_ratio,
        "bottom_ratio": cfg.bottom_ratio,
        "roi": list(roi),
        "stride": cfg.stride,
        "image_params": params.to_dict(),
        "detect": detect_cfg.__dict__,
        "stats": stats,
    }
    (out_dir / "export.json").write_text(
        json.dumps(export_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats
