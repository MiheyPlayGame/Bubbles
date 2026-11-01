"""CINE → dataset folder (enhanced crops) + optional YOLO pre-labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from bubble_yolo import CLASS_NAMES
from bubble_yolo.cine_io import iter_frames
from bubble_yolo.floor import crop_box_from_lines, default_rising_y, detect_floor_y
from bubble_yolo.image_args import add_image_args, image_params_from_args, image_params_from_dict
from bubble_yolo.image_ops import ImageParams, enhance_image
from bubble_yolo.paths import ARTIFACTS_DIR, REPO_ROOT, resolve_source
from bubble_yolo.preview import apply_roi


def resolve_geometry(
    sample_frame,
    *,
    floor_y: int | None,
    rising_y: int | None,
    crop_height: int,
    side_crop_ratio: float,
):
    gray = sample_frame if sample_frame.ndim == 2 else cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
    fy = floor_y if floor_y is not None else detect_floor_y(gray)
    ry = rising_y if rising_y is not None else default_rising_y(fy, crop_height)
    if ry >= fy:
        raise ValueError(f"rising_y ({ry}) must be < floor_y ({fy})")
    roi = crop_box_from_lines(sample_frame.shape, fy, ry, side_crop_ratio=side_crop_ratio)
    return roi, fy, ry


def export_images(
    source: Path,
    out_dir: Path,
    *,
    stride: int,
    floor_y: int | None,
    rising_y: int | None,
    crop_height: int,
    side_crop_ratio: float,
    image_params: ImageParams,
) -> dict:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    roi = None
    fy = floor_y
    ry = rising_y
    n = 0
    for frame_idx, frame in iter_frames(source, stride=stride):
        if roi is None:
            roi, fy, ry = resolve_geometry(
                frame,
                floor_y=floor_y,
                rising_y=rising_y,
                crop_height=crop_height,
                side_crop_ratio=side_crop_ratio,
            )
            print(f"crop   : roi={roi} floor_y={fy} rising_y={ry}")

        cropped = apply_roi(frame, roi)
        enhanced = enhance_image(cropped, image_params)
        stem = f"frame_{frame_idx:06d}"
        cv2.imwrite(str(images_dir / f"{stem}.png"), enhanced)
        n += 1

    if n == 0:
        raise RuntimeError(f"No frames exported from {source}")

    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    meta = {
        "source": str(source.resolve()),
        "frames": n,
        "stride": stride,
        "floor_y": fy,
        "rising_y": ry,
        "side_crop_ratio": side_crop_ratio,
        "roi": list(roi) if roi else None,
        "image_params": image_params.to_dict(),
    }
    (out_dir / "export.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def default_weights() -> Path:
    best = ARTIFACTS_DIR / "runs" / "bubbles_yolo11s" / "weights" / "best.pt"
    if best.is_file():
        return best
    # fallback: any best.pt under runs
    matches = sorted(ARTIFACTS_DIR.glob("runs/*/weights/best.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No YOLO weights found under {ARTIFACTS_DIR / 'runs'}. Train first or pass --weights."
    )


def run_predict_on_export(
    out_dir: Path,
    *,
    weights: Path,
    conf: float,
    imgsz: int,
    save_overlay: bool,
) -> int:
    from bubble_yolo.predict import main as predict_main

    argv = [
        "--source",
        str(out_dir / "images"),
        "--weights",
        str(weights),
        "--out",
        str(out_dir / "labels"),
        "--conf",
        str(conf),
        "--imgsz",
        str(imgsz),
    ]
    if save_overlay:
        argv.append("--save-overlay")
    return predict_main(argv)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export CINE → data/dataset_<stem> (crops + optional YOLO labels)"
    )
    p.add_argument("source", type=str, help="CINE path/name/stem, e.g. 24 or data/films/24.cine")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--stride", type=int, default=500)
    p.add_argument("--floor-y", type=int, default=None)
    p.add_argument("--rising-y", type=int, default=None)
    p.add_argument("--side-crop-ratio", type=float, default=None)
    p.add_argument("--crop-height", type=int, default=80)
    p.add_argument(
        "--from-preview",
        type=Path,
        default=None,
        help="preview.json to reuse lines + image_params",
    )
    p.add_argument(
        "--predict",
        action="store_true",
        help="After export, run YOLO and write labels/",
    )
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save-overlay", action="store_true")
    add_image_args(p)
    args = p.parse_args(argv)

    if args.stride < 1:
        print("error: --stride must be >= 1", file=sys.stderr)
        return 2

    floor_y = args.floor_y
    rising_y = args.rising_y
    side_crop_ratio = args.side_crop_ratio
    crop_height = args.crop_height
    base_params = ImageParams()

    if args.from_preview is not None:
        meta = json.loads(Path(args.from_preview).read_text(encoding="utf-8"))
        floor_y = floor_y if floor_y is not None else meta.get("floor_y")
        rising_y = rising_y if rising_y is not None else meta.get("rising_y")
        if side_crop_ratio is None:
            side_crop_ratio = float(meta.get("side_crop_ratio", 0.25))
        if meta.get("crop_height") is not None and args.crop_height == 80:
            crop_height = int(meta["crop_height"])
        base_params = image_params_from_dict(meta.get("image_params") or {})

    if side_crop_ratio is None:
        side_crop_ratio = 0.25
    if not (0.0 <= side_crop_ratio < 0.5):
        print("error: --side-crop-ratio must be in [0, 0.5)", file=sys.stderr)
        return 2

    params = image_params_from_args(args, base=base_params)

    try:
        source = resolve_source(args.source)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = args.output or (REPO_ROOT / "data" / f"dataset_{source.stem}")
    print(f"export source : {source}")
    print(f"export output : {out}")
    print(f"stride        : {args.stride}")
    print(f"image         : {params.to_dict()}")

    try:
        meta = export_images(
            source,
            out,
            stride=args.stride,
            floor_y=floor_y,
            rising_y=rising_y,
            crop_height=crop_height,
            side_crop_ratio=side_crop_ratio,
            image_params=params,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"done: frames={meta['frames']} → {out / 'images'}")

    if args.predict:
        try:
            weights = (args.weights or default_weights()).resolve()
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"predict weights: {weights}")
        code = run_predict_on_export(
            out,
            weights=weights,
            conf=args.conf,
            imgsz=args.imgsz,
            save_overlay=args.save_overlay,
        )
        if code != 0:
            return code
        print(f"labels: {out / 'labels'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
