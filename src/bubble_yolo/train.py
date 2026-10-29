"""Train a YOLO detector on prepared bubble crops."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bubble_yolo.paths import ARTIFACTS_DIR, REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train YOLO on yolo/dataset")
    p.add_argument(
        "--data",
        type=Path,
        default=ARTIFACTS_DIR / "dataset" / "data.yaml",
        help="Path to data.yaml from prepare-dataset",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained weights (yolov8n.pt recommended for small data)",
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or 0")
    p.add_argument("--name", type=str, default="bubbles")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    data = args.data.resolve()
    if not data.is_file():
        print(
            f"error: {data} not found. Run: python -m bubble_yolo prepare --all-clean",
            file=sys.stderr,
        )
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "error: ultralytics not installed. Run:\n  pip install -e .\n"
            f"  (from {REPO_ROOT})",
            file=sys.stderr,
        )
        return 2

    train_kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ARTIFACTS_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        seed=args.seed,
        rect=True,
        patience=40,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model = YOLO(args.model)
    results = model.train(**train_kwargs)
    best = ARTIFACTS_DIR / "runs" / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    if best.is_file():
        print("next: python -m bubble_yolo predict --source data/dataset_XX/images")
    return 0 if results is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
