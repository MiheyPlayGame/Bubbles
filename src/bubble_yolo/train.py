"""Train a YOLO detector on prepared bubble crops."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bubble_yolo.paths import ARTIFACTS_DIR, REPO_ROOT

# Sensible sweep for small bubble crops; override with --models.
DEFAULT_MODEL_SWEEP = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolo11n.pt",
    "yolo11s.pt",
)


def _run_name(base: str, model: str, multi: bool) -> str:
    if not multi:
        return base
    stem = Path(model).stem  # yolov8n / yolo11s
    return f"{base}_{stem}"


def train_one(
    *,
    data: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    name: str,
    workers: int,
    seed: int,
    freeze: int | None,
) -> Path:
    from ultralytics import YOLO

    train_kwargs = dict(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ARTIFACTS_DIR / "runs"),
        name=name,
        exist_ok=True,
        workers=workers,
        seed=seed,
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
        close_mosaic=15,
    )
    if device is not None:
        train_kwargs["device"] = device
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    print(f"\n=== train {model} -> runs/{name} ===")
    YOLO(model).train(**train_kwargs)
    best = ARTIFACTS_DIR / "runs" / name / "weights" / "best.pt"
    print(f"best weights: {best}")
    return best


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
        default=None,
        help="Single pretrained weights file (default: yolov8n.pt if no --models)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Train several models and compare (e.g. yolov8n.pt yolov8s.pt yolo11n.pt)",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help=f"Train default sweep: {' '.join(DEFAULT_MODEL_SWEEP)}",
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or 0")
    p.add_argument("--name", type=str, default="bubbles")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze first N backbone layers (helps on tiny datasets, e.g. 10)",
    )
    args = p.parse_args(argv)

    data = args.data.resolve()
    if not data.is_file():
        print(
            f"error: {data} not found. Run: python -m bubble_yolo prepare --all-clean",
            file=sys.stderr,
        )
        return 2

    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print(
            "error: ultralytics not installed. Run:\n  pip install -e .\n"
            f"  (from {REPO_ROOT})",
            file=sys.stderr,
        )
        return 2

    if args.sweep:
        models = list(DEFAULT_MODEL_SWEEP)
    elif args.models:
        models = list(args.models)
    elif args.model:
        models = [args.model]
    else:
        models = ["yolov8n.pt"]

    multi = len(models) > 1
    results: list[tuple[str, Path]] = []
    for model in models:
        name = _run_name(args.name, model, multi)
        best = train_one(
            data=data,
            model=model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=name,
            workers=args.workers,
            seed=args.seed,
            freeze=args.freeze,
        )
        results.append((model, best))

    print("\n=== summary ===")
    for model, best in results:
        print(f"{model}: {best}")
    if multi:
        print("Compare val metrics in each runs/<name>/results.csv (look at mAP50 / mAP50-95).")
    else:
        print("next: python -m bubble_yolo predict --source data/dataset_XX/images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
