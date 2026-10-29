"""Run trained YOLO on images and write YOLO .txt labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bubble_yolo.paths import ARTIFACTS_DIR, IMAGE_EXTS, REPO_ROOT, resolve_images_dir


def default_weights() -> Path:
    best = ARTIFACTS_DIR / "runs" / "bubbles" / "weights" / "best.pt"
    last = ARTIFACTS_DIR / "runs" / "bubbles" / "weights" / "last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"No weights at {best}. Train first: python -m bubble_yolo train")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict bubbles and save YOLO labels")
    p.add_argument("--source", type=Path, required=True, help="images/ folder or dataset root")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to .pt (default: yolo/runs/bubbles/weights/best.pt)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output labels folder (default: <source_parent>/labels_pred)",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--save-overlay",
        action="store_true",
        help="Also write preview images with boxes",
    )
    args = p.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "error: ultralytics not installed. Run:\n  pip install -e .\n"
            f"  (from {REPO_ROOT})",
            file=sys.stderr,
        )
        return 2

    try:
        source = resolve_images_dir(args.source)
        weights = (args.weights or default_weights()).resolve()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not weights.is_file():
        print(f"error: weights not found: {weights}", file=sys.stderr)
        return 2

    images = sorted(
        p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"error: no images in {source}", file=sys.stderr)
        return 2

    out_labels = args.out
    if out_labels is None:
        out_labels = source.parent / "labels_pred"
    out_labels = out_labels.resolve()
    out_labels.mkdir(parents=True, exist_ok=True)

    out_overlay = None
    if args.save_overlay:
        out_overlay = out_labels.parent / f"{out_labels.name}_overlay"
        out_overlay.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    names = model.names
    print(f"weights : {weights}")
    print(f"source  : {source} ({len(images)} images)")
    print(f"labels  : {out_labels}")
    print(f"classes : {names}")

    predict_kwargs = dict(
        source=[str(p) for p in images],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        stream=True,
        verbose=False,
    )
    if args.device is not None:
        predict_kwargs["device"] = args.device

    n_det = 0
    for result in model.predict(**predict_kwargs):
        img_path = Path(result.path)
        lines: list[str] = []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls.item())
                xywhn = box.xywhn.cpu().numpy().reshape(-1)
                xc, yc, bw, bh = map(float, xywhn[:4])
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                n_det += 1
        (out_labels / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        if out_overlay is not None:
            import cv2

            plotted = result.plot()
            cv2.imwrite(str(out_overlay / f"{img_path.stem}.jpg"), plotted)

    print(f"done: {len(images)} images, {n_det} boxes")
    if out_overlay is not None:
        print(f"overlay: {out_overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
