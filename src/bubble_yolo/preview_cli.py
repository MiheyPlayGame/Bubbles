"""CLI wrapper for geometry/enhance preview."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bubble_yolo.image_args import add_image_args, image_params_from_args
from bubble_yolo.paths import REPO_ROOT, resolve_source
from bubble_yolo.preview import run_preview


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Preview crop guides + enhance on one CINE frame")
    p.add_argument("source", type=str, help="CINE path/name/stem, e.g. 24")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--frame-index", type=int, default=None)
    p.add_argument("--floor-y", type=int, default=None)
    p.add_argument("--rising-y", type=int, default=None)
    p.add_argument("--side-crop-ratio", type=float, default=0.25)
    p.add_argument("--crop-height", type=int, default=80)
    add_image_args(p)
    args = p.parse_args(argv)

    if not (0.0 <= args.side_crop_ratio < 0.5):
        print("error: --side-crop-ratio must be in [0, 0.5)", file=sys.stderr)
        return 2

    try:
        source = resolve_source(args.source)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = args.output or (REPO_ROOT / "data" / f"preview_{source.stem}")
    params = image_params_from_args(args)
    print(f"preview source: {source}")
    print(f"preview output: {out}")
    try:
        meta = run_preview(
            source,
            out,
            frame_index=args.frame_index,
            floor_y=args.floor_y,
            rising_y=args.rising_y,
            side_crop_ratio=args.side_crop_ratio,
            crop_height=args.crop_height,
            image_params=params,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("\nNext: export with the same geometry/enhance:")
    print(
        f"  python -m bubble_yolo export {source.stem} "
        f"--from-preview {meta} --stride 500 --predict"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
