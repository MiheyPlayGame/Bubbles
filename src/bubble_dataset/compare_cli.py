"""CLI: compare detection variants and write a visual palette."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bubble_dataset.compare import compare_source


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare bubble detection variants on one frame.")
    p.add_argument("source", type=Path, help="Path to .cine or image folder")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/compare"),
        help="Output directory for palette and cells",
    )
    p.add_argument("--stride", type=int, default=500, help="Frame sampling stride")
    p.add_argument("--crop-height", type=int, default=80, help="Crop height above floor")
    p.add_argument(
        "--side-crop-ratio",
        type=float,
        default=0.25,
        help="Fraction of width cut from each side (default: 0.25)",
    )
    p.add_argument("--floor-y", type=int, default=None, help="Manual floor row")
    p.add_argument("--frame-index", type=int, default=None, help="Exact frame index to use")
    p.add_argument("--cols", type=int, default=1, help="Palette grid columns (default: 1 for full width)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.crop_height < 1:
        print("error: --crop-height must be >= 1", file=sys.stderr)
        return 2
    if not (0.0 <= args.side_crop_ratio < 0.5):
        print("error: --side-crop-ratio must be in [0, 0.5)", file=sys.stderr)
        return 2
    compare_source(
        args.source,
        args.output,
        stride=args.stride,
        crop_height=args.crop_height,
        side_crop_ratio=args.side_crop_ratio,
        floor_y=args.floor_y,
        frame_index=args.frame_index,
        cols=args.cols,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
