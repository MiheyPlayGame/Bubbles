"""CLI helpers for ImageParams flags."""

from __future__ import annotations

import argparse
from dataclasses import fields

from bubble_yolo.image_ops import ImageParams


def add_image_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("image enhancement")
    g.add_argument("--contrast", type=float, default=None)
    g.add_argument("--brightness", type=float, default=None)
    g.add_argument("--gamma", type=float, default=None)
    g.add_argument("--clahe", action="store_true")
    g.add_argument("--no-clahe", action="store_true")
    g.add_argument("--clahe-clip", type=float, default=None)
    g.add_argument("--clahe-grid", type=int, default=None)
    g.add_argument("--blur", type=int, default=None)
    g.add_argument("--invert", action="store_true")
    g.add_argument("--equalize", action="store_true")


def image_params_from_args(
    args: argparse.Namespace, base: ImageParams | None = None
) -> ImageParams:
    p = base or ImageParams()
    if args.contrast is not None:
        p.contrast = args.contrast
    if args.brightness is not None:
        p.brightness = args.brightness
    if args.gamma is not None:
        p.gamma = args.gamma
    if args.clahe_clip is not None:
        p.clahe_clip = args.clahe_clip
    if args.clahe_grid is not None:
        p.clahe_grid = args.clahe_grid
    if args.blur is not None:
        p.blur = args.blur
    if args.invert:
        p.invert = True
    if args.equalize:
        p.equalize = True
    if args.clahe:
        p.clahe = True
    if args.no_clahe:
        p.clahe = False
    return p


def image_params_from_dict(raw: dict) -> ImageParams:
    valid = {f.name for f in fields(ImageParams)}
    return ImageParams(**{k: v for k, v in raw.items() if k in valid})
