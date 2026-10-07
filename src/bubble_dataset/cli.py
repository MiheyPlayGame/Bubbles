"""CLI: preview palette + export with search preset and image enhance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bubble_dataset.image_ops import ImageParams
from bubble_dataset.paths import resolve_source
from bubble_dataset.pipeline import PipelineConfig, run_pipeline
from bubble_dataset.presets import all_presets, detect_from_dict, get_preset
from bubble_dataset.preview import run_preview


def _add_source_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "source",
        type=str,
        help="CINE path/name/stem, e.g. 24, 24.cine, data/films/24.cine",
    )


def _add_image_args(p: argparse.ArgumentParser) -> None:
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


def _image_params_from_args(args: argparse.Namespace, base: ImageParams | None = None) -> ImageParams:
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


def _image_params_from_preview(meta: dict) -> ImageParams:
    from dataclasses import fields

    raw = meta.get("image_params") or {}
    valid = {f.name for f in fields(ImageParams)}
    return ImageParams(**{k: v for k, v in raw.items() if k in valid})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bubble-dataset",
        description="Bubble dataset: preview search presets, then export YOLO labels.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prev = sub.add_parser("preview", help="Guides + enhance + search preset palette")
    _add_source_arg(prev)
    prev.add_argument("-o", "--output", type=Path, default=None)
    prev.add_argument("--frame-index", type=int, default=None)
    prev.add_argument("--floor-y", type=int, default=None)
    prev.add_argument("--rising-y", type=int, default=None)
    prev.add_argument("--side-crop-ratio", type=float, default=0.25)
    prev.add_argument("--crop-height", type=int, default=80)
    _add_image_args(prev)

    exp = sub.add_parser("export", help="Export enhanced PNG + YOLO labels for one preset")
    _add_source_arg(exp)
    exp.add_argument("-o", "--output", type=Path, default=None)
    exp.add_argument("--stride", type=int, default=500)
    exp.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Search preset from palette, e.g. 03_blackhat or blackhat",
    )
    exp.add_argument("--from-preview", type=Path, default=None)
    exp.add_argument("--floor-y", type=int, default=None)
    exp.add_argument("--rising-y", type=int, default=None)
    exp.add_argument("--side-crop-ratio", type=float, default=None)
    exp.add_argument("--bottom-ratio", type=float, default=None)
    exp.add_argument("--debug", action="store_true")
    _add_image_args(exp)

    sub.add_parser("list-presets", help="List search presets")
    return p


def _cmd_preview(args: argparse.Namespace) -> int:
    source = resolve_source(args.source)
    out = args.output or Path("data") / f"preview_{source.stem}"
    params = _image_params_from_args(args)
    print(f"preview source: {source}")
    print(f"preview output: {out}")
    run_preview(
        source,
        out,
        frame_index=args.frame_index,
        floor_y=args.floor_y,
        rising_y=args.rising_y,
        side_crop_ratio=args.side_crop_ratio,
        default_crop_height=args.crop_height,
        image_params=params,
    )
    print("\nNext: pick a preset from palette, then:")
    print(
        f"  python -m bubble_dataset export {source.stem} "
        f"--from-preview {out / 'preview.json'} --preset <NAME> --stride 500 --debug"
    )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    source = resolve_source(args.source)
    out = args.output or Path("data") / f"dataset_{source.stem}"

    floor_y = args.floor_y
    rising_y = args.rising_y
    side_crop_ratio = args.side_crop_ratio
    base_params = ImageParams()
    preview_meta = None

    if args.from_preview is not None:
        preview_meta = json.loads(Path(args.from_preview).read_text(encoding="utf-8"))
        floor_y = floor_y if floor_y is not None else preview_meta.get("floor_y")
        rising_y = rising_y if rising_y is not None else preview_meta.get("rising_y")
        if side_crop_ratio is None:
            side_crop_ratio = float(preview_meta.get("side_crop_ratio", 0.25))
        base_params = _image_params_from_preview(preview_meta)

    if side_crop_ratio is None:
        side_crop_ratio = 0.25

    try:
        preset = get_preset(args.preset)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    detect = preset.detect
    bottom_ratio = args.bottom_ratio if args.bottom_ratio is not None else preset.bottom_ratio
    if preview_meta is not None:
        for item in preview_meta.get("presets", []):
            if item.get("name") == preset.name:
                detect = detect_from_dict(item["detect"])
                if args.bottom_ratio is None:
                    bottom_ratio = float(item.get("bottom_ratio", bottom_ratio))
                break

    params = _image_params_from_args(args, base=base_params)

    if args.stride < 1:
        print("error: --stride must be >= 1", file=sys.stderr)
        return 2
    if not (0.0 <= side_crop_ratio < 0.5):
        print("error: --side-crop-ratio must be in [0, 0.5)", file=sys.stderr)
        return 2

    cfg = PipelineConfig(
        stride=args.stride,
        floor_y=floor_y,
        rising_y=rising_y,
        side_crop_ratio=side_crop_ratio,
        bottom_ratio=bottom_ratio,
        image_params=params,
        detect=detect,
        preset_name=preset.name,
        save_debug=args.debug,
    )

    print(f"export source : {source}")
    print(f"export output : {out}")
    print(f"preset        : {preset.name} ({preset.category}/{detect.method})")
    print(f"stride        : {cfg.stride}")
    print(f"floor_y       : {cfg.floor_y}")
    print(f"rising_y      : {cfg.rising_y}")
    print(f"image         : {params.to_dict()}")

    stats = run_pipeline(source, out, cfg)
    print(
        "done: "
        f"frames={stats['frames']}, "
        f"dets={stats['detections']}, "
        f"rising={stats['rising']}, "
        f"sitting={stats['sitting']}, "
        f"empty={stats['empty']}"
    )
    return 0


def _cmd_list_presets(_: argparse.Namespace) -> int:
    for preset in all_presets():
        print(f"{preset.name}\t{preset.category}\tmethod={preset.detect.method}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preview":
        return _cmd_preview(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "list-presets":
        return _cmd_list_presets(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
