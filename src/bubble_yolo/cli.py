"""CLI entry: preview / export / prepare / train / predict."""

from __future__ import annotations

import argparse
import sys

COMMANDS = ("preview", "export", "prepare", "train", "predict")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        prog="bubble-yolo",
        description="CINE → dataset + YOLO pre-labeling for rising / sitting bubbles",
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("preview", help="Crop guides + enhance preview on one frame")
    sub.add_parser("export", help="CINE → data/dataset_* images (+ optional YOLO labels)")
    sub.add_parser("prepare", help="Build train/val from dataset_*_clean")
    sub.add_parser("train", help="Train YOLO detector")
    sub.add_parser("predict", help="Run detector and write YOLO labels")

    if not argv:
        p.print_help()
        return 2

    command = argv[0]
    rest = argv[1:]
    if command not in COMMANDS:
        p.print_help()
        return 2

    if command == "preview":
        from bubble_yolo.preview_cli import main as preview_main

        return preview_main(rest)
    if command == "export":
        from bubble_yolo.export_cine import main as export_main

        return export_main(rest)
    if command == "prepare":
        from bubble_yolo.prepare_dataset import main as prepare_main

        return prepare_main(rest)
    if command == "train":
        from bubble_yolo.train import main as train_main

        return train_main(rest)
    from bubble_yolo.predict import main as predict_main

    return predict_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
