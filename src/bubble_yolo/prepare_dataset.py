"""Merge dataset_*_clean folders into train/val for Ultralytics."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

from bubble_yolo.paths import ARTIFACTS_DIR, IMAGE_EXTS, discover_clean_datasets, resolve_dataset


def paired_samples(root: Path) -> list[tuple[Path, Path | None]]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    pairs: list[tuple[Path, Path | None]] = []
    for img in images:
        lab = labels_dir / f"{img.stem}.txt"
        pairs.append((img, lab if lab.is_file() else None))
    return pairs


def load_class_names(roots: list[Path]) -> list[str]:
    for root in roots:
        path = root / "classes.txt"
        if path.is_file():
            names = [
                ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            if names:
                return names
    return ["rising", "sitting"]


def prepare(
    datasets: list[Path],
    out_dir: Path,
    val_ratio: float,
    seed: int,
) -> Path:
    samples: list[tuple[str, Path, Path | None]] = []
    for root in datasets:
        for img, lab in paired_samples(root):
            key = f"{root.name}__{img.stem}"
            samples.append((key, img, lab))

    if not samples:
        raise RuntimeError("No images found in the given datasets")

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(round(len(samples) * val_ratio))) if len(samples) > 1 else 1
    if len(samples) <= 2:
        n_val = 1
    n_val = min(n_val, len(samples) - 1) if len(samples) > 1 else 1
    val_set = {s[0] for s in samples[:n_val]}
    train_set = [s for s in samples if s[0] not in val_set]
    val_list = [s for s in samples if s[0] in val_set]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)

    def copy_split(items: list[tuple[str, Path, Path | None]], split: str) -> int:
        for key, img, lab in items:
            dst_img = out_dir / split / "images" / f"{key}{img.suffix.lower()}"
            shutil.copy2(img, dst_img)
            dst_lab = out_dir / split / "labels" / f"{key}.txt"
            if lab is not None:
                shutil.copy2(lab, dst_lab)
            else:
                dst_lab.write_text("", encoding="utf-8")
        return len(items)

    n_train = copy_split(train_set, "train")
    n_val_copied = copy_split(val_list, "val")
    names = load_class_names(datasets)

    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"datasets : {', '.join(d.name for d in datasets)}")
    print(f"train    : {n_train}")
    print(f"val      : {n_val_copied}")
    print(f"classes  : {names}")
    print(f"yaml     : {yaml_path}")
    return yaml_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Prepare train/val YOLO dataset from dataset_*_clean folders"
    )
    p.add_argument(
        "datasets",
        nargs="*",
        help="Dataset folders or stems (12, x1, dataset_12_clean). Empty + --all-clean = all clean.",
    )
    p.add_argument(
        "--all-clean",
        action="store_true",
        help="Use every data/dataset_*_clean folder",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "dataset",
        help="Output folder (default: yolo/dataset)",
    )
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation fraction (default 0.2)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not (0.0 < args.val_ratio < 1.0):
        print("error: --val-ratio must be in (0, 1)", file=sys.stderr)
        return 2
    if not args.datasets and not args.all_clean:
        print("error: pass dataset names or --all-clean", file=sys.stderr)
        return 2

    try:
        if args.all_clean:
            roots = discover_clean_datasets()
        else:
            roots = [resolve_dataset(d) for d in args.datasets]
        prepare(roots, args.output.resolve(), args.val_ratio, args.seed)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
