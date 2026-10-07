"""Convert a YOLO export folder into Label Studio JSON with embedded images."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

from PIL import Image

from bubble_dataset import CLASS_NAMES
from bubble_dataset.paths import resolve_dataset_dir

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_class_names(root: Path) -> list[str]:
    classes_path = root / "classes.txt"
    if classes_path.is_file():
        names = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if names:
            return names
    return list(CLASS_NAMES)


def yolo_line_to_result(
    line: str,
    class_names: list[str],
    width: int,
    height: int,
) -> dict | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(f"class_id {class_id} out of range for classes {class_names}")

    x = (xc - bw / 2.0) * 100.0
    y = (yc - bh / 2.0) * 100.0
    w = bw * 100.0
    h = bh * 100.0

    return {
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rotation": 0,
            "rectanglelabels": [class_names[class_id]],
        },
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "manual",
    }


def image_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_tasks(root: Path) -> list[dict]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels folder: {labels_dir}")

    class_names = load_class_names(root)
    images = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    tasks: list[dict] = []
    for img_path in images:
        with Image.open(img_path) as im:
            width, height = im.size

        label_path = labels_dir / f"{img_path.stem}.txt"
        results: list[dict] = []
        if label_path.is_file():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                item = yolo_line_to_result(line, class_names, width, height)
                if item is not None:
                    results.append(item)

        task: dict = {
            "data": {"image": image_to_data_uri(img_path)},
            "meta": {"image_name": img_path.name},
        }
        if results:
            task["annotations"] = [{"result": results}]
        tasks.append(task)
    return tasks


def run(name: str, output: Path | None = None, data_dir: Path = Path("data")) -> Path:
    root = resolve_dataset_dir(name, data_dir=data_dir)
    out = output or (root / "ls_tasks_embedded.json")
    tasks = build_tasks(root)
    out.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="YOLO dataset -> Label Studio JSON with base64-embedded images."
    )
    p.add_argument(
        "name",
        help="Dataset name/stem, e.g. 24 -> data/dataset_24",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/dataset_<name>/ls_tasks_embedded.json)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Parent folder for dataset_* (default: data)",
    )
    args = p.parse_args(argv)

    try:
        out = run(args.name, output=args.output, data_dir=args.data_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tasks = json.loads(out.read_text(encoding="utf-8"))
    with_boxes = sum(1 for t in tasks if t.get("annotations"))
    print(f"OK: {out}")
    print(f"tasks: {len(tasks)} (with boxes: {with_boxes}, empty: {len(tasks) - with_boxes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
