"""Convert Label Studio JSON export back to YOLO labels + copy images."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from bubble_dataset import CLASS_NAMES
from bubble_dataset.paths import resolve_dataset_dir

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_tasks(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("tasks", "annotations", "data"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError(f"Unrecognized Label Studio JSON shape in {path}")


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


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def data_uri_sha1(uri: str) -> str | None:
    if not isinstance(uri, str) or not uri.startswith("data:"):
        return None
    try:
        _, payload = uri.split(",", 1)
    except ValueError:
        return None
    h = hashlib.sha1()
    h.update(base64.b64decode(payload))
    return h.hexdigest()


def name_from_image_ref(ref: str) -> str | None:
    if not isinstance(ref, str) or not ref or ref.startswith("data:"):
        return None
    parsed = urlparse(ref)
    qs = parse_qs(parsed.query)
    if "d" in qs and qs["d"]:
        return Path(unquote(qs["d"][0])).name
    # /data/upload/.../frame_001500.png or plain path
    candidate = Path(unquote(parsed.path if parsed.scheme or parsed.path else ref)).name
    if candidate and Path(candidate).suffix.lower() in IMAGE_EXTS:
        return candidate
    return None


def pick_annotation(task: dict) -> dict | None:
    anns = task.get("annotations") or task.get("completions") or []
    if not anns:
        return None
    # Prefer completed annotations; fall back to last entry.
    completed = [a for a in anns if not a.get("was_cancelled")]
    pool = completed or anns
    return max(pool, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def results_to_yolo_lines(results: list[dict], class_names: list[str]) -> list[str]:
    name_to_id = {n: i for i, n in enumerate(class_names)}
    lines: list[str] = []
    for item in results:
        item_type = item.get("type")
        if item_type not in (None, "rectanglelabels"):
            continue
        value = item.get("value") or {}
        labels = value.get("rectanglelabels") or value.get("labels") or []
        if not labels:
            continue
        label = labels[0]
        if label not in name_to_id:
            raise ValueError(f"Unknown label '{label}'. Expected one of {class_names}")
        x = float(value["x"])
        y = float(value["y"])
        w = float(value["width"])
        h = float(value["height"])
        xc = (x + w / 2.0) / 100.0
        yc = (y + h / 2.0) / 100.0
        bw = w / 100.0
        bh = h / 100.0
        lines.append(f"{name_to_id[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def resolve_image_name(
    task: dict,
    hash_to_name: dict[str, str],
    fallback_names: list[str],
    index: int,
) -> str:
    meta = task.get("meta") or {}
    for key in ("image_name", "filename", "file_name", "name"):
        if meta.get(key):
            return Path(str(meta[key])).name

    data = task.get("data") or {}
    image_ref = data.get("image")
    if isinstance(image_ref, str):
        by_ref = name_from_image_ref(image_ref)
        if by_ref:
            return by_ref
        digest = data_uri_sha1(image_ref)
        if digest and digest in hash_to_name:
            return hash_to_name[digest]

    if index < len(fallback_names):
        return fallback_names[index]
    raise ValueError(f"Cannot resolve image name for task index {index}")


def run(
    name: str,
    ls_json: Path,
    output: Path | None = None,
    data_dir: Path = Path("data"),
) -> Path:
    root = resolve_dataset_dir(name, data_dir=data_dir)
    images_dir = root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")

    images = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise RuntimeError(f"No images in {images_dir}")

    hash_to_name = {file_sha1(p): p.name for p in images}
    fallback_names = [p.name for p in images]
    class_names = load_class_names(root)
    tasks = load_tasks(ls_json)

    out_root = output or (data_dir / f"{root.name}_clean")
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    empty = 0
    for idx, task in enumerate(tasks):
        image_name = resolve_image_name(task, hash_to_name, fallback_names, idx)
        src = images_dir / image_name
        if not src.is_file():
            # allow stem match if extension differs
            matches = list(images_dir.glob(Path(image_name).stem + ".*"))
            if not matches:
                raise FileNotFoundError(f"Image not found in dataset: {image_name}")
            src = matches[0]
            image_name = src.name

        shutil.copy2(src, out_images / image_name)

        ann = pick_annotation(task)
        results = (ann or {}).get("result") or []
        lines = results_to_yolo_lines(results, class_names)
        label_path = out_labels / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
        if not lines:
            empty += 1

    (out_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"OK: {out_root}")
    print(f"frames: {written} (empty labels: {empty})")
    return out_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Label Studio JSON export -> YOLO labels using original dataset images."
    )
    p.add_argument("name", help="Dataset name/stem, e.g. 53 -> data/dataset_53")
    p.add_argument("ls_json", type=Path, help="Path to Label Studio JSON export")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: data/dataset_<name>_clean)",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    try:
        run(args.name, args.ls_json, output=args.output, data_dir=args.data_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
