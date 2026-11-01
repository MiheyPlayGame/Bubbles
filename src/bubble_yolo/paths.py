"""Shared path helpers for the YOLO package."""

from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent.parent
# Local artifacts stay under yolo/ (gitignored): prepared dataset + training runs.
ARTIFACTS_DIR = REPO_ROOT / "yolo"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_dataset(name: str | Path) -> Path:
    """Resolve a labeled dataset. Prefer dataset_<stem>_clean under data/."""
    raw = Path(name)
    stem = raw.name
    if stem.startswith("dataset_"):
        stem = stem[len("dataset_") :]
    if stem.endswith("_clean"):
        stem = stem[: -len("_clean")]

    candidates = [
        raw,
        PKG_DIR / raw,
        REPO_ROOT / raw,
        REPO_ROOT / "data" / raw.name,
        REPO_ROOT / "data" / f"dataset_{stem}_clean",
        REPO_ROOT / "data" / f"dataset_{stem}",
        REPO_ROOT / "data" / f"dataset_{raw.name}",
    ]
    if raw.name.startswith("dataset_"):
        candidates.append(REPO_ROOT / "data" / raw.name)

    for cand in candidates:
        if cand.is_dir() and (cand / "images").is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Dataset not found: {name}")


def discover_clean_datasets(data_dir: Path | None = None) -> list[Path]:
    root = data_dir or (REPO_ROOT / "data")
    found = sorted(
        p for p in root.glob("dataset_*_clean") if p.is_dir() and (p / "images").is_dir()
    )
    if not found:
        raise FileNotFoundError(f"No dataset_*_clean folders in {root}")
    return found


def resolve_images_dir(path: Path) -> Path:
    raw = Path(path)
    candidates = [
        raw,
        PKG_DIR / raw,
        REPO_ROOT / raw,
        REPO_ROOT / "data" / raw.name,
        REPO_ROOT / "data" / f"dataset_{raw.name}",
    ]
    for cand in candidates:
        if cand.is_dir():
            if (cand / "images").is_dir():
                return (cand / "images").resolve()
            return cand.resolve()
    raise FileNotFoundError(f"Source folder not found: {path}")


def resolve_source(name: str | Path, films_dir: Path | None = None) -> Path:
    """
    Accept a full path, a filename, or a stem like '24' / '24.cine'.
    Search order: as given → data/films / name → data/films / name.cine
    """
    films = films_dir or (REPO_ROOT / "data" / "films")
    raw = Path(name)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                films / raw,
                films / raw.name,
                films / f"{raw.stem}.cine",
                REPO_ROOT / "data" / "films" / f"{raw.stem}.cine",
                Path("data") / "films" / f"{raw.stem}.cine",
            ]
        )
        if raw.suffix == "":
            candidates.append(films / f"{raw.name}.cine")

    seen: set[str] = set()
    for cand in candidates:
        resolved = cand.resolve() if cand.exists() else cand
        key = resolved.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"Source not found: {name}. Tried: " + ", ".join(str(c) for c in candidates[:6])
    )
