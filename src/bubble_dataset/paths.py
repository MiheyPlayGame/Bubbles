"""Resolve cine paths and small shared helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_source(name: str | Path, films_dir: Path = Path("data/films")) -> Path:
    """
    Accept a full path, a filename, or a stem like '24' / '24.cine'.
    Search order: as given → films_dir / name → films_dir / name.cine
    """
    raw = Path(name)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                films_dir / raw,
                films_dir / raw.name,
                films_dir / f"{raw.stem}.cine",
                Path("data") / "films" / f"{raw.stem}.cine",
            ]
        )
        if raw.suffix == "":
            candidates.append(films_dir / f"{raw.name}.cine")

    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve() if cand.exists() else cand
        key = cand.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"Source not found: {name}. Tried: "
        + ", ".join(str(c) for c in candidates[:6])
    )


def resolve_dataset_dir(name: str | Path, data_dir: Path = Path("data")) -> Path:
    """
    Accept '24', 'dataset_24', or a path -> data/dataset_24 (or the path as given).
    """
    raw = Path(name)
    if raw.exists() and raw.is_dir():
        return raw.resolve()

    text = str(name).strip().replace("\\", "/")
    if text.startswith("data/"):
        cand = Path(text)
        if cand.is_dir():
            return cand.resolve()
        raise FileNotFoundError(f"Dataset folder not found: {cand}")

    stem = text
    if stem.startswith("dataset_"):
        stem = stem[len("dataset_") :]
    if stem.endswith(".cine"):
        stem = Path(stem).stem

    cand = data_dir / f"dataset_{stem}"
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(
        f"Dataset folder not found: {cand}. "
        f"Expected images/ + labels/ after export (pass name like '24')."
    )
