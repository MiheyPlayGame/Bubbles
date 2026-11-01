"""Load frames from Phantom CINE or an image directory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalize any bit-depth frame to uint8 grayscale or BGR."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        if frame.dtype == np.uint8:
            return frame
        frame_f = frame.astype(np.float32)
        frame_f -= frame_f.min()
        max_v = frame_f.max()
        if max_v > 0:
            frame_f = frame_f / max_v * 255.0
        return frame_f.astype(np.uint8)

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint8:
        return gray

    gray_f = gray.astype(np.float32)
    gray_f -= gray_f.min()
    max_v = gray_f.max()
    if max_v > 0:
        gray_f = gray_f / max_v * 255.0
    return gray_f.astype(np.uint8)


def _read_cine_cinereader(path: Path, stride: int) -> Iterator[tuple[int, np.ndarray]]:
    import cinereader as cr

    metadata = cr.read_metadata(str(path))
    first = int(getattr(metadata, "FirstImageNo", 0))
    count = int(getattr(metadata, "ImageCount", 0))
    if count <= 0:
        raise RuntimeError(f"CINE has no frames: {path}")

    if hasattr(cr, "read_image"):
        for offset in range(0, count, stride):
            frame_no = first + offset
            frame = cr.read_image(metadata, str(path), frame_start=frame_no)
            yield frame_no, _to_uint8(np.asarray(frame))
        return

    _, images, _ = cr.read(str(path), start=first, count=count)
    if images is None:
        raise RuntimeError(f"cinereader returned no images for {path}")
    for i, frame in enumerate(images):
        if i % stride != 0:
            continue
        yield first + i, _to_uint8(np.asarray(frame))


def iter_frames(source: Path, stride: int = 5) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, uint8 image) from a .cine file or image folder."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise RuntimeError(f"No images found in {source}")
        for i, path in enumerate(files):
            if i % stride != 0:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            yield i, _to_uint8(img)
        return

    if source.suffix.lower() != ".cine":
        raise ValueError(f"Unsupported source (need .cine or directory): {source}")

    try:
        yield from _read_cine_cinereader(source, stride)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to read CINE via cinereader. "
            "Install with `pip install cinereader` and verify the file opens in PCC. "
            f"Original error: {exc}"
        ) from exc
