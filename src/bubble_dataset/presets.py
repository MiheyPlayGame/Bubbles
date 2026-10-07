"""Search-algorithm presets for bubble detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

from bubble_dataset.detect import DetectConfig


@dataclass(frozen=True)
class Preset:
    name: str
    detect: DetectConfig
    bottom_ratio: float = 0.25
    category: str = ""


def all_presets() -> list[Preset]:
    """
    Categories:
      1) Contours — otsu, adaptive, blackhat, tophat, dog, laplacian, sobel_mag
      2) Blobs    — blob, hough, msers
      3) Motion   — bg_absdiff, frame_diff
    """
    base = DetectConfig(min_area=25, max_area=50_000, min_circularity=0.25, morph_kernel=3)

    return [
        # --- contours ---
        Preset("01_otsu", replace(base, method="otsu"), category="contours"),
        Preset("02_adaptive", replace(base, method="adaptive"), category="contours"),
        Preset("03_blackhat", replace(base, method="blackhat", hat_ksize=17), category="contours"),
        Preset("04_tophat", replace(base, method="tophat", hat_ksize=17), category="contours"),
        Preset(
            "05_dog",
            replace(base, method="dog", dog_sigma1=1.0, dog_sigma2=3.0),
            category="contours",
        ),
        Preset("06_laplacian", replace(base, method="laplacian"), category="contours"),
        Preset("07_sobel_mag", replace(base, method="sobel_mag"), category="contours"),
        # --- blobs / circles ---
        Preset(
            "08_blob",
            replace(base, method="blob", min_area=20, max_area=8000, min_circularity=0.4),
            category="blobs",
        ),
        Preset(
            "09_hough",
            replace(base, method="hough", min_area=30, hough_param2=11, hough_max_radius=35),
            category="blobs",
        ),
        Preset(
            "10_msers",
            replace(base, method="msers", min_area=20, min_circularity=0.2),
            category="blobs",
        ),
        # --- background / motion ---
        Preset("11_bg_absdiff", replace(base, method="bg_absdiff"), category="motion"),
        Preset("12_frame_diff", replace(base, method="frame_diff"), category="motion"),
    ]


def get_preset(name: str) -> Preset:
    key = name.strip()
    presets = all_presets()
    for preset in presets:
        if preset.name == key:
            return preset
    for preset in presets:
        short = preset.name.split("_", 1)[-1]
        if key == short or preset.name.endswith("_" + key):
            return preset
    known = ", ".join(p.name for p in presets)
    raise KeyError(f"Unknown preset '{name}'. Known: {known}")


def preset_to_dict(preset: Preset) -> dict:
    return {
        "name": preset.name,
        "category": preset.category,
        "bottom_ratio": preset.bottom_ratio,
        "detect": asdict(preset.detect),
    }


def detect_from_dict(data: dict) -> DetectConfig:
    valid = {f.name for f in fields(DetectConfig)}
    cleaned = {k: v for k, v in data.items() if k in valid}
    return DetectConfig(**cleaned)
