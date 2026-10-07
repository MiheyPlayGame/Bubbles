"""Bubble search algorithms (presets). Image enhance is applied separately."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float
    circularity: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class DetectConfig:
    """
    Search methods:
      Contours: otsu, adaptive, blackhat, tophat, dog, laplacian, sobel_mag
      Blobs:    blob, hough, msers
      Motion:   bg_absdiff, frame_diff
    """

    method: str = "otsu"
    invert: bool = False
    morph_kernel: int = 3
    min_area: float = 25.0
    max_area: float = 50_000.0
    min_circularity: float = 0.25

    adaptive_block: int = 31
    adaptive_c: int = 5
    hat_ksize: int = 17
    dog_sigma1: float = 1.0
    dog_sigma2: float = 3.0
    edge_blur: int = 3

    hough_dp: float = 1.2
    hough_min_dist: float = 10.0
    hough_param1: float = 80.0
    hough_param2: float = 12.0
    hough_min_radius: int = 3
    hough_max_radius: int = 40

    mser_delta: int = 5
    mser_min_area: int = 20
    mser_max_area: int = 8000


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _morph_cleanup(binary: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    mk = _odd(cfg.morph_kernel)
    if mk <= 1:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def _otsu(work: np.ndarray, invert: bool) -> np.ndarray:
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(work, 0, 255, flag + cv2.THRESH_OTSU)
    return binary


def _filter_contour(contour: np.ndarray, cfg: DetectConfig) -> Detection | None:
    area = float(cv2.contourArea(contour))
    if area < cfg.min_area or area > cfg.max_area:
        return None
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return None
    circularity = float(4.0 * np.pi * area / (peri * peri))
    if circularity < cfg.min_circularity:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if w < 2 or h < 2:
        return None
    return Detection(x=x, y=y, w=w, h=h, area=area, circularity=circularity)


def find_bubbles_from_mask(binary: np.ndarray, cfg: DetectConfig) -> list[Detection]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Detection] = []
    for contour in contours:
        det = _filter_contour(contour, cfg)
        if det is not None:
            out.append(det)
    return out


def _mask_from_detections(shape: tuple[int, int], detections: list[Detection]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for det in detections:
        cv2.rectangle(mask, (det.x, det.y), (det.x + det.w, det.y + det.h), 255, -1)
    return mask


def _build_response_map(
    gray: np.ndarray,
    cfg: DetectConfig,
    background: np.ndarray | None,
    prev_gray: np.ndarray | None,
) -> np.ndarray:
    """Return a single-channel response image ready for thresholding / blobbing."""
    method = cfg.method
    work = gray

    if method == "bg_absdiff":
        if background is None:
            return gray
        bg = background
        if bg.shape[:2] != gray.shape[:2]:
            bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.absdiff(gray, bg)

    if method == "frame_diff":
        if prev_gray is None:
            return np.zeros_like(gray)
        prev = prev_gray
        if prev.shape[:2] != gray.shape[:2]:
            prev = cv2.resize(prev, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.absdiff(gray, prev)

    if method == "blackhat":
        hk = _odd(cfg.hat_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hk, hk))
        return cv2.morphologyEx(work, cv2.MORPH_BLACKHAT, kernel)

    if method == "tophat":
        hk = _odd(cfg.hat_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hk, hk))
        return cv2.morphologyEx(work, cv2.MORPH_TOPHAT, kernel)

    if method == "dog":
        g1 = cv2.GaussianBlur(work, (0, 0), cfg.dog_sigma1)
        g2 = cv2.GaussianBlur(work, (0, 0), cfg.dog_sigma2)
        dog = cv2.absdiff(g1, g2)
        return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if method == "laplacian":
        blur = cv2.GaussianBlur(work, (_odd(cfg.edge_blur), _odd(cfg.edge_blur)), 0)
        lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
        return cv2.convertScaleAbs(lap)

    if method == "sobel_mag":
        blur = cv2.GaussianBlur(work, (_odd(cfg.edge_blur), _odd(cfg.edge_blur)), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return work


def _threshold_response(response: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    method = cfg.method
    # Dark bubbles on bright field often need INV on raw/otsu/adaptive.
    # For residual maps (bg/frame/dog/edges/hats) bright peaks → BINARY.
    residual_methods = {
        "bg_absdiff",
        "frame_diff",
        "dog",
        "laplacian",
        "sobel_mag",
        "blackhat",
        "tophat",
    }
    invert = cfg.invert
    if method in residual_methods and not cfg.invert:
        invert = False
    elif method in {"otsu", "adaptive"} and not cfg.invert:
        # Prefer dark objects as white for typical bubble footage.
        invert = True

    if method == "adaptive":
        block = _odd(cfg.adaptive_block)
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            response,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            flag,
            block,
            cfg.adaptive_c,
        )
    else:
        binary = _otsu(response, invert=invert)
    return _morph_cleanup(binary, cfg)


def _detect_blob(work: np.ndarray, cfg: DetectConfig) -> list[Detection]:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(cfg.min_area)
    params.maxArea = float(min(cfg.max_area, 10000))
    params.filterByCircularity = True
    params.minCircularity = float(cfg.min_circularity)
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False
    detector = cv2.SimpleBlobDetector_create(params)
    image = cv2.bitwise_not(work) if cfg.invert else work
    # Default: dark blobs → invert so detector sees dark-on-light as expected? 
    # SimpleBlobDetector by default finds dark blobs. Dark bubbles on light: use work as-is.
    if not cfg.invert:
        image = work
    else:
        image = cv2.bitwise_not(work)
    keypoints = detector.detect(image)
    detections: list[Detection] = []
    for kp in keypoints:
        r = max(2.0, float(kp.size) / 2.0)
        x = int(round(kp.pt[0] - r))
        y = int(round(kp.pt[1] - r))
        s = int(round(2 * r))
        detections.append(
            Detection(x=max(0, x), y=max(0, y), w=s, h=s, area=float(np.pi * r * r), circularity=1.0)
        )
    return detections


def _detect_hough(work: np.ndarray, cfg: DetectConfig) -> list[Detection]:
    circles = cv2.HoughCircles(
        work,
        cv2.HOUGH_GRADIENT,
        dp=float(cfg.hough_dp),
        minDist=float(cfg.hough_min_dist),
        param1=float(cfg.hough_param1),
        param2=float(cfg.hough_param2),
        minRadius=int(cfg.hough_min_radius),
        maxRadius=int(cfg.hough_max_radius),
    )
    detections: list[Detection] = []
    if circles is None:
        return detections
    for x, y, r in np.round(circles[0]).astype(int):
        area = float(np.pi * r * r)
        if area < cfg.min_area or area > cfg.max_area:
            continue
        detections.append(
            Detection(
                x=max(0, x - r),
                y=max(0, y - r),
                w=2 * r,
                h=2 * r,
                area=area,
                circularity=1.0,
            )
        )
    return detections


def _detect_msers(work: np.ndarray, cfg: DetectConfig) -> list[Detection]:
    mser = cv2.MSER_create()
    mser.setDelta(int(cfg.mser_delta))
    mser.setMinArea(int(cfg.mser_min_area))
    mser.setMaxArea(int(cfg.mser_max_area))
    regions, _ = mser.detectRegions(work)
    detections: list[Detection] = []
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts)
        area = float(w * h)
        if area < cfg.min_area or area > cfg.max_area:
            continue
        # Circularity approx from filled contour.
        contour = pts.reshape(-1, 1, 2)
        c_area = float(cv2.contourArea(contour))
        peri = cv2.arcLength(contour, True)
        circ = float(4.0 * np.pi * c_area / (peri * peri)) if peri > 1e-6 else 0.0
        if circ < cfg.min_circularity:
            continue
        detections.append(Detection(x=x, y=y, w=w, h=h, area=c_area or area, circularity=circ))
    return detections


def detect_bubbles(
    frame: np.ndarray,
    cfg: DetectConfig,
    *,
    background: np.ndarray | None = None,
    prev_frame: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Detection]]:
    """
    Run a search preset on an already-enhanced grayscale (or BGR) crop.
    Returns (debug_mask, detections).
    """
    gray = _to_gray(frame)
    prev = None if prev_frame is None else _to_gray(prev_frame)
    method = cfg.method

    if method == "blob":
        dets = _detect_blob(gray, cfg)
        return _mask_from_detections(gray.shape, dets), dets
    if method == "hough":
        dets = _detect_hough(gray, cfg)
        return _mask_from_detections(gray.shape, dets), dets
    if method == "msers":
        dets = _detect_msers(gray, cfg)
        return _mask_from_detections(gray.shape, dets), dets

    response = _build_response_map(gray, cfg, background, prev)
    binary = _threshold_response(response, cfg)
    return binary, find_bubbles_from_mask(binary, cfg)
