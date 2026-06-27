"""Image preprocessing for document OCR (deskew / denoise / binarize + quality).

Uses OpenCV when available and falls back to pure numpy/Pillow so the package
works with only core deps. Quality metrics drive the agent's D1 routing.
(Ported from P07 dococr.)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..config import PreprocessConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _np():
    import numpy as np
    return np


def _cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def to_gray(img):
    np = _np()
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return arr.astype(np.uint8)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def otsu_threshold(gray) -> int:
    np = _np()
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = w_b = 0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def binarize(gray, method: str = "adaptive"):
    np = _np()
    if method == "none":
        return gray
    cv2 = _cv2()
    if method == "adaptive" and cv2 is not None:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    t = otsu_threshold(gray)
    return np.where(gray > t, 255, 0).astype(np.uint8)


def denoise(gray):
    np = _np()
    cv2 = _cv2()
    if cv2 is not None:
        return cv2.medianBlur(gray, 3)
    from PIL import Image, ImageFilter
    return np.asarray(Image.fromarray(gray).filter(ImageFilter.MedianFilter(3)))


def estimate_skew(gray, max_deg: float = 15.0) -> float:
    np = _np()
    from PIL import Image
    binary = (binarize(gray, "otsu") < 128).astype(np.uint8) * 255
    if binary.mean() < 1:
        return 0.0
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_deg, max_deg + 0.1, 1.0):
        rot = np.asarray(Image.fromarray(binary).rotate(angle, resample=Image.NEAREST, fillcolor=0))
        proj = (rot > 128).sum(axis=1).astype(np.float32)
        score = float(np.var(proj))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def deskew(gray, max_deg: float = 15.0):
    np = _np()
    from PIL import Image
    angle = estimate_skew(gray, max_deg)
    if abs(angle) < 0.5:
        return gray
    return np.asarray(Image.fromarray(gray).rotate(angle, resample=Image.BILINEAR, fillcolor=255), dtype=np.uint8)


def quality_metrics(gray) -> Dict[str, float]:
    np = _np()
    g = gray.astype(np.float32)
    lap = (np.abs(np.diff(g, n=2, axis=0)).mean() + np.abs(np.diff(g, n=2, axis=1)).mean()) / 2.0
    ink_ratio = float((binarize(gray, "otsu") < 128).mean())
    contrast = float(g.std() / 64.0)
    blur = float(min(1.0, lap / 12.0))
    score = max(0.0, min(1.0, 0.5 * blur + 0.3 * min(1.0, contrast) + 0.2 * min(1.0, ink_ratio * 8)))
    return {"blur": round(blur, 4), "ink_ratio": round(ink_ratio, 4),
            "contrast": round(contrast, 4), "quality": round(score, 4)}


def preprocess_image(img, cfg: PreprocessConfig) -> Tuple[Any, Dict[str, float]]:
    from PIL import Image
    gray = to_gray(img)
    if cfg.denoise:
        gray = denoise(gray)
    if cfg.deskew:
        gray = deskew(gray, cfg.max_skew_deg)
    metrics = quality_metrics(gray)
    return Image.fromarray(gray).convert("RGB"), metrics


__all__ = ["to_gray", "binarize", "denoise", "deskew", "quality_metrics",
           "preprocess_image", "otsu_threshold", "estimate_skew"]
