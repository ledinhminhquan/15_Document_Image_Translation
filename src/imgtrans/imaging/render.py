"""Layout-preserving overlay renderer - the headline value-add of P15.

Re-draws each translated block INTO the original bounding box, auto-shrinking the
font and word-wrapping so the (usually longer) translation fits. Pillow-only - no
OpenCV needed. When Pillow/fonts are unavailable (e.g. CI), a pure-python geometric
estimator computes the same fit decisions so layout-fidelity metrics still work and
tests run with no image stack.

Public API
----------
* ``discover_font(cfg)``                    -> a usable TTF path (Noto/DejaVu/Arial) or ""
* ``fit_text_to_box(text, box, font, cfg)`` -> FitResult(font_size, lines, fits, overflow)
* ``render_overlay(image, blocks, cfg)``    -> (PIL.Image, fidelity dict)
* ``render_side_by_side(image, blocks, cfg)``-> PIL.Image (source | translation panel)
* ``layout_fidelity(blocks, cfg)``          -> dict(fit_rate, mean_shrink, n_overflow, ...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..config import RenderConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

# candidate fonts that cover Latin (+ broad Unicode for fr/de/es/vi etc.)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "DejaVuSans.ttf",  # PIL ships this; truetype() resolves by name
]


@dataclass
class FitResult:
    font_size: int
    lines: List[str]
    fits: bool
    overflow: float          # >0 means the text exceeds the box (fraction of box height)
    method: str = "estimate"  # "pil" | "estimate"


def discover_font(cfg: RenderConfig) -> str:
    if cfg.font_path and os.path.exists(cfg.font_path):
        return cfg.font_path
    for cand in _FONT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    # last resort: let PIL try to resolve a bundled font by name
    return "DejaVuSans.ttf"


def _load_font(font_path: str, size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _wrap_greedy(text: str, measure, max_w: int) -> List[str]:
    """Greedy word-wrap using a ``measure(str)->width`` callable."""
    words = (text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if measure(trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    # hard-break any single token wider than the box
    out: List[str] = []
    for ln in lines:
        if measure(ln) <= max_w or " " in ln:
            out.append(ln)
            continue
        buf = ""
        for ch in ln:
            if measure(buf + ch) <= max_w or not buf:
                buf += ch
            else:
                out.append(buf)
                buf = ch
        if buf:
            out.append(buf)
    return out


def _fit_pil(text: str, box: Tuple[int, int, int, int], font_path: str, cfg: RenderConfig) -> FitResult:
    from PIL import Image, ImageDraw
    _img = Image.new("RGB", (4, 4))
    draw = ImageDraw.Draw(_img)
    bx, by, bw, bh = box
    inner_w = max(1, bw - 2 * cfg.box_padding)
    inner_h = max(1, bh - 2 * cfg.box_padding)

    def line_height(font):
        a = draw.textbbox((0, 0), "Ahg", font=font)
        return (a[3] - a[1]) * cfg.line_spacing

    lo, hi = cfg.min_font_size, cfg.max_font_size
    best: Optional[FitResult] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)

        def measure(s: str, _f=font):
            b = draw.textbbox((0, 0), s, font=_f)
            return b[2] - b[0]

        lines = _wrap_greedy(text, measure, inner_w)
        total_h = line_height(font) * len(lines)
        fits = total_h <= inner_h
        if fits:
            best = FitResult(mid, lines, True, 0.0, "pil")
            lo = mid + 1
        else:
            hi = mid - 1
    if best is not None:
        return best
    # nothing fit, even at min font: report overflow at min size
    font = _load_font(font_path, cfg.min_font_size)

    def measure_min(s: str, _f=font):
        b = draw.textbbox((0, 0), s, font=_f)
        return b[2] - b[0]

    lines = _wrap_greedy(text, measure_min, inner_w)
    total_h = line_height(font) * len(lines)
    overflow = round(max(0.0, (total_h - inner_h) / max(1, inner_h)), 4)
    return FitResult(cfg.min_font_size, lines, False, overflow, "pil")


def _fit_estimate(text: str, box: Tuple[int, int, int, int], cfg: RenderConfig) -> FitResult:
    """No-PIL geometric estimate: avg glyph width ~= 0.55*size, line height ~= 1.2*size."""
    bx, by, bw, bh = box
    inner_w = max(1, bw - 2 * cfg.box_padding)
    inner_h = max(1, bh - 2 * cfg.box_padding)
    text = text or ""

    def wrap_at(size: int) -> List[str]:
        char_w = max(1.0, 0.55 * size)
        max_chars = max(1, int(inner_w / char_w))
        return _wrap_greedy(text, lambda s: int(len(s) * char_w), inner_w) if max_chars > 1 \
            else [text[i:i + 1] for i in range(len(text))]

    lo, hi = cfg.min_font_size, cfg.max_font_size
    best: Optional[FitResult] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        lines = wrap_at(mid)
        total_h = len(lines) * mid * cfg.line_spacing
        if total_h <= inner_h:
            best = FitResult(mid, lines, True, 0.0, "estimate")
            lo = mid + 1
        else:
            hi = mid - 1
    if best is not None:
        return best
    lines = wrap_at(cfg.min_font_size)
    total_h = len(lines) * cfg.min_font_size * cfg.line_spacing
    overflow = round(max(0.0, (total_h - inner_h) / max(1, inner_h)), 4)
    return FitResult(cfg.min_font_size, lines, False, overflow, "estimate")


def fit_text_to_box(text: str, box: Tuple[int, int, int, int], cfg: RenderConfig,
                    font_path: Optional[str] = None) -> FitResult:
    try:
        from PIL import ImageDraw  # noqa: F401
        fp = font_path or discover_font(cfg)
        return _fit_pil(text, box, fp, cfg)
    except Exception as exc:
        logger.debug("PIL fit unavailable (%s); using geometric estimate.", exc)
        return _fit_estimate(text, box, cfg)


def layout_fidelity(blocks, cfg: RenderConfig) -> dict:
    """Quantify how well translations fit their source boxes (the overlay quality)."""
    translatable = [b for b in blocks if (getattr(b, "translation", "") or "").strip()
                    and getattr(b, "kind", "paragraph") != "blank"]
    if not translatable:
        return {"n_blocks": 0, "fit_rate": 1.0, "n_overflow": 0,
                "mean_shrink": 1.0, "mean_overflow": 0.0}
    font_path = discover_font(cfg)
    n_fit = 0
    shrinks: List[float] = []
    overflows: List[float] = []
    for b in translatable:
        src_fit = fit_text_to_box(b.text, b.bbox, cfg, font_path)
        tgt_fit = fit_text_to_box(b.translation, b.bbox, cfg, font_path)
        if tgt_fit.fits:
            n_fit += 1
        overflows.append(tgt_fit.overflow)
        base = max(1, src_fit.font_size)
        shrinks.append(tgt_fit.font_size / base)
    n = len(translatable)
    return {
        "n_blocks": n,
        "fit_rate": round(n_fit / n, 4),
        "n_overflow": n - n_fit,
        "mean_shrink": round(sum(shrinks) / n, 4),
        "mean_overflow": round(sum(overflows) / n, 4),
    }


def render_overlay(image, blocks, cfg: RenderConfig):
    """Whiteout each source box and draw the fitted translation. Returns (image, fidelity)."""
    from PIL import Image, ImageDraw
    img = image.convert("RGB").copy() if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_path = discover_font(cfg)
    n_fit = 0
    n = 0
    for b in blocks:
        tr = (getattr(b, "translation", "") or "").strip()
        if not tr or getattr(b, "kind", "paragraph") == "blank":
            continue
        n += 1
        bx, by, bw, bh = b.bbox
        draw.rectangle([bx, by, bx + bw, by + bh], fill=cfg.fill_color)
        if cfg.draw_box_border:
            draw.rectangle([bx, by, bx + bw, by + bh], outline="red")
        fit = fit_text_to_box(tr, b.bbox, cfg, font_path)
        if fit.fits:
            n_fit += 1
        font = _load_font(font_path, fit.font_size)
        y = by + cfg.box_padding
        lh = fit.font_size * cfg.line_spacing
        for ln in fit.lines:
            draw.text((bx + cfg.box_padding, y), ln, fill=cfg.text_color, font=font)
            y += lh
    fidelity = layout_fidelity(blocks, cfg)
    fidelity["rendered_blocks"] = n
    return img, fidelity


def render_side_by_side(image, blocks, cfg: RenderConfig):
    """Original on the left, a clean translation panel on the right."""
    from PIL import Image, ImageDraw
    src = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
    w, h = src.size
    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    font_path = discover_font(cfg)
    for b in blocks:
        tr = (getattr(b, "translation", "") or "").strip()
        if not tr or getattr(b, "kind", "paragraph") == "blank":
            continue
        fit = fit_text_to_box(tr, b.bbox, cfg, font_path)
        font = _load_font(font_path, fit.font_size)
        bx, by, _, _ = b.bbox
        y = by + cfg.box_padding
        lh = fit.font_size * cfg.line_spacing
        for ln in fit.lines:
            draw.text((bx + cfg.box_padding, y), ln, fill=cfg.text_color, font=font)
            y += lh
    out = Image.new("RGB", (w * 2 + 8, h), "white")
    out.paste(src, (0, 0))
    out.paste(panel, (w + 8, 0))
    return out


def save_image(image, path: str) -> str:
    p = str(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    image.save(p)
    return p


__all__ = ["FitResult", "discover_font", "fit_text_to_box", "layout_fidelity",
           "render_overlay", "render_side_by_side", "save_image"]
