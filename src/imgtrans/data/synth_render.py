"""Synthetic document-image generator (the PRIMARY data source for P15).

No public in-image / document-image translation benchmark with gold parallel text
exists, so we synthesize one: render source sentences (from the MT corpus or the
seed pairs) onto clean page images with varied fonts / sizes / light degradation,
and embed the gold layout spec (source text, gold translation, bounding boxes) into
the PNG metadata. That gives reproducible ``(image, gold_source_text,
gold_target_translation, boxes)`` quadruples, which let us measure OCR CER/WER, MT
chrF/BLEU and the end-to-end image-translation chrF, and exercise the overlay
renderer + layout-fidelity metric.

The embedded spec is read back by ``models.ocr_engine.SeedEngine`` so the entire
pipeline runs offline (no tesseract). On Colab/H100 the real Tesseract engine reads
the same images for an honest CER/WER number.

PIL is imported lazily; generation requires Pillow, but importing this module does not.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import DataConfig
from ..imaging.render import discover_font
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _measure(draw, text: str, font) -> Tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def render_page(spec: Dict, *, font_path: Optional[str] = None, font_size: int = 28,
                margin: int = 60, line_gap: int = 22, degrade: float = 0.0, seed: int = 0):
    """Render a page spec -> PIL.Image (RGB) with the gold spec embedded in PNG info.

    ``spec`` may already carry boxes (seed pages) - they are recomputed here from the
    actual rendered geometry so the embedded boxes match the pixels exactly.
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    rng = random.Random(seed)

    fp = font_path or discover_font_safe()
    try:
        font = ImageFont.truetype(fp, font_size)
    except Exception:
        font = ImageFont.load_default()

    width = int(spec.get("width", 1000))
    blocks_in = spec.get("blocks", [])
    # first pass on a scratch image to measure heights
    scratch = Image.new("RGB", (width, 10), "white")
    sdraw = ImageDraw.Draw(scratch)

    placed: List[Dict] = []
    y = margin
    line_h = _measure(sdraw, "Ahg", font)[1]
    for i, blk in enumerate(blocks_in):
        text = blk.get("text", "")
        tw, th = _measure(sdraw, text, font)
        bbox = [margin, y, min(width - 2 * margin, tw + 8), max(line_h, th) + 6]
        placed.append({"text": text, "translation": blk.get("translation", ""),
                       "bbox": bbox, "block": i, "line": i,
                       "kind": blk.get("kind", "heading" if i == 0 else "paragraph")})
        y += max(line_h, th) + line_gap
    height = y + margin

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for blk in placed:
        x, by = blk["bbox"][0], blk["bbox"][1]
        draw.text((x, by), blk["text"], fill="black", font=font)

    if degrade > 0:
        img = _degrade(img, degrade, rng)

    out_spec = {"width": width, "height": height,
                "src_lang": spec.get("src_lang", "en"), "tgt_lang": spec.get("tgt_lang", "fr"),
                "blocks": placed}
    # stash for in-memory consumers (SeedEngine reads image.info)
    img.info["imgtrans_spec"] = json.dumps(out_spec, ensure_ascii=False)
    return img, out_spec


def _degrade(img, amount: float, rng):
    """Light, realistic scan degradation: rotate a touch + gaussian blur + jpeg-ish."""
    from PIL import ImageFilter
    angle = (rng.random() - 0.5) * 2.0 * amount * 2.5     # +-(amount*2.5) degrees
    if abs(angle) > 0.1:
        img = img.rotate(angle, resample=2, fillcolor="white", expand=False)
    radius = amount * 1.2
    if radius > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def discover_font_safe() -> str:
    from ..config import RenderConfig
    return discover_font(RenderConfig())


def save_png_with_spec(img, spec: Dict, path: str) -> str:
    """Save a PNG that PERSISTS the gold spec in a tEXt chunk (survives reload)."""
    from PIL.PngImagePlugin import PngInfo
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = PngInfo()
    meta.add_text("imgtrans_spec", json.dumps(spec, ensure_ascii=False))
    img.save(str(p), pnginfo=meta)
    return str(p)


def generate_dataset(pairs: List[Dict[str, str]], out_dir: str, *, n_pages: int = 80,
                     lines_per_page: int = 6, cfg: Optional[DataConfig] = None,
                     degrade: float = 0.15, seed: int = 42) -> Dict:
    """Render ``n_pages`` synthetic document images from (src,tgt) pairs.

    Writes ``page_XXXX.png`` (with embedded spec) + a ``manifest.jsonl`` (one row per
    page) under ``out_dir``. Returns a small summary dict.
    """
    cfg = cfg or DataConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width = cfg.image_width
    manifest_path = out / "manifest.jsonl"
    n = min(n_pages, max(1, math.ceil(len(pairs) / lines_per_page))) if pairs else 0
    written = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for pi in range(n_pages):
            chunk = [pairs[(pi * lines_per_page + j) % len(pairs)] for j in range(lines_per_page)] if pairs else []
            if not chunk:
                break
            spec = {"width": width, "src_lang": cfg.src_lang, "tgt_lang": cfg.tgt_lang,
                    "blocks": [{"text": c["src"], "translation": c["tgt"]} for c in chunk]}
            fsz = 24 + (pi % 5) * 4          # vary font size 24..40
            deg = degrade * ((pi % 4) / 3.0)  # vary degradation
            img, full = render_page(spec, font_size=fsz, degrade=deg, seed=seed + pi)
            fname = f"page_{pi:04d}.png"
            save_png_with_spec(img, full, str(out / fname))
            mf.write(json.dumps({"image": fname, "blocks": full["blocks"],
                                 "src_lang": full["src_lang"], "tgt_lang": full["tgt_lang"]},
                                ensure_ascii=False) + "\n")
            written += 1
    logger.info("generated %d synthetic pages -> %s", written, out)
    return {"pages": written, "dir": str(out), "manifest": str(manifest_path)}


__all__ = ["render_page", "save_png_with_spec", "generate_dataset", "discover_font_safe"]
