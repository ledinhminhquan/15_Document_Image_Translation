"""Layout analysis & reading order: OCR words -> ordered translatable text blocks.

Ingests page images / PDFs (born-digital PDFs are read directly via PyMuPDF - no
OCR needed; scanned pages are rasterized for OCR), groups OCR words into blocks,
orders them (XY-cut, multi-column aware) and classifies them. Each ``TextBlock``
carries the source text + bbox + (later) the translation, so the renderer can put
the translated text back where the source text was. Heavy imports are lazy.
(Adapted from P07 dococr.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config import AppConfig, LayoutConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TextBlock:
    text: str
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    kind: str = "paragraph"              # heading | paragraph | list | header_footer | blank
    conf: float = 1.0
    reading_index: int = 0
    translation: str = ""                # filled by the MT stage
    status: str = "pending"             # pending | translated | skipped_lowconf | needs_review

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "bbox": list(self.bbox), "kind": self.kind,
                "conf": round(self.conf, 4), "reading_index": self.reading_index,
                "translation": self.translation, "status": self.status}


@dataclass
class PageInput:
    index: int
    born_digital: bool
    image: Any = None                    # PIL image for scanned pages
    digital_blocks: List[TextBlock] = field(default_factory=list)
    width: int = 0
    height: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion (image / PDF; born-digital detection)
# ─────────────────────────────────────────────────────────────────────────────
def ingest(path: str, cfg: AppConfig) -> List[PageInput]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return _ingest_pdf(p, cfg)
    from PIL import Image
    img = Image.open(p)
    img = img.convert("RGB") if img.mode != "RGB" else img
    return [PageInput(index=0, born_digital=False, image=img, width=img.width, height=img.height)]


def ingest_image(img, cfg: AppConfig) -> List[PageInput]:
    """Wrap an in-memory PIL image (used by the API + synthetic pipeline)."""
    return [PageInput(index=0, born_digital=False, image=img,
                      width=getattr(img, "width", 0), height=getattr(img, "height", 0))]


def _ingest_pdf(path: Path, cfg: AppConfig) -> List[PageInput]:
    import fitz  # PyMuPDF, lazy
    doc = fitz.open(str(path))
    pages: List[PageInput] = []
    for i, page in enumerate(doc):
        if i >= cfg.serving.max_pages:
            break
        text = page.get_text().strip()
        if len(text) >= cfg.layout.born_digital_min_chars:
            blocks = []
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
                if btext.strip():
                    blocks.append(TextBlock(text=btext.strip(),
                                            bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0))))
            pages.append(PageInput(index=i, born_digital=True, digital_blocks=blocks,
                                   width=int(page.rect.width), height=int(page.rect.height)))
        else:
            pix = page.get_pixmap(dpi=cfg.ocr.dpi)
            from PIL import Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(PageInput(index=i, born_digital=False, image=img, width=pix.width, height=pix.height))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Words -> blocks -> reading order -> classification
# ─────────────────────────────────────────────────────────────────────────────
def words_to_blocks(ocr_result, cfg: LayoutConfig) -> List[TextBlock]:
    groups: Dict[int, list] = {}
    for w in ocr_result.words:
        if w.text.strip():
            groups.setdefault(w.block, []).append(w)
    blocks: List[TextBlock] = []
    for _, ws in groups.items():
        xs = [w.bbox[0] for w in ws]
        ys = [w.bbox[1] for w in ws]
        x2 = [w.bbox[0] + w.bbox[2] for w in ws]
        y2 = [w.bbox[1] + w.bbox[3] for w in ws]
        x, y = min(xs), min(ys)
        bbox = (x, y, max(x2) - x, max(y2) - y)
        if bbox[2] * bbox[3] < cfg.min_region_area:
            continue
        line_groups: Dict[int, list] = {}
        for w in ws:
            line_groups.setdefault(w.line, []).append(w)
        text = " ".join(" ".join(t.text for t in line_groups[k]) for k in sorted(line_groups))
        conf = sum(w.conf for w in ws) / len(ws)
        blocks.append(TextBlock(text=text, bbox=bbox, conf=conf))
    return blocks


def reading_order(blocks: List[TextBlock], page_w: int, cfg: LayoutConfig) -> List[TextBlock]:
    if not blocks:
        return blocks
    if cfg.reading_order == "xycut" and page_w > 0:
        mid = page_w / 2
        centers = [(b.bbox[0] + b.bbox[2] / 2) for b in blocks]
        left = [c for c in centers if c < mid * 0.85]
        right = [c for c in centers if c > mid * 1.15]
        if left and right and len(blocks) >= 4:
            def key(b):
                cx = b.bbox[0] + b.bbox[2] / 2
                return (0 if cx < mid else 1, b.bbox[1], b.bbox[0])
            ordered = sorted(blocks, key=key)
        else:
            ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    else:
        ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    for idx, b in enumerate(ordered):
        b.reading_index = idx
    return ordered


_LIST_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


def classify_block(b: TextBlock, page_w: int, page_h: int) -> str:
    t = b.text.strip()
    if not t:
        return "blank"
    first = t.split("\n", 1)[0]
    y = b.bbox[1]
    if page_h and (y < page_h * 0.06 or y > page_h * 0.94) and len(t) <= 80:
        return "header_footer"
    if _LIST_RE.match(first):
        return "list"
    if len(t) <= 70 and "\n" not in t and not first.endswith((".", ",", ";")) and (first.isupper() or first.istitle()):
        return "heading"
    return "paragraph"


def classify_blocks(blocks: List[TextBlock], page_w: int, page_h: int, cfg: LayoutConfig) -> List[TextBlock]:
    if cfg.classify_blocks:
        for b in blocks:
            b.kind = classify_block(b, page_w, page_h)
    return blocks


def blocks_from_page(page: PageInput, ocr_engine, cfg: AppConfig) -> List[TextBlock]:
    """End-to-end: born-digital -> use embedded blocks; scanned -> OCR -> blocks."""
    if page.born_digital and page.digital_blocks:
        blocks = page.digital_blocks
    else:
        ocr = ocr_engine.recognize(page.image)
        page.width = page.width or ocr.width
        page.height = page.height or ocr.height
        blocks = words_to_blocks(ocr, cfg.layout)
    blocks = reading_order(blocks, page.width, cfg.layout)
    blocks = classify_blocks(blocks, page.width, page.height, cfg.layout)
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Structured output (source + translation)
# ─────────────────────────────────────────────────────────────────────────────
def _md(blocks: List[TextBlock], field_name: str) -> str:
    out: List[str] = []
    for b in blocks:
        t = (getattr(b, field_name) or "").strip()
        if not t or b.kind == "blank":
            continue
        if b.kind == "heading":
            out.append(f"## {t}")
        elif b.kind == "list":
            out.append("\n".join(f"- {re.sub(_LIST_RE, '', ln).strip()}" for ln in t.split("\n") if ln.strip()))
        elif b.kind == "header_footer":
            out.append(f"<sub>{t}</sub>")
        else:
            out.append(t.replace("\n", " "))
    return "\n\n".join(out)


def assemble(blocks: List[TextBlock]) -> Dict[str, Any]:
    src = "\n\n".join(b.text.strip() for b in blocks if b.text.strip() and b.kind != "blank")
    tgt = "\n\n".join((b.translation or "").strip() for b in blocks
                      if (b.translation or "").strip() and b.kind != "blank")
    return {"source_text": src, "translated_text": tgt,
            "source_markdown": _md(blocks, "text"), "translated_markdown": _md(blocks, "translation"),
            "blocks": [b.to_dict() for b in blocks if b.kind != "blank"]}


__all__ = ["TextBlock", "PageInput", "ingest", "ingest_image", "words_to_blocks", "reading_order",
           "classify_block", "classify_blocks", "blocks_from_page", "assemble"]
