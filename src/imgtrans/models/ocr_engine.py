"""OCR engines (the document front-end): page image -> words + boxes + confidence.

* ``TesseractEngine`` (default real engine) - per-word boxes, confidence and the
  block/paragraph/line hierarchy via ``image_to_data``.
* ``EasyOcrEngine`` - optional alternative (handles many scripts).
* ``SeedEngine`` - the OFFLINE deterministic engine: reads the gold layout spec that
  ``data/synth_render`` embeds in a synthetic image's PNG metadata (``imgtrans_spec``)
  and reconstructs words+boxes, optionally injecting a controllable char-noise. This
  lets the agent, eval and tests run with NO tesseract binary while still exercising
  the full OCR -> layout -> MT -> render path on realistic boxes.
* ``StubEngine`` - last-resort empty result (a blank image with no spec).

All heavy imports are lazy; ``load_ocr_engine`` never raises.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..config import OcrConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Word:
    text: str
    conf: float                 # 0..1
    bbox: tuple                 # (x, y, w, h)
    block: int = 0
    par: int = 0
    line: int = 0


@dataclass
class OcrResult:
    words: List[Word] = field(default_factory=list)
    engine: str = "stub"
    width: int = 0
    height: int = 0

    @property
    def mean_conf(self) -> float:
        cs = [w.conf for w in self.words if w.text.strip()]
        return round(sum(cs) / len(cs), 4) if cs else 0.0

    def lines_text(self) -> List[str]:
        groups: dict = {}
        for w in self.words:
            if w.text.strip():
                groups.setdefault((w.block, w.par, w.line), []).append(w)
        return [" ".join(t.text for t in v) for _, v in sorted(groups.items())]

    @property
    def full_text(self) -> str:
        return "\n".join(self.lines_text())


def _read_spec(image) -> Optional[dict]:
    """Pull the gold layout spec embedded by the synthetic generator, if present."""
    info = getattr(image, "info", None) or {}
    raw = info.get("imgtrans_spec")
    if raw is None and isinstance(getattr(image, "text", None), dict):
        raw = image.text.get("imgtrans_spec")
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None


def _noisify(text: str, rate: float, rng) -> str:
    if rate <= 0:
        return text
    chars = list(text)
    confuse = {"m": "rn", "rn": "m", "0": "O", "O": "0", "1": "l", "l": "1",
               "I": "l", "5": "S", "S": "5", "8": "B", "B": "8"}
    out: List[str] = []
    for c in chars:
        if rng.random() < rate:
            r = rng.random()
            if r < 0.4 and c in confuse:
                out.append(confuse[c])
            elif r < 0.7:
                continue  # deletion
            else:
                out.append(c + c)  # duplication
        else:
            out.append(c)
    return "".join(out)


class StubEngine:
    name = "stub"

    def __init__(self, cfg: Optional[OcrConfig] = None):
        self.cfg = cfg

    def recognize(self, image) -> OcrResult:
        w, h = (image.size if hasattr(image, "size") else (0, 0))
        return OcrResult(words=[], engine="stub", width=w, height=h)


class SeedEngine:
    """Offline engine that reconstructs words from a synthetic image's gold spec."""

    name = "seed"

    def __init__(self, cfg: Optional[OcrConfig] = None, noise: float = 0.0, seed: int = 0):
        self.cfg = cfg
        self.noise = noise
        self.seed = seed

    def recognize(self, image) -> OcrResult:
        import random
        spec = _read_spec(image)
        w, h = (image.size if hasattr(image, "size") else (0, 0))
        if not spec:
            return OcrResult(words=[], engine="seed", width=w, height=h)
        rng = random.Random(self.seed)
        words: List[Word] = []
        for bi, blk in enumerate(spec.get("blocks", [])):
            text = blk.get("text", "")
            bx, by, bw, bh = blk.get("bbox", [0, 0, 0, 0])
            toks = text.split()
            if not toks:
                continue
            total = sum(len(t) for t in toks) + max(1, len(toks) - 1)
            cursor = bx
            for ti, tok in enumerate(toks):
                frac = (len(tok) + 1) / total
                ww = max(1, int(bw * frac))
                noisy = _noisify(tok, self.noise, rng)
                if noisy.strip():
                    words.append(Word(text=noisy, conf=round(0.97 - self.noise, 4),
                                      bbox=(int(cursor), int(by), int(ww), int(bh)),
                                      block=bi, par=0, line=blk.get("line", 0)))
                cursor += ww
        return OcrResult(words=words, engine="seed",
                         width=int(spec.get("width", w)), height=int(spec.get("height", h)))


class TesseractEngine:
    name = "tesseract"

    def __init__(self, cfg: OcrConfig):
        import pytesseract  # lazy; raises if unavailable
        self._pt = pytesseract
        self.cfg = cfg
        self._pt.get_tesseract_version()

    def recognize(self, image) -> OcrResult:
        from PIL import Image
        img = image if hasattr(image, "size") else Image.fromarray(image)
        cfg_str = f"--psm {self.cfg.psm}"
        data = self._pt.image_to_data(img, lang=self.cfg.lang, config=cfg_str,
                                      output_type=self._pt.Output.DICT)
        words: List[Word] = []
        for i in range(len(data["text"])):
            txt = data["text"][i]
            conf = float(data["conf"][i])
            if not txt.strip() or conf < 0:
                continue
            words.append(Word(text=txt, conf=conf / 100.0,
                              bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                              block=data["block_num"][i], par=data["par_num"][i], line=data["line_num"][i]))
        return OcrResult(words=words, engine="tesseract", width=img.width, height=img.height)


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, cfg: OcrConfig):
        import easyocr  # lazy
        self._reader = easyocr.Reader([cfg.lang[:2] if cfg.lang != "eng" else "en"], gpu=False)
        self.cfg = cfg

    def recognize(self, image) -> OcrResult:
        import numpy as np
        arr = np.asarray(image)
        res = self._reader.readtext(arr)
        words: List[Word] = []
        for li, (box, text, conf) in enumerate(res):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            words.append(Word(text=text, conf=float(conf),
                              bbox=(x, y, int(max(xs) - x), int(max(ys) - y)), block=li, par=0, line=0))
        h, w = arr.shape[:2]
        return OcrResult(words=words, engine="easyocr", width=w, height=h)


_ENGINES = {"tesseract": TesseractEngine, "easyocr": EasyOcrEngine,
            "seed": SeedEngine, "stub": StubEngine}


def has_spec(image) -> bool:
    return _read_spec(image) is not None


def load_ocr_engine(cfg: OcrConfig, engine: Optional[str] = None, image: Any = None):
    """Pick an OCR engine. If the image carries a gold spec (synthetic) and no real
    engine is forced, prefer the deterministic ``SeedEngine`` so offline runs work."""
    requested = engine or cfg.engine
    if requested == "stub":
        return StubEngine(cfg)
    if requested == "seed":
        return SeedEngine(cfg)
    if requested == "auto" and image is not None and has_spec(image):
        return SeedEngine(cfg)
    order: List[str] = []
    if requested and requested != "auto":
        order.append(requested)
    for e in ("tesseract", "easyocr"):
        if e not in order:
            order.append(e)
    for name in order:
        cls = _ENGINES.get(name)
        if cls is None or name in ("stub", "seed"):
            continue
        try:
            inst = cls(cfg)
            logger.info("OCR engine: %s", inst.name)
            return inst
        except Exception as exc:
            logger.info("OCR engine %s unavailable (%s); trying next", name, exc)
    if image is not None and has_spec(image):
        return SeedEngine(cfg)
    logger.info("No OCR engine available; using stub (no recognition).")
    return StubEngine(cfg)


__all__ = ["Word", "OcrResult", "TesseractEngine", "EasyOcrEngine", "SeedEngine", "StubEngine",
           "load_ocr_engine", "has_spec"]
