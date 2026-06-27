"""Machine translation - the TRAINED NLP core + a dependency-light baseline.

Reuses the P13/P14 translator design:
* ``DictionaryTranslator`` - a word-lookup baseline (no torch): the offline floor +
  the fallback that lets the whole pipeline run with no GPU/network.
* ``TransformerTranslator`` - wraps a fine-tuned/pretrained seq2seq model
  (``facebook/m2m100_418M`` default; NLLB / Marian supported), handling per-family
  source/target language-code conventions.

Both expose ``translate(text)`` / ``translate_batch(texts)`` plus ``name`` / ``version``.
``load_translator`` picks the best available (fine-tuned > pretrained base > dictionary).

For the agent's D4 back-translation verify gate, ``TransformerTranslator.reversed()``
returns a translator that shares the same model/tokenizer but swaps src<->tgt (m2m100 is
many-to-many, so the reverse direction is free). The dictionary baseline has no reverse
(``reversed()`` returns ``None``) so D4 is skipped offline - reported honestly.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from ..config import MtConfig
from ..logging_utils import get_logger
from ..models.model_registry import resolve_latest

logger = get_logger(__name__)

_NLLB_CODES = {"en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
               "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "vi": "vie_Latn"}

_WORD = re.compile(r"[A-Za-zÀ-ÿ']+|\d+|[^\w\s]", re.IGNORECASE)


def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


class DictionaryTranslator:
    """Word-lookup baseline + offline fallback. ``reverse`` flips the lookup table."""

    name = "dictionary"
    version = "dict-1.0"

    def __init__(self, table: Optional[dict] = None, reverse: bool = False):
        if table is None:
            from ..data import samples
            table = samples.dictionary()
        if reverse:
            table = {v: k for k, v in table.items()}
        self.table = {k.lower(): v for k, v in table.items()}

    def translate(self, text: str) -> str:
        out: List[str] = []
        for tok in _WORD.findall(text or ""):
            if re.match(r"[^\w\s]", tok):
                if out:
                    out[-1] = out[-1] + tok
                else:
                    out.append(tok)
                continue
            low = _strip_accents(tok.lower())
            mapped = self.table.get(low, self.table.get(tok.lower(), tok))
            if mapped:
                out.append(mapped)
        s = " ".join(out)
        s = re.sub(r"\s+([.,!?;:])", r"\1", s).strip()
        return s

    def translate_batch(self, texts: List[str]) -> List[str]:
        return [self.translate(t) for t in texts]

    def reversed(self):
        return None  # dictionary back-translation is unreliable; D4 skipped offline


class TransformerTranslator:
    name = "transformer"

    def __init__(self, model, tok, cfg: MtConfig, version: str = "mt-1.0"):
        self.model = model
        self.tok = tok
        self.cfg = cfg
        self.version = version
        self._family = self._detect_family()
        self._device = self._to_device()

    def _detect_family(self) -> str:
        cls = type(self.tok).__name__.lower()
        if "m2m100" in cls or hasattr(self.tok, "get_lang_id"):
            return "m2m100"
        if "nllb" in cls or hasattr(self.tok, "lang_code_to_id"):
            return "nllb"
        return "generic"

    def _to_device(self):
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(dev)
            return dev
        except Exception:
            return "cpu"

    @classmethod
    def from_pretrained(cls, model_path: str, cfg: MtConfig) -> "TransformerTranslator":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()
        return cls(model, tok, cfg, version=_read_version(model_path))

    def _forced_bos(self):
        if self._family == "m2m100":
            return self.tok.get_lang_id(self.cfg.tgt_lang)
        if self._family == "nllb":
            code = _NLLB_CODES.get(self.cfg.tgt_lang, self.cfg.tgt_lang)
            try:
                return self.tok.convert_tokens_to_ids(code)
            except Exception:
                return None
        return None

    def _set_src(self):
        if self._family == "m2m100":
            self.tok.src_lang = self.cfg.src_lang
        elif self._family == "nllb":
            self.tok.src_lang = _NLLB_CODES.get(self.cfg.src_lang, self.cfg.src_lang)

    def translate_batch(self, texts: List[str]) -> List[str]:
        import torch
        if not texts:
            return []
        self._set_src()
        enc = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True,
                       max_length=self.cfg.max_source_length).to(self._device)
        kwargs = {"num_beams": self.cfg.num_beams, "max_length": self.cfg.max_target_length}
        fb = self._forced_bos()
        if fb is not None:
            kwargs["forced_bos_token_id"] = fb
        with torch.no_grad():
            gen = self.model.generate(**enc, **kwargs)
        return [s.strip() for s in self.tok.batch_decode(gen, skip_special_tokens=True)]

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0] if text else ""

    def reversed(self) -> "TransformerTranslator":
        """A view that translates tgt -> src (shares weights; m2m100 is many-to-many)."""
        rcfg = replace(self.cfg, src_lang=self.cfg.tgt_lang, tgt_lang=self.cfg.src_lang)
        view = TransformerTranslator.__new__(TransformerTranslator)
        view.model = self.model
        view.tok = self.tok
        view.cfg = rcfg
        view.version = self.version + "-rev"
        view._family = self._family
        view._device = self._device
        return view


def _read_version(model_path: str) -> str:
    meta = Path(model_path) / "model_meta.json"
    if meta.exists():
        try:
            import json
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "mt-1.0")
        except Exception:
            pass
    return "mt-base"


def load_translator(cfg: MtConfig, *, prefer: str = "transformer"):
    """Fine-tuned transformer > pretrained base > dictionary baseline."""
    if prefer == "transformer":
        latest = resolve_latest(cfg.output_dir)
        if latest is not None:
            try:
                return TransformerTranslator.from_pretrained(str(latest), cfg)
            except Exception as exc:
                logger.info("fine-tuned MT unavailable (%s); trying pretrained base.", exc)
        try:
            return TransformerTranslator.from_pretrained(cfg.base_model, cfg)
        except Exception as exc:
            logger.info("pretrained MT unavailable (%s); using dictionary baseline.", exc)
    return DictionaryTranslator()


__all__ = ["DictionaryTranslator", "TransformerTranslator", "load_translator"]
