"""Prefetch + sanity-check the datasets (no large files committed).

A streaming probe confirms the OPUS-100 fine-tune corpus is reachable and reports its
schema WITHOUT downloading it in full (OPUS-100 en-fr is 1M pairs). Degrades gracefully:
the built-in seed always works. Also (optionally) renders the synthetic eval pages.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _probe(loader) -> Dict[str, Any]:
    try:
        return {"ok": True, **loader()}
    except Exception as exc:  # pragma: no cover - network dependent
        return {"ok": False, "error": str(exc)}


def download_all(cfg: AppConfig, render_synthetic: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mt": {}, "ocr_text": {}, "seed": {}, "synthetic": {}}
    dc = cfg.data

    def mt_probe():
        from datasets import load_dataset
        ds = load_dataset(dc.mt_dataset, dc.mt_config, split="train", streaming=True)
        first = next(iter(ds))
        return {"dataset": dc.mt_dataset, "config": dc.mt_config, "reachable": True,
                "columns": list(first.keys())}

    def ocr_text_probe():
        from datasets import load_dataset
        ds = load_dataset(dc.ocr_text_dataset, dc.ocr_text_config, split="train", streaming=True)
        first = next(iter(ds))
        return {"dataset": dc.ocr_text_dataset, "config": dc.ocr_text_config, "reachable": True,
                "columns": list(first.keys())}

    out["mt"] = _probe(mt_probe)
    out["ocr_text"] = _probe(ocr_text_probe)

    from . import samples
    out["seed"] = {"ok": True, "pairs": len(samples.pairs()), "pages": len(samples.seed_pages()),
                   "dict_size": len(samples.dictionary())}

    if render_synthetic:
        try:
            from .dataset import build_synthetic_eval
            out["synthetic"] = {"ok": True, **build_synthetic_eval(cfg)}
        except Exception as exc:
            out["synthetic"] = {"ok": False, "error": str(exc)}

    logger.info("download_all: mt=%s ocr_text=%s seed=%d pairs / %d pages",
                out["mt"].get("ok"), out["ocr_text"].get("ok"),
                out["seed"]["pairs"], out["seed"]["pages"])
    return out


__all__ = ["download_all"]
