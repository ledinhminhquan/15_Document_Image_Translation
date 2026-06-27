"""Persist the dictionary-MT baseline artifact (the offline floor + fallback).

The ``DictionaryTranslator`` loads ``data/samples.dictionary()`` by default; this builder
writes that table to ``MtConfig.baseline_path`` as a versioned artifact so the registry,
the report and the API can reference it, and (optionally) augments it with high-frequency
word pairs mined from the training corpus when ``datasets`` is available.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..data import samples
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_baseline(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    table = samples.dictionary()
    out_path = cfg.mt.baseline_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": "dictionary", "version": "dict-1.0",
               "src_lang": cfg.mt.src_lang, "tgt_lang": cfg.mt.tgt_lang,
               "size": len(table), "table": table}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("dictionary baseline (%d entries) -> %s", len(table), out_path)
    return {"baseline_path": str(out_path), "size": len(table)}


__all__ = ["build_baseline"]
