"""Data loading: parallel MT pairs (fine-tune) + document-page specs (OCR/e2e eval).

* ``load_pairs`` - flat (src, tgt) sentence pairs from OPUS-100 en-fr (the MT fine-tune
  corpus); falls back to the built-in seed so everything runs offline.
* ``load_eval_pages`` - document-page specs for the OCR / end-to-end / layout-fidelity
  eval: a rendered ``manifest.jsonl`` if one was generated, else the synthetic seed pages.

``datasets`` is imported lazily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, data_dir
from ..logging_utils import get_logger
from . import samples

logger = get_logger(__name__)


@dataclass
class Pair:
    id: str
    src: str
    tgt: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "src": self.src, "tgt": self.tgt}


def load_pairs(cfg: AppConfig, split: str = "train", limit: Optional[int] = None) -> List[Pair]:
    """Flat sentence pairs from OPUS-100 en-fr; fall back to the seed."""
    dc = cfg.data
    cap = limit or (dc.max_train_samples if split == "train" else dc.max_eval_samples)
    if dc.use_hf:
        try:
            from datasets import load_dataset  # lazy
            ds = load_dataset(dc.mt_dataset, dc.mt_config, split=split, streaming=True)
            out: List[Pair] = []
            for i, r in enumerate(ds):
                if len(out) >= cap:
                    break
                tr = r.get("translation") or {}
                src = str(tr.get(dc.src_lang, "") or "").strip()
                tgt = str(tr.get(dc.tgt_lang, "") or "").strip()
                if src and tgt:
                    out.append(Pair(id=f"{split}{i:06d}", src=src[:1000], tgt=tgt[:1000]))
            if len(out) > 2:
                logger.info("Loaded %d %s pairs from %s", len(out), split, dc.mt_dataset)
                return out
        except Exception as exc:
            logger.warning("Could not load %s (%s); using seed.", dc.mt_dataset, exc)
    return load_seed_pairs()


def load_seed_pairs() -> List[Pair]:
    return [Pair(id=p["id"], src=p["src"], tgt=p["tgt"]) for p in samples.pairs()]


def seed_split(seed: int = 42, eval_frac: float = 0.3):
    import random
    items = load_seed_pairs()
    rng = random.Random(seed)
    rng.shuffle(items)
    n_eval = max(2, int(len(items) * eval_frac))
    return items[n_eval:], items[:n_eval]


def load_eval_pairs(cfg: AppConfig, limit: Optional[int] = None) -> List[Pair]:
    if cfg.data.use_hf:
        for split in ("validation", "test"):
            try:
                out = load_pairs(cfg, split=split, limit=limit or cfg.data.max_eval_samples)
                if len(out) > 2:
                    return out
            except Exception:
                continue
    _, ev = seed_split(cfg.data.seed)
    return ev


# ─────────────────────────────────────────────────────────────────────────────
# Document-page specs (for OCR / end-to-end / layout-fidelity eval)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_dir(cfg: AppConfig, split: str = "eval") -> Path:
    return data_dir() / "synthetic" / split


def load_manifest(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def load_eval_pages(cfg: AppConfig, split: str = "eval") -> List[Dict]:
    """Return page specs for evaluation. Prefer a generated synthetic manifest (which
    points at real PNGs); otherwise return the built-in synthetic seed pages."""
    manifest = synthetic_dir(cfg, split) / "manifest.jsonl"
    rows = load_manifest(manifest)
    if rows:
        base = synthetic_dir(cfg, split)
        pages = []
        for r in rows:
            pages.append({"image_path": str(base / r["image"]) if r.get("image") else None,
                          "width": 0, "height": 0,
                          "src_lang": r.get("src_lang", cfg.data.src_lang),
                          "tgt_lang": r.get("tgt_lang", cfg.data.tgt_lang),
                          "blocks": r.get("blocks", [])})
        logger.info("Loaded %d synthetic eval pages from %s", len(pages), manifest)
        return pages
    return samples.seed_pages()


def build_synthetic_eval(cfg: AppConfig, n_pages: Optional[int] = None, split: str = "eval") -> Dict:
    """Render synthetic eval pages from real (or seed) pairs into ``synthetic/<split>/``."""
    from .synth_render import generate_dataset
    pairs = load_eval_pairs(cfg)
    pair_dicts = [{"src": p.src, "tgt": p.tgt} for p in pairs]
    return generate_dataset(pair_dicts, str(synthetic_dir(cfg, split)),
                            n_pages=n_pages or cfg.data.synth_eval_pages,
                            lines_per_page=cfg.data.lines_per_page, cfg=cfg.data, seed=cfg.data.seed)


__all__ = ["Pair", "load_pairs", "load_seed_pairs", "seed_split", "load_eval_pairs",
           "synthetic_dir", "load_manifest", "load_eval_pages", "build_synthetic_eval"]
