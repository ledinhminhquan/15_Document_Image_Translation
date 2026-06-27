"""Lightweight inference-quality sweep for the document-image-translation pipeline.

Sweeps the MT decoding beam width and reports end-to-end image-translation chrF +
latency on the seed pages - a cheap, offline proxy for the quality/speed trade-off.
(With the dictionary baseline, beams is a no-op, so a single trial is reported; the
real gain shows on the fine-tuned model on Colab.)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..data.dataset import load_eval_pages
from ..logging_utils import get_logger
from ..training import metrics as M

logger = get_logger(__name__)


def tune(cfg: AppConfig, beams: Optional[List[int]] = None, save: bool = True, load_model: bool = True) -> Dict:
    beams = beams or [1, 4, 8]
    from ..agent.imgtrans_agent import ImgTransAgent
    from ..mt.translator import load_translator
    translator = load_translator(cfg.mt, prefer="transformer" if load_model else "dictionary")
    is_transformer = getattr(translator, "name", "") == "transformer"
    pages = load_eval_pages(cfg)[:20]

    trials: List[Dict[str, Any]] = []
    for nb in beams:
        cfg.mt.num_beams = nb
        agent = ImgTransAgent(cfg, load_model=load_model, translator=translator)
        hyps, refs = [], []
        t0 = time.perf_counter()
        for spec in pages:
            job = agent.run(spec=spec, mode="text_only", save=False)
            hyps.append(job.translated_text)
            refs.append("\n\n".join(b.get("translation", "") for b in spec.get("blocks", []) if b.get("text")))
        dt = (time.perf_counter() - t0) * 1000 / max(1, len(pages))
        trials.append({"num_beams": nb, "e2e_chrf": M.chrf(hyps, refs), "latency_ms_per_page": round(dt, 1)})
        if not is_transformer:
            break   # dictionary baseline is beam-invariant

    best = max(trials, key=lambda t: t["e2e_chrf"]) if trials else {}
    result = {"trials": trials, "best": best}
    if save:
        out = run_dir() / "tune"
        out.mkdir(parents=True, exist_ok=True)
        (out / "tune.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("tune: best beams=%s e2e_chrf=%s", best.get("num_beams"), best.get("e2e_chrf"))
    return result


__all__ = ["tune"]
