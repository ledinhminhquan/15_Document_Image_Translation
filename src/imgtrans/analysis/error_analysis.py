"""End-to-end error analysis (offline): chrF buckets + OCR/skip/fit issues.

Runs the agent on the synthetic seed pages, scores each page's end-to-end translation
chrF vs the gold target, and buckets good/medium/poor; records pages that needed review
(low OCR confidence / low verify chrF / poor overlay fit). Short keys ``good``/``medium``/
``poor`` feed the charts.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..training.metrics import chrf

logger = get_logger(__name__)


def error_analysis(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..agent.imgtrans_agent import ImgTransAgent
        from ..data import samples
        agent = ImgTransAgent(cfg, load_model=False)
        pages = samples.seed_pages()
    except Exception as exc:
        return _stub(str(exc), save)
    if limit:
        pages = pages[:limit]

    good = medium = poor = needs_review = skipped = low_fit = 0
    chrfs, fits = [], []
    examples: List[Dict] = []
    for i, spec in enumerate(pages):
        try:
            job = agent.run(spec=spec, mode="text_only", save=False)
        except Exception:
            continue
        gold = "\n\n".join(b.get("translation", "") for b in spec.get("blocks", []) if b.get("text"))
        score = chrf([job.translated_text], [gold])
        chrfs.append(score)
        if job.fit_rate is not None:
            fits.append(job.fit_rate)
        if job.needs_review:
            needs_review += 1
        skipped += job.n_skipped_lowconf
        if job.fit_rate is not None and job.fit_rate < cfg.agent.min_fit_rate:
            low_fit += 1
        if score >= 60:
            good += 1
        elif score >= 35:
            medium += 1
        else:
            poor += 1
            if len(examples) < 8:
                examples.append({"page": i, "chrf": round(score, 2), "fit_rate": job.fit_rate})
    n = max(1, len(chrfs))
    result = {"n_pages": len(chrfs), "mean_chrf": round(sum(chrfs) / n, 2),
              "mean_fit_rate": round(sum(fits) / len(fits), 4) if fits else None,
              "good": good, "medium": medium, "poor": poor,
              "needs_review": needs_review, "skipped_lowconf": skipped, "low_fit": low_fit,
              "worst_examples": examples}
    if save:
        _save(result)
    logger.info("error analysis: good=%d medium=%d poor=%d needs_review=%d low_fit=%d",
                good, medium, poor, needs_review, low_fit)
    return result


def _stub(error: str, save: bool) -> Dict:
    result = {"n_pages": 0, "mean_chrf": 0.0, "mean_fit_rate": None, "good": 0, "medium": 0, "poor": 0,
              "needs_review": 0, "skipped_lowconf": 0, "low_fit": 0, "worst_examples": [], "error": error}
    if save:
        _save(result)
    return result


def _save(result: Dict) -> None:
    try:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("error_analysis: could not save (%s)", exc)


__all__ = ["error_analysis"]
