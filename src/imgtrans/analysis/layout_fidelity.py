"""Layout-fidelity report (the P15 special quality analysis).

Runs the agent on the seed pages and aggregates the overlay-render quality: the
fit-rate (fraction of translated blocks that fit their source box), mean font shrink
(translated vs source size) and mean overflow. This quantifies how well the
layout-preserving overlay - the project's headline value-add - actually works.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..imaging import render as rendermod
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def layout_fidelity_report(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..agent.imgtrans_agent import ImgTransAgent
        from ..data import samples
        from ..imaging.layout import TextBlock
        agent = ImgTransAgent(cfg, load_model=False)
        pages = samples.seed_pages()
    except Exception as exc:
        return _stub(str(exc), save)
    if limit:
        pages = pages[:limit]

    fit_rates: List[float] = []
    shrinks: List[float] = []
    overflows: List[float] = []
    overlay = side_by_side = 0
    for spec in pages:
        job = agent.run(spec=spec, mode="overlay", save=False)
        fid = job.fidelity or {}
        if fid.get("fit_rate") is not None:
            fit_rates.append(fid["fit_rate"])
        if fid.get("mean_shrink") is not None:
            shrinks.append(fid["mean_shrink"])
        if fid.get("mean_overflow") is not None:
            overflows.append(fid["mean_overflow"])
        if job.render_mode_used == "side_by_side":
            side_by_side += 1
        else:
            overlay += 1

    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    result = {"n_pages": len(pages), "mean_fit_rate": _avg(fit_rates), "mean_shrink": _avg(shrinks),
              "mean_overflow": _avg(overflows), "overlay_pages": overlay, "side_by_side_pages": side_by_side,
              "min_fit_rate_threshold": cfg.agent.min_fit_rate}
    if save:
        _save(result)
    logger.info("layout fidelity: mean_fit_rate=%s overlay=%d side_by_side=%d",
                result["mean_fit_rate"], overlay, side_by_side)
    return result


def _stub(error: str, save: bool) -> Dict:
    result = {"n_pages": 0, "mean_fit_rate": None, "mean_shrink": None, "mean_overflow": None,
              "overlay_pages": 0, "side_by_side_pages": 0, "error": error}
    if save:
        _save(result)
    return result


def _save(result: Dict) -> None:
    try:
        d = run_dir() / "layout_fidelity"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"fidelity-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.info("layout_fidelity: could not save (%s)", exc)


__all__ = ["layout_fidelity_report"]
