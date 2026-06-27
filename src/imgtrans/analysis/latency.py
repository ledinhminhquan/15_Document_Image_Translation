"""Latency benchmark: end-to-end document-image-translation agent throughput (offline).

Builds a single ``ImgTransAgent`` (SeedEngine + dictionary MT) and times ``.run(spec=...)``
over the seed pages. Reports p50/p95/p99/mean per-page latency_ms, throughput,
decision-presence (D1-D5), and the status mix.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _pct(xs: List[float]) -> Dict[str, float]:
    import numpy as np
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "mean": round(float(a.mean()), 2)}


def benchmark(cfg: AppConfig = None, n: int = 10, warmup: int = 2, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    try:
        from ..agent.imgtrans_agent import ImgTransAgent
        from ..data import samples
        agent = ImgTransAgent(cfg, load_model=False)
        pages = samples.seed_pages() or []
    except Exception as exc:
        logger.warning("benchmark: could not build agent (%s)", exc)
        out = {"device": device, "n": n, "warmup": warmup, "error": str(exc),
               "latency_ms": _pct([]), "throughput_per_s": 0.0, "decision_presence": {}, "statuses": {}}
        if save:
            _save(out)
        return out

    total = n + warmup
    workload = (pages * (total // max(1, len(pages)) + 1))[:total]
    for spec in workload[:warmup]:
        try:
            agent.run(spec=spec, mode="text_only", save=False)
        except Exception:
            pass

    lat: List[float] = []
    decisions: Dict[str, int] = {}
    statuses: Dict[str, int] = {}
    n_ok = 0
    for spec in workload[warmup:]:
        t0 = time.perf_counter()
        try:
            job = agent.run(spec=spec, mode="text_only", save=False)
            wall = (time.perf_counter() - t0) * 1000.0
            lat.append(float(job.metrics.get("latency_ms", wall)))
            n_ok += 1
            statuses[job.status.value] = statuses.get(job.status.value, 0) + 1
            for dec in job.decisions:
                decisions[dec.id] = decisions.get(dec.id, 0) + 1
        except Exception as exc:
            logger.info("benchmark iteration failed (%s)", exc)

    import numpy as np
    mean_ms = float(np.mean(lat)) if lat else 0.0
    presence = {d: round(decisions[d] / n_ok, 3) for d in sorted(decisions)} if n_ok else {}
    out = {"device": device, "n": n_ok, "warmup": warmup,
           "mt": getattr(agent.translator, "name", "?"),
           "latency_ms": _pct(lat),
           "throughput_per_s": round(1000.0 / max(1e-6, mean_ms), 2) if lat else 0.0,
           "decision_presence": presence, "statuses": statuses}
    if save:
        _save(out)
    logger.info("benchmark: n=%d p50=%.1fms p95=%.1fms throughput=%.1f/s",
                n_ok, out["latency_ms"]["p50"], out["latency_ms"]["p95"], out["throughput_per_s"])
    return out


def _save(out: Dict) -> None:
    try:
        d = run_dir() / "benchmark"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"benchmark-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.info("benchmark: could not save (%s)", exc)


__all__ = ["benchmark"]
