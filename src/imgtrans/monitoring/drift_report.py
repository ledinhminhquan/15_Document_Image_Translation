"""Production monitoring report from the serving job log (image_translation JSONL).

Turns raw ``image_translation`` events into a health picture: request volume, the
input-kind mix, OCR-engine mix, status mix, the **needs-review rate** (human post-edit
load), the mean overlay fit-rate, latency (mean + p95), and a drift signal comparing a
recent window to an earlier baseline (a rising needs-review rate or falling fit-rate is
the tell-tale of harder/lower-quality incoming documents or a degrading model). Stdlib
only; never raises past its entrypoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_EVENT = "image_translation"


def _read_logs(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("could not read job log %s: %s", path, exc)
        return rows
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 1)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(float(ordered[rank]), 1)


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _window_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    statuses: Dict[str, int] = {}
    kinds: Dict[str, int] = {}
    engines: Dict[str, int] = {}
    needs_review = 0
    lats: List[float] = []
    fits: List[float] = []
    for r in rows:
        statuses[str(r.get("status", "?"))] = statuses.get(str(r.get("status", "?")), 0) + 1
        kinds[str(r.get("input_kind", "?"))] = kinds.get(str(r.get("input_kind", "?")), 0) + 1
        engines[str(r.get("ocr_engine", "?"))] = engines.get(str(r.get("ocr_engine", "?")), 0) + 1
        if bool(r.get("needs_review")):
            needs_review += 1
        if _is_num(r.get("fit_rate")):
            fits.append(float(r["fit_rate"]))
        metrics = r.get("metrics") or {}
        if isinstance(metrics, dict) and _is_num(metrics.get("latency_ms")):
            lats.append(float(metrics["latency_ms"]))
    return {"n": n, "status_distribution": statuses, "input_kind_distribution": kinds,
            "ocr_engine_distribution": engines, "needs_review_rate": round(needs_review / n, 4),
            "mean_fit_rate": _mean(fits), "mean_latency_ms": _mean(lats),
            "p95_latency_ms": _percentile(lats, 95)}


def _delta(base: Dict[str, Any], recent: Dict[str, Any], key: str) -> Optional[float]:
    a, b = base.get(key), recent.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(b) - float(a), 4)
    return None


def _drift(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rows) < 6:
        return {"available": False, "reason": "need >=6 events to split baseline/recent windows"}
    half = len(rows) // 2
    base = _window_stats(rows[:half])
    recent = _window_stats(rows[half:])
    d_nr = _delta(base, recent, "needs_review_rate")
    d_fit = _delta(base, recent, "mean_fit_rate")
    d_lat = _delta(base, recent, "mean_latency_ms")
    flags: List[str] = []
    if (d_nr or 0) > 0.15:
        flags.append("rising_needs_review_rate")
    if d_fit is not None and d_fit < -0.05:
        flags.append("falling_layout_fit_rate")
    if d_lat is not None and base.get("mean_latency_ms"):
        if d_lat / (base["mean_latency_ms"] or 1.0) > 0.5:
            flags.append("latency_regression")
    return {"available": True, "baseline_window": base, "recent_window": recent,
            "delta_needs_review_rate": d_nr, "delta_mean_fit_rate": d_fit, "delta_mean_latency_ms": d_lat,
            "flags": flags, "alert": bool(flags)}


def _recommendations(overall: Dict[str, Any], drift: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    nr = overall.get("needs_review_rate") or 0.0
    fit = overall.get("mean_fit_rate")
    flags = drift.get("flags") or []
    if nr > 0.4:
        recs.append("High needs-review rate ({:.0%}): check the OCR-confidence + verify thresholds, "
                    "and consider re-fine-tuning the MT core on the failing domain.".format(nr))
    if _is_num(fit) and fit < 0.7:
        recs.append("Mean overlay fit-rate {:.2f} is low: translations are overflowing their boxes - "
                    "tune the renderer (min font / wrapping) or default to side-by-side.".format(fit))
    if "rising_needs_review_rate" in flags or "falling_layout_fit_rate" in flags:
        recs.append("Quality is drifting vs the baseline window: incoming images may be harder "
                    "(lower DPI / new languages) - collect a fresh slice and re-evaluate.")
    if not recs:
        recs.append("No action needed: monitoring metrics within healthy operating ranges.")
    return recs


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    path = Path(log_path) if log_path else cfg.serving.job_log_path
    rows = _read_logs(path)
    events = [r for r in rows if r.get("event", _EVENT) == _EVENT]
    if not events:
        logger.info("monitoring: no image_translation events at %s", path)
        result = {"status": "no_data", "log_path": str(path), "n_events": 0, "request_volume": 0,
                  "overall": {"n": 0}, "drift": {"available": False, "reason": "no events"},
                  "recommendations": ["No job logs yet: exercise the agent / API to populate the log."],
                  "generated_at": utc_stamp()}
    else:
        overall = _window_stats(events)
        drift = _drift(events)
        result = {"status": "ok", "log_path": str(path), "n_events": len(events),
                  "request_volume": len(events), "overall": overall, "drift": drift,
                  "recommendations": _recommendations(overall, drift), "generated_at": utc_stamp()}
    if save:
        try:
            out = run_dir() / "monitoring"
            out.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result, indent=2, ensure_ascii=False)
            (out / f"monitor-{utc_stamp()}.json").write_text(payload, encoding="utf-8")
            (out / "latest.json").write_text(payload, encoding="utf-8")
        except Exception as exc:
            logger.warning("monitoring: could not save report: %s", exc)
    logger.info("monitoring: %s events, needs_review=%.0f%% mean_fit_rate=%s p95=%s ms, drift_alert=%s",
                result["n_events"], 100 * (result["overall"].get("needs_review_rate") or 0.0),
                result["overall"].get("mean_fit_rate"), result["overall"].get("p95_latency_ms"),
                result.get("drift", {}).get("alert", False))
    return result


__all__ = ["monitoring_report"]
