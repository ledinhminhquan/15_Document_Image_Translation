"""Collect generated run artifacts into one dict for the report + slides generators.

Reads the JSON written under ``run_dir()`` - the MT/OCR/e2e eval, the error analysis,
the layout-fidelity report, a latency benchmark, and a monitoring snapshot - plus the
trained-MT metadata. Every read is defensive: missing/malformed -> ``None``/``{}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig, run_dir
from ..models.model_registry import read_metadata, resolve_latest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_artifacts(cfg: AppConfig) -> Dict[str, Any]:
    rd = run_dir()
    arts: Dict[str, Any] = {
        "eval": _load_json(rd / "eval.json"),
        "error_analysis": _load_json(rd / "error_analysis" / "latest.json"),
        "layout_fidelity": _load_json(rd / "layout_fidelity" / "latest.json"),
        "benchmark": _load_json(rd / "benchmark" / "latest.json"),
        "tune": _load_json(rd / "tune" / "tune.json"),
        "monitoring": _load_json(rd / "monitoring" / "latest.json"),
    }
    try:
        latest = resolve_latest(cfg.mt.output_dir)
        arts["model_meta"] = read_metadata(latest) if latest else {}
    except Exception:
        arts["model_meta"] = {}
    return arts


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def mt_systems(arts: Dict[str, Any]) -> Dict[str, Any]:
    return ((arts.get("eval") or {}).get("mt") or {}).get("systems") or {}


def model_system_name(arts: Dict[str, Any]) -> Optional[str]:
    sys = mt_systems(arts)
    for k in sys:
        if k not in ("identity", "dictionary"):
            return k
    return "dictionary" if "dictionary" in sys else None


def mt_metric(arts: Dict[str, Any], system: str, key: str) -> Optional[float]:
    return _num((mt_systems(arts).get(system) or {}).get(key))


def e2e_metric(arts: Dict[str, Any], key: str) -> Optional[float]:
    block = ((arts.get("eval") or {}).get("ocr_e2e") or {}).get("end_to_end") or {}
    return _num(block.get(key))


def ocr_metric(arts: Dict[str, Any], key: str) -> Optional[float]:
    block = ((arts.get("eval") or {}).get("ocr_e2e") or {}).get("ocr") or {}
    return _num(block.get(key))


def headline(arts: Dict[str, Any], key: str) -> Optional[float]:
    return _num(((arts.get("eval") or {}).get("headline") or {}).get(key))


def has_eval(arts: Dict[str, Any]) -> bool:
    return bool(mt_systems(arts))


def model_version(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("version") or (arts.get("eval") or {}).get("model_version") or "untrained (dictionary fallback)")


def base_model(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("base_model") or "facebook/m2m100_418M")


def buckets(arts: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ea = arts.get("error_analysis") or {}
    return {"good": _num(ea.get("good")), "medium": _num(ea.get("medium")), "poor": _num(ea.get("poor"))}


def latency(arts: Dict[str, Any], pct: str = "p50") -> Optional[float]:
    b = (arts.get("benchmark") or {}).get("latency_ms") or {}
    return _num(b.get(pct))


def read_doc(name: str) -> str:
    p = repo_root() / "docs" / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


__all__ = ["load_artifacts", "read_doc", "repo_root", "mt_systems", "model_system_name", "mt_metric",
           "e2e_metric", "ocr_metric", "headline", "has_eval", "model_version", "base_model",
           "buckets", "latency"]
