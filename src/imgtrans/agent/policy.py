"""Decision-point logic for the document-image-translation agent (pure, testable).

Five explicit decision points on the cascade's intermediate artifacts:
* **D1** input router + page-quality gate (image / pdf / spec / text; blur/contrast/ink).
* **D2** born-digital vs scanned routing (skip OCR when a real text layer exists).
* **D3** per-block OCR-confidence gate (skip translating low-confidence / garbage blocks).
* **D4** translation verification (round-trip back-translation chrF [soft] + length ratio).
* **D5** render-fit feasibility gate (fit-rate -> overlay vs side_by_side vs needs_review).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..config import AgentConfig

_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def letter_ratio(text: str) -> float:
    s = re.sub(r"\s", "", text or "")
    if not s:
        return 0.0
    return sum(1 for c in s if _LETTERS.match(c)) / len(s)


def quality_gate(quality: Optional[float], cfg: AgentConfig) -> Dict[str, Any]:
    """D1 — page-quality routing."""
    if quality is None:
        return {"ok": True, "branch": "no_quality_check"}
    if quality < cfg.quality_min:
        return {"ok": False, "branch": "low_quality", "quality": quality}
    return {"ok": True, "branch": "ok", "quality": quality}


def block_translatable(text: str, conf: float, cfg: AgentConfig) -> Dict[str, Any]:
    """D3 — should this OCR block be translated? (enough chars + confidence)."""
    t = (text or "").strip()
    if len(t) < cfg.min_block_chars or letter_ratio(t) < 0.2:
        return {"ok": False, "branch": "too_short"}
    if conf is not None and conf < cfg.ocr_confidence_min:
        return {"ok": False, "branch": "low_confidence", "conf": round(conf, 3)}
    return {"ok": True, "branch": "ok", "conf": round(conf, 3) if conf is not None else None}


def translate_sane(src: str, hyp: str, cfg: AgentConfig) -> Dict[str, Any]:
    """D4 (per block) — basic sanity of one translation (non-empty, plausible length ratio)."""
    if not (hyp or "").strip():
        return {"ok": False, "branch": "empty"}
    ls, lh = max(1, len(src)), len(hyp)
    ratio = lh / ls
    if ratio < cfg.length_ratio_low or ratio > cfg.length_ratio_high:
        return {"ok": False, "branch": "length_ratio", "ratio": round(ratio, 3)}
    return {"ok": True, "branch": "ok", "ratio": round(ratio, 3)}


def verify_gate(verify_chrf: Optional[float], n_bad: int, cfg: AgentConfig) -> Dict[str, Any]:
    """D4 (page) — round-trip chrF (soft) + per-block sanity failures."""
    if cfg.verify_enabled and verify_chrf is not None and verify_chrf < cfg.verify_min_chrf:
        return {"ok": False, "branch": "low_similarity", "chrf": verify_chrf}
    if n_bad > 0:
        return {"ok": False, "branch": "sanity_failures", "n_bad": n_bad}
    return {"ok": True, "branch": "ok"}


def render_gate(fit_rate: Optional[float], mode: str, cfg: AgentConfig) -> Dict[str, Any]:
    """D5 — can the translation be overlaid? fall back to side_by_side when fit is poor."""
    if mode == "text_only":
        return {"ok": True, "branch": "text_only", "mode": "text_only"}
    if fit_rate is None:
        return {"ok": True, "branch": mode, "mode": mode}
    if mode == "overlay" and fit_rate < cfg.min_fit_rate:
        return {"ok": True, "branch": "fallback_side_by_side", "mode": "side_by_side",
                "fit_rate": fit_rate}
    return {"ok": True, "branch": mode, "mode": mode, "fit_rate": fit_rate}


__all__ = ["letter_ratio", "quality_gate", "block_translatable", "translate_sane",
           "verify_gate", "render_gate"]
