"""Matplotlib charts for the document-image-translation report/slides.

  * an **MT quality** bar chart - chrF + BLEU for identity vs dictionary vs model;
  * an **end-to-end** chart - image-translation chrF (0-100), OCR accuracy (1-CER)x100, fit-rate x100;
  * a **chrF bucket** chart (good / medium / poor) from error analysis.

Returns saved PNG paths under ``run_dir()/report``; matplotlib lazy-imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger
from . import artifact_loader as AL

logger = get_logger(__name__)

_IDENTITY = "#cbd5e0"
_DICT = "#9aa7b4"
_MODEL = "#2b6cb0"
_GOOD = "#2f855a"
_MED = "#dd6b20"
_POOR = "#c53030"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _f(d: Dict[str, Any], key: str) -> Optional[float]:
    v = (d or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def quality_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    systems = AL.mt_systems(arts)
    if not systems:
        return None
    model_name = AL.model_system_name(arts)
    try:
        plt = _mpl()
        metrics = [("chrf", "chrF"), ("bleu", "BLEU")]
        series = [("identity", systems.get("identity") or {}, _IDENTITY),
                  ("dictionary", systems.get("dictionary") or {}, _DICT)]
        if model_name and model_name not in ("identity", "dictionary"):
            series.append((model_name, systems.get(model_name) or {}, _MODEL))
        x = list(range(len(metrics)))
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for si, (label, data, color) in enumerate(series):
            vals = [(_f(data, k) or 0.0) for k, _ in metrics]
            offs = [i + (si - (len(series) - 1) / 2) * width for i in x]
            bars = ax.bar(offs, vals, width=width, label=label, color=color)
            for rect, v in zip(bars, vals):
                if v > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.5, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in metrics])
        ax.set_ylabel("score (0-100)"); ax.set_ylim(0, 105)
        ax.set_title("MT quality: model vs dictionary vs identity floor")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("quality_chart skipped (%s)", exc)
        return None


def e2e_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    e2e = AL.e2e_metric(arts, "chrf")
    cer = AL.ocr_metric(arts, "cer")
    fit = AL.headline(arts, "mean_fit_rate")
    if e2e is None and cer is None and fit is None:
        return None
    try:
        plt = _mpl()
        labels = ["e2e image chrF", "OCR acc\n(1-CER)x100", "fit-rate x100"]
        vals = [(e2e or 0.0), (1.0 - (cer or 0.0)) * 100, (fit or 0.0) * 100]
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        ax.bar(labels, vals, color=[_MODEL, _GOOD, "#553c9a"])
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("score")
        ax.set_title("End-to-end: image-translation chrF + OCR accuracy + overlay fit")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("e2e_chart skipped (%s)", exc)
        return None


def buckets_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    err = arts.get("error_analysis") or {}
    vals = [err.get("good"), err.get("medium"), err.get("poor")]
    if not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return None
    try:
        plt = _mpl()
        labels = ["good\n(chrF>=60)", "medium\n(35-60)", "poor\n(<35)"]
        nums = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0 for v in vals]
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.bar(labels, nums, color=[_GOOD, _MED, _POOR])
        for i, v in enumerate(nums):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("# pages"); ax.set_title("End-to-end quality buckets (by chrF)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("buckets_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []
    charts: List[Tuple[str, Path]] = []
    jobs = [("quality", lambda p: quality_chart(arts, p)),
            ("e2e", lambda p: e2e_chart(arts, p)),
            ("buckets", lambda p: buckets_chart(arts, p))]
    for name, fn in jobs:
        try:
            p = fn(out_dir / f"{name}.png")
        except Exception as exc:
            logger.info("chart %s skipped (%s)", name, exc)
            p = None
        if p:
            charts.append((name, p))
    return charts


__all__ = ["quality_chart", "e2e_chart", "buckets_chart", "build_all"]
