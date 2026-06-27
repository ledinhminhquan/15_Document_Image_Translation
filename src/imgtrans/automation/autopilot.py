"""One-button autopilot: data -> baseline -> train-mt -> evaluate -> tune -> error-analysis
-> layout-fidelity -> benchmark -> demo -> monitoring -> report + slides + grade + bundle.

Each stage is isolated in its own try/except and never aborts the run. The MT fine-tune is
skipped when torch/transformers/datasets are unavailable; everything else degrades to the
dictionary MT + SeedEngine OCR + numpy metrics. A zipped submission bundle is written.
"""

from __future__ import annotations

import json
import time
import zipfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import AppConfig, artifacts_dir, ensure_dirs
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)


def _step(steps: List[Dict], name: str, fn: Callable[[], Any], skip: bool = False) -> Optional[Any]:
    if skip:
        logger.info("autopilot step %s skipped", name)
        steps.append({"step": name, "status": "skipped", "seconds": 0.0})
        return None
    t0 = time.perf_counter()
    try:
        out = fn()
        steps.append({"step": name, "status": "ok", "seconds": round(time.perf_counter() - t0, 2)})
        return out
    except Exception as exc:
        logger.warning("autopilot step %s failed: %s", name, exc)
        steps.append({"step": name, "status": "error", "error": str(exc),
                      "seconds": round(time.perf_counter() - t0, 2)})
        return None


def _training_available() -> bool:
    return all(find_spec(m) is not None for m in ("torch", "transformers", "datasets"))


def _demo_agent(cfg: AppConfig) -> Dict:
    from ..agent.imgtrans_agent import ImgTransAgent
    from ..data import samples
    agent = ImgTransAgent(cfg, load_model=False)
    out: List[Dict] = []
    for i, spec in enumerate(samples.seed_pages()):
        job = agent.run(spec=spec, mode="text_only", save=False)
        out.append({"page": i, "status": job.status.value, "n_blocks": job.n_blocks,
                    "n_translatable": job.n_translatable, "fit_rate": job.fit_rate,
                    "decisions": [(x.id, x.branch) for x in job.decisions]})
    return {"demos": out}


def run_autopilot(cfg: AppConfig, title: str = None, author: str = None,
                  train: bool = True, limit: Optional[int] = None) -> Dict:
    ensure_dirs()
    title = title or cfg.project_title
    author = author or cfg.author
    steps: List[Dict] = []

    can_train = train and _training_available()

    _step(steps, "prepare_data", lambda: __import__(
        "imgtrans.data.download_dataset", fromlist=["download_all"]).download_all(cfg))
    _step(steps, "train_baseline", lambda: __import__(
        "imgtrans.training.train_baseline", fromlist=["build_baseline"]).build_baseline(cfg, limit=limit))

    if train and not can_train:
        logger.info("training requested but torch/transformers/datasets unavailable - skipping")
    _step(steps, "train_mt", lambda: __import__(
        "imgtrans.training.train_mt", fromlist=["train_mt"]).train_mt(cfg, limit=limit), skip=not can_train)

    _step(steps, "evaluate", lambda: __import__(
        "imgtrans.training.evaluate", fromlist=["evaluate"]).evaluate(cfg, save=True, load_model=can_train))
    _step(steps, "tune", lambda: __import__(
        "imgtrans.training.tune", fromlist=["tune"]).tune(cfg, save=True, load_model=can_train))
    _step(steps, "error_analysis", lambda: __import__(
        "imgtrans.analysis.error_analysis", fromlist=["error_analysis"]).error_analysis(cfg, save=True))
    _step(steps, "layout_fidelity", lambda: __import__(
        "imgtrans.analysis.layout_fidelity", fromlist=["layout_fidelity_report"]).layout_fidelity_report(cfg, save=True))
    _step(steps, "benchmark", lambda: __import__(
        "imgtrans.analysis.latency", fromlist=["benchmark"]).benchmark(cfg, n=8, warmup=2, save=True))
    _step(steps, "demo_agent", lambda: _demo_agent(cfg))
    _step(steps, "monitoring", lambda: __import__(
        "imgtrans.monitoring.drift_report", fromlist=["monitoring_report"]).monitoring_report(cfg))

    stamp = utc_stamp()
    sub = artifacts_dir() / "submission" / f"submission-{stamp}"
    sub.mkdir(parents=True, exist_ok=True)
    report = _step(steps, "report", lambda: __import__(
        "imgtrans.autoreport.report_pdf", fromlist=["generate_report"]).generate_report(
        cfg, title=title, author=author, out_path=sub / "report.pdf"))
    slides = _step(steps, "slides", lambda: __import__(
        "imgtrans.autoreport.slides_pptx", fromlist=["generate_slides"]).generate_slides(
        cfg, title=title, author=author, out_path=sub / "slides.pptx"))

    repo_root = Path(__file__).resolve().parents[3]
    checklist = _step(steps, "grading", lambda: __import__(
        "imgtrans.grading.checklist", fromlist=["build_checklist"]).build_checklist(repo_root))

    manifest = {"generated_at": utc_now_iso(), "title": title, "author": author,
                "student_id": cfg.student_id, "steps": steps, "grading_checklist": checklist}
    try:
        (sub / "submission_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("manifest write failed: %s", exc)

    zip_path = None
    try:
        zip_path = sub / "submission_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sub.iterdir():
                if f.is_file() and f.name != "submission_bundle.zip":
                    z.write(f, f.name)
    except Exception as exc:
        logger.warning("bundle zip failed: %s", exc)
        zip_path = None

    logger.info("Autopilot done -> %s", sub)
    return {"steps": steps, "submission_dir": str(sub),
            "zip": str(zip_path) if zip_path else None,
            "report": str(report) if report else None,
            "slides": str(slides) if slides else None,
            "grade_summary": (checklist or {}).get("summary")}


__all__ = ["run_autopilot"]
