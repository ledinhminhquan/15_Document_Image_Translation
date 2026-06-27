"""Evaluation: MT quality + OCR quality + end-to-end image-translation + layout fidelity.

Four complementary measurements:
* **MT** (the trainable core): chrF / BLEU on held-out sentence pairs, vs the dictionary
  and identity baselines (the floors the fine-tuned model must beat).
* **OCR** (front-end): CER / WER of the OCR text vs the gold source text on the synthetic
  document pages (perfect-OCR offline via SeedEngine; realistic with Tesseract on Colab).
* **End-to-end**: the headline number - run the FULL agent (OCR -> MT) on each page image
  and score the assembled translation against the gold target with chrF / BLEU.
* **Layout fidelity**: mean overlay fit-rate over the rendered blocks.

Everything runs offline on the seed pages (no torch / tesseract); on Colab the same code
reads the rendered PNGs through Tesseract and the fine-tuned model for honest numbers.
Results are written to ``run_dir/eval.json``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..data.dataset import load_eval_pages, load_eval_pairs
from ..logging_utils import get_logger
from ..models.baseline import IdentityTranslator
from ..mt.translator import DictionaryTranslator
from . import metrics as M

logger = get_logger(__name__)


def evaluate_mt(cfg: AppConfig, translator=None, *, limit: Optional[int] = None) -> Dict[str, Any]:
    pairs = load_eval_pairs(cfg, limit=limit or cfg.data.max_eval_samples)
    srcs = [p.src for p in pairs]
    refs = [p.tgt for p in pairs]
    systems: Dict[str, Any] = {}

    def score(name, hyps):
        systems[name] = {"chrf": M.chrf(hyps, refs), "bleu": M.bleu(hyps, refs)}

    score("identity", IdentityTranslator().translate_batch(srcs))
    score("dictionary", DictionaryTranslator().translate_batch(srcs))
    if translator is not None and getattr(translator, "name", "") not in ("dictionary", "identity"):
        try:
            score(getattr(translator, "name", "model"), translator.translate_batch(srcs))
        except Exception as exc:
            logger.info("model MT eval skipped (%s)", exc)
    return {"n": len(pairs), "systems": systems}


def evaluate_ocr_e2e(cfg: AppConfig, agent=None, *, limit: Optional[int] = None,
                     ocr_noise: float = 0.0, load_model: bool = True) -> Dict[str, Any]:
    from ..agent.imgtrans_agent import ImgTransAgent
    agent = agent or ImgTransAgent(cfg, load_model=load_model)
    pages = load_eval_pages(cfg)
    if limit:
        pages = pages[:limit]

    have_pil = _have_pil()
    ocr_hyps: List[str] = []
    ocr_refs: List[str] = []
    e2e_hyps: List[str] = []
    e2e_refs: List[str] = []
    fit_rates: List[float] = []

    for spec in pages:
        blocks = spec.get("blocks", [])
        gold_src = "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
        gold_tgt = "\n\n".join(b.get("translation", "") for b in blocks if b.get("text"))
        img = _page_image(spec, ocr_noise) if have_pil else None
        if img is not None:
            job = agent.run(image=img, mode="text_only", save=False)
        else:
            job = agent.run(spec=spec, mode="text_only", save=False)
        ocr_hyps.append(job.source_text)
        ocr_refs.append(gold_src)
        e2e_hyps.append(job.translated_text)
        e2e_refs.append(gold_tgt)
        if job.fit_rate is not None:
            fit_rates.append(job.fit_rate)

    return {
        "n_pages": len(pages),
        "ocr": M.ocr_metrics(ocr_hyps, ocr_refs),
        "end_to_end": {"chrf": M.chrf(e2e_hyps, e2e_refs), "bleu": M.bleu(e2e_hyps, e2e_refs)},
        "layout": {"mean_fit_rate": round(sum(fit_rates) / len(fit_rates), 4) if fit_rates else None},
        "ocr_engine": getattr(agent, "ocr_engine", None) or "seed/spec",
    }


def _have_pil() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:
        return False


def _page_image(spec: Dict, noise: float):
    """Render a page spec to an image (so the real OCR engine has pixels to read)."""
    try:
        if spec.get("image_path"):
            from PIL import Image
            return Image.open(spec["image_path"])
        from ..data.synth_render import render_page
        img, _ = render_page(spec, degrade=noise, seed=spec.get("seed", 0))
        return img
    except Exception as exc:
        logger.info("page render skipped (%s)", exc)
        return None


def evaluate(cfg: AppConfig, *, limit: Optional[int] = None, ocr_noise: float = 0.0,
             save: bool = True, load_model: bool = True) -> Dict[str, Any]:
    from ..agent.imgtrans_agent import ImgTransAgent
    agent = ImgTransAgent(cfg, load_model=load_model)
    report = {
        "mt": evaluate_mt(cfg, translator=agent.translator, limit=limit),
        "ocr_e2e": evaluate_ocr_e2e(cfg, agent=agent, limit=limit, ocr_noise=ocr_noise),
        "model_version": agent.translator.__dict__.get("version", getattr(agent.translator, "version", "?")),
    }
    headline = report["ocr_e2e"]["end_to_end"]["chrf"]
    report["headline"] = {"end_to_end_chrf": headline,
                          "mt_chrf": _best_model_chrf(report["mt"]),
                          "ocr_cer": report["ocr_e2e"]["ocr"]["cer"],
                          "mean_fit_rate": report["ocr_e2e"]["layout"]["mean_fit_rate"]}
    if save:
        out = run_dir() / "eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("eval -> %s (e2e chrF=%s, OCR CER=%s)", out, headline, report["headline"]["ocr_cer"])
    return report


def _best_model_chrf(mt: Dict[str, Any]) -> Optional[float]:
    systems = mt.get("systems", {})
    model = {k: v for k, v in systems.items() if k not in ("identity", "dictionary")}
    pool = model or {k: v for k, v in systems.items() if k == "dictionary"}
    return max((v["chrf"] for v in pool.values()), default=None)


__all__ = ["evaluate", "evaluate_mt", "evaluate_ocr_e2e"]
