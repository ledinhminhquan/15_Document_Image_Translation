"""The document-image-translation agent - a deterministic FSM over the cascade.

    ingest (D1 input router + page-quality) -> ocr (D2 born-digital vs scanned)
        -> translate (D3 per-block OCR-confidence gate) -> verify (D4 round-trip back-translation)
        -> render (D5 fit-feasibility gate: overlay | side_by_side | text_only)

Holds the MT translator (+ an optional reverse translator for D4) loaded once. Runs fully
offline (SeedEngine reading the embedded gold spec / a raw-text splitter + dictionary MT) and
upgrades to Tesseract + a fine-tuned seq2seq when present. Low-confidence OCR blocks are
skipped (not mistranslated); blocks whose translation cannot fit fall back to a side-by-side
panel; anything low-confidence is flagged for human review (needs_review). Every step is timed
and traced; same input + same models + brain disabled => identical output.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import JsonlLogger, get_logger
from . import tools
from .llm_orchestrator import LLMBrain
from .state import JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


class ImgTransAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True,
                 translator=None, back_translator=None, ocr_engine=None):
        self.cfg = cfg or AppConfig()
        if translator is None:
            from ..mt.translator import load_translator
            translator = load_translator(self.cfg.mt, prefer="transformer" if load_model else "dictionary")
        self.translator = translator
        self.ocr_engine = ocr_engine
        self.back_translator = back_translator
        if (self.back_translator is None and load_model and self.cfg.agent.verify_enabled
                and hasattr(translator, "reversed")):
            try:
                self.back_translator = translator.reversed()
            except Exception as exc:
                logger.info("reverse translator unavailable (%s); D4 round-trip will skip", exc)
        self.brain = LLMBrain(self.cfg.agent)
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.job_log_path) if self.cfg.serving.log_jobs else None

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def run(self, *, image=None, pdf_path: str = "", spec: dict = None, text: str = None,
            filename: str = "", mode: str = "", render_path: str = "", save: bool = True) -> JobState:
        job = JobState(src_lang=self.cfg.mt.src_lang, tgt_lang=self.cfg.mt.tgt_lang,
                       filename=filename, mode=mode or self.cfg.render.mode)
        job._image = image          # type: ignore[attr-defined]
        job._pdf_path = pdf_path or None  # type: ignore[attr-defined]
        job._spec = spec            # type: ignore[attr-defined]
        job._text = text            # type: ignore[attr-defined]

        if image is None and not pdf_path and spec is None and text is None:
            job.status = JobStatus.FAILED
            return job

        t0 = time.perf_counter()
        job = self._step(job, "ingest", lambda: tools.tool_ingest(job, self.cfg),
                         summary="input router + quality (D1)")
        if job.status is not JobStatus.FAILED:
            job = self._step(job, "ocr", lambda: tools.tool_ocr(job, self.cfg, ocr_engine=self.ocr_engine),
                             summary="born-digital/scanned + OCR (D2)")
            job = self._step(job, "translate", lambda: tools.tool_translate(job, self.cfg, translator=self.translator),
                             summary="OCR-confidence gate + translate (D3)")
            job = self._step(job, "verify", lambda: tools.tool_verify(job, self.cfg, back_translator=self.back_translator),
                             summary="round-trip verify (D4)")
            job = self._step(job, "render", lambda: tools.tool_render(job, self.cfg, render_path=render_path or None),
                             summary="render-fit gate + assemble (D5)")

        if self.brain.available() and job.translated_text:
            note = self.brain.consistency_note(job.src_lang, job.tgt_lang, job.translated_text)
            if note:
                job.metrics["brain_note"] = note
                job.metrics["brain_used"] = True

        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        for attr in ("_image", "_pdf_path", "_spec", "_text", "_pages", "_blocks", "_page_image", "_n_bad"):
            if hasattr(job, attr):
                delattr(job, attr)
        if save and self._log is not None:
            try:
                self._log.log("image_translation", input_kind=job.input_kind, src_lang=job.src_lang,
                              tgt_lang=job.tgt_lang, status=job.status.value, needs_review=job.needs_review,
                              n_translatable=job.n_translatable, ocr_engine=job.ocr_engine,
                              fit_rate=job.fit_rate, metrics=job.metrics)
            except Exception:
                pass
        return job

    def translate_image(self, image, *, mode: str = "", render_path: str = "") -> dict:
        job = self.run(image=image, mode=mode, render_path=render_path, save=False)
        return self._summary(job)

    def translate_spec(self, spec: dict, *, mode: str = "text_only", render_path: str = "") -> dict:
        job = self.run(spec=spec, mode=mode, render_path=render_path, save=False)
        return self._summary(job)

    def translate_text(self, text: str, *, mode: str = "text_only") -> dict:
        job = self.run(text=text, mode=mode, save=False)
        return self._summary(job)

    @staticmethod
    def _summary(job: JobState) -> dict:
        return {"translated_text": job.translated_text, "translated_markdown": job.translated_markdown,
                "source_text": job.source_text, "fit_rate": job.fit_rate,
                "render_mode_used": job.render_mode_used, "rendered_path": job.rendered_path,
                "needs_review": job.needs_review, "ocr_engine": job.ocr_engine,
                "model_version": job.model_versions.get("mt", "?"), "status": job.status.value}


_AGENT: Optional[ImgTransAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> ImgTransAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = ImgTransAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["ImgTransAgent", "get_agent"]
