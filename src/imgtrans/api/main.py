"""FastAPI service for the document-image machine-translation system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /translate-text``   - {text, mode?} -> translated text blocks (no OCR; demo/offline)
* ``POST /translate-image``  - upload an image / PDF -> translated text (+ optional rendered overlay PNG, base64)

The upload route is registered only when ``python-multipart`` is importable (so ``api.main``
imports anywhere). Low-confidence OCR is skipped; low-confidence / poor-fit output is flagged.
"""

from __future__ import annotations

import base64
import io
import tempfile
from importlib.util import find_spec

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import HealthResponse, TranslateResponse, TranslateTextRequest

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)

_HAS_MULTIPART = find_spec("multipart") is not None or find_spec("python_multipart") is not None


def _resp(sd: dict, rendered_b64: str = None) -> TranslateResponse:
    return TranslateResponse(
        input_kind=sd["input_kind"], src_lang=sd["src_lang"], tgt_lang=sd["tgt_lang"],
        source_text=sd["source_text"], translated_text=sd["translated_text"],
        translated_markdown=sd["translated_markdown"], status=sd["status"], ocr_engine=sd["ocr_engine"],
        n_blocks=sd["n_blocks"], n_translatable=sd["n_translatable"],
        n_skipped_lowconf=sd["n_skipped_lowconf"], mean_ocr_conf=sd["mean_ocr_conf"],
        fit_rate=sd["fit_rate"], render_mode_used=sd["render_mode_used"], verify_chrf=sd["verify_chrf"],
        needs_review=sd["needs_review"], rationale=sd["rationale"], rendered_image_b64=rendered_b64,
        blocks=sd["blocks"], decisions=sd["decisions"], metrics=sd["metrics"],
        model_versions=sd["model_versions"])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", mt=getattr(agent.translator, "name", "?"),
                          ocr=cfg.ocr.engine, direction=f"{cfg.mt.src_lang}->{cfg.mt.tgt_lang}",
                          version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "mt": getattr(agent.translator, "version", "?"),
            "direction": f"{cfg.mt.src_lang}->{cfg.mt.tgt_lang}", "model_version": cfg.serving.model_version}


@app.post("/translate-text", response_model=TranslateResponse)
def translate_text(req: TranslateTextRequest) -> TranslateResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="provide text to translate")
    job = get_agent().run(text=req.text, mode="text_only", save=True)
    return _resp(job.to_dict())


if _HAS_MULTIPART:
    from fastapi import File, Form, UploadFile

    @app.post("/translate-image", response_model=TranslateResponse)
    def translate_image(file: "UploadFile" = File(...), mode: str = Form("overlay")) -> TranslateResponse:
        raw = file.file.read()
        name = (file.filename or "upload").lower()
        agent = get_agent()
        rendered_b64 = None
        if name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf.write(raw)
                tmp = tf.name
            job = agent.run(pdf_path=tmp, filename=file.filename or "", mode=mode, save=True)
        else:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(raw))
            except Exception:
                raise HTTPException(status_code=422, detail="could not read image")
            render_path = ""
            if mode in ("overlay", "side_by_side"):
                render_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            job = agent.run(image=img, filename=file.filename or "", mode=mode,
                            render_path=render_path, save=True)
            if job.rendered_path:
                try:
                    with open(job.rendered_path, "rb") as fh:
                        rendered_b64 = base64.b64encode(fh.read()).decode("ascii")
                except Exception:
                    rendered_b64 = None
        return _resp(job.to_dict(), rendered_b64=rendered_b64)
else:  # pragma: no cover
    logger.info("python-multipart not installed; /translate-image disabled (/translate-text still works).")


__all__ = ["app"]
