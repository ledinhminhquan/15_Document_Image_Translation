"""Agent tools - each operates on the JobState and returns it.

Tools wrap the OCR front-end + layout + the MT core + the overlay renderer + the
D1-D5 policy. They run against the offline stack (SeedEngine reading the embedded gold
spec / a raw-text splitter + dictionary MT) so the whole pipeline runs offline for
tests/CI, and upgrade to Tesseract + a fine-tuned seq2seq when present. The
orchestrator wraps each call with timing/trace; tools never raise past it.

Private artifacts carried on the JobState between tools:
  job._image (PIL) | job._pdf_path | job._spec (dict) | job._text  -> inputs
  job._pages (List[PageInput]) | job._blocks (List[TextBlock]) | job._page_image  -> derived
"""

from __future__ import annotations

from typing import List, Optional

from ..config import AppConfig
from ..imaging import layout as layoutmod
from ..imaging import render as rendermod
from ..logging_utils import get_logger
from ..models.ocr_engine import load_ocr_engine
from ..training.metrics import chrf
from . import policy
from .state import Decision, JobState, JobStatus

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Block builders for the no-image input kinds
# ─────────────────────────────────────────────────────────────────────────────
def _blocks_from_spec(spec: dict) -> List[layoutmod.TextBlock]:
    blocks: List[layoutmod.TextBlock] = []
    for i, blk in enumerate(spec.get("blocks", [])):
        bbox = tuple(blk.get("bbox", [40, 40 + i * 60, 600, 40]))
        blocks.append(layoutmod.TextBlock(text=blk.get("text", ""), bbox=bbox,
                                          kind=blk.get("kind", "paragraph"), conf=0.99,
                                          reading_index=i))
    return blocks


def _blocks_from_text(text: str, width: int = 1000) -> List[layoutmod.TextBlock]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    blocks: List[layoutmod.TextBlock] = []
    y = 40
    idx = 0
    for ln in lines:
        if not ln:
            y += 24
            continue
        w = min(width - 80, 40 + len(ln) * 11)
        blocks.append(layoutmod.TextBlock(text=ln, bbox=(40, y, w, 44), kind="paragraph",
                                          conf=0.99, reading_index=idx))
        y += 64
        idx += 1
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# D1 - ingest + input routing + page-quality gate
# ─────────────────────────────────────────────────────────────────────────────
def tool_ingest(job: JobState, cfg: AppConfig) -> JobState:
    spec = getattr(job, "_spec", None)
    text = getattr(job, "_text", None)
    pdf_path = getattr(job, "_pdf_path", None)
    image = getattr(job, "_image", None)

    if spec is not None:
        job.input_kind = "spec"
        job._blocks = _blocks_from_spec(spec)  # type: ignore[attr-defined]
        job._page_image = getattr(job, "_image", None)  # type: ignore[attr-defined]
        job.born_digital = True
        job.page_quality = 1.0
        branch = "spec"
    elif text is not None:
        job.input_kind = "text"
        job._blocks = _blocks_from_text(text, cfg.data.image_width)  # type: ignore[attr-defined]
        job._page_image = None  # type: ignore[attr-defined]
        job.page_quality = 1.0
        branch = "text"
    elif pdf_path:
        job.input_kind = "pdf"
        job._pages = layoutmod.ingest(pdf_path, cfg)  # type: ignore[attr-defined]
        job.born_digital = bool(job._pages) and all(p.born_digital for p in job._pages)
        job._page_image = next((p.image for p in job._pages if p.image is not None), None)  # type: ignore[attr-defined]
        branch = "pdf"
    elif image is not None:
        job.input_kind = "image"
        job._pages = layoutmod.ingest_image(image, cfg)  # type: ignore[attr-defined]
        job._page_image = image  # type: ignore[attr-defined]
        try:
            from ..imaging.preprocess import preprocess_image
            _, qm = preprocess_image(image, cfg.preprocess)
            job.page_quality = qm.get("quality")
        except Exception:
            job.page_quality = None
        branch = "image"
    else:
        job.status = JobStatus.FAILED
        job.add_decision(Decision("D1", "input_router", "no_input", detail="no image/pdf/spec/text"))
        return job

    qgate = policy.quality_gate(job.page_quality, cfg.agent)
    if not qgate["ok"]:
        job.needs_review = True
    sub = "" if qgate["branch"] in ("ok", "no_quality_check") else f"/{qgate['branch']}"
    job.add_decision(Decision("D1", "input_router", branch + sub, score=job.page_quality,
                              detail=f"kind={job.input_kind}, quality={job.page_quality}"))
    job.status = JobStatus.INGESTED
    return job


# ─────────────────────────────────────────────────────────────────────────────
# D2 - born-digital vs scanned routing + OCR
# ─────────────────────────────────────────────────────────────────────────────
def tool_ocr(job: JobState, cfg: AppConfig, *, ocr_engine=None) -> JobState:
    if job.input_kind in ("spec", "text"):
        job.ocr_engine = job.input_kind
        job.n_blocks = len(getattr(job, "_blocks", []))
        job.mean_ocr_conf = 1.0
        job.add_decision(Decision("D2", "ocr_router", "no_ocr_" + job.input_kind, score=job.n_blocks,
                                  detail=f"blocks={job.n_blocks} (ground-truth text layer)"))
        job.status = JobStatus.OCR_DONE
        return job

    pages = getattr(job, "_pages", [])
    blocks: List[layoutmod.TextBlock] = []
    engine_name = ""
    confs: List[float] = []
    n_born = 0
    for page in pages:
        if page.born_digital and page.digital_blocks:
            pb = layoutmod.reading_order(page.digital_blocks, page.width, cfg.layout)
            pb = layoutmod.classify_blocks(pb, page.width, page.height, cfg.layout)
            engine_name = engine_name or "born_digital"
            n_born += 1
        else:
            eng = ocr_engine or load_ocr_engine(cfg.ocr, image=page.image)
            ocr = eng.recognize(page.image)
            engine_name = eng.name
            page.width = page.width or ocr.width
            page.height = page.height or ocr.height
            pb = layoutmod.words_to_blocks(ocr, cfg.layout)
            pb = layoutmod.reading_order(pb, page.width, cfg.layout)
            pb = layoutmod.classify_blocks(pb, page.width, page.height, cfg.layout)
            confs.append(ocr.mean_conf)
        blocks.extend(pb)

    job._blocks = blocks  # type: ignore[attr-defined]
    job.ocr_engine = engine_name or "stub"
    job.n_blocks = len(blocks)
    job.mean_ocr_conf = round(sum(confs) / len(confs), 4) if confs else (1.0 if n_born else 0.0)
    branch = "born_digital" if (pages and n_born == len(pages)) else ("scanned" if confs else "mixed")
    job.add_decision(Decision("D2", "ocr_router", branch, score=job.mean_ocr_conf,
                              detail=f"engine={job.ocr_engine}, blocks={job.n_blocks}, mean_conf={job.mean_ocr_conf}"))
    job.status = JobStatus.OCR_DONE
    return job


# ─────────────────────────────────────────────────────────────────────────────
# D3 - per-block OCR-confidence gate + translate (with per-block retry)
# ─────────────────────────────────────────────────────────────────────────────
def tool_translate(job: JobState, cfg: AppConfig, *, translator) -> JobState:
    blocks = getattr(job, "_blocks", [])
    keep: List[layoutmod.TextBlock] = []
    skipped = 0
    for b in blocks:
        if b.kind == "blank":
            b.status = "skipped"
            continue
        gate = policy.block_translatable(b.text, b.conf, cfg.agent)
        if gate["ok"]:
            keep.append(b)
        else:
            b.status = "skipped_lowconf" if gate["branch"] == "low_confidence" else "skipped"
            b.translation = ""   # leave the source pixels in place on overlay
            skipped += 1

    hyps = translator.translate_batch([b.text for b in keep]) if keep else []
    n_retrans = 0
    n_bad = 0
    for b, h in zip(keep, hyps):
        if not policy.translate_sane(b.text, h, cfg.agent)["ok"] and cfg.agent.max_retranslate > 0:
            retry = translator.translate(b.text)
            n_retrans += 1
            if policy.translate_sane(b.text, retry, cfg.agent)["ok"]:
                h = retry
        b.translation = h
        b.status = "translated"
        if not policy.translate_sane(b.text, h, cfg.agent)["ok"]:
            n_bad += 1
            b.status = "needs_review"

    job._n_bad = n_bad  # type: ignore[attr-defined]
    job.n_translatable = len(keep)
    job.n_skipped_lowconf = skipped
    job.n_retranslated = n_retrans
    job.model_versions["mt"] = f"{getattr(translator, 'name', '?')}:{getattr(translator, 'version', '?')}"
    job.add_decision(Decision("D3", "ocr_confidence_gate", "ok" if skipped == 0 else "some_skipped",
                              score=job.n_translatable,
                              detail=f"translatable={len(keep)}, skipped={skipped}, retranslated={n_retrans}"))
    job.status = JobStatus.TRANSLATED
    return job


# ─────────────────────────────────────────────────────────────────────────────
# D4 - translation verification (round-trip back-translation chrF + sanity)
# ─────────────────────────────────────────────────────────────────────────────
def tool_verify(job: JobState, cfg: AppConfig, *, back_translator=None) -> JobState:
    blocks = getattr(job, "_blocks", [])
    n_bad = getattr(job, "_n_bad", 0)
    verify_chrf = None
    if cfg.agent.verify_enabled and back_translator is not None:
        sample = [b for b in blocks if b.status == "translated" and b.translation][:20]
        if sample:
            try:
                backs = back_translator.translate_batch([b.translation for b in sample])
                srcs = [b.text for b in sample]
                verify_chrf = round(chrf(backs, srcs) / 100.0, 4)
            except Exception as exc:
                logger.info("verify back-translation skipped (%s)", exc)
    job.verify_chrf = verify_chrf
    gate = policy.verify_gate(verify_chrf, n_bad, cfg.agent)
    if not gate["ok"]:
        job.needs_review = True
    job.add_decision(Decision("D4", "verify_gate", gate["branch"], score=verify_chrf,
                              detail=f"verify_chrf={verify_chrf}, sanity_failures={n_bad}"))
    return job


# ─────────────────────────────────────────────────────────────────────────────
# D5 - render-fit feasibility gate + assemble outputs
# ─────────────────────────────────────────────────────────────────────────────
def tool_render(job: JobState, cfg: AppConfig, *, render_path: Optional[str] = None) -> JobState:
    blocks = getattr(job, "_blocks", [])
    fidelity = rendermod.layout_fidelity(blocks, cfg.render)
    job.fidelity = fidelity
    job.fit_rate = fidelity.get("fit_rate")

    gate = policy.render_gate(job.fit_rate, job.mode, cfg.agent)
    mode_used = gate.get("mode", job.mode)
    job.render_mode_used = mode_used

    page_image = getattr(job, "_page_image", None)
    if render_path and mode_used != "text_only" and page_image is not None:
        try:
            if mode_used == "overlay":
                out_img, _ = rendermod.render_overlay(page_image, blocks, cfg.render)
            else:
                out_img = rendermod.render_side_by_side(page_image, blocks, cfg.render)
            rendermod.save_image(out_img, render_path)
            job.rendered_path = render_path
        except Exception as exc:
            logger.info("overlay render skipped (%s)", exc)

    asm = layoutmod.assemble(blocks)
    job.source_text = asm["source_text"]
    job.translated_text = asm["translated_text"]
    job.source_markdown = asm["source_markdown"]
    job.translated_markdown = asm["translated_markdown"]
    job.blocks = asm["blocks"]

    job.add_decision(Decision("D5", "render_gate", gate["branch"], score=job.fit_rate,
                              detail=f"mode={mode_used}, fit_rate={job.fit_rate}, "
                                     f"mean_shrink={fidelity.get('mean_shrink')}"))
    job.status = JobStatus.NEEDS_REVIEW if job.needs_review else JobStatus.COMPLETED
    if not job.rationale:
        job.rationale = (f"Translated {job.src_lang}->{job.tgt_lang} document image "
                         f"({job.n_translatable} blocks via {job.model_versions.get('mt', 'MT')}); "
                         f"OCR={job.ocr_engine}, fit_rate={job.fit_rate}, mode={mode_used}, "
                         f"needs_review={job.needs_review}.")
    return job


__all__ = ["tool_ingest", "tool_ocr", "tool_translate", "tool_verify", "tool_render"]
