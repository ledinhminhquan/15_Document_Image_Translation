"""Generate the submission report.pdf for the imgtrans document-image-translation system.

A 10-15 page report covering every Section-I deliverable: problem & use cases, data & the
synthetic generator (license flags), the OCR->MT->render pipeline + models, the trainable MT
core, the agent (D1-D5), the evaluation (MT chrF/BLEU + OCR CER/WER + end-to-end chrF + layout
fidelity), deployment, continual learning & monitoring, privacy & robustness, and ethics. Live
numbers come from ``run_dir()`` artifacts; missing metrics degrade to placeholders. reportlab
lazy-imported; a Markdown fallback is written if absent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso
from . import charts as charts_mod
from .artifact_loader import (base_model, e2e_metric, has_eval, headline, latency, load_artifacts,
                              model_system_name, model_version, mt_metric, ocr_metric, read_doc)

logger = get_logger(__name__)

_SUBTITLE = ("Document-image machine translation: read the text inside an image / scanned document "
             "with an OCR front-end (Tesseract), translate it with a TRAINABLE machine-translation "
             "core (fine-tuned m2m100), then render the translation BACK onto the page preserving "
             "spatial layout (overlay). A deterministic agent (D1-D5) gates OCR confidence, "
             "translation quality and overlay fit. Default en->fr.")

_SECTIONS = [
    ("1. Problem Definition & Use Cases", "problem_definition.md"),
    ("2. Data & The Synthetic Generator", "data_description.md"),
    ("3. The OCR -> MT -> Render Pipeline", "architecture.md"),
    ("4. Model Selection (MT core + OCR)", "model_selection.md"),
    ("5. Agent Architecture (Decisions D1-D5)", "agent_architecture.md"),
    ("6. Evaluation & Layout Fidelity", "layout_fidelity_evaluation.md"),
    ("7. Deployment", "deployment.md"),
    ("8. Continual Learning & Monitoring", "continual_learning_monitoring.md"),
    ("9. Data Privacy & Robustness", "privacy_robustness.md"),
    ("10. Ethics & Responsible AI", "ethics_statement.md"),
]


def _builtin_sections(cfg: AppConfig, arts: Dict[str, Any]) -> Dict[str, str]:
    e2e = e2e_metric(arts, "chrf")
    cer = ocr_metric(arts, "cer")
    fit = headline(arts, "mean_fit_rate")
    if e2e is not None:
        res_line = (f"In the latest eval the end-to-end image-translation chrF is **{e2e:.1f}**"
                    + (f", OCR CER **{cer:.3f}**" if cer is not None else "")
                    + (f", overlay fit-rate **{fit:.2f}**." if fit is not None else "."))
    else:
        res_line = "Run `imgtrans evaluate` to populate the live numbers here."
    return {
        "problem_definition.md": f"""
## What it does
Given a **document image** (a photo, a scan, a screenshot, or a born-digital PDF), read the
text inside it, **translate** it ({cfg.mt.src_lang} -> {cfg.mt.tgt_lang}, configurable), and
render the translation **back onto the page** preserving the spatial layout (overlay), like the
Google Translate camera. Cascade: **OCR -> MT -> layout-preserving render**. The only trainable
component is the MT model (the NLP heart); OCR + layout + render are pretrained/algorithmic.

## The job-to-be-done
- **Travelers / accessibility** - translate signage, menus, forms from a photo.
- **Back-office / legal / finance** - translate scanned contracts, invoices, letters.
- **Localization** - translate screenshots, comics/manga, infographics keeping the layout.

## Why it is more than "OCR then translate"
The value-add is the **layout-preserving overlay** (re-fit the longer translation into the
original box with auto font-shrink + wrapping) plus the **confidence/verification gates**:
low-confidence OCR is skipped (not mistranslated), a round-trip back-translation flags doubtful
translations, and a fit-feasibility gate falls back to side-by-side when the text cannot fit.

## Success metrics
- **Technical:** MT **chrF (headline)** + BLEU; OCR **CER** + WER; **end-to-end image-translation
  chrF**; **layout fidelity** (overlay fit-rate).
- **Business:** human post-edit rate, the share of pages flagged for review.
{res_line}
""",
        "data_description.md": f"""
## Why synthetic data
**No public in-image / document-image translation benchmark with gold parallel text exists.**
So the PRIMARY data is a reproducible **synthetic generator** (`data/synth_render.py`) that
renders source sentences onto page images with varied fonts/sizes/light degradation and embeds
the gold layout spec (source text, gold translation, boxes) into the PNG metadata - giving
`(image, gold_source, gold_target, boxes)` quadruples that measure OCR CER/WER, MT chrF/BLEU and
the end-to-end chrF at once.

## MT fine-tune corpus
- **`{cfg.data.mt_dataset}`** config **`{cfg.data.mt_config}`** (`translation` {{en, fr}}, 1M
  pairs). **License flag: unknown** (research/educational).

## OCR-noise text source (optional)
- **`{cfg.data.ocr_text_dataset}`** config **`{cfg.data.ocr_text_config}`** (**CC0**) - real
  OCR-degraded / corrected text for OCR-robustness slices.

## Offline backbone
Built-in **synthetic seed pages** + an en->fr **dictionary** ship in `data/samples.py`, and the
`SeedEngine` reads the embedded gold spec, so the entire pipeline runs with **no network, no
tesseract, no torch**.
""",
        "model_selection.md": f"""
## The trainable MT core (reused from P13/P14)
- **Base:** `{base_model(arts)}` (default `facebook/m2m100_418M`, **MIT**, many-to-many ~100
  languages). Alternatives: `Helsinki-NLP/opus-mt-en-fr` (Apache, en->fr only, T4 fallback),
  `facebook/mbart-large-50-many-to-many-mmt` (MIT, H100 upgrade), `facebook/nllb-200-distilled-600M`
  (**CC-BY-NC, flagged - not shipped**).
- **Baselines:** zero-shot base MT (floor to beat), a **dictionary** word-lookup (offline floor +
  fallback), and an **identity** copy-source floor.

## The OCR front-end (pretrained, NOT trained)
- **Tesseract** via `pytesseract` (**Apache-2.0**) - `image_to_data` gives the word boxes + conf
  the overlay step needs; PyMuPDF routes born-digital vs scanned; `SeedEngine` is the offline
  fallback. Alternatives: docTR / PaddleOCR / EasyOCR (Apache, pip libs), `microsoft/trocr-base-printed`
  (MIT neural upgrade), Surya (**CC-BY-NC-SA, flagged**). End-to-end OCR-VLMs (`stepfun-ai/GOT-OCR-2.0-hf`,
  `google/pix2struct-base`, both Apache) are documented alternatives, not the chosen cascade core.

## Optimization
HF `Seq2SeqTrainer`, {cfg.mt.num_train_epochs} epochs, lr {cfg.mt.learning_rate:g}, label
smoothing {cfg.mt.label_smoothing}, `predict_with_generate` ({cfg.mt.num_beams} beams), bf16+tf32
on Ampere+/H100, fp16 on T4; selected on **chrF**; early stopping.
{res_line}
""",
        "agent_architecture.md": f"""
## FSM
A deterministic finite-state machine; every tool returns a uniform dict and every transition is
logged to a trace. States: `ingest -> ocr -> translate -> verify -> render`. An optional LLM
**brain** (`{cfg.agent.llm_model}`, OFF by default) only writes an advisory note; rules win and
the agent runs with **zero paid API calls**.

## Five decisions (each acts on an intermediate artifact)
- **D1 - input router + page-quality gate.** Route image / PDF / spec / raw-text; for photos,
  estimate blur/contrast/ink (quality < {cfg.agent.quality_min} -> flag).
- **D2 - born-digital vs scanned.** A born-digital PDF text layer skips OCR entirely; scanned
  pages go through the OCR engine.
- **D3 - per-block OCR-confidence gate.** Blocks below conf {cfg.agent.ocr_confidence_min} (or too
  short) are **skipped, not mistranslated** - the source pixels stay in place.
- **D4 - translation verification.** Round-trip back-translation chrF (< {cfg.agent.verify_min_chrf}
  -> flag) + a length-ratio sanity check; re-translate (budget {cfg.agent.max_retranslate}).
- **D5 - render-fit feasibility gate.** Compute the overlay fit-rate; below
  {cfg.agent.min_fit_rate} the agent falls back from overlay to **side-by-side**; low overall
  confidence -> **needs_review**.

The agent emits `{{translated_text, blocks, fit_rate, rendered_path, needs_review, decisions[],
trace[]}}`. Low-confidence / poor-fit output is **flagged for human review**.
""",
        "layout_fidelity_evaluation.md": f"""
## Four measurements
- **MT** (the trainable core): **chrF (headline)** + BLEU on held-out pairs (model vs dictionary
  vs identity floor).
- **OCR** (front-end): **CER (headline)** + WER vs the gold source text (perfect-OCR offline via
  `SeedEngine`; realistic with Tesseract on Colab).
- **End-to-end**: the headline - run the full agent (OCR -> MT) on each rendered page and score
  the assembled translation against the gold target with **chrF** + BLEU.
- **Layout fidelity**: the mean overlay **fit-rate** (fraction of blocks whose translation fits
  the source box) + mean shrink + overflow.
{res_line}
""",
    }


def _esc(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b\x00").replace("</b>", "\x00/b\x00")
    s = s.replace("<font face='Courier'>", "\x00f\x00").replace("</font>", "\x00/f\x00")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = (s.replace("\x00b\x00", "<b>").replace("\x00/b\x00", "</b>")
          .replace("\x00f\x00", "<font face='Courier'>").replace("\x00/f\x00", "</font>"))
    return s


def _md_to_flowables(md: str, styles, max_lines: int = 300):
    from reportlab.platypus import Paragraph, Preformatted, Spacer
    flow, lines, in_code, code, bullet = [], md.splitlines()[:max_lines], False, [], []

    def flush():
        nonlocal bullet
        for b in bullet:
            flow.append(Paragraph("- " + _esc(b), styles["Body"]))
        bullet = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code), styles["Code"])); code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(ln); continue
        s = ln.rstrip()
        if not s:
            flush(); flow.append(Spacer(1, 5)); continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            flow.append(Paragraph(_esc(s.lstrip("#").strip()), styles["H2" if level <= 2 else "H3"]))
        elif s.lstrip().startswith(("- ", "* ")):
            bullet.append(s.lstrip()[2:])
        else:
            flush(); flow.append(Paragraph(_esc(s), styles["Body"]))
    flush()
    return flow


def _results_tables(arts: Dict[str, Any], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    model_name = model_system_name(arts) or "model"
    flow = [Paragraph("Results - MT quality (held-out pairs)", styles["H3"])]
    rows = [["Metric", "identity floor", "dictionary", model_name]]
    if has_eval(arts):
        for key, label in [("chrf", "chrF ^"), ("bleu", "BLEU ^")]:
            iv = mt_metric(arts, "identity", key)
            dv = mt_metric(arts, "dictionary", key)
            mv = mt_metric(arts, model_name, key)
            rows.append([label, f"{iv:.1f}" if iv is not None else "-",
                         f"{dv:.1f}" if dv is not None else "-",
                         f"{mv:.1f}" if mv is not None else "-"])
    else:
        rows.append(["run `evaluate`", "-", "-", "-"])
    t = Table(rows, hAlign="LEFT", colWidths=[120, 100, 100, 110])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    flow += [t, Spacer(1, 8)]

    flow.append(Paragraph("Results - OCR, end-to-end & layout fidelity", styles["H3"]))
    drows = [["Metric", "value"],
             ["End-to-end image-translation chrF ^", _fmt(e2e_metric(arts, "chrf"), 1)],
             ["End-to-end BLEU ^", _fmt(e2e_metric(arts, "bleu"), 1)],
             ["OCR CER (lower better) v", _fmt(ocr_metric(arts, "cer"), 3)],
             ["OCR WER (lower better) v", _fmt(ocr_metric(arts, "wer"), 3)],
             ["Overlay fit-rate ^", _fmt(headline(arts, "mean_fit_rate"), 3)]]
    dt = Table(drows, hAlign="LEFT", colWidths=[300, 120])
    dt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f855a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf5ee")])]))
    flow += [dt, Spacer(1, 6),
             Paragraph("A translation that wins chrF but overflows its box is a poor overlay, so "
                       "translation quality and layout fidelity are reported jointly.", styles["Body"])]
    lat = latency(arts, "p50")
    if lat is not None:
        flow.append(Paragraph(f"Agent latency: per-page p50 ~ {lat:.0f} ms "
                              f"(p95 ~ {latency(arts, 'p95') or 0:.0f} ms).", styles["Body"]))
    flow.append(Spacer(1, 8))
    return flow


def _fmt(v, nd):
    return (f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "-")


def generate_report(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    title = title or cfg.project_title
    author = author or cfg.author
    arts = load_artifacts(cfg)
    out = Path(out_path) if out_path else run_dir() / "report" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    builtins = _builtin_sections(cfg, arts)

    def section_md(fname: str) -> str:
        doc = read_doc(fname)
        if doc.strip():
            lines = doc.splitlines()
            return "\n".join(lines[:46]) if len(lines) > 46 else doc
        return builtins.get(fname, "")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer)
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("T", parent=base["Title"], fontSize=22, leading=26),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], textColor="#1a365d", spaceBefore=10),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], textColor="#2b6cb0"),
        "Body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13),
        "Code": ParagraphStyle("C", parent=base["Code"], fontSize=7.5, leading=9, backColor="#f4f6f8"),
        "Meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=11, leading=15),
    }
    try:
        built = dict(charts_mod.build_all(arts, out.parent / "charts"))
    except Exception as exc:
        logger.info("charts skipped (%s)", exc)
        built = {}

    story: List[Any] = [
        Spacer(1, 5 * cm), Paragraph(title, styles["Title"]), Spacer(1, 1 * cm),
        Paragraph(f"<b>{author}</b> - Student {cfg.student_id}", styles["Meta"]),
        Paragraph("NLP in Industry - Final Assignment (P15)", styles["Meta"]),
        Paragraph(_SUBTITLE, styles["Meta"]),
        Paragraph(f"Generated {utc_now_iso()}", styles["Body"]),
        Paragraph(f"MT core: <b>{model_version(arts)}</b> (base {base_model(arts)})", styles["Body"]),
    ]
    story.append(PageBreak())
    story += _results_tables(arts, styles)
    for name in ("quality", "e2e", "buckets"):
        if name in built:
            story += [Image(str(built[name]), width=13 * cm, height=7.0 * cm), Spacer(1, 6)]
    story.append(PageBreak())

    for heading, fname in _SECTIONS:
        story.append(Paragraph(heading, styles["H2"]))
        story += _md_to_flowables(section_md(fname), styles)
        story.append(Spacer(1, 10))

    try:
        SimpleDocTemplate(str(out), pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                          leftMargin=1.8 * cm, rightMargin=1.8 * cm, title=title, author=author).build(story)
    except Exception as exc:
        logger.warning("reportlab build failed (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)
    logger.info("Report -> %s", out)
    return str(out)


__all__ = ["generate_report"]
