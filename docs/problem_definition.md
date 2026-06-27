# P15 — Problem Definition: Document-Image Machine Translation

> Project: **P15 Document-Image Machine Translation** (package `imgtrans`, folder `15_Document_Image_Translation`)
> Author: **Le Dinh Minh Quan**, student **23127460**
> Default direction: **en → fr** (configurable; the trainable core `facebook/m2m100_418M` is many-to-many)
> Companion docs: [`DESIGN_BRIEF.md`](DESIGN_BRIEF.md) (full technical spec)

---

## 1. The problem in one sentence

**Translate the text that appears *inside* an image, a scanned document, or a born-digital PDF, and render the translation *back onto the page* while preserving the original spatial layout** — the "Google Translate camera" experience for documents.

The input is **pixels (or a page)**, not a clean string. The output is a **new image** (plus the extracted/translated text) in which every block of source-language text has been replaced, in place, by its target-language translation, fitted into the box where the original text sat. This is fundamentally different from plain document translation (text in → text out): here the system must *read* the page, *translate* what it read, and *redraw* the result so the page still looks like the original — only in another language.

We solve this with a **cascade** of three stages, of which **only the middle stage is trained**:

```
image / PDF ──▶ [OCR front-end] ──▶ source text + block boxes + per-word confidence
  (pretrained,        │
   algorithmic)        ▼
                    [MT core]   ◀── the ONLY trained component (m2m100_418M, fine-tuned)
                       │
                       ▼
            translated text ──▶ [layout-preserving render/overlay] ──▶ output image
                                       (pure-Pillow algorithm)
```

- **OCR front-end** — pretrained Tesseract (via `pytesseract`, `image_to_data`) gives word/block boxes + confidence; a PyMuPDF router decides born-digital vs scanned; an offline `SeedEngine` reads the gold spec embedded in synthetic images. **Not trained.**
- **MT core** — `facebook/m2m100_418M` (MIT), fine-tuned with the Hugging Face `Seq2SeqTrainer` (headline metric **chrF**). **The only trained stage.**
- **Render / overlay** — a PIL-only fit-to-box engine (binary-search font size + greedy word-wrap, script-aware Noto/DejaVu fonts) that erases the source text and draws the translation back in place. **Pure algorithm, not trained.**

Wrapping the cascade is a **deterministic 5-decision agent** (the mandatory agentic component) that gates on each stage's own signals and degrades gracefully — `overlay` → `side_by_side` → `needs_review` — instead of emitting a broken page.

---

## 2. Why this matters — real-world use cases

In-image translation is one of the most-used "AI in the wild" features (phone camera translate, manga scanlation, document-localization pipelines), yet it is rarely treated as an end-to-end engineering problem with honest metrics. Concrete settings where P15's task appears:

- **Travel signage and menus.** A photo of a street sign, restaurant menu, transit board, or product label. Text is short, high-contrast, but on cluttered/colored backgrounds and often rotated — the canonical "point your camera" case.
- **Scanned contracts and legal documents.** A scanned (image-only) PDF of a contract, agreement, or certificate that must be read in another language. The layout (clauses, headers, signature blocks) carries meaning, so the translation must stay *in place* rather than collapse to a flat text dump. These pages frequently contain **PII** (names, IDs, addresses) — see §8.
- **Forms and official paperwork.** Tax forms, immigration paperwork, medical intake sheets, application forms — field labels must be translated next to the right field, which only works if layout is preserved.
- **Screenshots and UI captures.** A screenshot of an app, error dialog, spreadsheet, or web page where the text is "trapped" in pixels (no selectable text layer). Common in support tickets and bug reports.
- **Manga, comics, and scanlation.** Speech bubbles and captions with text laid out in tight, irregular regions — the archetypal layout-preserving overlay task (and the origin of tools like *manga-image-translator* / *BallonsTranslator* that this pipeline mirrors).
- **Born-digital PDFs.** Slide decks, reports, and brochures exported to PDF that already carry a text layer — here OCR can be *bypassed* entirely (decision **D2**), giving zero OCR error.

In every case the user wants the *page back*, translated — not a disembodied list of strings. That "render it back where it was" requirement is the heart of P15.

---

## 3. Inputs and outputs

### 3.1 Inputs (in scope)

| Input | Handling | Notes |
|-------|----------|-------|
| **Raster image** (`.png`, `.jpg`, `.webp`) | Treated as scanned → full OCR path | The core "camera" case. |
| **PDF** (born-digital) | PyMuPDF extracts the embedded text layer + boxes → **OCR bypassed** (D2) | Lossless, zero OCR error. |
| **PDF** (scanned / image-only) | Rasterized per page → OCR path like an image | OCR error applies. |
| **Plain `.txt`** | Degenerate input that **skips OCR**, jumps straight to MT + (text) render | Lets the MT core be exercised directly. |
| **Synthetic spec / manifest** (`SeedEngine`) | Offline path: gold source text + boxes read from the embedded spec | Powers reproducible, dependency-free tests and CER scoring. |

Default language direction is **en → fr**, configurable to any pair the m2m100 core supports (one fine-tuned checkpoint serves en→fr *and* the multilingual requirement).

### 3.2 Outputs

- **Primary:** a **new image / page** the same size as the input, with each source block erased (whiteout or simple-inpaint) and replaced by the **fitted, wrapped translation** drawn in place (the layout-preserving overlay).
- **Structured side-channel:** the extracted **source text**, the **translated text**, the **block boxes** + per-block OCR confidence, and the chosen **render mode** (`overlay` / `side_by_side` / `needs_review`).
- **Degrade outputs:** when in-place overlay is infeasible, a **side-by-side panel** (original untouched + translated caption beside it), or a **`needs_review`** payload (boxes + raw translation, no destructive render) when OCR/MT confidence is too low to trust.
- **Deployment surface:** FastAPI (`POST /translate-text` JSON; `POST /translate-image` upload → translated text + base64 overlay PNG) and a Gradio `/ui`.

---

## 4. Scope and non-goals

### 4.1 In scope

- The **OCR → MT → render** cascade end to end, with the MT core the single trained component.
- **Layout-preserving overlay** with shrink-to-fit, greedy word-wrap (per-character for CJK), script-aware font selection, and contrast-aware text color.
- A reproducible **synthetic document-image generator** (`data/synth_render.py`) producing `(image, gold_source, gold_target, boxes)` quadruples — the **primary data source** (see §6).
- A deterministic **5-decision agent** with confidence/verification gates and a graceful degradation ladder.
- A full **offline fallback** path (`SeedEngine` OCR + dictionary MT + DejaVu fonts + pure-Python metrics) so tests run with only stdlib + Pillow.
- Honest evaluation: **MT chrF/BLEU**, **OCR CER/WER**, **end-to-end image-translation chrF/BLEU**, and **layout-fidelity** metrics (fit-rate, shrink, overflow).

### 4.2 Non-goals (explicitly out of scope)

- **Training a custom OCR engine or text detector.** OCR is pretrained Tesseract; detection is fused into OCR via `image_to_data`. We do *not* train CRAFT/PaddleDet-style detectors.
- **Learned image inpainting.** Source-text erasure is an **algorithmic** whiteout / horizontal-smear, not a trained inpainting model.
- **End-to-end OCR-VLM image→translated-image models as the shipped core.** `stepfun-ai/GOT-OCR-2.0-hf` and `google/pix2struct-base` are documented as *upgrades / alternatives only* — they are readers, not in-place translators, and are not trained here.
- **Real-time / streaming latency guarantees.** P15 optimizes correctness and layout fidelity, not video-rate throughput.
- **Perfect typographic reproduction** (font matching, kerning, exact color of the original ink). We approximate ink/background from pixels and accept an imperfect but legible re-render.
- **Shipping any non-commercial component.** `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0) and Surya (`vikp/surya_rec2` + `surya_det3`, CC-BY-NC-SA-4.0) are **flagged non-commercial** and documented as research-only — **never the default**. The opus-100 corpus license must be verified per pair before commercial use.

---

## 5. Why a cascade (OCR → MT → render), not end-to-end

An end-to-end image→translated-image model (a single VLM that ingests pixels and emits a translated page) is conceptually appealing, but a cascade is the right choice for P15:

1. **Only one stage needs training.** OCR is a mature pretrained system (Tesseract); rendering is deterministic geometry (Pillow). Isolating MT as the single trainable core means the project trains exactly one model with the proven HF `Seq2SeqTrainer` stack reused from P13/P14 — focused, reproducible, and cheap enough to fine-tune on a free T4.
2. **No end-to-end training data exists.** Dataset research returned **null**: there is *no* public in-image / document-image translation benchmark with gold parallel text. An end-to-end model would have nothing to train or honestly evaluate on. The cascade lets us train MT on ordinary parallel text (opus-100 en-fr) and *synthesize* the image side (§6), which an end-to-end model cannot exploit as cleanly.
3. **Interpretability and graceful failure.** Each stage emits an inspectable intermediate signal — OCR confidence, round-trip back-translation chrF, render fit-rate. The agent gates on these to skip garbage, catch hallucinations, and degrade to side-by-side or `needs_review`. A monolithic model is a black box that fails silently; the cascade fails *visibly and recoverably*.
4. **Component-level metrics.** The cascade lets us measure MT in isolation (chrF on clean gold source), OCR in isolation (CER/WER), and the full pipeline (end-to-end chrF) — and quantify the **OCR-error gap** = clean-MT-chrF − end-to-end-chrF. An end-to-end model collapses these into one opaque number.
5. **Permissive, swappable parts.** Every shipped component is MIT or Apache-2.0, and any stage can be upgraded independently (Tesseract → docTR/TrOCR; m2m100 → mBART; whiteout → simple-inpaint) without retraining the others.

The cost — error propagation from OCR into MT — is real and is mitigated by the per-block **OCR-confidence gate (D3)**, which never sends garbled text to MT, and the **round-trip verification (D4)**, which catches downstream drift.

---

## 6. Data: synthetic-first, because no benchmark exists

Because **no verified public in-image/document-image translation benchmark with gold parallel text exists** (research returned null), the primary data source is a **reproducible synthetic generator** (`data/synth_render.py`). It renders source sentences drawn from the MT corpus onto page images with varied fonts/sizes and mild, OCR-survivable degradation, and **embeds the gold layout spec** (source text, gold translation, boxes). Each sample is a deterministic, seed-reproducible **quadruple**:

> `(image, gold_source_text, gold_target_text, boxes)`

This single artifact enables all four evaluation families: OCR (read image vs `gold_source`), MT (translate `gold_source` vs `gold_target`), end-to-end (translate `OCR(image)` vs `gold_target`), and layout fidelity (rendered boxes vs gold boxes).

| Asset | Identifier | License | Role | Flag |
|-------|-----------|---------|------|------|
| MT fine-tune corpus | `Helsinki-NLP/opus-100` (en-fr, ~1M pairs) | license unknown | Source side rendered onto images; target side = gold translation | **Flag — verify per pair before commercial use** |
| Optional OCR-noise text | `PleIAs/Post-OCR-Correction` (english) | CC0 | Realistic OCR-error text source | clean |
| Offline backbone | built-in synthetic seed pages + en→fr dictionary | in-repo | Fully offline pipeline + tests | clean |

This synthetic-first stance is also the project's **biggest risk** (§10): we must *not* overclaim generalization to photos of real-world signage. The degradation suite (rotation, blur, noise, brightness jitter, JPEG recompression) is the mitigation, and any future `hub_repo_details`-verified real benchmark would be added as a held-out test split.

---

## 7. Success criteria

P15 succeeds when the full cascade runs end to end, the agent degrades gracefully, and the metrics tell an honest story across all four families.

### 7.1 Metric families

- **MT (headline):** **chrF** (primary) + BLEU, computed `MT(gold_source)` vs `gold_target` on clean gold source so MT is isolated from OCR noise.
- **OCR (headline):** **CER** (primary) + WER, computed `OCR(image)` vs `gold_source`.
- **End-to-end (headline):** image-translation **chrF** + BLEU, computed `MT(OCR(rendered_page))` vs `gold_target`. The gap to clean-MT-chrF quantifies the OCR cost.
- **Layout fidelity:** overlay **fit-rate** (fraction of blocks whose translation fits the source box) + **mean shrink** + **overflow** count; complemented by box-retention IoU and no-overlap rate.

### 7.2 Verified offline-seed reference numbers

The offline seed evaluation (`SeedEngine` OCR + dictionary MT + pure-Python metrics) produced:

| Metric | Value | Reading |
|--------|-------|---------|
| MT chrF (dictionary) | **79.9** vs identity floor **22.4** | dictionary MT beats the identity floor by a wide margin |
| OCR CER | **0.0** | perfect via `SeedEngine` (reads embedded gold); realistic CER comes from Tesseract on Colab |
| End-to-end chrF | **76.4** | full pipeline on rendered pages |
| Mean fit-rate | **1.0** | every translation fit its source box |

> **Honesty note:** the dictionary MT **saturates on the seed** because seed (src, tgt) pairs overlap the in-repo dictionary. On real opus-100 eval pairs the fine-tuned `m2m100` is expected to dominate dictionary/identity — that is the **non-saturated honest floor**. Offline tests assert *pipeline wiring and metric plumbing*, not translation quality.

### 7.3 Baselines (required for honest comparison)

1. **Identity** — copy source through unchanged (the chrF floor; near-zero across scripts).
2. **Dictionary MT** — glossary lookup + identity passthrough for OOV (also the offline MT).
3. **Zero-shot m2m100** — the core *without* fine-tuning, to quantify what fine-tuning buys.

The fine-tuned m2m100 must beat all three on real eval pairs for the project to claim value.

---

## 8. Ethics, privacy, and robustness

Document images routinely contain **PII** — IDs, passports, medical and legal documents, forms. P15 is built to assist, not to assert certainty:

- **Privacy by default:** local processing, no raw-image retention by default, explicit consent assumed for any upload.
- **Human-in-the-loop:** the tool **assists** translation and **flags low-confidence output for human review** (the `needs_review` branch). It never claims certainty on a translation it could not verify.
- **No silent garbage:** low-confidence OCR blocks are **skipped, not mistranslated** (D3); MT hallucination/truncation is caught by round-trip + length-ratio checks (D4); poor-fit pages degrade to side-by-side (D5).

**Robustness concerns** explicitly handled: degraded scans, rotation/blur, multi-column layouts, mixed scripts, and **OCR error propagation into MT** (mitigated by the post-OCR confidence gate and the round-trip verification gate).

---

## 9. Assignment mapping

P15 satisfies the production-NLP assignment requirements as follows:

| Requirement | How P15 meets it |
|-------------|------------------|
| **Real, non-trivial NLP task** | In-image / document machine translation with layout-preserving render — a deployed-product-grade task (camera translate, manga translate, doc localization). |
| **A trained model** | `facebook/m2m100_418M` (MIT) fine-tuned with HF `Seq2SeqTrainer`, headline metric chrF — the single trainable stage. |
| **Baselines** | Identity (floor), dictionary MT (offline), zero-shot m2m100 — fine-tune must beat all three. |
| **Honest, multi-faceted metrics** | MT chrF/BLEU, OCR CER/WER, end-to-end image-translation chrF/BLEU, layout fidelity (fit-rate / shrink / overflow / IoU / no-overlap). |
| **Mandatory agentic component** | A deterministic 5-decision FSM (`src/imgtrans/agent/`): D1 input router → D2 born-digital vs scanned → D3 per-block OCR-confidence gate → D4 round-trip + length-ratio verification → D5 render-fit feasibility, with an overlay → side_by_side → needs_review degradation ladder. Optional advisory LLM brain (`anthropic`), OFF by default, never rewrites. |
| **Value-add beyond a naïve script** | Layout-preserving overlay with auto font-fit + the confidence/verification gates — strictly more than "OCR then translate then print the string." |
| **Reproducibility** | Per-index seeded synthetic generator; committed tiny fixtures; fully offline path (`SeedEngine` + dictionary MT) so CI runs on stdlib + Pillow alone. |
| **Deployment** | FastAPI (`/translate-text`, `/translate-image`) + Gradio `/ui` + Docker (tesseract-ocr + libGL) + HF Space. |
| **Ethics & licensing rigor** | PII-aware privacy posture, human-review flagging; every shipped component MIT/Apache-2.0, with non-commercial options (NLLB, Surya) explicitly flagged and excluded. |
| **Reuse / engineering maturity** | OCR plumbing + layout + preprocess ported from P07 dococr; MT translator + chrF/BLEU metrics + config/logging/registry/autoreport/monitoring/grading/autopilot templates from P13/P14. New for P15: the fit-to-box overlay engine, the synthetic doc-image generator, the `SeedEngine` offline OCR, and the 5-decision agent. |

---

## 10. Known risks (carried into the design)

- **No real benchmark** — everything rests on the synthetic generator; do not overclaim generalization to real photos. Mitigated by the degradation suite.
- **Translation expansion breaks overlay** — targets are routinely longer than source (esp. → fr and CJK → EN); the box overflows. This is the *expected* common case, handled by D5's side-by-side branch and the fit-rate metric — not a bug.
- **OCR error propagation** — garbled OCR feeding MT, mitigated by the D3 confidence gate (skip) and D4 round-trip verification (catch).
- **CJK / RTL rendering** — per-character wrap for CJK; RTL shaping is best-effort without optional `python-bidi` / `arabic-reshaper`. Offline-no-Noto renders tofu for CJK/Arabic (pure-offline test mode only).
- **Non-commercial traps** — NLLB (CC-BY-NC-4.0) and Surya (CC-BY-NC-SA-4.0) are tempting for quality; never ship them. Verify each opus-100 pair license.
