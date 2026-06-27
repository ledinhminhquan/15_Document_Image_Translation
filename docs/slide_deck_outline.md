# P15 — Document-Image Machine Translation: Slide Deck Outline

> Presentation outline for **P15 Document-Image Machine Translation** (package `imgtrans`).
> Author: **Le Dinh Minh Quan** — student **23127460**.
> ~12 slides, one slide per section below, each with 3–5 sub-bullets. Speaker notes in italics.
> Default direction **en→fr** (configurable; the m2m100 core is many-to-many).

---

## Slide 1 — Title

- **Document-Image Machine Translation** — translate the text *inside* an image / scanned page / PDF and render the translation **back onto the page**, preserving layout (the "Google Translate camera" experience).
- Author: **Le Dinh Minh Quan**, student **23127460**; package `imgtrans`, repo folder `15_Document_Image_Translation`.
- One-line framing: a **cascade** `OCR → MT → layout-preserving overlay`, where **MT is the only trained stage** and OCR + render are pretrained/algorithmic.
- Default pair **en→fr**, multilingual-capable from one checkpoint (`facebook/m2m100_418M`, MIT).
- *Speaker note: emphasise that the deliverable is a new image, not just a text string — input and output are both pixels.*

---

## Slide 2 — Problem and use cases

- **In-image MT** = image-with-text → new image with the text translated *in place*; distinct from plain document translation (text-in / text-out) because the translated text must be drawn back where the source was.
- **Use cases:** travel/signage and menus (camera translate), scanned contracts and forms, manga/comics, screenshots and slide decks, born-digital and scanned PDFs.
- **Inputs in scope:** raster images (png/jpg/webp) and PDFs (born-digital *and* scanned); plain `.txt` accepted as a degenerate input that skips OCR.
- **Out of scope:** training a custom OCR or text detector, learned inpainting, real-time/streaming latency targets.
- *Speaker note: the hard part is layout — translations are usually longer than the source, so the box overflows.*

---

## Slide 3 — Why a cascade (not end-to-end)

- Pipeline: `image → [OCR] source text + boxes + conf → [MT] (trainable core) → [RENDER/OVERLAY] → output image`.
- **One trainable component only** = `facebook/m2m100_418M`, fine-tuned with the HF `Seq2SeqTrainer` stack reused verbatim from P13/P14; OCR is a pretrained system library, render is pure algorithm.
- **Reuse over reinvention:** OCR plumbing + born-digital router from **P07 dococr**; MT translator + chrF/BLEU metrics + config/logging/registry/autoreport templates from **P13 s2st / P14 doctrans**.
- **End-to-end OCR-VLM alternatives documented, not shipped:** `stepfun-ai/GOT-OCR-2.0-hf` (Apache-2.0), `google/pix2struct-base` (Apache-2.0) — they read text, they don't produce a translated image.
- *Speaker note: cascade keeps each stage debuggable and lets us isolate OCR cost vs MT quality in the metrics.*

---

## Slide 4 — Data: the synthetic generator

- **No public in-image/document-translation benchmark with gold parallel text exists** (datasets research returned `null`) → the primary data source is a **reproducible synthetic generator** (`data/synth_render.py`).
- Renders MT-corpus source sentences onto page images with varied fonts/sizes and **mild, OCR-survivable degradations** (rotation ±3°, blur, pixel noise, brightness/contrast jitter, JPEG recompression); a CLEAN mode gives an OCR upper bound.
- Emits **`(image, gold_source, gold_target, boxes)` quadruples** + JSONL manifest; **per-index seeding** (`rng = Random(BASE_SEED*1_000_003 + i)`) so the same index always yields the same image (reproducible CER + fixture tests).
- **MT fine-tune corpus:** `Helsinki-NLP/opus-100` en-fr (~1M pairs, **license unknown → flag and verify per pair**). Optional real OCR-noise text: `PleIAs/Post-OCR-Correction` (CC0). Offline backbone = built-in synthetic seed pages + en→fr dictionary.
- *Speaker note: the gold layout/text/translation are known by construction — that is what makes the synthetic set the evaluation floor.*

---

## Slide 5 — The MT core + training (the only trained stage)

- **Default trainable core:** `facebook/m2m100_418M` (**MIT**) — many-to-many ~100 languages, so one fine-tuned checkpoint serves en→fr **and** the multilingual requirement.
- **Training:** HF `Seq2SeqTrainer`, headline metric **chrF** (+ BLEU secondary), corpus loaded in the Colab setup cell (no corpus committed).
- **GPU tiers:** T4 (free Colab, fp16 + grad-accum) → A10/L4 default → H100 upgrade; on T4 OOM, drop to the en→fr-only baseline.
- **Alternatives:** `Helsinki-NLP/opus-mt-en-fr` (Apache-2.0, en→fr only, T4/CPU fallback); `facebook/mbart-large-50-many-to-many-mmt` (MIT, H100 upgrade); **`facebook/nllb-200-distilled-600M` is CC-BY-NC-4.0 → NON-COMMERCIAL, FLAGGED, do not ship.**
- *Speaker note: every shipped component is MIT or Apache-2.0 — fully permissive; the non-commercial options are research-only.*

---

## Slide 6 — OCR front-end (pretrained, not trained)

- **Default:** **Tesseract** via `pytesseract` (Apache-2.0) — `image_to_data` returns **word/block boxes + per-word confidence** in one call, fusing detection and recognition.
- **Born-digital vs scanned router** (PyMuPDF `get_text()` coverage probe, ported from P07): if a real text layer exists, **bypass OCR entirely** (lossless, zero OCR error).
- **Offline fallback = SeedEngine:** reads the gold spec embedded in synthetic images so MT + render + metrics run end-to-end with no tesseract binary; optional deterministic corruption gives a known CER.
- **Alternatives:** docTR / PaddleOCR / EasyOCR (Apache, pip libs), `microsoft/trocr-base-printed` (MIT, neural upgrade); **Surya (`vikp/surya_rec2`+`surya_det3`) is CC-BY-NC-SA-4.0 → NON-COMMERCIAL, FLAGGED.**
- *Speaker note: aggregate to block/paragraph boxes (Tesseract level 2/3) before MT — word-by-word translation destroys meaning and wrecks wrapping.*

---

## Slide 7 — The overlay renderer (the value-add)

- **Pillow-only** fit-to-box engine (no OpenCV, no SciPy; Pillow ≥ 9.2): for each block — **erase → fit → wrap → draw** the translated target back into the source box; output is a new RGB image the same size as input.
- **Erase:** whiteout using the per-channel median of a 2px ring just outside the box (beats hard white on colored/scanned paper); optional horizontal-smear "simple inpaint" + light blur for textured backgrounds. **Erase all boxes before drawing any** translation.
- **Fit-to-box:** binary-search font size + greedy pixel word-wrap (per-character wrap for CJK); returns `fit_ok` that drives the fit-rate metric and decision **D5**.
- **Script-aware fonts** (Noto by Unicode script, SIL OFL 1.1; DejaVuSans shipped inside Pillow as the always-present fallback) + contrast-aware text color (luminance test); RTL is best-effort without optional `python-bidi`/`arabic-reshaper`.
- *Speaker note: verified on Pillow 12.2.0 — a long sentence in a 380×100 box fits at size 33 over 3 lines; this layout-preserving overlay is the single most visible deliverable.*

---

## Slide 8 — The 5-decision agent (the agentic component)

- A **deterministic finite state machine** (not an LLM agent) in `src/imgtrans/agent/` that routes on the pipeline's own intermediate signals and always emits a defensible output instead of a silent failure.
- **D1 ingest** — input router (image / pdf / spec / text + magic-bytes + page-quality gate); **D2 ocr** — born-digital vs scanned routing, skip OCR when a real text layer exists.
- **D3 translate** — per-block OCR-confidence gate (skip translating low-confidence/garbage blocks, never mistranslate); **D4 verify** — round-trip back-translation chrF (soft) + length-ratio sanity, no reference needed.
- **D5 render** — render-fit feasibility gate driven by `fit_ok` → **overlay** (fits) / **side-by-side** (too long) / **needs_review** (flagged or no fit) — a graceful degradation ladder.
- Optional LLM brain (anthropic) is **OFF by default, advisory only, never rewrites**; the whole agent runs **fully offline** (SeedEngine + dictionary MT).
- *Speaker note: this self-checking degradation ladder — confidence + verification + fit gates — is what beats a plain "OCR then translate then print" script.*

---

## Slide 9 — Metrics and results (verified numbers)

- **MT:** chrF (headline) + BLEU, computed on **clean gold source** to isolate MT from OCR noise. **OCR:** CER (headline) + WER. **End-to-end:** image-translation chrF/BLEU = `MT(OCR(rendered page))` vs gold target.
- **Layout fidelity:** overlay **fit-rate** (fraction of blocks whose translation fits the source box) + mean shrink + overflow (also box-retention IoU and no-overlap rate).
- **Verified offline seed eval:** MT dictionary chrF **79.9** vs identity floor **22.4**; OCR CER **0.0** (perfect-OCR via SeedEngine); end-to-end chrF **76.4**; mean fit-rate **1.0**.
- **Honest caveat:** the dictionary **saturates on the seed** because seed pairs overlap the dictionary; on real opus-100 eval pairs the **fine-tuned m2m100 dominates** — that is the non-saturated floor.
- **Baselines:** identity (the floor) + dictionary word-lookup (offline fallback) + zero-shot m2m100 (quantifies what fine-tuning buys).
- *Speaker note: report the gap (clean MT-chrF − end-to-end-chrF) — it quantifies the OCR cost directly.*

---

## Slide 10 — Deployment

- **FastAPI:** `POST /translate-text` (JSON in/out) and `POST /translate-image` (upload image/PDF → translated text + **base64 overlay PNG**); the image route is gated on `python-multipart`.
- **Gradio UI** at `/ui` for interactive upload-and-translate demos.
- **Docker:** image needs the `tesseract-ocr` binary + `libGL`; ships with the offline fallback so it runs without downloads.
- **Hugging Face Space** for a hosted public demo.
- *Speaker note: the offline backbone (SeedEngine + dictionary MT + DejaVu) means CI and the smoke demo pass with only Pillow installed.*

---

## Slide 11 — Ethics and privacy

- **PII risk:** document images can contain IDs, passports, medical and legal documents → consent, **local processing**, and **no raw-image retention by default**.
- **Assistive framing:** the tool **assists** translation and **flags low-confidence output for human review** (the D3/D4 gates and `needs_review` branch) — it never asserts certainty.
- **Robustness concerns:** degraded scans, rotation/blur, multi-column layouts, mixed scripts, and **OCR-error propagation into MT** — mitigated by the post-OCR-confidence gate (D3) and round-trip verification (D4).
- **Licensing discipline:** ship only MIT/Apache-2.0 components; non-commercial models (NLLB CC-BY-NC, Surya CC-BY-NC-SA) are flagged and excluded from the default; verify each opus-100 pair license.
- *Speaker note: low-confidence OCR is skipped (not mistranslated); poor-fit goes side-by-side; flagged blocks go needs_review — graceful, never silently wrong.*

---

## Slide 12 — Conclusion and future work

- **Delivered:** a layout-preserving in-image translation cascade with one trained stage (m2m100), a reproducible synthetic data generator, an offline-capable SeedEngine, a Pillow fit-to-box overlay, and a deterministic 5-decision self-checking agent.
- **Value-add recap:** layout-preserving overlay with auto font-fit **plus** confidence/verification/fit gates — strictly more than "OCR then translate".
- **Future work:** locate or build a real `hub_repo_details`-verified in-image benchmark; learned inpainting for cleaner erase on textured backgrounds; full RTL shaping (`python-bidi` + `arabic-reshaper`); H100 upgrades (mbart-50, TrOCR-large, GOT-OCR-2.0).
- **Honesty stance:** no public benchmark exists, so we do **not** overclaim generalization to photos of real signage — the synthetic suite plus degradations is the floor, not a guarantee.
- *Speaker note: close on the reuse story — P07 + P13 + P14 components recombined, with the render engine, generator, SeedEngine, and agent as the genuinely new P15 work.*
