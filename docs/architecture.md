# P15 Document-Image Machine Translation — System Architecture

Author: Le Dinh Minh Quan (student 23127460) · Package `imgtrans` · Folder `15_Document_Image_Translation`

This document describes the end-to-end system architecture of P15: the pipeline, the module map under `src/imgtrans`, the data-flow between stages, and the offline / degradation design that lets the whole system run with nothing but the Python standard library plus Pillow.

---

## 1. What the system does

P15 translates the text that appears **inside** an image, scanned document, or born-digital PDF and renders the translation **back onto the page** while preserving spatial layout — the "Google Translate camera" experience. Input and output are *pixels*, not strings, which is what distinguishes in-image translation from plain document translation.

The system is a **cascade**, not an end-to-end model:

> **OCR front-end → MT (the only trained stage) → layout-preserving overlay render.**

Only the MT core is fine-tuned (`facebook/m2m100_418M`, MIT). OCR, layout analysis, and rendering are pretrained or purely algorithmic. The default direction is **en→fr**, but because m2m100 is many-to-many the *same* checkpoint serves the multilingual requirement from one set of weights; the direction is configurable.

Wrapping the cascade is a **deterministic finite-state agent** (5 decision points) that gates on each stage's own intermediate signals and chooses how to present the result: full `overlay`, `side_by_side`, or an honest `needs_review`. This self-checking degradation ladder — not a bigger model — is the agentic value-add.

---

## 2. High-level pipeline

Six logical stages, each tagged **PRETRAINED**, **TRAINED**, or **ALGORITHMIC**, collapsed to fit the reuse from P07 / P13 / P14.

| # | Stage | Status | Component |
|---|-------|--------|-----------|
| 1 | Text-region detection | PRETRAINED (fused with OCR) | Tesseract `image_to_data` returns word/line/block boxes + per-word confidence in one call, so detection and recognition are fused — no standalone detector. |
| 2 | OCR / recognition | PRETRAINED | `pytesseract.image_to_data` (boxes + conf); PyMuPDF born-digital-vs-scanned router; `SeedEngine` offline fallback. |
| 3 | Mask / erase | ALGORITHMIC | Pillow whiteout (median border-ring color) + optional horizontal-smear "simple inpaint". |
| 4 | Machine translation | **TRAINED** | `facebook/m2m100_418M` fine-tuned with HF `Seq2SeqTrainer` (chrF/BLEU). The **only** trainable component. Dictionary + identity baselines. |
| 5 | Typeset / re-render | ALGORITHMIC | Pillow fit-to-box: binary-search font size + greedy pixel/CJK word-wrap, script-aware font selection, contrast-aware ink color. |
| 6 | Compose output | ALGORITHMIC | Erase-all-boxes-first then draw-all → new RGB image the same size as the input (or a side-by-side panel in the degrade branch). |

---

## 3. Data-flow diagram

```
                                   ┌──────────────────────────────────────────────┐
   image / pdf / .txt / spec ─────▶│  AGENT FSM  (src/imgtrans/agent)              │
                                   │  deterministic router, 5 decision points     │
                                   └──────────────────────────────────────────────┘
                                                     │
        ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
        ▼                                             ▼                                             ▼
 ┌─────────────┐   D1 input router        ┌────────────────────┐   D2 born-digital vs scanned
 │   INGEST    │── image → scanned ──────▶│       OCR          │── digital PDF: extract text layer ─┐
 │ (page-qual  │── pdf   → D2 ───────────▶│  ocr_engine        │   (PyMuPDF, BYPASS OCR)            │
 │  gate)      │── .txt/spec → skip OCR ─▶│  + SeedEngine      │── scanned: rasterize → Tesseract  │
 └─────────────┘                          └────────────────────┘   image_to_data (boxes+conf)      │
                                                     │                                              │
                                                     ▼                                              │
                                          ┌────────────────────┐                                    │
                                          │      LAYOUT        │  aggregate words → block/para      │
                                          │  imaging/layout    │  boxes (Tesseract level 2/3),      │
                                          │                    │  reading order, geometry           │
                                          └────────────────────┘                                    │
                                                     │  blocks: (text_src, box, conf)               │
                                                     ▼                                              │
                                          ┌────────────────────┐   D3 per-block OCR-confidence gate │
                                          │      VERIFY-IN     │  conf>=HIGH → translate            │
                                          │   (D3 conf gate)   │  LOW..HIGH  → retry / VLM fallback │
                                          │                    │  conf<LOW   → drop, needs_review   │
                                          └────────────────────┘                                    │
                                                     │  accepted blocks ◀───────────────────────────┘
                                                     ▼
                                          ┌────────────────────┐
                                          │        MT          │  m2m100_418M (fine-tuned) OR
                                          │   mt/translator    │  dictionary MT (offline)
                                          └────────────────────┘
                                                     │  translated text per block
                                                     ▼
                                          ┌────────────────────┐   D4 round-trip + length-ratio
                                          │      VERIFY        │  back-translate chrF (soft)        
                                          │   (D4 MT check)    │  length-ratio ∈ [0.4, 3.0]        
                                          │                    │  fail → re-decode once / flag      
                                          └────────────────────┘
                                                     │  text + verify flags + length-ratio
                                                     ▼
                                          ┌────────────────────┐   D5 render-fit feasibility gate
                                          │      RENDER        │  fit_ok at >= min font →  OVERLAY  
                                          │  imaging/render    │  too long           → SIDE_BY_SIDE
                                          │  (erase → fit →    │  flagged/infeasible → NEEDS_REVIEW
                                          │   wrap → draw)     │
                                          └────────────────────┘
                                                     │
                                                     ▼
                                   translated text + overlay PNG (+ boxes, metrics, decision trace)
```

The five decision points (D1–D5) live in the agent but each consults the stage it gates; the agent never re-implements OCR/MT/render logic, it only routes.

---

## 4. Module map (`src/imgtrans`)

The package mirrors the proven layout of P13 s2st / P14 doctrans; only the imaging, synthetic-data, SeedEngine, and image-translation agent are net-new for P15.

### `config`
Single source of truth for paths, language pair, thresholds, and model ids. Holds the decision-point constants (`OCR_CONF_HIGH≈75`, `OCR_CONF_LOW≈40`, round-trip `TAU`, length-ratio band `[0.4, 3.0]`, `font_size_minimum`, fit-rate cutoff), the model registry defaults (`facebook/m2m100_418M`), and the `P15_OFFLINE` flag. Typed dataclass config reused from P13/P14 so every module reads one object.

### `data`
Corpus loading and the **synthetic generator** (`data/synth_render.py`, new for P15). The generator is the **primary data source** because no public in-image translation benchmark with gold parallel text exists (dataset research returned null). It renders source sentences (from the MT corpus) onto page images with varied fonts/sizes and mild, OCR-survivable degradations, embedding the gold layout spec, and emits `(image_png, gold_source_text, gold_target_text, boxes_json)` quadruples plus a JSONL manifest.

- **Determinism:** per sample `i`, `rng = random.Random(BASE_SEED * 1_000_003 + i)`; the `(src, tgt)` pair is chosen by deterministic **index** (not rng) for full corpus coverage, so the same index always yields the same image — exact CER scoring and committed fixtures.
- **MT fine-tune corpus:** `Helsinki-NLP/opus-100` en-fr (~1M pairs, **license unknown → flag and verify per pair before commercial use**); the generator is corpus-agnostic and accepts any `list[(src, tgt)]`. Optional real OCR-noise text: `PleIAs/Post-OCR-Correction` english (CC0).
- **CLEAN mode** (degradations off) gives an OCR upper bound; tiny committed fixtures (3–5 fixed-seed PNGs) drive unit tests.

### `models/ocr_engine`
The OCR front-end, ported from **P07 dococr**. Default path is `pytesseract.image_to_data` (Apache-2.0) returning word boxes + per-word confidence; PyMuPDF probes a PDF's embedded text layer to route born-digital vs scanned (drives **D2**). Includes the **`SeedEngine`** offline fallback (new for P15): for a synthetic image with an available manifest it returns `manifest[i].src` directly (no pixel reading) so the rest of the pipeline runs end-to-end offline, with optional seeded ~3% deterministic character corruption to exercise the CER code at a known error rate. A capability probe (`shutil.which('tesseract')` / `try import`) selects real-vs-stub at runtime — **same code path**. Documented neural upgrades: `microsoft/trocr-base-printed` (MIT); end-to-end OCR-VLMs `stepfun-ai/GOT-OCR-2.0-hf` and `google/pix2struct-base` (Apache). **Surya is CC-BY-NC-SA → flagged, never shipped.**

### `mt/translator`
The trainable MT core plus baselines, ported from **P13/P14**. Default `facebook/m2m100_418M` (MIT, many-to-many) fine-tuned with HF `Seq2SeqTrainer`. Baselines (also the offline MT): **identity** (copy source — the chrF floor) and **dictionary** word-lookup (`' '.join(dict.get(tok.lower(), tok) ...)`) — deterministic, instant, no download. Capability probe swaps in the fine-tuned m2m100 when `transformers`+`torch` are present. Alternates: `Helsinki-NLP/opus-mt-en-fr` (Apache, en→fr only, T4 fallback), `facebook/mbart-large-50-many-to-many-mmt` (MIT, H100 upgrade). **`facebook/nllb-200-distilled-600M` is CC-BY-NC-4.0 → non-commercial, flagged, do not ship.**

### `imaging/preprocess`
Page-quality and image-prep utilities ported from P07: deskew/rotation handling, gamma/upscale/invert retries (used by the D3 low-confidence retry path), born-digital rasterization. Pillow-only, no OpenCV.

### `imaging/layout`
Geometry layer: aggregates Tesseract word boxes to **block/paragraph boxes** (level 2/3) — translating word-by-word destroys meaning and wrecks wrapping — computes reading order, and supplies the boxes consumed by render and by the layout-fidelity metrics (fit-rate, box-IoU, no-overlap).

### `imaging/render`
**New for P15** — the headline deliverable. Pure-Pillow fit-to-box overlay engine (Pillow ≥ 9.2, verified on 12.2.0; no OpenCV/SciPy). Per block: **erase every box first, then draw any translation** (so an enlarged translation never paints over a not-yet-erased neighbor). Steps:
- **Erase:** whiteout with the per-channel median of a 2px ring just outside the box (beats hard-white on colored/scanned paper); optional horizontal-smear simple-inpaint + light blur when border variance is high.
- **Font selection:** majority Unicode script of the target → `{script: path}` map (Noto Sans / Noto CJK / Noto Arabic / etc., SIL OFL 1.1); **DejaVuSans shipped inside Pillow** as the always-present offline fallback.
- **Fit-to-box:** `O(log max_sz)` binary search on font size + greedy pixel word-wrap (per-character for CJK), measured live with `multiline_textbbox` / `textlength`. Returns `fit_ok`, which feeds the fit-rate metric and decision point **D5**.
- **Draw:** `multiline_text` with luminance-contrast ink color; best-effort RTL (`python-bidi` / `arabic-reshaper` if available).

### `training`
HF `Seq2SeqTrainer` harness reused verbatim from P13/P14 to fine-tune m2m100 with the chrF metric (BLEU secondary). Corpus loading via `data`; checkpoints registered through `config`. Trains exactly one stage — the MT core.

### `agent`
**New for P15** — the mandatory agentic component: a deterministic FSM in `src/imgtrans/agent/` with 5 decision points and states `ingest → ocr → translate → verify → render`:
- **D1 ingest** — input router (image/pdf/spec/text by magic-bytes/MIME) + page-quality gate; `.txt`/spec skip OCR; unsupported → `needs_review`.
- **D2 ocr** — born-digital vs scanned routing (PyMuPDF coverage probe); skip OCR when a real text layer exists (lossless, zero OCR error).
- **D3 translate** — per-block OCR-confidence gate: `conf≥HIGH` accept; `LOW..HIGH` retry (gamma/upscale/invert) or optional VLM fallback; `conf<LOW`/empty → drop block, tag `needs_review` (never translate garbage).
- **D4 verify** — round-trip back-translation chrF (**soft** gate) + target/source length-ratio sanity, both from the model's own output (no reference needed); fail → re-decode once with alternate params, else flag `low_confidence`; length-ratio fed forward to D5.
- **D5 render** — render-fit feasibility: `fit_ok` at ≥ min legible font → **OVERLAY**; too long (common →fr / CJK→EN expansion) → **SIDE_BY_SIDE**; flagged/infeasible → **NEEDS_REVIEW**.

An optional LLM brain (`anthropic`) is **OFF by default, advisory only, never rewrites** output. The agent runs fully offline on SeedEngine + dictionary MT. Value-add: layout-preserving overlay with auto font-fit plus the confidence/verification gates — strictly more than "OCR then translate."

### `api`
**FastAPI** service: `POST /translate-text` (JSON in/out) and `POST /translate-image` (upload image/PDF → translated text + base64 overlay PNG); the image route is gated on `python-multipart`. A **Gradio** UI is mounted at `/ui`. Packaged with **Docker** (needs `tesseract-ocr` + `libGL`) and an HF Space.

### `analysis`
Computes and tabulates the metric suite from agent runs: MT chrF (headline) + BLEU; OCR CER (headline) + WER; **end-to-end image-translation chrF** + BLEU (`MT(OCR(rendered_page))` vs gold translation); and layout fidelity — overlay **fit-rate** + mean shrink + overflow, box-retention IoU, no-overlap rate. Verified offline seed numbers: dictionary-MT chrF 79.9 vs identity floor 22.4; OCR CER 0.0 (perfect SeedEngine OCR; realistic with Tesseract on Colab); end-to-end chrF 76.4; mean fit-rate 1.0. (Dictionary saturates on the seed because seed pairs overlap the dictionary; on real opus-100 eval pairs the fine-tuned m2m100 dominates — the honest non-saturated floor.)

### `autoreport`
Auto-generates the run report (config + metrics + decision-trace counts + sample overlays) from a single command — template reused from P13/P14.

### `monitoring`
Run-time signal capture: per-stage timings, decision-point branch counts (how many blocks took overlay vs side-by-side vs needs_review), OCR/MT confidence distributions. Template reused from P13/P14.

### `automation`
Autopilot driver that chains generate → OCR → translate → render → evaluate → report end-to-end for reproducible benchmark runs. Reused template.

### `grading`
Self-grading harness that scores a run against the project rubric (metrics present, baselines beaten, offline path green, agent branches exercised) for the assignment deliverable. Reused template.

---

## 5. Offline & degradation design

The defining engineering property of P15 is that the **entire pipeline runs deterministically with only the Python standard library plus Pillow** — no tesseract binary, no torch, no Noto TTFs, no sacrebleu. Tests pass in CI/Colab with nothing downloaded. Three mechanisms make this work.

### 5.1 Lazy imports + capability probes (one code path)
Heavy dependencies (`torch`, `transformers`, `pytesseract`, `fitz`, `sacrebleu`, `anthropic`) are imported lazily inside the functions that need them, never at module top level. Each stage runs a capability probe — `try import` / `shutil.which('tesseract')` — and selects the real component when present, the stub when absent. The env flag `P15_OFFLINE=1` pins stub mode for reproducible tests. Crucially, **the probe upgrades each stage in place; the surrounding code and the tests are identical online and offline.**

### 5.2 SeedEngine (offline OCR)
`SeedEngine` is the offline OCR backbone (new for P15). For a synthetic image with an available manifest it returns the gold source text by index (`OCR(image_i) := manifest[i].src`) without reading pixels, so MT + render + metrics execute end-to-end offline with CER = 0 against gold (clean). An optional seeded per-index corruption (`random.Random(seed+i)`, ~3% char substitute/drop) produces a **known, reproducible CER** to exercise the error-rate code — mirroring P07's stub-OCR pattern. When `pytesseract` + the binary exist, the probe switches to the real P07 path (`image_to_data` boxes + conf, born-digital router).

### 5.3 Dictionary MT + font fallback + pure-Python metrics
- **Dictionary MT** (no torch / no m2m100): a small in-repo bilingual glossary with identity passthrough for OOV — the P13/P14 baseline reused as **both** baseline **and** offline MT. Swaps to fine-tuned m2m100 when transformers+torch are present.
- **Font fallback** (no Noto): `PIL/fonts/DejaVuSans.ttf` is always inside the Pillow wheel, then `ImageFont.load_default()` last. Latin renders correctly; **CJK/Arabic render tofu in pure-offline mode only — flagged**; real runs ship Noto.
- **Metrics** (no sacrebleu/jiwer): CER/WER use a pure-Python two-row Levenshtein; chrF/BLEU use a minimal pure-Python chrF (char n-gram F, n=1..6, β=2), with real sacrebleu when installed (a tolerance test asserts they agree). Layout metrics are pure geometry — already dependency-free and fully meaningful offline.

**Net:** with stdlib + Pillow only, `image → OCR(SeedEngine) → MT(dictionary) → render(DejaVu) → all metrics` runs deterministically. The offline chrF/BLEU are deliberately low (they assert pipeline wiring, not translation quality), while CER/WER, fit-rate, IoU, and no-overlap remain fully meaningful because they measure deterministic geometry and stub corruption.

---

## 6. Reuse map

- **From P07 dococr:** Tesseract `image_to_data` front-end, the PyMuPDF born-digital-vs-scanned router (→ D2 verbatim), and the stub-OCR offline pattern (→ SeedEngine).
- **From P13 s2st / P14 doctrans:** the m2m100 + `Seq2SeqTrainer` fine-tune harness; dictionary + identity baselines (→ baselines **and** offline MT); sacrebleu chrF/BLEU plumbing; the config / logging / registry / autoreport / monitoring / grading / autopilot templates.
- **New for P15:** `imaging/render.py` fit-to-box overlay engine; `data/synth_render.py` synthetic doc-image generator; the `SeedEngine` offline OCR; the layout-fidelity metrics; and the 5-decision image-translation agent.

---

## 7. Ethics, privacy & robustness (architectural commitments)

Document images routinely contain PII (IDs, passports, medical/legal documents). The architecture therefore: processes locally by default, **retains no raw image** by default, treats the tool as an **assistant** that flags low-confidence output for human review (D3/D4/D5 → `needs_review`) and **never asserts certainty**, and requires consent for any retention. Robustness is engineered, not assumed: degraded scans, rotation/blur, multi-column layouts, and mixed scripts are exercised by the synthetic degradation suite, and **OCR-error propagation into MT** is mitigated by the post-OCR confidence gate (D3) and the round-trip verification (D4). Translation expansion overflowing boxes (esp. →fr and CJK→EN) is an *expected* outcome handled by the D5 side-by-side branch and the fit-rate metric — a designed degradation, not a bug.
