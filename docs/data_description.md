# P15 Document-Image Machine Translation — Data Description

> Project: **P15 Document-Image Machine Translation** (package `imgtrans`, folder `15_Document_Image_Translation`)
> Author: **Le Dinh Minh Quan**, student **23127460**
> Default direction: **en → fr** (configurable; the trainable MT core `facebook/m2m100_418M` is many-to-many, so one fine-tuned checkpoint also serves other pairs)

This document describes every data source P15 uses, why the primary data source is **synthetic by necessity**, how the synthetic generator renders pages and embeds a gold layout/translation spec, the text corpora that feed it, the offline seed pages used by the dependency-free fallback path, the train/validation/test splits and sizes, the licensing posture (with explicit non-commercial flags), and the preprocessing applied at each stage.

---

## 1. Why there is no real in-image translation benchmark

P15's task is **in-image machine translation**: take an image (photo, scan, or born-digital PDF) that contains text, translate that text, and **render the translation back onto the page preserving the original spatial layout** — the "Google Translate camera" / "manga translate" experience. The cascade is **OCR → MT → layout-preserving overlay**, where the **only trained stage is the MT core**; OCR, layout routing, and the render engine are pretrained or purely algorithmic.

To train and — more importantly — to **evaluate end to end**, we need data of the form:

```
(page image, gold source text, gold target translation, gold text-region boxes)
```

i.e. an image whose embedded text is known exactly, whose correct translation is known exactly, **and** whose text-region geometry is known exactly so layout fidelity can be scored.

**No such public benchmark exists.** The dedicated datasets research for this project returned `null`: there is no verified, ready-made, in-image / document-image translation dataset on the Hugging Face Hub (or elsewhere) that ships gold *parallel* text aligned to *boxes* on *page images*. The reasons are structural:

- **OCR datasets** (text recognition / detection) give image + transcription + boxes, but **no translation**.
- **Machine-translation corpora** (OPUS, Tatoeba, WikiMatrix, etc.) give source + target sentence pairs, but **no images and no layout**.
- **Document-understanding / scene-text datasets** give images + text + sometimes boxes, but are **monolingual** — there is no gold translation to compare against.
- The handful of "camera translate" products are closed and ship no labelled, layout-aligned parallel evaluation set.

Consequently, P15 **does not claim** a public in-image translation benchmark and **does not depend** on one. Instead, the primary data source is a **reproducible synthetic generator** (Section 3) that *constructs* the missing quadruples by rendering real parallel sentences onto page images and recording the gold layout by construction. Because the generator knows the source text, the gold translation, and the exact boxes it drew, all four families of metric — OCR CER/WER, MT chrF/BLEU, end-to-end image-translation chrF/BLEU, and layout fidelity — become exactly computable.

> If a real, license-clean, `hub_repo_details`-verified in-image translation benchmark is located later, it can be wired in as a **held-out test split**. Until then, synthetic data is the evaluation floor, and we are explicit that results do **not** automatically generalize to photos of real-world signage.

---

## 2. Data sources at a glance

| # | Source | Role in P15 | Form | License | Flag |
|---|--------|-------------|------|---------|------|
| 1 | **Synthetic generator** (`data/synth_render.py`) | **PRIMARY** — produces `(image, gold_source, gold_target, boxes)` quadruples for end-to-end + layout evaluation | generated PNG + JSONL manifest | code is ours (project license); inherits text license from the corpus it renders | inherits corpus flag |
| 2 | **`Helsinki-NLP/opus-100` (en-fr)** | MT fine-tune corpus — the parallel `(src, tgt)` text rendered onto images and used as translation gold | parallel sentences | **unknown / per-pair — FLAGGED**, verify before commercial use | **license unknown** |
| 3 | **`PleIAs/Post-OCR-Correction` (english)** | Optional real OCR-noise text source — realistic post-OCR error patterns for robustness | text (OCR'd + corrected) | **CC0** (public domain) | clean |
| 4 | **Built-in offline seed pages** (`SeedEngine`) | Dependency-free fallback — committed synthetic pages + en→fr dictionary for offline tests/CI | committed PNG + embedded gold spec | ours (project license) | clean |
| 5 | **Fonts** (Noto Sans family; DejaVuSans bundled in Pillow) | Rendering glyphs for synthetic pages and the overlay | TTF/TTC | **SIL OFL 1.1** (Noto) / bundled (DejaVu) | clean, redistributable |

**Nothing is committed at scale.** The large text corpora (#2, #3) are downloaded in the Colab/setup cell and never checked into the repo. Only the **tiny fixtures** — a handful of committed seed PNGs (#4) — live in version control so that unit tests and CI run with nothing downloaded.

---

## 3. PRIMARY data: the synthetic document-image generator

`data/synth_render.py` is the heart of P15's data story. It is a **deterministic, per-index-seeded** generator that renders real parallel sentences onto page images and emits the gold layout/translation spec alongside each image. It is **corpus-agnostic**: it accepts any `list[(src, tgt)]` and renders it, so the text corpus (Section 4) is swappable without touching the generator.

### 3.1 What each sample contains

For sample index `i`, the generator produces a **quadruple**:

```
(image_i.png, gold_source_text_i, gold_target_text_i, boxes_i)
```

emitted as an image file plus one JSONL manifest row:

```json
{
  "id": 42,
  "src": "The quick brown fox jumps over the lazy dog.",
  "tgt": "Le renard brun rapide saute par-dessus le chien paresseux.",
  "src_lang": "en",
  "tgt_lang": "fr",
  "font": "NotoSans-Regular.ttf",
  "size": 28,
  "boxes": [[40, 30, 600, 78], [40, 86, 600, 134]],
  "degrade_params": {"rotation": 1.4, "blur": 0.6, "noise": 4, "jpeg_q": 80},
  "seed": 1000003042
}
```

The `boxes` are the **gold layout** — the exact tight bounding boxes the generator drew the source text into. Because the same greedy pixel word-wrap used by the render engine (Section 6) is used to lay the source text out, the boxes are realistic and the layout-fidelity metrics are meaningful.

### 3.2 Determinism (reproducibility contract)

Every sample is seeded **per index**, never from global state:

```python
rng = random.Random(BASE_SEED * 1_000_003 + i)
# and, if numpy is used:
np_rng = np.random.default_rng(BASE_SEED * 1_000_003 + i)
```

- **Every** random choice (canvas, font, size, layout, ink color, degradations) draws from this per-index `rng`. No leakage to the global `random` / `numpy` state is permitted — such a leak would break reproducible CER scoring and fixture tests.
- The `(src, tgt)` pair is selected by **deterministic index** (the `i`-th pair), *not* by `rng`, which guarantees **full corpus coverage** and that the same index always yields the same sentence.
- Net effect: **the same index always yields the same image**, enabling reproducible unit tests and exact, byte-stable scoring.

### 3.3 Per-sample render pipeline

1. **Pick the pair** — take the `i`-th `(src, tgt)` pair by index.
2. **Canvas** — size ∈ `{(640,200), (800,256), (1024,300)}`; background is solid paper (off-white / cream), a 2-color vertical gradient, or faint procedural noise. The background choice is recorded.
3. **Font** — `rng.choice` over a fixed list (Noto Sans, DejaVuSans, a serif, a mono if present); size ∈ `[18, 40]`. The resolved font path is recorded.
4. **Layout** — lay the **source** sentence into **1–3 blocks** using the *same* greedy pixel word-wrap as the render engine; random margins and line spacing ∈ `[1.0, 1.4]`. Each block's tight bbox is recorded via `multiline_textbbox` → `boxes`. **These boxes are the gold layout.**
5. **Ink** — render in a high-contrast ink color drawn from a small ink palette.
6. **Degradations** — mild, **OCR-survivable**, each toggled by `rng` with a fixed probability and applied in a **fixed order** so they compose deterministically:
   - rotation `uniform(-3, 3)°`, `expand=True`, fill = background;
   - Gaussian blur radius ∈ `{0, 0, 0, 0.6, 1.0}`;
   - Gaussian pixel noise σ ∈ `{0, 4, 8}`, clipped to `0–255`;
   - brightness / contrast jitter `uniform(0.85, 1.15)`;
   - JPEG recompression to an in-memory `BytesIO` at quality ∈ `{95, 80, 60}` and reload.
   A **CLEAN mode** (all degradations off) is provided to establish an OCR / end-to-end **upper bound**.
7. **Emit** — write `image.png` plus the JSONL manifest row above.

### 3.4 What the quadruples let us measure

Because every quantity is known by construction, the generator output drives **four families of metric**:

| Metric family | Computation |
|---------------|-------------|
| **OCR** CER (headline) + WER | OCR(image) vs `gold_source_text` |
| **MT** chrF (headline) + BLEU | MT(`gold_source_text`) vs `gold_target_text` — on **clean gold source**, isolating MT from OCR noise |
| **End-to-end** image-translation chrF + BLEU | MT(OCR(image)) vs `gold_target_text` — the full pipeline on the rendered/degraded page; the gap vs clean MT-chrF quantifies the OCR cost |
| **Layout fidelity** | rendered boxes vs `gold boxes`: overlay **fit-rate** (fraction of blocks whose translation fits the source box), **mean shrink**, **overflow**, box-retention IoU, no-overlap rate |

### 3.5 Scale and fixtures

- **Evaluation scale:** default **200–2000 samples**, streamable for more.
- **Unit-test fixtures:** **3–5 samples**, fixed seed, with **committed PNGs**, so geometry and metric plumbing are tested deterministically in CI with nothing downloaded.

---

## 4. Text corpora rendered onto the images

The synthetic generator needs real parallel text to render and to use as translation gold. This text comes from a standard MT corpus; it is **the text side only** — the generator turns it into images.

### 4.1 MT fine-tune corpus — `Helsinki-NLP/opus-100` (en-fr)

- **What it is:** the OPUS-100 multilingual parallel corpus; the **en-fr** pair provides roughly **1M sentence pairs**.
- **Role:** the **source** side is rendered onto the synthetic page images; the **target** side is the **translation gold**. It is also the corpus the MT core (`facebook/m2m100_418M`) is fine-tuned on with the HF `Seq2SeqTrainer` (metric chrF).
- **License:** **unknown / per-pair — FLAGGED.** OPUS aggregates many upstream sources with mixed terms. The exact en-fr pair license **must be verified** (`hub_repo_details` + upstream provenance) before any commercial use. For this academic project it is used for research/evaluation; **do not assume a permissive license**.
- **Swappability:** the generator accepts any `list[(src, tgt)]`. Cleaner, smaller, mostly **CC-BY** pairs (Tatoeba / WikiMatrix-style, carried over from P14 doctrans) are a drop-in alternative for fixtures and require CC-BY attribution. **No corpus is committed to the repo**; it is loaded in the Colab setup cell.

### 4.2 Optional OCR-noise text source — `PleIAs/Post-OCR-Correction` (english)

- **What it is:** a corpus of real OCR output paired with corrected text, capturing **authentic post-OCR error distributions** (character confusions, broken words, spacing artifacts).
- **Role:** an **optional, realistic** noise source. Rather than relying solely on the generator's synthetic degradations, P15 can use these real OCR-error patterns to stress-test how OCR noise propagates into MT — exercising the per-block OCR-confidence gate (agent D3) and the MT round-trip verification (agent D4).
- **License:** **CC0** — public domain, no restrictions. Clean to use and redistribute.
- **Status:** optional enrichment, not on the critical path. The synthetic degradation suite (Section 3.3, step 6) is the default noise model.

---

## 5. Offline seed pages (dependency-free fallback)

P15 must run **fully offline** for CI and for the agent's offline mode — **no Tesseract binary, no torch, no downloaded fonts**. The offline backbone is two committed, deterministic pieces:

### 5.1 The `SeedEngine` and committed seed pages

- **Built-in synthetic seed pages:** a small set of committed PNGs produced by the same generator, each carrying an **embedded gold spec** (source text, gold translation, boxes) so the full `image → OCR → MT → render → metrics` path runs with only the Python standard library + Pillow.
- **`SeedEngine` offline OCR:** instead of reading pixels, the `SeedEngine` reads the **gold spec embedded in the synthetic image / manifest** and returns the known source text. This makes OCR "perfect" offline (`CER = 0.0`) so the rest of the pipeline (MT, render, metrics) can be exercised end to end without a real OCR engine.
  - Optionally, a **deterministic corruption mode** (seeded per index, substituting/dropping ~3% of characters) yields a **known, reproducible CER** to exercise the error-rate code, mirroring P07's stub-OCR pattern.
  - When `pytesseract` + the Tesseract binary are present, a capability probe upgrades this stage to the **real** P07 path (`image_to_data` boxes + confidence, born-digital router) — **same code path, no test changes**.

### 5.2 The en→fr dictionary (offline MT)

- A small in-repo **bilingual glossary** `{src_token: tgt_token}` plus identity passthrough for out-of-vocabulary tokens:
  `translate(s) = ' '.join(dict.get(tok.lower(), tok) for tok in s.split())`.
- It is deterministic, instant, and requires no download. It serves as **both** the dictionary baseline **and** the offline MT engine. When `transformers` + `torch` are present, the fine-tuned `m2m100_418M` is swapped in via the same probe mechanism.

> **Saturation caveat (honest reporting):** on the offline **seed** set, the dictionary baseline scores artificially high because the seed sentences overlap the dictionary's coverage. The verified offline seed eval shows **MT dictionary chrF 79.9** vs the **identity floor 22.4**, OCR **CER 0.0** (perfect-OCR via `SeedEngine`; realistic CER appears with Tesseract on Colab), **end-to-end chrF 76.4**, and **mean fit-rate 1.0**. These numbers test **pipeline wiring and metric plumbing**, not translation quality. On the real `opus-100` eval pairs (which do **not** overlap the dictionary), the fine-tuned `m2m100` dominates the dictionary baseline — that is the honest, non-saturated floor. Geometry metrics (CER/WER, fit-rate, IoU, no-overlap) are fully meaningful even offline.

---

## 6. Splits and sizes

| Split | Source | Size (default) | Purpose |
|-------|--------|----------------|---------|
| **MT train** | `opus-100` en-fr (text only) | up to ~1M pairs (subsampled to fit GPU tier) | fine-tune `m2m100_418M` with `Seq2SeqTrainer` |
| **MT validation** | `opus-100` en-fr held-out | a few thousand pairs | chrF/BLEU during training, early stopping |
| **MT test (clean)** | `opus-100` en-fr held-out | a few thousand pairs | headline MT chrF/BLEU on clean gold source |
| **Image eval set** | synthetic generator over held-out pairs | **200–2000** quadruples | OCR CER/WER, end-to-end chrF/BLEU, layout fidelity |
| **Image eval (CLEAN mode)** | synthetic generator, degradations off | same indices, clean | OCR / end-to-end **upper bound** |
| **Unit-test fixtures** | committed seed PNGs | **3–5** samples, fixed seed | deterministic geometry + metric-plumbing tests in CI |

Notes on splitting:

- The **synthetic image eval set is built from held-out parallel pairs** (pairs not seen during MT fine-tuning), so the end-to-end evaluation is not contaminated by training text.
- The image eval set has a **CLEAN twin** (same indices, all degradations off) so the OCR-error gap `(clean − degraded)` is measured on matched content.
- GPU tier governs MT train subsampling: on a free **T4** the `m2m100_418M` fine-tune uses fp16 + small batch + grad-accum (and may fall back to `opus-mt-en-fr` for an en→fr-only demo); the **default** mid-GPU tier (A10/L4) trains the full `m2m100_418M`.

---

## 7. Licenses and non-commercial flags

| Asset | License | Verdict |
|-------|---------|---------|
| Synthetic generator code (`synth_render.py`), seed pages, dictionary | project license (ours) | ship |
| **`Helsinki-NLP/opus-100` (en-fr)** | **unknown / per-pair — FLAGGED** | research/eval here; **verify each pair before commercial use** |
| **`PleIAs/Post-OCR-Correction` (english)** | **CC0** (public domain) | ship / redistribute freely |
| Tatoeba / WikiMatrix-style pairs (alt) | mostly **CC-BY** | ship **with attribution** |
| Noto Sans fonts | **SIL OFL 1.1** | ship / redistribute |
| DejaVuSans (bundled in Pillow) | bundled with Pillow | ship |

**Explicit non-commercial flags (data and model adjacent):**

- **MT:** `facebook/nllb-200-distilled-600M` is **CC-BY-NC-4.0 → NON-COMMERCIAL**. It is a quality/coverage upgrade for research only and is **never shipped** as the default. The shipped MT default `facebook/m2m100_418M` is **MIT**.
- **OCR:** **Surya** (`vikp/surya_rec2` + `vikp/surya_det3`) is **CC-BY-NC-SA-4.0 → NON-COMMERCIAL + share-alike**. Documented as a research-quality alternative only; **never shipped**. The shipped OCR front-end (Tesseract / `pytesseract`) is **Apache-2.0**.
- **Corpus:** treat each `opus-100` pair's license as **unknown until verified**.

The default shipped stack is deliberately **all MIT / Apache-2.0 / OFL / CC0** — fully permissive — with every non-commercial option flagged and excluded from the default.

---

## 8. Preprocessing

Preprocessing spans the text corpus, the synthetic image generation, and the OCR/MT/render path.

### 8.1 Text-corpus preprocessing (MT side)

- **Pair loading** reuses the P13/P14 parallel-corpus pattern: stream `opus-100` en-fr, drop empty / whitespace-only pairs.
- **Length filtering:** drop pathologically long or near-empty sentences that would not render legibly into a page block or would blow the MT max-token budget.
- **Tokenization** is handled by the `m2m100` tokenizer at fine-tune time; the dictionary baseline uses simple whitespace tokenization with lowercased lookup keys.
- **Deterministic indexing:** the generator selects the `i`-th pair by index (not by RNG) for full coverage and reproducibility.

### 8.2 Synthetic-image preprocessing (generation side)

- **Layout** uses the **same greedy pixel word-wrap** as the render engine, so gold boxes match the algorithm under test (CJK, if present, wraps per-character since there are no spaces).
- **Degradations** are applied in a **fixed order** (rotate → blur → noise → brightness/contrast → JPEG recompress) so they compose deterministically; each is toggled by the per-index RNG. A **CLEAN mode** disables all of them.
- **Gold spec** (source, target, boxes, font, size, degrade params, seed) is written to the JSONL manifest at emit time.

### 8.3 OCR / routing preprocessing (front-end)

- **Input routing (agent D1):** file magic-bytes / MIME + extension decide image vs PDF vs `.txt`; a page-quality gate screens unreadable inputs.
- **Born-digital vs scanned (agent D2):** PyMuPDF probes `page.get_text()` coverage (reused verbatim from P07 dococr). Born-digital PDFs **bypass OCR** entirely — the embedded text layer + word boxes are extracted losslessly (zero OCR error). Scanned PDFs are rasterized and routed through OCR like images.
- **OCR aggregation:** Tesseract `image_to_data` returns word/line/block boxes + per-word confidence. P15 aggregates to **block / paragraph boxes** (`level=2/3`) **before MT** — translating word-by-word destroys meaning and wrecks wrapping.
- **Per-block confidence gate (agent D3):** blocks below a confidence / alpha-ratio / min-length threshold are **skipped, not mistranslated** (tagged `needs_review`); mid-confidence blocks may be re-scored after gamma-correct / upscale / invert.
- **Optional retry preprocessing:** gamma correction, upscaling, and inversion on low-confidence crops before re-OCR.

### 8.4 Render / overlay preprocessing (output side)

- **Erase-all-then-draw:** every box is erased (median-of-border-ring whiteout, optional horizontal-smear inpaint) **before** any translation is drawn, so an enlarged translation never paints over a not-yet-erased neighbor.
- **Script-aware font selection:** the target string's majority Unicode script picks the font (Noto by script; DejaVuSans fallback for uncovered glyphs / offline mode — CJK/Arabic render as tofu offline, **flagged**, affecting pure-offline mode only).
- **Fit-to-box:** binary-search font size + greedy word-wrap returns `fit_ok`, which feeds the layout fit-rate metric and the render-fit decision (agent D5: overlay vs side-by-side vs needs_review).
- **Contrast-aware text color:** chosen by background luminance (`0.299R + 0.587G + 0.114B < 128 → white else black`).

---

## 9. Ethics and privacy of the data

Document images are an unusually sensitive input class: they can contain **PII** (IDs, passports, medical and legal documents). P15's data handling reflects this:

- **No real user documents are collected or committed.** All committed data is synthetic (generator output / seed pages); the text corpora are public and downloaded at runtime, never bundled.
- **Local processing by default**, with **no raw-image retention** by default in the deployed service.
- The tool **assists** translation and **flags low-confidence output for human review** (agent D3/D4/D5); it never asserts certainty over a translated document.
- **Consent** is expected for any real document a user submits; the synthetic-first design means development and evaluation never require real sensitive material.

---

## 10. Summary

- There is **no public in-image translation benchmark** with gold parallel text aligned to boxes on page images (datasets research = `null`), so P15's **primary data is a reproducible synthetic generator** that *constructs* the missing `(image, gold_source, gold_target, boxes)` quadruples.
- The generator is **deterministic** (per-index seed `BASE_SEED * 1_000_003 + i`), **corpus-agnostic**, and renders real parallel sentences with composable, OCR-survivable degradations plus a CLEAN-mode upper bound.
- Text comes from **`opus-100` en-fr** (~1M pairs, **license unknown — FLAGGED**), with optional real OCR-noise from **`PleIAs/Post-OCR-Correction`** (**CC0**). Offline, the **`SeedEngine` + committed seed pages + en→fr dictionary** keep the whole pipeline runnable with only stdlib + Pillow.
- Splits keep the **image eval set on held-out pairs**, with a CLEAN twin to isolate OCR cost, and tiny committed fixtures for CI.
- The default stack is **fully permissive** (MIT / Apache-2.0 / OFL / CC0); the non-commercial traps (**NLLB CC-BY-NC-4.0**, **Surya CC-BY-NC-SA-4.0**) are flagged and **never shipped**.
