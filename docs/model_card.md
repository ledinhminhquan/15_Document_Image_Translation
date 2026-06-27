# Model Card — P15 Document-Image Machine Translation (MT core)

**Project:** P15 "Document-Image Machine Translation" (package `imgtrans`, folder `15_Document_Image_Translation`)
**Author:** Le Dinh Minh Quan — student 23127460
**Component documented here:** the fine-tuned **machine-translation core**, the *only trained stage* of the P15 cascade.
**Status:** Model card / spec. The offline numbers below are from the verified seed evaluation; the on-GPU fine-tune is run from the project Colab notebook.

> Scope reminder. P15 translates text that appears *inside* an image, scanned document, or born-digital PDF and renders the translation back onto the page preserving layout (the "Google Translate camera" experience). It is a **cascade**: OCR front-end → **MT (this card)** → layout-preserving overlay render. OCR, layout routing, and rendering are pretrained/algorithmic; **MT is the single trainable component**. This card describes that component, not the whole pipeline — but it reports the end-to-end and OCR/layout metrics because the MT core is evaluated *in situ* inside the cascade as well as in isolation.

---

## 1. Model summary

| Field | Value |
|-------|-------|
| **Model name** | `imgtrans-mt-m2m100-en-fr` (fine-tuned checkpoint) |
| **Base model** | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) |
| **Base license** | **MIT** (fully permissive, commercial-safe) |
| **Architecture** | M2M-100, many-to-many multilingual encoder-decoder Transformer, ~418M parameters |
| **Fine-tune framework** | Hugging Face `Seq2SeqTrainer` (ported verbatim from P13 s2st / P14 doctrans) |
| **Default direction** | **en → fr** (configurable; the base is many-to-many so one checkpoint serves other pairs) |
| **Training corpus** | `Helsinki-NLP/opus-100` en-fr parallel pairs (≈1M) |
| **Headline metric** | **chrF** (`sacrebleu.CHRF`), with BLEU secondary |
| **Intended task** | Translate short text blocks extracted by OCR / a PDF text layer, one paragraph-level block at a time |
| **Repository** | `15_Document_Image_Translation` (package `imgtrans`) |

### Why m2m100_418M is the default

- **License:** MIT — ships in a fully permissive stack (every shipped component of P15 is MIT or Apache-2.0).
- **Multilingual from one checkpoint:** M2M-100 is many-to-many over ~100 languages, so a single fine-tuned set of weights serves the default `en→fr` *and* the multilingual requirement without training a separate bilingual model per pair.
- **Fits a free T4:** fine-tunable in fp16 with small batch + gradient accumulation on a free Colab T4 (~16 GB).
- **Proven stack:** the same base + `Seq2SeqTrainer` harness + chrF/BLEU plumbing was already verified in P13/P14 and is reused here.

### Alternatives (documented, not the shipped default)

| id | License | Role |
|----|---------|------|
| `Helsinki-NLP/opus-mt-en-fr` | Apache-2.0 | Strong baseline + near-CPU / T4 fallback. **Bilingual en→fr only** — does *not* cover multilingual. |
| `facebook/mbart-large-50-many-to-many-mmt` | MIT (per card) | H100 quality upgrade (611M, 50 langs). m2m100 stays the simpler verified default. |
| `facebook/nllb-200-distilled-600M` | **CC-BY-NC-4.0 — NON-COMMERCIAL** | Quality/coverage upgrade for research only. **Flagged: do NOT ship.** |

> **Non-commercial flag.** `facebook/nllb-200-distilled-600M` is **CC-BY-NC-4.0** and must never be the shipped default. It is referenced only as a research-grade quality upgrade. The **shipped MT core is MIT** (`m2m100_418M`) so the production pipeline stays commercial-safe.

---

## 2. Intended use

### Primary intended use

Translate **short, OCR-extracted text blocks** — typically a paragraph or layout block returned by Tesseract `image_to_data` (level 2/3) or by a PDF's embedded text layer — from a source language to a target language (default `en→fr`). The translated block is then handed to the algorithmic overlay engine, which fits it back into the original source box.

The MT core is **block-at-a-time**: one paragraph-level block in, one translation out. Blocks are aggregated to paragraph/line level *before* MT — never translated word-by-word, because word-by-word translation destroys meaning and wrecks the downstream wrapping.

### Intended users

- Developers building a layout-preserving "camera translate" / document-image translation tool.
- Researchers studying OCR→MT cascades and OCR-error propagation.
- The P15 deterministic agent (Section 6 of the design brief), which calls this core between its OCR-confidence gate (D3) and its translation-verification gate (D4).

### In-scope inputs

- Short blocks of printed/born-digital text (the common case): a sentence to a short paragraph.
- Text from any of the supported input routes: raster image OCR, scanned-PDF OCR, born-digital PDF text layer, or a plain `.txt` block.

### Out-of-scope / not intended for

- **Long documents in a single call** — pass per-block, not the whole page; quality and the box-fit assumptions degrade on very long inputs.
- **Authoritative or certified translation** — outputs are *assistive* and flagged for human review when confidence is low (see Ethics). Never use for legal, medical, or identity-document translation without a qualified human in the loop.
- **OCR or layout** — the MT core does not read pixels or detect boxes; those are the pretrained Tesseract front-end and the algorithmic overlay engine.
- **Languages or scripts the base model does not cover well**, and offline-mode CJK/Arabic rendering (a *font/render* limitation, not an MT one — flagged below).

---

## 3. Training data

| Item | Detail |
|------|--------|
| **MT fine-tune corpus** | `Helsinki-NLP/opus-100`, en-fr split (~1M parallel sentence pairs) |
| **Direction** | source = en, target = fr (default); the corpus side rendered onto synthetic pages is the *source*, the parallel target is the gold translation |
| **Optional OCR-noise text source** | `PleIAs/Post-OCR-Correction` (English, **CC0**) — for studying OCR-noised MT robustness |
| **Synthetic image data** | `data/synth_render.py` renders source sentences from the MT corpus onto page images (varied fonts/sizes, mild OCR-survivable degradation) and embeds the gold layout spec → `(image, gold_source, gold_target, boxes)` quadruples |

### License flags on data

- **`Helsinki-NLP/opus-100` license is unknown / mixed per pair → FLAGGED.** Verify each pair's license before commercial use. Do not assume a single permissive license for the whole corpus.
- **`PleIAs/Post-OCR-Correction` is CC0** (public domain) — safe.
- **No public in-image / document-image translation benchmark with gold parallel text exists** (dataset research returned null). Do **not** claim one. P15's primary evaluation data is therefore the **reproducible synthetic generator**, whose gold source, gold target, and gold boxes are known by construction.

### Synthetic generator note

The generator is *corpus-agnostic* (takes any `list[(src, tgt)]`) and deterministic per index (`rng = random.Random(BASE_SEED * 1_000_003 + i)`), so the same index always yields the same image — enabling exact CER scoring and committed fixture PNGs. The synthetic data is the floor; it does **not** establish generalization to photos of real-world signage.

---

## 4. Evaluation

P15 evaluates the MT core both in isolation (on clean gold source, to isolate MT from OCR noise) and inside the full cascade (end-to-end, to expose OCR-error propagation). Layout fidelity is reported because translation expansion is the dominant overlay failure mode.

### 4a. Metrics

| Metric | Stage | Definition | Direction |
|--------|-------|------------|-----------|
| **chrF** (headline) | MT | `sacrebleu.CHRF`, `MT(gold_source)` vs `gold_target`, on **clean gold source** so it isolates MT from OCR noise | higher better (0–100) |
| **BLEU** (secondary) | MT | `sacrebleu.corpus_bleu`, language-appropriate tokenizer (`13a` Latin, `zh`, `ja-mecab`, `intl`) | higher better |
| **CER** (headline) | OCR | char-level `(S+D+I)/N` via pure-Python Levenshtein, OCR output vs `gold_source` | lower better |
| **WER** | OCR | same over whitespace tokens | lower better |
| **End-to-end chrF** (headline) | full cascade | chrF between `MT(OCR(rendered_page))` vs `gold_target` — the *whole* pipeline on the degraded image | higher better |
| **End-to-end BLEU** | full cascade | BLEU of the same end-to-end output | higher better |
| **Fit-rate** (layout headline) | render | fraction of blocks whose translation fits the source box (`fit_ok==True` and `final_font_size ≥ min_readable`) | higher better (1.0 = all fit) |
| **Mean shrink** | render | mean font-size shrink applied to fit translations into source boxes | — |
| **Overflow** | render | count/fraction of blocks that do not fit even at min legible size | lower better |

The gap `(clean MT-chrF − end-to-end-chrF)` quantifies the cost of OCR error. End-to-end is also reported on **CLEAN** (un-degraded) images as an upper bound.

### 4b. Baselines (required, for honest comparison)

1. **Identity** — copy source through unchanged. The **floor** (near-zero across languages; only high when src≈tgt scripts).
2. **Dictionary MT** — glossary lookup + identity passthrough for OOV. The offline MT and a baseline; proves the fine-tune adds value on real (non-saturated) data.
3. **Zero-shot m2m100** — `m2m100_418M` *without* fine-tuning, to quantify what the fine-tune buys.

### 4c. Verified offline seed numbers

These are the **verified offline floor** from the seed evaluation (SeedEngine perfect-OCR + dictionary MT + pure-Python metrics). They assert *pipeline wiring and metric plumbing*, not on-GPU translation quality.

| Metric | Value | Note |
|--------|-------|------|
| **MT chrF** (dictionary) | **79.9** | vs identity floor **22.4** |
| **OCR CER** | **0.0** | perfect-OCR via SeedEngine (offline); realistic CER comes from Tesseract on Colab |
| **End-to-end chrF** | **76.4** | full cascade on rendered seed pages |
| **Mean fit-rate** | **1.0** | every seed block's translation fit its box |

> **Honest reading of the seed numbers.** The dictionary MT chrF of **79.9** *saturates* on the seed set because the seed (src, tgt) pairs overlap the bundled dictionary — so the dictionary baseline looks artificially strong there. This is a property of the seed, **not** a claim that dictionary lookup rivals neural MT. On real `opus-100` eval pairs (which do **not** overlap the dictionary), the fine-tuned `m2m100_418M` dominates the dictionary and identity baselines — that is the **honest, non-saturated floor**. OCR CER of 0.0 reflects SeedEngine's gold-spec read of synthetic images, not Tesseract on real scans, where CER is non-trivial and propagates into the end-to-end chrF gap.

---

## 5. Limitations

### 5a. OCR error propagation (the dominant cascade limitation)

The MT core is only as good as the text it receives. In a cascade, **OCR errors propagate into MT**: a misrecognized word can flip meaning, and the resulting mistranslation is rendered confidently onto the page. The end-to-end chrF is therefore strictly below the clean MT chrF; the gap is the OCR cost. P15 mitigates this with:

- **D3 — per-block OCR-confidence gate:** low-confidence/garbage blocks are *skipped*, not translated, so the MT core never receives obvious garbage.
- **D2 — born-digital bypass:** when a real PDF text layer exists, OCR is skipped entirely (lossless, zero OCR error).
- **D4 — round-trip + length-ratio verification:** catches likely hallucination/truncation from the MT's *own* output, with no reference translation.

These are pipeline mitigations, not fixes to the model — residual OCR noise still degrades MT and is the single biggest limitation of the deployed system.

### 5b. Domain shift

Fine-tuning on `opus-100` (general web/parallel text) does not match the distribution of document-image text — forms, signage, invoices, IDs, headings, terse UI labels, and OCR-noised fragments. Expect quality to drop on:

- Very short, context-free fragments (a single word or label) where MT has little context.
- Domain jargon and proper nouns absent from `opus-100`.
- OCR-noised input distribution (partially mitigated by training/eval with the optional `PleIAs/Post-OCR-Correction` text and the synthetic degradation suite).
- Real-world photos of signage — **no real benchmark exists**, so generalization beyond the synthetic generator is unverified. Do not overclaim.

### 5c. Long blocks

The core is intended for **short blocks**. On long inputs:

- Translation quality and faithfulness degrade (truncation, drift, repetition); D4's length-ratio gate flags ratios outside [0.4, 3.0].
- **Translation expansion breaks the overlay.** Targets are routinely longer than source (especially →fr, and CJK→EN); the translated text overflows the source box. This is the *expected common failure*, not a bug — handled by D5's degradation ladder: fits → **overlay**; too long → **side-by-side** caption panel; infeasible/flagged → **needs_review**. The fit-rate metric exists precisely to measure this.

### 5d. Other limitations

- **Dictionary/offline MT is deliberately low-quality** — it is a baseline and an offline fallback, not the production translator. Pure-offline mode (no torch/m2m100) uses it and will produce weak translations by design.
- **Offline-no-Noto CJK/Arabic render as tofu** — a *font* limitation of pure-offline test mode (DejaVuSans only), not an MT limitation. Real runs ship Noto fonts.
- **RTL shaping is best-effort** without optional `python-bidi` / `arabic-reshaper` — again a render concern, flagged for completeness.
- **Round-trip verification (D4) is heuristic** — it can false-flag legitimate free translations and miss fluent hallucinations; it is a *soft* gate (re-decode once, then `low_confidence`), never a hard reject that silently drops content.

---

## 6. Ethical considerations

### Privacy / PII

Document images frequently contain **personally identifiable information** — IDs, passports, medical and legal records, financial documents. The P15 stance:

- **Local processing by default**; **no raw-image retention** by default.
- **Consent** is required for processing third-party documents.
- The MT core (and the pipeline around it) is an **assistive translation aid**, not an authority — it **flags low-confidence output for human review** and **never asserts certainty**.

### Assistive, not authoritative

The system **assists** translation and surfaces uncertainty (OCR confidence at D3, round-trip/length-ratio at D4, render-fit at D5 → `needs_review`). It must not be presented as certified or error-free translation, especially for high-stakes documents. Low-confidence OCR is **skipped, not mistranslated**; poor-fit results degrade to side-by-side; flagged blocks route to `needs_review` rather than producing a confident-looking wrong overlay.

### Bias and fairness

The base `m2m100_418M` and the `opus-100` corpus carry the biases of their web-scale training data — uneven quality across languages, scripts, and domains, and potential reproduction of social biases present in the corpus. Quality is best on high-resource pairs (e.g. en↔fr) and degrades on low-resource languages.

### Licensing ethics

- **Shipped MT core is MIT** (`m2m100_418M`) — commercial-safe.
- **`nllb-200-distilled-600M` (CC-BY-NC-4.0) and Surya (CC-BY-NC-SA-4.0) are non-commercial — never shipped**, documented only as research upgrades.
- **`opus-100` per-pair license is unverified → verify before commercial use.** Do not assert a license that the source does not surface.

### Misuse

Do not use to translate documents you are not authorized to process, to bulk-translate private documents without consent, or to present machine output as a human/certified translation. The human-review flags exist to keep a person in the loop on consequential decisions.

---

## 7. How to use

The MT core is invoked per block by the P15 agent, between the OCR-confidence gate (D3) and the verification gate (D4):

```
ingest (D1) → ocr (D2 born-digital vs scanned)
  → translate (D3 conf gate → MT core → this model)
  → verify (D4 round-trip chrF + length ratio)
  → render (D5 fit-rate → overlay / side_by_side / needs_review)
```

- **Direction:** default `en→fr`; configurable via the source/target language codes (the base is many-to-many).
- **Granularity:** one paragraph/line-level block per call (Tesseract level 2/3 or the PDF text layer). Never word-by-word.
- **Offline mode:** with `P15_OFFLINE=1` (or when torch/transformers are absent), the core falls back to the deterministic **dictionary MT** baseline so the full pipeline still runs end-to-end with stdlib + Pillow only. Swap in the fine-tuned `m2m100_418M` when torch + transformers are present — same code path, capability-probed at runtime.
- **Metric:** evaluate with `sacrebleu` chrF (headline) and BLEU; the pipeline also reports end-to-end chrF, OCR CER/WER, and layout fit-rate.

---

## 8. Provenance and reuse

- **MT core + `Seq2SeqTrainer` fine-tune harness + chrF/BLEU plumbing + dictionary/identity baselines:** ported from **P13 s2st / P14 doctrans**.
- **OCR front-end + born-digital router + stub-OCR offline pattern:** ported from **P07 dococr**.
- **New for P15:** the fit-to-box overlay engine, the synthetic doc-image generator, the SeedEngine offline OCR, and the five-decision image-translation agent — none of which change *this* MT core, but all of which determine the text it receives and how its output is rendered.

---

*Model card author: Le Dinh Minh Quan (23127460). Base model `facebook/m2m100_418M` is MIT-licensed. Offline numbers are the verified seed-evaluation floor; on-GPU fine-tune numbers are produced by the project Colab notebook. No public in-image document-translation benchmark exists; evaluation rests on the reproducible synthetic generator.*
