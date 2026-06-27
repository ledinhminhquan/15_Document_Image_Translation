# P15 — Layout Fidelity and Evaluation

> The evaluation methodology for **Document-Image Machine Translation** (`imgtrans`).
> Author: Le Dinh Minh Quan (student 23127460).
> Scope: how every stage of the **OCR → MT → render** cascade is measured, what each number means, what the baselines are, and how to read the offline floor against the full Colab run.

This is the project's *special quality document*. P15 is a cascade in which only the **MT core** (`facebook/m2m100_418M`, MIT) is trained; OCR (Tesseract / `pytesseract`) and the render/overlay engine (Pillow) are pretrained or algorithmic. Because there is **no public in-image / document-image translation benchmark with gold parallel text** (datasets research returned `null`), the primary evaluation data is a **reproducible synthetic generator** (`data/synth_render.py`) that renders source sentences onto page images and embeds the gold layout spec — yielding `(image, gold_source, gold_target, boxes)` quadruples. Those quadruples are what make every metric below computable by construction.

---

## 1. Why a bespoke evaluation methodology

In-image translation is not one task with one score. It is a cascade of three sub-systems, each of which can fail independently, plus a spatial-rendering problem that has no analogue in plain text MT:

- **OCR can mis-read** the page → garbage flows into MT (error propagation).
- **MT can mistranslate, hallucinate, or truncate** → wrong text, correctly placed.
- **The render can be geometrically infeasible** — the translation is correct but does not fit the source box (French is routinely longer than English; CJK→EN expands hard).

A single headline number would hide which stage broke. So P15 measures **four families** of metrics, each isolating a different failure mode:

| Family | Question it answers | Stage isolated |
|--------|---------------------|----------------|
| **MT quality** (chrF / BLEU) | Is the translation good *given clean text*? | MT only (fed gold source) |
| **OCR quality** (CER / WER) | Did we read the page correctly? | OCR only (vs gold source) |
| **End-to-end** (image-translation chrF / BLEU) | Is the whole pipeline good *on a real degraded image*? | OCR + MT composed |
| **Layout fidelity** (fit-rate / mean-shrink / overflow) | Does the translation actually *fit back into the layout*? | render / overlay |

The gap between **clean MT chrF** and **end-to-end chrF** is a direct, interpretable measurement of the cost of OCR error. The layout-fidelity family is the one that is unique to in-image translation and is the project's headline deliverable.

The synthetic generator is what makes all four families exact. For sample `i` it emits `(image.png, gold_source, gold_target, boxes_json)` deterministically (`rng = random.Random(BASE_SEED * 1_000_003 + i)`), so the same index always yields the same image. Because we *know* the gold source text, gold translation, and gold boxes, we can score OCR against gold source, MT against gold target, and rendered boxes against gold boxes with zero ambiguity.

---

## 2. MT quality — chrF (headline) and BLEU

The MT core is the only trained stage, so its intrinsic quality is reported on **clean gold source**, deliberately bypassing OCR so the number isolates translation quality from reading error.

- **chrF (primary).** `sacrebleu.CHRF` between `MT(gold_source_text)` and `gold_target_text`. Character-n-gram F-score (n = 1..6, β = 2), range **0–100**, higher is better. chrF is the headline because it is tokenizer-light, robust to morphology, and degrades gracefully on the partial-credit cases that dominate document translation — the same choice made in P13 (s2st) and P14 (doctrans), so the numbers are directly comparable across projects.
- **BLEU (secondary).** `sacrebleu.corpus_bleu` with a language-appropriate tokenizer (`13a` for Latin scripts, `zh`, `ja-mecab`, `intl` otherwise). Reported alongside chrF for continuity with the wider MT literature; treated as secondary because BLEU is brittle on short document blocks.

Both are computed at the **block level** (Tesseract `level=2/3` paragraph boxes), never word-by-word — word-by-word translation destroys meaning and is explicitly out of scope.

**Offline note.** When `sacrebleu` is absent, a pure-Python chrF (char n-gram precision/recall F, n = 1..6, β = 2) is used; a tolerance test asserts it agrees with the real `sacrebleu` chrF when both are installed.

---

## 3. OCR quality — CER (headline) and WER

OCR is a pretrained front-end (Tesseract via `pytesseract.image_to_data`, giving word boxes + per-word confidence). It is **never trained**; we only measure it, because its errors propagate into MT.

- **CER (primary).** Character Error Rate = `(S + D + I) / N_chars` — substitutions + deletions + insertions over reference character count, via a pure-Python two-row Levenshtein (zero dependencies). Lower is better; `0.0` is perfect. (Verified: `kitten` → `sitting` gives CER 0.5.) Optional whitespace/case normalization is available.
- **WER (secondary).** Same edit-distance metric over whitespace-delimited tokens. (Verified: `"the cat sat"` vs `"the cat sit"` → WER 0.333.)

Reference = `gold_source_text` from the manifest. CER is the headline for OCR because document text is character-dense and a single misread character can corrupt downstream MT; WER is the coarser, more human-readable companion.

**Two OCR regimes you will see in the numbers:**
- **SeedEngine / stub OCR (offline):** for a synthetic image with an available manifest, OCR returns `manifest[i].src` directly (reads the embedded gold spec, not pixels), so CER vs gold is **0.0** — a *perfect-OCR upper bound*, not a claim about real reading. An optional seeded corruption mode (~3% char substitution per index) produces a *known, reproducible* non-zero CER to exercise the error-rate code.
- **Tesseract (Colab / real run):** reads pixels off the degraded image; CER is realistically non-zero and rises with blur, noise, rotation, and JPEG recompression. This is the honest OCR number.

---

## 4. End-to-end image-translation quality (headline)

This is the metric that scores the **whole pipeline on a real rendered, degraded page** — the number that matters for the actual product.

- **End-to-end chrF / BLEU.** chrF (and BLEU) between `MT(OCR(synthetic_image))` and `gold_target_text`. The image is the rendered, degradation-applied page; OCR reads it, MT translates the OCR output, and the result is compared to the gold translation.

The single most informative derived quantity is the **OCR-cost gap**:

```
OCR cost  =  (clean MT chrF)  −  (end-to-end chrF)
```

A large gap means OCR error is the bottleneck; a small gap means MT is the bottleneck. We additionally report end-to-end chrF on **CLEAN** images (all degradations off) as an upper bound — the gap between clean-image and degraded-image end-to-end chrF isolates the cost of the degradation suite specifically.

---

## 5. Layout fidelity — the metric family unique to P15

Layout fidelity asks the question no plain-MT metric can: *once translated, does the text fit back into the page?* It is computed purely from geometry (the `fit_box` binary-search result and the rendered bounding boxes), so it is **fully meaningful even offline** — no model, no OCR quality required.

The render engine's `fit_box(text, font, box_w, box_h)` returns `(best_size, best_wrapped, fit_ok)`: it binary-searches the largest font size at which a greedy word-wrap of the translation fits inside the source box. The layout metrics read directly off that result.

### 5.1 Fit-rate (dominant signal)

> **fit-rate = fraction of blocks where `fit_ok == True` and `final_font_size >= min_readable`.**

`1.0` means every translated block fit inside its original box at a legible size. This is the dominant overlay signal because the most common in-image failure is **translation expansion** (target longer than source — especially en→fr and CJK→EN) overflowing the box. Fit-rate exposes exactly that.

### 5.2 Mean-shrink

> **mean-shrink = mean over blocks of `final_font_size / source_font_size`** (≤ 1.0).

How much the renderer had to *shrink* the text to make it fit. `1.0` = no shrink needed; lower values mean the overlay is making text smaller (and harder to read) to fit longer translations. Mean-shrink is the legibility-cost companion to fit-rate: a fit-rate of 1.0 achieved only by shrinking everything to the minimum size is worse than a 1.0 achieved at full size, and mean-shrink surfaces that distinction.

### 5.3 Overflow

> **overflow = fraction of blocks that did NOT fit even at `min_size`** (`fit_ok == False`).

The complement of the feasible set: blocks where no legible size fits the box. These are the blocks that drive the agent's D5 decision away from `overlay` toward `side_by_side`. Overflow is reported as a rate so it is comparable across pages of different block counts.

### 5.4 Optional geometric extras

Reported as components when the full geometric suite is enabled (they are pure geometry, dependency-free):

- **box-retention (mean IoU):** mean IoU between each rendered translation's actual drawn bbox (`multiline_textbbox` of the wrapped target, offset to the box origin) and the original source box. Measures how well the overlay stays *inside* the original footprint.
- **no-overlap rate:** `1 − (#new overlapping rendered pairs / #originally-disjoint pairs)` — penalizes long translations bleeding into neighbor blocks. The erase-all-then-draw ordering rule exists to keep this high.
- **aggregate layout score (optional):** `mean(fit_rate, mean_box_IoU, no_overlap_rate)`, fit-rate weighted highest; always report components separately too.

---

## 6. Baselines

Three baselines establish the floors against which the fine-tuned MT core is judged. Without them, a chrF of 76 is meaningless — you cannot tell whether the model learned anything or whether the task was trivial.

| Baseline | What it does | Role | Expected behavior |
|----------|--------------|------|-------------------|
| **Identity** | Pass source text through unchanged (`tgt := src`) | The absolute **floor** | chrF near-zero across different scripts; non-trivial only when src ≈ tgt. The hard floor any real system must beat. |
| **Dictionary MT** | Per-token glossary lookup + identity passthrough for OOV (`' '.join(dict.get(tok.lower(), tok) ...)`) | Offline MT **and** baseline | Deterministic, instant, no download. Proves the fine-tune adds value over naive word substitution. Doubles as the offline-mode translator. |
| **Zero-shot m2m100** | `facebook/m2m100_418M` **without** fine-tuning | Quantifies the fine-tune's value | Whatever the pretrained checkpoint scores; the fine-tune's lift = `(fine-tuned chrF − zero-shot chrF)`. |

The dictionary baseline is special: it is reused as **both** the comparison floor **and** the offline fallback translator (the same code path P13/P14 used). This is why, on the synthetic *seed* pages, the dictionary can look deceptively strong — see §8.

---

## 7. Verified seed numbers and the metrics table

The numbers below are the **verified offline seed evaluation** — the deterministic floor that CI and the smoke notebook reproduce with stdlib + Pillow only (SeedEngine stub OCR + dictionary MT + DejaVu fonts + pure-Python metrics). They assert *pipeline wiring and metric plumbing*, not translation quality.

| Metric | Family | Value (offline seed) | Direction | Reading |
|--------|--------|----------------------|-----------|---------|
| **MT chrF (dictionary)** | MT quality | **79.9** | higher better | Dictionary MT on the seed pairs. Saturated — see §8. |
| MT chrF (identity floor) | MT quality | **22.4** | higher better | The floor. Dictionary beats it by **+57.5 chrF**, proving translation is happening, not copying. |
| **OCR CER** | OCR quality | **0.0** | lower better | Perfect-OCR via SeedEngine (reads embedded gold). Realistic/non-zero with Tesseract on Colab. |
| **End-to-end chrF** | End-to-end | **76.4** | higher better | `MT(OCR(image))` vs gold target on the rendered page. OCR-cost gap here = `79.9 − 76.4 = 3.5` chrF. |
| **Mean fit-rate** | Layout fidelity | **1.0** | higher better | Every translated block fit its box. Expected on synthetic — boxes were generated by the *same* wrap algorithm the renderer uses. |

How to read the headline contrast at a glance: **dictionary chrF 79.9 vs identity 22.4** is the honest "does the system translate?" signal on the seed; **OCR CER 0.0** is the perfect-OCR ceiling; **end-to-end chrF 76.4** shows the full pipeline holds up on a rendered page; **fit-rate 1.0** shows the overlay is geometrically feasible on synthetic layouts.

---

## 8. How to read these numbers honestly (offline floor vs Colab story)

These are good numbers, but they must be read with the right caveats. The single most important point of intellectual honesty in this project:

> **The dictionary MT saturates on the seed because the seed (src, tgt) pairs overlap the dictionary's own glossary.** On the seed, dictionary chrF (79.9) therefore *looks* competitive with — even ahead of — a fine-tuned model. **This is an artifact, not a result.** On real held-out `opus-100` en→fr eval pairs (out-of-vocabulary for the glossary), the dictionary collapses and the **fine-tuned m2m100 dominates** — that is the honest, non-saturated floor.

Three more reading rules:

1. **OCR CER 0.0 is a ceiling, not a claim.** It comes from SeedEngine reading the embedded gold spec, not from reading pixels. The *realistic* OCR number is whatever Tesseract produces on the degraded image in Colab — non-zero, and rising with blur/noise/rotation/JPEG. Always pair the offline `0.0` with the Colab Tesseract CER when reporting.
2. **Fit-rate 1.0 is expected on synthetic, and is not a generalization claim.** The synthetic boxes were laid out with the *same* greedy pixel word-wrap the renderer uses, so source text fits by construction; translation expansion is the only thing that can break it, and on the seed the expansion is mild. On real photos of signage, expect fit-rate well below 1.0 and the D5 side-by-side branch to fire — that is the designed-for common case, not a bug.
3. **The offline floor tests plumbing; Colab tests quality.** The offline stack (stub OCR, dictionary MT, DejaVu, pure-Python chrF) deliberately produces *low, stable* quality numbers whose job is to prove the cascade is wired correctly and every metric computes. The geometry metrics (CER/WER mechanics, fit-rate, mean-shrink, overflow, IoU, no-overlap) are **fully meaningful offline**. Translation *quality* (the real chrF lift from the fine-tune) is only established on Colab with the real corpus and real OCR.

| Aspect | Offline floor (CI / stub) | Colab / real run |
|--------|---------------------------|------------------|
| OCR | SeedEngine reads gold → CER 0.0 | Tesseract reads pixels → realistic CER > 0 |
| MT | Dictionary (saturates on seed) | Fine-tuned m2m100, dominates on held-out opus-100 |
| Fonts | DejaVuSans (CJK/Arabic → tofu, flagged) | Noto split-by-script, correct glyphs |
| Metrics | Pure-Python chrF / Levenshtein | sacrebleu chrF/BLEU + jiwer-equivalent |
| What it proves | Wiring + metric plumbing + geometry | Translation quality + real OCR cost |

---

## 9. How layout fidelity feeds the agent (D4 → D5)

Layout-fidelity measurement is not just reporting — it is the input to the agent's render decision, so evaluation and runtime share one computation:

- **D4 (translation verification)** emits a target/source **length-ratio** sanity signal (accept band `[0.4, 3.0]`); an over-long translation is fed *forward* to D5 because it predicts box overflow.
- **D5 (render-fit feasibility)** consumes `fit_box`'s returned `fit_ok` and the D4 length-ratio to choose:
  - **fits at ≥ min legible font → `overlay`** (the full camera experience: erase + re-render in-box);
  - **does not fit (overflow) → `side_by_side`** (original untouched + translated caption panel);
  - **fit/inpaint fails entirely, or block was D3/D4-flagged → `needs_review`** (emit boxes + raw translation, no destructive render).

So the **overflow** metric directly predicts the rate of `side_by_side` outputs, and **fit-rate** predicts the rate of clean `overlay` outputs, on any given corpus. A low-fit-rate corpus is not a failed run — it is a corpus the agent correctly routes to side-by-side. This is the self-checking degradation ladder that distinguishes P15 from a plain "OCR → translate → print the string" script.

---

## 10. Reproducing the evaluation

- **Offline floor (no downloads):** runs with stdlib + Pillow only. SeedEngine stub OCR + dictionary MT + DejaVu + pure-Python chrF reproduce the §7 table deterministically (`P15_OFFLINE=1` pins stub mode). This is what CI and the smoke notebook cell execute.
- **Colab / real:** install `pytesseract` + tesseract binary, `transformers` + `torch` (fine-tuned m2m100), `sacrebleu`, and the Noto fonts; point the generator at real `opus-100` en→fr pairs. Capability probes (`shutil.which('tesseract')`, `try import`) upgrade each stage automatically — **same code path**, no test changes.
- **Scale:** 200–2000 synthetic samples for eval (streamable for more); the committed 3–5-sample fixed-seed PNG fixtures back the unit tests.

### Dataset / license flags (evaluation-relevant)

- The MT fine-tune corpus `Helsinki-NLP/opus-100` (en-fr) license is **unknown per-pair → verify before commercial use**.
- **Do not ship** non-commercial models even though they would raise quality: `facebook/nllb-200-distilled-600M` (**CC-BY-NC-4.0**) and **Surya** (`vikp/surya_rec2` + `surya_det3`, **CC-BY-NC-SA-4.0**). They may appear in research-quality comparison tables only, clearly labeled non-commercial.
- Optional real OCR-noise text source `PleIAs/Post-OCR-Correction` (english) is **CC0** (safe).
- Everything in the shipped, scored default stack — `facebook/m2m100_418M` (MIT), Tesseract/`pytesseract` (Apache-2.0), Pillow, Noto fonts (OFL 1.1) — is fully permissive.

---

## 11. Summary

P15 is evaluated along four independent axes so that no failure mode hides behind a single score: **MT chrF/BLEU** (translation quality on clean text), **OCR CER/WER** (reading accuracy), **end-to-end image-translation chrF/BLEU** (the whole pipeline on a degraded page, with the OCR-cost gap making error propagation explicit), and **layout fidelity** (fit-rate / mean-shrink / overflow — the metrics unique to in-image translation that drive the agent's overlay → side-by-side → needs_review ladder). The verified offline seed floor (dictionary chrF **79.9** vs identity **22.4**, OCR CER **0.0**, end-to-end chrF **76.4**, fit-rate **1.0**) proves the cascade is wired and every metric computes; the honest quality story lives on Colab, where the dictionary's seed saturation disappears and the fine-tuned m2m100 dominates on held-out `opus-100`.
