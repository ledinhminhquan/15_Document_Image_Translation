# Privacy & Robustness — P15 Document-Image Machine Translation

> Package: `imgtrans` · Author: Le Dinh Minh Quan (23127460)
> Scope: translate text that appears *inside* an image / scanned document / born-digital PDF and render the translation back onto the page (`OCR → MT → layout-preserving overlay`).

This document covers two failure surfaces that matter more for in-image document translation than for plain text MT:

1. **Privacy** — the inputs are *pictures of documents*, which routinely carry the most sensitive personally identifiable information a person owns (IDs, passports, medical records, contracts, bank statements). The system is designed to keep those pixels under the user's control.
2. **Robustness** — the input is *uncontrolled pixels* (blurred phone photos, skewed scans, multi-column layouts, mixed scripts), and OCR errors propagate downstream into the MT core. The agentic state machine exists precisely to catch each class of degradation and degrade gracefully instead of emitting a confident-but-wrong translated image.

Both concerns share one design principle: **the tool assists translation and flags uncertainty for human review; it never asserts certainty over a sensitive document.**

---

## 1. Privacy

### 1.1 Why document images are a high-risk input class

P15's input is not anonymous web text. A document image is frequently a photograph or scan of:

- **Identity documents** — national ID cards, passports, driver's licences, residence permits.
- **Medical and health records** — prescriptions, lab results, discharge summaries, insurance cards.
- **Legal documents** — contracts, court filings, notarised letters, immigration paperwork.
- **Financial documents** — bank statements, payslips, tax forms, invoices.

These are exactly the documents a user would reach for the "camera translate" use case (e.g. a traveller translating a foreign medical form, or an immigrant translating a contract). The threat is therefore concrete: **the raw image, the OCR'd text, and the rendered overlay can each leak the same PII**, and the OCR step turns pixels into machine-readable strings that are trivially indexable and exfiltratable.

Because the PII risk is intrinsic to the task and not incidental, privacy is treated as a first-class design constraint, not an add-on.

### 1.2 Local / on-device processing by default

Every shipped component of the cascade runs **locally** with no network egress at inference time:

| Stage | Component | Network at inference |
|-------|-----------|----------------------|
| OCR | Tesseract via `pytesseract` (system binary) | none — local binary |
| Born-digital routing | PyMuPDF `get_text()` probe | none — local parse |
| MT | `facebook/m2m100_418M` (MIT), local weights | none — local inference |
| Erase / render | Pillow fit-to-box overlay | none — pure algorithm |
| Metrics | `sacrebleu` / pure-Python fallback | none |

The only network access in the whole project is at **setup time** (downloading model weights, Noto fonts, and the parallel corpus in the Colab/Docker setup cell). Once those artifacts are cached, **a document image is processed without leaving the machine.** This is a deliberate contrast to cloud OCR/translation APIs, where the sensitive image is uploaded to a third party.

The FastAPI service and Gradio UI are likewise designed to run **on the user's own host** (or their own private deployment). When self-hosted, no document image ever transits an external service.

### 1.3 No raw-image retention by default

The pipeline is **stateless with respect to user content** by default:

- The `/translate-image` route accepts an upload, processes it in memory, returns the translated text plus a base64 overlay PNG **in the HTTP response**, and **does not persist the input image, the OCR text, or the output image to disk.**
- No request body, OCR result, or rendered page is written to a log, cache, or database as part of normal operation.
- Temporary buffers (the in-memory `BytesIO` used for JPEG re-encode in the synthetic generator, the throwaway `ImageDraw` canvases in the render engine) are scoped to the request and garbage-collected; they are never the synthetic-vs-user distinction made permanent.
- Any debugging artifact retention (e.g. saving intermediate crops) is **opt-in** and off by default, and must be documented to the operator before it is enabled.

The synthetic dataset is the *only* image data that lives on disk in the repo (the committed fixture PNGs in §3c of the design brief), and that data is generated, contains no real PII, and exists solely for reproducible tests.

### 1.4 The LLM brain is OFF by default

The agent in `src/imgtrans/agent/` is a **deterministic finite-state machine** (five decision points D1–D5). It needs no LLM to function.

An **optional advisory LLM brain (`anthropic`)** can be attached, but:

- It is **OFF by default** and must be explicitly enabled.
- It is **advisory only** — it can comment on a routing decision but **never rewrites the translation, never overrides a gate, and never sees a decision it can silently change.**
- When off (the default), **no document text or image is sent to any external LLM API**, preserving the local-only privacy guarantee.

This is the single most important privacy lever for a PII-bearing document tool: the default configuration sends nothing to a hosted model. Enabling the LLM brain is an informed, operator-level choice that trades the local-only guarantee for advisory commentary, and that trade-off is documented at the point of configuration.

### 1.5 Consent and operator obligations

Because the inputs are sensitive, the project documents an explicit consent posture:

- **Consent to process.** The operator/user must have the right to process the document they submit. The tool is for translating documents the user is entitled to read (their own records, documents shared with them) — not for harvesting third-party PII.
- **Purpose limitation.** Translated output exists to help the user *understand* a document, not to build a corpus of others' personal data.
- **Transparency.** The UI and API responses make clear that translation is machine-generated and may contain errors (see §1.7), so a user never mistakes an overlay for a certified translation of, say, a legal contract.

### 1.6 Redaction and minimisation

P15's architecture gives several natural minimisation points:

- **Block-level processing.** Because OCR returns block/paragraph boxes (Tesseract `level=2/3`), the operator can choose to **process only selected blocks** and leave the rest untouched, rather than OCR-ing the entire page. A user translating one paragraph of a passport need not extract the document number.
- **The confidence gate (D3) drops, it does not store.** Low-confidence or garbage blocks are **dropped and tagged `needs_review`** — they are never translated and never propagate text downstream. This incidentally avoids materialising garbled-but-still-sensitive strings.
- **Whiteout erase removes source pixels.** The default overlay path erases each source box (median border-ring fill) before drawing the translation. For redaction-style use, the erase step can be used to **suppress** a region rather than re-render it, producing a page with the sensitive source text removed.
- **No-retention default (§1.3)** is itself the strongest minimisation: data that is never stored cannot be leaked.

### 1.7 The tool assists; it never asserts certainty

For PII-bearing legal/medical/financial documents, an over-confident wrong translation is a *harm*, not just a quality miss. P15 structurally refuses to assert certainty:

- **D4 round-trip + length-ratio verification** flags likely hallucinations or truncation as `low_confidence` instead of presenting them as fact.
- **D5 feasibility gate** falls back to **side-by-side** (original preserved, translation in a caption panel) when the translation cannot be faithfully overlaid, so the *original* sensitive text is never destroyed by a bad fit.
- **`needs_review`** is a first-class output: when OCR confidence is too low (D3) or rendering is infeasible, the system emits the boxes and raw translation **without a destructive render** and tells the human to check it.

This degradation ladder means the worst case for a sensitive document is "the tool says *I'm not sure, please review*", not "the tool silently produced a confident, wrong, official-looking translated image."

### 1.8 Privacy summary

| Risk | Mitigation in P15 |
|------|-------------------|
| Document PII uploaded to a third party | Local/on-device cascade; self-hosted API/UI; no inference-time network egress |
| Sensitive image/text retained and later leaked | No raw-image / OCR-text / overlay retention by default; stateless requests |
| External LLM sees the document | LLM brain OFF by default; advisory-only when on; never rewrites |
| Over-extraction of PII | Block-level processing; D3 drops low-confidence blocks; whiteout supports redaction |
| Confident wrong translation of a legal/medical doc | D4/D5 gates + `needs_review`; tool assists and flags, never asserts certainty |

---

## 2. Robustness

The input is uncontrolled pixels. Robustness in P15 means: for each realistic way a document image can be hard to read, there is (a) a preprocessing or routing response that tries to recover, and (b) a **gate that catches the failure and degrades gracefully** when recovery is not possible — so the system never spills garbled text out of a box and calls it done.

The synthetic generator (`data/synth_render.py`) deliberately injects many of these degradations (rotation, blur, pixel noise, brightness/contrast jitter, JPEG recompression) so robustness is **measured**, not assumed. A **CLEAN mode** (all degradations off) provides the OCR/end-to-end upper bound, and the gap to the degraded score quantifies real-world cost.

### 2.1 Degraded scans (blur, rotation, skew, low DPI, noise)

**The problem.** Phone photos and cheap scanners introduce motion/focus blur, page rotation and skew, low effective resolution, sensor noise, uneven lighting, and JPEG artifacts. Each lowers OCR character accuracy, and OCR errors then propagate into the MT core (§2.4).

**What P15 does.**

- **Modelled, not hand-waved.** The synthetic generator composes degradations deterministically per index: rotation `uniform(-3,3)°` with `expand=True`, Gaussian blur (radius up to 1.0), Gaussian pixel noise (σ up to 8), brightness/contrast jitter `uniform(0.85,1.15)`, and JPEG recompression at quality down to 60. This produces a degraded test bed with **known gold text and boxes**, so CER/WER and end-to-end chrF are measured against truth.
- **Confidence-gated recovery (D3).** Tesseract's per-block mean confidence (0–100) is read from `image_to_data`. When a block lands in the **mid band (≈40–75)**, the agent runs a **retry path — gamma correction / upscale / invert, then re-score** — before giving up. This directly targets low-DPI and low-contrast scans.
- **Drop, never guess.** When confidence stays below `LOW` (≈40) or the block is empty / fails the alpha-ratio and `min_text_length` sanity check, the block is **dropped and tagged `needs_review`.** Garbage from an unreadable smudge is never translated.
- **Metric exposure.** The `(clean MT-chrF − end-to-end-chrF)` gap is reported explicitly as the OCR-error cost, so scan degradation shows up as a number rather than a silent quality loss.

> Note: the shipped synthetic degradation is intentionally **mild and OCR-survivable** (the brief's stance). Severe blur/skew on real photos is handled at runtime by the D3 retry + drop ladder, not by claiming the synthetic suite covers all photographic conditions — see the "do not overclaim real signage" risk in the design brief.

### 2.2 Multi-column and complex layouts

**The problem.** Newspapers, forms, and contracts use multiple columns; naïve OCR can read across column boundaries and scramble sentence order, and word-by-word handling destroys meaning and wrecks line wrapping.

**What P15 does.**

- **Block/paragraph boxes, not word boxes.** The pipeline aggregates to Tesseract block/paragraph boxes (`level=2/3`) before MT. Translating per-word destroys meaning and ruins wrapping; translating per-block keeps a coherent unit and gives the render engine room to wrap. This is a hard rule, not a preference.
- **Per-block independent rendering.** Each block is translated, erased, and re-rendered in its own box. Multi-column pages are handled as **many independent single-column blocks**, which is exactly the geometry the fit-to-box engine expects.
- **Erase-all-then-draw ordering.** Every source box is erased *before any* translation is drawn, so an enlarged translation that spills slightly never paints over a not-yet-erased neighbouring column.
- **No-overlap metric.** Layout fidelity includes a **no-overlap rate** (`1 − new-overlapping-rendered-pairs / originally-disjoint-pairs`), which penalises a long translation in one column bleeding into the next — making multi-column bleed measurable.

### 2.3 Mixed scripts and non-Latin / RTL text

**The problem.** A single page can mix scripts (Latin captions over a CJK body, an Arabic form with English fields). The translation may also be in a script whose width/shaping differs from the source, and RTL languages need right alignment and shaping.

**What P15 does.**

- **Target-language-aware font routing.** The render engine scans the *target* codepoints, picks the majority Unicode script, and resolves a `{script: path}` Noto font map (Latin/Cyrillic/Greek, CJK, Arabic, Devanagari, Hebrew, Thai). One CJK file (`NotoSansCJK-Regular.ttc`) covers zh/ja/ko.
- **Per-character wrap for no-space scripts.** Greedy word-wrap splits on spaces for Latin; **CJK (no spaces) wraps per character** — break when the next char exceeds `max_w`. This prevents a single un-wrappable "word" from overflowing.
- **RTL handling.** Arabic/Hebrew render with `align='right'` anchored at the box right edge; correct shaping uses optional `python-bidi` + `arabic-reshaper` when available, else the logical string is drawn and shaping is **flagged as best-effort** (honest about the limitation rather than silently mis-shaping).
- **Glyph-coverage fallback.** Any codepoint uncovered by the chosen Noto font (or in pure-offline mode with no Noto installed) degrades to `DejaVuSans.ttf` shipped inside Pillow, then to `ImageFont.load_default()` — **without crashing.** In pure-offline mode CJK/Arabic render as tofu boxes; this is **flagged** as affecting offline test mode only (real runs ship Noto).

### 2.4 OCR error propagation into MT

**The problem.** This is the defining robustness risk of a cascade: an OCR misread becomes the MT input, and the MT core may then confidently "translate" a corrupted string into fluent nonsense. Errors compound across stage boundaries.

**What P15 does — three independent firewalls.**

1. **Eliminate the error where possible (D2 born-digital bypass).** For born-digital PDFs, the PyMuPDF coverage probe (reused verbatim from P07) detects a real text layer and **bypasses OCR entirely**, extracting the digital text and word boxes losslessly. On the common digital-PDF case, OCR error is **zero** because OCR never runs.
2. **Stop bad text before MT (D3 confidence gate).** Low-confidence/garbage blocks are dropped before translation (§2.1). The MT core only ever sees text OCR was reasonably sure about, so the "fluent translation of garbage" failure is cut off at the source.
3. **Catch drift after MT (D4 round-trip + length ratio).** Two reference-free signals validate each MT output: **round-trip back-translation chrF** (translate back to source, compare) and **target/source length ratio** (∈ [0.4, 3.0]). Round-trip below `TAU` (likely hallucination/drift) triggers **one re-decode with alternate params**, then a `low_confidence` flag; an out-of-band length ratio (truncation or runaway repetition) is flagged **and fed forward to D5**, since an over-long translation will not fit the box.

The result: OCR noise is bypassed when avoidable, blocked when detectable pre-MT, and caught as drift post-MT — three chances to stop a corrupted string from becoming a confident wrong overlay.

> D4 is explicitly a **soft** gate: round-trip chrF can false-flag legitimate free translations and can miss fluent hallucinations. It re-decodes once and then flags `low_confidence`; it **never hard-rejects and silently drops content.** This avoids the failure mode where a heuristic quietly deletes a correct translation.

### 2.5 Translation expansion breaking the overlay

**The problem.** Target text is routinely longer than source (notably en→fr, and CJK→EN expansion), so the translation does not fit the original box. This is the *expected common case*, not a bug.

**What P15 does.**

- **Fit-to-box with shrink (Step C).** A binary search over font size plus greedy wrap finds the largest legible size at which the translation fits, returning `fit_ok`. Complexity is `O(log max_sz)` measurements per block — negligible.
- **D5 feasibility gate drives graceful degradation.** Using the translated char count, box dimensions, and the D4 length ratio:
  - fits at ≥ minimum legible font → **OVERLAY** (the full camera experience);
  - does not fit (translation too long) → **SIDE-BY-SIDE** (original untouched + translated caption panel), so nothing is clipped into illegibility and the source is preserved;
  - infeasible or D3/D4-flagged → **`needs_review`** (boxes + raw translation, no destructive render).
- **fit-rate metric.** Layout fidelity reports the fraction of blocks with `fit_ok==True` at ≥ minimum readable size, so expansion-driven overflow is quantified rather than hidden.

### 2.6 Adversarial, garbled, and out-of-distribution input

**The problem.** Real input includes blank pages, decorative non-text glyphs misread as characters, watermarks, handwriting, and outright noise — anything that produces a string that is not meaningful source text.

**What P15 does.**

- **Magic-byte input routing (D1).** Unsupported / malformed inputs are routed to `needs_review` up front based on file magic bytes + MIME + extension, not on a fragile extension-only guess. A `.txt` is accepted as a degenerate input that skips OCR.
- **Alpha-ratio + length sanity (D3).** Beyond raw confidence, blocks must pass an alpha-ratio and `min_text_length` check, filtering symbol soup and stray punctuation marks that OCR sometimes hallucinates from texture.
- **Length-ratio sanity (D4).** Runaway repetition (a known seq2seq failure mode on garbled input) blows the length ratio out of band and is flagged, then routed to side-by-side / `needs_review` rather than rendered.
- **Page-quality gate at ingest (D1).** A page-quality check at the ingest stage screens degenerate pages before they consume OCR/MT compute.

The net adversarial posture: a page the system cannot read confidently produces an honest `needs_review`, never a confidently mangled overlay.

### 2.7 Deterministic, reproducible failure handling

Robustness claims are only credible if they reproduce:

- **Per-index seeded RNG.** The generator and the offline stub-OCR corruption use *only* the per-index seed (`BASE_SEED * 1_000_003 + i`), never global `random`/`numpy` state. The same index always yields the same image and the same injected errors, so CER scoring and fixture tests are exactly reproducible. This determinism discipline is a stated robustness requirement (a leak to global state breaks reproducible scoring).
- **Offline degradation is total.** With stdlib + Pillow only — no Tesseract binary, no torch, no Noto — every stage falls back to a deterministic stub (seed-text stub OCR, dictionary MT, DejaVu fonts, pure-Python chrF/CER) selected automatically by capability probes (`shutil.which('tesseract')`, `try import`). `P15_OFFLINE=1` pins stub mode. The pipeline `image → OCR(stub) → MT(dict) → render(DejaVu) → metrics` runs end-to-end with nothing downloaded, so CI verifies the wiring even when no heavy component is present. **Same code path** — the probes upgrade each stage when the real component appears, with no test changes.
- **Pillow version floor.** `multiline_textbbox` / `textlength` require Pillow ≥ 9.2 (verified on 12.2.0); the floor is pinned so the fit algorithm behaves identically.
- **T4 memory fallback.** If `m2m100_418M` fine-tuning OOMs on a free T4 even with fp16 + small batch + grad-accum, the documented fallback is `opus-mt-en-fr` for the en→fr-only demo, with the loss of multilingual coverage noted rather than silently dropped.

### 2.8 Robustness summary — degradation → response → gate

| Degradation | Recovery attempt | Gate / graceful fallback |
|-------------|------------------|--------------------------|
| Blur / skew / low DPI / noise | D3 retry: gamma / upscale / invert + re-score | conf < LOW → drop block, `needs_review` |
| Multi-column / complex layout | Block/paragraph boxes; per-block render; erase-all-then-draw | no-overlap metric; side-by-side on bleed |
| Mixed scripts / RTL / CJK | Target-aware Noto routing; per-char wrap; bidi shaping | DejaVu → `load_default` glyph fallback (no crash) |
| OCR error → MT | D2 born-digital bypass; D3 pre-MT gate | D4 round-trip + length ratio → `low_confidence` |
| Translation expansion | Fit-to-box binary search (shrink-to-fit) | D5: overlay → side-by-side → `needs_review` |
| Adversarial / garbled input | D1 magic-byte routing; D3 alpha/length sanity | unsupported / unreadable → `needs_review` |
| Missing components (offline) | Capability probes auto-select stubs | deterministic stub OCR / dict MT / DejaVu |

---

## 3. Cross-cutting principle

Privacy and robustness converge on the same guarantee: **for a sensitive, hard-to-read document, P15 prefers an honest `needs_review` over a confident wrong answer.** Local processing keeps the document under the user's control; the D1–D5 gate ladder ensures the worst output is a flagged one a human can check — never a destructive, official-looking, silently-wrong translated image of someone's passport, prescription, or contract.
