# P15 Document Image Machine Translation — Design Brief

> Status: spec (build from this).
> One-line scope: translate the text that appears *inside* an image and render the translation back into the same layout — a cascade **OCR → MT → render** where the **only trainable component is the MT core** (`facebook/m2m100_418M`, MIT).

---

## 1. Problem and scope

**In-image machine translation** ("camera translate" / "manga translate" / Google Translate camera) takes an **image that contains text** and produces a **new image** in which the text has been translated into the target language *while preserving the original layout*. It is distinct from plain document translation (text in → text out) because the input and output are pixels and the translated text must be drawn back where the source text was.

This is a **cascade**, not an end-to-end model:

```
image ──▶ [OCR] ──▶ source text + word/block boxes + conf
                         │
                         ▼
                      [MT]  (the trainable core)
                         │
                         ▼
              translated text ──▶ [RENDER/OVERLAY] ──▶ output image
```

**Decisive scope decisions:**

- **The trainable core is MT only.** OCR is a pretrained system library (Tesseract). The render/overlay engine is pure algorithm (Pillow). We fine-tune exactly one component — `facebook/m2m100_418M` — with the HF `Seq2SeqTrainer` stack reused verbatim from P13/P14. Everything else is pretrained or deterministic code.
- **Default language pair = en→fr**, but the m2m100 core is many-to-many, so the *same* checkpoint also serves the multilingual requirement from one set of weights.
- **Inputs in scope:** raster images (png/jpg/webp) and PDFs (both born-digital and scanned). Plain `.txt` is accepted as a degenerate input that skips OCR.
- **Out of scope:** training a custom OCR or detector; learned inpainting (we use an algorithmic whiteout/smear); real-time/streaming latency targets. End-to-end OCR-VLM alternatives (GOT-OCR-2.0, Pix2Struct) are documented as upgrades but are **not** the shipped core.
- **Headline deliverable:** the layout-preserving overlay with text-fit, plus a deterministic self-checking agent that degrades gracefully instead of producing broken images.

---

## 2. Pipeline architecture

Six stages, mirroring the canonical manga-image-translator / BallonsTranslator / Google-camera flow, collapsed to fit our reuse from P07/P13/P14. Each stage is tagged **PRETRAINED**, **TRAINED**, or **ALGORITHMIC**.

| # | Stage | Status | What we use |
|---|-------|--------|-------------|
| 1 | **Text-region detection** | PRETRAINED (fused with OCR) | Tesseract `image_to_data` returns word/line/block boxes + per-word confidence in one call, so detection and OCR are fused. (Reference tools use separate CTD/CRAFT/PaddleDet detectors; we do not need a standalone detector.) |
| 2 | **OCR / recognition** | PRETRAINED | `pytesseract.image_to_data` (boxes + conf). P07 born-digital-vs-scanned router (PyMuPDF) decides whether OCR is even needed. Stub-OCR offline fallback. |
| 3 | **Mask / erase** | ALGORITHMIC | Pillow whiteout (modal border-ring background color) — replaces the reference inpainting stage. Optional Pillow-only horizontal-smear "simple inpaint". |
| 4 | **Machine translation** | **TRAINED** | `facebook/m2m100_418M` fine-tuned with HF `Seq2SeqTrainer` (chrF/BLEU). The **only** trainable component. Dictionary + identity baselines. |
| 5 | **Typeset / re-render** | ALGORITHMIC | Pillow fit-to-box: binary-search font size + greedy pixel word-wrap, script-aware font selection, contrast-aware text color. |
| 6 | **Compose output** | ALGORITHMIC | Erase-all-boxes-first, then draw-all, producing a new RGB image the same size as input (or a side-by-side panel in the degrade branch). |

Wrapping all six stages is the **agentic state machine** (Section 6): a deterministic router with five decision points that gates on each stage's own intermediate signals and chooses `overlay` / `side-by-side` / `needs_review`.

---

## 3. Datasets

> **Findings note:** the dedicated datasets research returned `null`. **There is no verified, ready-made in-image document-translation benchmark to depend on.** Therefore the design is built around a **synthetic generator** as the primary data source, with a real parallel MT corpus rendered onto images. Any HF dataset id below must be re-verified with `hub_repo_details` before wiring (corpus selection is the MT-researcher's lane; the generator is corpus-agnostic).

### 3a. Corpus role (text side, to be rendered)

| Corpus | Suggested license | Role | Flag |
|--------|-------------------|------|------|
| `Helsinki-NLP/opus-100` (or the OPUS pairs already used in P14 doctrans) | mixed/permissive — **verify per pair** | Parallel (src, tgt) sentences. **Source** side is rendered onto the synthetic image; **target** side is the translation gold. | Verify the exact pair license before commercial use. |
| Tatoeba / WikiMatrix-style pairs (P14 carry-over) | mostly CC-BY | Same role; smaller, cleaner for fixtures. | CC-BY attribution required. |

The generator takes any `list[(src, tgt)]`, so corpus choice is swappable. **No corpus is committed**; it is loaded in the Colab setup cell.

### 3b. Real in-image benchmark

**None verified to exist.** Do **not** claim a public image-translation benchmark. Our evaluation rests on the synthetic generator (3c), whose gold layout/text/translation are known by construction. (If a real benchmark is later located and `hub_repo_details`-verified, add it as a held-out test split; until then, synthetic is the floor.)

### 3c. SYNTHETIC generator (PRIMARY data source) — required

Deterministic, seeded **per index**, producing `(image_png, gold_source_text, gold_target_text, boxes_json)` triples so the *same index always yields the same image* (reproducible tests + exact CER scoring).

**Determinism:** per sample `i`, `rng = random.Random(BASE_SEED * 1_000_003 + i)` (and `np.random.default_rng(BASE_SEED*1_000_003+i)` if numpy is used). Every random choice draws from this rng — no global state. The (src,tgt) pair is chosen by **deterministic index** (not rng) to guarantee full corpus coverage.

**Per-sample pipeline:**
1. Pick the `i`-th (src, tgt) pair by index.
2. **Canvas:** size ∈ {(640,200),(800,256),(1024,300)}; background = solid paper (off-white/cream) OR 2-color vertical gradient OR faint procedural noise. Record bg.
3. **Font:** `rng.choice` over a fixed list (Noto Sans, DejaVuSans, a serif, a mono if present); size ∈ [18,40]; path recorded.
4. **Layout:** lay the **source** sentence into 1–3 blocks using the *same* greedy pixel word-wrap as the render engine; random margins, line spacing ∈ [1.0,1.4]. Record each block's tight bbox via `multiline_textbbox` → `boxes_json`. **These boxes are the gold layout.**
5. Render in a high-contrast ink color (rng from a small ink palette).
6. **Degradations** (mild, OCR-survivable, each toggled by rng with fixed probability, applied in fixed order so they compose deterministically):
   - rotation `uniform(-3,3)°`, `expand=True`, fill=bg;
   - Gaussian blur radius ∈ {0,0,0,0.6,1.0};
   - Gaussian pixel noise σ ∈ {0,4,8}, clip 0–255;
   - brightness/contrast jitter `uniform(0.85,1.15)`;
   - JPEG recompression to in-memory `BytesIO` at quality ∈ {95,80,60} and reload.
   Provide a **CLEAN mode** (all degradations off) for an OCR upper bound.
7. **Emit:** `image.png` + JSONL manifest row `{id, src, tgt, src_lang, tgt_lang, font, size, boxes:[...], degrade_params:{...}, seed}`.

**Outputs enable four measurements:** OCR CER/WER (OCR vs gold_source), MT chrF/BLEU (MT(gold_source) vs gold_target), **end-to-end image-translation chrF** (MT(OCR(image)) vs gold_target), and layout fidelity (rendered boxes vs gold boxes).

**Scale:** default 200–2000 samples for eval, streamable for more; tiny fixtures (3–5 samples, fixed seed, **committed PNGs**) for unit tests.

### 3d. Non-commercial flags

- MT: **`facebook/nllb-200-distilled-600M` is CC-BY-NC-4.0 — non-commercial.** Research/quality upgrade only; never the shipped default.
- OCR: **Surya (`vikp/surya_rec2` + `vikp/surya_det3`) is CC-BY-NC-SA-4.0 — non-commercial + share-alike.** Mention as research-quality alternative only.
- Verify each `opus-100` pair's license before commercial use.

---

## 4. Models

All ids below were **verified to resolve** in the models findings. The shipped stack is **every component MIT or Apache-2.0** (fully permissive). Non-commercial options are flagged and excluded from the default.

### 4a. MT slot (the trainable core)

| id | License | Tier / role |
|----|---------|-------------|
| **`facebook/m2m100_418M`** | **MIT** | **DEFAULT trainable core.** Many-to-many ~100 langs → one fine-tuned checkpoint serves en→fr **and** multilingual. Already the verified P13/P14 stack. **Ship this.** |
| `Helsinki-NLP/opus-mt-en-fr` | Apache-2.0 | **Strong baseline + T4/CPU fallback.** Tiny, fast, near-CPU. Bilingual only (en→fr) so it does **not** cover multi. |
| `facebook/mbart-large-50-many-to-many-mmt` | MIT (per card; no Hub tag) | **H100 upgrade.** 611M, 50 langs many-to-many, higher quality; m2m100 stays the simpler verified default. |
| `facebook/nllb-200-distilled-600M` | **CC-BY-NC-4.0 (NON-COMMERCIAL)** | Quality/coverage upgrade for research only. **Do not ship.** |

### 4b. OCR slot (pretrained front-end, not trained)

| id | License | Tier / role |
|----|---------|-------------|
| **Tesseract / `pytesseract`** | Apache-2.0 (engine + wrapper) | **DEFAULT front-end** (system binary, not HF weights). `image_to_data` → boxes + conf for the overlay step; P07 PyMuPDF router; stub-OCR offline fallback. **Ship this.** |
| `easyocr` (JaidedAI) | Apache-2.0 | **T4-tier light alternative** (boxes+text+conf, GPU-optional, easy install). |
| `docTR` (mindee) / PaddleOCR | Apache-2.0 | Permissive detection+recognition toolkits; cleaner boxes on some layouts. Default-tier alternative. |
| `microsoft/trocr-base-printed` | MIT upstream (**no Hub license tag — flag**) | Neural recognizer upgrade; needs an external detector for line crops. |
| `microsoft/trocr-large-printed` | MIT upstream (no Hub tag — flag) | **H100-tier** higher-accuracy recognizer. |
| `kha-white/manga-ocr-base` | Apache-2.0 | Specialist for vertical/Japanese manga text only. |
| `microsoft/dit-base` | MIT upstream (no Hub tag — flag) | Document-**image** classifier for scanned-vs-born-digital / layout routing — **does not read text**, not an OCR engine. |
| Surya (`vikp/surya_rec2`+`surya_det3`) | **CC-BY-NC-SA-4.0 (NON-COMMERCIAL)** | Research-quality alternative only. **Do not ship.** |

### 4c. End-to-end OCR-VLM alternatives (documented, not the core)

`stepfun-ai/GOT-OCR-2.0-hf` (Apache-2.0, prefer the `-hf` native-transformers id over the `trust_remote_code` original) and `google/pix2struct-base` (Apache-2.0) are unified image→text readers. **Alternatives only** — they are OCR readers, not image→translated-image pipelines; the trainable core stays m2m100.

### 4d. GPU tiers (decisive defaults)

- **T4 fallback (free Colab ~16 GB):** OCR = Tesseract (CPU) or easyocr; MT = `m2m100_418M` fp16 fine-tune (fits T4) or `opus-mt-en-fr` (near-CPU); small batch + grad-accum.
- **DEFAULT (mid GPU A10/L4):** Tesseract or docTR front-end + `m2m100_418M` (MIT) fine-tuned + algorithmic overlay.
- **H100 upgrade (80 GB):** OCR recognizer → `trocr-large-printed` or end-to-end `GOT-OCR-2.0-hf` for tough photos; MT → `mbart-large-50-many-to-many-mmt` for higher quality. All permissive.

---

## 5. Render / overlay engine

**Hard dependency: Pillow ≥ 9.2** (needs `textbbox`/`textlength`/`multiline_textbbox`; verified on Pillow 12.2.0). **No OpenCV, no SciPy.** For each OCR block `(text_src, box=(x0,y0,x1,y1), conf)`: translate → erase → draw fitted+wrapped target. Output is a new RGB image the same size as input.

**Ordering rule:** erase **every** box before drawing **any** translation, so an enlarged translation spilling slightly never paints over a not-yet-erased neighbor.

### Step A — Erase / whiteout (no OpenCV)

- **WHITEOUT (default):** `bg_color` = per-channel **median of a 2px ring just outside the box** (sample an inflated box border). Fallback to corner-pixel median if the box touches the image edge. `draw.rectangle([x0,y0,x1,y1], fill=bg_color)`. Beats hard white on colored/scanned paper.
- **SIMPLE INPAINT (optional):** when border-ring color variance is high, for each row interpolate per-channel between the pixel just left of `x0` and just right of `x1` and write across the row (horizontal smear). A light `GaussianBlur(1)` on the erased patch only hides seams.

### Step B — Font selection (target-language aware)

Scan target codepoints, pick majority Unicode script → font (resolve a `{script: path}` map once at startup):

| Script | Font |
|--------|------|
| Latin / Cyrillic / Greek | `NotoSans-Regular.ttf` |
| CJK (Han/Hiragana/Katakana/Hangul) | `NotoSansCJK-Regular.ttc` (one file = zh/ja/ko) |
| Arabic / Persian / Urdu (RTL) | `NotoSansArabic-Regular.ttf` |
| Devanagari/Hindi | `NotoSansDevanagari-Regular.ttf` |
| Hebrew (RTL) | `NotoSansHebrew-Regular.ttf` |
| Thai | `NotoSansThai-Regular.ttf` |
| **uncovered glyph / offline fallback** | **`DejaVuSans.ttf` shipped inside Pillow** at `os.path.join(os.path.dirname(PIL.__file__),'fonts','DejaVuSans.ttf')` — always present |

**Fonts: Google Noto Sans, SIL OFL 1.1** (permissive, redistributable). Split by script to keep downloads small; download in the Colab setup cell, cache, record exact path per block. Missing Noto files degrade to DejaVuSans without crashing.

### Step C — Fit-to-box (binary search + greedy pixel wrap) — verified

Measure live with a throwaway `ImageDraw` on a 1×1 image. `box_w = x1-x0-2*pad`, `box_h = y1-y0-2*pad`, `pad≈2`.

Primitives: `draw.textlength(s, font)` (line advance width, for wrapping); `draw.multiline_textbbox((0,0), block, font)` → `(l,t,r,b)` (block extent, for fit test).

```
fit_box(text, font_path, box_w, box_h, min_sz=6, max_sz=box_h):
    lo, hi = min_sz, max_sz; best=min_sz; best_wrapped=text; fit_ok=False
    while lo <= hi:
        mid  = (lo+hi)//2
        font = ImageFont.truetype(font_path, mid)
        wrapped  = greedy_wrap(text, font, box_w)
        bw, bh   = extent of multiline_textbbox(wrapped)
        if bw <= box_w and bh <= box_h:
            best, best_wrapped, fit_ok = mid, wrapped, True; lo = mid+1   # try bigger
        else:
            hi = mid-1                                                    # too big
    return best, best_wrapped, fit_ok
```

`greedy_wrap`: split on spaces, accumulate words while `textlength(line+' '+w) <= max_w` else start a new line; join with `\n`. **CJK (no spaces): wrap per-character** — break when adding the next char exceeds `max_w`.

`fit_ok` is **returned** and feeds the layout fit-rate metric and decision point **D5**. If `fit_ok` is False at `min_sz`, render clipped at `min_sz` (or optionally ellipsize) and count the block as an overflow. Complexity: `O(log max_sz)` measurements per block — negligible vs OCR/MT.

> **Verified (Pillow 12.2.0):** "The quick brown fox…" in a 380×100 box → size 33, 3 lines, `fit_ok=True`; an unfittable string in a 30×12 box → clamped to size 6, `fit_ok=False`.

### Step D — Draw

`draw.multiline_text((x0+pad, y0+pad), best_wrapped, font=font_best, fill=text_color, spacing=line_gap, align='left')`.

- **text_color:** high-contrast vs `bg_color` by luminance — `0.299R+0.587G+0.114B < 128 → white else black`. Optionally sample the original ink (darkest pixel cluster inside the box) before erasing.
- **Vertical centering (optional):** `start_y = y0 + (box_h - block_h)//2`.
- **RTL (Arabic/Hebrew):** `align='right'`, anchor at `x1`; for correct shaping use `python-bidi` + `arabic-reshaper` **if available**, else draw the logical string (**flag** that shaping is best-effort without those optional deps). Both are pure-Python optional extras, not hard deps.
- **Use block/paragraph boxes** (Tesseract `level=2/3`) not word boxes — wrapping needs room and word-by-word translation destroys meaning.

---

## 6. Agentic component

A **deterministic state machine** (not an LLM agent) that routes on the pipeline's own intermediate signals and always produces a defensible output (`overlay`, `side-by-side`, or an honest `needs_review`) instead of a silent failure. Five decision points:

### D1 — Input router
- **Gates on:** file magic-bytes / MIME + extension.
- **Branches:** image (png/jpg/webp) → scanned branch (OCR required); pdf → **D2**; `text/.txt` → skip OCR, jump to **D4**; unsupported → `needs_review`.

### D2 — Born-digital vs scanned (PDF only)
- **Gates on:** PyMuPDF embedded-text probe — `page.get_text()` char count / text-coverage ratio per page (reused **verbatim** from P07 dococr router).
- **Branches:** embedded text present (ratio above threshold) → extract the digital text layer + word boxes directly, **BYPASS OCR** (lossless, zero OCR error); little/no embedded text (scanned PDF) → rasterize and route through OCR like an image.

### D3 — Per-block OCR-confidence gate
- **Gates on:** Tesseract `image_to_data` per-word/per-block mean confidence (0–100) + char-length sanity (alpha ratio, `min_text_length`).
- **Branches:** `conf >= HIGH` (≈75) → accept, send to MT; `LOW <= conf < HIGH` (≈40–75) → retry path (gamma correct / upscale / invert, re-score) **or** optional VLM-OCR fallback; `conf < LOW` or empty → drop block + tag `needs_review` (never translate garbage).

### D4 — Translation verification (round-trip + length ratio)
- **Gates on:** two self-emitted signals on each MT output — (a) **round-trip back-translation** chrF/BLEU between source and back-translated source; (b) target/source **length ratio**. Both computed from the model's own intermediate output (**no reference translation needed**).
- **Branches:** round-trip ≥ TAU **and** length-ratio ∈ [0.4, 3.0] → accept; round-trip < TAU (likely hallucination/drift) → re-decode once with alternate params (more beams, sampling off), else flag `low_confidence`; length-ratio out of band (truncation or runaway/repetition) → flag, **and feed the ratio forward to D5** (an over-long translation will not fit the box).

### D5 — Render-fit feasibility gate
- **Gates on:** computed required font size to fit the translated string into the original box at min legible size (uses translated char count, box w/h, and the **D4 length-ratio**): does the text wrap within box height when shrunk only to `font_size_minimum`? Driven by `fit_box`'s returned `fit_ok`.
- **Branches:** fits at ≥ min legible font → **OVERLAY** (erase + re-render in-box, the full camera experience); does **not** fit (translation too long, common JA/ZH→EN expansion) → **SIDE-BY-SIDE** (original untouched + translated caption panel beside it); inpaint/fit fails entirely or block was D3/D4-flagged → **`needs_review`** (emit boxes + raw translation, no destructive render).

### Value-add (why this beats "OCR → translate → print the string")

1. **Layout-preserving overlay with text-fit** — translations are drawn back where the source text was (detect → erase → re-render with shrink-to-fit, wrap, alignment, direction). This is exactly what separates camera-translate from a plain OCR+MT script and is the single most visible deliverable.
2. **Confidence + verification gates on the model's own intermediate outputs** — OCR confidence (D3) so garbled text is never translated; round-trip + length-ratio (D4) so MT hallucination/truncation is caught **without any reference translation**; render-fit (D5) so the system **degrades gracefully** (overlay → side-by-side → needs_review) instead of spilling text out of boxes. This self-checking degradation ladder is the agentic part. Bonus: D2's born-digital bypass eliminates OCR error entirely on the common digital-PDF case.

---

## 7. Metrics, baselines, and offline floor

### MT (the trainable core)
- **chrF (primary):** `sacrebleu.CHRF` (chrF++ `word_order=2` or plain chrF) between `MT(gold_source_text)` and `gold_target_text`, on **clean gold source** so it isolates MT from OCR noise. Range 0–100, higher better. Matches P13/P14.
- **BLEU (secondary):** `sacrebleu.corpus_bleu` with language-appropriate tokenizer (`13a` Latin, `zh`, `ja-mecab`, `intl`).

### OCR (front-end)
- **CER:** char-level `(S+D+I)/N_chars` via pure-Python two-row Levenshtein (zero deps; verified kitten/sitting → 0.5). Optional whitespace/case normalization.
- **WER:** same over whitespace tokens (verified "the cat sat" vs "the cat sit" → 0.333).

### End-to-end (headline)
- **End-to-end image-translation chrF:** chrF between `MT(OCR(synthetic_image))` and `gold_target_text` — the full pipeline on the rendered/degraded image. The gap `(clean MT-chrF − end-to-end-chrF)` quantifies OCR cost. Also report end-to-end on **CLEAN** images as an upper bound.

### Layout fidelity
- **fit-rate (dominant overlay signal):** fraction of blocks with `fit_ok==True` and `final_font_size >= min_readable`. 1.0 = every translation fit. Exposes box-overflow (translations are usually longer than source).
- **box-retention (IoU):** mean IoU between each rendered translation's actual drawn bbox (`multiline_textbbox` of the wrapped target at chosen size, offset to box origin) and the original source box.
- **no-overlap rate:** `1 − (#new overlapping rendered pairs / #originally-disjoint pairs)`. Penalizes long translations bleeding into neighbors.
- **aggregate layout score (optional):** `mean(fit_rate, mean_box_IoU, no_overlap_rate)`, fit-rate weighted highest; report components separately too.

### Baselines (required, for honest comparison)
1. **Dictionary MT** — glossary lookup + identity passthrough for OOV (also the offline MT). Low chrF, proves the fine-tune adds value.
2. **Identity** — pass the source text through unchanged (chrF floor when src≈tgt scripts; near-zero across languages).
3. **Zero-shot MT** — `m2m100_418M` / `opus-mt-en-fr` *without* fine-tuning, to quantify what the fine-tune buys.

### Offline floor expectation
With the offline stack (stub OCR + dictionary MT + DejaVu + pure-Python chrF), **chrF/BLEU are deliberately low** — tests assert *pipeline wiring and metric plumbing*, not translation quality. The CER/WER, fit-rate, IoU, and no-overlap metrics are fully meaningful offline (geometry + deterministic stub corruption). Expect: dictionary MT chrF well below the fine-tuned m2m100; fit-rate high on synthetic (boxes generated by the same wrap algorithm); end-to-end chrF below clean MT chrF by the OCR-error gap.

---

## 8. Offline fallback design

Tests run with **no tesseract binary, no torch, no Noto TTFs** — every component degrades to a deterministic pure-Python stub so CI passes with only Pillow (Pillow itself optional for metric-only tests). Selection is automatic via capability probes (`try import` / `shutil.which('tesseract')`); env flag `P15_OFFLINE=1` pins stub mode for reproducible tests. **Same code path — probes upgrade each stage when the real component is present; no test changes.**

1. **Stub OCR (no tesseract):**
   - (a) For a synthetic image with an available manifest: `OCR(image_i) := manifest[i].src` (no pixel reading) — lets MT + render + metrics run end-to-end offline; CER vs gold = 0 (clean).
   - (b) Optional deterministic corruption: seeded per index `random.Random(seed+i)`, substitute/drop a fixed fraction (~3%) of chars → a **known, reproducible CER** to exercise the error-rate code (mirrors P07's stub-OCR pattern).
   - (c) If `pytesseract` + binary exist → real P07 path (`image_to_data` boxes+conf, born-digital router). Probe picks real-vs-stub at runtime.
2. **Dictionary MT (no torch / no m2m100):** small in-repo bilingual glossary `{src_token: tgt_token}` + identity passthrough for OOV: `translate(s) = ' '.join(dict.get(tok.lower(), tok) for tok in s.split())`. Deterministic, instant, no download. This is the P13/P14 dictionary/identity baseline reused as **both** baseline **and** offline MT. Swap in fine-tuned m2m100 when transformers+torch are present.
3. **Font fallback (no Noto):** `PIL/fonts/DejaVuSans.ttf` (always inside the Pillow wheel), then `ImageFont.load_default()` (bitmap) last resort. Latin renders correctly; **CJK/Arabic render tofu offline — FLAGGED, affects pure-offline mode only**. With `load_default` the fit binary search is skipped (fixed size) and `fit_ok` is computed at the single available size.
4. **Metrics (no sacrebleu/jiwer):** CER/WER use the pure-Python Levenshtein. chrF/BLEU → minimal pure-Python chrF (char n-gram precision/recall F, n=1..6, β=2) when sacrebleu missing; real sacrebleu when installed (a tolerance test asserts they agree). Layout metrics are pure geometry — already dependency-free.

**Net:** with stdlib + Pillow only, `image → OCR(stub) → MT(dict) → render(DejaVu) → all metrics` executes deterministically; the smoke notebook cell and unit tests run in Colab/CI with nothing downloaded.

---

## 9. Reuse map

### Ported from **P07 dococr**
- Tesseract `image_to_data` OCR front-end (boxes + per-word confidence).
- **Born-digital-vs-scanned router** (PyMuPDF `get_text()` coverage probe) → drives **D2** verbatim.
- **Stub-OCR offline fallback** pattern (return seed text + light deterministic noise) → drives offline §8.1.

### Ported from **P13 s2st / P14 doctrans**
- MT core `facebook/m2m100_418M` + HF `Seq2SeqTrainer` fine-tune harness.
- **Dictionary + identity baselines** → reused as baselines **and** offline MT (§8.2).
- sacrebleu chrF/BLEU metric plumbing; the parallel-corpus loading pattern (OPUS/Tatoeba).

### NEW for P15
- **Render / overlay engine** (Section 5): Pillow whiteout/smear erase, script-aware font routing, **fit-to-box binary search + greedy pixel/CJK wrap**, contrast-aware draw, erase-all-then-draw ordering.
- **Synthetic doc-image-translation generator** (Section 3c): deterministic seeded `(image, gold_src, gold_tgt, boxes)` triples with composable degradations + JSONL manifest.
- **Layout-fidelity metrics**: fit-rate, box-retention IoU, no-overlap rate, aggregate layout score.
- **Five-point agentic state machine** (Section 6): D1 input router, D4 round-trip + length-ratio MT verification, D5 render-fit feasibility + overlay/side-by-side/needs_review degradation ladder (D2/D3 are reused-but-rewired P07 signals).
- **End-to-end image-translation chrF** wiring (`MT(OCR(image))` vs gold_target).

---

## 10. Risks and gotchas

- **No real benchmark (datasets findings = `null`).** Everything rests on the synthetic generator; do not overclaim generalization to photos of real signage. Mitigate with the degradation suite and, if found, a `hub_repo_details`-verified held-out set.
- **License tags missing on TrOCR/DiT.** `trocr-*-printed` and `dit-base` carry **no Hub license tag** (MIT upstream). Treat as MIT but document the gap; do not assert a tag that is not surfaced.
- **Non-commercial traps.** `nllb-200-distilled-600M` (CC-BY-NC-4.0) and Surya (CC-BY-NC-SA-4.0) are tempting for quality — **never ship them**. Also verify each `opus-100` pair license.
- **Translation expansion breaks overlay.** Targets are routinely longer than source (esp. →fr, and CJK→EN); the box overflows. D5's side-by-side branch and the fit-rate metric exist precisely for this — it is the expected common failure, not a bug.
- **OCR returns no reliable text color.** We estimate ink/background from pixels (luminance contrast / darkest-cluster sample); on textured or multi-color backgrounds the whiteout patch can be visible — fall back to simple-inpaint smear + blur, and accept imperfect erase.
- **CJK / RTL rendering.** No-space wrapping must be per-character; RTL shaping is **best-effort** without optional `python-bidi`/`arabic-reshaper` (flag it). Offline-no-Noto renders **tofu** for CJK/Arabic — affects pure-offline test mode only; real runs ship Noto.
- **Word-box vs block-box.** Translating word-by-word destroys meaning and wrecks wrapping — always aggregate to Tesseract block/paragraph boxes (`level=2/3`) before MT.
- **Round-trip verification is heuristic.** D4's back-translation chrF can false-flag legitimate free translations and miss fluent hallucinations; treat it as a soft gate (re-decode once, then `low_confidence`), never a hard reject that silently drops content.
- **Determinism discipline.** The generator and stub corruption must use only the per-index seeded rng (`BASE_SEED*1_000_003 + i`) — any leak to global `random`/`numpy` state breaks reproducible CER scoring and fixture tests.
- **Pillow version floor.** `multiline_textbbox`/`textlength` require Pillow ≥ 9.2; pin it. The fit algorithm is verified on 12.2.0.
- **T4 memory.** Fine-tuning m2m100_418M on a free T4 needs fp16 + small batch + grad-accum; if it still OOMs, drop to `opus-mt-en-fr` for the en→fr-only demo and note the loss of multilingual coverage.
