# P15 Document-Image Machine Translation — Model Selection

Author: Le Dinh Minh Quan (student 23127460)
Package: `imgtrans` · Folder: `15_Document_Image_Translation`
Default direction: **en→fr** (configurable; the MT core is many-to-many)

---

## 1. Scope of this document

P15 translates the text that appears **inside** an image, scanned page, or born-digital PDF, then renders the translation **back onto the page** while preserving the original spatial layout — the "Google Translate camera" experience. It is a **cascade**, not an end-to-end model:

```
image / PDF ──▶ [OCR front-end] ──▶ source text + boxes + conf
                                          │
                                          ▼
                                   [MT core]  ← the ONLY trained stage
                                          │
                                          ▼
                          translated text ──▶ [layout-preserving render] ──▶ output image
```

Three model/algorithm slots must be chosen, and they are not equal in nature:

| Slot | Nature | Trained here? | This doc decides |
|------|--------|---------------|------------------|
| **MT core** | neural seq2seq | **YES** (HF `Seq2SeqTrainer`, metric chrF) | which checkpoint to fine-tune and ship |
| **OCR front-end** | pretrained / algorithmic | no | which recognizer is the default + fallbacks |
| **Render / overlay** | pure algorithm | no | which library + rendering strategy |

Plus a transverse decision: **cascade vs end-to-end OCR-VLM**, and a **GPU-tier table** that pins concrete model/precision/batch defaults per accelerator.

The governing constraint across every slot: **the shipped default stack must be fully permissive (MIT or Apache-2.0).** Non-commercial-licensed options (NLLB, Surya) are documented as research-only upgrades and are explicitly **flagged and excluded** from the default.

---

## 2. MT core — the only trainable component

The MT stage is the single component we fine-tune. Every other stage is pretrained or deterministic code, so the entire engineering and licensing weight of "the trained model" rests here. Four candidates were verified to resolve on the HF Hub.

### 2.1 Candidate comparison

| id | License | Params | Coverage | Verdict |
|----|---------|--------|----------|---------|
| **`facebook/m2m100_418M`** | **MIT** | 418M | many-to-many, ~100 langs | **DEFAULT — ship this** |
| `Helsinki-NLP/opus-mt-en-fr` | Apache-2.0 | ~74M | bilingual, en→fr only | strong baseline + T4/CPU fallback |
| `facebook/mbart-large-50-many-to-many-mmt` | MIT (card; no Hub tag) | 611M | many-to-many, 50 langs | H100 quality upgrade |
| `facebook/nllb-200-distilled-600M` | **CC-BY-NC-4.0 (NON-COMMERCIAL)** | 600M | 200 langs | **FLAGGED — research only, do NOT ship** |

### 2.2 Why `m2m100_418M` is the default

**License is decisive and clean.** m2m100 is **MIT** — fully permissive, commercially usable, redistributable. This single fact removes the largest legal risk in the stack and is the reason NLLB (better raw quality, broader coverage) is rejected for the shipped default.

**One checkpoint satisfies two requirements.** The default pair is en→fr, but the assignment also wants multilingual capability. m2m100 is **directly many-to-many across ~100 languages**, so a *single* fine-tuned checkpoint serves en→fr **and** the multilingual requirement from one set of weights — no per-pair model zoo, no separate bilingual models to maintain. The translation direction is a runtime argument (forced BOS / `tgt_lang`), not a different model.

**It fits the hardware floor.** At 418M parameters, fine-tuning fits a free-Colab **T4 (~16 GB)** under fp16 with small batch + gradient accumulation. mBART-50 (611M) and NLLB (600M) are tighter on a T4; m2m100 is the largest many-to-many model that comfortably trains on the lowest-common-denominator GPU.

**Zero new infrastructure.** m2m100 is already the **verified P13 (s2st) / P14 (doctrans) stack.** The HF `Seq2SeqTrainer` harness, the chrF/BLEU (`sacrebleu`) metric plumbing, and the OPUS/Tatoeba corpus-loading pattern are ported **verbatim**. P15 reuses a battle-tested fine-tune path and spends its novelty budget on the render engine, synthetic generator, and agent instead.

### 2.3 Why the alternatives are not the default

**`Helsinki-NLP/opus-mt-en-fr` (Apache-2.0) — baseline and fallback, not the core.** Tiny (~74M), fast, near-CPU-viable, permissive. It is an excellent **zero-shot/fine-tune baseline** and the **T4/CPU emergency fallback** when m2m100 OOMs. But it is **bilingual (en→fr only)**: shipping it as the core would silently drop the multilingual requirement. It is positioned as a fallback that explicitly trades away multilingual coverage, never as the primary.

**`facebook/mbart-large-50-many-to-many-mmt` (MIT) — H100 upgrade.** Also many-to-many (50 langs), MIT, generally higher quality than m2m100 at 611M params. It is the documented **quality upgrade for an H100 tier**. It is not the default because (a) it is heavier on a T4, (b) it covers fewer languages than m2m100's ~100, and (c) its license is asserted by the model card but **carries no Hub license tag** — we treat it as MIT but document that gap rather than depend on an unverified tag for the shipped default.

**`facebook/nllb-200-distilled-600M` — FLAGGED, excluded.** NLLB is tempting: 200 languages, strong quality. But it is **CC-BY-NC-4.0 — non-commercial.** A production system that may be deployed commercially **cannot ship a non-commercial model.** NLLB is documented as a research/quality reference only and is **never wired into the default pipeline.** This is the most important licensing trap in the MT slot and is called out explicitly in the brief's non-commercial flags.

### 2.4 Training and baselines

- **Fine-tune:** HF `Seq2SeqTrainer`, headline metric **chrF** (`sacrebleu.CHRF`), secondary **BLEU** (`sacrebleu.corpus_bleu`, language-appropriate tokenizer). chrF is computed on **clean gold source** so it isolates MT quality from OCR noise.
- **Corpus:** `Helsinki-NLP/opus-100` en-fr (~1M pairs) for fine-tuning; **license is unverified per-pair → flag and verify before commercial use.** Optional real OCR-noise text: `PleIAs/Post-OCR-Correction` english (CC0). No corpus is committed to the repo; it is loaded in the Colab setup cell.
- **Required baselines (honest comparison):**
  1. **Identity** — copy source through unchanged (the chrF floor; near-zero across scripts).
  2. **Dictionary MT** — glossary lookup + identity passthrough for OOV (also the offline MT).
  3. **Zero-shot m2m100 / opus-mt** — un-fine-tuned, to quantify exactly what the fine-tune buys.
- **Offline-seed note (do not over-read):** on the built-in synthetic seed pages, dictionary-MT chrF (79.9) saturates vs the identity floor (22.4) **because the seed (src,tgt) pairs overlap the dictionary by construction.** On real held-out opus-100 eval pairs the fine-tuned m2m100 dominates — that is the honest, non-saturated floor. The seed numbers test *pipeline wiring*, not translation quality.

---

## 3. OCR front-end — pretrained, never trained

The OCR stage is a **pretrained system component**, not something P15 trains. Its job is to return, in one call, the **text plus per-block boxes plus per-word confidence** that the overlay step needs to place translations back on the page. That last requirement — geometry + confidence in a single, dependency-light call — drives the choice as much as raw accuracy does.

### 3.1 Candidate comparison

| id / tool | License | Role |
|-----------|---------|------|
| **Tesseract (`pytesseract`)** | **Apache-2.0** | **DEFAULT front-end — ship this** |
| `easyocr` (JaidedAI) | Apache-2.0 | T4-tier light alternative |
| `docTR` (mindee) / PaddleOCR | Apache-2.0 | cleaner boxes on some layouts |
| `microsoft/trocr-base-printed` | MIT upstream (no Hub tag — flag) | neural recognizer upgrade (needs external detector) |
| `microsoft/trocr-large-printed` | MIT upstream (no Hub tag — flag) | H100-tier high-accuracy recognizer |
| `kha-white/manga-ocr-base` | Apache-2.0 | specialist: vertical/Japanese manga only |
| Surya (`vikp/surya_rec2` + `surya_det3`) | **CC-BY-NC-SA-4.0 (NON-COMMERCIAL + SHARE-ALIKE)** | **FLAGGED — research only, do NOT ship** |
| **SeedEngine** (in-repo) | project code | **offline fallback** — reads gold spec embedded in synthetic images |

### 3.2 Why Tesseract is the default

**It fuses detection + recognition + confidence in one call.** `pytesseract.image_to_data` returns word/line/block boxes **and** per-word confidence in a single call. The reference camera-translate stack (manga-image-translator, BallonsTranslator) needs a *separate* text detector (CRAFT / CTD / PaddleDet) feeding a recognizer; Tesseract collapses stages 1–2 of the pipeline into one, so we **never build or train a standalone detector.** The block-level boxes (`level=2/3`) feed the overlay directly, and the per-word confidence feeds agent decision point **D3** (skip translating low-confidence garbage).

**Permissive and dependency-light.** Apache-2.0 engine + Apache-2.0 Python wrapper. It is a **system binary** (no multi-GB neural weights to download), installs cleanly in Docker (`tesseract-ocr`), and runs on CPU — so the default stack needs no GPU for OCR at all.

**Direct reuse from P07 dococr.** The Tesseract `image_to_data` front-end, the **PyMuPDF born-digital-vs-scanned router** (which drives agent decision **D2**, bypassing OCR entirely when a real text layer exists), and the **stub-OCR offline-fallback pattern** are all ported from P07. Proven plumbing.

### 3.3 Why the alternatives are not the default

**`easyocr` / `docTR` / PaddleOCR (all Apache-2.0) — permissive peers, kept as alternatives.** All return boxes + text + confidence and are fully permissive. easyocr is the **T4-tier light alternative** (GPU-optional, very easy install); docTR / PaddleOCR give **cleaner boxes on some layouts** (multi-column, dense). They are first-class swap-ins behind the same interface, but Tesseract wins the default on the smallest install footprint, CPU-only operation, and verbatim P07 reuse.

**`microsoft/trocr-base/large-printed` (MIT upstream) — neural upgrade, with a caveat.** TrOCR is a stronger *recognizer* for tough printed text and is the **H100-tier** accuracy upgrade (`-large-printed`). Two reasons it is not the default: (1) TrOCR is a **recognizer only** — it reads a pre-cropped line and **needs an external detector** to find lines first, re-introducing the separate-detector stage Tesseract avoids; (2) it **carries no Hub license tag** (MIT is the upstream claim) — we treat it as MIT but **document the gap** rather than assert an unsurfaced tag.

**`kha-white/manga-ocr-base` (Apache-2.0) — out-of-scope specialist.** Vertical / Japanese manga text only. Useful for a manga-translate demo, irrelevant to the en→fr document default.

**Surya — FLAGGED, excluded.** Surya (`vikp/surya_rec2` + `vikp/surya_det3`) has excellent layout/recognition quality, but it is **CC-BY-NC-SA-4.0 — non-commercial AND share-alike.** Both clauses are disqualifying for a potentially commercial deliverable (the share-alike clause would additionally try to relicense downstream code). It is mentioned as a research-quality alternative and **never shipped** — the OCR-slot twin of the NLLB trap.

### 3.4 SeedEngine — the offline OCR fallback

For tests and offline runs (`P15_OFFLINE=1`, or simply no tesseract binary present), **SeedEngine** is the deterministic stub OCR. For a synthetic image with an available manifest it returns `manifest[i].src` **without reading pixels**, so the MT → render → metrics chain runs end-to-end offline with **CER = 0 (perfect OCR)**. It can optionally inject a seeded, reproducible character-corruption fraction to exercise the CER/WER code with a *known* error rate. A runtime capability probe (`shutil.which('tesseract')` / `try import`) picks the real Tesseract path vs SeedEngine — **same code path, no test changes.** This is why the verified offline-seed eval shows OCR CER 0.0; on Colab with real Tesseract on degraded scans the CER is realistic and non-zero, and the gap `(clean MT-chrF − end-to-end-chrF)` quantifies the OCR cost.

---

## 4. Render / overlay engine — pure algorithm (Pillow)

The render stage is **not a model** — it is deterministic code, and deliberately so. It is the single most visible deliverable (translations drawn back into the original boxes) and must run offline with zero heavyweight dependencies.

### 4.1 Why PIL-only (no OpenCV, no SciPy, no neural inpainting)

- **Hard dependency: Pillow ≥ 9.2** (needs `textbbox` / `textlength` / `multiline_textbbox`; the fit algorithm is verified on Pillow 12.2.0). **No OpenCV, no SciPy** — a tiny, portable, offline-friendly footprint that installs everywhere and adds no GPU requirement.
- **Erase = algorithmic whiteout, not learned inpainting.** The reference stacks use a neural inpainting model to remove source text. P15 deliberately uses a **Pillow whiteout** (`bg_color` = per-channel median of a 2px ring just outside the box; corner-pixel-median fallback at edges) and an optional **horizontal-smear "simple inpaint"** for high-variance backgrounds. This trades some visual polish for **zero model weights, full determinism, and offline operation** — the right trade for a tool whose value is layout fidelity, not photo retouching.
- **Fit-to-box = binary search + greedy wrap.** Font size is found by binary search (`O(log max_size)` measurements per block, negligible vs OCR/MT); words are wrapped greedily by measured pixel width (**per-character** for CJK, which has no spaces). The search returns a `fit_ok` flag that feeds the layout fit-rate metric **and** agent decision **D5** (overlay vs side-by-side vs needs_review).
- **Script-aware font selection.** Majority Unicode script of the *target* string picks the font: Noto Sans (Latin/Cyrillic/Greek), Noto Sans CJK, Noto Sans Arabic/Hebrew (RTL), Devanagari, Thai. **Fonts: Google Noto, SIL OFL 1.1** (permissive, redistributable). The universal offline fallback is **`DejaVuSans.ttf` shipped inside the Pillow wheel** — always present, so rendering never crashes for lack of a font. (Offline-with-no-Noto renders **tofu** for CJK/Arabic — flagged, affects pure-offline test mode only; real runs ship Noto.)
- **Erase-all-then-draw ordering** so an enlarged translation spilling slightly never paints over a not-yet-erased neighbor. **Contrast-aware text color** by luminance (`0.299R+0.587G+0.114B < 128 → white else black`). **RTL shaping is best-effort** without optional `python-bidi` / `arabic-reshaper` (flagged).

There is **no model-selection decision** in this slot — only the engineering decision to keep it algorithmic, offline, and dependency-light rather than introducing a neural inpainter or a heavy CV stack.

---

## 5. Cascade vs end-to-end OCR-VLM

A genuine architectural alternative exists: skip the cascade and use a single image→text vision-language model. Two verified, Apache-2.0 candidates were considered and **documented as upgrades only.**

| id | License | What it is |
|----|---------|------------|
| `stepfun-ai/GOT-OCR-2.0-hf` | Apache-2.0 | unified image→text reader (prefer `-hf` native-transformers id over the `trust_remote_code` original) |
| `google/pix2struct-base` | Apache-2.0 | unified image→text reader |

### Why the cascade is the chosen core

1. **They read, they do not re-render.** GOT-OCR-2.0 and Pix2Struct are image→**text** readers — they are OCR-VLMs, **not** image→translated-image pipelines. The headline P15 deliverable is the **layout-preserving overlay** (translation drawn back *where the source text was*). A VLM that emits a text string discards the geometry the overlay needs; we would still have to bolt the render engine on top.
2. **The trainable core stays MT.** P15's design contract is "the only trained stage is MT." Folding OCR into a learned VLM would either (a) make OCR a second training target (out of scope, no in-image translation benchmark exists to train it on) or (b) freeze the VLM and lose the explicit per-block confidence the agent needs.
3. **The cascade exposes the intermediate signals the agent gates on.** The five-decision agent (D1–D5) routes on *intermediate* signals — Tesseract per-block confidence (D3), born-digital text-coverage ratio (D2), round-trip back-translation chrF and length-ratio (D4), render fit-rate (D5). An end-to-end VLM is a black box that **hides exactly these signals**, defeating the self-checking degradation ladder that is the agentic value-add.
4. **Cost, latency, and offline operation.** The cascade runs fully offline (SeedEngine + dictionary MT + Pillow) on CPU. A VLM adds GPU cost and latency for a stage Tesseract already handles permissively.

GOT-OCR-2.0-hf is therefore positioned as an **H100-tier OCR upgrade for tough photographs** (drop-in for the Tesseract recognizer behind the same boxes+text interface), not as the architecture. Pix2Struct is documented as a peer alternative. **The cascade is the shipped core.**

---

## 6. GPU-tier table (decisive defaults)

Concrete model / precision / batch defaults per accelerator tier. Every entry in the default and upgrade rows is **MIT or Apache-2.0** (fully permissive); the non-commercial NLLB/Surya never appear.

| GPU tier | VRAM | OCR front-end | MT model + precision | Batch / training | Render | Notes |
|----------|------|---------------|----------------------|------------------|--------|-------|
| **CPU / offline** | — | SeedEngine stub (or Tesseract CPU) | Dictionary MT (no torch) | n/a (no training) | Pillow + DejaVuSans | `P15_OFFLINE=1`; deterministic; CI/tests; tofu for CJK/Arabic |
| **T4 (free Colab)** | ~16 GB | Tesseract (CPU) or `easyocr` | **`m2m100_418M` fp16** (fine-tune) — fallback `opus-mt-en-fr` (near-CPU, en→fr only) | small batch + **grad-accum**; fp16 | Pillow + Noto | the hardware floor; if m2m100 OOMs → opus-mt and note loss of multilingual coverage |
| **DEFAULT (A10 / L4)** | 24 GB | **Tesseract** (or docTR) | **`m2m100_418M` fp16** (fine-tuned) | comfortable batch; fp16 | Pillow + Noto | **the shipped default stack** — every component MIT/Apache |
| **A100** | 40–80 GB | Tesseract / docTR | `m2m100_418M` fp16/bf16, larger batch | large batch; bf16 | Pillow + Noto | faster training, same checkpoint |
| **H100 (upgrade)** | 80 GB | `trocr-large-printed` or `GOT-OCR-2.0-hf` (tough photos) | **`mbart-large-50-many-to-many-mmt`** (MIT) for higher quality | large batch; bf16 | Pillow + Noto | quality-max; all permissive; mBART/TrOCR carry no Hub tag (treated MIT, gap documented) |

---

## 7. Non-commercial flags — summary

Two strong models are deliberately **excluded from every shipped tier** for licensing reasons, plus one corpus to verify:

- **`facebook/nllb-200-distilled-600M` — CC-BY-NC-4.0 (NON-COMMERCIAL).** Best coverage (200 langs) in the MT slot; **never shipped.** Research/quality reference only.
- **Surya (`vikp/surya_rec2` + `vikp/surya_det3`) — CC-BY-NC-SA-4.0 (NON-COMMERCIAL + SHARE-ALIKE).** Best layout quality in the OCR slot; **never shipped.** Research alternative only.
- **`Helsinki-NLP/opus-100` en-fr corpus — per-pair license unverified.** Verify the exact pair license before commercial use; do not claim a license that is not surfaced.
- **License-tag gaps (treat as upstream license, document the gap, do not assert an unsurfaced tag):** `mbart-large-50-many-to-many-mmt` (MIT per card, no Hub tag); `microsoft/trocr-base/large-printed` and `microsoft/dit-base` (MIT upstream, no Hub tag).

The **default stack is fully permissive end-to-end**: Tesseract (Apache-2.0) → m2m100_418M (MIT) → Pillow + Noto (SIL OFL 1.1 / Pillow's own DejaVu). No non-commercial component is reachable from any default code path.

---

## 8. Decision summary

| Slot | Chosen default | License | One-line reason |
|------|----------------|---------|-----------------|
| **MT core (trained)** | `facebook/m2m100_418M` | MIT | permissive + many-to-many (one checkpoint = en→fr **and** multilingual) + fits a T4 + verified P13/P14 reuse |
| **OCR front-end** | Tesseract / `pytesseract` | Apache-2.0 | boxes + conf in one call (no separate detector) + CPU + P07 reuse |
| **Offline OCR** | SeedEngine (in-repo) | project code | deterministic, pixel-free, CER-0 for offline tests |
| **Render / overlay** | Pillow fit-to-box | Pillow + SIL OFL fonts | algorithmic, offline, dependency-light; no neural inpainter |
| **Architecture** | cascade (OCR → MT → render) | — | exposes the intermediate signals the agent gates on; preserves layout; MT-only training |
| **Excluded (NC)** | NLLB-200, Surya | CC-BY-NC | strong quality, but non-commercial — flagged, never shipped |
