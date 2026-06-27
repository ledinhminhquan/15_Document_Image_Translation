# P15 Document-Image Machine Translation — Data Card

> Package `imgtrans` · folder `15_Document_Image_Translation`
> Author: Le Dinh Minh Quan (student 23127460)
> Task: translate the text that appears *inside* an image / scanned document / born-digital PDF and render the translation back onto the page preserving layout. Cascade **OCR → MT → overlay render**; the **only trained stage is MT**.
> Default direction: **en → fr** (configurable — the `m2m100_418M` core is many-to-many, so one checkpoint serves multiple pairs).

This card documents every data source the project touches: the parallel **MT fine-tune corpus** (the text that gets rendered and translated), an optional **post-OCR noise corpus**, the **synthetic document-image generator** that is the *primary* data source for the end-to-end task, and the built-in **offline seed pages + dictionary**. For each it states id, role, size, schema, license, provenance, known biases/limitations, and intended vs out-of-scope use.

---

## 0. Why a synthetic generator is the primary data source

Dedicated datasets research for an in-image / document-image translation benchmark with **gold parallel text** (an image containing source text, paired with the gold translation of that text *and* the gold layout boxes) returned **`null`**. No verified public benchmark of this exact shape exists. There is plenty of OCR data, plenty of parallel MT text, and plenty of document-layout data — but not a ready-made `(image, gold_source, gold_target, boxes)` corpus.

Consequently P15 is built around a **reproducible synthetic generator** (`data/synth_render.py`) that renders source sentences drawn from a real parallel MT corpus onto page images and **embeds the gold layout spec** (source text, gold translation, block boxes, render parameters, seed). This yields `(image, gold_source, gold_target, boxes)` quadruples *by construction*, which is exactly what lets us score:

- **OCR** CER/WER — OCR output vs `gold_source`;
- **MT** chrF/BLEU — `MT(gold_source)` vs `gold_target`;
- **End-to-end image translation** chrF/BLEU — `MT(OCR(image))` vs `gold_target`;
- **Layout fidelity** — rendered boxes vs gold boxes (fit-rate, IoU, no-overlap).

The trade-off — synthetic clean renders are *not* photographs of real signage — is documented explicitly in §3.7 (Limitations vs real photos). If a real `hub_repo_details`-verified benchmark is later located, it is added as a held-out test split; until then, **synthetic is the floor and the only end-to-end ground truth**.

---

## 1. Dataset inventory (at a glance)

| # | Dataset / source | Role | Trained on? | License | Ship status |
|---|------------------|------|-------------|---------|-------------|
| 1 | `Helsinki-NLP/opus-100` (en-fr config) | MT fine-tune + eval corpus (text rendered onto images; target = gold) | **Yes — MT only** | Unknown / mixed per pair — **FLAGGED**, verify before commercial use | Default trainable corpus |
| 2 | `PleIAs/Post-OCR-Correction` (english) | Optional real OCR-noise text source for robustness eval | No | **CC0-1.0** (public domain) | Optional, permissive |
| 3 | **`data/synth_render.py`** synthetic generator | **PRIMARY** end-to-end data: `(image, gold_source, gold_target, boxes)` | No (consumes corpus #1 text) | Code MIT (this repo); renders text under its source corpus license | Default primary data |
| 4 | Built-in **SeedEngine seed pages** + **en→fr dictionary** | Offline backbone (deterministic OCR + dictionary MT for tests/offline mode) | No | MIT (this repo) | Default offline fallback |
| 5 | Google **Noto Sans** / Pillow **DejaVuSans** fonts | Rendering glyphs (generator + overlay) | No | Noto: **SIL OFL 1.1**; DejaVu: Bitstream Vera / public-domain-ish permissive | Shipped (Noto downloaded in setup; DejaVu inside Pillow wheel) |

**Non-commercial datasets/models flagged and NOT shipped** (kept out of every default path): `facebook/nllb-200-distilled-600M` (**CC-BY-NC-4.0**) and Surya `vikp/surya_rec2`+`vikp/surya_det3` (**CC-BY-NC-SA-4.0**). These are model weights, not datasets, but are listed here so the licensing picture is complete — they are research-quality upgrades only and must never be wired into a shipped pipeline.

---

## 2. Text corpora (the "what gets translated" side)

The generator and the MT trainer are **corpus-agnostic**: both consume any `list[(src, tgt)]` of parallel sentences. No corpus is committed to the repo; it is downloaded in the Colab/training setup cell. The defaults below are the verified choices.

### 2.1 `Helsinki-NLP/opus-100` — primary MT corpus (DEFAULT)

| Field | Value |
|-------|-------|
| **HF id** | `Helsinki-NLP/opus-100` (config `en-fr`) |
| **Role** | The **only** dataset the trainable component (MT) is fine-tuned on. The **source** side is rendered onto synthetic page images; the **target** side is the gold translation against which MT and end-to-end chrF/BLEU are scored. |
| **Direction** | en → fr by default; the corpus is bilingual aligned, the `m2m100_418M` core is many-to-many so other OPUS-100 configs can be substituted by changing the config + `src_lang`/`tgt_lang`. |
| **Size** | ~1M sentence pairs for the en-fr config (OPUS-100 caps high-resource pairs at 1M train; with ~2k dev + ~2k test). P15 typically fine-tunes on a sampled slice (tens of thousands of pairs) under Colab time/compute limits. |
| **Schema** | `{ "translation": { "en": <str>, "fr": <str> } }` per example (HF `translation` feature). Loader flattens to `(src, tgt)` tuples. |
| **License** | **Unknown / mixed — FLAGGED.** OPUS-100 is sampled from the OPUS collection, whose constituent corpora carry heterogeneous licenses. The aggregate is generally treated as research-permissive, but **the exact per-pair license is not surfaced as a single clean tag**. Verify each pair's provenance before any commercial deployment. |
| **Provenance** | Derived from the OPUS project (Tiedemann); `opus-100` is an English-centric 100-language multilingual MT benchmark sampled from OPUS, distributed via the Helsinki-NLP org on the HF Hub. Originally released with the paper on massively multilingual NMT. |

**Known biases / limitations:**
- **Domain skew.** OPUS aggregates web-crawled and institutional text (subtitles, software localization, religious texts, EU/UN proceedings). Register is uneven — conversational subtitle lines sit next to formal legal boilerplate — and this bias propagates into the fine-tuned MT and therefore into rendered translations.
- **Alignment noise.** Automatically aligned pairs include partial mismatches, length-mismatched sentences, and the occasional misaligned pair. Not human-curated to gold standard.
- **License opacity.** As above — the headline risk. Treated as research-use; flagged for commercial review.
- **Length distribution.** Sentence-level pairs; very long paragraphs are rare, so the rendered images mostly carry 1–3 short blocks (matches the generator's 1–3-block layout, but under-represents dense multi-paragraph pages).

**Alternative (cleaner, smaller — for fixtures):** Tatoeba / WikiMatrix-style pairs carried over from P14 doctrans, **mostly CC-BY** (attribution required). Smaller and cleaner; useful for committed test fixtures where a permissive, attributable license matters more than scale.

### 2.2 `PleIAs/Post-OCR-Correction` (english) — optional OCR-noise text (NOT trained on)

| Field | Value |
|-------|-------|
| **HF id** | `PleIAs/Post-OCR-Correction` (english subset) |
| **Role** | **Optional, evaluation-side only.** Supplies realistic OCR-corrupted English text (with cleaned references) so the MT and the D3 confidence / D4 verification gates can be exercised against *real* OCR noise rather than only the generator's synthetic degradations. Never used to fine-tune the MT core. |
| **Size** | Large (PleIAs post-OCR collection is corpus-scale); P15 samples a small slice for robustness probes. |
| **Schema** | Paired `text` (OCR-noisy) / corrected `text` columns (PleIAs post-OCR-correction layout). |
| **License** | **CC0-1.0 (public domain).** Fully permissive — no attribution required, commercial-safe. |
| **Provenance** | PleIAs open-data initiative; OCR'd historical/public-domain materials with correction targets. |

**Known biases / limitations:** historical / public-domain skew (older orthography, period-specific vocabulary); English-only as used here; OCR noise distribution reflects the source scanning pipelines, which may differ from Tesseract-on-Colab error patterns. Used as a *stress signal*, not a training target.

---

## 3. Synthetic document-image generator — `data/synth_render.py` (PRIMARY)

This is the heart of P15's data story: the only source that provides true end-to-end ground truth `(image, gold_source, gold_target, boxes)`.

### 3.1 Role and outputs

For each sample index `i` the generator emits:

- `image.png` — a rendered page image (the model input), optionally degraded;
- a JSONL manifest row carrying the **gold spec**:
  `{ id, src, tgt, src_lang, tgt_lang, font, size, boxes:[[x0,y0,x1,y1], ...], degrade_params:{...}, seed }`.

`src` is the rendered source text (gold OCR target), `tgt` is the gold translation, `boxes` are the gold block layout boxes. These four facts — image, gold source, gold target, gold boxes — enable **all four metric families** (OCR, MT, end-to-end, layout) without any external labels.

### 3.2 Determinism (the central guarantee)

The generator is **seeded per index**, so the same index always yields the same image, the same gold spec, and therefore the same CER/chrF/fit-rate scores — fully reproducible tests and exact scoring.

- Per sample `i`: `rng = random.Random(BASE_SEED * 1_000_003 + i)` (and `np.random.default_rng(BASE_SEED*1_000_003 + i)` where numpy is used).
- **Every** random choice (canvas, font, size, layout jitter, ink color, each degradation toggle) draws from this `rng` — no leakage to global `random`/`numpy` state. Any such leak would break reproducible CER scoring and is called out as a hard discipline rule.
- The `(src, tgt)` pair is selected by **deterministic index** (not by `rng`), guaranteeing full, gap-free coverage of the corpus slice rather than random resampling.

### 3.3 Per-sample render pipeline

1. Pick the `i`-th `(src, tgt)` pair by index.
2. **Canvas:** size ∈ {(640,200), (800,256), (1024,300)}; background = solid paper (off-white / cream) **or** 2-color vertical gradient **or** faint procedural noise. Background recorded.
3. **Font:** `rng.choice` over a fixed list (Noto Sans, DejaVuSans, a serif, a mono where present); size ∈ [18, 40]; exact path recorded.
4. **Layout:** lay the **source** sentence into 1–3 blocks using the *same* greedy pixel word-wrap as the production render engine; random margins, line spacing ∈ [1.0, 1.4]. Each block's tight bbox (`multiline_textbbox`) is recorded → **these boxes are the gold layout**.
5. **Ink:** render in a high-contrast ink color drawn from a small palette.
6. **Degradations** (mild, OCR-survivable, each toggled by `rng` at a fixed probability, applied in a **fixed order** so they compose deterministically):
   - rotation `uniform(-3, 3)°`, `expand=True`, fill = background;
   - Gaussian blur radius ∈ {0, 0, 0, 0.6, 1.0};
   - Gaussian pixel noise σ ∈ {0, 4, 8}, clipped 0–255;
   - brightness/contrast jitter `uniform(0.85, 1.15)`;
   - JPEG recompression (in-memory `BytesIO`) at quality ∈ {95, 80, 60} then reload.
   A **CLEAN mode** (all degradations off) provides an OCR/end-to-end upper bound.
7. **Emit:** `image.png` + the JSONL manifest row above.

### 3.4 Coverage

- **Text coverage:** deterministic index selection guarantees every pair in the requested corpus slice is rendered exactly once — no resampling gaps or duplicates.
- **Visual coverage:** 3 canvas sizes × 3 background styles × a fixed font list × continuous size range [18,40] × 1–3 block layouts × 5 composable degradation axes. The cross-product spans a broad range of clean-document appearances.
- **Layout coverage:** 1–3 blocks with varied margins and line spacing exercises single-block, multi-block, and wrap-heavy cases that feed the fit-rate / IoU / no-overlap metrics.
- **Language coverage:** inherits from the corpus — en→fr by default; other directions render whatever scripts the chosen Noto fonts cover (Latin/Cyrillic/Greek default; CJK/Arabic/Devanagari/Hebrew/Thai via script-specific Noto files when present).

### 3.5 Scale

- **Eval:** default **200–2000 samples**, streamable for more.
- **Fixtures:** tiny **3–5 sample** sets at a fixed seed with **committed PNGs** for unit tests (so geometry/CER assertions are byte-stable across machines).

### 3.6 Intended use

- Primary end-to-end evaluation of the full OCR→MT→overlay cascade.
- Isolated OCR scoring (CER/WER vs `gold_source`) and isolated MT scoring (chrF/BLEU vs `gold_target`).
- Layout-fidelity scoring (fit-rate, box-retention IoU, no-overlap) against gold boxes.
- Deterministic regression fixtures and CI smoke tests.
- Ablations: clean-vs-degraded gap quantifies OCR cost; per-degradation sweeps probe robustness.

### 3.7 Limitations vs real photographs (read this before claiming generalization)

The generator produces **clean, programmatically rendered documents with mild synthetic degradations** — it is **not** a photograph of real-world signage, packaging, handwriting, or a phone-camera capture. Specifically it does **not** model:

- **real camera optics / lighting** — perspective warp, vignetting, glare, shadows, uneven illumination, motion blur beyond a small Gaussian;
- **complex/natural backgrounds** — textured surfaces, photographic scenes behind text, occlusion;
- **handwriting, stylized/decorative fonts, artistic typography, curved or wrapped text** (manga SFX, logos);
- **severe degradation** — heavy noise, creases, stains, ink bleed, faded scans, dense multi-column newspaper layouts, tables, forms;
- **mixed-script pages and right-to-left shaping subtleties** at scale.

Because the generator's boxes are produced by the **same greedy wrap algorithm** the render engine uses, the synthetic **fit-rate is optimistically high** (text was authored to fit). Real OCR boxes are noisier, so on real inputs fit-rate, IoU and CER will be **worse than the synthetic numbers** — the synthetic results are a **floor / wiring proof, not a generalization claim**. Per the design risk register: *do not overclaim generalization to photos of real signage.*

### 3.8 Known biases

- **Layout self-consistency bias:** gold boxes come from the same algorithm as rendering → flatters layout metrics.
- **Font/script bias:** limited to installed Noto/DejaVu fonts; offline-without-Noto renders **tofu** (boxes) for CJK/Arabic — affects pure-offline mode only and is flagged.
- **Degradation realism gap:** synthetic degradations are mild and analytic; they under-represent real-world severity (see §3.7).
- **Corpus inheritance:** all text-side biases of the source corpus (§2.1) carry straight into the images.

---

## 4. Offline backbone — SeedEngine seed pages + en→fr dictionary

To let tests and offline mode run with **no tesseract binary, no torch, no Noto fonts** (stdlib + Pillow only), the repo ships a deterministic backbone. Selection is automatic via capability probes (`try import` / `shutil.which('tesseract')`); env flag `P15_OFFLINE=1` pins stub mode.

### 4.1 SeedEngine (offline OCR)

| Field | Value |
|-------|-------|
| **Role** | Deterministic stand-in for Tesseract so the pipeline + metrics run end-to-end with nothing downloaded. |
| **Mechanism** | For a synthetic image with an available manifest, `OCR(image_i) := manifest[i].src` (reads the embedded gold spec, **not** pixels) → CER vs gold = **0.0** (perfect, clean). Optional deterministic corruption: seeded `random.Random(seed + i)` substitutes/drops a fixed ~3% of characters → a **known, reproducible** CER for exercising the error-rate code (mirrors P07's stub-OCR pattern). When `pytesseract` + binary are present, the real P07 path (`image_to_data` boxes+conf, born-digital router) is used instead — same code path, probe upgrades it. |
| **License** | MIT (this repo). |
| **Limitation** | It does **not read pixels** in stub mode — it reads the gold spec. CER 0.0 offline is a **wiring proof, not an OCR-quality claim**; realistic CER comes only from running real Tesseract on the rendered images (on Colab). |

### 4.2 Built-in en→fr dictionary (offline / baseline MT)

| Field | Value |
|-------|-------|
| **Role** | Doubles as the **dictionary baseline** and the **offline MT** when torch/m2m100 are absent. `translate(s) = ' '.join(dict.get(tok.lower(), tok) for tok in s.split())` — glossary lookup + identity passthrough for OOV. Deterministic, instant, no download. |
| **Size** | Small in-repo bilingual glossary `{src_token: tgt_token}` (hundreds of common tokens). |
| **License** | MIT (this repo). |
| **Known limitation / measurement caveat** | On the **offline seed pages the dictionary saturates** — verified seed eval shows **dictionary chrF 79.9 vs identity floor 22.4**, end-to-end chrF 76.4, fit-rate 1.0 — **because the seed pairs overlap the dictionary's vocabulary**. This is an artifact of the closed seed set, **not** evidence the dictionary rivals real MT. On real `opus-100` eval pairs the fine-tuned `m2m100_418M` dominates the dictionary — that is the **honest, non-saturated floor**. Word-by-word lookup ignores word order, morphology, and context, so it is a deliberately weak baseline. |

---

## 5. Fonts (rendering assets)

| Asset | License | Role | Note |
|-------|---------|------|------|
| Google **Noto Sans** family (split by script: Latin/Cyrillic/Greek, CJK `.ttc`, Arabic, Devanagari, Hebrew, Thai) | **SIL OFL 1.1** (permissive, redistributable) | Glyphs for the generator and the overlay render engine; script-aware selection by majority Unicode script of the target text. | Downloaded in the setup cell, cached, exact path recorded per block. Missing Noto files degrade to DejaVuSans without crashing. |
| **DejaVuSans.ttf** (shipped inside the Pillow wheel) | Permissive (Bitstream Vera / DejaVu license) | Always-present fallback for uncovered glyphs / offline mode. | Latin renders correctly offline; **CJK/Arabic render tofu offline — FLAGGED, affects pure-offline mode only.** Real runs ship Noto. |

---

## 6. Intended use (all data)

- Train and evaluate the **MT core** (`facebook/m2m100_418M`, MIT) for in-image document translation, default en→fr.
- Reproducibly evaluate the **full cascade** (OCR → MT → layout overlay) on synthetic document images with gold layout.
- Quantify OCR cost (clean-vs-degraded gap), MT quality (chrF/BLEU), and layout fidelity (fit-rate, IoU, no-overlap).
- Run fully **offline / in CI** via SeedEngine + dictionary MT + DejaVu fonts + pure-Python metrics.
- Research, coursework, and demonstration of a layout-preserving camera-translate pipeline with confidence/verification gates.

## 7. Out-of-scope use

- **Not a real-world benchmark.** Do **not** report synthetic numbers as evidence of accuracy on real photos of signage, packaging, handwriting, scanned forms, or phone-camera captures. The synthetic results are a floor/wiring proof (§3.7).
- **No high-stakes / authoritative translation.** The tool **assists** translation and **flags low-confidence output for human review**; it must never be presented as certified, certainty-asserting, or legally/medically authoritative output.
- **PII-sensitive documents.** Document images can contain personal data (IDs, passports, medical/legal records). These corpora and the generator are **not** licensed or designed for processing such PII; doing so requires consent, local processing, and no raw-image retention (default), per the ethics/privacy stance.
- **No commercial shipping of flagged sources.** `facebook/nllb-200-distilled-600M` (**CC-BY-NC-4.0**) and Surya (**CC-BY-NC-SA-4.0**) are non-commercial — research/quality comparison only, **never shipped**. Each `Helsinki-NLP/opus-100` pair's license must be **verified before commercial use**.
- **Not a training set for OCR or layout.** OCR and layout are pretrained/algorithmic in P15; the synthetic images are **not** intended as a corpus for training a custom OCR or detector (and the self-consistent gold boxes would bias such training).
- **Dictionary baseline is not a quality claim.** The saturated offline dictionary chrF (§4.2) must not be cited as competitive MT quality; it is a deliberately weak baseline that saturates only because seed vocabulary overlaps the glossary.

---

## 8. Provenance, licensing summary, and flags

| Source | License | Commercial-safe? | Action |
|--------|---------|------------------|--------|
| `Helsinki-NLP/opus-100` (en-fr) | Unknown / mixed per pair | **Unclear — FLAGGED** | Verify per-pair license before commercial use |
| `PleIAs/Post-OCR-Correction` | CC0-1.0 | Yes | None |
| Synthetic generator code (`data/synth_render.py`) | MIT (repo) | Yes | Rendered text inherits corpus license |
| SeedEngine + dictionary | MIT (repo) | Yes | None |
| Noto Sans | SIL OFL 1.1 | Yes | Redistribute with OFL notice |
| DejaVuSans (via Pillow) | Permissive | Yes | None |
| `facebook/nllb-200-distilled-600M` (model) | **CC-BY-NC-4.0** | **No — non-commercial** | Research only; **do not ship** |
| Surya `vikp/surya_rec2`+`surya_det3` (model) | **CC-BY-NC-SA-4.0** | **No — non-commercial + share-alike** | Research only; **do not ship** |

All HF ids above were confirmed to resolve on the Hugging Face Hub during research. Any id newly wired into the pipeline must be re-verified with `hub_repo_details`, and any per-pair `opus-100` license re-checked, before commercial deployment.
