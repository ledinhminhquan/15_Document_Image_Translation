# P15 Agent Architecture — Document-Image Machine Translation

> Package `imgtrans` · Author Le Dinh Minh Quan (student 23127460)
> Component: the **mandatory agentic layer** that wraps the OCR → MT → render cascade.
> Source: `src/imgtrans/agent/`. Default direction en→fr (configurable; the m2m100 core is many-to-many).

---

## 1. What the agent is (and what it is not)

P15 translates the text that appears **inside** an image, a scanned document, or a born-digital PDF, and renders the translation **back onto the page** preserving the original spatial layout — the Google-Translate-camera experience. The underlying pipeline is a **cascade**, not an end-to-end model:

```
ingest ──▶ OCR front-end ──▶ MT (the ONLY trained stage) ──▶ layout-preserving overlay render
```

The **agent** is the controller that sits on top of this cascade. It is a **deterministic finite-state machine (FSM)** — not an LLM agent and not a learned policy. It routes on the pipeline's **own intermediate signals** (file type, embedded-text coverage, OCR confidence, round-trip back-translation, length ratio, render-fit feasibility) and always emits a **defensible output**: a clean `overlay`, a `side_by_side` panel, or an honest `needs_review` — never a silently broken image.

Three properties define it:

- **Deterministic.** Given the same input and the same `AgentConfig`, the agent visits the same states, makes the same branch decisions, and produces byte-identical output. No randomness, no hidden global state, no network call required. This is what makes the offline seed evaluation reproducible (MT dictionary chrF 79.9 vs identity floor 22.4; OCR CER 0.0; end-to-end chrF 76.4; mean fit-rate 1.0).
- **Fail-soft.** Every decision point has a degrade branch. Low-confidence OCR is **skipped, not mistranslated**; a translation that will not fit its box becomes a **side-by-side** caption instead of spilling over neighbors; anything genuinely uncertain is surfaced as **`needs_review`** rather than asserted as correct.
- **Self-checking without a reference.** The verification gate (D4) uses the model's own back-translation and length statistics — it needs **no gold translation at inference time**, so it works on real user uploads, not just on synthetic eval pages.

**Not** in the agent's job description: training (only the MT core is trained, via `Seq2SeqTrainer`), OCR (pretrained Tesseract / SeedEngine), layout detection (Tesseract `image_to_data` block boxes), or rendering geometry (the PIL fit-to-box engine in `imaging/render.py`). The agent **orchestrates** those; it does not reimplement them.

---

## 2. The state machine

Five states, executed left to right, each guarded by one decision point (D1–D5). The decision points gate on the intermediate signal that state produces and choose the next edge.

```
        D1                 D2                  D3                    D4                      D5
   ┌─────────┐        ┌─────────┐        ┌───────────┐        ┌────────────┐         ┌──────────┐
──▶│ INGEST  │──pdf──▶│   OCR   │──block▶│ TRANSLATE │──block▶│  VERIFY    │──block─▶│  RENDER  │──▶ result
   │ (router │        │ (route  │  text  │ (per-block │  MT    │ (round-trip│  fit?   │ (fit gate│
   │ +quality│        │  scan   │        │  conf gate)│ output │  + length) │         │  overlay │
   │  gate)  │        │  vs born│        │            │        │            │         │  /SbS    │
   └────┬────┘        │ digital)│        └─────┬─────┘        └─────┬──────┘         │ /review) │
        │ image       └────┬────┘              │ low-conf           │ low/len             └────┬─────┘
        │ ──────────────▶  │                   │ block              │ block                    │
        │ text ─────────────────────────────────────────────────▶ (skip OCR, jump to D4)      │
        │ unsupported ─▶ needs_review          ▼ drop block         ▼ flag low_confidence       ▼
        ▼                                  needs_review tag    (soft, forward to D5)    overlay | side_by_side
   page-quality gate                                                                    | needs_review
```

Each stage records a `ToolTrace` entry (Section 5) so the whole run is auditable. State transitions are **edges chosen by D1–D5**; there is no back-edge except the bounded, single-shot retries inside D3 (re-OCR) and D4 (re-decode).

### State responsibilities

| State | Reads | Produces | Gate |
|-------|-------|----------|------|
| **ingest** | raw bytes (image/pdf/spec/text) | input kind, page list, quality flag | **D1** |
| **ocr** | page image or PDF | per-block `(text, box, conf)` | **D2** |
| **translate** | accepted source blocks | per-block MT target | **D3** |
| **verify** | MT target + back-translation | accept / re-decode / `low_confidence` | **D4** |
| **render** | targets + boxes + fit flags | overlay PNG / side-by-side / review payload | **D5** |

---

## 3. The five decision points

All thresholds below are fields of **`AgentConfig`** (in `src/imgtrans/agent/config.py`) so the entire decision surface is tunable from one dataclass and serialized into every run's `ToolTrace`. The values shown are the shipped defaults.

### D1 — Input router + page-quality gate (state: `ingest`)

- **Gates on:** file magic-bytes / MIME + extension, then a cheap page-quality probe (resolution and, for raster input, an estimated blur/contrast score).
- **Branches:**
  - **image** (`png`/`jpg`/`webp`) → scanned branch, OCR required → go to **ocr** (D3 applies; D2 is a no-op for raster).
  - **pdf** → go to **ocr** with the **D2** born-digital-vs-scanned router engaged.
  - **spec** (a synthetic-generator manifest row, offline) → SeedEngine reads the embedded gold spec; OCR is satisfied from the manifest.
  - **text / `.txt`** → degenerate input, **skip OCR entirely**, jump straight to **D4** (translate then verify; there is no image to render onto, so D5 emits text-only).
  - **unsupported** MIME/extension → `needs_review` immediately (no guessing).
- **Page-quality gate:** if a page is below `min_page_dpi` or its blur/contrast score is worse than `page_quality_min`, the page is **tagged** (not dropped) so D3 can interpret low OCR confidence in context and D5 can prefer `needs_review` over a confident-looking-but-wrong overlay.
- **Config:** `supported_image_exts`, `supported_doc_exts`, `min_page_dpi`, `page_quality_min`.

### D2 — Born-digital vs scanned routing (state: `ocr`, PDF only)

- **Gates on:** the PyMuPDF embedded-text probe ported **verbatim from P07 dococr** — `page.get_text()` character count and text-coverage ratio per page.
- **Branches:**
  - **embedded text present** (coverage ≥ `born_digital_text_ratio`) → **extract the digital text layer + word boxes directly and BYPASS OCR.** This is lossless: zero OCR error on the common digital-PDF case, and the boxes are exact.
  - **little / no embedded text** (scanned PDF, coverage below threshold) → **rasterize** the page at render DPI and route it through OCR exactly like a raster image.
- **Why it matters:** the born-digital bypass eliminates the entire OCR-error term on a large fraction of real documents — it is the cheapest correctness win in the pipeline.
- **Config:** `born_digital_text_ratio`, `raster_dpi`.

### D3 — Per-block OCR-confidence gate (state: `translate` entry)

- **Gates on:** Tesseract `image_to_data` per-block **mean confidence** (0–100), aggregated to block/paragraph level (`level=2/3` — never per-word, which would destroy meaning and wrecking wrapping), plus a **character-sanity** check (alpha ratio and `min_text_length`).
- **Branches:**
  - **`conf ≥ ocr_conf_high`** (default ≈ 75) → **accept**, send the block to MT.
  - **`ocr_conf_low ≤ conf < ocr_conf_high`** (default ≈ 40–75) → **single bounded retry**: gamma-correct / upscale / invert the block crop and re-OCR; if the re-scored confidence clears `ocr_conf_high`, accept; otherwise treat as low. (An optional VLM-OCR fallback is documented but off by default.)
  - **`conf < ocr_conf_low`** OR block fails the char-sanity check OR is empty → **drop the block and tag `needs_review`.** The system **never translates garbage** — a confidently-rendered mistranslation of OCR noise is worse than an honest gap.
- **Config:** `ocr_conf_high`, `ocr_conf_low`, `min_text_length`, `min_alpha_ratio`, `ocr_retry_enabled`.

### D4 — Translation verification: round-trip + length ratio (state: `verify`)

- **Gates on:** two signals computed from the **model's own output**, requiring **no reference translation**:
  - **(a) round-trip back-translation chrF** — translate the target back to the source language and score chrF/BLEU between the original source and the back-translated source.
  - **(b) target/source length ratio** — character (or token) length of the target divided by the source.
- **Branches:**
  - **round-trip ≥ `roundtrip_tau`** (default ≈ 0.45) **AND** length-ratio ∈ `[len_ratio_min, len_ratio_max]` (default `[0.4, 3.0]`) → **accept**.
  - **round-trip < `roundtrip_tau`** (likely hallucination/drift) → **re-decode once** with alternate params (more beams, sampling off); if it still fails, flag the block **`low_confidence`** — a **soft** flag, never a silent drop.
  - **length-ratio out of band** (truncation, or runaway repetition) → flag, **and forward the ratio to D5**: an over-long translation is a strong predictor that the box will overflow, so D5 can pre-emptively prefer side-by-side.
- **Soft-gate discipline:** round-trip chrF can false-flag legitimate free translations and miss fluent hallucinations, so D4 is deliberately a soft gate (re-decode once, then mark `low_confidence`) — it never hard-rejects and silently discards content.
- **Config:** `roundtrip_tau`, `len_ratio_min`, `len_ratio_max`, `verify_redecode_enabled`, `redecode_num_beams`.

### D5 — Render-fit feasibility gate (state: `render`)

- **Gates on:** the **`fit_ok`** flag returned by the render engine's `fit_box` binary search (`imaging/render.py`) for each block — does the translated string wrap within the box height when shrunk only down to `font_size_min` legible pixels? The decision uses translated char count, box width/height, and the **D4 length ratio**.
- **Branches:**
  - **fits at ≥ min legible font** (`fit_ok == True` and `final_font_size ≥ font_size_min`) → **OVERLAY** — erase the box (whiteout/smear) and re-render the fitted, wrapped translation in place. The full camera experience.
  - **does not fit** (translation too long — the common JA/ZH→EN and →fr expansion case) → **SIDE_BY_SIDE** — leave the original page untouched and emit a translated caption panel beside it. No destructive, overflowing render.
  - **fit fails entirely** OR the block was flagged `needs_review` by D3 OR `low_confidence` by D4 → **`needs_review`** — emit the boxes and the raw translation without any destructive render, for a human to confirm.
- **Page-level aggregation:** the page result is the **worst-acceptable** branch across its blocks — if `fit_rate` (fraction of blocks with `fit_ok`) ≥ `overlay_fit_rate_min` and no block is `needs_review`, the page renders as `overlay`; otherwise it degrades to `side_by_side`; if any block is review-flagged it is `needs_review`.
- **Config:** `font_size_min`, `overlay_fit_rate_min`, `side_by_side_enabled`.

---

## 4. Decision table

| Point | State | Intermediate signal | Threshold (`AgentConfig` default) | Accept branch | Degrade / alternate branch | Fail branch |
|-------|-------|---------------------|-----------------------------------|---------------|----------------------------|-------------|
| **D1** | ingest | MIME + magic-bytes + extension; page-quality score | `supported_*_exts`; `min_page_dpi`, `page_quality_min` | image→ocr · pdf→ocr(+D2) · text→jump D4 · spec→SeedEngine | low-quality page → **tag** (carries to D3/D5) | unsupported type → **needs_review** |
| **D2** | ocr (PDF) | PyMuPDF `get_text()` coverage ratio (P07 router) | `born_digital_text_ratio` | coverage ≥ thr → **use text layer, BYPASS OCR** | — | coverage < thr → rasterize → OCR |
| **D3** | translate-in | Tesseract block mean conf (0–100) + char sanity | `ocr_conf_high≈75`, `ocr_conf_low≈40`, `min_text_length`, `min_alpha_ratio` | conf ≥ high → **MT** | low ≤ conf < high → re-OCR (gamma/upscale/invert), re-score | conf < low / empty / non-text → **drop + needs_review** |
| **D4** | verify | round-trip back-translation chrF; target/source length ratio | `roundtrip_tau≈0.45`; `len_ratio_min=0.4`, `len_ratio_max=3.0` | rt ≥ τ AND ratio in-band → **accept** | rt < τ → re-decode once (more beams); else **low_confidence** | ratio out-of-band → flag + **forward ratio to D5** |
| **D5** | render | `fit_ok` from `fit_box` binary search + D4 ratio | `font_size_min`, `overlay_fit_rate_min` | fits ≥ min font → **OVERLAY** | does not fit → **SIDE_BY_SIDE** caption panel | fit fails / D3 / D4 flagged → **needs_review** |

---

## 5. ToolTrace audit

Every state appends a structured record to a per-run **`ToolTrace`** (a list of step dicts, serializable to JSON and emitted alongside the result). Each entry captures:

- **`step`** — the state name (`ingest`/`ocr`/`translate`/`verify`/`render`).
- **`decision`** — which decision point fired (`D1`…`D5`) and the **branch taken** (e.g. `D2:born_digital_bypass`, `D3:drop_low_conf`, `D5:side_by_side`).
- **`signal`** — the actual measured value the branch turned on (coverage ratio, block confidence, round-trip chrF, length ratio, `fit_ok`/`final_font_size`).
- **`threshold`** — the `AgentConfig` value it was compared against, so the run is reproducible and the decision is explainable after the fact.
- **`tool`** — which backend served the step (`tesseract` vs `seed_engine`; `m2m100` vs `dictionary`; `pymupdf` text-layer vs raster), making the offline/online path explicit.
- **`block_id`** and per-block outcome where the step is per-block (D3/D4/D5 operate block-wise).

The ToolTrace is the **audit and debugging spine**: it answers "why is this block a side-by-side caption and not an overlay?" (because `fit_ok=False` at `font_size_min`), or "why was this region left blank?" (because D3 dropped it at conf 31 < `ocr_conf_low`). It also feeds the autoreport/monitoring templates reused from P13/P14, and is what makes the agent **explainable rather than a black box**.

---

## 6. Optional LLM brain (off by default, advisory only)

The agent ships with an **optional LLM "brain"** (an `anthropic` client) that is **OFF by default** and, when enabled, is **strictly advisory**:

- It runs **fully offline without it.** The complete decision ladder D1–D5 is deterministic threshold logic over numeric signals; the LLM is never on the critical path. Disabling it changes nothing about the output for a given input.
- When enabled, it may **annotate** a borderline decision (e.g. suggest that a `low_confidence` D4 flag is likely a legitimate free translation, or summarize why a page degraded to side-by-side) — but it **never rewrites the translation, never overrides a D1–D5 branch, and never moves a threshold.** The FSM remains the sole authority on routing.
- This keeps the agent **reproducible, auditable, and privacy-preserving** (Section 8): no document content has to leave the machine for the agent to function, which matters because document images frequently contain PII.

The LLM brain is therefore a comfort feature for human reviewers, not a control component. Determinism and the no-network guarantee are preserved precisely because it is advisory and default-off.

---

## 7. Fail-soft behaviour (the degradation ladder)

The agent's central design commitment is that it **degrades gracefully instead of producing a broken image or a confident lie.** Three concrete mechanisms:

1. **Skip low-confidence OCR (D3).** Blocks below `ocr_conf_low`, or that fail the character-sanity check, are **dropped and tagged `needs_review`**, never translated. Garbled OCR is the most common source of nonsense translations; refusing to translate it is strictly safer than rendering a fluent mistranslation of noise.
2. **Side-by-side fallback (D5).** When a translation cannot fit its source box at the minimum legible font — the routine →fr and CJK→EN expansion case — the agent does **not** spill text over neighboring blocks. It leaves the original untouched and renders a **side-by-side caption panel**. This is an expected, designed outcome, not a failure; the fit-rate metric exists to measure exactly this.
3. **`needs_review` honesty (D1/D3/D4/D5).** Unsupported inputs (D1), dropped garbage blocks (D3), unrecoverable verification flags (D4), and infeasible renders (D5) all converge on a `needs_review` result that **emits the boxes and the raw translation without a destructive render.** The tool **assists** translation and flags uncertain output for human review — it never asserts certainty it does not have.

The single-shot retries (D3 re-OCR, D4 re-decode) are bounded — each fires at most once per block — so the FSM always terminates and never loops.

---

## 8. Value-add — why this beats "OCR → translate → print the string"

A naive `OCR → MT → print` script reads text, translates it, and dumps the string. P15's agent adds two things that are exactly what separates a real camera-translate product from that script:

1. **Layout-preserving overlay with auto font-fit.** Translations are drawn **back where the source text was** — detect block → erase (whiteout / smear) → re-render the target with binary-search shrink-to-fit, greedy pixel/CJK word-wrap, script-aware font selection, contrast-aware color, and erase-all-then-draw ordering. This is the single most visible deliverable and is impossible without the overlay engine plus the D5 fit gate.
2. **Confidence + verification gates on the model's own intermediate outputs.** OCR confidence (D3) so garbled text is never translated; round-trip back-translation + length ratio (D4) so MT hallucination/truncation is caught **without any reference translation**; render-fit feasibility (D5) so the system degrades `overlay → side_by_side → needs_review` instead of overflowing boxes. This self-checking degradation ladder **is** the agentic part. Bonus: D2's born-digital bypass eliminates OCR error entirely on the common digital-PDF case.

Together these turn a brittle one-shot script into a system that produces a **defensible output on every input** — a clean overlay when it can, an honest fallback when it cannot.

---

## 9. Offline operation and reproducibility

The agent runs **fully offline** with no model downloads: the **SeedEngine** satisfies OCR by reading the gold spec embedded in synthetic images (D1 `spec` branch / D3 perfect confidence), and the **dictionary MT** baseline serves the translate stage. Under this stack the verified seed evaluation is MT dictionary chrF 79.9 vs identity floor 22.4, OCR CER 0.0 (perfect-OCR via SeedEngine; realistic CER appears with Tesseract on Colab), end-to-end chrF 76.4, and mean fit-rate 1.0. The dictionary saturates on the seed because seed pairs overlap the dictionary; on real `opus-100` eval pairs the fine-tuned `facebook/m2m100_418M` dominates — the honest, non-saturated floor. The same code path upgrades to Tesseract + fine-tuned m2m100 when those components are present; no agent code changes between offline and online runs, only the `tool` recorded in each ToolTrace entry.

> **License note.** The shipped agent stack is fully permissive: `facebook/m2m100_418M` (MIT) MT core, Tesseract / `pytesseract` (Apache-2.0) OCR, PIL-only render with Noto/DejaVu (OFL/permissive) fonts. **Non-commercial models are flagged and never routed to by default:** `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0) and Surya (CC-BY-NC-SA-4.0) are documented as research-quality upgrades only and are **not shippable**. The `opus-100` corpus license must be verified per pair before commercial use.

---

## 10. Ethics, privacy, and robustness

Document images routinely contain **PII** — IDs, passports, medical and legal documents. The agent is built to respect that: it runs **locally by default** (the offline SeedEngine + dictionary stack needs no network, and the LLM brain is off), retains **no raw image** by default, and the tool **assists** translation while **flagging low-confidence output for human review** (the `needs_review` and `low_confidence` paths) rather than asserting certainty. On robustness, the FSM is explicitly designed for degraded scans, rotation/blur, multi-column layouts, mixed scripts, and **OCR-error propagation into MT** — the post-OCR confidence gate (D3) and the round-trip verification gate (D4) are precisely the mitigations for that error chain, and D5 ensures a failed fit never produces a corrupted page.
