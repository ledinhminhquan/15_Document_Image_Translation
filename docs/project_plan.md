# P15 Document-Image Machine Translation — Project Plan

> Package: `imgtrans` · Folder: `15_Document_Image_Translation`
> Author: Le Dinh Minh Quan (student 23127460)
> Default direction: en→fr (configurable; the `m2m100` core is many-to-many)
> Companion documents: `docs/DESIGN_BRIEF.md` (authoritative spec), `docs/project_plan.md` (this file).

This plan turns the design brief into an executable schedule: what gets built, in what order, on what hardware, with which risks tracked and who owns what. It is deliberately concrete to P15 — an **OCR → MT → layout-preserving render** cascade in which the **MT core (`facebook/m2m100_418M`, MIT) is the only trained stage**, wrapped by a deterministic five-decision agent.

---

## 1. Goal and definition of done

**Goal.** Translate the text that appears *inside* an image, scanned page, or PDF and render the translation **back onto the page preserving spatial layout** (the Google-Translate-camera experience), while degrading gracefully — side-by-side or `needs_review` — instead of producing a broken image.

**Definition of done (ship criteria).**

1. One fine-tuned `m2m100_418M` checkpoint beats the dictionary and zero-shot baselines on **MT chrF** on a held-out **opus-100 en-fr** split (the honest, non-saturated floor).
2. The synthetic generator (`data/synth_render.py`) emits reproducible `(image, gold_source, gold_target, boxes)` quadruples and drives all four metric families: **MT chrF/BLEU, OCR CER/WER, end-to-end image-translation chrF/BLEU, layout fidelity** (fit-rate / mean shrink / overflow).
3. The render engine produces a layout-preserving overlay with auto font-fit; the five-decision agent routes `overlay` / `side_by_side` / `needs_review` deterministically.
4. The whole pipeline runs **fully offline** (SeedEngine OCR + dictionary MT + DejaVu fonts + pure-Python metrics) so CI passes with only Pillow.
5. FastAPI (`/translate-text`, `/translate-image`) + Gradio (`/ui`) + Docker (tesseract-ocr + libGL) + a published HF Space.
6. Autoreport, monitoring, grading, and autopilot templates wired (ported from P13/P14).

**Verified offline-seed numbers to reproduce** (sanity floor, not the headline result): MT dictionary chrF **79.9** vs identity floor **22.4**; OCR CER **0.0** (perfect SeedEngine OCR; realistic CER comes from Tesseract on Colab); end-to-end chrF **76.4**; mean fit-rate **1.0**. The dictionary saturates on the seed because seed pairs overlap the glossary — on real opus-100 eval pairs the fine-tuned m2m100 dominates, which is the number that goes in the report.

---

## 2. Pipeline diagram

```
                                  ┌─────────────────────────────────────────────┐
                                  │              AGENT (deterministic FSM)        │
                                  │   5 decision points, runs fully offline       │
                                  └─────────────────────────────────────────────┘

  input (image / PDF / .txt / spec)
        │
        ▼
  ┌───────────┐   D1 input router + page-quality gate
  │  INGEST   │── image → scanned branch ── pdf → D2 ── .txt → skip OCR → D3/translate
  └───────────┘   unsupported → needs_review
        │
        ▼
  ┌───────────┐   D2 born-digital vs scanned (PyMuPDF text-coverage probe, from P07)
  │    OCR    │── embedded text layer present → BYPASS OCR (lossless word boxes)
  │ front-end │── scanned / image → Tesseract image_to_data → words+boxes+conf
  └───────────┘     (SeedEngine offline fallback reads gold spec from synthetic image)
        │   text_src, boxes, conf  (aggregate to block/paragraph boxes, level 2/3)
        ▼
  ┌───────────┐   D3 per-block OCR-confidence gate
  │ TRANSLATE │── conf ≥ HIGH → MT ; LOW..HIGH → retry (gamma/upscale/invert)
  │  (TRAINED)│── conf < LOW or garbage → DROP block, tag needs_review (never mistranslate)
  └───────────┘     MT core = facebook/m2m100_418M fine-tuned (HF Seq2SeqTrainer, chrF)
        │   text_tgt
        ▼
  ┌───────────┐   D4 verify (soft gate): round-trip back-translation chrF + length-ratio
  │  VERIFY   │── ok → render ; round-trip < TAU → re-decode once → else low_confidence
  └───────────┘── length-ratio out of band → flag + feed ratio forward to D5
        │
        ▼
  ┌───────────┐   D5 render-fit feasibility gate (fit_box → fit_ok)
  │  RENDER   │── fits at ≥ min legible font → OVERLAY (erase-all-then-draw, fit-to-box)
  │ (PIL only)│── too long → SIDE_BY_SIDE (original + translated caption panel)
  └───────────┘── infeasible / flagged → NEEDS_REVIEW (boxes + raw translation, no destructive render)
        │
        ▼
   output image (same-size overlay PNG)  +  translated text  +  per-block decision trace
```

Only the **TRANSLATE** stage is trained. OCR is pretrained (Tesseract / SeedEngine), and RENDER is pure algorithm (Pillow fit-to-box: binary-search font size + greedy word-wrap, script-aware Noto/DejaVu fonts).

---

## 3. Milestones and timeline

Plan spans **8 weeks** of part-time effort (one student author) organised into nine milestones. Weeks overlap where a milestone is unblocked by an earlier deliverable. Each milestone lists its **exit criterion** — the concrete, checkable thing that marks it done.

### Milestone table

| # | Milestone | Week(s) | Key deliverables | Exit criterion | Hardware |
|---|-----------|---------|------------------|----------------|----------|
| **M0** | Research & scaffold | 1 | Verify every HF id (`hub_repo_details`), confirm licenses, port P02 repo/notebook/docs template, port config/logging/registry from P13/P14 | All ids resolve; non-commercial ids (nllb, Surya) flagged; offline smoke cell runs on stdlib+Pillow | CPU / free T4 |
| **M1** | Synthetic data generator | 1–2 | `data/synth_render.py` deterministic `(image, gold_src, gold_tgt, boxes)` quadruples; JSONL manifest; degradation suite; CLEAN mode; committed tiny fixtures (3–5 PNGs) | Same index → same image (hash-stable); 200–2000 sample eval set streams; fixtures committed | CPU |
| **M2** | MT fine-tune core | 2–3 | `m2m100_418M` fine-tuned via HF `Seq2SeqTrainer` on opus-100 en-fr; chrF/BLEU eval; dictionary + identity + zero-shot baselines | Fine-tuned chrF > zero-shot > dictionary > identity on held-out opus-100 split | **T4 (fp16) → A10/L4 default; H100 for mbart upgrade** |
| **M3** | OCR integration | 3–4 | Port P07 Tesseract `image_to_data` front-end + PyMuPDF born-digital router; SeedEngine offline OCR; CER/WER metrics | OCR CER/WER computed on synthetic (CLEAN + degraded); D2 bypass verified on a born-digital PDF | CPU (Tesseract) |
| **M4** | Overlay render engine | 4–5 | NEW `imaging/render.py`: whiteout/smear erase, script-aware font routing, `fit_box` (binary search + greedy/CJK wrap), contrast-aware draw, erase-all-then-draw | Verified Pillow case reproduces (380×100 box → size 33, 3 lines, `fit_ok=True`); fit-rate metric live | CPU |
| **M5** | Agentic FSM | 5–6 | `src/imgtrans/agent/` 5 decision points (D1–D5); overlay/side_by_side/needs_review ladder; optional advisory LLM brain OFF by default | Agent routes all three outcomes on crafted inputs; runs fully offline (SeedEngine+dict) | CPU |
| **M6** | End-to-end eval | 6 | End-to-end image-translation chrF/BLEU (`MT(OCR(image))` vs gold); layout fidelity (fit-rate/shrink/overflow); autoreport tables/plots | All four metric families reported; clean-vs-degraded gap (OCR cost) quantified | T4/A10 |
| **M7** | Deploy | 7 | FastAPI (`/translate-text`, `/translate-image` gated on python-multipart), Gradio `/ui`, Dockerfile (tesseract-ocr+libGL), HF Space | Image upload → translated overlay PNG via API; Space live; Docker builds | CPU host + HF Space |
| **M8** | Hardening & docs | 8 | Robustness pass (rotation/blur/multi-column/mixed-script), ethics/privacy doc, monitoring/grading/autopilot, final README + report | Grading harness green; ethics section complete; report reproduces headline numbers | CPU |

### Narrative timeline

- **Week 1 — Research & data start (M0, M1).** Re-verify all model/dataset ids against the brief's VERIFIED STACK; do not invent new ids. Stand up the P02 repo skeleton and the P13/P14 config/logging/registry/autoreport templates. Begin the synthetic generator — it is on the critical path because every downstream metric depends on its `(image, gold_src, gold_tgt, boxes)` quadruples.
- **Week 2 — Generator done, MT starts (M1, M2).** Lock determinism (`rng = random.Random(BASE_SEED*1_000_003 + i)`, index-chosen pairs for full coverage), commit fixtures, then kick off the m2m100 fine-tune on opus-100 en-fr.
- **Week 3 — MT lands, OCR integrates (M2, M3).** Confirm the fine-tune beats all three baselines on a real held-out split (the honest floor). Port the P07 OCR front-end and born-digital router; wire SeedEngine for offline.
- **Weeks 4–5 — Render + agent (M4, M5).** Build the NEW `imaging/render.py` fit-to-box engine, then the five-decision FSM on top of it. This is the headline value-add.
- **Week 6 — End-to-end eval (M6).** Run `MT(OCR(image))` vs gold; report layout fidelity; quantify the OCR-error gap (clean MT chrF − end-to-end chrF).
- **Week 7 — Deploy (M7).** FastAPI + Gradio + Docker + HF Space.
- **Week 8 — Hardening & docs (M8).** Robustness suite, ethics/privacy, monitoring/grading/autopilot, final report.

### Critical path

`M1 (generator) → M2 (MT) / M3 (OCR) → M4 (render) → M5 (agent) → M6 (end-to-end eval)`.
The generator (M1) is the single most upstream dependency: it produces the gold that scores OCR, MT, end-to-end, **and** layout. M4→M5 is the headline deliverable; slipping it is more costly than slipping deploy (M7), which can be reduced to API-only if time runs short.

---

## 4. Risk register

Severity = Low / Med / High. Likelihood = Low / Med / High. Ordered by combined exposure.

| ID | Risk | Likelihood | Severity | Mitigation | Owner |
|----|------|------------|----------|------------|-------|
| R1 | **No real in-image benchmark exists** (datasets research returned null) — risk of overclaiming generalization to photos of real signage | High | High | Build everything on the synthetic generator; degradation suite simulates scan noise; if a `hub_repo_details`-verified set is later found, add as held-out test. Do **not** claim a public benchmark. | Author |
| R2 | **Non-commercial license trap** — nllb-200-distilled-600M (CC-BY-NC-4.0) and Surya (CC-BY-NC-SA-4.0) are tempting for quality | Med | High | Hard rule: never ship them. Default stack is all MIT/Apache. opus-100 per-pair license verified before commercial claims. Flagged in registry. | Author |
| R3 | **Translation expansion breaks overlay** — targets routinely longer than source (esp. →fr, CJK→EN); box overflows | High | Med | This is expected, not a bug. D5 side-by-side branch + fit-rate metric exist precisely for it. `fit_box` shrinks to min legible, then degrades. | Author |
| R4 | **T4 OOM fine-tuning m2m100_418M** on free Colab | Med | Med | fp16 + small batch + grad-accum. If still OOM, drop to `opus-mt-en-fr` (en→fr only) and note loss of multilingual coverage. A10/L4 is the comfortable default. | Author |
| R5 | **OCR error propagates into MT** — garbled OCR mistranslated | Med | Med | D3 per-block confidence gate drops low-conf blocks (never translate garbage); D4 round-trip + length-ratio catches drift; D2 born-digital bypass eliminates OCR error on digital PDFs. | Author |
| R6 | **Whiteout patch visible** on textured/multi-color backgrounds (no reliable ink/bg color) | Med | Low | Estimate bg from 2px border-ring median; fall back to simple-inpaint horizontal smear + light blur; accept imperfect erase. | Author |
| R7 | **Determinism leak** to global `random`/`numpy` state breaks reproducible CER and fixtures | Med | Med | Per-index seeded rng only (`BASE_SEED*1_000_003+i`); fixture-hash test in CI; no global state in generator or stub corruption. | Author |
| R8 | **CJK/RTL rendering** — per-char wrap needed; RTL shaping best-effort without python-bidi/arabic-reshaper; offline-no-Noto renders tofu | Med | Low | Per-character wrap for no-space scripts; `align='right'` + optional bidi/reshaper extras (flag best-effort); ship Noto in real runs (tofu affects pure-offline test mode only). | Author |
| R9 | **Round-trip verification false-flags** legit free translations / misses fluent hallucinations | Med | Low | Treat D4 as a soft gate: re-decode once, then `low_confidence` — never a hard reject that silently drops content. | Author |
| R10 | **Pillow version floor** — `multiline_textbbox`/`textlength` need Pillow ≥ 9.2 | Low | Med | Pin Pillow ≥ 9.2 (fit algorithm verified on 12.2.0); CI guards the floor. | Author |
| R11 | **Missing/untagged license** on TrOCR/DiT (MIT upstream, no Hub tag) | Low | Low | Treat as MIT but document the gap; do not assert a tag the Hub does not surface. These are upgrades, not the shipped core. | Author |
| R12 | **opus-100 pair license unknown** | Med | Med | Verify the exact en-fr pair license before any commercial claim; flag in datasets doc. Tatoeba/WikiMatrix (CC-BY) as cleaner fixture fallback. | Author |
| R13 | **Schedule slip on M4/M5** (the headline render + agent) | Med | Med | These are the highest-value deliverables — protect their weeks; reduce M7 deploy to API-only (drop Docker/Space polish) before cutting render/agent scope. | Author |

---

## 5. Resource needs

### Compute (Colab tiers, per the brief's GPU defaults)

| Tier | Hardware | Used for | When |
|------|----------|----------|------|
| **Free** | Colab **T4** (~16 GB) | m2m100_418M fp16 fine-tune (small batch + grad-accum); Tesseract OCR (CPU); all offline tests | M2 first pass, all CPU-side work, CI |
| **Default** | **A10 / L4** (mid GPU) | Comfortable m2m100 fine-tune + eval; docTR front-end if used | M2 main runs, M6 eval |
| **Upgrade** | **H100 / A100** (80 GB) | mbart-large-50 quality upgrade; trocr-large-printed / GOT-OCR-2.0-hf on tough photos | Optional quality experiments only |

OCR (Tesseract), the render engine (Pillow), the agent FSM, and all metrics run **CPU-only**. GPU is needed exclusively for the MT fine-tune and any neural-OCR upgrade experiment.

### Software / accounts

- **HF Pro** (recommended): faster Hub downloads, Space hosting for the demo, model/dataset hosting for the fine-tuned checkpoint.
- **Hugging Face account** (authenticated as `ledinhminhquan`) for `hub_repo_details` id verification and the published Space.
- **Anthropic API key** (optional): the advisory LLM "brain" in the agent is **OFF by default** and never rewrites output — only needed if the optional advisory path is demoed.
- System packages for Docker/deploy: `tesseract-ocr`, `libGL` (PyMuPDF/Pillow runtime).
- Python deps: `transformers`, `torch`, `sacrebleu`, `pytesseract`, `PyMuPDF`, `Pillow ≥ 9.2`, `fastapi`, `gradio`, `python-multipart` (the `/translate-image` route is gated on it), optional `python-bidi` + `arabic-reshaper` for RTL shaping.
- Fonts: Google **Noto Sans** (SIL OFL 1.1, split by script) downloaded in the Colab setup cell and cached; **DejaVuSans** (shipped inside the Pillow wheel) as the always-present offline fallback.

### Data

- **MT corpus:** `Helsinki-NLP/opus-100` en-fr (~1M pairs; license unknown → verify per pair, flagged). No corpus committed — loaded in the Colab setup cell; the generator is corpus-agnostic (`list[(src, tgt)]`).
- **Optional OCR-noise text:** `PleIAs/Post-OCR-Correction` english (CC0).
- **Primary eval data:** synthetic, generated by `data/synth_render.py` (gold is known by construction).
- **Offline backbone:** built-in synthetic seed pages + in-repo en→fr dictionary.

---

## 6. Division of work

Single author (Le Dinh Minh Quan, 23127460). "Division of work" is organised by **workstream** so dependencies and reuse are explicit, not by team. The agentic tools/automation templates assist with scaffolding and reporting.

| Workstream | Scope | Reuse source | New code |
|------------|-------|--------------|----------|
| **Data** | Synthetic generator, manifest, degradations, fixtures; corpus loading | corpus-loading pattern from P14 | `data/synth_render.py` (NEW) |
| **MT core** | m2m100 fine-tune harness, baselines, chrF/BLEU | P13 s2st / P14 doctrans Seq2SeqTrainer + dictionary/identity baselines + sacrebleu plumbing | config/training glue only |
| **OCR** | Tesseract front-end, born-digital router, SeedEngine, CER/WER | P07 dococr OCR + PyMuPDF router + stub-OCR pattern | SeedEngine offline OCR (NEW), CER/WER wiring |
| **Render** | Erase, font routing, fit-to-box, draw, layout metrics | — | `imaging/render.py` (NEW), fit-rate/IoU/no-overlap metrics (NEW) |
| **Agent** | 5-decision FSM, degradation ladder, optional advisory brain | autopilot/monitoring/grading templates from P13/P14 | `src/imgtrans/agent/` 5 decision points (NEW), end-to-end chrF wiring (NEW) |
| **Deploy** | FastAPI, Gradio, Docker, HF Space | deploy template from P13/P14 | route handlers + Dockerfile (tesseract+libGL) |
| **Docs/QA** | Design brief, this plan, README, ethics/privacy, report | P02 docs template | ethics/privacy section, final report |

**Reuse summary.** OCR plumbing + layout + preprocess come from **P07 dococr**; the MT translator + chrF/BLEU metrics + config/logging/registry/autoreport/monitoring/grading/autopilot templates come from **P13 s2st / P14 doctrans**. **NEW for P15:** the `imaging/render.py` fit-to-box overlay engine, `data/synth_render.py` synthetic generator, the SeedEngine offline OCR, and the five-decision image-translation agent.

---

## 7. Ethics, privacy, and robustness (carried into the plan)

These are not an afterthought — they are scheduled work in **M8** and a constraint on **M7**.

- **PII.** Document images can contain passports, IDs, medical/legal records. Defaults: local processing, **no raw-image retention**, explicit consent language in the UI. The tool **assists** translation and **flags low-confidence output for human review** (D3/D4/`needs_review`) — it never asserts certainty.
- **Honest claims.** No public in-image-translation benchmark exists; the report says so plainly and rests results on the synthetic generator. Non-commercial models (nllb, Surya) are flagged and excluded from the shipped default.
- **Robustness work (M8).** Degraded scans, rotation/blur, multi-column, mixed scripts, and OCR-error propagation into MT — mitigated by the post-OCR confidence gate (D3), round-trip + length-ratio verification (D4), and the overlay→side-by-side→needs_review degradation ladder (D5).

---

## 8. Open items to verify before/at kickoff

1. Re-run `hub_repo_details` on every shipped id (`facebook/m2m100_418M`, `Helsinki-NLP/opus-mt-en-fr`, `facebook/mbart-large-50-many-to-many-mmt`, Tesseract is a system binary) to confirm license tags still resolve.
2. Confirm the **opus-100 en-fr** pair license before any commercial claim (R12).
3. Confirm T4 fits the m2m100_418M fp16 fine-tune; if not, fall back to `opus-mt-en-fr` (R4).
4. Pin **Pillow ≥ 9.2** in the environment and guard it in CI (R10).
5. Verify Noto-per-script font download URLs and cache path; confirm DejaVuSans path inside the installed Pillow wheel (offline fallback).
