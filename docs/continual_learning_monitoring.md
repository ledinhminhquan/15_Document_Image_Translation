# Continual Learning & Monitoring — P15 Document-Image Machine Translation

> Package `imgtrans` · folder `15_Document_Image_Translation` · author Le Dinh Minh Quan (23127460)
> Scope of this document: how the deployed cascade (**OCR → MT → layout-preserving overlay**) is observed in production, how human corrections of flagged pages are captured and fed back, and how the **single trainable component — the MT core `facebook/m2m100_418M` (MIT)** — is periodically re-fine-tuned under a dataset/version registry with explicit retraining triggers.

P15 is a cascade in which **only the MT stage is trained**. OCR (Tesseract), the born-digital router (PyMuPDF), the synthetic generator, and the PIL fit-to-box overlay are pretrained or algorithmic. That shapes everything below: there is exactly **one model to retrain**, but **three quality surfaces to monitor** (OCR quality, translation quality, layout/overlay fidelity) plus the **agent's own routing decisions** (needs-review rate, side-by-side rate). Continual learning therefore means "keep the MT checkpoint good and keep the cascade honest," not "retrain a monolith."

---

## 1. What we monitor and why

The agent (`src/imgtrans/agent/`, the deterministic five-decision FSM) already emits a structured signal at every decision point. Monitoring is built **on top of those signals** rather than bolted on. Each job that flows `ingest → ocr → translate → verify → render` writes one JSONL job-log row; `monitoring/drift_report.py` aggregates those rows into health metrics and drift checks.

The four monitoring surfaces and the agent decision point each is fed by:

| Surface | Headline metric(s) | Source signal (decision point) | Why it matters for P15 |
|---|---|---|---|
| **Routing health** | needs-review rate, side-by-side rate, overlay rate | D1/D5 final route | Rising needs-review = the cascade is silently giving up more often (bad inputs, OCR collapse, or MT drift). |
| **OCR quality** | mean OCR confidence, low-conf-block rate, born-digital-bypass rate | D3 per-block Tesseract confidence; D2 router | OCR error propagates into MT; this is the upstream drift surface. |
| **Translation quality** | round-trip chrF (proxy), length-ratio distribution, low-confidence flag rate | D4 round-trip back-translation + length ratio | The only reference-free in-production signal for MT quality drift. |
| **Layout fidelity** | fit-rate, mean shrink (final/requested font ratio), overflow rate | D5 `fit_box` returned `fit_ok` + chosen font size | Overlay is the headline deliverable; falling fit-rate = visibly worse output. |
| **Latency / cost** | p50/p95 end-to-end latency, per-stage latency (OCR / MT / render), pages/job | timing wrappers around each stage | OCR and MT dominate; latency drift flags model/binary/throughput regressions. |

Note that **D4 round-trip chrF is a soft proxy, not ground truth** (see the design brief, risk: "round-trip verification is heuristic"). It can false-flag legitimate free translations and miss fluent hallucinations. We monitor its *distribution over time* (drift), not its absolute value, and never treat a single round-trip drop as proof of MT regression — it is a trigger to look, not a verdict.

---

## 2. Job logs — the raw telemetry

Every production request (FastAPI `POST /translate-image` or `/translate-text`, or a Gradio `/ui` submission) appends exactly one row to a rolling JSONL log under `runs/serve/jobs-YYYYMMDD.jsonl`. The row is emitted by the agent's terminal state regardless of outcome (overlay, side-by-side, or needs_review), so the logs are complete by construction — there is no "success-only" sampling bias.

One job-log row (illustrative schema; all numbers are produced by the cascade itself, no human in the loop):

```json
{
  "job_id": "2026-06-27T10:14:22Z-7f3a",
  "ts": "2026-06-27T10:14:22Z",
  "model_version": "imgtrans-mt-m2m100-418M-ft-2026-05",
  "input_kind": "pdf",
  "route_d1": "pdf",
  "route_d2": "scanned",
  "born_digital_bypass": false,
  "n_pages": 3,
  "n_blocks": 41,
  "src_lang": "en",
  "tgt_lang": "fr",
  "ocr": {"engine": "tesseract", "mean_conf": 81.4, "low_conf_blocks": 3, "dropped_blocks": 1},
  "mt": {"backend": "m2m100_ft", "roundtrip_chrf_mean": 68.2, "len_ratio_mean": 1.18,
         "low_confidence_blocks": 2, "redecoded_blocks": 1},
  "render": {"fit_rate": 0.93, "mean_shrink": 0.86, "overflow_blocks": 3, "route_d5": "overlay"},
  "final_route": "overlay",
  "needs_review": false,
  "latency_ms": {"ocr": 2140, "mt": 880, "render": 310, "total": 3460},
  "flags": ["low_conf_block_dropped"]
}
```

Privacy constraint (carried from the ethics section): **the raw image and the recognized source/target text are NOT written to the job log by default.** Job logs hold *metrics and routing decisions only*. Per-block raw text is retained **only** when (a) the operator has enabled feedback capture for that tenant, and (b) the page was flagged `needs_review` or `side_by_side` — i.e. only the pages a human will actually review (Section 4). This keeps PII exposure (IDs, passports, medical/legal scans) bounded to the review queue, with local-processing and no-raw-retention-by-default as the operating defaults.

---

## 3. `monitoring/drift_report.py` — from job logs to health and drift

`monitoring/drift_report.py` is the periodic aggregator (reused and rewired from the P13/P14 monitoring template, extended with P15's OCR/layout surfaces). It takes a window of JSONL job logs plus a frozen **reference baseline** (the metric distribution measured on the held-out synthetic eval set at the time the current checkpoint was promoted) and produces a single report.

### 3.1 Aggregations (per window, default daily + 7-day rolling)

- **Routing mix**: `overlay% / side_by_side% / needs_review%`, plus born-digital-bypass rate.
- **OCR**: mean confidence, low-conf-block rate, dropped-block rate, bypass rate.
- **MT**: round-trip chrF mean and percentiles, length-ratio mean/IQR, low-confidence flag rate, re-decode rate.
- **Layout**: mean fit-rate, mean shrink, overflow rate.
- **Latency**: p50/p95 total and per stage.
- **Volume**: jobs, pages, blocks; input-kind mix (image/pdf/text); language-pair mix.

### 3.2 Drift checks (production window vs reference baseline)

Drift is flagged when a metric moves beyond a configured tolerance from the promotion-time baseline. Two complementary methods:

1. **Threshold drift** on the headline scalars — e.g. `mean_ocr_conf` drops > 8 points, `fit_rate` drops > 0.05, `needs_review_rate` rises > 0.03 absolute, p95 latency rises > 30%.
2. **Distribution drift** on the continuous signals — **Population Stability Index (PSI)** on the length-ratio distribution and the round-trip-chrF distribution, and PSI/KS on the OCR-confidence histogram. PSI > 0.2 on any of these raises a `distribution_drift` warning even if the mean has not moved (catches "same average, fatter tails" — e.g. a new document type that OCRs bimodally).

### 3.3 Output

`drift_report.py` writes `runs/monitoring/drift-YYYYMMDD.json` (machine-readable, for triggers) and a short markdown summary appended to the autoreport (human-readable). Each drift check resolves to `ok | watch | trigger`. A `trigger` status on a retraining-eligible metric is what feeds Section 7. The report is intended to be run on a schedule (cron / GitHub Action / HF Space scheduled job) and is also runnable on demand by an operator.

### 3.4 The two drift surfaces, separated

Because P15 is a cascade, OCR drift and MT drift must be **attributed separately**, otherwise a falling end-to-end quality number is un-actionable:

- **OCR-quality drift** (upstream): falling mean OCR confidence, rising low-conf/dropped-block rate, or an OCR-confidence histogram PSI breach. Causes: new scanner/phone-camera profiles, new fonts/scripts, degraded scan quality, a Tesseract/locale upgrade in the Docker base, or a shift in born-digital-vs-scanned mix. **OCR is pretrained and not retrained by us** — so OCR drift does *not* trigger an MT retrain. It triggers (a) an alert, (b) a check of the OCR engine/version and preprocessing (gamma/upscale/invert retry path in D3), and (c) inclusion of the new hard pages in the corrupted-OCR training set so the MT core becomes *more robust to that OCR noise* (Section 5).
- **Translation-quality drift** (the trained surface): round-trip chrF distribution sinking, length-ratio drifting out of band, low-confidence flag rate rising — *after* controlling for OCR (i.e. drift persists on born-digital-bypass jobs where OCR error is zero). This is the surface that genuinely points at the MT checkpoint and is the legitimate input to an MT retraining trigger.

The born-digital-bypass jobs are precious here: they have **no OCR error by construction**, so MT-quality drift measured on the bypass subset is a clean, OCR-free signal. `drift_report.py` reports MT metrics both overall and on the bypass-only slice for exactly this attribution.

---

## 4. Feedback capture — human corrections of flagged pages

The agent already separates pages into `overlay` (confident), `side_by_side` (didn't fit), and `needs_review` (low-confidence OCR or MT, or render infeasible). **Feedback capture is concentrated on the latter two**, because those are the pages where a human is expected to look — and they are also the pages most informative for retraining (the failure tail).

### 4.1 What is captured

When feedback capture is enabled for a tenant, a flagged page carries forward into a review record under `data/feedback/inbox/`:

```json
{
  "feedback_id": "fb-2026-06-27-001",
  "job_id": "2026-06-27T10:14:22Z-7f3a",
  "page": 2,
  "block_id": 17,
  "reason": "low_confidence",          // d3_low_conf | d4_low_conf | d5_no_fit | side_by_side
  "src_lang": "en", "tgt_lang": "fr",
  "ocr_text": "Pateint must fast for 8 huors",   // raw OCR (may contain OCR error)
  "mt_text": "Le patient doit jeûner pendant 8 heures",
  "human_src_correction": "Patient must fast for 8 hours",   // optional, human-typed
  "human_tgt_correction": "Le patient doit être à jeun pendant 8 heures",
  "human_verdict": "mt_wrong",         // ocr_wrong | mt_wrong | both_wrong | acceptable
  "reviewer": "anon-hash",
  "ts": "2026-06-27T11:02:00Z"
}
```

The reviewer surface is the existing Gradio `/ui` extended with a review tab: a flagged page shows the original crop, the OCR text, the proposed translation, and two editable fields (corrected source, corrected target) plus a verdict dropdown. Corrections are optional — even a bare verdict (`acceptable` / `mt_wrong` / `ocr_wrong`) is a useful label.

### 4.2 Why the verdict field matters for a cascade

The `human_verdict` field is what lets us **route the correction to the right stage**:

- `ocr_wrong` → the human source correction becomes a **(degraded-image-region → clean-source)** pair. We do *not* retrain Tesseract, but we add `(corrupted_ocr_text → human_src_correction → gold_target)` to the **OCR-robustness fine-tune set** so the MT core learns to translate *through* that class of OCR noise. It also flags a preprocessing gap to investigate in D3.
- `mt_wrong` (with OCR correct) → a clean **(source, human_tgt_correction)** parallel pair: the highest-value retraining signal, a real-world hard example the corpus missed.
- `both_wrong` → both corrections are captured; the pair contributes to both sets.
- `acceptable` → a positive label; used as a regression guard (these must *stay* acceptable after retraining) and to calibrate the D4 round-trip threshold (TAU).

### 4.3 Curation before it touches training

Raw feedback is **never** fed straight into a retrain. It passes through `data/feedback/curate.py`:

1. **PII scrub / consent check** — a page enters the training-eligible pool only if its tenant consented to feedback use; otherwise the correction improves *that job's output only* and is then discarded. Aggressive redaction of obvious PII spans (numbers matching ID/passport/SSN-like patterns) before any pair is persisted to the training pool.
2. **De-duplication** against existing training pairs (exact + near-dup via normalized text hash).
3. **Sanity filters** — length ratio in band, language-id check on both sides (corrected target really is the target language), drop empty/garbage.
4. **Reviewer agreement** — where two reviewers touched the same block, require agreement or escalate; single-reviewer corrections are tagged lower-weight.

Curated pairs land in `data/feedback/curated/feedback-YYYYMM.parquet`, versioned and registered (Section 6), ready to be mixed into the next fine-tune.

---

## 5. Periodic re-fine-tuning of the MT core

The retrain target is **always and only** `facebook/m2m100_418M` (MIT, many-to-many), fine-tuned with the HF `Seq2SeqTrainer` harness reused verbatim from P13/P14, headline metric **chrF**, secondary **BLEU**. No OCR, router, or render component is "trained" — they are upgraded by version bumps of the pretrained dependency, not by gradient descent here.

### 5.1 Training mixture for a refresh

Each refresh assembles a versioned mixture:

1. **Base corpus** — the `Helsinki-NLP/opus-100` en-fr pairs (license **flagged**: verify per-pair license before any commercial release; treat as the existing baseline corpus). The bulk of the data, keeps general competence.
2. **Curated feedback pairs** (Section 4.3) — the real-world hard tail. **Up-weighted** (e.g. oversampled 3–10×) relative to base, because they are exactly the cases production gets wrong, but capped so they cannot overwhelm the base distribution and cause catastrophic forgetting.
3. **OCR-robustness pairs** — `(corrupted_source → gold_target)` pairs, where the corrupted source is produced by the synthetic generator's **seeded deterministic OCR corruption** (the §8 stub corruption: substitute/drop ~3% of chars at a known seed) *and* by real OCR error harvested from `ocr_wrong` feedback. Optionally augmented from `PleIAs/Post-OCR-Correction` (english, **CC0** — safe to ship) as a real OCR-noise source. This is what makes the MT core robust to the cascade's own upstream noise — the single most cascade-specific reason P15 retrains.
4. **Synthetic regeneration** — `data/synth_render.py` is corpus-driven, so any new corpus or feedback batch can also be *rendered to images* to refresh the end-to-end eval set, keeping evaluation aligned with the new training distribution.

### 5.2 Retrain procedure (reproducible)

1. Freeze the mixture as a registered dataset version (Section 6).
2. Fine-tune `m2m100_418M` (fp16, small batch + grad-accum on the default A10/L4 tier; T4-compatible per the design brief) with the identical `Seq2SeqTrainer` config, logging chrF/BLEU per eval step.
3. Evaluate the candidate on **three frozen held-out sets**: (a) clean-source MT chrF/BLEU on the opus-100 held-out split (isolates MT), (b) end-to-end image-translation chrF on the synthetic held-out pages (`MT(OCR(image))` vs gold target), (c) a **feedback regression set** of past `acceptable` and corrected pairs (must not regress).
4. **Gate promotion** on a champion–challenger comparison (Section 5.3).

### 5.3 Champion / challenger promotion gate

A new checkpoint is promoted **only if** it beats the current production champion on the frozen eval sets by a margin exceeding noise, and regresses nothing material:

- end-to-end image-translation **chrF** improves (the headline production metric), AND
- clean-source MT chrF/BLEU does not regress beyond tolerance, AND
- the feedback regression set's `acceptable` pairs stay acceptable (no new failures introduced), AND
- no material rise in projected needs-review rate when the challenger is shadow-run over a recent job-log window.

If the gate passes, the challenger becomes champion and the production `model_version` string is bumped (e.g. `imgtrans-mt-m2m100-418M-ft-2026-08`). If it fails, the champion stays; the failed run is still registered for audit. A new champion resets the monitoring **baseline** in `drift_report.py` (the reference distribution is re-measured on the new checkpoint), so drift is always relative to the deployed model, not a stale one.

### 5.4 License discipline at retrain time

Retraining must not silently introduce a non-shippable model or corpus. The pipeline hard-blocks the **flagged non-commercial** options from the design brief: `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0), Surya (CC-BY-NC-SA-4.0). These may be used **only** in clearly-labelled research experiments that are never promoted to the shipped checkpoint. The corpus license check (opus-100 per-pair) and the CC0 status of `PleIAs/Post-OCR-Correction` are re-asserted on every mixture build and recorded in the dataset registry.

---

## 6. Dataset & model version registry

Continual learning is only safe if every promoted checkpoint is reproducible from registered inputs. P15 reuses the registry template from P13/P14 and records both **dataset versions** and **model versions**, linked.

### 6.1 Dataset registry (`runs/registry/datasets.jsonl`)

One row per frozen training/eval mixture:

```json
{
  "dataset_version": "ds-2026-08",
  "created": "2026-08-01",
  "components": [
    {"name": "opus-100-en-fr", "rows": 1000000, "license": "mixed/verify-per-pair", "flag": "verify_commercial"},
    {"name": "feedback-curated-2026-06..07", "rows": 1840, "license": "tenant-consented", "weight": 5},
    {"name": "ocr-robustness-synth+feedback", "rows": 22000, "license": "derived"},
    {"name": "post-ocr-correction-en", "rows": 50000, "license": "CC0"}
  ],
  "synth_eval": {"generator": "data/synth_render.py", "base_seed": 1234, "n_pages": 2000},
  "content_hash": "sha256:...",
  "notes": "feedback batch up-weighted 5x; nllb/surya excluded by license gate"
}
```

The `content_hash`, `base_seed`, and generator path make the synthetic eval set and the mixture **bit-reproducible** — the design brief's determinism discipline (`BASE_SEED*1_000_003 + i`, no global rng leakage) is what makes this hash stable across machines.

### 6.2 Model registry (`runs/registry/models.jsonl`)

One row per fine-tune run (promoted or not):

```json
{
  "model_version": "imgtrans-mt-m2m100-418M-ft-2026-08",
  "base_model": "facebook/m2m100_418M",
  "base_license": "MIT",
  "trained_on": "ds-2026-08",
  "git_sha": "....",
  "eval": {"mt_chrf_clean": 61.7, "mt_bleu_clean": 38.2,
           "e2e_chrf": 57.9, "feedback_regression_pass": true},
  "promoted": true,
  "champion_compared_to": "imgtrans-mt-m2m100-418M-ft-2026-05",
  "baseline_reset": true,
  "created": "2026-08-02"
}
```

Every production job-log row carries `model_version`, so any production metric or drift report can be joined back to the exact checkpoint, its training mixture, and the code SHA that produced it. This closes the loop: **job log → drift report → trigger → registered dataset → registered model → deployed `model_version` → job log.**

---

## 7. Retraining triggers

Retraining is **trigger-driven and gated**, not blindly periodic, but with a periodic floor so the model never goes stale silently. `monitoring/drift_report.py` emits the trigger conditions; a human (or the autopilot template) decides to act on a `trigger` status. Triggers are evaluated on the **MT-attributable** surface (Section 3.4) so OCR drift never spuriously fires an MT retrain.

| # | Trigger | Threshold (tunable) | Acts on |
|---|---|---|---|
| **T1 — Calendar floor** | scheduled refresh | every 1–3 months | Always retrain to fold in accumulated feedback; promote only if the gate passes. |
| **T2 — Feedback volume** | curated MT-correction pairs accumulated | ≥ N (e.g. 500) new `mt_wrong`/`both_wrong` pairs | Enough new hard examples to justify a refresh. |
| **T3 — Translation-quality drift** | round-trip chrF distribution PSI / mean drop on the **born-digital-bypass slice** (OCR-error-free) | PSI > 0.2 or mean drop > tolerance, sustained over the rolling window | Real MT drift → retrain candidate. |
| **T4 — Needs-review / side-by-side rise** | `needs_review_rate` or `side_by_side_rate` rises | > 0.03 absolute over baseline, sustained | The cascade is degrading gracefully more often — investigate cause, retrain if MT-attributable. |
| **T5 — Length-ratio drift** | length-ratio distribution drifts out of the D4/D5 band | PSI > 0.2 or median outside [0.4, 3.0] band shifting | Truncation/runaway behavior → MT decoding regression; retrain or re-tune decode params. |
| **T6 — Base dependency change** | new m2m100 base or HF/transformers/Pillow/Tesseract major bump | on dependency update | Re-fine-tune and re-baseline against the new stack; never auto-promote a dependency bump without the eval gate. |

Two surfaces that look like triggers but are **handled differently** because OCR is not trained by us:

- **OCR-quality drift** (falling confidence, rising dropped-block rate, OCR-confidence histogram PSI breach) → **alert + investigate, not an MT retrain**. Response order: (1) check OCR engine/locale/version in the Docker base and the D3 preprocessing retry path; (2) verify the input-mix shift (new scanner/camera/script); (3) harvest the hard pages into the OCR-robustness set so the *next scheduled* MT refresh becomes more noise-tolerant. An MT retrain is fired by OCR drift only indirectly, via T1's scheduled fold-in of those robustness pairs.
- **Latency drift** → ops alert (throughput, batch size, hardware tier, OCR/MT/render stage timings), **never** a model-quality retrain. p95 latency regression points at the serving environment, not the checkpoint.

A trigger **proposes** a retrain; the **champion–challenger gate (Section 5.3) disposes**. No checkpoint reaches production without beating the incumbent on end-to-end chrF and passing the feedback regression set — so a noisy trigger can at worst waste a training run, never ship a worse model.

---

## 8. Operating cadence (putting it together)

- **Continuously** — every job appends a metrics-only row to `runs/serve/jobs-*.jsonl`; flagged pages (needs_review / side_by_side, consented tenants) carry raw text into `data/feedback/inbox/`.
- **Daily** — `monitoring/drift_report.py` runs on a schedule, aggregates the window vs the current champion's baseline, writes `runs/monitoring/drift-*.json` + an autoreport markdown summary, and resolves each check to `ok | watch | trigger`.
- **Weekly** — operator triages the review queue in the Gradio review tab; `curate.py` promotes consented, scrubbed, de-duplicated corrections into `data/feedback/curated/`.
- **Monthly–quarterly (or on a fired trigger)** — assemble + freeze a registered dataset version, fine-tune `m2m100_418M`, evaluate on the three frozen sets, run the champion–challenger gate; on pass, promote, bump `model_version`, and reset the drift baseline.

This keeps the **one trained component** improving on exactly the data production struggles with, keeps the **two untrained quality surfaces** (OCR, layout) observed and attributed rather than conflated, and keeps every promoted checkpoint reproducible and license-clean — the non-commercial `nllb`/Surya options stay permanently fenced out of the shipped path.
