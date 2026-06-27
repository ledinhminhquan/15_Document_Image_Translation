# Document-Image Machine Translation (`imgtrans`)

> Translate the text **inside** an image, scan, or born-digital PDF, and render the translation
> **back onto the page** preserving the layout — like the Google Translate camera, but as a
> debuggable, license-clean, production-grade pipeline.
>
> **NLP in Industry — Final Assignment, Project #15.** Author: **Le Dinh Minh Quan** (Student 23127460).

`imgtrans` is a **cascade**: an **OCR** front-end reads the text + boxes, a **trainable
machine-translation core** (`facebook/m2m100_418M`, fine-tuned) translates each block, and a
**layout-preserving overlay renderer** re-draws the translation into the original boxes. A
deterministic **agent (D1–D5)** orchestrates it and gates OCR confidence, translation quality, and
overlay fit. Only the MT model is trained; OCR + layout + render are pretrained/algorithmic.

```
image / scan / PDF ──► OCR (Tesseract) ──► layout (blocks+boxes) ──► MT (m2m100, trained)
                                                                          │
        translated image (overlay) ◄── render (fit-to-box) ◄── verify (round-trip) ◄┘
```

---

## How this repo meets each assignment requirement

| Requirement | Where it is delivered |
|---|---|
| **Problem definition** | [docs/problem_definition.md](docs/problem_definition.md) |
| **Data description + data card** | [docs/data_description.md](docs/data_description.md), [docs/data_card.md](docs/data_card.md); synthetic generator [src/imgtrans/data/synth_render.py](src/imgtrans/data/synth_render.py) |
| **Model selection + baseline** | [docs/model_selection.md](docs/model_selection.md); MT core [src/imgtrans/mt/translator.py](src/imgtrans/mt/translator.py); baselines [src/imgtrans/models/baseline.py](src/imgtrans/models/baseline.py) |
| **Training + evaluation** | [src/imgtrans/training/train_mt.py](src/imgtrans/training/train_mt.py), [evaluate.py](src/imgtrans/training/evaluate.py); [docs/layout_fidelity_evaluation.md](docs/layout_fidelity_evaluation.md) |
| **Agentic AI component** | [src/imgtrans/agent/](src/imgtrans/agent/) — 5 decision points; [docs/agent_architecture.md](docs/agent_architecture.md) |
| **Deployment / serving** | FastAPI [src/imgtrans/api/main.py](src/imgtrans/api/main.py) + Gradio [ui.py](src/imgtrans/api/ui.py); [docs/deployment.md](docs/deployment.md) |
| **Continual learning + monitoring** | [src/imgtrans/monitoring/drift_report.py](src/imgtrans/monitoring/drift_report.py); [docs/continual_learning_monitoring.md](docs/continual_learning_monitoring.md) |
| **Privacy + robustness** | [docs/privacy_robustness.md](docs/privacy_robustness.md) |
| **Project plan** | [docs/project_plan.md](docs/project_plan.md) |
| **Ethics** | [docs/ethics_statement.md](docs/ethics_statement.md) |
| **Report + slides (auto-generated)** | [src/imgtrans/autoreport/](src/imgtrans/autoreport/) → `report.pdf` + `slides.pptx` |
| **Reproducible training** | H100 notebook [notebooks/ImgTrans_Colab_Training_H100_AUTOPILOT.ipynb](notebooks/ImgTrans_Colab_Training_H100_AUTOPILOT.ipynb) + [COLAB_GUIDE.md](notebooks/COLAB_GUIDE.md) |

---

## Repository layout

```
15_Document_Image_Translation/
├── src/imgtrans/
│   ├── config.py  cli.py  logging_utils.py
│   ├── data/        # samples (seed pages + dict), synth_render (generator), dataset, download
│   ├── models/      # ocr_engine (Tesseract/Seed/Stub), baseline, model_registry
│   ├── mt/          # translator (m2m100 + dictionary, reverse for back-translation)
│   ├── imaging/     # preprocess, layout (blocks+reading order), render (fit-to-box overlay)
│   ├── training/    # train_mt, train_baseline, evaluate, tune, metrics (chrF/BLEU/CER/WER)
│   ├── agent/       # state, policy (D1-D5), tools, llm_orchestrator, imgtrans_agent
│   ├── api/         # schemas, dependencies, main (FastAPI), ui (Gradio), app_combined
│   ├── analysis/    # error_analysis, latency, layout_fidelity
│   ├── autoreport/  # artifact_loader, charts, report_pdf, slides_pptx
│   ├── monitoring/  # drift_report (job-log monitor)
│   ├── automation/  # autopilot (one button)
│   └── grading/     # checklist (rubric self-check)
├── configs/         # train.yaml, infer.yaml
├── docs/            # 14 markdown docs + DESIGN_BRIEF.md
├── notebooks/       # H100 AUTOPILOT .ipynb + COLAB_GUIDE.md
├── app/  deploy/  sample_data/  scripts/  tests/
├── Dockerfile  docker-compose.yml  Makefile
├── pyproject.toml  requirements.txt  requirements_colab.txt
└── .github/workflows/ci.yml  LICENSE (MIT)  README.md
```

---

## Models & data (all verified on the HF Hub)

| Slot | Default (shipped) | License | Alternatives |
|---|---|---|---|
| **MT core (trained)** | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) | **MIT** | `Helsinki-NLP/opus-mt-en-fr` (Apache, T4); `facebook/mbart-large-50-many-to-many-mmt` (MIT, H100); `facebook/nllb-200-distilled-600M` (**CC-BY-NC — flagged**) |
| **OCR front-end** | Tesseract via `pytesseract` | **Apache-2.0** | docTR / PaddleOCR / EasyOCR (Apache); `microsoft/trocr-base-printed` (MIT); Surya (**CC-BY-NC-SA — flagged**) |
| **OCR-VLM (documented)** | — | — | `stepfun-ai/GOT-OCR-2.0-hf` (Apache), `google/pix2struct-base` (Apache) |
| **MT fine-tune data** | [`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100) `en-fr` | license unknown → flag | — |
| **OCR-noise data** | [`PleIAs/Post-OCR-Correction`](https://huggingface.co/datasets/PleIAs/Post-OCR-Correction) | **CC0** | — |
| **Primary data** | **synthetic generator** (`data/synth_render.py`) | — | no real in-image-translation benchmark exists |

**Why synthetic?** No public in-image / document-image translation benchmark with gold parallel
text exists, so the primary data renders source sentences onto pages and embeds the gold
`(source, translation, boxes)` spec — letting us measure OCR CER/WER, MT chrF/BLEU **and**
end-to-end chrF on the same images. The offline `SeedEngine` reads that spec so the whole pipeline
runs with **no Tesseract, no torch, no network**.

---

## Quickstart

```bash
pip install -e .                 # core: runs offline (dictionary MT + SeedEngine OCR + fit-estimate)
pip install -e .[all]            # + torch/transformers, Tesseract wrapper, FastAPI/Gradio, reportlab
# system OCR (for real images): apt-get install tesseract-ocr  (Windows: install the Tesseract binary)

imgtrans demo-agent --fast                         # run the 5-decision agent on the seed pages
imgtrans translate-text --file sample_data/sample_lines_en.txt --fast
imgtrans translate-image --image sample_data/sample_document_en.png --mode overlay --out out.png --fast
imgtrans evaluate --fast                           # MT chrF/BLEU + OCR CER/WER + end-to-end + fit-rate
imgtrans autopilot --no-train                      # report.pdf + slides.pptx + grade + bundle
bash scripts/smoke.sh                              # full offline smoke
```

`--fast` uses the dictionary baseline (no model download). Drop it to use the fine-tuned m2m100.

## Train on Colab (H100, auto-adapts A100/L4/T4)

Push this folder to GitHub (or upload to Drive), open
`notebooks/ImgTrans_Colab_Training_H100_AUTOPILOT.ipynb`, set the controls in cell 0, and
**Runtime → Run all**. It installs Tesseract + fonts, fine-tunes the MT core (resume-safe), runs
the full evaluation, and writes `report.pdf` + `slides.pptx` + the submission bundle to Drive. See
[notebooks/COLAB_GUIDE.md](notebooks/COLAB_GUIDE.md).

## The agent (the mandatory agentic component)

A deterministic FSM `ingest → ocr → translate → verify → render` with **five decision points**:

| # | Decision | Gates on | Branches |
|---|---|---|---|
| **D1** | input router + quality | input kind + blur/contrast | image / pdf / spec / text (low-quality → flag) |
| **D2** | born-digital vs scanned | PDF text layer | skip OCR (born-digital) / OCR (scanned) |
| **D3** | OCR-confidence gate | per-block conf | translate / **skip low-confidence** (no mistranslation) |
| **D4** | translation verify | round-trip back-translation chrF + length ratio | ok / re-translate / flag |
| **D5** | render-fit feasibility | overlay fit-rate | overlay / **side-by-side fallback** / needs_review |

An optional LLM brain (`anthropic`) is **off by default** — the agent runs fully on rules with zero
paid API calls. Every step is timed and traced; same input → identical output.

## Serving

```bash
imgtrans serve --ui          # FastAPI on :8000 + Gradio demo at /ui
# POST /translate-image  (upload image/PDF -> translated text + base64 overlay PNG)
# POST /translate-text   (JSON {text, mode})        GET /healthz  /version
docker compose up --build    # containerized (Tesseract + Noto/DejaVu fonts + libGL baked in)
```

## Verified offline results

On the synthetic seed pages (offline, dictionary MT + perfect-OCR `SeedEngine`):
**MT dictionary chrF 79.9 vs identity floor 22.4**, **OCR CER 0.0**, **end-to-end image-translation
chrF 76.4**, **overlay fit-rate 1.0**, **grade 0.97**, **all 5 decision points fire**, 22 tests pass.
The offline floor saturates because the seed pairs overlap the dictionary; on real OPUS-100 eval
pairs the fine-tuned m2m100 dominates — the honest, non-saturated comparison happens on Colab.

## Tests

```bash
pytest -q        # CPU-only, no downloads (HF_HUB_OFFLINE); graceful fallbacks everywhere
```

## Documentation

[Problem](docs/problem_definition.md) · [Data](docs/data_description.md) · [Data card](docs/data_card.md) ·
[Models](docs/model_selection.md) · [Architecture](docs/architecture.md) · [Agent](docs/agent_architecture.md) ·
[Evaluation](docs/layout_fidelity_evaluation.md) · [Deployment](docs/deployment.md) ·
[Continual learning & monitoring](docs/continual_learning_monitoring.md) ·
[Privacy & robustness](docs/privacy_robustness.md) · [Ethics](docs/ethics_statement.md) ·
[Project plan](docs/project_plan.md) · [Model card](docs/model_card.md) · [Slides outline](docs/slide_deck_outline.md) ·
[Design brief](docs/DESIGN_BRIEF.md)

## Ethics & license

Document images are **highly sensitive PII** (IDs, contracts, medical/financial records): the
default path processes images transiently, logs metadata only, and the LLM brain is off. The tool
**assists** translation and **flags** low-confidence output for human review — it never asserts
certainty on high-stakes documents. Code is **MIT** ([LICENSE](LICENSE)); the shipped model stack
is permissive (m2m100 MIT + Tesseract Apache); non-commercial options (NLLB, Surya) are flagged and
not shipped.
