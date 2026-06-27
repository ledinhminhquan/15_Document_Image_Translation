# Colab Guide — training imgtrans on an H100 (auto-adapts A100/L4/T4)

Run `ImgTrans_Colab_Training_H100_AUTOPILOT.ipynb`: upload the repo, set a few controls, **Run all**.

## 0. What gets trained
- **Only the MT core** is fine-tuned — HF `Seq2SeqTrainer` on OPUS-100 sentence pairs, selected
  on **chrF**, resume-safe.
- **The OCR front-end** (Tesseract), **layout analysis** and the **overlay renderer** are
  pretrained/algorithmic, not trained. The notebook installs Tesseract + fonts (cell 2), renders
  a synthetic eval set, and reports MT chrF/BLEU + OCR CER/WER + end-to-end image-translation
  chrF + overlay fit-rate.

## 1. Put the repo where Colab can see it (pick ONE)
- **GitHub (recommended):** push this folder to `https://github.com/<you>/imgtrans`, set
  `GIT_REPO_URL` in cell 0.
- **Drive:** upload `15_Document_Image_Translation/` to `MyDrive/imgtrans/imgtrans` (repo root
  = `.../imgtrans/imgtrans`); leave `GIT_REPO_URL` as the placeholder.

```
MyDrive/imgtrans/
├── imgtrans/       <- the repo, if using Drive
└── artifacts/      <- created automatically; the MT model + reports persist here
```

## 2. Runtime
`Runtime -> Change runtime type -> GPU`. H100 ideal but optional — cell 7 auto-profiles
batch/precision for **H100/A100/L4/T4** (T4 has no bf16 -> fp16).

## 3. Controls (cell 0)
- `MT_BASE` — the trainable MT core (`facebook/m2m100_418M` MIT default; `opus-mt-en-fr` Apache,
  cheapest, en->fr only; `mbart-large-50` MIT, H100; `nllb-200-distilled-600M` stronger but
  **CC-BY-NC**).
- `SRC_LANG`/`TGT_LANG`, `MT_CONFIG` (e.g. `en-fr`) — the direction.
- `TESS_LANGS` — Tesseract language packs to install (e.g. `eng fra`); match the source script.
- `MAX_TRAIN_SAMPLES`, `EPOCHS` — training budget.

## 4. Run all
The **autopilot** (cell 10) does everything: baseline -> fine-tune MT -> evaluate (MT + OCR +
end-to-end + fidelity) -> analysis -> **report.pdf + slides.pptx + grading + submission_bundle.zip**.
Resume-safe: re-run cell 10 after a disconnect.

## 5. Read the results (cell 12)
Look for the fine-tuned MT core's **chrF** beating the dictionary / identity baselines, a low
**OCR CER**, a high **end-to-end image-translation chrF**, and an overlay **fit-rate** near 1.0.

## 6. Test the trained model (cell 13)
Cell 13 translates `sample_data/sample_document_en.png` and shows the **source vs the translated
overlay** side by side — the translated text should sit in the original boxes in the target
language.

## 7. Deliverables (cell 14)
`report.pdf`, `slides.pptx`, `submission_bundle.zip` under
`artifacts/submission/submission-<stamp>/` (on Drive).

## Troubleshooting
- **"Set GIT_REPO_URL..."** — neither a repo URL nor a Drive copy was found; do step 1.
- **`TesseractNotFoundError`** — cell 2 didn't finish; re-run it (`apt-get install tesseract-ocr`).
- **Wrong-script OCR** — add the language pack to `TESS_LANGS` (e.g. `eng fra deu`) and set
  `ocr.lang` accordingly.
- **bf16 error on T4** — Turing has no bf16; cell 7 falls back to fp16.
- **OOM** — lower `MAX_TRAIN_SAMPLES`, pick `m2m100_418M` / `opus-mt-en-fr`, or reduce batch.
- **License** — the redistributable stack is m2m100 (MIT) + Tesseract (Apache); opus-100 is
  license-unknown (training only); NLLB / Surya are CC-BY-NC (flagged, not shipped).
