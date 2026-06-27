# models/

Trained-model artifacts land here at runtime and are **not committed** (see `.gitignore`).

- `imgtrans train-mt` writes the fine-tuned MT core to `$IMGTRANS_MODEL_DIR/mt/<version>/`
  (a `latest` pointer + `model_meta.json` track the active version).
- `imgtrans train-baseline` writes `dictionary_mt.json` (the offline dictionary baseline).

The default base model `facebook/m2m100_418M` (MIT) is downloaded from the Hugging Face Hub on
first use into `HF_HOME` — it is not stored here. The OCR engine (Tesseract) is a system binary,
not a model file.
