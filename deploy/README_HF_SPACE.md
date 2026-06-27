# Deploying imgtrans as a Hugging Face Space

Two options: a **Gradio** Space (simplest) or a **Docker** Space (full API + UI + Tesseract).

## Option A — Gradio Space (recommended for a demo)

1. Create a new Space → SDK: **Gradio**.
2. Add these files to the Space repo:
   - `app.py` → copy `app/gradio_app.py` (or `from imgtrans.api.ui import build_demo; build_demo().launch()`).
   - the `src/imgtrans/` package (or `pip install` it from your GitHub repo in `requirements.txt`).
   - `requirements.txt` → at least: `transformers datasets accelerate sentencepiece sacrebleu pytesseract PyMuPDF Pillow gradio reportlab python-pptx matplotlib`.
   - a `packages.txt` with the system OCR + fonts:
     ```
     tesseract-ocr
     tesseract-ocr-fra
     fonts-dejavu-core
     fonts-noto-core
     ```
3. The first request lazily downloads `facebook/m2m100_418M` (MIT) into the Space cache.
   If no GPU is attached the model still runs on CPU (slower); with no torch at all the
   Space falls back to the dictionary MT so the demo never hard-fails.

## Option B — Docker Space (full FastAPI + Gradio)

1. Create a new Space → SDK: **Docker**.
2. Add the repo's `Dockerfile` (it already installs `tesseract-ocr`, the `fra` language
   pack, Noto/DejaVu fonts and `libGL`).
3. The container serves `imgtrans.api.app_combined:app` on port 8000:
   - REST: `POST /translate-image` (upload an image/PDF → translated text + base64 overlay PNG),
     `POST /translate-text`, `GET /healthz`, `GET /version`.
   - UI: the Gradio demo is mounted at **`/ui`**.

## Notes
- Set `IMGTRANS_INFER_CONFIG=/app/configs/infer.yaml` to change direction / model / thresholds.
- The optional LLM brain is OFF by default; set `IMGTRANS_LLM_API_KEY` + `agent.llm_fallback_enabled: true` to enable advisory notes.
- Document images can contain PII — the Space processes uploads transiently and logs metadata only.
