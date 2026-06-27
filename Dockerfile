# Document-Image Machine Translation — serving image.
# Includes Tesseract (OCR), Noto/DejaVu fonts (overlay rendering) and libGL (image stack).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    IMGTRANS_ARTIFACTS_DIR=/app/artifacts

# System deps: tesseract OCR + a Latin-capable language pack, fonts for the overlay
# renderer, and libGL/libglib for Pillow/PyMuPDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra \
        fonts-dejavu-core fonts-noto-core \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

EXPOSE 8000
# Serve the FastAPI app with the Gradio UI mounted at /ui.
CMD ["uvicorn", "imgtrans.api.app_combined:app", "--host", "0.0.0.0", "--port", "8000"]
