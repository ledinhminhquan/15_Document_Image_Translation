# Sample data

- **`sample_document_en.png`** — a synthetic English document page (1000×420) produced by
  `imgtrans.data.synth_render`. It embeds the gold layout spec (source text + gold French
  translation + boxes) in the PNG metadata, so the offline `SeedEngine` can "read" it without
  Tesseract. Use it to try the pipeline:

  ```bash
  # text + overlay (writes the translated overlay PNG)
  imgtrans translate-image --image sample_data/sample_document_en.png --mode overlay --out /tmp/translated.png --fast

  # quick text-only translation of raw lines (no OCR)
  imgtrans translate-text --file sample_data/sample_lines_en.txt --fast
  ```

- **`sample_lines_en.txt`** — a few English lines for the `translate-text` demo (one block per line).

> `--fast` uses the dictionary MT baseline (no model download). Drop it to use the fine-tuned
> `facebook/m2m100_418M` (downloads on first use). On a real photo/scan install Tesseract
> (`apt-get install tesseract-ocr`) so the OCR front-end reads real pixels.
