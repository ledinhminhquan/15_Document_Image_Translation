# P15 Deployment — Document-Image Machine Translation

This document describes how to serve **P15 Document-Image Machine Translation** (`imgtrans`) in production: the FastAPI service and its endpoints, the Gradio UI, the Docker image and its system dependencies, the Hugging Face Space, scaling and latency behaviour, GPU-vs-CPU placement of each cascade stage, environment-variable configuration, and a worked request/response example.

P15 is a **cascade** — OCR front-end (pretrained Tesseract) → MT core (the only trained stage, `facebook/m2m100_418M`, MIT) → layout-preserving overlay render (algorithmic Pillow). The deployment surface reflects that cascade: a JSON text-only path that skips OCR and render, and an image/PDF path that runs the full pipeline behind the deterministic five-decision agent and returns both the translated text and a base64 overlay PNG.

Author: Le Dinh Minh Quan (student 23127460). Default direction `en→fr` (configurable; the m2m100 core is many-to-many so the same checkpoint serves other pairs from one set of weights).

---

## 1. Service overview

| Component | Path / port | Purpose |
|-----------|-------------|---------|
| FastAPI app | `src/imgtrans/serve/app.py`, port `8000` | REST API: health, text translation, image/PDF translation with overlay. |
| Gradio UI | mounted at `/ui` | Browser demo: upload an image/PDF, see the overlay PNG and per-block decisions. |
| Docker image | `Dockerfile` | Reproducible runtime with `tesseract-ocr` + `libGL` system deps. |
| HF Space | `spaces/imgtrans` | Public Gradio demo (SeedEngine + dictionary MT offline floor by default). |

The service is built so that **it always starts** — every cascade stage degrades to a deterministic pure-Python stub (SeedEngine offline OCR, dictionary MT, DejaVuSans font) when its real backend (Tesseract binary, torch + m2m100, Noto fonts) is absent. Capability probes (`shutil.which('tesseract')`, `try import torch`) pick real-vs-stub at runtime; `P15_OFFLINE=1` pins stub mode. This means the container boots and `/healthz` returns ready even with no model weights downloaded — important for cold HF Spaces and CI smoke tests.

---

## 2. FastAPI endpoints

The API is a thin transport layer over `imgtrans.agent` (the five-decision FSM). The agent does the routing; the endpoints marshal bytes in and JSON/base64 out.

### 2.1 `GET /healthz`

Liveness/readiness probe. Returns the resolved runtime configuration so an operator can confirm which backends are live without reading logs.

```http
GET /healthz
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "mode": "online",
  "ocr": {"engine": "tesseract", "version": "5.3.4", "available": true},
  "mt": {"model": "facebook/m2m100_418M", "device": "cuda:0", "fine_tuned": true},
  "render": {"fonts": ["NotoSans", "NotoSansCJK", "DejaVuSans"], "pillow": "12.2.0"},
  "default_direction": {"src": "en", "tgt": "fr"},
  "offline": false
}
```

`status` is `"ok"` whenever the app can serve a request — including offline stub mode (`mode: "offline"`, `ocr.engine: "seedengine"`, `mt.model: "dictionary"`). Use this as the Kubernetes/Space readiness probe. A `503` is returned only if model warm-up was requested (`P15_EAGER_LOAD=1`) and failed.

### 2.2 `POST /translate-text`

Text-in, text-out. **Skips OCR and render entirely** — this is the degenerate `D1 → D4` path of the agent (input router classifies `text/plain` and jumps straight to translation + verification). This is the cheapest, GPU-bound-only endpoint and the right one for callers that already have clean text.

Request:

```json
{
  "text": "The quick brown fox jumps over the lazy dog.",
  "src_lang": "en",
  "tgt_lang": "fr",
  "verify": true
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `text` | string (required) | — | Source text. Multi-line allowed; blocks are translated independently. |
| `src_lang` | string | `P15_SRC_LANG` (`en`) | m2m100 BCP-47/ISO code. |
| `tgt_lang` | string | `P15_TGT_LANG` (`fr`) | Any pair the many-to-many checkpoint supports. |
| `verify` | bool | `true` | Run D4 round-trip back-translation chrF + length-ratio gate. |

Response:

```json
{
  "src_lang": "en",
  "tgt_lang": "fr",
  "translation": "Le renard brun rapide saute par-dessus le chien paresseux.",
  "verification": {
    "round_trip_chrf": 84.1,
    "length_ratio": 1.18,
    "status": "accept"
  }
}
```

`verification.status` is one of `accept`, `low_confidence` (D4 round-trip below `P15_VERIFY_TAU`, or length-ratio outside `[0.4, 3.0]` — the translation is still returned, never silently dropped), or `retried` (re-decoded once with more beams before accepting). The gate is **soft**: it annotates, it does not censor.

### 2.3 `POST /translate-image`

The headline endpoint: upload an image or PDF, get back the translated text **and** a layout-preserving overlay PNG. This runs the **full cascade** through all five agent decision points.

This route is **gated on `python-multipart`** — `multipart/form-data` upload parsing is a hard dependency for file uploads in Starlette/FastAPI. If `python-multipart` is not installed the route is **not registered** at startup and `/healthz` reports `"upload": false`; the JSON `/translate-text` path still works. This keeps a minimal text-only deployment dependency-light while the full image deployment opts into the upload stack. The Docker image installs `python-multipart`, so the upload route is always present in the container.

Request (`multipart/form-data`):

```http
POST /translate-image
Content-Type: multipart/form-data

file=@invoice.png
src_lang=en
tgt_lang=fr
mode=overlay
return_image=true
```

| Form field | Type | Default | Notes |
|------------|------|---------|-------|
| `file` | upload (required) | — | `png`/`jpg`/`webp`/`pdf`/`txt`. Sniffed by magic bytes + MIME (D1). |
| `src_lang` | string | `en` | |
| `tgt_lang` | string | `fr` | |
| `mode` | enum | `auto` | `auto` lets D5 choose `overlay`/`side_by_side`; `overlay` and `side_by_side` force a strategy; `text_only` returns blocks without rendering. |
| `return_image` | bool | `true` | When `false`, returns only text + boxes + decisions (no base64 PNG) — lighter payload for headless callers. |
| `dpi` | int | `200` | PDF rasterization DPI for the scanned branch. |

Response:

```json
{
  "input_kind": "image",
  "src_lang": "en",
  "tgt_lang": "fr",
  "render_mode": "overlay",
  "blocks": [
    {
      "box": [42, 60, 520, 96],
      "source": "The quick brown fox jumps over the lazy dog.",
      "translation": "Le renard brun rapide saute par-dessus le chien paresseux.",
      "ocr_conf": 91.3,
      "font_size": 28,
      "fit_ok": true,
      "decision": "overlay"
    }
  ],
  "layout": {"fit_rate": 1.0, "mean_shrink": 1.0, "overflow": 0},
  "verification": {"round_trip_chrf": 83.7, "length_ratio": 1.18, "status": "accept"},
  "overlay_png_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "needs_review": false
}
```

`overlay_png_base64` is a base64-encoded PNG the **same pixel size as the input** (overlay mode) or a side-by-side panel (degrade mode). Decode and `data:image/png;base64,...` it directly into an `<img>` tag. When `return_image=false` the field is omitted.

The per-block `decision` and the top-level `render_mode` / `needs_review` come straight from the agent decision points:

- **D2** (PDF only) — born-digital PDFs bypass OCR; `blocks[].ocr_conf` is reported as `100.0` and `input_kind` is `pdf_digital`.
- **D3** — blocks with `ocr_conf < P15_OCR_LOW` are dropped (not translated) and surfaced under a `skipped_blocks` array with `reason: "low_ocr_conf"`.
- **D4** — populates `verification`.
- **D5** — drives `render_mode`: `overlay` when `layout.fit_rate` is high, `side_by_side` when translations overflow boxes (the common `→fr` / CJK→EN expansion case), `needs_review` when fit fails or blocks were D3/D4-flagged (then `needs_review: true` and no destructive render is performed).

### 2.4 Error contract

| Status | Condition |
|--------|-----------|
| `200` | Success — including `needs_review: true` (a defensible non-destructive result, not an error). |
| `400` | Unsupported file type (D1 router → `unsupported`), empty upload, or malformed JSON. |
| `413` | Upload exceeds `P15_MAX_UPLOAD_MB`. |
| `415` | `/translate-image` called without `multipart/form-data`, or upload route disabled (no `python-multipart`). |
| `422` | FastAPI/Pydantic validation (bad `src_lang`, etc.). |
| `503` | Eager model load configured and failed. |

---

## 3. Gradio UI (`/ui`)

A Gradio `Blocks` app is mounted onto the FastAPI server at `/ui` via `gr.mount_gradio_app(fastapi_app, demo, path="/ui")`, so a single process serves both the REST API and the demo (one port, one container).

The UI exposes:

- **Upload** an image or PDF (drag-drop), plus source/target language dropdowns (defaulting to en/fr) and a `mode` selector (`auto` / `overlay` / `side_by_side`).
- **Output panel:** the overlay PNG rendered inline, a table of per-block `(source → translation, ocr_conf, font_size, fit_ok, decision)`, and the aggregate `fit_rate` / `mean_shrink` / `overflow` layout metrics.
- **Decision badges:** each block is tagged `overlay` / `side_by_side` / `needs_review` so the user sees *why* the agent chose a strategy — this exposes the agentic value-add (low-confidence OCR skipped, poor-fit downgraded to side-by-side) rather than hiding it.
- **Privacy banner:** document images can contain PII (IDs, passports, medical/legal scans). The UI states that processing is local, raw images are not retained by default, and that the tool *assists* translation and flags low-confidence output for human review — it never asserts certainty.

The UI calls the same `imgtrans.agent` entrypoint as the API (no logic duplication), so what you see in the demo is exactly what `/translate-image` returns.

---

## 4. Docker

The container must ship the **Tesseract binary** (Python `pytesseract` is only a wrapper around it) and **libGL** (pulled in transitively by image/PDF stacks; required so PDF rasterization and imaging deps load on a headless slim base). These are system packages, not pip installs — the two most common deployment failures for this project are a missing `tesseract-ocr` binary (OCR silently falls back to SeedEngine stub) and a missing `libGL.so.1` (import error at boot).

```dockerfile
FROM python:3.11-slim

# System deps: Tesseract OCR engine (+ French/multilingual traineddata),
# libGL for the imaging/PDF stack, and poppler/mupdf runtime for PDF raster.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
        libgl1 \
        libglib2.0-0 \
        fonts-noto-core \
        fonts-noto-cjk \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# python-multipart here enables the /translate-image upload route.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV P15_SRC_LANG=en \
    P15_TGT_LANG=fr \
    P15_DEVICE=auto \
    P15_MAX_UPLOAD_MB=25 \
    TESSERACT_CMD=/usr/bin/tesseract \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "imgtrans.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes:

- `tesseract-ocr-fra` installs the French `traineddata`. Add `tesseract-ocr-<lang>` packages for any other OCR source language you need (Tesseract recognizes the language *on the page*, which for in-image MT is the **source** side).
- `fonts-noto-cjk` is large (~hundreds of MB). Omit it for a Latin-only (en→fr) deployment to keep the image small; the render engine falls back to DejaVuSans and only CJK targets would render tofu.
- For a GPU image, base on `nvidia/cuda:12.x-runtime` instead of `python:3.11-slim`, install the same apt packages, and `pip install` the CUDA torch wheel. OCR still runs on CPU (see §6).
- `--no-install-recommends` + `rm -rf /var/lib/apt/lists/*` keep the layer lean.

Run:

```bash
docker build -t imgtrans:latest .
docker run --rm -p 8000:8000 \
  -e P15_TGT_LANG=fr \
  -e P15_DEVICE=cpu \
  imgtrans:latest
# GPU:
docker run --rm --gpus all -p 8000:8000 -e P15_DEVICE=cuda imgtrans:latest
```

---

## 5. Hugging Face Space

The public demo is a **Gradio Space**. Spaces does not let you `apt-get` arbitrarily in the simple SDK config, so the Space ships either:

1. **Docker Space** (recommended for full fidelity) — reuses the `Dockerfile` above so `tesseract-ocr` is present and the real OCR path runs on the free CPU hardware; or
2. **Gradio SDK Space** with a `packages.txt` listing the apt deps:

```
tesseract-ocr
tesseract-ocr-fra
libgl1
fonts-noto-core
fonts-dejavu-core
```

and `requirements.txt` for the Python deps.

Space defaults and behaviour:

- Free CPU Spaces have no GPU, so MT runs the m2m100 checkpoint on **CPU** (slow but correct) or, to keep the demo snappy and downloads small, the Space defaults to the **offline floor** (`P15_OFFLINE=1`: SeedEngine OCR + dictionary MT + DejaVuSans). The dictionary-MT offline floor (seed chrF 79.9 vs identity floor 22.4) is enough to demonstrate the full overlay pipeline without a multi-GB model download on cold start.
- Set `P15_OFFLINE=0` and select a GPU Space to serve the fine-tuned m2m100 checkpoint for real translation quality.
- The MT checkpoint and any Noto fonts are pulled from the Hub on first request and cached under `HF_HOME` (set a persistent path on paid Spaces so cold starts stay warm).
- **Non-commercial models are not shipped on the Space.** `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0) and Surya (CC-BY-NC-SA-4.0) are documented as research-only upgrades and must never be the Space default. The shipped stack is entirely MIT/Apache-2.0.

---

## 6. GPU vs CPU serving

The cascade has a clean device split, and the deployment exploits it:

| Stage | Device | Why |
|-------|--------|-----|
| **OCR (Tesseract)** | **CPU** | Tesseract is a CPU C++ engine; it does not use the GPU. Pinning it to CPU frees GPU memory entirely for MT. The born-digital PDF branch (D2) does no OCR at all — pure PyMuPDF text extraction on CPU. |
| **MT (m2m100_418M)** | **GPU** (`cuda`), CPU fallback | The only neural, GPU-benefiting stage. fp16 on GPU is the throughput path; on CPU it still runs (slower), which is the free-Space default. |
| **Render (Pillow)** | **CPU** | Pure-Python/Pillow geometry (binary-search font-fit + greedy wrap + whiteout). Negligible cost; no GPU. |

`P15_DEVICE=auto` resolves to `cuda` when `torch.cuda.is_available()` else `cpu`. Because OCR and render are CPU-only regardless, a single mid-tier GPU (A10/L4) hosts the MT model while OCR/render run on the box's CPU cores in parallel — OCR of page *N+1* can overlap MT decode of page *N*.

Tier guidance (from the design brief):

- **T4 / free tier:** m2m100_418M fp16 (fits ~16 GB), or `opus-mt-en-fr` (Apache-2.0, near-CPU speed) for an en→fr-only demo if you must avoid GPU.
- **Default (A10/L4):** Tesseract (CPU) + fine-tuned m2m100_418M (GPU) + algorithmic overlay (CPU).
- **H100 upgrade:** swap MT to `mbart-large-50-many-to-many-mmt` (MIT) for higher quality; OCR recognizer can move to `trocr-large-printed` / `GOT-OCR-2.0-hf` for hard photos. All permissive.

---

## 7. Scaling and latency

**Latency budget** for `/translate-image` is dominated, in order, by **MT decode** and **OCR**; render is sub-10ms per block.

| Path | Typical bottleneck | Notes |
|------|--------------------|-------|
| `/translate-text` | MT decode only | No OCR, no render. ~tens of ms/sentence on GPU; hundreds on CPU. |
| `/translate-image` (born-digital PDF, D2) | MT decode | OCR fully bypassed → fastest image path; zero OCR error. |
| `/translate-image` (scanned/photo) | Tesseract OCR (CPU) + MT decode | OCR scales with page size/DPI; cap `dpi` at 200–300. |
| `/translate-image` with `verify=true` | MT decode **×2** (round-trip back-translation) | D4 doubles decode cost. Disable for throughput-critical batch jobs. |

**Scaling strategy:**

- **Stateless workers.** The service holds no per-request state (raw images are not retained by default — see ethics). Scale horizontally with multiple uvicorn workers / replicas behind a load balancer. Model weights load once per worker.
- **Batching.** Within one image, all blocks above the D3 confidence gate are translated in a single batched MT call (pad to the longest block) rather than one call per block — the largest single throughput win on multi-block pages.
- **Worker count.** With a single GPU, run **one** GPU worker (the model is loaded once, GPU is the serializing resource) and rely on intra-image batching; scale OCR/render parallelism via the GPU worker's CPU threads. For CPU-only deployments, run `min(cores, N)` workers.
- **Upload limit.** `P15_MAX_UPLOAD_MB` (default 25) rejects oversized scans early with `413` to protect OCR latency and memory.
- **Timeouts.** Set the front proxy read timeout above the worst-case multi-page PDF (OCR + MT scale with page count); paginate or async-queue very large PDFs rather than translating 100 pages in one synchronous request.
- **Warm-up.** Set `P15_EAGER_LOAD=1` to load the MT model at startup (slower boot, no first-request latency spike). Leave unset for fast cold starts (lazy load on first translate call), which is preferable on autoscaling Spaces.
- **Caching.** `HF_HOME` on a persistent volume avoids re-downloading the checkpoint and Noto fonts on every cold start.

---

## 8. Configuration (environment variables)

All runtime configuration is via env vars (12-factor), readable in `/healthz`. No secrets are required for the offline floor; `ANTHROPIC_API_KEY` is needed **only** if the optional advisory LLM brain is enabled.

| Variable | Default | Purpose |
|----------|---------|---------|
| `P15_SRC_LANG` | `en` | Default source language. |
| `P15_TGT_LANG` | `fr` | Default target language (m2m100 is many-to-many). |
| `P15_DEVICE` | `auto` | `auto` / `cuda` / `cpu` — MT placement. OCR/render stay CPU regardless. |
| `P15_MT_MODEL` | `facebook/m2m100_418M` | MT checkpoint or local path to the fine-tuned dir. Do **not** set to a non-commercial id (`nllb-200-distilled-600M`) in production. |
| `P15_OFFLINE` | `0` | `1` pins SeedEngine OCR + dictionary MT + DejaVuSans (no downloads). |
| `P15_EAGER_LOAD` | `0` | `1` loads MT at startup instead of first request. |
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to the Tesseract binary for `pytesseract`. |
| `P15_OCR_HIGH` | `75` | D3 high-confidence threshold (accept block). |
| `P15_OCR_LOW` | `40` | D3 low-confidence threshold (drop block → `needs_review`). |
| `P15_VERIFY` | `1` | Enable D4 round-trip verification by default. |
| `P15_VERIFY_TAU` | `0.5` | D4 round-trip chrF acceptance threshold (normalized). |
| `P15_FIT_MIN_FONT` | `6` | Minimum legible font size for D5 fit feasibility. |
| `P15_MAX_UPLOAD_MB` | `25` | Reject uploads larger than this with `413`. |
| `P15_PDF_DPI` | `200` | Default rasterization DPI for scanned PDFs. |
| `P15_LLM_BRAIN` | `0` | Enable the optional advisory LLM (anthropic). OFF by default; advisory only, never rewrites output. |
| `ANTHROPIC_API_KEY` | — | Only required when `P15_LLM_BRAIN=1`. |
| `HF_HOME` | `~/.cache/huggingface` | Model/font cache dir; point at a persistent volume in production. |
| `PORT` | `8000` | Server port. |

---

## 9. Worked example

Translate a scanned English image to French with the overlay, via `curl`:

```bash
curl -s -X POST http://localhost:8000/translate-image \
  -F "file=@/path/to/notice_en.png" \
  -F "src_lang=en" \
  -F "tgt_lang=fr" \
  -F "mode=auto" \
  -F "return_image=true" \
  -o response.json
```

Response (`response.json`, truncated base64):

```json
{
  "input_kind": "image",
  "src_lang": "en",
  "tgt_lang": "fr",
  "render_mode": "overlay",
  "blocks": [
    {
      "box": [42, 60, 520, 96],
      "source": "The quick brown fox jumps over the lazy dog.",
      "translation": "Le renard brun rapide saute par-dessus le chien paresseux.",
      "ocr_conf": 91.3,
      "font_size": 28,
      "fit_ok": true,
      "decision": "overlay"
    }
  ],
  "skipped_blocks": [],
  "layout": {"fit_rate": 1.0, "mean_shrink": 1.0, "overflow": 0},
  "verification": {"round_trip_chrf": 83.7, "length_ratio": 1.18, "status": "accept"},
  "overlay_png_base64": "iVBORw0KGgoAAAANSUhEUgAAB...<truncated>...g==",
  "needs_review": false
}
```

Save the overlay PNG from the response:

```bash
python -c "import json,base64; d=json.load(open('response.json')); open('overlay_fr.png','wb').write(base64.b64decode(d['overlay_png_base64']))"
```

Text-only call (skips OCR + render, GPU-MT only):

```bash
curl -s -X POST http://localhost:8000/translate-text \
  -H "Content-Type: application/json" \
  -d '{"text":"The quick brown fox jumps over the lazy dog.","src_lang":"en","tgt_lang":"fr"}'
```

```json
{
  "src_lang": "en",
  "tgt_lang": "fr",
  "translation": "Le renard brun rapide saute par-dessus le chien paresseux.",
  "verification": {"round_trip_chrf": 84.1, "length_ratio": 1.18, "status": "accept"}
}
```

---

## 10. Deployment checklist

- [ ] `tesseract-ocr` binary on `PATH` (or `TESSERACT_CMD` set) — else OCR silently degrades to SeedEngine stub.
- [ ] `tesseract-ocr-<src_lang>` traineddata installed for each OCR source language.
- [ ] `libgl1` present on slim/headless bases — else import error at boot.
- [ ] `python-multipart` installed — else `/translate-image` upload route is disabled.
- [ ] Noto fonts (esp. CJK) installed if any non-Latin target is served; DejaVuSans is the always-present fallback.
- [ ] `P15_DEVICE=cuda` and a CUDA torch wheel for GPU MT; confirm via `/healthz` (`mt.device`).
- [ ] `P15_MT_MODEL` points at the fine-tuned checkpoint, **not** a non-commercial id.
- [ ] `HF_HOME` on a persistent volume for warm cold starts.
- [ ] `/healthz` wired as the readiness probe.
- [ ] PII handling reviewed: no raw-image retention by default; low-confidence output flagged for human review.
