"""Generate the H100 AUTOPILOT Colab notebook as valid .ipynb JSON.

Run:  python notebooks/_build_notebook.py
Produces: notebooks/ImgTrans_Colab_Training_H100_AUTOPILOT.ipynb

Mirrors the resume-safe, GPU-auto-profiling, Colab-safe-install pattern proven in P02-P14,
plus an apt install of Tesseract + fonts (this project's OCR front-end + overlay renderer).
"""

from __future__ import annotations

import json
from pathlib import Path

NB = "ImgTrans_Colab_Training_H100_AUTOPILOT.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(lines)}


cells = []

cells.append(md(
    "# Document-Image Machine Translation - Colab Training (H100 AUTOPILOT, resume-safe)\n",
    "\n",
    "Fine-tunes the **MT core** (`facebook/m2m100_418M`, HF Seq2SeqTrainer, chrF). The OCR front-end\n",
    "(Tesseract), layout analysis and the layout-preserving overlay renderer are pretrained/algorithmic\n",
    "- only the MT model is trained.\n",
    "\n",
    "**How to use:** set the controls in cell 0, then **Runtime -> Run all**. Resume-safe (re-run cell 10).\n",
    "Auto-adapts H100/A100/L4/T4. Default direction English -> French.\n",
    "\n",
    "> Low-confidence OCR is skipped; low-confidence / poor-fit output is flagged for human review.\n",
))

cells.append(code(
    "#@title 0) Controls - set these, then `Runtime -> Run all`  { display-mode: \"form\" }\n",
    "GIT_REPO_URL = \"https://github.com/<your-username>/imgtrans\"  #@param {type:\"string\"}\n",
    "GIT_BRANCH   = \"main\"  #@param {type:\"string\"}\n",
    "USE_DRIVE    = True     #@param {type:\"boolean\"}\n",
    "DRIVE_SUBDIR = \"imgtrans\"  #@param {type:\"string\"}\n",
    "\n",
    "MT_BASE      = \"facebook/m2m100_418M\"  #@param [\"facebook/m2m100_418M\", \"Helsinki-NLP/opus-mt-en-fr\", \"facebook/mbart-large-50-many-to-many-mmt\", \"facebook/nllb-200-distilled-600M\"]\n",
    "SRC_LANG     = \"en\"  #@param {type:\"string\"}\n",
    "TGT_LANG     = \"fr\"  #@param {type:\"string\"}\n",
    "MT_DATASET   = \"Helsinki-NLP/opus-100\"  #@param {type:\"string\"}\n",
    "MT_CONFIG    = \"en-fr\"  #@param {type:\"string\"}\n",
    "MAX_TRAIN_SAMPLES = 50000  #@param {type:\"integer\"}\n",
    "EPOCHS       = 3     #@param {type:\"integer\"}\n",
    "TESS_LANGS   = \"eng fra\"  #@param {type:\"string\"}\n",
    "RUN_AUTOPILOT = True  #@param {type:\"boolean\"}\n",
    "HF_TOKEN     = \"\"      #@param {type:\"string\"}\n",
    "print('Controls set. MT =', MT_BASE, '| direction =', SRC_LANG, '->', TGT_LANG)\n",
))

cells.append(code(
    "#@title 1) Check the GPU\n",
    "import subprocess\n",
    "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout or 'No GPU - Runtime>Change runtime type>GPU')\n",
))

cells.append(code(
    "#@title 2) System deps: Tesseract OCR + fonts (the OCR front-end + overlay renderer)\n",
    "import os, subprocess\n",
    "langs = ' '.join('tesseract-ocr-' + l for l in TESS_LANGS.split())\n",
    "os.system('apt-get -qq update')\n",
    "os.system(f'apt-get -qq install -y tesseract-ocr {langs} fonts-dejavu-core fonts-noto-core libgl1 >/dev/null 2>&1')\n",
    "out = subprocess.run(['tesseract', '--version'], capture_output=True, text=True).stdout\n",
    "print(out.splitlines()[0] if out else 'tesseract installed')\n",
))

cells.append(code(
    "#@title 3) Mount Drive + artifact paths & HF caches  (BEFORE importing torch)\n",
    "import os\n",
    "ART = '/content/artifacts'\n",
    "if USE_DRIVE:\n",
    "    try:\n",
    "        from google.colab import drive\n",
    "        drive.mount('/content/drive')\n",
    "        ART = f'/content/drive/MyDrive/{DRIVE_SUBDIR}/artifacts'\n",
    "    except Exception as e:\n",
    "        print('Drive mount skipped:', e)\n",
    "os.makedirs(ART, exist_ok=True)\n",
    "os.environ['IMGTRANS_ARTIFACTS_DIR'] = ART\n",
    "os.environ['HF_HOME'] = f'{ART}/hf_cache'\n",
    "os.makedirs(os.environ['HF_HOME'], exist_ok=True)\n",
    "if HF_TOKEN:\n",
    "    os.environ['HF_TOKEN'] = HF_TOKEN; os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN\n",
    "print('Artifacts ->', ART)\n",
))

cells.append(code(
    "#@title 4) Get the project source (git clone, or copy from Drive)\n",
    "import os\n",
    "os.chdir('/content')\n",
    "if os.path.isdir('/content/imgtrans'):\n",
    "    os.chdir('/content/imgtrans'); os.system('git pull')\n",
    "elif GIT_REPO_URL and '<your-username>' not in GIT_REPO_URL:\n",
    "    os.system(f'git clone -b {GIT_BRANCH} {GIT_REPO_URL} /content/imgtrans'); os.chdir('/content/imgtrans')\n",
    "else:\n",
    "    drive_src = f'/content/drive/MyDrive/{DRIVE_SUBDIR}/imgtrans'\n",
    "    if os.path.isdir(drive_src):\n",
    "        os.system(f'cp -r {drive_src} /content/imgtrans'); os.chdir('/content/imgtrans')\n",
    "    else:\n",
    "        raise SystemExit('Set GIT_REPO_URL to your repo, or upload the project to Drive at ' + drive_src)\n",
    "print('cwd =', os.getcwd()); print(sorted(os.listdir('.'))[:20])\n",
))

cells.append(code(
    "#@title 5) Install dependencies (Colab-safe: NEVER reinstall torch)\n",
    "!pip -q install -r requirements_colab.txt\n",
    "!pip -q install -e . --no-deps\n",
    "print('deps installed')\n",
))

cells.append(code(
    "#@title 6) Verify environment + performance knobs (TF32)\n",
    "import torch\n",
    "print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    torch.backends.cuda.matmul.allow_tf32 = True\n",
    "    torch.backends.cudnn.allow_tf32 = True\n",
    "    print('GPU:', torch.cuda.get_device_name(0))\n",
    "import imgtrans, transformers, datasets, pytesseract\n",
    "print('imgtrans', imgtrans.__version__, '| transformers', transformers.__version__,\n",
    "      '| tesseract', pytesseract.get_tesseract_version())\n",
))

cells.append(code(
    "#@title 7) Auto GPU profile (MT fine-tune batch + precision; eff. batch ~32)\n",
    "import torch\n",
    "name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'\n",
    "n = name.upper()\n",
    "if 'H100' in n:     BATCH = 32\n",
    "elif 'A100' in n:   BATCH = 16\n",
    "elif 'L4' in n:     BATCH = 8\n",
    "elif 'T4' in n:     BATCH = 4\n",
    "else:               BATCH = 4\n",
    "if any(k in MT_BASE.lower() for k in ('600m','nllb','mbart','large')): BATCH = max(2, BATCH // 2)\n",
    "BF16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False\n",
    "FP16 = (not BF16) and torch.cuda.is_available()\n",
    "TF32 = ('H100' in n or 'A100' in n or 'L4' in n)\n",
    "GRAD_ACCUM = max(1, 32 // max(1, BATCH))\n",
    "print(f'GPU={name} -> batch={BATCH} grad_accum={GRAD_ACCUM} precision={\"bf16\" if BF16 else (\"fp16\" if FP16 else \"fp32\")}')\n",
))

cells.append(code(
    "#@title 8) Write the Colab training config  (configs/train_colab.yaml)\n",
    "import yaml, os\n",
    "cfg = {\n",
    "  'project_title': 'Document-Image Machine Translation System', 'author': 'Le Dinh Minh Quan', 'student_id': '23127460',\n",
    "  'data': {'mt_dataset': MT_DATASET, 'mt_config': MT_CONFIG, 'src_lang': SRC_LANG, 'tgt_lang': TGT_LANG,\n",
    "           'use_hf': True, 'max_train_samples': int(MAX_TRAIN_SAMPLES), 'max_eval_samples': 2000,\n",
    "           'synth_eval_pages': 80, 'lines_per_page': 6, 'image_width': 1000, 'seed': 42},\n",
    "  'mt': {'base_model': MT_BASE, 'src_lang': SRC_LANG, 'tgt_lang': TGT_LANG, 'max_source_length': 200,\n",
    "         'max_target_length': 200, 'num_beams': 4, 'num_train_epochs': int(EPOCHS), 'learning_rate': 3.0e-5,\n",
    "         'per_device_train_batch_size': int(BATCH), 'gradient_accumulation_steps': int(GRAD_ACCUM),\n",
    "         'label_smoothing': 0.1, 'early_stopping_patience': 3,\n",
    "         'bf16': bool(BF16), 'fp16': bool(FP16), 'tf32': bool(TF32), 'eval_steps': 500, 'save_steps': 500},\n",
    "  'ocr': {'engine': 'auto', 'lang': SRC_LANG[:3] if SRC_LANG != 'en' else 'eng', 'dpi': 200, 'psm': 3},\n",
    "  'render': {'mode': 'overlay', 'max_font_size': 48, 'min_font_size': 8},\n",
    "  'agent': {'quality_min': 0.30, 'ocr_confidence_min': 0.50, 'verify_min_chrf': 0.30,\n",
    "            'max_retranslate': 2, 'min_fit_rate': 0.60},\n",
    "}\n",
    "os.makedirs('configs', exist_ok=True)\n",
    "yaml.safe_dump(cfg, open('configs/train_colab.yaml','w'), sort_keys=False)\n",
    "print(open('configs/train_colab.yaml').read())\n",
))

cells.append(code(
    "#@title 9) Render synthetic document-image eval pages + sanity-check datasets\n",
    "!PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml gen-synthetic\n",
    "!PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml data\n",
))

cells.append(md(
    "## ONE BUTTON - autopilot (resume-safe)\n",
    "Persists the dictionary baseline, fine-tunes the MT core, evaluates MT chrF/BLEU + OCR CER/WER +\n",
    "end-to-end image-translation chrF + layout fit-rate, runs analysis, and writes **report.pdf +\n",
    "slides.pptx + grading + bundle**. Re-run to resume from the last checkpoint.\n",
))

cells.append(code(
    "#@title 10) ONE BUTTON autopilot  (re-run to resume)\n",
    "import os\n",
    "if RUN_AUTOPILOT:\n",
    "    os.system('PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml autopilot '\n",
    "              f'--limit {int(MAX_TRAIN_SAMPLES)}')\n",
    "else:\n",
    "    print('RUN_AUTOPILOT is off - use the individual steps below.')\n",
))

cells.append(md("## Individual steps (optional) - idempotent + resume-safe\n"))

cells.append(code(
    "#@title 11a) Fine-tune the MT core (resumes from the last checkpoint)\n",
    "!PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml train-mt --limit $MAX_TRAIN_SAMPLES --base-model \"$MT_BASE\"\n",
))

cells.append(code(
    "#@title 11b) Baseline + evaluate (MT chrF/BLEU + OCR CER/WER + end-to-end + fidelity) + tune\n",
    "!PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml train-baseline\n",
    "!PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml evaluate --ocr-noise 0.05\n",
    "!PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml tune\n",
))

cells.append(code(
    "#@title 12) Diagnostics: eval headline + model metadata\n",
    "import json, glob, os\n",
    "rd = os.path.join(os.environ['IMGTRANS_ARTIFACTS_DIR'], 'runs', 'eval.json')\n",
    "if os.path.exists(rd):\n",
    "    print(json.dumps(json.load(open(rd)).get('headline', {}), indent=2))\n",
    "for m in glob.glob(os.path.join(os.environ['IMGTRANS_ARTIFACTS_DIR'], 'models', 'mt', '*', 'model_meta.json')):\n",
    "    print(m); print(json.dumps(json.load(open(m)), indent=2)[:500])\n",
))

cells.append(md("## Test the trained model\n"))

cells.append(code(
    "#@title 13) Translate the sample document image (overlay) with the trained MT core\n",
    "!PYTHONPATH=src python -m imgtrans.cli --config configs/train_colab.yaml translate-image \\\n",
    "  --image sample_data/sample_document_en.png --mode overlay --out /content/translated_overlay.png\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "fig, ax = plt.subplots(1, 2, figsize=(14, 6))\n",
    "ax[0].imshow(Image.open('sample_data/sample_document_en.png')); ax[0].set_title('source'); ax[0].axis('off')\n",
    "ax[1].imshow(Image.open('/content/translated_overlay.png')); ax[1].set_title('translated overlay'); ax[1].axis('off')\n",
    "plt.tight_layout(); plt.show()\n",
))

cells.append(code(
    "#@title 14) Locate deliverables (report.pdf + slides.pptx + bundle)\n",
    "import glob, os\n",
    "base = os.environ['IMGTRANS_ARTIFACTS_DIR']\n",
    "for pat in ['submission/*/report.pdf', 'submission/*/slides.pptx', 'submission/*/submission_bundle.zip']:\n",
    "    for f in glob.glob(os.path.join(base, pat)):\n",
    "        print(round(os.path.getsize(f)/1024, 1), 'KB', f)\n",
))

cells.append(code(
    "#@title 15) (Optional) Serve the API + Gradio demo\n",
    "# !PYTHONPATH=src python -m imgtrans.cli --config configs/infer.yaml serve --ui --port 7860\n",
    "print('Uncomment to serve. On Colab add a tunnel (e.g. cloudflared) to expose :7860.')\n",
))

cells.append(md(
    "## Final checklist\n",
    "- [ ] Tesseract installed (cell 2) + GPU profile picked a sensible batch/precision\n",
    "- [ ] `train-mt` wrote `models/mt/<version>/`; `train-baseline` wrote `dictionary_mt.json`\n",
    "- [ ] `evaluate` shows the fine-tuned MT core beating the dictionary / identity baselines on **chrF**\n",
    "- [ ] OCR **CER** is low + end-to-end image-translation **chrF** is high + overlay **fit-rate** ~ 1.0\n",
    "- [ ] the overlay image (cell 13) reads correctly in the target language\n",
    "- [ ] `report.pdf` + `slides.pptx` + `submission_bundle.zip` exist under `artifacts/submission/`\n",
    "- [ ] Remember: opus-100 is license-unknown, NLLB/Surya are CC-BY-NC; the clean stack is m2m100 (MIT) + Tesseract (Apache)\n",
))


def main():
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(__file__).resolve().parent / NB
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
