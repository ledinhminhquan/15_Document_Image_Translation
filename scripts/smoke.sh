#!/usr/bin/env bash
# Offline smoke test: no torch / tesseract / network needed (dictionary MT + SeedEngine).
set -euo pipefail
cd "$(dirname "$0")/.."

export IMGTRANS_ARTIFACTS_DIR="${IMGTRANS_ARTIFACTS_DIR:-/tmp/imgtrans_smoke}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "== compile ==";        python -m compileall -q src/imgtrans
echo "== data probe ==";     python -m imgtrans.cli data || true
echo "== demo agent ==";     python -m imgtrans.cli demo-agent --fast
echo "== evaluate ==";       python -m imgtrans.cli evaluate --fast
echo "== translate-text =="; python -m imgtrans.cli translate-text --file sample_data/sample_lines_en.txt --fast
echo "== grade ==";          python -m imgtrans.cli grade | python -c "import json,sys;print(json.load(sys.stdin)['summary'])"
echo "OK"
