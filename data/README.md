# data/

This directory holds dataset caches and the **generated synthetic document-image pages** at
runtime — none of it is committed (see `.gitignore`).

- The MT fine-tune corpus (`Helsinki-NLP/opus-100` en-fr) is streamed by `imgtrans data` /
  `imgtrans train-mt` and cached under `HF_HOME`.
- The synthetic eval pages are rendered by `imgtrans gen-synthetic` into
  `$IMGTRANS_DATA_DIR/synthetic/<split>/` (PNGs + `manifest.jsonl`), each PNG embedding its gold
  layout spec.
- The **offline backbone** (synthetic seed pages + the en→fr dictionary) is code, not data — it
  lives in [`src/imgtrans/data/samples.py`](../src/imgtrans/data/samples.py), so the whole pipeline
  runs with no network.

Nothing here is required to import the package or run the tests.
