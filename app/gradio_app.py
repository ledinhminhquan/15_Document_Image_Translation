"""Standalone Gradio app entrypoint (also used by the Hugging Face Space).

Run locally:  python app/gradio_app.py
HF Space:     this file is the Space's `app.py` (rename or point the Space at it).
"""

from __future__ import annotations

import os
import sys

# allow running from the repo root without an install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from imgtrans.api.ui import build_demo  # noqa: E402


def main() -> None:
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
