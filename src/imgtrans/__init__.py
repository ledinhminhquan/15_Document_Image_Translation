"""imgtrans - Document-Image Machine Translation.

An end-to-end, production-grade system that translates the text appearing INSIDE
an image or scanned document and (optionally) renders the translation back onto
the page preserving spatial layout. Cascade: OCR -> MT (the only trained stage)
-> layout-preserving overlay render. Built for the "NLP in Industry" final
assignment (project #15).
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import AppConfig, load_config, save_config, ensure_dirs  # noqa: E402

__all__ = ["AppConfig", "load_config", "save_config", "ensure_dirs", "__version__"]
