"""Combined ASGI app: the FastAPI service with the Gradio demo mounted at ``/ui``.

Used by ``imgtrans serve --ui`` and the Hugging Face Space. If Gradio is unavailable
the API still serves on its own.
"""

from __future__ import annotations

from .main import app
from ..logging_utils import get_logger

logger = get_logger(__name__)

try:
    import gradio as gr  # noqa: F401
    from .ui import build_demo

    demo = build_demo()
    app = gr.mount_gradio_app(app, demo, path="/ui")
    logger.info("Gradio demo mounted at /ui")
except Exception as exc:  # pragma: no cover
    logger.info("Gradio UI not mounted (%s); API-only.", exc)


__all__ = ["app"]
