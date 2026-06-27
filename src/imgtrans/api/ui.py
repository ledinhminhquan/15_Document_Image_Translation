"""Gradio demo UI for document-image machine translation.

Upload a document image (or use a synthetic sample) -> the agent OCRs the text,
translates it, and renders the translation back onto the page (overlay) or beside it.
Heavy deps (gradio/PIL) are imported lazily so the package imports without them.
"""

from __future__ import annotations

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_demo(cfg: AppConfig = None, load_model: bool = True):
    import gradio as gr
    from ..agent.imgtrans_agent import ImgTransAgent

    cfg = cfg or AppConfig()
    agent = ImgTransAgent(cfg, load_model=load_model)

    def run(image, mode):
        if image is None:
            return None, "Please upload a document image.", ""
        import tempfile
        render_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name if mode != "text_only" else ""
        job = agent.run(image=image, mode=mode, render_path=render_path, save=False)
        out_img = None
        if job.rendered_path:
            try:
                from PIL import Image
                out_img = Image.open(job.rendered_path)
            except Exception:
                out_img = None
        info = (f"status={job.status.value} | OCR={job.ocr_engine} | blocks={job.n_blocks} "
                f"| translated={job.n_translatable} | skipped={job.n_skipped_lowconf} "
                f"| fit_rate={job.fit_rate} | mode={job.render_mode_used} "
                f"| needs_review={job.needs_review}")
        return out_img, job.translated_text, info

    with gr.Blocks(title=cfg.serving.api_title) as demo:
        gr.Markdown(f"# {cfg.serving.api_title}\n"
                    f"Translate the text inside a document image ({cfg.mt.src_lang} -> {cfg.mt.tgt_lang}). "
                    "OCR -> MT -> layout-preserving overlay. Low-confidence OCR is skipped; "
                    "poor-fit output falls back to side-by-side.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Document image")
                mode = gr.Radio(["overlay", "side_by_side", "text_only"], value="overlay", label="Output mode")
                btn = gr.Button("Translate", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Translated image")
                out_txt = gr.Textbox(label="Translated text", lines=8)
                out_info = gr.Textbox(label="Pipeline trace", lines=2)
        btn.click(run, inputs=[inp, mode], outputs=[out_img, out_txt, out_info])
        gr.Markdown("_Machine translation; flagged output should be reviewed by a human._")
    return demo


def launch(cfg: AppConfig = None, share: bool = False, **kwargs):
    build_demo(cfg).launch(share=share, **kwargs)


__all__ = ["build_demo", "launch"]
