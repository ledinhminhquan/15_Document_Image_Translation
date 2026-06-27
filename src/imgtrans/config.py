"""Typed configuration + YAML loader for the imgtrans Document-Image Machine
Translation system.

Single source of truth for the trainable MT core, the OCR front-end, layout
analysis, the layout-preserving overlay renderer, the agent decision thresholds
(D1-D5), the datasets, and serving. Paths come from environment variables so
nothing is hard-coded (required by the assignment). Default direction:
English -> French (configurable via ``DataConfig`` / ``MtConfig``).

Pipeline (cascade): image/PDF -> OCR (boxes+conf) -> layout -> MT (the only
trained stage) -> render translated text back onto the page (overlay) OR export
text/markdown. OCR + layout + render are pretrained/algorithmic; only the MT
model is fine-tuned.

Environment overrides
---------------------
* ``IMGTRANS_ARTIFACTS_DIR`` - base for data/models/runs (Drive on Colab)
* ``IMGTRANS_DATA_DIR``      - dataset cache / generated synthetic images
* ``IMGTRANS_MODEL_DIR``     - trained models (the fine-tuned MT core)
* ``IMGTRANS_RUN_DIR``       - eval/benchmark/analysis JSON
* ``IMGTRANS_OUTPUT_DIR``    - translated documents / rendered images
* ``HF_HOME``                - HuggingFace cache
* ``IMGTRANS_LLM_API_KEY``   - optional key for the LLM agent brain

Verified ids (confirmed on the HF Hub during research - keep exact):
  mt fine-tune  Helsinki-NLP/opus-100 (en-fr; license unknown flag)
  mt model      facebook/m2m100_418M (MIT, default) | nllb-200-distilled-600M (CC-BY-NC flag) |
                Helsinki-NLP/opus-mt-en-fr (Apache, cheapest) - see docs/DESIGN_BRIEF.md
  ocr           Tesseract (system, Apache-2.0) default; docTR / EasyOCR optional
  ocr eval data PleIAs/Post-OCR-Correction (CC0) - real OCR-noise text source
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("IMGTRANS_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("IMGTRANS_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("IMGTRANS_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def run_dir() -> Path:
    return Path(_env("IMGTRANS_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


def output_dir() -> Path:
    return Path(_env("IMGTRANS_OUTPUT_DIR", str(artifacts_dir() / "outputs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Parallel MT corpus (fine-tune) + the synthetic document-image generator.

    There is NO public in-image / document-image translation benchmark with gold
    parallel text, so the PRIMARY data is a reproducible SYNTHETIC generator
    (``data/synth_render.py``) that renders source sentences (from the MT corpus)
    onto page images with varied fonts / backgrounds / light degradation, giving
    (image, gold_source_text, gold_target_translation) triples. This lets us
    measure OCR CER/WER, MT chrF/BLEU AND the end-to-end image-translation chrF.
    """
    # PRIMARY fine-tune corpus (VERIFIED): Helsinki-NLP/opus-100 en-fr (translation {en,fr}).
    mt_dataset: str = "Helsinki-NLP/opus-100"
    mt_config: str = "en-fr"
    # optional real OCR-noise text source (VERIFIED, CC0) for OCR robustness slices
    ocr_text_dataset: str = "PleIAs/Post-OCR-Correction"
    ocr_text_config: str = "english"
    src_lang: str = "en"
    tgt_lang: str = "fr"
    use_hf: bool = True
    max_train_samples: int = 50000
    max_eval_samples: int = 2000
    # synthetic document-image generator
    synth_train_pages: int = 400      # rendered page images for the OCR/e2e eval slices
    synth_eval_pages: int = 80
    lines_per_page: int = 6           # sentences rendered per synthetic page
    image_width: int = 1000
    seed: int = 42


@dataclass
class MtConfig:
    """The TRAINABLE MT core (the NLP heart of this project)."""
    base_model: str = "facebook/m2m100_418M"  # MIT. alt nllb-200-distilled-600M (CC-BY-NC) / opus-mt-en-fr (Apache)
    src_lang: str = "en"
    tgt_lang: str = "fr"
    max_source_length: int = 200
    max_target_length: int = 200
    num_beams: int = 4
    # training (HF Seq2SeqTrainer)
    num_train_epochs: int = 3
    learning_rate: float = 3.0e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = False
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    seed: int = 42
    output_subdir: str = "mt"
    baseline_filename: str = "dictionary_mt.json"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir

    @property
    def baseline_path(self) -> Path:
        return self.output_dir / self.baseline_filename


@dataclass
class OcrConfig:
    """OCR engine for the document front-end (page image -> words + boxes + conf)."""
    engine: str = "auto"                   # "auto"|"tesseract"|"easyocr"|"stub"
    lang: str = "eng"                      # tesseract lang (source language script)
    dpi: int = 200                         # rasterise scanned PDFs at 200-300 DPI
    min_word_conf: float = 0.0             # keep all words; conf used for gating
    psm: int = 3                           # tesseract page segmentation mode (auto)


@dataclass
class LayoutConfig:
    """Layout analysis & reading order (group OCR words into translatable blocks)."""
    detect_layout: bool = True
    reading_order: str = "xycut"           # "xycut" | "topdown"
    min_region_area: int = 200
    born_digital_min_chars: int = 40       # PDFs with >= this many extractable chars => born-digital
    classify_blocks: bool = True


@dataclass
class PreprocessConfig:
    deskew: bool = True
    denoise: bool = True
    binarize: str = "adaptive"             # "adaptive" | "otsu" | "none"
    max_skew_deg: float = 15.0


@dataclass
class RenderConfig:
    """The layout-preserving overlay renderer (translated text -> back onto the page).

    The headline value-add of P15: re-draw each translated block INTO the original
    bounding box, auto-shrinking the font + word-wrapping so it fits (translated
    text is often longer than the source). Pillow-only (no OpenCV required).
    """
    mode: str = "overlay"                  # "overlay" | "side_by_side" | "text_only"
    font_path: str = ""                    # explicit TTF; "" => auto-discover Noto/DejaVu
    max_font_size: int = 48
    min_font_size: int = 8
    box_padding: int = 3                   # px padding inside a block box
    line_spacing: float = 1.12
    fill_color: str = "white"              # whiteout the original text region
    text_color: str = "black"
    draw_box_border: bool = False          # debug: outline translated boxes


@dataclass
class AgentConfig:
    """Document-image-translation agent decision thresholds (D1-D5) + optional LLM brain."""
    # D1 - input routing handled structurally (image / pdf / born-digital / raw-text)
    # D1b - page-quality routing (blur/contrast/ink)
    quality_min: float = 0.30
    # D3 - per-block OCR-confidence gate (low-confidence block -> skip translation / flag)
    ocr_confidence_min: float = 0.50
    min_block_chars: int = 2               # ignore tiny/garbage OCR fragments
    # D4 - translation verification: round-trip back-translation chrF + length-ratio sanity
    verify_enabled: bool = True
    verify_min_chrf: float = 0.30          # round-trip chrF (0..1) below this -> flag block
    length_ratio_low: float = 0.3
    length_ratio_high: float = 3.5
    max_retranslate: int = 2               # D4 -> retranslate budget (N)
    # D5 - render-fit feasibility gate (can the translation fit the box?)
    min_fit_rate: float = 0.60             # below this fraction of blocks fitting -> side_by_side fallback
    needs_review_conf: float = 0.45        # overall confidence below this -> needs_review
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "IMGTRANS_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Document-Image Machine Translation API"
    api_version: str = "1.0.0"
    log_jobs: bool = True
    job_log_subdir: str = "job_logs"
    max_file_mb: int = 25
    max_pages: int = 30

    @property
    def job_log_path(self) -> Path:
        return run_dir() / self.job_log_subdir / "jobs.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Document-Image Machine Translation System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    mt: MtConfig = field(default_factory=MtConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "mt": MtConfig, "ocr": OcrConfig, "layout": LayoutConfig,
             "preprocess": PreprocessConfig, "render": RenderConfig, "agent": AgentConfig,
             "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(),
            "runs": run_dir(), "outputs": output_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "MtConfig", "OcrConfig", "LayoutConfig", "PreprocessConfig",
           "RenderConfig", "AgentConfig", "ServingConfig", "AppConfig",
           "load_config", "save_config", "ensure_dirs",
           "artifacts_dir", "data_dir", "model_dir", "run_dir", "output_dir"]
