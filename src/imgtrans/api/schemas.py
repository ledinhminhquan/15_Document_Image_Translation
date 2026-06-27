"""Pydantic request/response schemas for the document-image-translation API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TranslateTextRequest(BaseModel):
    text: str = Field(..., description="Raw text blocks (one per line) to translate")
    mode: str = Field("text_only", description="overlay | side_by_side | text_only")


class TranslateResponse(BaseModel):
    input_kind: str = ""
    src_lang: str = ""
    tgt_lang: str = ""
    source_text: str = ""
    translated_text: str = ""
    translated_markdown: str = ""
    status: str = ""
    ocr_engine: str = ""
    n_blocks: int = 0
    n_translatable: int = 0
    n_skipped_lowconf: int = 0
    mean_ocr_conf: Optional[float] = None
    fit_rate: Optional[float] = None
    render_mode_used: str = ""
    verify_chrf: Optional[float] = None
    needs_review: bool = False
    rationale: str = ""
    rendered_image_b64: Optional[str] = None
    blocks: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    model_versions: Dict[str, str] = {}
    disclaimer: str = ("Machine translation of in-image text. Low-confidence OCR is skipped and "
                       "low-confidence/poor-fit output is flagged for human review.")


class HealthResponse(BaseModel):
    status: str
    mt: str
    ocr: str
    direction: str
    version: str


__all__ = ["TranslateTextRequest", "TranslateResponse", "HealthResponse"]
