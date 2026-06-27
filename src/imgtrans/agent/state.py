"""Shared state types for the document-image-translation agent (deterministic FSM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    INGESTED = "ingested"
    OCR_DONE = "ocr_done"
    TRANSLATED = "translated"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"   # a quality gate fired (low OCR conf / low verify chrF / poor fit)
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str               # D1..D5
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class JobState:
    # ---- inputs --------------------------------------------------------------
    input_kind: str = "image"      # image | pdf | spec | text
    filename: str = ""
    src_lang: str = "en"
    tgt_lang: str = "fr"
    mode: str = "overlay"          # overlay | side_by_side | text_only
    # ---- derived -------------------------------------------------------------
    status: JobStatus = JobStatus.PENDING
    born_digital: bool = False
    page_quality: Optional[float] = None
    ocr_engine: str = ""
    n_blocks: int = 0
    n_translatable: int = 0
    n_skipped_lowconf: int = 0
    mean_ocr_conf: Optional[float] = None
    # ---- outputs -------------------------------------------------------------
    source_text: str = ""
    translated_text: str = ""
    source_markdown: str = ""
    translated_markdown: str = ""
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    fit_rate: Optional[float] = None
    fidelity: Dict[str, Any] = field(default_factory=dict)
    verify_chrf: Optional[float] = None
    n_retranslated: int = 0
    rendered_path: str = ""
    render_mode_used: str = ""
    needs_review: bool = False
    rationale: str = ""
    # ---- audit ---------------------------------------------------------------
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_kind": self.input_kind, "filename": self.filename,
            "src_lang": self.src_lang, "tgt_lang": self.tgt_lang, "mode": self.mode,
            "status": self.status.value, "born_digital": self.born_digital,
            "page_quality": self.page_quality, "ocr_engine": self.ocr_engine,
            "n_blocks": self.n_blocks, "n_translatable": self.n_translatable,
            "n_skipped_lowconf": self.n_skipped_lowconf, "mean_ocr_conf": self.mean_ocr_conf,
            "source_text": self.source_text, "translated_text": self.translated_text,
            "source_markdown": self.source_markdown, "translated_markdown": self.translated_markdown,
            "blocks": self.blocks, "fit_rate": self.fit_rate, "fidelity": self.fidelity,
            "verify_chrf": self.verify_chrf, "n_retranslated": self.n_retranslated,
            "rendered_path": self.rendered_path, "render_mode_used": self.render_mode_used,
            "needs_review": self.needs_review, "rationale": self.rationale,
            "decisions": [d.to_dict() for d in self.decisions],
            "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics, "model_versions": self.model_versions,
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "JobState"]
