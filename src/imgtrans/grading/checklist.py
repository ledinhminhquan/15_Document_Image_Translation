"""Rubric completeness self-check (PASS/WARN/FAIL over assignment deliverables)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)

PKG = "imgtrans"

_SUBPACKAGES = ["data", "models", "mt", "imaging", "training", "agent", "api",
                "analysis", "autoreport", "monitoring", "automation", "grading"]
_REQUIRED_SRC = [
    f"src/{PKG}/config.py", f"src/{PKG}/cli.py", f"src/{PKG}/logging_utils.py",
    f"src/{PKG}/data/dataset.py", f"src/{PKG}/data/samples.py", f"src/{PKG}/data/download_dataset.py",
    f"src/{PKG}/data/synth_render.py", f"src/{PKG}/models/ocr_engine.py", f"src/{PKG}/mt/translator.py",
    f"src/{PKG}/imaging/layout.py", f"src/{PKG}/imaging/render.py", f"src/{PKG}/imaging/preprocess.py",
    f"src/{PKG}/training/train_mt.py", f"src/{PKG}/training/evaluate.py", f"src/{PKG}/training/metrics.py",
    f"src/{PKG}/agent/imgtrans_agent.py", f"src/{PKG}/agent/policy.py", f"src/{PKG}/api/main.py",
]
_REQUIRED_DIRS = ["src", "docs", "notebooks", "tests", "configs", "app", "deploy", "sample_data", ".github"]
_REQUIRED_ROOT = ["README.md", "LICENSE", "requirements.txt", "requirements_colab.txt",
                  "pyproject.toml", "Dockerfile", "docker-compose.yml", "Makefile"]
_REQUIRED_DOCS = [
    "problem_definition", "data_description", "data_card", "model_selection", "deployment",
    "agent_architecture", "continual_learning_monitoring", "privacy_robustness", "project_plan",
    "ethics_statement", "architecture", "layout_fidelity_evaluation", "model_card",
    "slide_deck_outline", "DESIGN_BRIEF",
]

_REQUIREMENTS = {
    "R1_problem_data": ("Problem definition + dataset/data card",
                        ["docs/problem_definition.md", "docs/data_description.md",
                         "docs/data_card.md", f"src/{PKG}/data/dataset.py"]),
    "R2_model_baseline": ("Model selection + baseline (dictionary/identity) vs MT core",
                          ["docs/model_selection.md", f"src/{PKG}/mt/translator.py",
                           f"src/{PKG}/training/train_baseline.py"]),
    "R3_training_eval": ("Training + evaluation with metrics (chrF/BLEU + OCR CER/WER + fit-rate)",
                         [f"src/{PKG}/training/train_mt.py", f"src/{PKG}/training/evaluate.py",
                          "docs/layout_fidelity_evaluation.md"]),
    "R4_agent": ("Agentic image-translation pipeline with explicit decision points",
                 [f"src/{PKG}/agent/imgtrans_agent.py", f"src/{PKG}/agent/policy.py", "docs/agent_architecture.md"]),
    "R5_serving": ("Deployment / serving surface (API + app)",
                   [f"src/{PKG}/api/main.py", "docs/deployment.md", "app", "deploy"]),
    "R6_monitoring": ("Continual learning + monitoring",
                      ["docs/continual_learning_monitoring.md", f"src/{PKG}/monitoring", f"src/{PKG}/automation"]),
    "R7_privacy_ethics": ("Privacy / robustness + ethics statement",
                          ["docs/privacy_robustness.md", "docs/ethics_statement.md"]),
    "R8_reproducibility": ("Reproducibility (notebook, requirements, Docker, CI)",
                           ["notebooks", "requirements.txt", "Dockerfile", ".github/workflows/ci.yml"]),
    "R9_reporting": ("Reporting / planning (model card, plan, slides, autoreport)",
                     ["docs/model_card.md", "docs/project_plan.md", "docs/slide_deck_outline.md",
                      f"src/{PKG}/autoreport"]),
}


def _exists(root: Path, rel: str) -> bool:
    p = root / rel
    return p.is_file() or p.is_dir()


def build_checklist(repo) -> Dict:
    root = Path(repo)
    items: List[Dict] = []

    def check(name: str, ok: bool, detail: str, optional: bool = False) -> None:
        status = "PASS" if ok else ("WARN" if optional else "FAIL")
        items.append({"name": name, "status": status, "detail": detail})

    pkg_dir = root / "src" / PKG
    check(f"Package: src/{PKG}/", pkg_dir.is_dir(), str(pkg_dir))
    check(f"Package init: src/{PKG}/__init__.py", (pkg_dir / "__init__.py").is_file(), f"src/{PKG}/__init__.py")
    for sub in _SUBPACKAGES:
        d = pkg_dir / sub
        check(f"Subpackage: {PKG}/{sub}/", (d / "__init__.py").is_file(), str(d))
    for rel in _REQUIRED_SRC:
        check(f"Module: {rel}", (root / rel).is_file(), str(root / rel))
    for rel in _REQUIRED_DIRS:
        check(f"Dir: {rel}/", (root / rel).is_dir(), str(root / rel))
    for rel in _REQUIRED_ROOT:
        optional = rel in ("requirements_colab.txt", "docker-compose.yml", "Makefile")
        check(f"File: {rel}", (root / rel).is_file(), str(root / rel), optional=optional)
    docs_dir = root / "docs"
    for d in _REQUIRED_DOCS:
        check(f"Doc: {d}.md", (docs_dir / f"{d}.md").is_file(), f"docs/{d}.md")

    nb_dir = root / "notebooks"
    nbs = list(nb_dir.glob("*.ipynb")) if nb_dir.is_dir() else []
    check("Notebook: >=1 .ipynb", len(nbs) >= 1, f"{len(nbs)} notebook(s)")
    check("Notebook: COLAB_GUIDE.md", (nb_dir / "COLAB_GUIDE.md").is_file(),
          "notebooks/COLAB_GUIDE.md", optional=True)
    tests_dir = root / "tests"
    tests = list(tests_dir.glob("test_*.py")) if tests_dir.is_dir() else []
    check("Tests: >=1 test_*.py", len(tests) >= 1, f"{len(tests)} test file(s)")
    cfg_dir = root / "configs"
    yamls = (list(cfg_dir.glob("*.yaml")) + list(cfg_dir.glob("*.yml"))) if cfg_dir.is_dir() else []
    check("Configs: configs/*.yaml", len(yamls) >= 1, f"{len(yamls)} config(s)")
    check("CI: .github/workflows/ci.yml", (root / ".github" / "workflows" / "ci.yml").is_file(),
          ".github/workflows/ci.yml")

    import_ok = _check_import()
    check("Functional: package imports", import_ok["ok"], import_ok["detail"], optional=not import_ok["ok"])
    agent_res = _check_agent_offline()
    check("Functional: agent runs offline (D1-D5)", agent_res["ok"], agent_res["detail"],
          optional=not agent_res["ok"])
    check("Agent: >=5 decision points fire", agent_res.get("n_decisions", 0) >= 5,
          f"{agent_res.get('n_decisions', 0)}/5 decisions", optional=True)
    eval_res = _check_evaluate()
    check("Functional: evaluate() produces MT chrF + end-to-end chrF", eval_res["ok"],
          eval_res["detail"], optional=not eval_res["ok"])

    requirement_coverage: Dict[str, Dict] = {}
    for rid, (label, artifacts) in _REQUIREMENTS.items():
        hit = next((a for a in artifacts if _exists(root, a)), None)
        covered = hit is not None
        requirement_coverage[rid] = {"label": label, "covered": covered, "artifact": hit if covered else None}
        check(f"Requirement: {rid} ({label})", covered, hit if covered else "no delivered artifact",
              optional=not covered)

    n_pass = sum(i["status"] == "PASS" for i in items)
    n_warn = sum(i["status"] == "WARN" for i in items)
    n_fail = sum(i["status"] == "FAIL" for i in items)
    total = len(items)
    score = round((n_pass + 0.5 * n_warn) / total, 4) if total else 0.0
    summary = {"PASS": n_pass, "WARN": n_warn, "FAIL": n_fail, "total": total, "score": score}
    logger.info("checklist: %s", summary)
    return {"summary": summary, "items": items, "requirement_coverage": requirement_coverage,
            "ok": n_fail == 0}


def _check_import() -> Dict:
    try:
        import importlib
        for m in (PKG, f"{PKG}.config", f"{PKG}.agent.imgtrans_agent", f"{PKG}.training.evaluate"):
            importlib.import_module(m)
        return {"ok": True, "detail": f"import {PKG} ok"}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "detail": f"import failed: {exc}"}


def _offline_cfg():
    from ..config import load_config
    cfg = load_config()
    try:
        cfg.data.use_hf = False
    except Exception:
        pass
    return cfg


def _check_agent_offline() -> Dict:
    try:
        from ..agent.imgtrans_agent import ImgTransAgent
        from ..data import samples
        cfg = _offline_cfg()
        agent = ImgTransAgent(cfg, load_model=False)
        job = agent.run(spec=samples.seed_pages()[0], mode="text_only", save=False)
        data = job.to_dict()
        ids = {d.get("id") for d in data.get("decisions", [])}
        n = len([x for x in ids if x])
        return {"ok": data.get("status") in ("completed", "needs_review") and n >= 5,
                "n_decisions": n,
                "detail": f"status={data.get('status')}, {n} decisions: {sorted(str(i) for i in ids)}"}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "n_decisions": 0, "detail": f"agent offline run failed: {exc}"}


def _check_evaluate() -> Dict:
    try:
        from ..training.evaluate import evaluate
        res = evaluate(_offline_cfg(), save=False, load_model=False)
        systems = ((res.get("mt") or {}).get("systems")) or {}
        e2e = (res.get("ocr_e2e") or {}).get("end_to_end") or {}
        ok = bool(systems) and ("chrf" in e2e)
        return {"ok": bool(ok), "detail": f"mt systems={list(systems)} e2e_chrf={e2e.get('chrf')}"}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "detail": f"evaluate() failed: {exc}"}


def write_checklist(repo, out_path: Optional[str] = None) -> Path:
    res = build_checklist(repo)
    if out_path is None:
        try:
            from ..config import run_dir
            out_path = run_dir() / "grading" / "checklist.json"
        except Exception:
            out_path = Path(repo) / "grading_checklist.json"
    p = Path(out_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        logger.warning("could not write checklist: %s", exc)
    return p


__all__ = ["build_checklist", "write_checklist"]
