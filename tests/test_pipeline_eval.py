"""End-to-end (offline): evaluate() + error analysis + grading + report/slides smoke."""

from __future__ import annotations


def test_evaluate_offline_produces_metrics(cfg):
    from imgtrans.training.evaluate import evaluate
    rep = evaluate(cfg, save=False, load_model=False)
    systems = rep["mt"]["systems"]
    assert "identity" in systems and "dictionary" in systems
    assert "chrf" in rep["ocr_e2e"]["end_to_end"]
    assert rep["ocr_e2e"]["ocr"]["cer"] is not None
    # dictionary beats the identity floor on the seed
    assert systems["dictionary"]["chrf"] >= systems["identity"]["chrf"]


def test_error_analysis_and_fidelity(cfg):
    from imgtrans.analysis.error_analysis import error_analysis
    from imgtrans.analysis.layout_fidelity import layout_fidelity_report
    ea = error_analysis(cfg, save=False)
    assert ea["n_pages"] >= 1
    lf = layout_fidelity_report(cfg, save=False)
    assert lf["mean_fit_rate"] is not None


def test_grading_runs(cfg):
    from pathlib import Path

    from imgtrans.grading.checklist import build_checklist
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["total"] > 0
    assert res["summary"]["FAIL"] == 0  # all required deliverables present


def test_dictionary_baseline_translates():
    from imgtrans.mt.translator import DictionaryTranslator
    t = DictionaryTranslator()
    out = t.translate("the museum is open every day")
    # word-lookup MT: known words are translated (it cannot disambiguate senses)
    assert "musee" in out and out != "the museum is open every day"
