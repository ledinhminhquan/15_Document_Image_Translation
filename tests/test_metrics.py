"""Metrics: chrF / BLEU (MT) + CER / WER (OCR), pure-python fallbacks."""

from __future__ import annotations

from imgtrans.training import metrics as M


def test_chrf_perfect_and_floor():
    hyps = ["le chat noir dort sur le canape."]
    assert M.chrf(hyps, hyps) > 95.0
    assert M.chrf(["aaaa"], ["zzzz"]) < 20.0


def test_bleu_monotone():
    refs = ["the meeting starts tomorrow at nine"]
    good = M.bleu(["the meeting starts tomorrow at nine"], refs)
    bad = M.bleu(["completely different words here"], refs)
    assert good > bad


def test_cer_wer():
    assert M.cer(["hello"], ["hello"]) == 0.0
    assert M.wer(["hello world"], ["hello world"]) == 0.0
    # one char substitution -> CER 0.2 on a 5-char reference
    assert abs(M.cer(["hallo"], ["hello"]) - 0.2) < 1e-6
    assert M.wer(["hello there"], ["hello world"]) == 0.5


def test_ocr_translation_metric_dicts():
    om = M.ocr_metrics(["a b c"], ["a b c"])
    assert set(om) == {"cer", "wer", "n"}
    tm = M.translation_metrics(["a"], ["a"])
    assert "chrf" in tm and "bleu" in tm
