"""Metrics for Document-Image Machine Translation.

Two families:
* MT quality - ``chrF`` (headline) + ``BLEU`` (via sacrebleu when available, with
  self-contained pure-python fallbacks so metrics compute with no heavy deps).
  Reused from P13/P14.
* OCR quality - ``CER`` (headline) + ``WER`` (character / word error rate) for the
  OCR front-end on the synthetic slice where the gold source text is known.

The end-to-end "image-translation chrF" simply runs OCR -> MT on a rendered image
and scores the result with ``chrf`` against the gold target translation; layout
fidelity (fit-rate / box-retention) is computed in ``imaging/render.py``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Sequence

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tok(s: str) -> List[str]:
    return _WORD.findall((s or "").lower())


def _char_ngrams(s: str, n: int) -> Counter:
    s = re.sub(r"\s+", "", s or "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def chrf(hyps: Sequence[str], refs: Sequence[str], max_n: int = 6, beta: float = 2.0) -> float:
    try:
        import sacrebleu
        return float(sacrebleu.corpus_chrf(list(hyps), [list(refs)]).score)
    except Exception:
        pass
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        tp = h = r = 0
        for hyp, ref in zip(hyps, refs):
            hn, rn = _char_ngrams(hyp, n), _char_ngrams(ref, n)
            tp += sum((hn & rn).values())
            h += sum(hn.values())
            r += sum(rn.values())
        if h:
            precisions.append(tp / h)
        if r:
            recalls.append(tp / r)
    if not precisions or not recalls:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return round(100.0 * (1 + b2) * p * r / (b2 * p + r), 4)


def bleu(hyps: Sequence[str], refs: Sequence[str], max_n: int = 4) -> float:
    try:
        import sacrebleu
        return float(sacrebleu.corpus_bleu(list(hyps), [list(refs)]).score)
    except Exception:
        pass
    weights = [1.0 / max_n] * max_n
    p_log = 0.0
    hyp_len = ref_len = 0
    clipped = [0] * max_n
    total = [0] * max_n
    for hyp, ref in zip(hyps, refs):
        ht, rt = _tok(hyp), _tok(ref)
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            hng = Counter(tuple(ht[i:i + n]) for i in range(len(ht) - n + 1))
            rng = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
            clipped[n - 1] += sum((hng & rng).values())
            total[n - 1] += max(0, len(ht) - n + 1)
    for n in range(max_n):
        if total[n] == 0 or clipped[n] == 0:
            return 0.0
        p_log += weights[n] * math.log(clipped[n] / total[n])
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    return round(100.0 * bp * math.exp(p_log), 4)


def _levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    tot_err = tot_words = 0
    for hyp, ref in zip(hyps, refs):
        h, r = _tok(hyp), _tok(ref)
        tot_words += len(r)
        tot_err += _levenshtein(h, r)
    return round(tot_err / max(1, tot_words), 4)


def cer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Character Error Rate: char-level edit distance / total reference chars."""
    tot_err = tot_chars = 0
    for hyp, ref in zip(hyps, refs):
        h = list((hyp or "").strip())
        r = list((ref or "").strip())
        tot_chars += len(r)
        tot_err += _levenshtein(h, r)
    return round(tot_err / max(1, tot_chars), 4)


def translation_metrics(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, Any]:
    return {"chrf": chrf(hyps, refs), "bleu": bleu(hyps, refs), "n": len(hyps)}


def ocr_metrics(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, Any]:
    return {"cer": cer(hyps, refs), "wer": wer(hyps, refs), "n": len(hyps)}


__all__ = ["chrf", "bleu", "wer", "cer", "translation_metrics", "ocr_metrics"]
