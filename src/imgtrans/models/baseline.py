"""Baselines for the MT core - the honest floors the fine-tuned model must beat.

* ``IdentityTranslator`` - copies the source (the absolute floor; chrF of source-vs-target).
* ``DictionaryTranslator`` (re-exported) - word-lookup MT, the offline fallback.
* zero-shot base model = ``TransformerTranslator.from_pretrained(base_model)`` without
  fine-tuning (built in ``training/evaluate.py``).
"""

from __future__ import annotations

from typing import List

from ..mt.translator import DictionaryTranslator


class IdentityTranslator:
    name = "identity"
    version = "identity-1.0"

    def translate(self, text: str) -> str:
        return text or ""

    def translate_batch(self, texts: List[str]) -> List[str]:
        return [t or "" for t in texts]

    def reversed(self):
        return IdentityTranslator()


def build_baseline(kind: str = "dictionary"):
    if kind == "identity":
        return IdentityTranslator()
    return DictionaryTranslator()


__all__ = ["IdentityTranslator", "DictionaryTranslator", "build_baseline"]
