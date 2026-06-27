"""OCR: the SeedEngine reconstructs words+boxes from an embedded gold spec; layout grouping."""

from __future__ import annotations

import json

from imgtrans.config import OcrConfig
from imgtrans.models.ocr_engine import OcrResult, SeedEngine, StubEngine, load_ocr_engine


class _FakeImage:
    """Minimal stand-in carrying an `info` dict (no PIL needed)."""

    def __init__(self, spec, size=(1000, 400)):
        self.info = {"imgtrans_spec": json.dumps(spec)}
        self.size = size


def _spec():
    return {"width": 1000, "height": 400, "src_lang": "en", "tgt_lang": "fr",
            "blocks": [{"text": "Hello world", "translation": "Bonjour le monde",
                        "bbox": [40, 40, 300, 40], "block": 0, "line": 0}]}


def test_seed_engine_reads_spec():
    eng = SeedEngine(OcrConfig())
    res = eng.recognize(_FakeImage(_spec()))
    assert isinstance(res, OcrResult)
    assert res.engine == "seed"
    texts = [w.text for w in res.words]
    assert texts == ["Hello", "world"]
    assert all(len(w.bbox) == 4 for w in res.words)
    assert res.mean_conf > 0.5


def test_load_ocr_engine_prefers_seed_when_spec_present():
    eng = load_ocr_engine(OcrConfig(), image=_FakeImage(_spec()))
    assert eng.name == "seed"


def test_stub_on_blank_image():
    eng = StubEngine(OcrConfig())
    res = eng.recognize(_FakeImage({"blocks": []}))
    assert res.words == []


def test_seed_noise_changes_text():
    clean = SeedEngine(OcrConfig(), noise=0.0).recognize(_FakeImage(_spec())).full_text
    noisy = SeedEngine(OcrConfig(), noise=0.9, seed=3).recognize(_FakeImage(_spec())).full_text
    assert clean != noisy  # noise actually perturbs the OCR text
