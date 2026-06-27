"""Config: defaults, YAML round-trip, unknown-key tolerance, env paths."""

from __future__ import annotations

import os


def test_defaults(cfg):
    assert cfg.mt.base_model == "facebook/m2m100_418M"
    assert cfg.mt.src_lang == "en" and cfg.mt.tgt_lang == "fr"
    assert cfg.render.mode in ("overlay", "side_by_side", "text_only")
    assert 0.0 <= cfg.agent.ocr_confidence_min <= 1.0


def test_yaml_roundtrip(tmp_path, cfg):
    from imgtrans.config import load_config, save_config
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    again = load_config(p)
    assert again.mt.base_model == cfg.mt.base_model
    assert again.agent.min_fit_rate == cfg.agent.min_fit_rate


def test_unknown_keys_ignored(tmp_path):
    from imgtrans.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("mt:\n  base_model: x\n  bogus_key: 1\nzzz: 2\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.mt.base_model == "x"


def test_env_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("IMGTRANS_ARTIFACTS_DIR", str(tmp_path / "arts"))
    from imgtrans import config
    assert str(tmp_path / "arts") in str(config.artifacts_dir())
    dirs = config.ensure_dirs()
    assert dirs["models"].exists()
