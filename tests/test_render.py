"""Render: fit-to-box (binary search + wrap) and layout fidelity, with the no-PIL estimate."""

from __future__ import annotations

from imgtrans.config import RenderConfig
from imgtrans.imaging import render as R


def test_fit_shrinks_for_long_text():
    cfg = RenderConfig()
    box = (0, 0, 200, 40)
    short = R.fit_text_to_box("Hi", box, cfg)
    long = R.fit_text_to_box("This is a much much much longer line of translated text", box, cfg)
    assert short.font_size >= long.font_size
    assert long.font_size >= cfg.min_font_size


def test_wrap_produces_multiple_lines():
    cfg = RenderConfig()
    res = R.fit_text_to_box("word " * 40, (0, 0, 120, 200), cfg)
    assert len(res.lines) >= 2


def test_layout_fidelity_on_blocks():
    from imgtrans.imaging.layout import TextBlock
    cfg = RenderConfig()
    blocks = [
        TextBlock(text="Hello", bbox=(0, 0, 300, 50), translation="Bonjour"),
        TextBlock(text="World", bbox=(0, 60, 300, 50), translation="Le monde"),
    ]
    fid = R.layout_fidelity(blocks, cfg)
    assert fid["n_blocks"] == 2
    assert 0.0 <= fid["fit_rate"] <= 1.0
    assert "mean_shrink" in fid


def test_fit_estimate_path_no_pil():
    cfg = RenderConfig()
    res = R._fit_estimate("Bonjour le monde", (0, 0, 200, 40), cfg)
    assert res.method == "estimate"
    assert cfg.min_font_size <= res.font_size <= cfg.max_font_size
