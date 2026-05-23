from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.core.config import GenerationConfig

pytest.importorskip("PIL", reason="metadata richiede Pillow")

from PIL import Image  # noqa: E402

from src.utils.metadata import (  # noqa: E402
    build_a1111_parameters_string,
    read_a1111_parameters,
    save_image_with_metadata,
)


def test_a1111_string_contains_core_fields():
    cfg = GenerationConfig(
        prompt="hello world",
        negative_prompt="bad",
        steps=25,
        cfg_scale=6.5,
        width=832,
        height=1216,
        model_id="pony-v6-xl",
    )
    s = build_a1111_parameters_string(cfg, seed=42)
    assert "hello world" in s
    assert "Negative prompt: bad" in s
    assert "Steps: 25" in s
    assert "Seed: 42" in s
    assert "Size: 832x1216" in s
    assert "pony-v6-xl" in s


def test_png_metadata_roundtrip():
    cfg = GenerationConfig(prompt="roundtrip")
    text = build_a1111_parameters_string(cfg, seed=7)
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    out = Path(tempfile.mkdtemp(prefix="vf_meta_")) / "x.png"
    save_image_with_metadata(img, out, text)
    assert out.exists()
    assert read_a1111_parameters(out) == text
