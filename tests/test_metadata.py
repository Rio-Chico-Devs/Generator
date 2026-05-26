from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PIL", reason="metadata richiede Pillow")

from PIL import Image  # noqa: E402

from src.utils.metadata import (  # noqa: E402
    build_a1111_parameters,
    embed_a1111_parameters,
    read_a1111_parameters,
    save_image_with_metadata,
)


def _record_dict() -> dict:
    return {
        "positive": "hello world",
        "negative": "bad",
        "steps": 25,
        "sampler": "dpmpp_2m",
        "cfg": 6.5,
        "seed": 42,
        "width": 832,
        "height": 1216,
        "model_id": "pony-v6-xl",
        "clip_skip": -2,
        "lora_name": "iris.safetensors",
        "lora_weight": 0.85,
    }


def test_a1111_string_contains_core_fields():
    s = build_a1111_parameters(_record_dict())
    assert "hello world" in s
    assert "Negative prompt: bad" in s
    assert "Steps: 25" in s
    assert "Seed: 42" in s
    assert "Size: 832x1216" in s
    assert "pony-v6-xl" in s
    # clip_skip viene normalizzato a valore positivo nello stile A1111
    assert "Clip skip: 2" in s
    assert "Lora: iris.safetensors:0.85" in s


def test_a1111_string_minimal_record():
    s = build_a1111_parameters({"positive": "solo prompt"})
    assert s.startswith("solo prompt")
    assert "Negative prompt:" not in s
    assert "Vihente Forge" in s


def test_png_metadata_roundtrip():
    text = build_a1111_parameters({"positive": "roundtrip", "seed": 7})
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    out = Path(tempfile.mkdtemp(prefix="vf_meta_")) / "x.png"
    save_image_with_metadata(img, out, text)
    assert out.exists()
    assert read_a1111_parameters(out) == text


def test_embed_into_existing_png_preserves_pixels_and_adds_params():
    out = Path(tempfile.mkdtemp(prefix="vf_meta_")) / "existing.png"
    Image.new("RGB", (24, 24), (1, 2, 3)).save(out)  # PNG senza metadati
    text = build_a1111_parameters(_record_dict())
    assert embed_a1111_parameters(out, text) is True
    assert read_a1111_parameters(out) == text
    # i pixel restano leggibili dopo il riscrittura
    assert Image.open(out).size == (24, 24)


def test_embed_non_png_returns_false():
    out = Path(tempfile.mkdtemp(prefix="vf_meta_")) / "not.txt"
    out.write_text("x")
    assert embed_a1111_parameters(out, "params") is False
