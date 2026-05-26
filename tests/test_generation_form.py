"""Test della logica pura del form di generazione (no Qt)."""
from __future__ import annotations

from src.core.generation_form import (
    SAMPLERS,
    SCHEDULERS,
    SIZE_PRESETS,
    GenerationForm,
    TokenStatus,
    approx_token_count,
    preset_for,
    token_status,
)


# ---------------------------------------------------------------------------
# Conteggio token
# ---------------------------------------------------------------------------


def test_empty_prompt_zero_tokens():
    assert approx_token_count("") == 0
    assert approx_token_count("   ") == 0


def test_token_count_words_and_punctuation():
    # 4 parole + 2 virgole = 6 token
    assert approx_token_count("masterpiece, best quality, 1girl") == 6


def test_token_status_levels():
    assert token_status("a b c").level == "ok"
    long_ok = ", ".join(["tag"] * 40)  # ~ sotto 75
    assert token_status(long_ok).level in ("ok", "warn")
    over = ", ".join(["tag"] * 120)  # ben oltre 75
    assert token_status(over).level in ("warn", "over")


def test_token_status_chunks():
    assert TokenStatus(count=0).chunks == 0
    assert TokenStatus(count=10).chunks == 1
    assert TokenStatus(count=75).chunks == 1
    assert TokenStatus(count=76).chunks == 2
    assert TokenStatus(count=150).chunks == 2
    assert TokenStatus(count=151).chunks == 3


def test_token_label():
    assert "token" in token_status("ciao").label()


# ---------------------------------------------------------------------------
# Preset dimensioni e liste
# ---------------------------------------------------------------------------


def test_size_presets_are_sdxl_ratios():
    # tutti vicini a 1 megapixel
    for p in SIZE_PRESETS:
        mp = (p.width * p.height) / 1_000_000
        assert 0.9 <= mp <= 1.1


def test_preset_for_matches_and_custom():
    assert preset_for(1024, 1024).key == "square"
    assert preset_for(832, 1216).key == "portrait"
    assert preset_for(500, 700) is None  # custom


def test_sampler_scheduler_lists_nonempty():
    assert "euler" in SAMPLERS
    assert "karras" in SCHEDULERS


# ---------------------------------------------------------------------------
# Costruzione parametri + config
# ---------------------------------------------------------------------------


def test_to_user_params_includes_all_keys():
    form = GenerationForm(prompt="x", negative="y", width=832, height=1216,
                          steps=25, cfg=6.5, seed=42, batch=3)
    p = form.to_user_params()
    assert p["prompt"] == "x"
    assert p["negative"] == "y"
    assert p["width"] == 832 and p["height"] == 1216
    assert p["steps"] == 25
    assert p["cfg"] == 6.5
    assert p["seed"] == 42
    assert p["batch"] == 3
    assert "sampler" in p and "scheduler" in p


def test_to_generation_config_maps_fields():
    form = GenerationForm(prompt="p", width=1024, height=1024, steps=20,
                          cfg=5.0, batch=2, extra_lora_count=2)
    cfg = form.to_generation_config()
    assert cfg.prompt == "p"
    assert cfg.width == 1024 and cfg.height == 1024
    assert cfg.steps == 20
    assert cfg.batch_size == 2
    assert len(cfg.loras) == 2


def test_estimate_reflects_settings():
    cheap = GenerationForm(width=512, height=512, steps=10, batch=1)
    expensive = GenerationForm(width=1024, height=1024, steps=40, batch=2,
                               extra_lora_count=3, upscale_factor=2.0,
                               use_ipadapter=True)
    assert expensive.estimate().total > cheap.estimate().total


def test_estimate_batch_multiplies():
    one = GenerationForm(width=1024, height=1024, steps=20, batch=1)
    four = GenerationForm(width=1024, height=1024, steps=20, batch=4)
    assert abs(four.estimate().total - one.estimate().per_image * 4) < 0.01


def test_upscale_increases_cost():
    base = GenerationForm(width=1024, height=1024, steps=20)
    up = GenerationForm(width=1024, height=1024, steps=20, upscale_factor=2.0)
    assert up.estimate().per_image > base.estimate().per_image
