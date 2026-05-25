"""Test per core/prompting: preset di dialetto, dedup, composizione prompt."""
from __future__ import annotations

from src.core.prompting import (
    DEFAULT_DIALECT,
    PRESETS,
    build_prompts,
    get_preset,
    merge_tags,
)


def test_all_presets_have_consistent_dialect_key():
    for key, preset in PRESETS.items():
        assert key == preset.dialect
        assert preset.clip_skip >= 1


def test_get_preset_unknown_falls_back_to_default():
    assert get_preset("nonexistent").dialect == DEFAULT_DIALECT
    assert get_preset(None).dialect == DEFAULT_DIALECT


def test_pony_preset_has_score_tags_and_clip_skip_2():
    p = get_preset("pony")
    assert "score_9" in p.positive_prefix
    assert p.clip_skip == 2


def test_illustrious_preset_no_score_tags_clip_skip_2():
    p = get_preset("illustrious")
    # Illustrious NON usa i tag score_* di Pony.
    assert "score_9" not in p.positive_prefix
    assert "masterpiece" in p.positive_prefix
    assert p.clip_skip == 2


def test_merge_tags_dedups_case_insensitive_preserving_order():
    out = merge_tags("a, B, c", "b, D, A")
    assert out == "a, B, c, D"


def test_merge_tags_skips_empty_chunks_and_blanks():
    assert merge_tags("", "x, , y", None or "") == "x, y"


def test_build_prompts_orders_quality_then_trigger_then_user():
    preset = get_preset("illustrious")
    b = build_prompts(
        preset,
        user_positive="1girl, forest",
        trigger_prefix="vf_iris_v1,",
        project_negative="ugly",
    )
    # quality-prefix prima, poi trigger, poi prompt utente
    assert b.positive.startswith("masterpiece")
    assert b.positive.index("vf_iris_v1") < b.positive.index("1girl")
    # negative: base (project) seguito dai quality-negative del preset
    assert b.negative.startswith("ugly")
    assert "worst quality" in b.negative
    assert b.clip_skip == 2


def test_build_prompts_user_negative_overrides_project_negative():
    preset = get_preset("pony")
    b = build_prompts(
        preset,
        user_positive="x",
        user_negative="my custom neg",
        project_negative="project default",
    )
    assert b.negative.startswith("my custom neg")
    assert "project default" not in b.negative


def test_build_prompts_dedups_quality_tags_with_trigger():
    preset = get_preset("illustrious")  # prefix contiene "masterpiece, best quality"
    b = build_prompts(
        preset,
        user_positive="scene",
        trigger_prefix="vf_x, masterpiece, best quality,",
    )
    # "masterpiece" deve comparire una sola volta nonostante sia in entrambi
    assert b.positive.lower().count("masterpiece") == 1
