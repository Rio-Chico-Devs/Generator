"""Test del Decision Diary (Pilastro C) — no Qt."""
from __future__ import annotations

from pathlib import Path

from src.core.guidance.diary import (
    DiaryEntry,
    append_entry,
    diary_stats,
    load_diary,
)


def _entry(session="s1", step="colors", chosen=True, reason=None, regen=False, techs=None):
    return DiaryEntry(
        session_id=session,
        project_slug="iris",
        step_id=step,
        prompt="iris",
        chosen_path=Path("chosen.png") if chosen else None,
        rejected_paths=[Path("a.png"), Path("b.png")],
        rejection_reason=reason,
        was_regeneration=regen,
        technique_refs_used=techs or [],
    )


# ---------------------------------------------------------------------------
# DiaryEntry
# ---------------------------------------------------------------------------


def test_entry_is_approval():
    assert _entry(chosen=True).is_approval
    assert not _entry(chosen=False).is_approval


def test_entry_roundtrip():
    e = _entry(reason="troppo scuro", regen=True, techs=["r1", "r2"])
    restored = DiaryEntry.from_dict(e.to_dict())
    assert restored.session_id == e.session_id
    assert restored.step_id == e.step_id
    assert restored.rejection_reason == "troppo scuro"
    assert restored.was_regeneration is True
    assert restored.technique_refs_used == ["r1", "r2"]
    assert restored.chosen_path == e.chosen_path
    assert restored.rejected_paths == e.rejected_paths


def test_entry_rejected_chosen_path_none():
    e = _entry(chosen=False)
    assert e.to_dict()["chosen_path"] is None
    assert DiaryEntry.from_dict(e.to_dict()).chosen_path is None


# ---------------------------------------------------------------------------
# append / load
# ---------------------------------------------------------------------------


def test_load_missing_returns_empty(tmp_path):
    assert load_diary(tmp_path / "nope.jsonl") == []


def test_append_then_load(tmp_path):
    p = tmp_path / "diary.jsonl"
    append_entry(p, _entry(step="composition"))
    append_entry(p, _entry(step="colors", chosen=False))
    entries = load_diary(p)
    assert len(entries) == 2
    assert entries[0].step_id == "composition"
    assert entries[1].is_approval is False


def test_append_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "diary.jsonl"
    append_entry(p, _entry())
    assert p.exists()


def test_append_is_one_line_per_entry(tmp_path):
    p = tmp_path / "diary.jsonl"
    for _ in range(3):
        append_entry(p, _entry())
    assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_load_skips_malformed_lines(tmp_path):
    p = tmp_path / "diary.jsonl"
    append_entry(p, _entry())
    with p.open("a", encoding="utf-8") as fh:
        fh.write("{ not valid json\n")
        fh.write("\n")  # riga vuota
    append_entry(p, _entry(step="shading"))
    entries = load_diary(p)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# diary_stats
# ---------------------------------------------------------------------------


def test_stats_empty():
    s = diary_stats([])
    assert s["total_entries"] == 0
    assert s["approval_rate"] == 0.0
    assert s["most_rejected_step"] is None


def test_stats_approval_rate():
    entries = [_entry(chosen=True), _entry(chosen=True), _entry(chosen=False), _entry(chosen=False)]
    s = diary_stats(entries)
    assert s["total_entries"] == 4
    assert s["approval_rate"] == 0.5


def test_stats_counts_sessions():
    entries = [_entry(session="a"), _entry(session="a"), _entry(session="b")]
    assert diary_stats(entries)["total_sessions"] == 2


def test_stats_most_rejected_step():
    entries = [
        _entry(step="colors", chosen=False),
        _entry(step="colors", chosen=False),
        _entry(step="shading", chosen=False),
        _entry(step="composition", chosen=True),
    ]
    assert diary_stats(entries)["most_rejected_step"] == "colors"


def test_stats_regeneration_count():
    entries = [_entry(regen=True), _entry(regen=False), _entry(regen=True)]
    assert diary_stats(entries)["regeneration_count"] == 2


def test_stats_rejection_reasons_counter():
    entries = [
        _entry(chosen=False, reason="troppo scuro"),
        _entry(chosen=False, reason="troppo scuro"),
        _entry(chosen=False, reason="anatomia"),
    ]
    reasons = diary_stats(entries)["rejection_reasons"]
    assert reasons["troppo scuro"] == 2
    assert reasons["anatomia"] == 1


def test_stats_technique_usage():
    entries = [_entry(techs=["r1", "r2"]), _entry(techs=["r1"])]
    usage = diary_stats(entries)["technique_usage"]
    assert usage["r1"] == 2
    assert usage["r2"] == 1
