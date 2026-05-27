"""Test del rilevamento stato 'esaurito' (Pilastro C) — no Qt."""
from __future__ import annotations

from pathlib import Path

from src.core.guidance.diary import DiaryEntry
from src.core.guidance.exhaustion import (
    check_exhaustion,
    consecutive_rejections,
    diagnose,
)


def _e(session, step, chosen):
    return DiaryEntry(
        session_id=session,
        project_slug="iris",
        step_id=step,
        prompt="iris",
        chosen_path=Path("c.png") if chosen else None,
    )


# ---------------------------------------------------------------------------
# consecutive_rejections
# ---------------------------------------------------------------------------


def test_no_entries_zero():
    assert consecutive_rejections([], "s1", "colors") == 0


def test_counts_only_matching_session_and_step():
    entries = [
        _e("s1", "colors", False),
        _e("s2", "colors", False),    # altra sessione
        _e("s1", "shading", False),   # altro step
        _e("s1", "colors", False),
    ]
    assert consecutive_rejections(entries, "s1", "colors") == 2


def test_approval_resets_streak():
    entries = [
        _e("s1", "colors", False),
        _e("s1", "colors", False),
        _e("s1", "colors", True),     # azzera
        _e("s1", "colors", False),
    ]
    assert consecutive_rejections(entries, "s1", "colors") == 1


def test_trailing_approval_means_zero():
    entries = [
        _e("s1", "colors", False),
        _e("s1", "colors", True),
    ]
    assert consecutive_rejections(entries, "s1", "colors") == 0


# ---------------------------------------------------------------------------
# check_exhaustion
# ---------------------------------------------------------------------------


def test_not_exhausted_below_threshold():
    entries = [_e("s1", "colors", False), _e("s1", "colors", False)]
    report = check_exhaustion(entries, "s1", "colors", threshold=3)
    assert report.exhausted is False
    assert report.consecutive_rejections == 2


def test_exhausted_at_threshold():
    entries = [_e("s1", "colors", False) for _ in range(3)]
    report = check_exhaustion(entries, "s1", "colors", threshold=3)
    assert report.exhausted is True
    assert report.consecutive_rejections == 3


def test_exhausted_above_threshold():
    entries = [_e("s1", "colors", False) for _ in range(5)]
    report = check_exhaustion(entries, "s1", "colors", threshold=3)
    assert report.exhausted is True


def test_default_threshold_is_three():
    entries = [_e("s1", "colors", False) for _ in range(3)]
    assert check_exhaustion(entries, "s1", "colors").exhausted is True


def test_report_carries_metadata():
    report = check_exhaustion([], "s1", "colors", threshold=4)
    assert report.step_id == "colors"
    assert report.threshold == 4


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


def test_diagnose_mentions_step():
    report = check_exhaustion([_e("s1", "colors", False)] * 3, "s1", "colors")
    msg = diagnose(report)
    assert "colors" in msg
    assert "Libreria Tecniche" in msg
