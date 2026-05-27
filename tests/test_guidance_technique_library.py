"""Test della Libreria Tecniche (Pilastro B) — no Qt."""
from __future__ import annotations

from pathlib import Path

from src.core.guidance.session import StepDefinition
from src.core.guidance.technique_library import (
    TechniqueLibrary,
    TechniqueRef,
)


def _ref(slot="skin", label="", weight=1.0) -> TechniqueRef:
    return TechniqueRef(slot=slot, image_path=Path(f"{slot}.png"), label=label, weight=weight)


def _step(slots) -> StepDefinition:
    return StepDefinition(id="x", label="X", denoise_strength=0.5, technique_slots=slots)


# ---------------------------------------------------------------------------
# TechniqueRef
# ---------------------------------------------------------------------------


def test_ref_roundtrip():
    r = _ref(slot="shadow", label="ombre dure", weight=0.8)
    restored = TechniqueRef.from_dict(r.to_dict())
    assert restored.ref_id == r.ref_id
    assert restored.slot == "shadow"
    assert restored.label == "ombre dure"
    assert restored.weight == 0.8


def test_ref_generates_unique_ids():
    assert _ref().ref_id != _ref().ref_id


# ---------------------------------------------------------------------------
# by_slot / add / remove / get
# ---------------------------------------------------------------------------


def test_by_slot_filters():
    lib = TechniqueLibrary(project_slug="iris", refs=[_ref("skin"), _ref("hair"), _ref("skin")])
    assert len(lib.by_slot("skin")) == 2
    assert len(lib.by_slot("hair")) == 1
    assert lib.by_slot("background") == []


def test_slots_in_use():
    lib = TechniqueLibrary(project_slug="iris", refs=[_ref("skin"), _ref("hair")])
    assert lib.slots_in_use() == {"skin", "hair"}


def test_add():
    lib = TechniqueLibrary(project_slug="iris")
    lib.add(_ref("skin"))
    assert len(lib.refs) == 1


def test_remove_existing():
    r = _ref("skin")
    lib = TechniqueLibrary(project_slug="iris", refs=[r])
    assert lib.remove(r.ref_id) is True
    assert lib.refs == []


def test_remove_missing():
    lib = TechniqueLibrary(project_slug="iris", refs=[_ref("skin")])
    assert lib.remove("nope") is False
    assert len(lib.refs) == 1


def test_get():
    r = _ref("hair")
    lib = TechniqueLibrary(project_slug="iris", refs=[r])
    assert lib.get(r.ref_id) is r
    assert lib.get("nope") is None


# ---------------------------------------------------------------------------
# active_for_step
# ---------------------------------------------------------------------------


def test_active_for_step_empty_slots_returns_none():
    lib = TechniqueLibrary(project_slug="iris", refs=[_ref("skin")])
    assert lib.active_for_step(_step([])) == []


def test_active_for_step_matches_slots():
    lib = TechniqueLibrary(
        project_slug="iris",
        refs=[_ref("skin"), _ref("hair"), _ref("shadow")],
    )
    active = lib.active_for_step(_step(["skin", "hair"]))
    assert {r.slot for r in active} == {"skin", "hair"}


def test_active_for_step_ignores_absent_slots():
    lib = TechniqueLibrary(project_slug="iris", refs=[_ref("skin")])
    assert lib.active_for_step(_step(["background"])) == []


# ---------------------------------------------------------------------------
# Persistenza
# ---------------------------------------------------------------------------


def test_load_missing_returns_empty(tmp_path):
    lib = TechniqueLibrary.load(tmp_path / "nope.json")
    assert lib.refs == []


def test_save_load_roundtrip(tmp_path):
    lib = TechniqueLibrary(
        project_slug="iris",
        refs=[_ref("skin", "ambra"), _ref("hair", "castano")],
    )
    path = tmp_path / "library.json"
    lib.save(path)
    loaded = TechniqueLibrary.load(path)
    assert loaded.project_slug == "iris"
    assert len(loaded.refs) == 2
    assert {r.label for r in loaded.refs} == {"ambra", "castano"}


def test_save_creates_parent_dirs(tmp_path):
    lib = TechniqueLibrary(project_slug="iris")
    path = tmp_path / "techniques" / "library.json"
    lib.save(path)
    assert path.exists()
