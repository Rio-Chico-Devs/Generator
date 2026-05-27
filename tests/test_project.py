from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.core.project import ActiveLora, BaseModelRef, Project, _slugify


def _tmp_parent() -> Path:
    return Path(tempfile.mkdtemp(prefix="vf_proj_"))


def test_slugify():
    assert _slugify("Sprite Gothic!") == "sprite-gothic"
    assert _slugify("  Multiple   Spaces  ") == "multiple-spaces"
    assert _slugify("") == "untitled"
    assert _slugify("---") == "untitled"


def test_create_project_structure_and_load():
    parent = _tmp_parent()
    p = Project.create(name="Iris Test", description="desc", parent_dir=parent)
    assert p.slug == "iris-test"
    assert p.activator_tag == "vf_iris_test_v1"
    assert p.root.exists()
    assert p.project_json_path.exists()
    for d in (
        p.dataset_images_dir,
        p.training_runs_dir,
        p.generations_dir,
        p.gallery_dir,
        p.prompts_dir,
    ):
        assert d.is_dir()
    assert p.dataset_manifest_path.exists()

    loaded = Project.load(p.root)
    assert loaded.name == "Iris Test"
    assert loaded.slug == p.slug
    assert loaded.activator_tag == p.activator_tag


def test_create_deconflicts_duplicate_names():
    parent = _tmp_parent()
    p1 = Project.create(name="Same", parent_dir=parent)
    p2 = Project.create(name="Same", parent_dir=parent)
    assert p1.slug != p2.slug
    assert p2.slug.startswith("same")


def test_save_updates_description_and_timestamp():
    parent = _tmp_parent()
    p = Project.create(name="Tstamp", parent_dir=parent)
    p.description = "changed"
    p.save()
    data = json.loads(p.project_json_path.read_text(encoding="utf-8"))
    assert data["description"] == "changed"
    assert data["updated_at"]


def test_load_missing_raises():
    parent = _tmp_parent()
    raised = False
    try:
        Project.load(parent / "does-not-exist")
    except FileNotFoundError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# fork (progetto variante)
# ---------------------------------------------------------------------------


def test_fork_inherits_and_records_lineage():
    parent = _tmp_parent()
    p = Project.create(
        name="Stile Base",
        base_model=BaseModelRef(id="pony", name="Pony V6"),
        parent_dir=parent,
    )
    p.default_negative_prompt = "blurry, deformed"
    p.active_lora = ActiveLora(run_id="run1", checkpoint="ckpt-2000", weight=0.9)
    p.save()

    child = p.fork(name="Stile Variante")
    assert child.parent_slug == p.slug
    assert child.slug != p.slug
    assert child.base_model is not None
    assert child.base_model.id == "pony"
    assert child.default_negative_prompt == "blurry, deformed"
    assert child.active_lora is not None
    assert child.active_lora.checkpoint == "ckpt-2000"
    # Il figlio ha un proprio tag attivatore, distinto dal genitore
    assert child.activator_tag != p.activator_tag


def test_fork_creates_independent_copies():
    parent = _tmp_parent()
    p = Project.create(name="Base", parent_dir=parent)
    p.active_lora = ActiveLora(run_id="run1", checkpoint="ckpt", weight=0.9)
    child = p.fork(name="Variante")
    # Mutare i parametri del figlio non tocca il genitore
    child.default_generation_params.steps = 99
    assert p.default_generation_params.steps != 99
    child.active_lora.weight = 0.1
    assert p.active_lora.weight == 0.9


def test_fork_persists_parent_slug():
    parent = _tmp_parent()
    p = Project.create(name="Base", parent_dir=parent)
    child = p.fork(name="Variante")
    reloaded = Project.load(child.root)
    assert reloaded.parent_slug == p.slug


def test_fork_starts_with_empty_dataset():
    parent = _tmp_parent()
    p = Project.create(name="Base", parent_dir=parent)
    child = p.fork(name="Variante")
    assert child.dataset_images_dir.is_dir()
    assert list(child.dataset_images_dir.iterdir()) == []
