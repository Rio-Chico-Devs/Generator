from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.core.project import Project, _slugify


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
