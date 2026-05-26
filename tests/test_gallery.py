"""Test della logica pura della galleria (no Qt)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.core.gallery import (
    REUSABLE_KEYS,
    GalleryItem,
    load_gallery,
    load_sidecar,
    remove_item,
    sidecar_path,
)


def _write_image(directory, name, meta=None, mtime=None):
    img = directory / name
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake png")
    if meta is not None:
        sidecar_path(img).write_text(json.dumps(meta), encoding="utf-8")
    if mtime is not None:
        os.utime(img, (mtime, mtime))
    return img


# ---------------------------------------------------------------------------
# Scansione e ordinamento
# ---------------------------------------------------------------------------


def test_load_gallery_missing_dir_is_empty(tmp_path):
    assert load_gallery(tmp_path / "nope") == []


def test_load_gallery_finds_images_only(tmp_path):
    _write_image(tmp_path, "a.png")
    _write_image(tmp_path, "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "a.json").write_text("{}")  # sidecar, non immagine
    items = load_gallery(tmp_path)
    names = {it.name for it in items}
    assert names == {"a.png", "b.jpg"}


def test_load_gallery_sorted_newest_first(tmp_path):
    _write_image(tmp_path, "old.png", mtime=time.time() - 1000)
    _write_image(tmp_path, "new.png", mtime=time.time())
    items = load_gallery(tmp_path)
    assert [it.name for it in items] == ["new.png", "old.png"]


def test_load_gallery_query_filters_by_prompt(tmp_path):
    _write_image(tmp_path, "a.png", meta={"positive": "a red dragon"})
    _write_image(tmp_path, "b.png", meta={"positive": "a blue cat"})
    items = load_gallery(tmp_path, query="dragon")
    assert [it.name for it in items] == ["a.png"]


def test_load_gallery_query_matches_filename(tmp_path):
    _write_image(tmp_path, "hero_shot.png", meta={"positive": "x"})
    _write_image(tmp_path, "other.png", meta={"positive": "y"})
    items = load_gallery(tmp_path, query="hero")
    assert [it.name for it in items] == ["hero_shot.png"]


# ---------------------------------------------------------------------------
# Metadati e accessori
# ---------------------------------------------------------------------------


def test_item_properties_from_metadata(tmp_path):
    _write_image(
        tmp_path, "x.png",
        meta={"positive": "p", "negative": "n", "seed": 123,
              "created_at": "2026-05-26T10:00:00+00:00"},
    )
    items = load_gallery(tmp_path)
    it = items[0]
    assert it.prompt == "p"
    assert it.negative == "n"
    assert it.seed == 123
    assert it.created_at.startswith("2026-05-26")
    assert it.has_metadata


def test_item_without_sidecar_has_empty_metadata(tmp_path):
    _write_image(tmp_path, "x.png")
    it = load_gallery(tmp_path)[0]
    assert not it.has_metadata
    assert it.seed is None
    assert it.prompt == ""


def test_seed_invalid_is_none():
    it = GalleryItem(path=Path("x.png"),
                     metadata={"seed": "abc"})
    assert it.seed is None


def test_caption_contains_key_fields():
    it = GalleryItem(
        path=Path("img.png"),
        metadata={"positive": "dragon", "width": 832, "height": 1216,
                  "seed": 42, "steps": 30, "sampler": "dpmpp_2m"},
    )
    cap = it.caption()
    assert "dragon" in cap
    assert "832×1216" in cap
    assert "42" in cap
    assert "dpmpp_2m" in cap


def test_caption_no_metadata():
    it = GalleryItem(path=Path("img.png"))
    assert "nessun metadato" in it.caption()


def test_reuse_params_subset():
    meta = {"positive": "p", "negative": "n", "width": 1024, "height": 1024,
            "steps": 25, "cfg": 6.0, "sampler": "euler", "seed": 7,
            "model_id": "pony", "dialect": "pony"}  # ultimi due non riusabili
    it = GalleryItem(path=Path("x.png"), metadata=meta)
    params = it.reuse_params()
    assert set(params) == set(REUSABLE_KEYS)
    assert "model_id" not in params
    assert params["seed"] == 7


# ---------------------------------------------------------------------------
# load_sidecar / remove_item
# ---------------------------------------------------------------------------


def test_load_sidecar_missing_returns_empty(tmp_path):
    img = _write_image(tmp_path, "x.png")
    assert load_sidecar(img) == {}


def test_load_sidecar_corrupt_returns_empty(tmp_path):
    img = _write_image(tmp_path, "x.png")
    sidecar_path(img).write_text("{ not json")
    assert load_sidecar(img) == {}


def test_remove_item_deletes_image_and_sidecar(tmp_path):
    img = _write_image(tmp_path, "x.png", meta={"positive": "p"})
    it = load_gallery(tmp_path)[0]
    remove_item(it)
    assert not img.exists()
    assert not sidecar_path(img).exists()
