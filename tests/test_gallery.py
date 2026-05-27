"""Test della logica pura della galleria (no Qt)."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from src.core.gallery import (
    RATING_KEEP,
    RATING_REJECT,
    RATING_UNTAGGED,
    RATING_VARIANT,
    REUSABLE_KEYS,
    GalleryItem,
    add_to_dataset,
    load_gallery,
    load_sidecar,
    remove_item,
    set_rating,
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


# ---------------------------------------------------------------------------
# Giudizio (rating): set_rating, property e filtro
# ---------------------------------------------------------------------------


def test_rating_property_reads_valid_value():
    it = GalleryItem(path=Path("x.png"), metadata={"rating": RATING_KEEP})
    assert it.rating == RATING_KEEP


def test_rating_property_ignores_unknown_value():
    it = GalleryItem(path=Path("x.png"), metadata={"rating": "bogus"})
    assert it.rating is None


def test_rating_property_none_when_absent():
    assert GalleryItem(path=Path("x.png")).rating is None


def test_set_rating_persists_to_sidecar(tmp_path):
    img = _write_image(tmp_path, "x.png", meta={"positive": "p"})
    set_rating(img, RATING_VARIANT)
    assert load_sidecar(img)["rating"] == RATING_VARIANT
    # gli altri parametri restano intatti
    assert load_sidecar(img)["positive"] == "p"


def test_set_rating_none_clears(tmp_path):
    img = _write_image(tmp_path, "x.png", meta={"positive": "p", "rating": RATING_REJECT})
    set_rating(img, None)
    assert "rating" not in load_sidecar(img)
    assert load_sidecar(img)["positive"] == "p"


def test_set_rating_invalid_raises(tmp_path):
    img = _write_image(tmp_path, "x.png")
    with pytest.raises(ValueError):
        set_rating(img, "nope")


def test_set_rating_creates_sidecar_when_absent(tmp_path):
    img = _write_image(tmp_path, "x.png")  # nessun sidecar
    assert not sidecar_path(img).exists()
    set_rating(img, RATING_KEEP)
    assert load_sidecar(img)["rating"] == RATING_KEEP


def test_set_rating_none_does_not_create_empty_sidecar(tmp_path):
    img = _write_image(tmp_path, "x.png")  # nessun sidecar
    set_rating(img, None)
    assert not sidecar_path(img).exists()


def test_load_gallery_filters_by_rating(tmp_path):
    _write_image(tmp_path, "k.png", meta={"rating": RATING_KEEP})
    _write_image(tmp_path, "v.png", meta={"rating": RATING_VARIANT})
    _write_image(tmp_path, "r.png", meta={"rating": RATING_REJECT})
    _write_image(tmp_path, "u.png", meta={"positive": "x"})  # non valutata
    assert [it.name for it in load_gallery(tmp_path, rating_filter=RATING_KEEP)] == ["k.png"]
    assert [it.name for it in load_gallery(tmp_path, rating_filter=RATING_VARIANT)] == ["v.png"]


def test_load_gallery_untagged_filter(tmp_path):
    _write_image(tmp_path, "k.png", meta={"rating": RATING_KEEP})
    _write_image(tmp_path, "u.png", meta={"positive": "x"})
    _write_image(tmp_path, "bare.png")  # senza sidecar → non valutata
    names = {it.name for it in load_gallery(tmp_path, rating_filter=RATING_UNTAGGED)}
    assert names == {"u.png", "bare.png"}


def test_load_gallery_no_filter_returns_all(tmp_path):
    _write_image(tmp_path, "k.png", meta={"rating": RATING_KEEP})
    _write_image(tmp_path, "u.png", meta={"positive": "x"})
    assert len(load_gallery(tmp_path)) == 2


def test_caption_includes_rating_label():
    it = GalleryItem(path=Path("x.png"), metadata={"positive": "p", "rating": RATING_KEEP})
    assert "Coerente" in it.caption()


# ---------------------------------------------------------------------------
# add_to_dataset
# ---------------------------------------------------------------------------


def test_add_to_dataset_copies_image(tmp_path):
    src_dir = tmp_path / "gallery"
    src_dir.mkdir()
    dst_dir = tmp_path / "dataset"
    img = _write_image(src_dir, "img.png", meta={"positive": "a red cat"})
    it = load_gallery(src_dir)[0]
    dest = add_to_dataset(it, dst_dir)
    assert dest.exists()
    assert dest.name == "img.png"
    assert dest.parent == dst_dir


def test_add_to_dataset_writes_caption(tmp_path):
    src_dir = tmp_path / "gallery"
    src_dir.mkdir()
    dst_dir = tmp_path / "dataset"
    img = _write_image(src_dir, "img.png", meta={"positive": "a red cat"})
    it = load_gallery(src_dir)[0]
    dest = add_to_dataset(it, dst_dir)
    caption_file = dest.with_suffix(".txt")
    assert caption_file.exists()
    assert caption_file.read_text(encoding="utf-8") == "a red cat"


def test_add_to_dataset_no_caption_when_no_prompt(tmp_path):
    src_dir = tmp_path / "gallery"
    src_dir.mkdir()
    dst_dir = tmp_path / "dataset"
    img = _write_image(src_dir, "img.png")
    it = load_gallery(src_dir)[0]
    dest = add_to_dataset(it, dst_dir)
    assert not dest.with_suffix(".txt").exists()


def test_add_to_dataset_deconflicts_name(tmp_path):
    src_dir = tmp_path / "gallery"
    src_dir.mkdir()
    dst_dir = tmp_path / "dataset"
    dst_dir.mkdir()
    # pre-occupa il nome
    (dst_dir / "img.png").write_bytes(b"existing")
    img = _write_image(src_dir, "img.png", meta={"positive": "p"})
    it = load_gallery(src_dir)[0]
    dest = add_to_dataset(it, dst_dir)
    assert dest.name == "img_2.png"
    assert dest.exists()
    assert (dst_dir / "img.png").read_bytes() == b"existing"  # originale intatto


def test_add_to_dataset_creates_dst_dir(tmp_path):
    src_dir = tmp_path / "gallery"
    src_dir.mkdir()
    dst_dir = tmp_path / "nonexistent" / "dataset"
    img = _write_image(src_dir, "img.png")
    it = load_gallery(src_dir)[0]
    dest = add_to_dataset(it, dst_dir)
    assert dest.exists()
