"""Test della logica pura del Dataset Inspector (no Qt)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.core.dataset_inspector import (
    DatasetItem,
    dataset_stats,
    remove_dataset_item,
    save_caption,
    scan_dataset,
    tag_frequency,
)


def _make_item(directory: Path, name: str, caption: str = "") -> DatasetItem:
    img = directory / name
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake png")
    txt = img.with_suffix(".txt")
    if caption:
        txt.write_text(caption, encoding="utf-8")
    return DatasetItem(image_path=img, caption_path=txt)


# ---------------------------------------------------------------------------
# scan_dataset
# ---------------------------------------------------------------------------


def test_scan_empty_dir_returns_empty(tmp_path):
    assert scan_dataset(tmp_path) == []


def test_scan_missing_dir_returns_empty(tmp_path):
    assert scan_dataset(tmp_path / "nope") == []


def test_scan_finds_images_only(tmp_path):
    _make_item(tmp_path, "a.png", "cat")
    _make_item(tmp_path, "b.jpg", "dog")
    (tmp_path / "notes.txt").write_text("not an image")
    items = scan_dataset(tmp_path)
    assert {it.name for it in items} == {"a.png", "b.jpg"}


def test_scan_sorted_by_name(tmp_path):
    _make_item(tmp_path, "c.png")
    _make_item(tmp_path, "a.png")
    _make_item(tmp_path, "b.png")
    names = [it.name for it in scan_dataset(tmp_path)]
    assert names == ["a.png", "b.png", "c.png"]


def test_scan_item_without_caption(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    items = scan_dataset(tmp_path)
    assert len(items) == 1
    assert items[0].caption == ""
    assert items[0].tags == []
    assert not items[0].has_caption


# ---------------------------------------------------------------------------
# DatasetItem properties
# ---------------------------------------------------------------------------


def test_tags_comma_separated(tmp_path):
    item = _make_item(tmp_path, "x.png", "masterpiece, 1girl, red hair")
    assert item.tags == ["masterpiece", "1girl", "red hair"]


def test_tags_strips_whitespace(tmp_path):
    item = _make_item(tmp_path, "x.png", "  tag1 ,  tag2  ")
    assert item.tags == ["tag1", "tag2"]


def test_tags_ignores_empty_slots(tmp_path):
    item = _make_item(tmp_path, "x.png", "tag1,,tag2,")
    assert item.tags == ["tag1", "tag2"]


def test_matches_tag_case_insensitive(tmp_path):
    item = _make_item(tmp_path, "x.png", "Red Hair, masterpiece")
    assert item.matches_tag("red hair")
    assert item.matches_tag("RED HAIR")
    assert not item.matches_tag("blue")


def test_matches_tag_substring(tmp_path):
    item = _make_item(tmp_path, "x.png", "long red hair")
    assert item.matches_tag("red")


# ---------------------------------------------------------------------------
# tag_frequency
# ---------------------------------------------------------------------------


def test_tag_frequency_counts_per_image(tmp_path):
    i1 = _make_item(tmp_path, "a.png", "cat, cute")
    i2 = _make_item(tmp_path, "b.png", "cat, fluffy")
    i3 = _make_item(tmp_path, "c.png", "dog")
    freq = tag_frequency([i1, i2, i3])
    assert freq["cat"] == 2
    assert freq["cute"] == 1
    assert freq["dog"] == 1


def test_tag_frequency_counts_tag_once_per_image(tmp_path):
    # Anche se il tag appare più volte nella caption, conta 1 per immagine
    item = _make_item(tmp_path, "x.png", "cat, cat, cat")
    freq = tag_frequency([item])
    assert freq["cat"] == 1


def test_tag_frequency_empty_returns_empty(tmp_path):
    item = _make_item(tmp_path, "x.png")
    assert tag_frequency([item]) == Counter()


# ---------------------------------------------------------------------------
# dataset_stats
# ---------------------------------------------------------------------------


def test_dataset_stats_basic(tmp_path):
    _make_item(tmp_path, "a.png", "cat, cute")
    _make_item(tmp_path, "b.png", "dog")
    _make_item(tmp_path, "c.png")  # senza caption
    items = scan_dataset(tmp_path)
    s = dataset_stats(items)
    assert s["total_images"] == 3
    assert s["images_without_caption"] == 1
    assert s["unique_tags"] == 3  # cat, cute, dog
    assert s["avg_tags_per_image"] == 1.0  # (2+1+0)/3


def test_dataset_stats_empty():
    s = dataset_stats([])
    assert s["total_images"] == 0
    assert s["avg_tags_per_image"] == 0.0


# ---------------------------------------------------------------------------
# save_caption / remove_dataset_item
# ---------------------------------------------------------------------------


def test_save_caption_creates_file(tmp_path):
    item = _make_item(tmp_path, "x.png")
    assert not item.caption_path.exists()
    save_caption(item, "  masterpiece, 1girl  ")
    assert item.caption_path.read_text(encoding="utf-8") == "masterpiece, 1girl"


def test_save_caption_updates_existing(tmp_path):
    item = _make_item(tmp_path, "x.png", "old tag")
    save_caption(item, "new tag")
    assert item.caption_path.read_text(encoding="utf-8") == "new tag"


def test_save_caption_empty_removes_file(tmp_path):
    item = _make_item(tmp_path, "x.png", "some tag")
    save_caption(item, "")
    assert not item.caption_path.exists()


def test_save_caption_empty_on_nonexistent_is_noop(tmp_path):
    item = _make_item(tmp_path, "x.png")
    save_caption(item, "")  # non deve sollevare
    assert not item.caption_path.exists()


def test_remove_dataset_item_deletes_both(tmp_path):
    item = _make_item(tmp_path, "x.png", "tag")
    remove_dataset_item(item)
    assert not item.image_path.exists()
    assert not item.caption_path.exists()


def test_remove_dataset_item_without_caption_is_ok(tmp_path):
    item = _make_item(tmp_path, "x.png")
    remove_dataset_item(item)  # non deve sollevare
    assert not item.image_path.exists()
