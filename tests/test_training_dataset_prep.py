"""Tests per src/training/dataset_prep.py."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.training.dataset_prep import (
    DatasetReport,
    _safe_tag,
    prepare_dataset,
)


# ---------------------------------------------------------------------------
# DatasetReport
# ---------------------------------------------------------------------------


class TestDatasetReport:
    def test_is_ready_false_when_no_images(self):
        r = DatasetReport()
        assert not r.is_ready

    def test_is_ready_false_below_minimum(self):
        r = DatasetReport(valid_images=4)
        assert not r.is_ready

    def test_is_ready_true_at_minimum(self):
        r = DatasetReport(valid_images=5)
        assert r.is_ready

    def test_readiness_message_no_images(self):
        r = DatasetReport(valid_images=0)
        assert "nessuna" in r.readiness_message.lower()

    def test_readiness_message_below_min(self):
        r = DatasetReport(valid_images=3)
        assert "3" in r.readiness_message
        assert "almeno" in r.readiness_message.lower()

    def test_readiness_message_ready(self):
        r = DatasetReport(valid_images=10)
        assert "10" in r.readiness_message
        assert "pronte" in r.readiness_message.lower()

    def test_default_repeats_is_1(self):
        r = DatasetReport()
        assert r.repeats == 1

    def test_default_warnings_empty(self):
        r = DatasetReport()
        assert r.warnings == []


# ---------------------------------------------------------------------------
# _safe_tag
# ---------------------------------------------------------------------------


class TestSafeTag:
    def test_simple_tag(self):
        assert _safe_tag("vf_iris_v1") == "vf_iris_v1"

    def test_spaces_replaced(self):
        assert " " not in _safe_tag("my tag name")

    def test_slashes_replaced(self):
        result = _safe_tag("a/b\\c")
        assert "/" not in result
        assert "\\" not in result

    def test_max_64_chars(self):
        long_tag = "a" * 100
        assert len(_safe_tag(long_tag)) <= 64

    def test_empty_becomes_concept(self):
        assert _safe_tag("") == "concept"


# ---------------------------------------------------------------------------
# Helpers for building fake image datasets
# ---------------------------------------------------------------------------


def _make_png(path: Path) -> None:
    """Crea un PNG 1×1 pixel valido (header minimo)."""
    # PNG magic + minimal IHDR + IDAT + IEND
    data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(data)


def _make_images(images_dir: Path, n: int, with_captions: bool = True) -> list[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img = images_dir / f"img_{i:04d}.png"
        _make_png(img)
        if with_captions:
            (images_dir / f"img_{i:04d}.txt").write_text(f"caption {i}", encoding="utf-8")
        paths.append(img)
    return paths


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------


class TestPrepareDataset:
    def test_no_images_dir_returns_report_with_warning(self, tmp_path):
        images_dir = tmp_path / "nonexistent"
        processed_dir = tmp_path / "processed"
        report = prepare_dataset(images_dir, processed_dir, "vf_test_v1")
        assert report.total_images == 0
        assert len(report.warnings) > 0
        assert not report.is_ready

    def test_too_few_images_not_ready(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 3)
        report = prepare_dataset(images_dir, tmp_path / "processed", "vf_test_v1")
        assert not report.is_ready
        assert report.valid_images == 3

    def test_enough_images_ready(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 10)
        report = prepare_dataset(images_dir, tmp_path / "processed", "vf_test_v1")
        assert report.is_ready
        assert report.valid_images == 10

    def test_creates_concept_dir(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 10)
        processed_dir = tmp_path / "processed"
        report = prepare_dataset(images_dir, processed_dir, "vf_test_v1")
        assert report.output_dir.exists()
        assert "vf_test_v1" in report.output_dir.name

    def test_copies_images(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 8)
        processed_dir = tmp_path / "processed"
        report = prepare_dataset(images_dir, processed_dir, "vf_test_v1")
        png_count = len(list(report.output_dir.glob("*.png")))
        assert png_count == 8

    def test_copies_captions(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 6, with_captions=True)
        processed_dir = tmp_path / "processed"
        report = prepare_dataset(images_dir, processed_dir, "tag")
        txt_count = len(list(report.output_dir.glob("*.txt")))
        assert txt_count == 6

    def test_missing_captions_creates_empty(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 6, with_captions=False)
        processed_dir = tmp_path / "processed"
        report = prepare_dataset(images_dir, processed_dir, "tag")
        assert report.missing_captions == 6
        txt_count = len(list(report.output_dir.glob("*.txt")))
        assert txt_count == 6  # uno per immagine, ma vuoti

    def test_missing_captions_warning(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 6, with_captions=False)
        report = prepare_dataset(images_dir, tmp_path / "p", "tag")
        assert any("caption" in w.lower() for w in report.warnings)

    def test_excluded_images_skipped(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 10)
        # Esclude le prime 3
        manifest = {
            "images": {f"img_{i:04d}.png": {"excluded_from_training": True} for i in range(3)}
        }
        report = prepare_dataset(images_dir, tmp_path / "p", "tag", manifest=manifest)
        assert report.skipped_excluded == 3
        assert report.valid_images == 7

    def test_excluded_images_warning(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 10)
        manifest = {
            "images": {"img_0000.png": {"excluded_from_training": True}}
        }
        report = prepare_dataset(images_dir, tmp_path / "p", "tag", manifest=manifest)
        assert any("esclus" in w.lower() for w in report.warnings)

    def test_repeats_calc_for_small_dataset(self, tmp_path):
        # 10 immagini → repeats = max(1, 200//10) = 20
        images_dir = tmp_path / "images"
        _make_images(images_dir, 10)
        report = prepare_dataset(images_dir, tmp_path / "p", "tag")
        assert report.repeats == 20

    def test_repeats_calc_for_large_dataset(self, tmp_path):
        # 200+ immagini → repeats = 1
        images_dir = tmp_path / "images"
        _make_images(images_dir, 250)
        report = prepare_dataset(images_dir, tmp_path / "p", "tag")
        assert report.repeats == 1

    def test_repeats_in_dir_name(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 10)
        report = prepare_dataset(images_dir, tmp_path / "p", "tag")
        assert str(report.repeats) in report.output_dir.name

    def test_clean_existing_removes_old_dir(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 8)
        processed_dir = tmp_path / "processed"
        # Prima run
        prepare_dataset(images_dir, processed_dir, "tag_v1")
        old_file = next(processed_dir.rglob("*.png"))
        # Seconda run con tag diverso: clean_existing=True
        prepare_dataset(images_dir, processed_dir, "tag_v2")
        assert not old_file.exists()

    def test_clean_existing_false_preserves_old(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 8)
        processed_dir = tmp_path / "processed"
        prepare_dataset(images_dir, processed_dir, "tag")
        old_dirs = list(processed_dir.iterdir())
        prepare_dataset(images_dir, processed_dir, "tag", clean_existing=False)
        # La directory vecchia è ancora lì
        new_dirs = list(processed_dir.iterdir())
        old_names = {d.name for d in old_dirs}
        new_names = {d.name for d in new_dirs}
        assert old_names <= new_names

    def test_total_images_includes_excluded(self, tmp_path):
        images_dir = tmp_path / "images"
        _make_images(images_dir, 10)
        manifest = {
            "images": {f"img_{i:04d}.png": {"excluded_from_training": True} for i in range(2)}
        }
        report = prepare_dataset(images_dir, tmp_path / "p", "tag", manifest=manifest)
        assert report.total_images == 10
