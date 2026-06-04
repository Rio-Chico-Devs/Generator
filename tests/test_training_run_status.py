"""Tests per src/training/run_status.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.run_status import (
    STATUS_ABORTED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PREPARING,
    STATUS_RUNNING,
    LossPoint,
    RunStatus,
)


# ---------------------------------------------------------------------------
# LossPoint
# ---------------------------------------------------------------------------


class TestLossPoint:
    def test_to_dict_roundtrip(self):
        lp = LossPoint(step=42, loss=0.1234)
        d = lp.to_dict()
        lp2 = LossPoint.from_dict(d)
        assert lp2.step == 42
        assert lp2.loss == pytest.approx(0.1234)

    def test_to_dict_keys(self):
        d = LossPoint(step=1, loss=0.5).to_dict()
        assert "step" in d and "loss" in d

    def test_from_dict_int_coercion(self):
        lp = LossPoint.from_dict({"step": "10", "loss": "0.5"})
        assert lp.step == 10
        assert lp.loss == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RunStatus defaults
# ---------------------------------------------------------------------------


class TestRunStatusDefaults:
    def test_status_is_preparing(self):
        rs = RunStatus()
        assert rs.status == STATUS_PREPARING

    def test_run_id_nonempty(self):
        rs = RunStatus()
        assert len(rs.run_id) >= 8

    def test_unique_run_ids(self):
        ids = {RunStatus().run_id for _ in range(20)}
        assert len(ids) == 20

    def test_started_at_nonempty(self):
        rs = RunStatus()
        assert rs.started_at

    def test_loss_history_empty(self):
        rs = RunStatus()
        assert rs.loss_history == []

    def test_instances_do_not_share_loss_history(self):
        a = RunStatus()
        b = RunStatus()
        a.loss_history.append(LossPoint(1, 0.5))
        assert b.loss_history == []


# ---------------------------------------------------------------------------
# is_resumable
# ---------------------------------------------------------------------------


class TestIsResumable:
    def test_not_resumable_when_running(self):
        rs = RunStatus(status=STATUS_RUNNING)
        assert not rs.is_resumable

    def test_not_resumable_when_done(self):
        rs = RunStatus(status=STATUS_DONE)
        assert not rs.is_resumable

    def test_not_resumable_when_aborted_no_checkpoint(self):
        rs = RunStatus(status=STATUS_ABORTED, last_checkpoint_path=None)
        assert not rs.is_resumable

    def test_resumable_when_aborted_with_checkpoint(self):
        rs = RunStatus(status=STATUS_ABORTED, last_checkpoint_path="/tmp/lora.safetensors")
        assert rs.is_resumable

    def test_not_resumable_when_error(self):
        rs = RunStatus(status=STATUS_ERROR)
        assert not rs.is_resumable


# ---------------------------------------------------------------------------
# progress_pct
# ---------------------------------------------------------------------------


class TestProgressPct:
    def test_zero_when_no_steps(self):
        rs = RunStatus(current_step=0, max_steps=0)
        assert rs.progress_pct == 0.0

    def test_correct_percentage(self):
        rs = RunStatus(current_step=50, max_steps=200)
        assert rs.progress_pct == pytest.approx(25.0)

    def test_hundred_pct(self):
        rs = RunStatus(current_step=1000, max_steps=1000)
        assert rs.progress_pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _full_rs(self) -> RunStatus:
        rs = RunStatus(
            run_id="abc123",
            preset_id="standard",
            status=STATUS_RUNNING,
            current_step=42,
            max_steps=1000,
            current_epoch=3,
            max_epochs=10,
            config_path="/tmp/config.toml",
            last_checkpoint_path="/tmp/lora_e2.safetensors",
            final_checkpoint_path=None,
            error_message=None,
        )
        rs.loss_history = [LossPoint(1, 0.8), LossPoint(5, 0.6)]
        return rs

    def test_to_dict_all_keys(self):
        d = self._full_rs().to_dict()
        for key in ("run_id", "preset_id", "status", "started_at",
                    "current_step", "max_steps", "current_epoch", "max_epochs",
                    "config_path", "loss_history"):
            assert key in d

    def test_from_dict_restores_fields(self):
        orig = self._full_rs()
        rs = RunStatus.from_dict(orig.to_dict())
        assert rs.run_id == orig.run_id
        assert rs.preset_id == orig.preset_id
        assert rs.status == orig.status
        assert rs.current_step == orig.current_step
        assert rs.max_epochs == orig.max_epochs

    def test_loss_history_preserved(self):
        orig = self._full_rs()
        rs = RunStatus.from_dict(orig.to_dict())
        assert len(rs.loss_history) == 2
        assert rs.loss_history[0].step == 1
        assert rs.loss_history[1].loss == pytest.approx(0.6)

    def test_unknown_keys_ignored(self):
        orig = self._full_rs()
        d = orig.to_dict()
        d["future_field"] = "some_value"
        rs = RunStatus.from_dict(d)
        assert rs.run_id == orig.run_id  # non crasha


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_status_json(self, tmp_path):
        rs = RunStatus(run_id="test01", status=STATUS_RUNNING)
        rs.save(tmp_path)
        assert (tmp_path / "status.json").exists()

    def test_save_and_load_roundtrip(self, tmp_path):
        rs = RunStatus(run_id="test02", status=STATUS_DONE, max_epochs=5, current_epoch=5)
        rs.save(tmp_path)
        loaded = RunStatus.load(tmp_path)
        assert loaded.run_id == "test02"
        assert loaded.status == STATUS_DONE
        assert loaded.max_epochs == 5

    def test_load_nonexistent_returns_default(self, tmp_path):
        rs = RunStatus.load(tmp_path / "nonexistent")
        assert rs.status == STATUS_PREPARING

    def test_save_idempotent(self, tmp_path):
        rs = RunStatus(run_id="x")
        rs.save(tmp_path)
        rs.current_step = 99
        rs.save(tmp_path)
        loaded = RunStatus.load(tmp_path)
        assert loaded.current_step == 99

    def test_no_tmp_residue(self, tmp_path):
        rs = RunStatus()
        rs.save(tmp_path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_save_updates_status(self, tmp_path):
        rs = RunStatus(run_id="upd")
        rs.status = STATUS_PREPARING
        rs.save(tmp_path)
        rs.status = STATUS_RUNNING
        rs.save(tmp_path)
        loaded = RunStatus.load(tmp_path)
        assert loaded.status == STATUS_RUNNING

    def test_loss_history_persisted(self, tmp_path):
        rs = RunStatus()
        rs.loss_history = [LossPoint(10, 0.42), LossPoint(20, 0.35)]
        rs.save(tmp_path)
        loaded = RunStatus.load(tmp_path)
        assert len(loaded.loss_history) == 2
        assert loaded.loss_history[1].step == 20


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------


class TestLoadAll:
    def _make_run(self, runs_dir: Path, run_id: str, status: str) -> None:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        rs = RunStatus(run_id=run_id, status=status)
        rs.save(run_dir)

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert RunStatus.load_all(tmp_path / "empty") == []

    def test_nonexistent_dir_returns_empty_list(self, tmp_path):
        assert RunStatus.load_all(tmp_path / "never_created") == []

    def test_loads_single_run(self, tmp_path):
        runs_dir = tmp_path / "runs"
        self._make_run(runs_dir, "20260601_100000", STATUS_DONE)
        runs = RunStatus.load_all(runs_dir)
        assert len(runs) == 1
        assert runs[0].run_id == "20260601_100000"

    def test_loads_multiple_runs_most_recent_first(self, tmp_path):
        runs_dir = tmp_path / "runs"
        self._make_run(runs_dir, "20260601_100000", STATUS_DONE)
        self._make_run(runs_dir, "20260602_100000", STATUS_RUNNING)
        self._make_run(runs_dir, "20260603_100000", STATUS_ERROR)
        runs = RunStatus.load_all(runs_dir)
        assert len(runs) == 3
        assert runs[0].run_id == "20260603_100000"

    def test_skips_dirs_without_status_json(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "orphan_dir").mkdir()
        self._make_run(runs_dir, "20260601_100000", STATUS_DONE)
        runs = RunStatus.load_all(runs_dir)
        assert len(runs) == 1

    def test_corrupt_status_json_skipped(self, tmp_path):
        runs_dir = tmp_path / "runs"
        bad_dir = runs_dir / "bad_run"
        bad_dir.mkdir(parents=True)
        (bad_dir / "status.json").write_text("{invalid json}", encoding="utf-8")
        self._make_run(runs_dir, "good_run", STATUS_DONE)
        runs = RunStatus.load_all(runs_dir)
        # bad_run skippato, good_run caricato
        assert len(runs) == 1
        assert runs[0].run_id == "good_run"

    def test_resumable_run_found(self, tmp_path):
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "aborted_run"
        run_dir.mkdir(parents=True)
        rs = RunStatus(
            run_id="aborted_run",
            status=STATUS_ABORTED,
            last_checkpoint_path="/tmp/lora.safetensors",
        )
        rs.save(run_dir)
        runs = RunStatus.load_all(runs_dir)
        resumable = [r for r in runs if r.is_resumable]
        assert len(resumable) == 1
