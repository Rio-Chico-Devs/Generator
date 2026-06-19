"""Test del modello GuidedSession (Pilastro A) — no Qt, no ComfyUI."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.guidance.session import (
    STATUS_ACTIVE,
    STATUS_COMPLETED,
    Candidate,
    GuidedSession,
    StepDefinition,
    StepResult,
    default_pipeline,
)


def _candidate(idx: int, directory: Path, approved=None) -> Candidate:
    img = directory / f"cand_{idx}.png"
    return Candidate(index=idx, image_path=img, is_approved=approved)


def _session(directory: Path, pipeline=None) -> GuidedSession:
    return GuidedSession(
        project_slug="iris",
        prompt="iris, full body",
        negative_prompt="lowres",
        seed=1234,
        lora_checkpoint="lora.safetensors",
        pipeline=pipeline if pipeline is not None else default_pipeline(),
    )


# ---------------------------------------------------------------------------
# StepDefinition
# ---------------------------------------------------------------------------


def test_step_definition_roundtrip():
    sd = StepDefinition(
        id="colors", label="Colori", denoise_strength=0.55,
        n_candidates=3, guidance_scale=5.5, steps=12, technique_slots=["skin"],
    )
    assert StepDefinition.from_dict(sd.to_dict()) == sd


def test_default_pipeline_has_four_steps():
    p = default_pipeline()
    assert [s.id for s in p] == ["composition", "structure", "colors", "shading"]


def test_default_pipeline_first_step_full_denoise():
    assert default_pipeline()[0].denoise_strength == 1.0


def test_default_pipeline_returns_independent_lists():
    a = default_pipeline()
    b = default_pipeline()
    a[1].technique_slots.append("x")
    assert "x" not in b[1].technique_slots


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


def test_candidate_roundtrip(tmp_path):
    c = Candidate(
        index=2,
        image_path=tmp_path / "c.png",
        sidecar_path=tmp_path / "c.json",
        is_approved=True,
    )
    restored = Candidate.from_dict(c.to_dict())
    assert restored == c


def test_candidate_optional_paths_none(tmp_path):
    c = Candidate(index=0, image_path=tmp_path / "c.png")
    d = c.to_dict()
    assert d["sidecar_path"] is None
    assert d["latent_path"] is None
    assert Candidate.from_dict(d) == c


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


def test_step_result_approved_candidate(tmp_path):
    cands = [_candidate(0, tmp_path), _candidate(1, tmp_path)]
    r = StepResult(step_def=default_pipeline()[0], candidates=cands, approved_index=1)
    assert r.approved_candidate.index == 1
    assert not r.all_rejected


def test_step_result_all_rejected(tmp_path):
    cands = [_candidate(0, tmp_path), _candidate(1, tmp_path)]
    r = StepResult(step_def=default_pipeline()[0], candidates=cands, approved_index=None)
    assert r.approved_candidate is None
    assert r.all_rejected


def test_step_result_empty_not_rejected():
    r = StepResult(step_def=default_pipeline()[0], candidates=[])
    assert not r.all_rejected


def test_step_result_roundtrip(tmp_path):
    cands = [_candidate(0, tmp_path, approved=False), _candidate(1, tmp_path, approved=True)]
    r = StepResult(
        step_def=default_pipeline()[1], candidates=cands,
        approved_index=1, rejection_reason="troppo scuro",
    )
    restored = StepResult.from_dict(r.to_dict())
    assert restored.approved_index == 1
    assert restored.rejection_reason == "troppo scuro"
    assert len(restored.candidates) == 2


# ---------------------------------------------------------------------------
# GuidedSession — stato derivato
# ---------------------------------------------------------------------------


def test_new_session_is_active(tmp_path):
    s = _session(tmp_path)
    assert s.status == STATUS_ACTIVE
    assert not s.is_finished
    assert s.current_step_index == 0


def test_current_step_def_tracks_progress(tmp_path):
    s = _session(tmp_path)
    assert s.current_step_def.id == "composition"
    s.results.append(StepResult(step_def=s.pipeline[0], candidates=[], approved_index=0))
    assert s.current_step_def.id == "structure"


def test_current_step_def_none_when_pipeline_done(tmp_path):
    s = _session(tmp_path, pipeline=[default_pipeline()[0]])
    s.results.append(StepResult(step_def=s.pipeline[0], candidates=[], approved_index=0))
    assert s.current_step_def is None


def test_step_input_image_none_at_start(tmp_path):
    s = _session(tmp_path)
    assert s.step_input_image() is None


def test_step_input_image_returns_last_approved(tmp_path):
    s = _session(tmp_path)
    cands = [_candidate(0, tmp_path), _candidate(1, tmp_path)]
    s.results.append(StepResult(step_def=s.pipeline[0], candidates=cands, approved_index=1))
    assert s.step_input_image() == cands[1].image_path


def test_step_input_image_none_if_last_rejected(tmp_path):
    s = _session(tmp_path)
    cands = [_candidate(0, tmp_path)]
    s.results.append(StepResult(step_def=s.pipeline[0], candidates=cands, approved_index=None))
    assert s.step_input_image() is None


# ---------------------------------------------------------------------------
# GuidedSession — record_result / completamento
# ---------------------------------------------------------------------------


def test_record_result_completes_on_last_step(tmp_path):
    s = _session(tmp_path, pipeline=[default_pipeline()[0]])
    cands = [_candidate(0, tmp_path, approved=True)]
    s.record_result(StepResult(step_def=s.pipeline[0], candidates=cands, approved_index=0))
    assert s.status == STATUS_COMPLETED
    assert s.final_image_path == cands[0].image_path
    assert s.is_finished


def test_record_result_not_complete_mid_pipeline(tmp_path):
    s = _session(tmp_path)  # 4 step
    cands = [_candidate(0, tmp_path, approved=True)]
    s.record_result(StepResult(step_def=s.pipeline[0], candidates=cands, approved_index=0))
    assert s.status == STATUS_ACTIVE


def test_record_result_last_step_rejected_not_completed(tmp_path):
    s = _session(tmp_path, pipeline=[default_pipeline()[0]])
    cands = [_candidate(0, tmp_path)]
    s.record_result(StepResult(step_def=s.pipeline[0], candidates=cands, approved_index=None))
    assert s.status == STATUS_ACTIVE
    assert s.final_image_path is None


# ---------------------------------------------------------------------------
# GuidedSession — persistenza
# ---------------------------------------------------------------------------


def test_session_save_load_roundtrip(tmp_path):
    s = _session(tmp_path)
    cands = [_candidate(0, tmp_path, approved=False), _candidate(1, tmp_path, approved=True)]
    s.record_result(StepResult(step_def=s.pipeline[0], candidates=cands, approved_index=1))

    path = tmp_path / "session.json"
    s.save(path)
    loaded = GuidedSession.load(path)

    assert loaded.session_id == s.session_id
    assert loaded.project_slug == "iris"
    assert loaded.seed == 1234
    assert len(loaded.pipeline) == 4
    assert len(loaded.results) == 1
    assert loaded.results[0].approved_index == 1


def test_session_save_creates_parent_dirs(tmp_path):
    s = _session(tmp_path)
    path = tmp_path / "nested" / "deep" / "session.json"
    s.save(path)
    assert path.exists()


def test_session_load_rejects_future_schema(tmp_path):
    s = _session(tmp_path)
    d = s.to_dict()
    d["schema_version"] = 999
    path = tmp_path / "s.json"
    import json
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError):
        GuidedSession.load(path)


def test_session_generates_unique_ids(tmp_path):
    a = _session(tmp_path)
    b = _session(tmp_path)
    assert a.session_id != b.session_id
