"""Test per RecipeWorker e _parametrize.

Livello 1 — _parametrize: test puri, nessun Qt richiesto.
Livello 2 — RecipeWorker QThread: usa qtbot di pytest-qt.

I test di livello 2 usano MockComfyClient: nessuna GPU necessaria.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.comfy.client import MockComfyClient
from src.comfy.workflow import WorkflowTemplate
from src.core.project import ActiveLora, BaseModelRef, GenerationDefaults, Project
from src.core.recipes import RECIPES, RecipeId
from src.utils.paths import get_assets_dir
from src.workers.recipe_worker import (
    RecipeWorker,
    _copy_to_comfy_input,
    _parametrize,
)


# ---------------------------------------------------------------------------
# Fixture condivise
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_recipe():
    return RECIPES[RecipeId.BASE]


@pytest.fixture()
def project_with_lora(tmp_path):
    """Progetto con base_model + LoRA attivo su disco (file dummy)."""
    p = Project.create(
        name="Test Character",
        base_model=BaseModelRef(id="pony-v6-xl", name="Pony V6 XL"),
        parent_dir=tmp_path / "projects",
    )
    lora_file = p.training_runs_dir / "character_v1.safetensors"
    lora_file.write_bytes(b"\x00" * 8)
    p.active_lora = ActiveLora(
        run_id="run_01",
        checkpoint="character_v1.safetensors",
        weight=0.85,
    )
    p.save()
    return p


@pytest.fixture()
def project_no_lora(tmp_path):
    """Progetto senza LoRA attivo."""
    return Project.create(
        name="No LoRA",
        base_model=BaseModelRef(id="pony-v6-xl", name="Pony V6 XL"),
        parent_dir=tmp_path / "projects",
    )


@pytest.fixture()
def base_wf():
    """WorkflowTemplate fresco per ogni test (evita contaminazione tra test)."""
    return WorkflowTemplate(
        get_assets_dir() / "workflows" / "base_txt2img.json"
    )


# ---------------------------------------------------------------------------
# Livello 1 — _parametrize (puro, no Qt)
# ---------------------------------------------------------------------------


class TestParametrize:
    def test_prompt_includes_trigger_prefix(self, base_wf, project_with_lora):
        # base_model pony-v6-xl → dialetto "pony": i quality-tag (score_*)
        # vanno in testa, poi il trigger del progetto, poi il prompt utente.
        project_with_lora.trigger_prompt_prefix = "vf_char_v1, masterpiece,"
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "running in a forest"},
            project_with_lora,
        )
        pos = base_wf.mapping["positive_prompt"]
        result = base_wf.get_param(pos["node"], pos["field"])
        assert result.startswith("score_9")
        assert "vf_char_v1" in result
        assert "running in a forest" in result

    def test_prompt_without_prefix(self, base_wf, project_with_lora):
        project_with_lora.trigger_prompt_prefix = ""
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "standing alone"},
            project_with_lora,
        )
        pos = base_wf.mapping["positive_prompt"]
        result = base_wf.get_param(pos["node"], pos["field"])
        assert "standing alone" in result
        assert result.startswith("score_9")  # quality-tag del dialetto pony

    def test_negative_prompt_explicit(self, base_wf, project_with_lora):
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "negative": "ugly, blurry"},
            project_with_lora,
        )
        neg = base_wf.mapping["negative_prompt"]
        result = base_wf.get_param(neg["node"], neg["field"])
        # Il negative dell'utente è rispettato e i quality-negative accodati.
        assert result.startswith("ugly, blurry")
        assert "worst quality" in result

    def test_negative_falls_back_to_project_default(self, base_wf, project_with_lora):
        project_with_lora.default_negative_prompt = "project_default_neg"
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x"},
            project_with_lora,
        )
        neg = base_wf.mapping["negative_prompt"]
        result = base_wf.get_param(neg["node"], neg["field"])
        assert result.startswith("project_default_neg")

    def test_clip_skip_applied_for_pony(self, base_wf, project_with_lora):
        # Pony vuole CLIP skip 2 → ComfyUI CLIPSetLastLayer = -2.
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x"},
            project_with_lora,
        )
        cs = base_wf.mapping["clip_skip"]
        assert base_wf.get_param(cs["node"], cs["field"]) == -2

    def test_record_captures_effective_params(self, base_wf, project_with_lora):
        rec, _ = _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "seed": 123},
            project_with_lora,
        )
        assert rec.model_id == "pony-v6-xl"
        assert rec.dialect == "pony"
        assert rec.seed == 123
        assert rec.clip_skip == 2
        assert rec.lora_name == "character_v1.safetensors"
        assert rec.positive.startswith("score_9")

    def test_seed_minus1_resolved_to_non_negative(self, base_wf, project_with_lora):
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "seed": -1},
            project_with_lora,
        )
        s = base_wf.mapping["seed"]
        actual = base_wf.get_param(s["node"], s["field"])
        assert isinstance(actual, int)
        assert actual >= 0

    def test_seed_explicit_preserved(self, base_wf, project_with_lora):
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "seed": 42},
            project_with_lora,
        )
        s = base_wf.mapping["seed"]
        assert base_wf.get_param(s["node"], s["field"]) == 42

    def test_dimensions_from_params(self, base_wf, project_with_lora):
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "width": 768, "height": 1152},
            project_with_lora,
        )
        w = base_wf.mapping["width"]
        h = base_wf.mapping["height"]
        assert base_wf.get_param(w["node"], w["field"]) == 768
        assert base_wf.get_param(h["node"], h["field"]) == 1152

    def test_dimensions_fall_back_to_project_defaults(self, base_wf, project_with_lora):
        project_with_lora.default_generation_params = GenerationDefaults(
            width=512, height=512
        )
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x"},
            project_with_lora,
        )
        w = base_wf.mapping["width"]
        h = base_wf.mapping["height"]
        assert base_wf.get_param(w["node"], w["field"]) == 512
        assert base_wf.get_param(h["node"], h["field"]) == 512

    def test_checkpoint_set_from_project_base_model(self, base_wf, project_with_lora):
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x"},
            project_with_lora,
        )
        ckpt = base_wf.mapping["checkpoint"]
        assert (
            base_wf.get_param(ckpt["node"], ckpt["field"])
            == "pony-v6-xl.safetensors"
        )

    def test_lora_weight_applied(self, base_wf, project_with_lora):
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "lora_weight": 0.6},
            project_with_lora,
        )
        lora_node = base_wf.mapping["lora"]["node"]
        assert base_wf.get_param(lora_node, "strength_model") == pytest.approx(0.6)
        assert base_wf.get_param(lora_node, "strength_clip") == pytest.approx(0.6)

    def test_lora_name_is_basename_only(self, base_wf, project_with_lora):
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x"},
            project_with_lora,
        )
        lora_node = base_wf.mapping["lora"]["node"]
        name = base_wf.get_param(lora_node, "lora_name")
        assert name == "character_v1.safetensors"
        assert "/" not in name and "\\" not in name

    def test_no_lora_does_not_crash(self, base_wf, project_no_lora):
        """Progetto senza LoRA: _parametrize non crasha, emette solo warning."""
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x"},
            project_no_lora,
        )

    def test_steps_and_cfg_applied_via_residual(self, base_wf, project_with_lora):
        """steps e cfg sono nel mapping e vengono applicati dal loop residuo."""
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "steps": 20, "cfg": 5.5},
            project_with_lora,
        )
        steps = base_wf.mapping["steps"]
        cfg = base_wf.mapping["cfg"]
        assert base_wf.get_param(steps["node"], steps["field"]) == 20
        assert base_wf.get_param(cfg["node"], cfg["field"]) == pytest.approx(5.5)

    def test_unknown_key_in_params_ignored(self, base_wf, project_with_lora):
        """Chiavi non nel mapping non causano errori."""
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "x", "banana": 99, "unknown_knob": "foo"},
            project_with_lora,
        )

    def test_build_after_parametrize_is_independent(self, base_wf, project_with_lora):
        """build() ritorna deepcopy: mutazioni successive non alterano il risultato."""
        _parametrize(
            base_wf,
            RECIPES[RecipeId.BASE],
            {"prompt": "scene A", "seed": 100},
            project_with_lora,
        )
        built = base_wf.build()
        # Muta il template originale
        base_wf.set_seed(999)
        s = base_wf.mapping["seed"]
        assert built[s["node"]]["inputs"][s["field"]] == 100


# ---------------------------------------------------------------------------
# TestCopyToComfyInput
# ---------------------------------------------------------------------------


class TestCopyToComfyInput:
    def test_copy_creates_file_in_dest(self, tmp_path):
        src = tmp_path / "pose.png"
        src.write_bytes(b"\x89PNG\r\n")
        comfy_in = tmp_path / "comfy_input"

        name, _ = _copy_to_comfy_input(src, comfy_in)

        assert (comfy_in / name).exists()
        assert (comfy_in / name).read_bytes() == b"\x89PNG\r\n"

    def test_copy_names_are_unique(self, tmp_path):
        src = tmp_path / "ref.png"
        src.write_bytes(b"\x89PNG")
        comfy_in = tmp_path / "ci"

        name1 = _copy_to_comfy_input(src, comfy_in)
        name2 = _copy_to_comfy_input(src, comfy_in)

        assert name1 != name2

    def test_copy_preserves_extension(self, tmp_path):
        src = tmp_path / "ref.jpg"
        src.write_bytes(b"\xff\xd8")
        name, _ = _copy_to_comfy_input(src, tmp_path / "ci")
        assert name.endswith(".jpg")

    def test_copy_creates_dest_dir_if_missing(self, tmp_path):
        src = tmp_path / "img.png"
        src.write_bytes(b"data")
        missing_dir = tmp_path / "does" / "not" / "exist"

        _copy_to_comfy_input(src, missing_dir)

        assert missing_dir.exists()


# ---------------------------------------------------------------------------
# Livello 2 — RecipeWorker QThread (richiede qtbot di pytest-qt)
# ---------------------------------------------------------------------------


class TestRecipeWorker:
    def test_worker_emits_progress_and_finished_ok(
        self, qtbot, tmp_path, project_with_lora
    ):
        client = MockComfyClient()
        output_dir = tmp_path / "generations"

        worker = RecipeWorker(
            recipe=RECIPES[RecipeId.BASE],
            user_params={"prompt": "character standing on a hill", "seed": 7},
            project=project_with_lora,
            client=client,
            output_dir=output_dir,
        )

        progress_calls: list[tuple[int, int]] = []
        ready_paths: list[Path] = []
        errors: list[str] = []

        worker.progress.connect(lambda s, t: progress_calls.append((s, t)))
        worker.image_ready.connect(ready_paths.append)
        worker.error.connect(errors.append)

        with qtbot.waitSignal(worker.finished_ok, timeout=15_000) as blocker:
            worker.start()

        assert not errors, f"Errori inattesi: {errors}"
        assert progress_calls, "Nessun signal progress emesso"
        assert progress_calls[-1][0] == progress_calls[-1][1], (
            "Ultimo step != totale"
        )
        final_paths: list[Path] = blocker.args[0]
        assert len(final_paths) >= 1
        assert all(isinstance(p, Path) for p in final_paths)
        assert ready_paths == final_paths
        assert all(p.exists() for p in final_paths)

    def test_worker_output_copied_to_output_dir(
        self, qtbot, tmp_path, project_with_lora
    ):
        output_dir = tmp_path / "my_generations"

        worker = RecipeWorker(
            recipe=RECIPES[RecipeId.BASE],
            user_params={"prompt": "test"},
            project=project_with_lora,
            client=MockComfyClient(),
            output_dir=output_dir,
        )

        with qtbot.waitSignal(worker.finished_ok, timeout=15_000) as blocker:
            worker.start()

        for path in blocker.args[0]:
            assert path.parent == output_dir

    def test_worker_emits_error_on_submit_failure(
        self, qtbot, tmp_path, project_with_lora
    ):
        class FailingClient(MockComfyClient):
            def submit(self, workflow: dict) -> str:
                raise RuntimeError("ComfyUI non raggiungibile")

        worker = RecipeWorker(
            recipe=RECIPES[RecipeId.BASE],
            user_params={"prompt": "test"},
            project=project_with_lora,
            client=FailingClient(),
            output_dir=tmp_path / "out",
        )

        errors: list[str] = []
        worker.error.connect(errors.append)

        with qtbot.waitSignal(worker.error, timeout=5_000):
            worker.start()

        assert errors
        assert "ComfyUI non raggiungibile" in errors[0]

    def test_worker_abort_calls_interrupt(
        self, qtbot, tmp_path, project_with_lora
    ):
        interrupted: list[bool] = []

        class TrackingClient(MockComfyClient):
            def interrupt(self) -> None:
                interrupted.append(True)

        worker = RecipeWorker(
            recipe=RECIPES[RecipeId.BASE],
            user_params={"prompt": "test"},
            project=project_with_lora,
            client=TrackingClient(),
            output_dir=tmp_path / "out",
        )

        worker.start()
        worker.abort()
        worker.wait(10_000)

        assert interrupted, "interrupt() non chiamato dopo abort()"

    def test_worker_no_lora_does_not_crash(
        self, qtbot, tmp_path, project_no_lora
    ):
        """Progetto senza LoRA: warning nel log ma nessun crash."""
        worker = RecipeWorker(
            recipe=RECIPES[RecipeId.BASE],
            user_params={"prompt": "test"},
            project=project_no_lora,
            client=MockComfyClient(),
            output_dir=tmp_path / "out",
        )

        errors: list[str] = []
        worker.error.connect(errors.append)

        with qtbot.waitSignal(worker.finished_ok, timeout=15_000):
            worker.start()

        assert not errors, f"Errori inattesi con progetto senza LoRA: {errors}"

    def test_worker_batch_produces_multiple_images(
        self, qtbot, tmp_path, project_with_lora
    ):
        from src.utils.gpu_monitor import GpuSnapshot, SafetyConfig

        worker = RecipeWorker(
            recipe=RECIPES[RecipeId.BASE],
            user_params={"prompt": "x", "variants": 3, "seed": 10},
            project=project_with_lora,
            client=MockComfyClient(),
            output_dir=tmp_path / "out",
            safety=SafetyConfig(cooldown_between_images_sec=0.01),
            gpu_read_fn=lambda: GpuSnapshot(available=False),
        )

        with qtbot.waitSignal(worker.finished_ok, timeout=20_000) as blocker:
            worker.start()

        paths = blocker.args[0]
        assert len(paths) == 3
        assert all(p.exists() for p in paths)

    def test_worker_writes_sidecar_with_resolved_seed(
        self, qtbot, tmp_path, project_with_lora
    ):
        import json as _json

        from src.utils.gpu_monitor import GpuSnapshot, SafetyConfig

        worker = RecipeWorker(
            recipe=RECIPES[RecipeId.BASE],
            user_params={"prompt": "x", "seed": 777},
            project=project_with_lora,
            client=MockComfyClient(),
            output_dir=tmp_path / "out",
            safety=SafetyConfig(enabled=False),
            gpu_read_fn=lambda: GpuSnapshot(available=False),
        )

        with qtbot.waitSignal(worker.finished_ok, timeout=15_000) as blocker:
            worker.start()

        img = blocker.args[0][0]
        sidecar = img.with_suffix(".json")
        assert sidecar.exists(), "sidecar JSON non scritto"
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["seed"] == 777
        assert meta["dialect"] == "pony"
        assert meta["model_id"] == "pony-v6-xl"
        assert meta["created_at"]
