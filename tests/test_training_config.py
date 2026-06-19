"""Tests per src/training/config.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.training.config import (
    TrainingParams,
    _clamp_float,
    _clamp_int,
    _write_sample_prompts,
    generate_toml,
    sdscripts_launch_cmd,
)
from src.training.presets import PresetId, get_preset


# ---------------------------------------------------------------------------
# _clamp_int / _clamp_float
# ---------------------------------------------------------------------------


class TestClampInt:
    def test_within_range(self):
        assert _clamp_int(10, 1, 20) == 10

    def test_clamps_low(self):
        assert _clamp_int(0, 1, 20) == 1

    def test_clamps_high(self):
        assert _clamp_int(200, 1, 128) == 128

    def test_at_boundaries(self):
        assert _clamp_int(1, 1, 1) == 1

    def test_returns_int(self):
        assert isinstance(_clamp_int(5.7, 1, 10), int)


class TestClampFloat:
    def test_within_range(self):
        assert _clamp_float(1e-4, 1e-6, 5e-3) == pytest.approx(1e-4)

    def test_clamps_low(self):
        assert _clamp_float(0.0, 1e-6, 5e-3) == pytest.approx(1e-6)

    def test_clamps_high(self):
        assert _clamp_float(1.0, 1e-6, 5e-3) == pytest.approx(5e-3)

    def test_returns_float(self):
        assert isinstance(_clamp_float(1e-4, 1e-6, 5e-3), float)


# ---------------------------------------------------------------------------
# TrainingParams defaults
# ---------------------------------------------------------------------------


class TestTrainingParamsDefaults:
    def test_epochs_default(self):
        p = TrainingParams()
        assert p.epochs == 10

    def test_optional_fields_none(self):
        p = TrainingParams()
        assert p.network_dim is None
        assert p.network_alpha is None
        assert p.learning_rate is None
        assert p.lr_scheduler is None
        assert p.optimizer_type is None

    def test_custom_sample_prompts_default_empty_list(self):
        p = TrainingParams()
        assert p.custom_sample_prompts == []

    def test_instances_do_not_share_list(self):
        a = TrainingParams()
        b = TrainingParams()
        a.custom_sample_prompts.append("x")
        assert b.custom_sample_prompts == []

    def test_noise_offset_default(self):
        assert TrainingParams().noise_offset == 0.0

    def test_train_text_encoder_default_false(self):
        assert TrainingParams().train_text_encoder is False


# ---------------------------------------------------------------------------
# _write_sample_prompts
# ---------------------------------------------------------------------------


class TestWriteSamplePrompts:
    def test_uses_custom_prompts(self, tmp_path):
        p = tmp_path / "prompts.txt"
        _write_sample_prompts(p, ["hello world"], "mytag")
        assert p.read_text() == "hello world"

    def test_default_prompts_contain_tag(self, tmp_path):
        p = tmp_path / "prompts.txt"
        _write_sample_prompts(p, [], "vf_iris_v1")
        content = p.read_text()
        assert "vf_iris_v1" in content
        assert content.count("\n") >= 1  # più di una riga

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "deep" / "prompts.txt"
        _write_sample_prompts(p, [], "tag")
        assert p.exists()


# ---------------------------------------------------------------------------
# generate_toml
# ---------------------------------------------------------------------------


def _minimal_generate_toml(
    tmp_path: Path,
    *,
    epochs: int = 5,
    preset_id: PresetId = PresetId.STANDARD,
    network_dim: int | None = None,
    train_text_encoder: bool = False,
    noise_offset: float = 0.0,
    min_snr_gamma: float = 5.0,
    custom_prompts: list[str] | None = None,
    resume_from: Path | None = None,
) -> tuple:
    import toml

    preset = get_preset(preset_id)
    params = TrainingParams(
        epochs=epochs,
        network_dim=network_dim,
        train_text_encoder=train_text_encoder,
        noise_offset=noise_offset,
        min_snr_gamma=min_snr_gamma,
        custom_sample_prompts=custom_prompts or [],
        resume_from=resume_from,
    )
    base_model = tmp_path / "model.safetensors"
    base_model.write_bytes(b"\x00" * 8)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    content = generate_toml(
        preset=preset,
        params=params,
        base_model_path=base_model,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        activator_tag="vf_test_v1",
    )
    # Deve essere TOML valido
    parsed = toml.loads(content)
    return content, parsed


class TestGenerateToml:
    def test_returns_string(self, tmp_path):
        content, _ = _minimal_generate_toml(tmp_path)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_valid_toml(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path)
        assert isinstance(parsed, dict)

    def test_epochs_capped_at_50(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, epochs=999)
        assert parsed["max_train_epochs"] == 50

    def test_epochs_at_least_1(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, epochs=0)
        assert parsed["max_train_epochs"] == 1

    def test_network_dim_capped_at_128(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, network_dim=512)
        assert parsed["network_dim"] == 128

    def test_network_dim_from_params(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, network_dim=32)
        assert parsed["network_dim"] == 32

    def test_network_dim_falls_back_to_preset(self, tmp_path):
        preset = get_preset(PresetId.STANDARD)
        _, parsed = _minimal_generate_toml(tmp_path, network_dim=None)
        assert parsed["network_dim"] == preset.network_dim

    def test_contains_pretrained_model_path(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path)
        assert "pretrained_model_name_or_path" in parsed

    def test_train_data_dir_present(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path)
        assert "train_data_dir" in parsed

    def test_output_dir_present(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path)
        assert "output_dir" in parsed

    def test_network_module_is_lora(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path)
        assert parsed["network_module"] == "networks.lora"

    def test_save_model_as_safetensors(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path)
        assert parsed["save_model_as"] == "safetensors"

    def test_noise_offset_included_when_nonzero(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, noise_offset=0.05)
        assert "noise_offset" in parsed
        assert abs(parsed["noise_offset"] - 0.05) < 1e-6

    def test_noise_offset_excluded_when_zero(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, noise_offset=0.0)
        assert "noise_offset" not in parsed

    def test_min_snr_excluded_when_default(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, min_snr_gamma=5.0)
        assert "min_snr_gamma" not in parsed

    def test_min_snr_included_when_custom(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, min_snr_gamma=3.0)
        assert "min_snr_gamma" in parsed

    def test_train_text_encoder_unet_only_false(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, train_text_encoder=True)
        assert parsed["network_train_unet_only"] is False

    def test_resume_path_in_toml(self, tmp_path):
        ckpt = tmp_path / "ckpt.safetensors"
        ckpt.write_bytes(b"\x00")
        _, parsed = _minimal_generate_toml(tmp_path, resume_from=ckpt)
        assert "resume" in parsed
        assert str(ckpt) in parsed["resume"]

    def test_resume_absent_when_none(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, resume_from=None)
        assert "resume" not in parsed

    def test_sdxl_preset_has_bucket(self, tmp_path):
        _, parsed = _minimal_generate_toml(tmp_path, preset_id=PresetId.STANDARD)
        preset = get_preset(PresetId.STANDARD)
        if preset.enable_bucket:
            assert "enable_bucket" in parsed

    def test_sample_prompts_file_written(self, tmp_path):
        _minimal_generate_toml(tmp_path, custom_prompts=["test prompt"])
        run_dir = tmp_path / "run"
        prompts_file = run_dir / "sample_prompts.txt"
        assert prompts_file.exists()
        assert "test prompt" in prompts_file.read_text()

    def test_checkpoints_dir_created(self, tmp_path):
        _minimal_generate_toml(tmp_path)
        assert (tmp_path / "run" / "checkpoints").is_dir()

    def test_logs_dir_created(self, tmp_path):
        _minimal_generate_toml(tmp_path)
        assert (tmp_path / "run" / "logs").is_dir()

    def test_samples_dir_created(self, tmp_path):
        _minimal_generate_toml(tmp_path)
        assert (tmp_path / "run" / "samples").is_dir()


# ---------------------------------------------------------------------------
# sdscripts_launch_cmd
# ---------------------------------------------------------------------------


class TestSdscriptsLaunchCmd:
    def test_returns_empty_when_no_script(self, tmp_path):
        cmd = sdscripts_launch_cmd(tmp_path, tmp_path / "config.toml")
        assert cmd == []

    def test_returns_list_with_sdxl_script(self, tmp_path):
        script = tmp_path / "sdxl_train_network.py"
        script.write_text("# dummy")
        config = tmp_path / "config.toml"
        config.write_text("")
        cmd = sdscripts_launch_cmd(tmp_path, config, "sdxl")
        assert isinstance(cmd, list)
        assert len(cmd) > 0
        assert cmd[0] == "accelerate"
        assert "sdxl_train_network.py" in cmd[-1] or any("sdxl" in c for c in cmd)

    def test_returns_list_with_sd15_script(self, tmp_path):
        script = tmp_path / "train_network.py"
        script.write_text("# dummy")
        config = tmp_path / "config.toml"
        config.write_text("")
        cmd = sdscripts_launch_cmd(tmp_path, config, "sd15")
        assert isinstance(cmd, list)
        assert cmd[0] == "accelerate"
        assert "--config_file" in cmd
        assert str(config) in cmd

    def test_config_file_in_cmd(self, tmp_path):
        script = tmp_path / "sdxl_train_network.py"
        script.write_text("")
        config = tmp_path / "my_config.toml"
        config.write_text("")
        cmd = sdscripts_launch_cmd(tmp_path, config, "sdxl")
        config_idx = cmd.index("--config_file")
        assert cmd[config_idx + 1] == str(config)

    def test_accelerate_launch_in_cmd(self, tmp_path):
        script = tmp_path / "sdxl_train_network.py"
        script.write_text("")
        cmd = sdscripts_launch_cmd(tmp_path, tmp_path / "c.toml", "sdxl")
        assert "launch" in cmd
