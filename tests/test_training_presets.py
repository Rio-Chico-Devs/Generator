"""Tests per src/training/presets.py."""
from __future__ import annotations

import pytest

from src.training.presets import (
    PRESETS,
    PresetId,
    TrainingPreset,
    get_preset,
)


class TestPresetId:
    def test_values(self):
        assert PresetId.VELOCE.value == "veloce"
        assert PresetId.STANDARD.value == "standard"
        assert PresetId.MASSIMO.value == "massimo"

    def test_is_str(self):
        assert isinstance(PresetId.VELOCE, str)

    def test_from_string(self):
        assert PresetId("veloce") is PresetId.VELOCE
        assert PresetId("standard") is PresetId.STANDARD
        assert PresetId("massimo") is PresetId.MASSIMO


class TestPresetsDict:
    def test_all_ids_present(self):
        for pid in PresetId:
            assert pid in PRESETS

    def test_all_values_are_training_preset(self):
        for pid, p in PRESETS.items():
            assert isinstance(p, TrainingPreset)

    def test_preset_ids_match_keys(self):
        for pid, p in PRESETS.items():
            assert p.id == pid

    def test_three_presets(self):
        assert len(PRESETS) == 3


class TestTrainingPresetFields:
    """Verifica che ogni preset abbia campi validi."""

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_name_nonempty(self, pid):
        assert PRESETS[pid].name

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_description_nonempty(self, pid):
        assert PRESETS[pid].description

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_model_family_valid(self, pid):
        assert PRESETS[pid].model_family in {"sd15", "sdxl"}

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_network_dim_positive(self, pid):
        assert PRESETS[pid].network_dim > 0

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_network_alpha_le_dim(self, pid):
        p = PRESETS[pid]
        assert p.network_alpha <= p.network_dim

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_learning_rate_in_range(self, pid):
        lr = PRESETS[pid].learning_rate
        assert 1e-6 <= lr <= 5e-3

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_train_batch_size_ge_1(self, pid):
        assert PRESETS[pid].train_batch_size >= 1

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_max_train_epochs_ge_1(self, pid):
        assert PRESETS[pid].max_train_epochs >= 1

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_resolution_is_string(self, pid):
        res = PRESETS[pid].resolution
        assert isinstance(res, str)
        # Formato sd-scripts: "512,512" o "1024,1024" (anche singolo "512")
        parts = res.split(",")
        assert all(p.strip().isdigit() for p in parts)

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_min_vram_positive(self, pid):
        assert PRESETS[pid].min_vram_gb > 0

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_save_every_n_epochs_ge_1(self, pid):
        assert PRESETS[pid].save_every_n_epochs >= 1


class TestPresetImmutable:
    def test_frozen(self):
        p = PRESETS[PresetId.STANDARD]
        with pytest.raises(Exception):
            p.network_dim = 999  # type: ignore[misc]


class TestVelocePreset:
    def setup_method(self):
        self.p = PRESETS[PresetId.VELOCE]

    def test_sd15_family(self):
        assert self.p.model_family == "sd15"

    def test_min_vram_low(self):
        # Veloce è per GPU consumer
        assert self.p.min_vram_gb <= 6.0

    def test_network_dim_small(self):
        # Veloce usa dim basso per essere leggero
        assert self.p.network_dim <= 16


class TestMassimoPreset:
    def setup_method(self):
        self.p = PRESETS[PresetId.MASSIMO]

    def test_sdxl_family(self):
        assert self.p.model_family == "sdxl"

    def test_larger_dim_than_veloce(self):
        assert self.p.network_dim >= PRESETS[PresetId.VELOCE].network_dim

    def test_more_epochs_than_veloce(self):
        assert self.p.max_train_epochs >= PRESETS[PresetId.VELOCE].max_train_epochs


class TestGetPreset:
    def test_returns_correct_preset(self):
        for pid in PresetId:
            p = get_preset(pid)
            assert p.id == pid

    def test_returns_training_preset(self):
        assert isinstance(get_preset(PresetId.STANDARD), TrainingPreset)

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError):
            get_preset("nonexistent")  # type: ignore[arg-type]


class TestPresetSampler:
    @pytest.mark.parametrize("pid", list(PresetId))
    def test_sample_sampler_nonempty(self, pid):
        assert PRESETS[pid].sample_sampler

    @pytest.mark.parametrize("pid", list(PresetId))
    def test_sample_every_n_ge_1(self, pid):
        assert PRESETS[pid].sample_every_n_epochs >= 1
