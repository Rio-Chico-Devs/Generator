from __future__ import annotations

from src.core.catalog import (
    AUX_CATALOG,
    CATALOG,
    VAE_CATALOG,
    aux_models_for_phase,
    suggest_models_for,
)
from src.core.recipes import RECIPES, RecipeId


def test_catalog_keys_match_entry_ids():
    assert CATALOG
    for key, entry in CATALOG.items():
        assert key == entry.id


def test_vae_references_resolve():
    for entry in CATALOG.values():
        if entry.requires_vae is not None:
            assert entry.requires_vae in VAE_CATALOG


def test_suggest_models_for_character_includes_pony():
    ids = suggest_models_for("character")
    assert "pony-v6-xl" in ids


def test_suggest_models_prefer_fast_puts_sd15_first():
    ids = suggest_models_for("illustration", prefer_fast=True)
    assert ids
    assert CATALOG[ids[0]].family == "sd15"


def test_suggest_models_default_puts_sdxl_first():
    ids = suggest_models_for("illustration")
    assert ids
    assert CATALOG[ids[0]].family == "sdxl"


# ---------------------------------------------------------------------------
# AUX_CATALOG
# ---------------------------------------------------------------------------


def test_aux_catalog_keys_match_entry_ids():
    assert AUX_CATALOG
    for key, entry in AUX_CATALOG.items():
        assert key == entry.id, f"Chiave '{key}' != entry.id '{entry.id}'"


def test_aux_catalog_required_fields_non_empty():
    for entry in AUX_CATALOG.values():
        assert entry.repo_id, f"{entry.id}: repo_id vuoto"
        assert entry.filename, f"{entry.id}: filename vuoto"
        assert entry.comfy_subdir, f"{entry.id}: comfy_subdir vuoto"
        assert entry.license, f"{entry.id}: license vuoto"
        assert entry.description, f"{entry.id}: description vuota"


def test_aux_catalog_size_positive():
    for entry in AUX_CATALOG.values():
        assert entry.size_gb > 0, f"{entry.id}: size_gb deve essere > 0"
        assert entry.min_vram_gb >= 0, f"{entry.id}: min_vram_gb deve essere >= 0"


def test_aux_catalog_phase_valid():
    for entry in AUX_CATALOG.values():
        assert 1 <= entry.required_for_phase <= 7, (
            f"{entry.id}: required_for_phase={entry.required_for_phase} fuori range 1-7"
        )


def test_recipe_required_models_in_aux_or_base_catalog():
    """Ogni ID in required_models delle ricette deve essere nel CATALOG o AUX_CATALOG."""
    all_known = set(CATALOG.keys()) | set(AUX_CATALOG.keys())
    for recipe in RECIPES.values():
        for mid in recipe.required_models:
            assert mid in all_known, (
                f"Ricetta '{recipe.id}': modello '{mid}' non trovato "
                f"in CATALOG né in AUX_CATALOG"
            )


def test_aux_models_for_phase_filters_correctly():
    phase1 = aux_models_for_phase(1)
    phase4 = aux_models_for_phase(4)
    assert len(phase4) >= len(phase1), "Fase 4 deve includere almeno i modelli di fase 1"
    for entry in phase1:
        assert entry.required_for_phase <= 1
    for entry in phase4:
        assert entry.required_for_phase <= 4
