from __future__ import annotations

from src.core.catalog import CATALOG, VAE_CATALOG, suggest_models_for


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
