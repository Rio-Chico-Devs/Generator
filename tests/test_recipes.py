from __future__ import annotations

from src.core.catalog import CATALOG
from src.core.recipes import RECIPES, RecipeId, get_recipe, recipes_for_phase


def test_all_recipe_ids_present():
    for rid in RecipeId:
        assert rid in RECIPES
        assert get_recipe(rid).id == rid


def test_recipe_metadata_sane():
    for r in RECIPES.values():
        assert r.workflow_file.endswith(".json")
        # required_models può essere vuoto (es. REFINE usa il modello base del
        # progetto, non un checkpoint fisso del catalogo); se presente, dev'essere
        # una lista di id stringa.
        assert isinstance(r.required_models, list)
        assert all(isinstance(m, str) for m in r.required_models)
        assert r.priority_phase >= 1


def test_base_recipe_models_in_catalog():
    base = get_recipe(RecipeId.BASE)
    assert all(m in CATALOG for m in base.required_models)


def test_recipes_for_phase_is_monotonic():
    p1 = {r.id for r in recipes_for_phase(1)}
    p5 = {r.id for r in recipes_for_phase(5)}
    assert p1.issubset(p5)
    assert RecipeId.BASE in p1
    assert RecipeId.CHARACTER_IN_POSE not in p1
    assert RecipeId.CHARACTER_IN_POSE in p5


def test_recipe_inputs_have_unique_keys():
    for r in RECIPES.values():
        keys = [i.key for i in r.inputs]
        assert len(keys) == len(set(keys)), f"chiavi duplicate in {r.id}"
