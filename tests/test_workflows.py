"""Validazione strutturale dei workflow ComfyUI e dei loro mapping.

Non esegue ComfyUI: controlla che i template JSON siano grafi ben formati,
che i link tra nodi puntino a nodi esistenti e che i ruoli semantici nei
``.map.json`` siano risolvibili. Cattura errori di battitura nei template
scritti a mano, senza bisogno di GPU/Qt/ComfyUI.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.comfy.workflow import WorkflowTemplate
from src.core.recipes import RECIPES
from src.utils.paths import get_assets_dir

WF_DIR = get_assets_dir() / "workflows"

# Ruoli che il RecipeWorker richiede in ogni workflow di generazione
REQUIRED_ROLES = ("checkpoint", "positive_prompt", "negative_prompt", "seed", "lora")


def _workflow_files() -> list[Path]:
    return sorted(
        p for p in WF_DIR.glob("*.json") if not p.name.endswith(".map.json")
    )


def _is_link(value: object) -> bool:
    """In formato API ComfyUI un link è ``[node_id, slot]`` (str, int)."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def test_at_least_the_known_workflows_present():
    names = {p.name for p in _workflow_files()}
    assert "base_txt2img.json" in names
    assert "correct_inpaint.json" in names
    assert "character_in_pose.json" in names
    assert "product_in_scene.json" in names


def test_every_recipe_workflow_file_exists():
    for recipe in RECIPES.values():
        path = WF_DIR / recipe.workflow_file
        assert path.exists(), f"Ricetta '{recipe.id}': manca {recipe.workflow_file}"


def test_workflows_are_valid_node_graphs():
    for path in _workflow_files():
        graph = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(graph, dict) and graph, f"{path.name}: grafo vuoto/non dict"
        for node_id, node in graph.items():
            assert isinstance(node, dict), f"{path.name}:{node_id} non è un oggetto"
            assert node.get("class_type"), f"{path.name}:{node_id} senza class_type"
            assert isinstance(node.get("inputs"), dict), (
                f"{path.name}:{node_id} senza dict 'inputs'"
            )


def test_node_links_reference_existing_nodes():
    for path in _workflow_files():
        graph = json.loads(path.read_text(encoding="utf-8"))
        for node_id, node in graph.items():
            for field, value in node["inputs"].items():
                if _is_link(value):
                    target = value[0]
                    assert target in graph, (
                        f"{path.name}:{node_id}.{field} -> nodo '{target}' inesistente"
                    )


def test_map_roles_resolve_to_existing_nodes_and_fields():
    for path in _workflow_files():
        wf = WorkflowTemplate(path)
        graph = wf.graph
        for role, spec in wf.mapping.items():
            if role.startswith("_"):  # _meta / documentazione
                continue
            assert isinstance(spec, dict), f"{path.name}: ruolo '{role}' non è dict"
            node_id = spec.get("node")
            assert node_id in graph, (
                f"{path.name}: ruolo '{role}' -> nodo '{node_id}' inesistente"
            )
            if "field" in spec:
                assert spec["field"] in graph[node_id]["inputs"], (
                    f"{path.name}: ruolo '{role}' -> campo '{spec['field']}' "
                    f"assente nel nodo {node_id}"
                )


def test_lora_role_targets_loader_with_named_fields():
    for path in _workflow_files():
        wf = WorkflowTemplate(path)
        lora = wf.mapping.get("lora")
        assert lora and "node" in lora, f"{path.name}: ruolo 'lora' mancante"
        node = wf.graph[lora["node"]]
        for fk in ("name_field", "model_weight_field", "clip_weight_field"):
            field = lora.get(fk)
            assert field and field in node["inputs"], (
                f"{path.name}: lora.{fk}='{field}' assente nel nodo {lora['node']}"
            )


def test_generation_workflows_have_required_roles():
    for path in _workflow_files():
        wf = WorkflowTemplate(path)
        for role in REQUIRED_ROLES:
            assert role in wf.mapping, f"{path.name}: ruolo richiesto '{role}' mancante"


def test_clip_skip_role_is_negative_layer_field():
    """Dove presente, clip_skip deve puntare a un CLIPSetLastLayer."""
    for path in _workflow_files():
        wf = WorkflowTemplate(path)
        spec = wf.mapping.get("clip_skip")
        if not spec:
            continue
        node = wf.graph[spec["node"]]
        assert node["class_type"] == "CLIPSetLastLayer", (
            f"{path.name}: clip_skip non punta a CLIPSetLastLayer"
        )
