from __future__ import annotations

from pathlib import Path

from src.comfy.workflow import WorkflowTemplate
from src.utils.paths import get_assets_dir


def _base_path() -> Path:
    return get_assets_dir() / "workflows" / "base_txt2img.json"


def test_base_workflow_loads_with_mapping():
    wf = WorkflowTemplate(_base_path())
    assert wf.graph
    for node in wf.graph.values():
        assert "class_type" in node
        assert "inputs" in node
    assert "seed" in wf.mapping
    assert "positive_prompt" in wf.mapping
    assert "lora" in wf.mapping


def test_set_seed_via_mapping():
    wf = WorkflowTemplate(_base_path())
    wf.set_seed(12345)
    spec = wf.mapping["seed"]
    assert wf.get_param(spec["node"], spec["field"]) == 12345


def test_set_prompt_positive_and_negative():
    wf = WorkflowTemplate(_base_path())
    wf.set_prompt("a cat", "ugly")
    pos = wf.mapping["positive_prompt"]
    neg = wf.mapping["negative_prompt"]
    assert wf.get_param(pos["node"], pos["field"]) == "a cat"
    assert wf.get_param(neg["node"], neg["field"]) == "ugly"


def test_set_dimensions():
    wf = WorkflowTemplate(_base_path())
    wf.set_dimensions(768, 1152)
    w = wf.mapping["width"]
    h = wf.mapping["height"]
    assert wf.get_param(w["node"], w["field"]) == 768
    assert wf.get_param(h["node"], h["field"]) == 1152


def test_set_lora_name_and_weights():
    wf = WorkflowTemplate(_base_path())
    wf.set_lora("/some/dir/MyStyle.safetensors", 0.7)
    node = wf.mapping["lora"]["node"]
    assert wf.get_param(node, "lora_name") == "MyStyle.safetensors"
    assert wf.get_param(node, "strength_model") == 0.7
    assert wf.get_param(node, "strength_clip") == 0.7


def test_set_input_image_uses_basename():
    wf = WorkflowTemplate(_base_path())
    wf.set_input_image("3", "/abs/path/pose.png")
    assert wf.get_param("3", "image") == "pose.png"


def test_build_is_independent_deepcopy():
    wf = WorkflowTemplate(_base_path())
    built = wf.build()
    wf.set_seed(99)
    spec = wf.mapping["seed"]
    assert built[spec["node"]]["inputs"][spec["field"]] != 99


def test_unknown_role_raises():
    wf = WorkflowTemplate(_base_path())
    raised = False
    try:
        wf.set_role("does_not_exist", 1)
    except KeyError:
        raised = True
    assert raised


def test_unknown_node_raises():
    wf = WorkflowTemplate(_base_path())
    raised = False
    try:
        wf.set_param("999", "x", 1)
    except KeyError:
        raised = True
    assert raised
