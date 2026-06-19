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


# ---------------------------------------------------------------------------
# set_loras — stacking di più LoRA
# ---------------------------------------------------------------------------


def _lora_node_id(wf: WorkflowTemplate) -> str:
    return wf.mapping["lora"]["node"]


def _count_lora_nodes(wf: WorkflowTemplate) -> int:
    return sum(1 for n in wf.graph.values() if n["class_type"] == "LoraLoader")


def test_set_loras_single_reuses_head_node():
    wf = WorkflowTemplate(_base_path())
    wf.set_loras([("/d/Style.safetensors", 0.9)])
    head = _lora_node_id(wf)
    assert _count_lora_nodes(wf) == 1
    assert wf.get_param(head, "lora_name") == "Style.safetensors"
    assert wf.get_param(head, "strength_model") == 0.9
    assert wf.get_param(head, "strength_clip") == 0.9


def test_set_loras_stacks_three_in_order():
    wf = WorkflowTemplate(_base_path())
    head = _lora_node_id(wf)
    specs = [
        ("/d/A.safetensors", 0.8),
        ("/d/B.safetensors", 0.6),
        ("/d/C.safetensors", 0.4),
    ]
    wf.set_loras(specs)
    assert _count_lora_nodes(wf) == 3

    # Il primo è il nodo head, alimentato dal checkpoint (nodo "4").
    assert wf.get_param(head, "lora_name") == "A.safetensors"
    assert wf.get_param(head, "model") == ["4", 0]
    assert wf.get_param(head, "clip") == ["4", 1]

    # I due cloni leggono dal precedente.
    b_id = f"{head}_lora1"
    c_id = f"{head}_lora2"
    assert wf.get_param(b_id, "lora_name") == "B.safetensors"
    assert wf.get_param(b_id, "model") == [head, 0]
    assert wf.get_param(c_id, "lora_name") == "C.safetensors"
    assert wf.get_param(c_id, "model") == [b_id, 0]


def test_set_loras_rewires_consumers_to_tail():
    wf = WorkflowTemplate(_base_path())
    head = _lora_node_id(wf)
    wf.set_loras([("/d/A.safetensors", 0.8), ("/d/B.safetensors", 0.6)])
    tail = f"{head}_lora1"
    # KSampler (nodo "3") legge il model dalla coda della catena.
    assert wf.get_param("3", "model") == [tail, 0]
    # CLIPSetLastLayer (nodo "11") legge il clip dalla coda.
    assert wf.get_param("11", "clip") == [tail, 1]


def test_set_loras_empty_bypasses_node():
    wf = WorkflowTemplate(_base_path())
    head = _lora_node_id(wf)
    wf.set_loras([])
    assert _count_lora_nodes(wf) == 0
    assert head not in wf.graph
    # I consumatori tornano a leggere dal checkpoint (nodo "4").
    assert wf.get_param("3", "model") == ["4", 0]
    assert wf.get_param("11", "clip") == ["4", 1]


def test_set_loras_uses_basename():
    wf = WorkflowTemplate(_base_path())
    wf.set_loras([("/long/abs/path/Hero.safetensors", 1.0)])
    head = _lora_node_id(wf)
    assert wf.get_param(head, "lora_name") == "Hero.safetensors"


def test_set_loras_build_is_independent():
    wf = WorkflowTemplate(_base_path())
    wf.set_loras([("/d/A.safetensors", 0.8), ("/d/B.safetensors", 0.6)])
    built = wf.build()
    # build() è un deepcopy: contiene la catena completa.
    head = _lora_node_id(wf)
    assert f"{head}_lora1" in built
