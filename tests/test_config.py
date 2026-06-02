"""Test per src/core/config — GenerationConfig e LoraSpec.

Nessuna dipendenza esterna. Copre: default, campi espliciti,
isolamento liste mutabili, tipi opzionali, campi img2img e ControlNet.
"""
from __future__ import annotations

from src.core.config import GenerationConfig, LoraSpec


# ---------------------------------------------------------------------------
# GenerationConfig — default
# ---------------------------------------------------------------------------


def test_default_prompt():
    c = GenerationConfig(prompt="hello")
    assert c.prompt == "hello"


def test_default_negative_prompt():
    assert GenerationConfig(prompt="x").negative_prompt == ""


def test_default_dimensions():
    c = GenerationConfig(prompt="x")
    assert c.width == 1024
    assert c.height == 1024


def test_default_steps():
    assert GenerationConfig(prompt="x").steps == 30


def test_default_cfg_scale():
    assert GenerationConfig(prompt="x").cfg_scale == 7.0


def test_default_sampler():
    assert GenerationConfig(prompt="x").sampler == "dpmpp_2m_karras"


def test_default_seed():
    assert GenerationConfig(prompt="x").seed == -1


def test_default_batch_size():
    assert GenerationConfig(prompt="x").batch_size == 1


def test_default_model_id_empty():
    assert GenerationConfig(prompt="x").model_id == ""


def test_default_loras_empty_list():
    assert GenerationConfig(prompt="x").loras == []


def test_default_init_image_none():
    assert GenerationConfig(prompt="x").init_image_path is None


def test_default_mask_image_none():
    assert GenerationConfig(prompt="x").mask_image_path is None


def test_default_strength():
    assert GenerationConfig(prompt="x").strength == 0.7


def test_default_controlnet_image_none():
    assert GenerationConfig(prompt="x").controlnet_image_path is None


def test_default_controlnet_model_none():
    assert GenerationConfig(prompt="x").controlnet_model is None


def test_default_controlnet_weight():
    assert GenerationConfig(prompt="x").controlnet_weight == 1.0


def test_default_project_slug_empty():
    assert GenerationConfig(prompt="x").project_slug == ""


def test_default_activator_tag_empty():
    assert GenerationConfig(prompt="x").activator_tag == ""


# ---------------------------------------------------------------------------
# GenerationConfig — valori espliciti
# ---------------------------------------------------------------------------


def test_explicit_values():
    c = GenerationConfig(
        prompt="iris",
        negative_prompt="blurry",
        width=512,
        height=768,
        steps=20,
        cfg_scale=6.5,
        sampler="euler_a",
        seed=42,
        batch_size=4,
        model_id="sdxl_base",
    )
    assert c.prompt == "iris"
    assert c.negative_prompt == "blurry"
    assert c.width == 512
    assert c.height == 768
    assert c.steps == 20
    assert c.cfg_scale == 6.5
    assert c.sampler == "euler_a"
    assert c.seed == 42
    assert c.batch_size == 4
    assert c.model_id == "sdxl_base"


def test_project_and_tag_fields():
    c = GenerationConfig(
        prompt="iris",
        project_slug="iris-proj",
        activator_tag="iris_v1",
    )
    assert c.project_slug == "iris-proj"
    assert c.activator_tag == "iris_v1"


# ---------------------------------------------------------------------------
# Isolamento liste mutabili
# ---------------------------------------------------------------------------


def test_loras_independent_between_instances():
    a = GenerationConfig(prompt="a")
    b = GenerationConfig(prompt="b")
    a.loras.append(LoraSpec(path="x.safetensors"))
    assert b.loras == []


def test_loras_do_not_share_list_with_default():
    c1 = GenerationConfig(prompt="a")
    c2 = GenerationConfig(prompt="b")
    c1.loras.append(LoraSpec(path="x.safetensors"))
    assert len(c2.loras) == 0


# ---------------------------------------------------------------------------
# img2img
# ---------------------------------------------------------------------------


def test_img2img_fields():
    c = GenerationConfig(
        prompt="x",
        init_image_path="/tmp/input.png",
        mask_image_path="/tmp/mask.png",
        strength=0.5,
    )
    assert c.init_image_path == "/tmp/input.png"
    assert c.mask_image_path == "/tmp/mask.png"
    assert c.strength == 0.5


def test_strength_zero():
    c = GenerationConfig(prompt="x", strength=0.0)
    assert c.strength == 0.0


def test_strength_one():
    c = GenerationConfig(prompt="x", strength=1.0)
    assert c.strength == 1.0


# ---------------------------------------------------------------------------
# ControlNet
# ---------------------------------------------------------------------------


def test_controlnet_fields():
    c = GenerationConfig(
        prompt="x",
        controlnet_image_path="/tmp/pose.png",
        controlnet_model="openpose",
        controlnet_weight=0.75,
    )
    assert c.controlnet_image_path == "/tmp/pose.png"
    assert c.controlnet_model == "openpose"
    assert c.controlnet_weight == 0.75


def test_controlnet_weight_zero():
    c = GenerationConfig(prompt="x", controlnet_weight=0.0)
    assert c.controlnet_weight == 0.0


# ---------------------------------------------------------------------------
# LoraSpec — default e valori espliciti
# ---------------------------------------------------------------------------


def test_lora_spec_default_weight():
    spec = LoraSpec(path="style.safetensors")
    assert spec.weight == 1.0


def test_lora_spec_default_adapter_name():
    assert LoraSpec(path="style.safetensors").adapter_name == "default"


def test_lora_spec_custom_weight():
    spec = LoraSpec(path="style.safetensors", weight=0.75)
    assert spec.weight == 0.75


def test_lora_spec_custom_adapter_name():
    spec = LoraSpec(path="x.safetensors", adapter_name="my_lora")
    assert spec.adapter_name == "my_lora"


def test_lora_spec_zero_weight():
    assert LoraSpec(path="x.safetensors", weight=0.0).weight == 0.0


def test_lora_spec_negative_weight():
    """Valori negativi non vengono validati qui (validazione a livello superiore)."""
    spec = LoraSpec(path="x.safetensors", weight=-0.5)
    assert spec.weight == -0.5


def test_lora_spec_path_preserved():
    path = "/some/path/to/style_v2.safetensors"
    assert LoraSpec(path=path).path == path


# ---------------------------------------------------------------------------
# GenerationConfig con più LoRA
# ---------------------------------------------------------------------------


def test_multiple_loras():
    specs = [LoraSpec(path=f"lora{i}.safetensors", weight=0.5) for i in range(3)]
    c = GenerationConfig(prompt="x", loras=specs)
    assert len(c.loras) == 3
    assert c.loras[0].path == "lora0.safetensors"
    assert c.loras[2].path == "lora2.safetensors"


def test_lora_list_accessible_after_creation():
    lo = LoraSpec(path="lora.safetensors", weight=0.8, adapter_name="custom")
    c = GenerationConfig(prompt="x", loras=[lo])
    assert c.loras[0].weight == 0.8
    assert c.loras[0].adapter_name == "custom"
