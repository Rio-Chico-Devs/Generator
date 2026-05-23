from __future__ import annotations

from src.core.config import GenerationConfig, LoraSpec


def test_generation_config_defaults():
    c = GenerationConfig(prompt="hello")
    assert c.prompt == "hello"
    assert c.width == 1024 and c.height == 1024
    assert c.steps == 30
    assert c.seed == -1
    assert c.batch_size == 1
    assert c.loras == []


def test_loras_independent_between_instances():
    a = GenerationConfig(prompt="a")
    b = GenerationConfig(prompt="b")
    a.loras.append(LoraSpec(path="x.safetensors"))
    assert b.loras == []


def test_lora_spec_defaults():
    spec = LoraSpec(path="style.safetensors")
    assert spec.weight == 1.0
    assert spec.adapter_name == "default"
