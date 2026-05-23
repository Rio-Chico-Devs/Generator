"""Embed metadata in PNG + sidecar JSON.

Formato A1111-compatible nel tEXt chunk con chiave "parameters", così le
immagini sono portabili verso AUTOMATIC1111 / ComfyUI / etc.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.core.config import GenerationConfig, GenerationResult

APP_VERSION = "0.1.0"


def build_a1111_parameters_string(cfg: GenerationConfig, seed: int) -> str:
    """Formato testuale stile AUTOMATIC1111 webui."""
    lines: list[str] = []
    lines.append(cfg.prompt)
    if cfg.negative_prompt:
        lines.append(f"Negative prompt: {cfg.negative_prompt}")

    meta_parts = [
        f"Steps: {cfg.steps}",
        f"Sampler: {cfg.sampler}",
        f"CFG scale: {cfg.cfg_scale}",
        f"Seed: {seed}",
        f"Size: {cfg.width}x{cfg.height}",
    ]
    if cfg.model_id:
        meta_parts.append(f"Model: {cfg.model_id}")
    if cfg.loras:
        lora_strs = [f"{l.adapter_name}:{l.weight:.2f}" for l in cfg.loras]
        meta_parts.append(f"Lora: {', '.join(lora_strs)}")
    meta_parts.append(f"App: Vihente Forge {APP_VERSION}")

    lines.append(", ".join(meta_parts))
    return "\n".join(lines)


def save_image_with_metadata(img: Image.Image, path: Path, parameters_text: str) -> None:
    """Salva PNG con chunk tEXt 'parameters'."""
    info = PngInfo()
    info.add_text("parameters", parameters_text)
    img.save(str(path), format="PNG", pnginfo=info, compress_level=4)


def read_a1111_parameters(path: Path) -> str | None:
    """Legge il chunk 'parameters' da una PNG generata. None se assente."""
    img = Image.open(str(path))
    return img.info.get("parameters")


def write_sidecar_json(path: Path, result: GenerationResult) -> None:
    """Serializza GenerationResult come sidecar."""
    d = _result_to_dict(result)
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_sidecar_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_to_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "image_path": result.image_path,
        "actual_seed": result.actual_seed,
        "generation_time_sec": result.generation_time_sec,
        "model_hash": result.model_hash,
        "app_version": result.app_version,
        "created_at": result.created_at,
        "config": asdict(result.config),
    }
