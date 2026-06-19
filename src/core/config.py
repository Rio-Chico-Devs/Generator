"""Configurazione di una richiesta di generazione."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoraSpec:
    """Specifica un LoRA da caricare con il suo peso."""

    path: str
    weight: float = 1.0
    adapter_name: str = "default"


@dataclass
class GenerationConfig:
    """Tutto ciò che serve per produrre un'immagine."""

    # Base
    prompt: str
    negative_prompt: str = ""

    # Dimensioni
    width: int = 1024
    height: int = 1024

    # Sampler
    steps: int = 30
    cfg_scale: float = 7.0
    sampler: str = "dpmpp_2m_karras"  # nome sampler ComfyUI
    seed: int = -1  # -1 = random

    # Batch
    batch_size: int = 1  # immagini generate in una chiamata

    # Modello
    model_id: str = ""  # id da CATALOG
    loras: list[LoraSpec] = field(default_factory=list)

    # img2img / inpaint (opzionali)
    init_image_path: Optional[str] = None
    mask_image_path: Optional[str] = None
    strength: float = 0.7  # img2img only

    # ControlNet (futuro)
    controlnet_image_path: Optional[str] = None
    controlnet_model: Optional[str] = None
    controlnet_weight: float = 1.0

    # Metadata extra
    project_slug: str = ""
    activator_tag: str = ""
