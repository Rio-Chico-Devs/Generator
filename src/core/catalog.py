"""Catalogo dei modelli base supportati.

Riferimento: docs/MODELS.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ModelFamily = Literal["sdxl", "sd15", "flux"]
UseCase = Literal["character", "photo", "illustration", "abstract", "logo"]


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    family: ModelFamily
    repo_id: str
    revision: Optional[str]
    size_gb: float
    license: str
    license_url: str
    suitable_for: tuple[UseCase, ...]
    requires_vae: Optional[str] = None
    min_vram_gb_inference: float = 5.0
    min_vram_gb_training: float = 7.0
    commercial_use_ok: bool = True
    description: str = ""


CATALOG: dict[str, ModelEntry] = {
    "pony-v6-xl": ModelEntry(
        id="pony-v6-xl",
        name="Pony Diffusion V6 XL",
        family="sdxl",
        repo_id="AstraliteHeart/pony-diffusion-v6-xl",
        revision=None,
        size_gb=7.0,
        license="Fair AI Public 1.0-SD",
        license_url="https://freedevproject.org/faipl-1.0-sd/",
        suitable_for=("character", "illustration"),
        requires_vae="sdxl-vae-fp16-fix",
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=True,
        description=(
            "Modello SDXL specializzato in character art e illustrazione, "
            "ottimizzato per tag Danbooru. Ottimo per personaggi originali, "
            "concept art, sprite stilizzati."
        ),
    ),
    "juggernaut-xl-v9": ModelEntry(
        id="juggernaut-xl-v9",
        name="Juggernaut XL v9",
        family="sdxl",
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        revision=None,
        size_gb=7.0,
        license="OpenRAIL++-M",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        suitable_for=("photo", "illustration"),
        requires_vae="sdxl-vae-fp16-fix",
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=True,
        description=(
            "Fine-tune SDXL ottimizzato per fotorealismo. "
            "Ottimo per render prodotti, mockup, asset realistici."
        ),
    ),
    "animagine-xl-4": ModelEntry(
        id="animagine-xl-4",
        name="Animagine XL 4.0",
        family="sdxl",
        repo_id="cagliostrolab/animagine-xl-4.0",
        revision=None,
        size_gb=7.0,
        license="Fair AI Public 1.0-SD",
        license_url="https://freedevproject.org/faipl-1.0-sd/",
        suitable_for=("character", "illustration"),
        requires_vae="sdxl-vae-fp16-fix",
        min_vram_gb_inference=5.5,
        min_vram_gb_training=7.5,
        commercial_use_ok=True,
        description=(
            "Modello SDXL per illustrazione anime/manga alta qualità. "
            "Alternativa a Pony con output più 'pulito'."
        ),
    ),
    "realistic-vision-v6": ModelEntry(
        id="realistic-vision-v6",
        name="Realistic Vision V6 (SD 1.5)",
        family="sd15",
        repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        revision=None,
        size_gb=4.0,
        license="CreativeML OpenRAIL-M",
        license_url="https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        suitable_for=("photo", "illustration"),
        min_vram_gb_inference=4.0,
        min_vram_gb_training=5.5,
        commercial_use_ok=True,
        description=(
            "Modello SD 1.5 leggero per fotorealismo veloce. "
            "Training in 30-45 min, ideale per esperimenti."
        ),
    ),
    "sd-1-5": ModelEntry(
        id="sd-1-5",
        name="Stable Diffusion 1.5",
        family="sd15",
        repo_id="runwayml/stable-diffusion-v1-5",
        revision=None,
        size_gb=4.0,
        license="CreativeML OpenRAIL-M",
        license_url="https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        suitable_for=("character", "illustration", "abstract"),
        min_vram_gb_inference=4.0,
        min_vram_gb_training=5.5,
        commercial_use_ok=True,
        description=(
            "Modello SD 1.5 base. Versatile, leggero, training rapido. "
            "Sweet spot per sprite, pixel art, esperimenti."
        ),
    ),
}


# VAE separati (sostituiscono i VAE buggy dei modelli base)
@dataclass(frozen=True)
class VaeEntry:
    id: str
    repo_id: str
    filename: str
    size_mb: float
    license: str


VAE_CATALOG: dict[str, VaeEntry] = {
    "sdxl-vae-fp16-fix": VaeEntry(
        id="sdxl-vae-fp16-fix",
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="sdxl_vae.safetensors",
        size_mb=335.0,
        license="MIT",
    ),
}


# Helper per suggerire modello in base al caso d'uso
def suggest_models_for(use_case: UseCase, prefer_fast: bool = False) -> list[str]:
    """Restituisce ID modelli ordinati per adeguatezza."""
    matches = [
        (mid, m) for mid, m in CATALOG.items() if use_case in m.suitable_for
    ]
    # Ordina: SDXL prima se prefer_fast=False, SD 1.5 prima se True
    matches.sort(
        key=lambda x: (
            x[1].family == "sd15" if not prefer_fast else x[1].family != "sd15",
            x[1].size_gb,
        )
    )
    return [mid for mid, _ in matches]
