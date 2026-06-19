"""Preset di training LoRA — configurazioni pre-bilanciate per RTX 3060/8GB VRAM.

Tre preset coprono i casi d'uso tipici senza esporre 30 parametri all'utente:
- Veloce:   SD 1.5, 512px, ~30-45 min, per esperimenti rapidi o dataset piccoli
- Standard: SDXL,   1024px, ~1.5-2.5 ore, default consigliato
- Massimo:  SDXL,   1024px alta fedeltà, ~3-5 ore, dataset grandi (150+ immagini)

Riferimento: docs/TRAINING.md
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PresetId(str, Enum):
    VELOCE = "veloce"
    STANDARD = "standard"
    MASSIMO = "massimo"


@dataclass(frozen=True)
class TrainingPreset:
    id: PresetId
    name: str
    description: str
    model_family: str          # "sd15" | "sdxl"

    # --- Resolution ---
    resolution: str            # es. "512,512" o "1024,1024"
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1536
    bucket_reso_steps: int = 64

    # --- Training loop ---
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_train_epochs: int = 10
    save_every_n_epochs: int = 2

    # --- LoRA network ---
    network_dim: int = 16      # rank: più alto = più capacità, più VRAM
    network_alpha: int = 8     # consigliato = dim / 2
    network_train_unet_only: bool = True  # esclude text encoder → -1.5 GB VRAM

    # --- Optimizer / LR ---
    learning_rate: float = 1e-4
    unet_lr: float = 1e-4
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 100
    optimizer_type: str = "AdamW8bit"

    # --- Precision / memory ---
    mixed_precision: str = "bf16"
    xformers: bool = True
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder_outputs: bool = False
    cache_text_encoder_outputs_to_disk: bool = False
    no_half_vae: bool = False          # VAE SDXL instabile in fp16

    # --- Misc ---
    max_data_loader_n_workers: int = 0  # Windows: deve essere 0
    clip_skip: int = 1

    # --- Sample images durante training ---
    sample_every_n_epochs: int = 2
    sample_sampler: str = "dpm_2"

    # --- Stima risorse ---
    min_vram_gb: float = 5.0
    estimated_time_per_100: str = "30-45 min"


PRESETS: dict[PresetId, TrainingPreset] = {
    PresetId.VELOCE: TrainingPreset(
        id=PresetId.VELOCE,
        name="Veloce",
        description="SD 1.5 · 512px · ~30-45 min · VRAM ≥5 GB",
        model_family="sd15",
        resolution="512,512",
        enable_bucket=False,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=10,
        save_every_n_epochs=2,
        network_dim=16,
        network_alpha=8,
        network_train_unet_only=True,
        learning_rate=1e-4,
        unet_lr=1e-4,
        lr_scheduler="cosine_with_restarts",
        lr_warmup_steps=50,
        optimizer_type="AdamW8bit",
        mixed_precision="fp16",  # fp16 su SD1.5 (bf16 non necessario)
        xformers=True,
        gradient_checkpointing=True,
        cache_latents=True,
        cache_latents_to_disk=False,
        no_half_vae=False,
        clip_skip=2,
        sample_every_n_epochs=2,
        sample_sampler="dpm_2",
        min_vram_gb=5.0,
        estimated_time_per_100="30-45 min",
    ),
    PresetId.STANDARD: TrainingPreset(
        id=PresetId.STANDARD,
        name="Standard",
        description="SDXL · 1024px · ~1.5-2.5 ore · VRAM ≥7.5 GB",
        model_family="sdxl",
        resolution="1024,1024",
        enable_bucket=True,
        min_bucket_reso=512,
        max_bucket_reso=1536,
        bucket_reso_steps=64,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        max_train_epochs=10,
        save_every_n_epochs=2,
        network_dim=16,
        network_alpha=8,
        network_train_unet_only=True,
        learning_rate=1e-4,
        unet_lr=1e-4,
        lr_scheduler="cosine_with_restarts",
        lr_warmup_steps=100,
        optimizer_type="AdamW8bit",
        mixed_precision="bf16",  # bf16 > fp16 su SDXL per stabilità numerica
        xformers=True,
        gradient_checkpointing=True,
        cache_latents=True,
        cache_latents_to_disk=True,
        cache_text_encoder_outputs=True,
        cache_text_encoder_outputs_to_disk=True,
        no_half_vae=True,
        max_data_loader_n_workers=0,
        clip_skip=1,
        sample_every_n_epochs=2,
        sample_sampler="dpmpp_2m",
        min_vram_gb=7.5,
        estimated_time_per_100="90-150 min",
    ),
    PresetId.MASSIMO: TrainingPreset(
        id=PresetId.MASSIMO,
        name="Massimo",
        description="SDXL · rank 32 · alta fedeltà · ~3-5 ore · VRAM ≥7.8 GB",
        model_family="sdxl",
        resolution="1024,1024",
        enable_bucket=True,
        min_bucket_reso=512,
        max_bucket_reso=1536,
        bucket_reso_steps=64,
        train_batch_size=1,
        gradient_accumulation_steps=8,
        max_train_epochs=15,
        save_every_n_epochs=2,
        network_dim=32,    # rank raddoppiato vs Standard
        network_alpha=16,
        network_train_unet_only=True,
        learning_rate=5e-5,  # LR più basso per training più lungo e stabile
        unet_lr=5e-5,
        lr_scheduler="cosine_with_restarts",
        lr_warmup_steps=150,
        optimizer_type="AdamW8bit",
        mixed_precision="bf16",
        xformers=True,
        gradient_checkpointing=True,
        cache_latents=True,
        cache_latents_to_disk=True,
        cache_text_encoder_outputs=True,
        cache_text_encoder_outputs_to_disk=True,
        no_half_vae=True,
        max_data_loader_n_workers=0,
        clip_skip=1,
        sample_every_n_epochs=3,
        sample_sampler="dpmpp_2m",
        min_vram_gb=7.8,
        estimated_time_per_100="180-300 min",
    ),
}


def get_preset(preset_id: PresetId) -> TrainingPreset:
    return PRESETS[preset_id]
