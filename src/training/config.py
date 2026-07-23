"""Generazione config TOML per sd-scripts (kohya).

La config è sempre generata da parametri validati, mai assemblata
via string concatenation. Usa il package `toml` (già in requirements.txt).

Sicurezza (da docs/TRAINING.md):
- network_dim cappato a 128 (evita LoRA enormi accidentali)
- max_train_epochs cappato a 50
- Tutti i path sono oggetti Path, non stringhe grezze utente
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml

from src.training.presets import TrainingPreset

# Limiti di sicurezza hardware (da docs/TRAINING.md)
_MAX_NETWORK_DIM = 128
_MAX_TRAIN_EPOCHS = 50
_MIN_TRAIN_EPOCHS = 1
_MAX_LR = 5e-3
_MIN_LR = 1e-6

# Prompt sample auto-generati se l'utente non ne fornisce
_DEFAULT_SAMPLE_PROMPTS = [
    "{tag}, 1girl, simple pose, white background",
    "{tag}, full body, dynamic angle, masterpiece",
    "{tag}, portrait, soft lighting, best quality",
]


@dataclass
class TrainingParams:
    """Parametri configurabili dall'utente oltre al preset selezionato."""

    epochs: int = 10
    network_dim: Optional[int] = None     # None = usa preset
    network_alpha: Optional[int] = None   # None = usa preset
    learning_rate: Optional[float] = None # None = usa preset
    lr_scheduler: Optional[str] = None    # None = usa preset
    optimizer_type: Optional[str] = None  # None = usa preset
    train_text_encoder: bool = False       # richiede VRAM extra
    noise_offset: float = 0.0             # 0.0-0.1, contrasto output
    min_snr_gamma: float = 5.0            # convergenza
    custom_sample_prompts: Optional[list[str]] = None
    resume_from: Optional[Path] = None    # checkpoint per resume da crash

    def __post_init__(self) -> None:
        if self.custom_sample_prompts is None:
            self.custom_sample_prompts = []


def generate_toml(
    preset: TrainingPreset,
    params: TrainingParams,
    base_model_path: Path,
    dataset_dir: Path,
    run_dir: Path,
    activator_tag: str,
    seed: int = 42,
) -> str:
    """Genera il contenuto TOML per sd-scripts e ritorna la stringa.

    Il chiamante scrive il file su disco atomicamente.
    Tutti i parametri vengono validati prima della generazione.
    """
    epochs = _clamp_int(params.epochs, _MIN_TRAIN_EPOCHS, _MAX_TRAIN_EPOCHS)
    dim = _clamp_int(params.network_dim or preset.network_dim, 4, _MAX_NETWORK_DIM)
    alpha = _clamp_int(params.network_alpha or preset.network_alpha, 1, dim)
    lr = _clamp_float(params.learning_rate or preset.learning_rate, _MIN_LR, _MAX_LR)
    scheduler = params.lr_scheduler or preset.lr_scheduler
    optimizer = params.optimizer_type or preset.optimizer_type

    # Sample prompts file
    prompts_path = run_dir / "sample_prompts.txt"
    _write_sample_prompts(prompts_path, params.custom_sample_prompts, activator_tag)

    # Cartelle di output
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    samples_dir = run_dir / "samples"
    for d in (checkpoints_dir, logs_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg: dict = {
        # Paths
        "pretrained_model_name_or_path": str(base_model_path),
        "train_data_dir": str(dataset_dir),
        # Il resto dell'app (Dataset Inspector, prepare_dataset) scrive/legge
        # caption in file .txt; senza dirlo esplicitamente a sd-scripts, il
        # suo default (.caption) fa ignorare TUTTE le caption in silenzio —
        # il training procede solo col tag attivatore, senza errori visibili.
        "caption_extension": ".txt",
        "output_dir": str(checkpoints_dir),
        "output_name": "lora",
        "save_model_as": "safetensors",
        "logging_dir": str(logs_dir),
        # Resolution
        "resolution": preset.resolution,
        # Training loop
        "train_batch_size": preset.train_batch_size,
        "gradient_accumulation_steps": preset.gradient_accumulation_steps,
        "max_train_epochs": epochs,
        "save_every_n_epochs": preset.save_every_n_epochs,
        # LoRA network
        "network_module": "networks.lora",
        "network_dim": dim,
        "network_alpha": alpha,
        "network_train_unet_only": (
            preset.network_train_unet_only and not params.train_text_encoder
        ),
        # Optimizer
        "learning_rate": lr,
        "unet_lr": lr,
        "lr_scheduler": scheduler,
        "lr_warmup_steps": preset.lr_warmup_steps,
        "optimizer_type": optimizer,
        # Precision / memory
        "mixed_precision": preset.mixed_precision,
        # Attention efficiente in memoria. Usiamo SDPA (integrato in PyTorch):
        # niente dipendenza xformers, che su Windows è fragile da agganciare
        # alla versione esatta di torch. Stessa resa in VRAM per LoRA SDXL.
        "sdpa": True,
        "gradient_checkpointing": preset.gradient_checkpointing,
        "cache_latents": preset.cache_latents,
        # Misc
        "seed": seed,
        "clip_skip": preset.clip_skip,
        "max_data_loader_n_workers": preset.max_data_loader_n_workers,
        # Sample images durante training
        "sample_every_n_epochs": preset.sample_every_n_epochs,
        "sample_prompts": str(prompts_path),
        "sample_sampler": preset.sample_sampler,
        "sample_at_first": True,
        # Stato per --resume (ottimizzatore/scheduler/RNG, non solo i pesi):
        # senza questo "Riprendi da crash" punterebbe --resume a un file
        # .safetensors, che sd-scripts rifiuta o interpreta male. Teniamo solo
        # l'ultimo stato salvato (save_last_n_epochs_state=1): ogni stato pesa
        # quanto il checkpoint stesso, tenerli tutti satura il disco in fretta.
        "save_state": True,
        "save_last_n_epochs_state": 1,
    }

    # Bucket (SDXL)
    if preset.enable_bucket:
        cfg["enable_bucket"] = True
        cfg["min_bucket_reso"] = preset.min_bucket_reso
        cfg["max_bucket_reso"] = preset.max_bucket_reso
        cfg["bucket_reso_steps"] = preset.bucket_reso_steps

    # Cache latents su disco (risparmia VRAM a regime)
    if preset.cache_latents_to_disk:
        cfg["cache_latents_to_disk"] = True

    # Cache text encoder outputs (risparmia VRAM significativa su SDXL).
    # Incompatibile con l'addestramento del text encoder (sd-scripts lo
    # rifiuta): se train_text_encoder è attivo, la cache va disabilitata
    # anche se il preset la richiederebbe di default.
    if preset.cache_text_encoder_outputs and not params.train_text_encoder:
        cfg["cache_text_encoder_outputs"] = True
    if preset.cache_text_encoder_outputs_to_disk and not params.train_text_encoder:
        cfg["cache_text_encoder_outputs_to_disk"] = True

    # VAE SDXL: no_half_vae evita artefatti in bf16
    if preset.no_half_vae:
        cfg["no_half_vae"] = True

    # Text encoder training (se abilitato, disabilita network_train_unet_only)
    if params.train_text_encoder:
        cfg["network_train_unet_only"] = False

    # Parametri avanzati
    if params.noise_offset > 0.0:
        cfg["noise_offset"] = round(params.noise_offset, 4)
    if params.min_snr_gamma != 5.0:
        cfg["min_snr_gamma"] = params.min_snr_gamma

    # Resume da crash
    if params.resume_from:
        cfg["resume"] = str(params.resume_from)

    return toml.dumps(cfg)


def _write_sample_prompts(path: Path, custom: list[str], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompts = custom if custom else [p.format(tag=tag) for p in _DEFAULT_SAMPLE_PROMPTS]
    path.write_text("\n".join(prompts), encoding="utf-8")


def _clamp_int(v: int | float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def sdscripts_launch_cmd(
    sdscripts_dir: Path,
    config_path: Path,
    model_family: str = "sdxl",
) -> list[str]:
    """Costruisce il comando per lanciare sd-scripts via accelerate.

    Non esegue nulla: ritorna la lista di argomenti per subprocess.Popen.

    Ritorna lista vuota se sd-scripts non è installato nella cartella attesa.
    """
    sdscripts_dir = Path(sdscripts_dir)
    script = (
        sdscripts_dir / "sdxl_train_network.py"
        if model_family == "sdxl"
        else sdscripts_dir / "train_network.py"
    )
    if not script.exists():
        return []

    # sd-scripts vive in un venv DEDICATO (le sue dipendenze romperebbero
    # quelle di ComfyUI nel venv principale). Usiamo l'accelerate di quel venv
    # se presente; altrimenti ci affidiamo al PATH.
    accelerate = _sdscripts_accelerate(sdscripts_dir)

    return [
        accelerate,
        "launch",
        "--num_cpu_threads_per_process", "2",
        str(script),
        "--config_file", str(config_path),
    ]


def _sdscripts_accelerate(sdscripts_dir: Path) -> str:
    """Percorso dell'eseguibile ``accelerate`` del venv dedicato di sd-scripts.

    Cerca un venv (``.venv`` o ``.venv-sdscripts``) dentro sdscripts_dir; se
    non lo trova, ritorna ``"accelerate"`` (risolto via PATH)."""
    import sys

    for venv in (sdscripts_dir / ".venv", sdscripts_dir / ".venv-sdscripts"):
        exe = (
            venv / "Scripts" / "accelerate.exe"
            if sys.platform == "win32"
            else venv / "bin" / "accelerate"
        )
        if exe.exists():
            return str(exe)
    return "accelerate"
