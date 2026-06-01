"""Embed dei parametri di generazione nel PNG (formato A1111) + lettura.

Il chunk tEXt ``parameters`` rende le immagini portabili: AUTOMATIC1111,
ComfyUI, Civitai e simili sanno leggerlo e ricostruire i parametri. Affianca
il sidecar JSON scritto dal RecipeWorker (che resta la fonte ad alta fedeltà,
machine-readable).

Questo modulo è volutamente disaccoppiato dal worker: ``build_a1111_parameters``
accetta un semplice mapping (es. ``dataclasses.asdict(GenerationRecord)``), così
non si crea dipendenza circolare con ``workers/recipe_worker``.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping, Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.utils.atomic import atomic_write_bytes

APP_VERSION = "0.1.0"


def build_a1111_parameters(data: Mapping[str, Any]) -> str:
    """Compone la stringa ``parameters`` in stile AUTOMATIC1111.

    Chiavi lette (tutte opzionali): positive, negative, steps, sampler, cfg,
    seed, width, height, model_id, clip_skip, lora_name, lora_weight.
    """
    positive = str(data.get("positive", "")).strip()
    negative = str(data.get("negative", "")).strip()

    lines: list[str] = [positive]
    if negative:
        lines.append(f"Negative prompt: {negative}")

    meta: list[str] = []
    if data.get("steps") is not None:
        meta.append(f"Steps: {data['steps']}")
    if data.get("sampler"):
        meta.append(f"Sampler: {data['sampler']}")
    if data.get("cfg") is not None:
        meta.append(f"CFG scale: {data['cfg']}")
    if data.get("seed") is not None:
        meta.append(f"Seed: {data['seed']}")
    if data.get("width") and data.get("height"):
        meta.append(f"Size: {data['width']}x{data['height']}")
    if data.get("model_id"):
        meta.append(f"Model: {data['model_id']}")
    if data.get("clip_skip") is not None:
        meta.append(f"Clip skip: {abs(int(data['clip_skip']))}")
    if data.get("lora_name"):
        weight = data.get("lora_weight")
        suffix = f":{weight}" if weight is not None else ""
        meta.append(f"Lora: {data['lora_name']}{suffix}")
    meta.append(f"App: Vihente Forge {APP_VERSION}")

    lines.append(", ".join(meta))
    return "\n".join(lines)


def embed_a1111_parameters(png_path: Path, parameters_text: str) -> bool:
    """Riscrive un PNG esistente aggiungendo il chunk tEXt ``parameters``.

    ComfyUI ha già salvato l'immagine su disco; qui la riapriamo e iniettiamo
    i parametri preservando eventuali metadati ComfyUI già presenti
    (``prompt``/``workflow``). Non solleva mai: ritorna False se il file non è
    un PNG leggibile/scrivibile, così l'embed non blocca la generazione.
    """
    png_path = Path(png_path)
    if png_path.suffix.lower() != ".png":
        return False
    try:
        img = Image.open(png_path)
        img.load()
    except (OSError, ValueError):
        return False

    info = PngInfo()
    for key, value in (img.info or {}).items():
        if key != "parameters" and isinstance(value, str):
            info.add_text(key, value)
    info.add_text("parameters", parameters_text)

    # Serializza in memoria e scrivi atomico: una save() in-place che crasha
    # a metà corromperebbe l'immagine già prodotta da ComfyUI.
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=info, compress_level=4)
        atomic_write_bytes(png_path, buf.getvalue())
    except OSError:
        return False
    return True


def save_image_with_metadata(img: Image.Image, path: Path, parameters_text: str) -> None:
    """Salva un PNG nuovo con il chunk tEXt ``parameters``."""
    info = PngInfo()
    info.add_text("parameters", parameters_text)
    img.save(str(path), format="PNG", pnginfo=info, compress_level=4)


def read_a1111_parameters(path: Path) -> Optional[str]:
    """Legge il chunk ``parameters`` da un PNG. None se assente."""
    img = Image.open(str(path))
    return img.info.get("parameters")
