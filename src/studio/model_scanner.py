"""Scanner modelli e LoRA per Studio Mode.

Scansiona le cartelle studio/, identifica la famiglia di ciascun modello
leggendo l'header safetensors (senza caricarlo in VRAM), legge metadata
LoRA, e valuta compatibilità base/LoRA.

Separazione totale da Forge: NON guarda mai le cartelle dei progetti.

Riferimento: docs/STUDIO_MODE.md
"""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from src.utils.paths import get_user_data_dir

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux"
    UNKNOWN = "unknown"


@dataclass
class ScannedModel:
    path: str
    name: str
    family: ModelFamily
    size_mb: float
    kind: str  # "checkpoint" | "lora" | "vae" | "embedding"
    trigger_words: list[str] = field(default_factory=list)
    base_model_hint: str = ""  # da metadata se presente
    metadata: dict = field(default_factory=dict)


def studio_root() -> Path:
    return get_user_data_dir() / "studio"


def studio_dirs() -> dict[str, Path]:
    root = studio_root()
    return {
        "checkpoint": root / "checkpoints",
        "lora": root / "loras",
        "vae": root / "vae",
        "embedding": root / "embeddings",
    }


def ensure_studio_dirs() -> None:
    for d in studio_dirs().values():
        d.mkdir(parents=True, exist_ok=True)


# --- Header safetensors -------------------------------------------------


# Cap di sicurezza sull'header: un file corrotto/malevolo potrebbe dichiarare
# un header_len enorme (fino a 2^64) e far tentare un'allocazione gigantesca.
# Gli header safetensors reali stanno ben sotto i pochi MB.
_MAX_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB


def read_safetensors_header(path: Path) -> dict:
    """Legge solo l'header JSON di un .safetensors (no caricamento pesi).

    Formato: primi 8 byte = lunghezza header (uint64 LE), poi JSON.
    """
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            return {}
        header_len = struct.unpack("<Q", length_bytes)[0]
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            logger.warning(
                "Header safetensors anomalo in %s: %d byte (cap %d) — ignorato",
                path.name, header_len, _MAX_HEADER_BYTES,
            )
            return {}
        header_bytes = f.read(header_len)
    try:
        return json.loads(header_bytes)
    except json.JSONDecodeError:
        return {}


def detect_family(header: dict) -> ModelFamily:
    """Identifica la famiglia dal contenuto dell'header safetensors.

    Euristiche:
    - metadata kohya 'ss_base_model_version' (LoRA) → diretto
    - chiavi/dimensioni dei tensori → checkpoint
    """
    meta = header.get("__metadata__", {})

    # LoRA kohya: metadata esplicito
    base_ver = str(meta.get("ss_base_model_version", "")).lower()
    if base_ver:
        if "xl" in base_ver or "sdxl" in base_ver:
            return ModelFamily.SDXL
        if "v1" in base_ver or "sd_v1" in base_ver or "1.5" in base_ver:
            return ModelFamily.SD15
        if "flux" in base_ver:
            return ModelFamily.FLUX

    keys = [k for k in header if k != "__metadata__"]
    keys_str = " ".join(keys)

    # FLUX: chiavi caratteristiche
    if "double_blocks" in keys_str or "single_blocks" in keys_str:
        return ModelFamily.FLUX

    # SDXL: ha due text encoder (conditioner.embedders con dim grandi)
    # SD1.5 vs SDXL: cross-attention context dim 768 vs 2048
    for k in keys:
        if "cross_attention" in k or "to_k" in k:
            shape = header[k].get("shape", [])
            if shape:
                dim = shape[-1] if len(shape) >= 1 else 0
                if dim == 2048:
                    return ModelFamily.SDXL
                if dim == 768:
                    return ModelFamily.SD15

    # Fallback su presenza di chiavi tipiche SDXL
    if "conditioner.embedders.1" in keys_str:
        return ModelFamily.SDXL

    return ModelFamily.UNKNOWN


def extract_lora_metadata(header: dict) -> tuple[list[str], str]:
    """Estrae trigger words e base model hint dai metadata LoRA kohya."""
    meta = header.get("__metadata__", {})
    triggers: list[str] = []

    # kohya salva i tag più frequenti del dataset in ss_tag_frequency
    tag_freq_raw = meta.get("ss_tag_frequency", "")
    if tag_freq_raw:
        try:
            tag_freq = json.loads(tag_freq_raw)
            # struttura: {dataset: {tag: count}}
            for _ds, tags in tag_freq.items():
                if isinstance(tags, dict):
                    top = sorted(tags.items(), key=lambda x: -x[1])[:5]
                    triggers.extend(t for t, _ in top)
        except (json.JSONDecodeError, AttributeError):
            pass

    base_hint = str(meta.get("ss_base_model_version", "")) or str(
        meta.get("ss_sd_model_name", "")
    )
    return triggers[:5], base_hint


# --- Scan ---------------------------------------------------------------


def scan_dir(directory: Path, kind: str) -> list[ScannedModel]:
    """Scansiona una cartella per file .safetensors."""
    results: list[ScannedModel] = []
    if not directory.exists():
        return results

    for f in sorted(directory.glob("*.safetensors")):
        try:
            header = read_safetensors_header(f)
            family = detect_family(header)
            size_mb = f.stat().st_size / (1024 * 1024)

            triggers: list[str] = []
            base_hint = ""
            if kind == "lora":
                triggers, base_hint = extract_lora_metadata(header)

            results.append(
                ScannedModel(
                    path=str(f),
                    name=f.stem,
                    family=family,
                    size_mb=round(size_mb, 1),
                    kind=kind,
                    trigger_words=triggers,
                    base_model_hint=base_hint,
                    metadata=header.get("__metadata__", {}),
                )
            )
        except Exception as e:
            logger.warning("Scan fallito per %s: %s", f, e)
    return results


def scan_all() -> dict[str, list[ScannedModel]]:
    """Scansiona tutte le cartelle studio. Ritorna dict per kind."""
    ensure_studio_dirs()
    dirs = studio_dirs()
    return {
        "checkpoint": scan_dir(dirs["checkpoint"], "checkpoint"),
        "lora": scan_dir(dirs["lora"], "lora"),
        "vae": scan_dir(dirs["vae"], "vae"),
        "embedding": scan_dir(dirs["embedding"], "embedding"),
    }


def compatible_loras(
    base: ScannedModel, loras: list[ScannedModel]
) -> list[ScannedModel]:
    """Filtra i LoRA compatibili con il modello base scelto."""
    return [
        lo
        for lo in loras
        if lo.family == base.family or lo.family == ModelFamily.UNKNOWN
    ]


def is_compatible(base: ScannedModel, lora: ScannedModel) -> bool:
    if lora.family == ModelFamily.UNKNOWN:
        return True  # non possiamo escludere, lasciamo provare con warning
    return lora.family == base.family
