"""Logica della galleria, separata da Qt.

Le immagini generate vivono in ``project.gallery_dir`` accanto a un sidecar
JSON con i parametri effettivi (vedi ``RecipeWorker._write_sidecar``). Qui
si scandisce quella cartella, si caricano i metadati e si ordina/filtra —
tutto senza PyQt6, così è testabile a parte. La GalleryView ci mette sopra
solo i widget.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

# Campi del sidecar riusabili come parametri di una nuova generazione.
REUSABLE_KEYS = (
    "positive",
    "negative",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler",
    "seed",
)


def sidecar_path(image: Path) -> Path:
    """Path del sidecar JSON dei parametri accanto all'immagine."""
    return image.with_suffix(".json")


@dataclass(frozen=True)
class GalleryItem:
    """Un'immagine generata con i suoi metadati (sidecar) e tempo di modifica."""

    path: Path
    metadata: dict = field(default_factory=dict)
    mtime: float = 0.0

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def prompt(self) -> str:
        return str(self.metadata.get("positive", ""))

    @property
    def negative(self) -> str:
        return str(self.metadata.get("negative", ""))

    @property
    def seed(self) -> Optional[int]:
        raw = self.metadata.get("seed")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @property
    def created_at(self) -> str:
        return str(self.metadata.get("created_at", ""))

    @property
    def has_metadata(self) -> bool:
        return bool(self.metadata)

    def matches(self, query: str) -> bool:
        """True se la query (case-insensitive) compare nel prompt o nel nome."""
        q = query.strip().lower()
        if not q:
            return True
        haystack = f"{self.name}\n{self.prompt}\n{self.negative}".lower()
        return q in haystack

    def caption(self) -> str:
        """Riepilogo leggibile dei parametri, per il pannello dettagli."""
        m = self.metadata
        if not m:
            return f"{self.name}\n\n(nessun metadato)"
        lines = [self.name, ""]
        if self.prompt:
            lines.append(f"Prompt: {self.prompt}")
        if self.negative:
            lines.append(f"Negativo: {self.negative}")
        dims = _fmt_dims(m)
        if dims:
            lines.append(f"Dimensioni: {dims}")
        for label, key in (
            ("Seed", "seed"),
            ("Step", "steps"),
            ("CFG", "cfg"),
            ("Sampler", "sampler"),
            ("Modello", "model_id"),
            ("LoRA", "lora_name"),
            ("Creato", "created_at"),
        ):
            val = m.get(key)
            if val not in (None, "", -1):
                lines.append(f"{label}: {val}")
        return "\n".join(lines)

    def reuse_params(self) -> dict:
        """Sottoinsieme dei metadati riusabili per pre-compilare il form."""
        return {k: self.metadata[k] for k in REUSABLE_KEYS if k in self.metadata}


def load_sidecar(image: Path) -> dict:
    """Carica il sidecar JSON di un'immagine; dict vuoto se assente/illeggibile."""
    sc = sidecar_path(image)
    if not sc.exists():
        return {}
    try:
        data = json.loads(sc.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Sidecar illeggibile per %s: %s", image.name, exc)
        return {}


def load_gallery(directory: Path, query: str = "") -> list[GalleryItem]:
    """Scandisce ``directory`` e ritorna gli item ordinati dal più recente.

    Filtra per ``query`` (prompt/nome) se fornita. Le immagini senza sidecar
    sono comunque incluse (con metadati vuoti)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    items: list[GalleryItem] = []
    for entry in directory.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append(
            GalleryItem(path=entry, metadata=load_sidecar(entry), mtime=mtime)
        )
    items.sort(key=lambda it: it.mtime, reverse=True)
    if query.strip():
        items = [it for it in items if it.matches(query)]
    return items


def remove_item(item: GalleryItem) -> None:
    """Elimina l'immagine e il suo sidecar dal disco (best-effort)."""
    for p in (item.path, sidecar_path(item.path)):
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Impossibile eliminare %s: %s", p, exc)


def _fmt_dims(m: dict) -> str:
    w, h = m.get("width"), m.get("height")
    if w and h:
        return f"{w}×{h}"
    return ""
