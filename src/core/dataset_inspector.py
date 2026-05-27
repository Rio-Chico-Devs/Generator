"""Logica del Dataset Inspector — scanning, statistiche tag, editing caption.

Separato da Qt per permettere test puri. La DatasetView ci mette sopra i widget.

Il dataset di Vihente Forge segue il formato kohya/sd-scripts:
- immagine: any di IMAGE_EXTS
- caption: stesso nome con estensione .txt, tag comma-separated

Questo modulo è usato sia per la cartella images/ del dataset che per la
cartella negatives/ (stessa struttura, scopo diverso).
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})


@dataclass(frozen=True)
class DatasetItem:
    """Un'immagine nel dataset con la sua caption."""

    image_path: Path
    caption_path: Path  # .txt accanto all'immagine; può non esistere

    @property
    def name(self) -> str:
        return self.image_path.name

    @property
    def caption(self) -> str:
        """Testo grezzo della caption; stringa vuota se il file non esiste."""
        if not self.caption_path.exists():
            return ""
        try:
            return self.caption_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @property
    def tags(self) -> list[str]:
        """Tag comma-separated dal file .txt; lista vuota se nessuna caption."""
        text = self.caption
        if not text:
            return []
        return [t.strip() for t in text.split(",") if t.strip()]

    @property
    def has_caption(self) -> bool:
        return bool(self.caption)

    def matches_tag(self, tag: str) -> bool:
        """True se il tag (case-insensitive, sottostringa) compare nei tag dell'item."""
        tag_l = tag.strip().lower()
        return any(tag_l in t.lower() for t in self.tags)


def scan_dataset(directory: Path) -> list[DatasetItem]:
    """Scansiona directory e ritorna un DatasetItem per ogni immagine trovata.

    Ordinato per nome file. Le immagini senza .txt sono incluse (caption vuota)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    items: list[DatasetItem] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTS:
            continue
        items.append(DatasetItem(image_path=entry, caption_path=entry.with_suffix(".txt")))
    return items


def tag_frequency(items: list[DatasetItem]) -> Counter:
    """Conta quante immagini contengono ciascun tag (una volta per immagine).

    Usa set per non sommare ripetizioni nella stessa caption. Ritorna Counter
    ordinabile per most_common()."""
    counter: Counter = Counter()
    for item in items:
        for tag in set(item.tags):
            counter[tag] += 1
    return counter


def dataset_stats(items: list[DatasetItem]) -> dict:
    """Statistiche aggregate: totale immagini, totale tag unici, media tag/img."""
    freq = tag_frequency(items)
    total_tags_per_item = sum(len(it.tags) for it in items)
    return {
        "total_images": len(items),
        "unique_tags": len(freq),
        "avg_tags_per_image": round(total_tags_per_item / len(items), 1) if items else 0.0,
        "images_without_caption": sum(1 for it in items if not it.has_caption),
    }


def save_caption(item: DatasetItem, text: str) -> None:
    """Scrive (o aggiorna) il file .txt della caption.

    Se text è vuoto, rimuove il file. Solleva OSError in caso di errore I/O."""
    text = text.strip()
    if not text:
        try:
            item.caption_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Impossibile rimuovere caption %s: %s", item.caption_path, exc)
        return
    try:
        item.caption_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("Impossibile salvare caption %s: %s", item.caption_path, exc)
        raise


def remove_dataset_item(item: DatasetItem) -> None:
    """Rimuove immagine e caption dal dataset (best-effort, non solleva)."""
    for p in (item.image_path, item.caption_path):
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Impossibile eliminare %s: %s", p, exc)
