"""Preparazione dataset per il training LoRA.

Fase 1 della pipeline di training (vedi docs/TRAINING.md):
1. Legge le immagini dalla cartella images/ del progetto
2. Filtra quelle escluse (excluded_from_training nel manifest)
3. Verifica che ogni immagine abbia un file .txt di caption
4. Crea la struttura di cartelle per sd-scripts sotto processed/
5. Ritorna un DatasetReport con conteggi e warning

La struttura attesa da sd-scripts (kohya format):
    processed/
      {repeats}_{concept}/
        image1.png
        image1.txt
        ...

Dove `concept` è il tag attivatore del progetto e `repeats` è il numero
di volte che sd-scripts ripete ogni immagine per epoch — calcolato
automaticamente in base alla dimensione del dataset.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_MIN_IMAGES = 5
_TARGET_STEPS_PER_EPOCH = 200   # step target per epoch per calcolo repeats


@dataclass
class DatasetReport:
    """Esito della preparazione del dataset."""

    total_images: int = 0
    valid_images: int = 0
    skipped_excluded: int = 0
    missing_captions: int = 0         # immagini senza .txt → caption vuota creata
    corrupt_images: int = 0
    output_dir: Path = field(default_factory=Path)
    repeats: int = 1
    warnings: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.valid_images >= _MIN_IMAGES

    @property
    def readiness_message(self) -> str:
        if self.valid_images == 0:
            return "Nessuna immagine valida trovata nel dataset."
        if self.valid_images < _MIN_IMAGES:
            return (
                f"Solo {self.valid_images} immagini valide. "
                f"Servono almeno {_MIN_IMAGES} per avviare il training."
            )
        return f"{self.valid_images} immagini pronte per il training."


def prepare_dataset(
    images_dir: Path,
    processed_dir: Path,
    activator_tag: str,
    manifest: dict | None = None,
    clean_existing: bool = True,
) -> DatasetReport:
    """Prepara la cartella dataset per sd-scripts.

    Parametri
    ---------
    images_dir:     cartella con le immagini originali del progetto
    processed_dir:  dove creare la struttura kohya format
    activator_tag:  tag attivatore del progetto (es. "vf_iris_v1")
    manifest:       dict manifest.json per leggere exclusions (opzionale)
    clean_existing: se True, cancella processed_dir prima di ricrearla

    Ritorna un DatasetReport con i conteggi e i warning.
    """
    report = DatasetReport()
    images_dir = Path(images_dir)
    processed_dir = Path(processed_dir)

    if not images_dir.exists():
        report.warnings.append(f"Cartella immagini non trovata: {images_dir}")
        return report

    # Raccoglie tutte le immagini
    all_images = [
        f for f in sorted(images_dir.iterdir())
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
    ]
    report.total_images = len(all_images)

    # Legge exclusions dal manifest
    excluded: set[str] = set()
    if manifest:
        images_meta = manifest.get("images", {})
        excluded = {
            name for name, meta in images_meta.items()
            if meta.get("excluded_from_training") is True
        }

    # Filtra immagini valide
    valid: list[Path] = []
    for img in all_images:
        if img.name in excluded:
            report.skipped_excluded += 1
            continue
        if not _is_readable_image(img):
            report.corrupt_images += 1
            logger.warning("Immagine corrotta o illeggibile, saltata: %s", img.name)
            continue
        valid.append(img)

    report.valid_images = len(valid)

    if not report.is_ready:
        return report

    # Calcola repeats: vogliamo ~200 step/epoch
    # step/epoch = ceil(n_images * repeats / batch_size)
    # Con batch_size=1: repeats ≈ TARGET / n_images
    report.repeats = max(1, _TARGET_STEPS_PER_EPOCH // max(1, len(valid)))

    # Crea struttura kohya: processed/{repeats}_{activator_tag}/
    concept_dir_name = f"{report.repeats}_{_safe_tag(activator_tag)}"
    concept_dir = processed_dir / concept_dir_name

    if clean_existing and processed_dir.exists():
        shutil.rmtree(processed_dir, ignore_errors=True)
    concept_dir.mkdir(parents=True, exist_ok=True)

    report.output_dir = concept_dir

    # Copia immagini e crea/copia caption .txt
    for img in valid:
        dest_img = concept_dir / img.name
        shutil.copy2(img, dest_img)

        # Caption: cerca prima {img.stem}.txt accanto all'immagine originale
        caption_src = images_dir / f"{img.stem}.txt"
        caption_dest = concept_dir / f"{img.stem}.txt"

        if caption_src.exists():
            shutil.copy2(caption_src, caption_dest)
        else:
            # Caption vuota: sd-scripts la accetta; il tag attivatore è nel nome dir
            caption_dest.write_text("", encoding="utf-8")
            report.missing_captions += 1

    if report.missing_captions > 0:
        report.warnings.append(
            f"{report.missing_captions} immagini senza caption .txt — "
            "usa il Dataset Inspector per aggiungere le descrizioni "
            "(migliora la qualità del LoRA)."
        )

    if report.skipped_excluded > 0:
        report.warnings.append(
            f"{report.skipped_excluded} immagini escluse dal training "
            "(marcate come exclude nel Dataset Inspector)."
        )

    logger.info(
        "Dataset preparato: %d immagini → %s (repeats=%d)",
        report.valid_images, concept_dir.name, report.repeats,
    )
    return report


def _is_readable_image(path: Path) -> bool:
    """Verifica che il file sia un'immagine leggibile (header check via Pillow)."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()   # rileva corruzione senza decodificare tutto
        return True
    except ImportError:
        # Pillow non disponibile: controllo size minimo
        return path.stat().st_size > 1024
    except Exception:
        return False


def _safe_tag(tag: str) -> str:
    """Rende sicuro il tag attivatore come nome di cartella."""
    safe = tag.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return safe[:64] or "concept"
