"""Gestione Pose Library.

Le ~500 reference di pose: import, categorizzazione, ricerca, cache
scheletri. NON è un dataset di training, è un catalogo di input
ControlNet.

Riferimento: docs/POSE_LIBRARY.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.atomic import atomic_write_text
from src.utils.paths import get_user_data_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass
class PoseEntry:
    id: str
    source_path: str
    skeleton_path: str = ""
    skeleton_preview: str = ""
    thumbnail: str = ""
    added_at: str = ""
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    person_count: int = 1
    confidence: float = 0.0
    user_notes: str = ""
    favorite: bool = False


class PoseLibrary:
    """Catalogo pose su filesystem, indicizzato in memoria."""

    def __init__(self) -> None:
        self.root = get_user_data_dir() / "pose_library"
        self._poses: dict[str, PoseEntry] = {}
        self._loaded = False

    # --- Path -----------------------------------------------------

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def skeletons_dir(self) -> Path:
        return self.root / "skeletons"

    @property
    def thumbnails_dir(self) -> Path:
        return self.root / "thumbnails"

    @property
    def index_path(self) -> Path:
        return self.root / "library.json"

    # --- Load / save ----------------------------------------------

    def ensure_dirs(self) -> None:
        for d in (self.sources_dir, self.skeletons_dir, self.thumbnails_dir):
            d.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        self.ensure_dirs()
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._poses = {
                pid: PoseEntry(id=pid, **{k: v for k, v in pdata.items() if k != "id"})
                for pid, pdata in data.get("poses", {}).items()
            }
        self._loaded = True
        logger.info("Pose library caricata: %d pose", len(self._poses))

    def save(self) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "poses": {pid: _entry_dict(e) for pid, e in self._poses.items()},
        }
        atomic_write_text(
            self.index_path, json.dumps(data, indent=2, ensure_ascii=False)
        )

    # --- Query ----------------------------------------------------

    def all(self) -> list[PoseEntry]:
        return list(self._poses.values())

    def get(self, pose_id: str) -> Optional[PoseEntry]:
        return self._poses.get(pose_id)

    def search(
        self,
        text: str = "",
        categories: Optional[list[str]] = None,
        favorites_only: bool = False,
    ) -> list[PoseEntry]:
        """Filtro in memoria su metadata."""
        results = []
        text_low = text.lower()
        for e in self._poses.values():
            if favorites_only and not e.favorite:
                continue
            if categories and not all(c in e.categories for c in categories):
                continue
            if text_low:
                haystack = " ".join([*e.tags, *e.categories, e.user_notes]).lower()
                if text_low not in haystack:
                    continue
            results.append(e)
        return results

    def all_categories(self) -> list[str]:
        cats: set[str] = set()
        for e in self._poses.values():
            cats.update(e.categories)
        return sorted(cats)

    # --- Mutations ------------------------------------------------

    def add(self, entry: PoseEntry) -> None:
        self._poses[entry.id] = entry

    def update(self, entry: PoseEntry) -> None:
        self._poses[entry.id] = entry

    def remove(self, pose_id: str) -> None:
        self._poses.pop(pose_id, None)

    def next_id(self) -> str:
        n = len(self._poses) + 1
        while f"pose_{n:04d}" in self._poses:
            n += 1
        return f"pose_{n:04d}"


def categorize_skeleton(joints: dict) -> list[str]:
    """Euristiche di categorizzazione automatica dallo scheletro.

    Stub: la versione completa analizza posizioni joint relative.
    Restituisce categorie come ["standing", "frontal", "arms_raised"].
    """
    categories: list[str] = []
    # Placeholder — la logica vera userà i joint DWPose:
    # - confronto y di polsi vs spalle → arms_raised
    # - presenza joint frontali/posteriori → orientamento
    # - distribuzione verticale → standing/sitting/lying
    return categories


def _entry_dict(e: PoseEntry) -> dict:
    d = asdict(e)
    d.pop("id", None)
    return d


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
