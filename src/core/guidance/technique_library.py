"""Libreria Tecniche — riferimenti visivi per IP-Adapter (Pilastro B).

Riferimento: docs/GUIDED_TRAINING.md

Raccolta di immagini classificate per slot (skin, shadow, hair, linework…).
Durante uno step della sessione guidata, i riferimenti degli slot richiesti
dallo step vengono passati a IP-Adapter come condizionamento visivo.

La libreria è opzionale: se vuota, la generazione guidata funziona comunque,
solo senza condizionamento IP-Adapter.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.core.guidance.session import StepDefinition
from src.utils.atomic import atomic_write_text

logger = logging.getLogger(__name__)

# Slot predefiniti con label leggibili. Si possono aggiungere slot custom.
DEFAULT_SLOTS: dict[str, str] = {
    "skin": "Colorazione pelle",
    "shadow": "Ombre e profondità",
    "light": "Sorgenti luce",
    "hair": "Capelli & texture",
    "linework": "Linee & contorni",
    "background": "Sfondi",
    "palette": "Palette cromatica",
}


@dataclass
class TechniqueRef:
    """Un riferimento visivo per uno slot tecnico."""

    slot: str
    image_path: Path
    label: str = ""
    notes: str = ""
    weight: float = 1.0  # peso IP-Adapter (0.0-2.0)
    ref_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: _utcnow_iso())

    def to_dict(self) -> dict:
        return {
            "ref_id": self.ref_id,
            "slot": self.slot,
            "image_path": str(self.image_path),
            "label": self.label,
            "notes": self.notes,
            "weight": self.weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TechniqueRef:
        return cls(
            slot=d["slot"],
            image_path=Path(d["image_path"]),
            label=d.get("label", ""),
            notes=d.get("notes", ""),
            weight=float(d.get("weight", 1.0)),
            ref_id=d.get("ref_id", uuid.uuid4().hex),
            created_at=d.get("created_at", _utcnow_iso()),
        )


@dataclass
class TechniqueLibrary:
    """Insieme dei riferimenti tecnici di un progetto."""

    project_slug: str
    refs: list[TechniqueRef] = field(default_factory=list)

    def by_slot(self, slot: str) -> list[TechniqueRef]:
        return [r for r in self.refs if r.slot == slot]

    def slots_in_use(self) -> set[str]:
        return {r.slot for r in self.refs}

    def active_for_step(self, step_def: StepDefinition) -> list[TechniqueRef]:
        """Riferimenti che alimentano IP-Adapter per uno step dato.

        Sono i ref i cui slot compaiono in step_def.technique_slots."""
        wanted = set(step_def.technique_slots)
        if not wanted:
            return []
        return [r for r in self.refs if r.slot in wanted]

    def add(self, ref: TechniqueRef) -> None:
        self.refs.append(ref)

    def remove(self, ref_id: str) -> bool:
        """Rimuove il ref dato per ref_id. True se trovato e rimosso."""
        before = len(self.refs)
        self.refs = [r for r in self.refs if r.ref_id != ref_id]
        return len(self.refs) < before

    def get(self, ref_id: str) -> TechniqueRef | None:
        for r in self.refs:
            if r.ref_id == ref_id:
                return r
        return None

    # --- Serializzazione ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "project_slug": self.project_slug,
            "refs": [r.to_dict() for r in self.refs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TechniqueLibrary:
        return cls(
            project_slug=d.get("project_slug", ""),
            refs=[TechniqueRef.from_dict(r) for r in d.get("refs", [])],
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        atomic_write_text(
            path,
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
        )
        logger.debug("TechniqueLibrary salvata: %s (%d ref)", path, len(self.refs))

    @classmethod
    def load(cls, path: Path) -> TechniqueLibrary:
        """Carica la libreria; ritorna libreria vuota se il file non esiste."""
        path = Path(path)
        if not path.exists():
            return cls(project_slug="")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
