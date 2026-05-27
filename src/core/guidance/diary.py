"""Decision Diary — registro append-only delle scelte (Pilastro C).

Riferimento: docs/GUIDED_TRAINING.md

Ogni approvazione/rifiuto in una sessione guidata produce una DiaryEntry,
appesa a un file JSONL (una entry per riga). Il formato è una coppia di
preferenza (chosen > rejected) compatibile con DPO futuro.

Il diario è APPEND-ONLY: mai modificare o cancellare entry esistenti. Per
"dimenticare" una sessione si scrive una entry con status diverso, ma i dati
storici restano per analisi retrospettiva.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DiaryEntry:
    """Una decisione registrata: chosen > rejected per uno step."""

    session_id: str
    project_slug: str
    step_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    chosen_path: Optional[Path] = None  # None = tutti rifiutati
    rejected_paths: list[Path] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    technique_refs_used: list[str] = field(default_factory=list)
    lora_checkpoint: str = ""
    was_regeneration: bool = False
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: _utcnow_iso())

    @property
    def is_approval(self) -> bool:
        return self.chosen_path is not None

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "project_slug": self.project_slug,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "chosen_path": str(self.chosen_path) if self.chosen_path else None,
            "rejected_paths": [str(p) for p in self.rejected_paths],
            "rejection_reason": self.rejection_reason,
            "technique_refs_used": list(self.technique_refs_used),
            "lora_checkpoint": self.lora_checkpoint,
            "was_regeneration": self.was_regeneration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiaryEntry:
        return cls(
            session_id=d["session_id"],
            project_slug=d["project_slug"],
            step_id=d["step_id"],
            prompt=d.get("prompt", ""),
            negative_prompt=d.get("negative_prompt", ""),
            seed=int(d.get("seed", -1)),
            chosen_path=Path(d["chosen_path"]) if d.get("chosen_path") else None,
            rejected_paths=[Path(p) for p in d.get("rejected_paths", [])],
            rejection_reason=d.get("rejection_reason"),
            technique_refs_used=list(d.get("technique_refs_used", [])),
            lora_checkpoint=d.get("lora_checkpoint", ""),
            was_regeneration=bool(d.get("was_regeneration", False)),
            entry_id=d.get("entry_id", uuid.uuid4().hex),
            timestamp=d.get("timestamp", _utcnow_iso()),
        )


def append_entry(diary_path: Path, entry: DiaryEntry) -> None:
    """Appende una entry al file JSONL. Crea il file e le cartelle se mancanti."""
    diary_path = Path(diary_path)
    diary_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.to_dict(), ensure_ascii=False)
    with diary_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    logger.debug("DiaryEntry appesa: %s (step=%s)", entry.entry_id, entry.step_id)


def load_diary(diary_path: Path) -> list[DiaryEntry]:
    """Legge tutte le entry dal JSONL. Righe malformate vengono saltate con warning.

    Ritorna lista vuota se il file non esiste."""
    diary_path = Path(diary_path)
    if not diary_path.exists():
        return []
    entries: list[DiaryEntry] = []
    for lineno, raw in enumerate(diary_path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entries.append(DiaryEntry.from_dict(json.loads(raw)))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Riga %d del diario malformata, saltata: %s", lineno, exc)
    return entries


def diary_stats(entries: list[DiaryEntry]) -> dict:
    """Statistiche aggregate sul diario per il pannello 'Cosa ho imparato'."""
    if not entries:
        return {
            "total_entries": 0,
            "total_sessions": 0,
            "approval_rate": 0.0,
            "regeneration_count": 0,
            "most_rejected_step": None,
            "rejection_reasons": Counter(),
            "technique_usage": Counter(),
        }

    sessions = {e.session_id for e in entries}
    approvals = sum(1 for e in entries if e.is_approval)

    rejected_by_step: Counter = Counter()
    reasons: Counter = Counter()
    techniques: Counter = Counter()
    for e in entries:
        if not e.is_approval:
            rejected_by_step[e.step_id] += 1
        if e.rejection_reason:
            reasons[e.rejection_reason.strip()] += 1
        for ref in e.technique_refs_used:
            techniques[ref] += 1

    most_rejected = rejected_by_step.most_common(1)[0][0] if rejected_by_step else None

    return {
        "total_entries": len(entries),
        "total_sessions": len(sessions),
        "approval_rate": round(approvals / len(entries), 3),
        "regeneration_count": sum(1 for e in entries if e.was_regeneration),
        "most_rejected_step": most_rejected,
        "rejection_reasons": reasons,
        "technique_usage": techniques,
    }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
