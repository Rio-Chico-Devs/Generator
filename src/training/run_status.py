"""Gestione stato di un run di training (status.json per run).

status.json vive in training/runs/{run_id}/status.json e viene aggiornato
atomicamente durante il training. Permette di riprendere da crash e di
mostrare informazioni all'utente al riavvio del progetto.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.atomic import atomic_write_text

# Stati possibili di un run
STATUS_PREPARING = "preparing"
STATUS_RUNNING = "running"
STATUS_ABORTED = "aborted"
STATUS_DONE = "done"
STATUS_ERROR = "error"


@dataclass
class LossPoint:
    step: int
    loss: float

    def to_dict(self) -> dict:
        return {"step": self.step, "loss": self.loss}

    @classmethod
    def from_dict(cls, d: dict) -> LossPoint:
        return cls(step=int(d["step"]), loss=float(d["loss"]))


@dataclass
class RunStatus:
    """Stato persistente di un run di training."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    preset_id: str = ""
    status: str = STATUS_PREPARING
    started_at: str = field(default_factory=lambda: _utcnow_iso())
    finished_at: Optional[str] = None

    # Progress
    current_step: int = 0
    max_steps: int = 0
    current_epoch: int = 0
    max_epochs: int = 0

    # Paths
    config_path: Optional[str] = None
    last_checkpoint_path: Optional[str] = None
    final_checkpoint_path: Optional[str] = None

    # Loss history (sampled, non tutti gli step)
    loss_history: list[LossPoint] = field(default_factory=list)

    # Error info
    error_message: Optional[str] = None

    @property
    def is_resumable(self) -> bool:
        """True se il run è interrotto e ha un checkpoint da cui riprendere."""
        return self.status == STATUS_ABORTED and self.last_checkpoint_path is not None

    @property
    def progress_pct(self) -> float:
        if self.max_steps <= 0:
            return 0.0
        return round(100.0 * self.current_step / self.max_steps, 1)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "preset_id": self.preset_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "current_epoch": self.current_epoch,
            "max_epochs": self.max_epochs,
            "config_path": self.config_path,
            "last_checkpoint_path": self.last_checkpoint_path,
            "final_checkpoint_path": self.final_checkpoint_path,
            "loss_history": [lp.to_dict() for lp in self.loss_history],
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunStatus:
        rs = cls(
            run_id=d.get("run_id", uuid.uuid4().hex[:12]),
            preset_id=d.get("preset_id", ""),
            status=d.get("status", STATUS_PREPARING),
            started_at=d.get("started_at", _utcnow_iso()),
            finished_at=d.get("finished_at"),
            current_step=int(d.get("current_step", 0)),
            max_steps=int(d.get("max_steps", 0)),
            current_epoch=int(d.get("current_epoch", 0)),
            max_epochs=int(d.get("max_epochs", 0)),
            config_path=d.get("config_path"),
            last_checkpoint_path=d.get("last_checkpoint_path"),
            final_checkpoint_path=d.get("final_checkpoint_path"),
            loss_history=[LossPoint.from_dict(lp) for lp in d.get("loss_history", [])],
            error_message=d.get("error_message"),
        )
        return rs

    def save(self, run_dir: Path) -> None:
        path = Path(run_dir) / "status.json"
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, run_dir: Path) -> RunStatus:
        path = Path(run_dir) / "status.json"
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def load_all(cls, training_runs_dir: Path) -> list[RunStatus]:
        """Carica tutti i run dalla cartella runs/ del progetto, dal più recente."""
        runs: list[RunStatus] = []
        if not training_runs_dir.exists():
            return runs
        for run_dir in sorted(training_runs_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and (run_dir / "status.json").exists():
                try:
                    runs.append(cls.load(run_dir))
                except Exception:
                    pass
        return runs


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
