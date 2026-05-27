"""Modello della sessione di training guidato (Pilastro A).

Riferimento: docs/GUIDED_TRAINING.md

Una GuidedSession descrive una pipeline di step (composizione → struttura →
colori → ombre). Ogni step genera N candidati; l'utente ne approva uno (o li
rifiuta tutti). L'immagine approvata diventa il punto di partenza dello step
successivo via img2img.

Nessuna dipendenza Qt né ComfyUI: solo dati + (de)serializzazione JSON.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Stati possibili di una sessione.
STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_EXHAUSTED = "exhausted"
STATUS_ABORTED = "aborted"


@dataclass(frozen=True)
class StepDefinition:
    """Definizione di una fase della pipeline. Immutabile."""

    id: str
    label: str
    denoise_strength: float  # 1.0 per il primo step, <1.0 per img2img successivi
    n_candidates: int = 4
    guidance_scale: float = 7.0
    steps: int = 20
    technique_slots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "denoise_strength": self.denoise_strength,
            "n_candidates": self.n_candidates,
            "guidance_scale": self.guidance_scale,
            "steps": self.steps,
            "technique_slots": list(self.technique_slots),
        }

    @classmethod
    def from_dict(cls, d: dict) -> StepDefinition:
        return cls(
            id=d["id"],
            label=d["label"],
            denoise_strength=float(d["denoise_strength"]),
            n_candidates=int(d.get("n_candidates", 4)),
            guidance_scale=float(d.get("guidance_scale", 7.0)),
            steps=int(d.get("steps", 20)),
            technique_slots=list(d.get("technique_slots", [])),
        )


@dataclass
class Candidate:
    """Un candidato generato per uno step. Mutabile (is_approved cambia)."""

    index: int
    image_path: Path
    sidecar_path: Optional[Path] = None
    latent_path: Optional[Path] = None  # v2: latent puro; v1 usa img2img su PNG
    is_approved: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "image_path": str(self.image_path),
            "sidecar_path": str(self.sidecar_path) if self.sidecar_path else None,
            "latent_path": str(self.latent_path) if self.latent_path else None,
            "is_approved": self.is_approved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Candidate:
        return cls(
            index=int(d["index"]),
            image_path=Path(d["image_path"]),
            sidecar_path=Path(d["sidecar_path"]) if d.get("sidecar_path") else None,
            latent_path=Path(d["latent_path"]) if d.get("latent_path") else None,
            is_approved=d.get("is_approved"),
        )


@dataclass
class StepResult:
    """Esito di uno step: i candidati generati e quale (se) approvato."""

    step_def: StepDefinition
    candidates: list[Candidate] = field(default_factory=list)
    approved_index: Optional[int] = None  # None = tutti rifiutati
    rejection_reason: Optional[str] = None

    @property
    def approved_candidate(self) -> Optional[Candidate]:
        if self.approved_index is None:
            return None
        for c in self.candidates:
            if c.index == self.approved_index:
                return c
        return None

    @property
    def all_rejected(self) -> bool:
        return bool(self.candidates) and self.approved_index is None

    def to_dict(self) -> dict:
        return {
            "step_def": self.step_def.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "approved_index": self.approved_index,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StepResult:
        return cls(
            step_def=StepDefinition.from_dict(d["step_def"]),
            candidates=[Candidate.from_dict(c) for c in d.get("candidates", [])],
            approved_index=d.get("approved_index"),
            rejection_reason=d.get("rejection_reason"),
        )


@dataclass
class GuidedSession:
    """Una sessione completa di training guidato."""

    project_slug: str
    prompt: str
    negative_prompt: str
    seed: int
    lora_checkpoint: str
    pipeline: list[StepDefinition]
    results: list[StepResult] = field(default_factory=list)
    status: str = STATUS_ACTIVE
    final_image_path: Optional[Path] = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: _utcnow_iso())
    schema_version: int = SCHEMA_VERSION

    # --- Stato derivato -------------------------------------------------

    @property
    def current_step_index(self) -> int:
        """Indice del prossimo step da eseguire (= numero di step completati)."""
        return len(self.results)

    @property
    def is_finished(self) -> bool:
        return self.status in (STATUS_COMPLETED, STATUS_EXHAUSTED, STATUS_ABORTED)

    @property
    def current_step_def(self) -> Optional[StepDefinition]:
        """Definizione del prossimo step, o None se la pipeline è esaurita."""
        idx = self.current_step_index
        if idx >= len(self.pipeline):
            return None
        return self.pipeline[idx]

    def step_input_image(self) -> Optional[Path]:
        """Immagine approvata dell'ultimo step = input img2img del prossimo.

        None se nessuno step è ancora stato completato (primo step parte da
        noise puro) oppure se l'ultimo step non ha un approvato."""
        if not self.results:
            return None
        last = self.results[-1]
        approved = last.approved_candidate
        return approved.image_path if approved else None

    # --- Mutazioni ------------------------------------------------------

    def record_result(self, result: StepResult) -> None:
        """Aggiunge l'esito di uno step. Se è l'ultimo e ha un approvato e non
        ci sono altri step, marca la sessione come completata."""
        self.results.append(result)
        if (
            result.approved_candidate is not None
            and self.current_step_index >= len(self.pipeline)
        ):
            self.status = STATUS_COMPLETED
            self.final_image_path = result.approved_candidate.image_path

    # --- Serializzazione ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "project_slug": self.project_slug,
            "created_at": self.created_at,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "lora_checkpoint": self.lora_checkpoint,
            "pipeline": [s.to_dict() for s in self.pipeline],
            "results": [r.to_dict() for r in self.results],
            "status": self.status,
            "final_image_path": str(self.final_image_path) if self.final_image_path else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GuidedSession:
        sv = d.get("schema_version", 1)
        if sv > SCHEMA_VERSION:
            raise ValueError(
                f"GuidedSession schema_version={sv} non supportato (max {SCHEMA_VERSION})."
            )
        return cls(
            project_slug=d["project_slug"],
            prompt=d.get("prompt", ""),
            negative_prompt=d.get("negative_prompt", ""),
            seed=int(d.get("seed", -1)),
            lora_checkpoint=d.get("lora_checkpoint", ""),
            pipeline=[StepDefinition.from_dict(s) for s in d.get("pipeline", [])],
            results=[StepResult.from_dict(r) for r in d.get("results", [])],
            status=d.get("status", STATUS_ACTIVE),
            final_image_path=Path(d["final_image_path"]) if d.get("final_image_path") else None,
            session_id=d.get("session_id", uuid.uuid4().hex),
            created_at=d.get("created_at", _utcnow_iso()),
            schema_version=sv,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("GuidedSession salvata: %s", path)

    @classmethod
    def load(cls, path: Path) -> GuidedSession:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# --- Pipeline predefinita SDXL ----------------------------------------


def default_pipeline() -> list[StepDefinition]:
    """Pipeline standard a 4 step. Funzione (non costante) per evitare
    condivisione accidentale delle liste mutabili technique_slots."""
    return [
        StepDefinition(
            id="composition",
            label="Composizione",
            denoise_strength=1.0,
            n_candidates=4,
            guidance_scale=7.0,
            steps=20,
            technique_slots=[],
        ),
        StepDefinition(
            id="structure",
            label="Struttura & Linework",
            denoise_strength=0.65,
            n_candidates=4,
            guidance_scale=6.0,
            steps=15,
            technique_slots=["linework"],
        ),
        StepDefinition(
            id="colors",
            label="Colori di base",
            denoise_strength=0.55,
            n_candidates=4,
            guidance_scale=5.5,
            steps=12,
            technique_slots=["skin", "hair"],
        ),
        StepDefinition(
            id="shading",
            label="Ombre e volume",
            denoise_strength=0.45,
            n_candidates=4,
            guidance_scale=5.0,
            steps=10,
            technique_slots=["shadow", "light"],
        ),
    ]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
