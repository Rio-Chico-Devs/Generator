"""Modello dati Progetto e I/O su filesystem.

Riferimento: docs/PROJECTS.md
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.atomic import atomic_write_text
from src.utils.paths import get_projects_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass
class BaseModelRef:
    id: str
    name: str
    hash: str = ""
    locked: bool = True


@dataclass
class ActiveLora:
    run_id: str
    checkpoint: str
    weight: float = 0.85


@dataclass
class GenerationDefaults:
    steps: int = 30
    cfg_scale: float = 7.0
    sampler: str = "dpmpp_2m_karras"
    width: int = 1024
    height: int = 1024


@dataclass
class ProjectStats:
    dataset_image_count: int = 0
    training_runs: int = 0
    generation_count: int = 0
    gallery_count: int = 0


@dataclass
class Project:
    schema_version: int = SCHEMA_VERSION
    name: str = ""
    slug: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    base_model: Optional[BaseModelRef] = None
    activator_tag: str = ""
    trigger_prompt_prefix: str = ""
    # Slug del progetto genitore se questo è una variante (fork). Vuoto = radice.
    parent_slug: str = ""
    default_negative_prompt: str = (
        "low quality, worst quality, blurry, deformed, bad anatomy"
    )
    default_generation_params: GenerationDefaults = field(default_factory=GenerationDefaults)
    active_lora: Optional[ActiveLora] = None
    stats: ProjectStats = field(default_factory=ProjectStats)

    # Non serializzato: path radice del progetto su disco
    _root: Optional[Path] = field(default=None, repr=False)

    # --- Path helpers ------------------------------------------------

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("Project root not set. Save or load the project first.")
        return self._root

    @property
    def project_json_path(self) -> Path:
        return self.root / "project.json"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def dataset_images_dir(self) -> Path:
        return self.dataset_dir / "images"

    @property
    def dataset_processed_dir(self) -> Path:
        return self.dataset_dir / "processed"

    @property
    def dataset_thumbnails_dir(self) -> Path:
        return self.dataset_dir / "thumbnails"

    @property
    def dataset_manifest_path(self) -> Path:
        return self.dataset_dir / "manifest.json"

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def training_runs_dir(self) -> Path:
        return self.training_dir / "runs"

    @property
    def guided_dir(self) -> Path:
        """Sessioni di training guidato (human-in-the-loop)."""
        return self.training_dir / "guided"

    @property
    def diary_path(self) -> Path:
        """Decision Diary append-only (JSONL) condiviso tra le sessioni guidate."""
        return self.guided_dir / "diary.jsonl"

    @property
    def techniques_dir(self) -> Path:
        return self.root / "techniques"

    @property
    def techniques_refs_dir(self) -> Path:
        return self.techniques_dir / "refs"

    @property
    def technique_library_path(self) -> Path:
        return self.techniques_dir / "library.json"

    def guided_session_dir(self, session_id: str) -> Path:
        return self.guided_dir / session_id

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations"

    @property
    def gallery_dir(self) -> Path:
        return self.root / "gallery"

    @property
    def negatives_dir(self) -> Path:
        """Immagini 👎 scartate: riferimento negativo per il training futuro."""
        return self.root / "negatives"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    # --- Serializzazione ----------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_root", None)
        return d

    @classmethod
    def from_dict(cls, data: dict, root: Path) -> Project:
        # Migrazioni schema in caso di versione vecchia (placeholder)
        sv = data.get("schema_version", 1)
        if sv > SCHEMA_VERSION:
            raise ValueError(
                f"Progetto schema_version={sv} non supportato (max {SCHEMA_VERSION})."
            )

        bm = data.get("base_model")
        al = data.get("active_lora")
        params = data.get("default_generation_params", {})
        stats = data.get("stats", {})

        p = cls(
            schema_version=sv,
            name=data.get("name", ""),
            slug=data.get("slug", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            base_model=BaseModelRef(**bm) if bm else None,
            activator_tag=data.get("activator_tag", ""),
            trigger_prompt_prefix=data.get("trigger_prompt_prefix", ""),
            parent_slug=data.get("parent_slug", ""),
            default_negative_prompt=data.get("default_negative_prompt", ""),
            default_generation_params=GenerationDefaults(**params),
            active_lora=ActiveLora(**al) if al else None,
            stats=ProjectStats(**stats),
        )
        p._root = root
        return p

    def save(self) -> None:
        self.updated_at = _utcnow_iso()
        atomic_write_text(
            self.project_json_path,
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
        )
        logger.debug("Project saved: %s", self.project_json_path)

    # --- Operazioni di alto livello -----------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        base_model: Optional[BaseModelRef] = None,
        parent_dir: Optional[Path] = None,
    ) -> Project:
        slug = _slugify(name)
        parent = parent_dir or get_projects_dir()
        parent.mkdir(parents=True, exist_ok=True)

        root = _resolve_unique_dir(parent / slug)
        root.mkdir(parents=True)

        # Tag attivatore unico, derivato dallo slug + suffisso versione
        activator = f"vf_{slug.replace('-', '_')}_v1"

        now = _utcnow_iso()
        project = cls(
            name=name,
            slug=root.name,  # slug effettivo (post-deconflitto)
            description=description,
            created_at=now,
            updated_at=now,
            base_model=base_model,
            activator_tag=activator,
            trigger_prompt_prefix=f"{activator}, masterpiece, best quality,",
        )
        project._root = root

        # Crea struttura
        for d in (
            project.dataset_images_dir,
            project.dataset_processed_dir,
            project.dataset_thumbnails_dir,
            project.training_runs_dir,
            project.guided_dir,
            project.generations_dir,
            project.gallery_dir,
            project.negatives_dir,
            project.prompts_dir,
            project.techniques_refs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Manifest vuoto
        project.dataset_manifest_path.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "images": {}}, indent=2),
            encoding="utf-8",
        )

        # README auto-generato
        (project.root / "README.md").write_text(
            f"# {name}\n\n{description}\n\nTag attivatore: `{activator}`\n",
            encoding="utf-8",
        )

        project.save()
        logger.info("Project created: %s (slug=%s)", name, project.slug)
        return project

    def fork(self, name: str, description: str = "") -> Project:
        """Crea un progetto variante che eredita da questo lo stile e i parametri.

        Il figlio nasce vuoto (dataset da popolare con le immagini 🔀 scelte) ma
        eredita: modello base, prompt negativo di default, parametri di
        generazione e il LoRA attivo come punto di partenza. Registra la
        discendenza in ``parent_slug``. Vive nella stessa cartella progetti del
        genitore."""
        child = Project.create(
            name=name,
            description=description or f"Variante di {self.name}",
            base_model=self.base_model,
            parent_dir=self.root.parent,
        )
        child.parent_slug = self.slug
        child.default_negative_prompt = self.default_negative_prompt
        child.default_generation_params = replace(self.default_generation_params)
        child.active_lora = replace(self.active_lora) if self.active_lora else None
        child.save()
        logger.info("Fork progetto: %s → %s", self.slug, child.slug)
        return child

    @classmethod
    def load(cls, root: Path) -> Project:
        root = Path(root).resolve()
        pj = root / "project.json"
        if not pj.exists():
            raise FileNotFoundError(f"project.json non trovato in {root}")
        data = json.loads(pj.read_text(encoding="utf-8"))
        return cls.from_dict(data, root)

    def delete(self) -> None:
        """Sposta nel trash di sistema. Non unlink diretto."""
        try:
            from send2trash import send2trash  # type: ignore

            send2trash(str(self.root))
        except ImportError:
            # Fallback: rinomina con prefisso _trashed_ invece di cancellare
            trashed = self.root.parent / f"_trashed_{self.root.name}_{_utcnow_iso()}"
            self.root.rename(trashed)
            logger.warning("send2trash non disponibile, rinominato a %s", trashed)

    def archive(self) -> Path:
        """Sposta in archive dir."""
        from src.utils.paths import get_archived_dir

        archived = get_archived_dir()
        archived.mkdir(parents=True, exist_ok=True)
        dest = _resolve_unique_dir(archived / self.slug)
        shutil.move(str(self.root), str(dest))
        self._root = dest
        return dest


# --- Helpers ----------------------------------------------------------


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "untitled"


def _resolve_unique_dir(candidate: Path) -> Path:
    """Se candidate esiste, aggiunge suffisso numerico."""
    if not candidate.exists():
        return candidate
    parent = candidate.parent
    base = candidate.name
    i = 2
    while True:
        new = parent / f"{base}-{i}"
        if not new.exists():
            return new
        i += 1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
