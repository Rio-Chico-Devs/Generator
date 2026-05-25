"""Worker QThread per esecuzione ricette via ComfyUI.

Data una RecipeDef + parametri utente + Project, parametrizza il
WorkflowTemplate della ricetta e lo esegue via ComfyClient, emettendo
signal Qt thread-safe verso la UI.

Caratteristiche chiave:
- **Freno termico**: prima di ogni immagine attende che la GPU sia abbastanza
  fredda (ThermalGovernor), con cooldown fisso tra un'immagine e l'altra.
  La priorità è non surriscaldare la GPU (caso laptop), anche a costo di
  generazioni più lente.
- **Loop a batch**: N immagini = N submit separati (uno per immagine), così
  il freno termico può intervenire tra l'una e l'altra e ogni immagine ha un
  seed proprio. Su 8GB questo è anche più sicuro di un batch unico.
- **Riproducibilità**: il seed effettivo di ogni immagine viene salvato in un
  sidecar JSON accanto al file, insieme a prompt/negative/parametri.

Tutto il lavoro pesante (I/O, WebSocket) avviene nel thread worker; la UI
riceve solo signal. MAI mutare widget Qt da qui.

Riferimento: docs/COMFY_ENGINE.md, docs/RECIPES.md
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.comfy.client import ComfyClientProtocol, ComfyInterrupted
from src.comfy.workflow import WorkflowTemplate
from src.core.catalog import dialect_for_model
from src.core.project import Project
from src.core.prompting import build_prompts, get_preset
from src.core.recipes import RecipeDef
from src.utils.gpu_monitor import CooldownOutcome, SafetyConfig, ThermalGovernor
from src.utils.paths import get_assets_dir
from src.utils.seed import resolve_seed

logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"
MAX_BATCH = 16


@dataclass
class GenerationRecord:
    """Parametri effettivi di una generazione, serializzati nel sidecar JSON.

    Permette di riprodurre o ispezionare un'immagine (incluso il seed reale,
    che con seed=-1 sarebbe altrimenti perso)."""

    model_id: str
    dialect: str
    positive: str
    negative: str
    width: int
    height: int
    clip_skip: int
    recipe_id: str = ""
    lora_name: Optional[str] = None
    lora_weight: Optional[float] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler: Optional[str] = None
    seed: int = -1
    app_version: str = APP_VERSION
    created_at: str = ""


class RecipeWorker(QThread):
    """Esegue una ricetta ComfyUI in un thread separato.

    Uso tipico:
        worker = RecipeWorker(recipe, params, project, client, out_dir)
        worker.progress.connect(progress_bar.setValue)
        worker.image_ready.connect(gallery.add_image)
        worker.cooling.connect(status.show_cooldown)
        worker.finished_ok.connect(on_done)
        worker.start()
    """

    progress    = pyqtSignal(int, int)  # step_corrente, totale_step
    image_ready = pyqtSignal(Path)      # path immagine appena prodotta
    error       = pyqtSignal(str)       # messaggio errore (fatale) o interruzione
    finished_ok = pyqtSignal(list)      # list[Path] — tutte le immagini
    cooling     = pyqtSignal(int, int)  # temp_corrente, temp_ripristino (freno termico)

    def __init__(
        self,
        recipe: RecipeDef,
        user_params: dict[str, Any],
        project: Project,
        client: ComfyClientProtocol,
        output_dir: Path,
        comfy_input_dir: Optional[Path] = None,
        safety: Optional[SafetyConfig] = None,
        gpu_read_fn: Optional[Callable[[], Any]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._recipe = recipe
        self._user_params = user_params
        self._project = project
        self._client = client
        self._output_dir = output_dir
        self._comfy_input_dir = comfy_input_dir
        self._safety = safety or SafetyConfig()
        if gpu_read_fn is not None:
            self._governor = ThermalGovernor(self._safety, read_fn=gpu_read_fn)
        else:
            self._governor = ThermalGovernor(self._safety)
        self._aborted = False

    def abort(self) -> None:
        """Richiede interruzione al client ComfyUI e segnala al thread."""
        self._aborted = True
        try:
            self._client.interrupt()
        except Exception as exc:
            logger.warning("interrupt() fallito: %s", exc)

    def run(self) -> None:
        try:
            self._run_impl()
        except ComfyInterrupted:
            self.error.emit("Generazione interrotta dall'utente.")
        except Exception as exc:
            logger.exception("Errore non gestito in RecipeWorker")
            self.error.emit(str(exc))

    def _run_impl(self) -> None:
        wf_path = get_assets_dir() / "workflows" / self._recipe.workflow_file
        wf = WorkflowTemplate(wf_path)

        record = _parametrize(
            wf,
            self._recipe,
            self._user_params,
            self._project,
            self._comfy_input_dir,
        )

        if self._aborted:
            self.error.emit("Generazione interrotta dall'utente.")
            return

        count = _resolve_count(self._user_params)
        base_seed = record.seed
        user_gave_seed = _int_or(self._user_params.get("seed"), -1) >= 0

        self._output_dir.mkdir(parents=True, exist_ok=True)
        final_paths: list[Path] = []

        for i in range(count):
            if self._aborted:
                break

            if not self._wait_until_cool():
                # ABORTED o TIMEOUT: messaggio già emesso (o abort silenzioso)
                if not self._aborted:
                    return  # timeout termico: stop senza finished_ok
                break

            # Seed proprio per ogni immagine: se l'utente lo ha fissato, le
            # varianti sono base_seed, base_seed+1, ...; altrimenti casuale.
            seed_i = (base_seed + i) if user_gave_seed else resolve_seed(-1)
            wf.set_seed(seed_i)

            prompt_id = self._client.submit(wf.build())
            logger.info(
                "Ricetta '%s' img %d/%d sottomessa — prompt_id=%s seed=%d",
                self._recipe.id, i + 1, count, prompt_id, seed_i,
            )
            outputs = self._client.wait_for_completion(
                prompt_id,
                progress_callback=lambda s, t: self.progress.emit(s, t),
            )

            if self._aborted:
                break

            for src in outputs:
                dest = _unique_dest(self._output_dir, src.name)
                if src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                _write_sidecar(dest, record, seed_i)
                final_paths.append(dest)
                self.image_ready.emit(dest)

            # Cooldown fisso tra un'immagine e la successiva (non dopo l'ultima).
            if i < count - 1 and self._safety.enabled:
                self._interruptible_sleep(self._safety.cooldown_between_images_sec)

        if self._aborted and not final_paths:
            self.error.emit("Generazione interrotta dall'utente.")
            return

        self.finished_ok.emit(final_paths)
        logger.info(
            "Ricetta '%s' completata: %d immagini", self._recipe.id, len(final_paths)
        )

    # --- Freno termico ----------------------------------------------------

    def _wait_until_cool(self) -> bool:
        """Attende il raffreddamento GPU. Ritorna False se non si può procedere.

        Emette `cooling` durante l'attesa. In caso di TIMEOUT termico emette
        `error` e ritorna False (la generazione si ferma per sicurezza)."""
        if not self._safety.enabled:
            return True
        outcome = self._governor.wait_until_safe(
            on_wait=lambda s: self.cooling.emit(
                s.temperature_c, self._governor.cfg.resume_temp_c
            ),
            should_abort=lambda: self._aborted,
        )
        if outcome == CooldownOutcome.TIMEOUT:
            self.error.emit(
                f"GPU troppo calda: non scende sotto "
                f"{self._governor.cfg.resume_temp_c}°C entro il tempo massimo. "
                "Generazione interrotta per sicurezza — controlla la ventilazione."
            )
            return False
        return outcome != CooldownOutcome.ABORTED

    def _interruptible_sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while not self._aborted:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.2, remaining))


# ---------------------------------------------------------------------------
# Parametrizzazione — funzione pura, testabile senza Qt
# ---------------------------------------------------------------------------

# Chiavi con semantica speciale: gestite esplicitamente sopra, non passate
# a set_role nel loop residuo.
_HANDLED_KEYS = frozenset(
    {"prompt", "negative", "seed", "width", "height", "lora_weight",
     "variants", "batch"}
)


def _parametrize(
    wf: WorkflowTemplate,
    recipe: RecipeDef,
    user_params: dict[str, Any],
    project: Project,
    comfy_input_dir: Optional[Path] = None,
) -> GenerationRecord:
    """Applica user_params + contesto progetto al WorkflowTemplate.

    Ordine deliberato: checkpoint → prompt+CLIP skip → seed → dimensioni →
    LoRA → immagini input → knob numerici residui.

    Funzione pura rispetto a Qt: nessun signal, nessun thread. Ritorna un
    :class:`GenerationRecord` con i parametri effettivamente applicati (per il
    sidecar di riproducibilità)."""
    # 1. Checkpoint dal progetto (base_model → nome file safetensors)
    model_id = project.base_model.id if project.base_model is not None else ""
    if model_id:
        _safe_set_role(wf, "checkpoint", f"{model_id}.safetensors")

    # 2. Prompt: applica il preset del dialetto del modello (quality-tag +
    #    CLIP skip) combinato con trigger del progetto e input utente.
    dialect = dialect_for_model(model_id)
    preset = get_preset(dialect)
    built = build_prompts(
        preset,
        user_positive=str(user_params.get("prompt", "")),
        trigger_prefix=project.trigger_prompt_prefix or "",
        user_negative=str(user_params.get("negative", "")),
        project_negative=project.default_negative_prompt or "",
    )
    _safe_set_prompt(wf, built.positive, built.negative)
    # CLIP skip: ComfyUI (CLIPSetLastLayer) usa il valore negativo.
    _safe_set_role(wf, "clip_skip", -abs(built.clip_skip))

    # 3. Seed: -1 viene risolto in un valore casuale riproducibile
    raw_seed = _int_or(user_params.get("seed"), -1)
    actual_seed = resolve_seed(raw_seed)
    _safe_set_seed(wf, actual_seed)

    # 4. Dimensioni: user_params > default progetto
    width = _int_or(
        user_params.get("width"), project.default_generation_params.width
    )
    height = _int_or(
        user_params.get("height"), project.default_generation_params.height
    )
    _safe_set_dimensions(wf, width, height)

    # 5. LoRA del personaggio — il cuore: senza questo il personaggio non esiste
    lora_name: Optional[str] = None
    lora_weight: Optional[float] = float(user_params.get("lora_weight", 0.85))
    if project.active_lora is not None:
        lora_path = _resolve_lora_path(project)
        if lora_path is not None:
            wf.set_lora(str(lora_path), lora_weight)
            lora_name = lora_path.name
        else:
            logger.warning(
                "LoRA '%s' non trovato su disco — generazione senza stile personaggio",
                project.active_lora.checkpoint,
            )
            lora_weight = None
    else:
        logger.warning(
            "Progetto '%s' senza active_lora — generazione senza stile personaggio",
            project.name,
        )
        lora_weight = None

    # 6. Immagini input (pose, personaggio, prodotto…)
    #    Copia nella dir input di ComfyUI e aggiorna il nodo LoadImage
    for inp in recipe.inputs:
        if inp.kind not in ("image", "pose", "character"):
            continue
        raw = user_params.get(inp.key)
        if raw is None:
            continue
        src_path = Path(str(raw))
        if not src_path.exists():
            logger.warning("Immagine input '%s' non trovata: %s", inp.key, src_path)
            continue
        if comfy_input_dir is None:
            logger.warning(
                "comfy_input_dir non fornita — immagine '%s' non caricata", inp.key
            )
            continue
        dest_name = _copy_to_comfy_input(src_path, comfy_input_dir)
        node_spec = wf.mapping.get(inp.key)
        if node_spec and "node" in node_spec:
            wf.set_input_image(node_spec["node"], dest_name)

    # 7. Knob numerici residui (steps, cfg, denoise…)
    #    Passati via set_role se presenti nel mapping; chiavi sconosciute ignorate.
    for key, value in user_params.items():
        if key in _HANDLED_KEYS:
            continue
        if key in wf.mapping:
            try:
                wf.set_role(key, value)
            except KeyError:
                pass  # mapping presente ma nodo assente nel grafo — skip silenzioso

    return GenerationRecord(
        model_id=model_id,
        dialect=dialect,
        positive=built.positive,
        negative=built.negative,
        width=width,
        height=height,
        clip_skip=built.clip_skip,
        recipe_id=getattr(recipe.id, "value", str(recipe.id)),
        lora_name=lora_name,
        lora_weight=lora_weight,
        steps=_role_or_none(wf, "steps"),
        cfg=_role_or_none(wf, "cfg"),
        sampler=_role_or_none(wf, "sampler"),
        seed=actual_seed,
    )


# ---------------------------------------------------------------------------
# Helper interni
# ---------------------------------------------------------------------------


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_count(user_params: dict[str, Any]) -> int:
    """Quante immagini generare (chiave 'variants' o 'batch'), clampato."""
    raw = user_params.get("variants", user_params.get("batch", 1))
    return max(1, min(_int_or(raw, 1), MAX_BATCH))


def _unique_dest(out_dir: Path, name: str) -> Path:
    """Path di destinazione che non sovrascrive file esistenti."""
    dest = out_dir / name
    if not dest.exists():
        return dest
    p = Path(name)
    return out_dir / f"{p.stem}_{uuid.uuid4().hex[:8]}{p.suffix}"


def _write_sidecar(image_path: Path, record: GenerationRecord, seed: int) -> None:
    """Scrive il sidecar JSON dei parametri accanto all'immagine."""
    data = asdict(record)
    data["seed"] = seed
    data["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sidecar = image_path.with_suffix(".json")
    try:
        sidecar.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Sidecar non scritto per %s: %s", image_path.name, exc)


def _role_or_none(wf: WorkflowTemplate, role: str) -> Any:
    spec = wf.mapping.get(role)
    if not spec or "node" not in spec or "field" not in spec:
        return None
    try:
        return wf.get_param(spec["node"], spec["field"])
    except KeyError:
        return None


def _safe_set_role(wf: WorkflowTemplate, role: str, value: Any) -> None:
    if role in wf.mapping:
        try:
            wf.set_role(role, value)
        except KeyError:
            pass  # ruolo mappato ma nodo assente nel grafo — skip


def _safe_set_prompt(wf: WorkflowTemplate, positive: str, negative: str) -> None:
    try:
        wf.set_prompt(positive, negative)
    except KeyError as exc:
        logger.warning("Mapping prompt mancante nel workflow: %s", exc)


def _safe_set_seed(wf: WorkflowTemplate, seed: int) -> None:
    try:
        wf.set_seed(seed)
    except KeyError as exc:
        logger.warning("Mapping seed mancante nel workflow: %s", exc)


def _safe_set_dimensions(wf: WorkflowTemplate, width: int, height: int) -> None:
    try:
        wf.set_dimensions(width, height)
    except KeyError as exc:
        logger.warning("Mapping dimensioni mancante nel workflow: %s", exc)


def _resolve_lora_path(project: Project) -> Optional[Path]:
    """Risolve il path su disco del LoRA attivo del progetto.

    Prova prima path assoluto, poi relativo alla cartella training runs.
    """
    if project.active_lora is None:
        return None
    lora_file = Path(project.active_lora.checkpoint)
    if lora_file.is_absolute() and lora_file.exists():
        return lora_file
    candidate = project.training_runs_dir / lora_file
    if candidate.exists():
        return candidate
    return None


def _copy_to_comfy_input(src: Path, comfy_input_dir: Path) -> str:
    """Copia src in comfy_input_dir con nome univoco. Ritorna il basename.

    Il nome univoco evita collisioni tra sessioni e tra ricette diverse
    in esecuzione concorrente.
    """
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"vf_{uuid.uuid4().hex[:12]}{src.suffix}"
    shutil.copy2(src, comfy_input_dir / unique_name)
    logger.debug("Input copiato in ComfyUI: %s → %s", src.name, unique_name)
    return unique_name
