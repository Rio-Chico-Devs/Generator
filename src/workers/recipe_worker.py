"""Worker QThread per esecuzione ricette via ComfyUI.

Data una RecipeDef + parametri utente + Project, parametrizza il
WorkflowTemplate della ricetta e lo esegue via ComfyClient, emettendo
signal Qt thread-safe verso la UI.

Tutto il lavoro pesante (I/O, WebSocket) avviene nel thread worker;
la UI riceve solo signal. MAI mutare widget Qt da qui.

Riferimento: docs/COMFY_ENGINE.md, docs/RECIPES.md
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.comfy.client import ComfyClientProtocol
from src.comfy.workflow import WorkflowTemplate
from src.core.project import Project
from src.core.recipes import RecipeDef
from src.utils.paths import get_assets_dir
from src.utils.seed import resolve_seed

logger = logging.getLogger(__name__)


class RecipeWorker(QThread):
    """Esegue una ricetta ComfyUI in un thread separato.

    Uso tipico:
        worker = RecipeWorker(recipe, params, project, client, out_dir)
        worker.progress.connect(progress_bar.setValue)
        worker.image_ready.connect(gallery.add_image)
        worker.finished_ok.connect(on_done)
        worker.start()
    """

    progress    = pyqtSignal(int, int)  # step_corrente, totale_step
    image_ready = pyqtSignal(Path)      # path immagine appena prodotta
    error       = pyqtSignal(str)       # messaggio errore (fatale)
    finished_ok = pyqtSignal(list)      # list[Path] — tutte le immagini

    def __init__(
        self,
        recipe: RecipeDef,
        user_params: dict[str, Any],
        project: Project,
        client: ComfyClientProtocol,
        output_dir: Path,
        comfy_input_dir: Optional[Path] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._recipe = recipe
        self._user_params = user_params
        self._project = project
        self._client = client
        self._output_dir = output_dir
        self._comfy_input_dir = comfy_input_dir
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
        except Exception as exc:
            logger.exception("Errore non gestito in RecipeWorker")
            self.error.emit(str(exc))

    def _run_impl(self) -> None:
        wf_path = get_assets_dir() / "workflows" / self._recipe.workflow_file
        wf = WorkflowTemplate(wf_path)

        _parametrize(
            wf,
            self._recipe,
            self._user_params,
            self._project,
            self._comfy_input_dir,
        )

        if self._aborted:
            self.error.emit("Generazione interrotta dall'utente.")
            return

        prompt_id = self._client.submit(wf.build())
        logger.info(
            "Ricetta '%s' sottomessa — prompt_id=%s", self._recipe.id, prompt_id
        )

        output_paths = self._client.wait_for_completion(
            prompt_id,
            progress_callback=lambda s, t: self.progress.emit(s, t),
        )

        if self._aborted:
            self.error.emit("Generazione interrotta dall'utente.")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        final_paths: list[Path] = []
        for src in output_paths:
            dest = self._output_dir / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            final_paths.append(dest)
            self.image_ready.emit(dest)

        self.finished_ok.emit(final_paths)
        logger.info(
            "Ricetta '%s' completata: %d immagini", self._recipe.id, len(final_paths)
        )


# ---------------------------------------------------------------------------
# Parametrizzazione — funzione pura, testabile senza Qt
# ---------------------------------------------------------------------------

# Chiavi con semantica speciale: gestite esplicitamente sopra, non passate
# a set_role nel loop residuo.
_HANDLED_KEYS = frozenset(
    {"prompt", "negative", "seed", "width", "height", "lora_weight"}
)


def _parametrize(
    wf: WorkflowTemplate,
    recipe: RecipeDef,
    user_params: dict[str, Any],
    project: Project,
    comfy_input_dir: Optional[Path] = None,
) -> None:
    """Applica user_params + contesto progetto al WorkflowTemplate.

    Ordine deliberato: checkpoint → prompt → seed → dimensioni → LoRA
    → immagini input → knob numerici residui.

    Funzione pura rispetto a Qt: nessun signal, nessun thread.
    """
    # 1. Checkpoint dal progetto (base_model → nome file safetensors)
    if project.base_model is not None:
        ckpt_name = f"{project.base_model.id}.safetensors"
        _safe_set_role(wf, "checkpoint", ckpt_name)

    # 2. Prompt: prepend trigger prefix del progetto se presente
    positive = str(user_params.get("prompt", ""))
    if project.trigger_prompt_prefix:
        positive = f"{project.trigger_prompt_prefix} {positive}".strip()
    negative = str(
        user_params.get("negative", project.default_negative_prompt)
    )
    _safe_set_prompt(wf, positive, negative)

    # 3. Seed: -1 viene risolto in un valore casuale riproducibile
    raw_seed = int(user_params.get("seed", -1))
    actual_seed = resolve_seed(raw_seed)
    _safe_set_seed(wf, actual_seed)

    # 4. Dimensioni: user_params > default progetto
    width = int(
        user_params.get("width", project.default_generation_params.width)
    )
    height = int(
        user_params.get("height", project.default_generation_params.height)
    )
    _safe_set_dimensions(wf, width, height)

    # 5. LoRA del personaggio — il cuore: senza questo il personaggio non esiste
    lora_weight = float(user_params.get("lora_weight", 0.85))
    if project.active_lora is not None:
        lora_path = _resolve_lora_path(project)
        if lora_path is not None:
            wf.set_lora(str(lora_path), lora_weight)
        else:
            logger.warning(
                "LoRA '%s' non trovato su disco — generazione senza stile personaggio",
                project.active_lora.checkpoint,
            )
    else:
        logger.warning(
            "Progetto '%s' senza active_lora — generazione senza stile personaggio",
            project.name,
        )

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


# ---------------------------------------------------------------------------
# Helper interni
# ---------------------------------------------------------------------------


def _safe_set_role(wf: WorkflowTemplate, role: str, value: Any) -> None:
    if role in wf.mapping:
        wf.set_role(role, value)


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
