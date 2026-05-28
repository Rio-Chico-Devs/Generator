"""Worker QThread per un singolo step della sessione di training guidato.

Genera N candidati per lo step corrente della GuidedSession e li consegna
alla UI uno alla volta tramite signal. NON decide: l'approvazione/rifiuto
avviene nella UI; il DiaryEntry viene scritto fuori da qui.

Step 1 (denoise=1.0, nessun input): usa il workflow base_txt2img.
Step 2+ (denoise<1.0, input image): usa guided_img2img (VAEEncode da PNG).

IP-Adapter (Pilastro B): se la TechniqueLibrary contiene riferimenti attivi
per lo step corrente, il worker lo registra nel sidecar candidato ma NON li
applica ancora — richiederebbe custom nodes ComfyUI (Fase 2b, TODO).

Riferimento: docs/GUIDED_TRAINING.md, docs/COMFY_ENGINE.md
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.comfy.client import ComfyClientProtocol, ComfyInterrupted
from src.comfy.workflow import WorkflowTemplate
from src.core.catalog import dialect_for_model
from src.core.guidance.session import Candidate, GuidedSession, StepDefinition
from src.core.guidance.technique_library import TechniqueLibrary
from src.core.project import Project
from src.core.prompting import build_prompts, get_preset
from src.utils.gpu_monitor import CooldownOutcome, SafetyConfig, ThermalGovernor
from src.utils.paths import get_assets_dir
from src.utils.seed import resolve_seed
from src.workers.recipe_worker import _resolve_lora_path

logger = logging.getLogger(__name__)

# Workflow da usare per ogni caso
_WF_TXT2IMG = "base_txt2img.json"
_WF_IMG2IMG = "guided_img2img.json"


@dataclass
class CandidateRecord:
    """Parametri effettivi di un candidato guidato, scritti nel sidecar JSON."""

    session_id: str
    step_id: str
    candidate_index: int
    model_id: str
    dialect: str
    positive: str
    negative: str
    width: int
    height: int
    clip_skip: int
    seed: int
    steps: int
    cfg: float
    denoise: float
    lora_name: Optional[str] = None
    lora_weight: Optional[float] = None
    input_image: Optional[str] = None        # basename del PNG input (step 2+)
    technique_refs: list[str] = None         # ref_id attivi (IP-Adapter TODO)
    is_img2img: bool = False
    app_version: str = "0.1.0"
    created_at: str = ""

    def __post_init__(self):
        if self.technique_refs is None:
            self.technique_refs = []


class GuidedWorker(QThread):
    """Esegue uno step della pipeline guidata, genera N candidati.

    Uso tipico:
        worker = GuidedWorker(session, project, client, session_step_dir,
                              comfy_input_dir, technique_library)
        worker.candidate_ready.connect(view.on_candidate)
        worker.step_complete.connect(view.on_step_complete)
        worker.error.connect(view.on_error)
        worker.start()

    Al termine di step_complete la UI mostra i candidati, aspetta la scelta
    dell'utente, poi crea un nuovo GuidedWorker per lo step successivo.
    """

    candidate_ready = pyqtSignal(int, Path)  # (candidate_index, image_path)
    step_complete   = pyqtSignal(list)       # list[Candidate]
    progress        = pyqtSignal(int, int)   # diffusion step corrente, totale
    cooling         = pyqtSignal(int, int)   # temp_gpu, target_temp
    error           = pyqtSignal(str)

    def __init__(
        self,
        session: GuidedSession,
        project: Project,
        client: ComfyClientProtocol,
        step_dir: Path,
        comfy_input_dir: Optional[Path] = None,
        technique_library: Optional[TechniqueLibrary] = None,
        attempt: int = 0,
        safety: Optional[SafetyConfig] = None,
        gpu_read_fn: Optional[Callable[[], Any]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._project = project
        self._client = client
        self._step_dir = step_dir
        self._comfy_input_dir = comfy_input_dir
        self._library = technique_library
        # Tentativo dello step (0 = primo, >0 = dopo "Rigenera tutti"): sposta
        # i seed così che ogni rigenerazione esplori candidati diversi.
        self._attempt = attempt
        self._safety = safety or SafetyConfig()
        if gpu_read_fn is not None:
            self._governor = ThermalGovernor(self._safety, read_fn=gpu_read_fn)
        else:
            self._governor = ThermalGovernor(self._safety)
        self._aborted = False

    def abort(self) -> None:
        self._aborted = True
        try:
            self._client.interrupt()
        except Exception as exc:
            logger.warning("interrupt() fallito: %s", exc)
        try:
            self._client.clear_queue()
        except Exception as exc:
            logger.warning("clear_queue() fallito: %s", exc)

    def stop(self, timeout_ms: int = 5000) -> None:
        """Interrompe il worker e attende la sua terminazione (max timeout_ms)."""
        if not self.isRunning():
            return
        self.abort()
        if not self.wait(timeout_ms):
            logger.warning("GuidedWorker non terminato in %dms — terminate()", timeout_ms)
            self.terminate()
            self.wait(1000)

    def run(self) -> None:
        try:
            self._run_impl()
        except ComfyInterrupted:
            self.error.emit("Generazione interrotta dall'utente.")
        except Exception as exc:
            logger.exception("Errore non gestito in GuidedWorker")
            self.error.emit(str(exc))

    def _run_impl(self) -> None:
        step_def = self._session.current_step_def
        if step_def is None:
            self.error.emit("La pipeline è già completata — nessuno step da generare.")
            return

        input_image = self._session.step_input_image()
        is_img2img = input_image is not None and step_def.denoise_strength < 1.0

        wf = self._load_workflow(is_img2img)
        self._step_dir.mkdir(parents=True, exist_ok=True)

        # Tecniche attive per questo step (segnate nel sidecar; IP-Adapter TODO)
        active_refs: list[str] = []
        if self._library:
            for ref in self._library.active_for_step(step_def):
                active_refs.append(ref.ref_id)
            if active_refs:
                logger.info(
                    "IP-Adapter [step=%s]: %d ref attivi — not yet implemented",
                    step_def.id, len(active_refs),
                )

        record_base = self._parametrize(wf, step_def, input_image, is_img2img)
        record_base.technique_refs = active_refs

        candidates: list[Candidate] = []

        for i in range(step_def.n_candidates):
            if self._aborted:
                break

            if not self._wait_until_cool():
                if not self._aborted:
                    return
                break

            if self._session.seed < 0:
                seed_i = resolve_seed(-1)  # sessione a seed casuale: ogni candidato random
            else:
                # Deterministico: l'offset per tentativo evita candidati identici
                # tra una rigenerazione e l'altra dello stesso step.
                # Modulo 2^32 per restare nel range valido per i sampler.
                seed_i = (self._session.seed + self._attempt * 10_000 + i) % (2**32)
            wf.set_seed(seed_i)

            prompt_id = self._client.submit(wf.build())
            logger.info(
                "Guided step '%s' candidato %d/%d — prompt_id=%s seed=%d",
                step_def.id, i + 1, step_def.n_candidates, prompt_id, seed_i,
            )
            outputs = self._client.wait_for_completion(
                prompt_id,
                progress_callback=lambda s, t: self.progress.emit(s, t),
            )

            if self._aborted:
                break

            for src in outputs:
                dest = self._step_dir / f"candidate_{i:02d}{src.suffix}"
                if src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)

                record_i = replace(
                    record_base,
                    candidate_index=i,
                    seed=seed_i,
                    technique_refs=list(record_base.technique_refs),
                )
                _write_candidate_sidecar(dest, record_i)

                sidecar_path = dest.with_suffix(".json")
                c = Candidate(
                    index=i,
                    image_path=dest,
                    sidecar_path=sidecar_path,
                )
                candidates.append(c)
                self.candidate_ready.emit(i, dest)
                break  # una immagine per candidato

            # Piccolo cooldown tra candidati (no freno termico, solo pausa)
            if i < step_def.n_candidates - 1 and self._safety.enabled:
                self._interruptible_sleep(self._safety.cooldown_between_images_sec)

        if self._aborted:
            self.error.emit("Generazione interrotta dall'utente.")
            return

        self.step_complete.emit(candidates)
        logger.info(
            "Guided step '%s' completato: %d candidati", step_def.id, len(candidates)
        )

    # --- Workflow setup --------------------------------------------------

    def _load_workflow(self, is_img2img: bool) -> WorkflowTemplate:
        wf_file = _WF_IMG2IMG if is_img2img else _WF_TXT2IMG
        wf_path = get_assets_dir() / "workflows" / wf_file
        return WorkflowTemplate(wf_path)

    def _parametrize(
        self,
        wf: WorkflowTemplate,
        step_def: StepDefinition,
        input_image: Optional[Path],
        is_img2img: bool,
    ) -> CandidateRecord:
        """Applica project + step al workflow. Ritorna un record base (seed=0)."""
        # Modello base
        model_id = self._project.base_model.id if self._project.base_model else ""
        if model_id:
            _safe_set(wf, "checkpoint", f"{model_id}.safetensors")

        # Prompt con quality-tag del dialetto
        dialect = dialect_for_model(model_id)
        preset = get_preset(dialect)
        built = build_prompts(
            preset,
            user_positive=self._session.prompt,
            trigger_prefix=self._project.trigger_prompt_prefix or "",
            user_negative=self._session.negative_prompt,
            project_negative=self._project.default_negative_prompt or "",
        )
        try:
            wf.set_prompt(built.positive, built.negative)
        except KeyError as exc:
            logger.warning("Mapping prompt mancante: %s", exc)
        _safe_set(wf, "clip_skip", -abs(built.clip_skip))

        # Parametri numerici dallo step
        _safe_set(wf, "steps", step_def.steps)
        _safe_set(wf, "cfg", step_def.guidance_scale)
        _safe_set(wf, "denoise", step_def.denoise_strength)

        # Dimensioni dal progetto
        w = self._project.default_generation_params.width
        h = self._project.default_generation_params.height
        if not is_img2img:
            # Solo txt2img ha width/height espliciti; img2img li eredita dall'input
            _safe_set(wf, "width", w)
            _safe_set(wf, "height", h)

        # LoRA del progetto
        lora_name: Optional[str] = None
        lora_weight: Optional[float] = None
        if self._project.active_lora is not None:
            lora_path = _resolve_lora_path(self._project)
            if lora_path is not None:
                weight = float(getattr(self._project.active_lora, "weight", 0.85) or 0.85)
                wf.set_lora(str(lora_path), weight)
                lora_name = lora_path.name
                lora_weight = weight
            else:
                logger.warning(
                    "LoRA '%s' non trovato — guided step senza stile",
                    self._project.active_lora.checkpoint,
                )

        # Immagine input per img2img
        input_basename: Optional[str] = None
        if is_img2img and input_image is not None:
            if self._comfy_input_dir is not None:
                input_basename = _copy_to_comfy_input(input_image, self._comfy_input_dir)
                _safe_set(wf, "input_image", input_basename)
            else:
                logger.warning(
                    "comfy_input_dir non fornita — img2img non applicabile per step '%s'",
                    step_def.id,
                )

        return CandidateRecord(
            session_id=self._session.session_id,
            step_id=step_def.id,
            candidate_index=0,
            model_id=model_id,
            dialect=dialect,
            positive=built.positive,
            negative=built.negative,
            width=w,
            height=h,
            clip_skip=built.clip_skip,
            seed=0,
            steps=step_def.steps,
            cfg=step_def.guidance_scale,
            denoise=step_def.denoise_strength,
            lora_name=lora_name,
            lora_weight=lora_weight,
            input_image=input_basename,
            is_img2img=is_img2img,
        )

    # --- Freno termico / sleep ------------------------------------------

    def _wait_until_cool(self) -> bool:
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
                f"{self._governor.cfg.resume_temp_c}°C. "
                "Generazione interrotta per sicurezza."
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
# Helper
# ---------------------------------------------------------------------------


def _safe_set(wf: WorkflowTemplate, role: str, value: Any) -> None:
    if role in wf.mapping:
        try:
            wf.set_role(role, value)
        except KeyError:
            pass


def _write_candidate_sidecar(image_path: Path, record: CandidateRecord) -> None:
    record.created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    data = {k: (str(v) if isinstance(v, Path) else v) for k, v in record.__dict__.items()}
    sidecar = image_path.with_suffix(".json")
    try:
        sidecar.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Sidecar candidato non scritto: %s", exc)


def _copy_to_comfy_input(src: Path, comfy_input_dir: Path) -> str:
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"vf_guided_{uuid.uuid4().hex[:12]}{src.suffix}"
    shutil.copy2(src, comfy_input_dir / unique_name)
    logger.debug("Input guided copiato: %s → %s", src.name, unique_name)
    return unique_name
