"""Worker QThread per il training LoRA via sd-scripts (kohya).

Pipeline (docs/TRAINING.md):
  1. Prepare dataset  — crea struttura kohya format in processed/
  2. Generate config  — scrive TOML validato in run_dir/config.toml
  3. Launch subprocess — accelerate launch sdxl_train_network.py
  4. Monitor loop     — parse stdout, emette segnali progress/loss/sample
  5. Finalize         — aggiorna status.json, project.active_lora

Il subprocess gira in isolamento (working dir = run_dir); se crasha,
status.json contiene last_checkpoint_path per resume dal punto esatto.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.project import ActiveLora, Project
from src.training.config import TrainingParams, generate_toml, sdscripts_launch_cmd
from src.training.dataset_prep import DatasetReport, prepare_dataset
from src.training.presets import PresetId, TrainingPreset, get_preset
from src.training.run_status import (
    STATUS_ABORTED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PREPARING,
    STATUS_RUNNING,
    LossPoint,
    RunStatus,
)
from src.utils.atomic import atomic_write_text

logger = logging.getLogger(__name__)

# Pattern per parsing stdout sd-scripts
_RE_STEP = re.compile(r"steps:\s*(\d+)%.*?(\d+)/(\d+)", re.IGNORECASE)
_RE_EPOCH = re.compile(r"epoch\s+(\d+)/(\d+)", re.IGNORECASE)
_RE_LOSS = re.compile(r"loss[:\s=]+([0-9]+\.[0-9]+)", re.IGNORECASE)
_RE_SAMPLE = re.compile(r"saving sample image.*?:\s*(.+\.png)", re.IGNORECASE)
_RE_CHECKPOINT = re.compile(r"saving checkpoint.*?:\s*(.+\.safetensors)", re.IGNORECASE)
_RE_LOSS_STEP = re.compile(
    r"step\s+loss:\s*([0-9.eE+\-]+).*?(\d+)", re.IGNORECASE
)

# Profilo di loss preoccupante (perdita instabile)
_LOSS_SPIKE_FACTOR = 5.0     # loss > 5× media → warning
_LOSS_WINDOW = 20             # finestra per media mobile


class TrainingWorker(QThread):
    """Esegue il training LoRA in un thread separato.

    Segnali (tutti thread-safe via Qt):
        progress(step, total_steps)       — avanzamento step
        epoch_changed(epoch, total_epochs)
        loss_update(loss, step)           — punto di loss per il grafico
        log_line(text)                    — riga grezza stdout/stderr
        sample_ready(path)                — nuova sample image disponibile
        checkpoint_saved(path, epoch)     — checkpoint intermedio salvato
        dataset_report(report)            — esito preparazione dataset
        finished_ok(checkpoint_path)      — training completato, path finale
        error(message)                    — errore fatale
        cooling(temp, target)             — pausa termica (futuro)
    """

    progress = pyqtSignal(int, int)          # step_corrente, step_totale
    epoch_changed = pyqtSignal(int, int)     # epoch, total_epochs
    loss_update = pyqtSignal(float, int)     # loss, step
    log_line = pyqtSignal(str)               # riga stdout grezza
    sample_ready = pyqtSignal(Path)          # nuova sample image
    checkpoint_saved = pyqtSignal(Path, int) # path checkpoint, epoch
    dataset_report = pyqtSignal(object)      # DatasetReport
    finished_ok = pyqtSignal(Path)           # path checkpoint finale
    error = pyqtSignal(str)

    def __init__(
        self,
        project: Project,
        preset_id: PresetId,
        params: TrainingParams,
        sdscripts_dir: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._preset_id = preset_id
        self._params = params
        self._sdscripts_dir = Path(sdscripts_dir)
        self._aborted = False
        self._process: Optional[subprocess.Popen] = None
        self._run_status: Optional[RunStatus] = None

    # --- Abort ------------------------------------------------------------

    def abort(self) -> None:
        """Richiede la terminazione pulita del subprocess di training."""
        self._aborted = True
        proc = self._process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    # --- Thread entry point -----------------------------------------------

    def run(self) -> None:
        preset = get_preset(self._preset_id)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self._project.training_runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        status = RunStatus(
            run_id=run_id,
            preset_id=self._preset_id.value,
            max_epochs=self._params.epochs,
        )
        self._run_status = status
        status.save(run_dir)

        # 1. Prepara dataset
        status.status = STATUS_PREPARING
        status.save(run_dir)

        try:
            manifest = _load_manifest(self._project)
            report = prepare_dataset(
                images_dir=self._project.dataset_images_dir,
                processed_dir=self._project.dataset_processed_dir,
                activator_tag=self._project.activator_tag,
                manifest=manifest,
            )
        except Exception as exc:
            self.error.emit(f"Preparazione dataset fallita: {exc}")
            status.status = STATUS_ERROR
            status.error_message = str(exc)
            status.save(run_dir)
            return

        self.dataset_report.emit(report)

        if not report.is_ready:
            self.error.emit(report.readiness_message)
            status.status = STATUS_ERROR
            status.error_message = report.readiness_message
            status.save(run_dir)
            return

        if self._aborted:
            status.status = STATUS_ABORTED
            status.save(run_dir)
            return

        # 2. Individua modello base
        base_model_path = _find_base_model(self._project, preset)
        if base_model_path is None:
            msg = (
                f"Modello base '{preset.model_family}' non trovato. "
                "Scarica il checkpoint e mettilo in models/base/."
            )
            self.error.emit(msg)
            status.status = STATUS_ERROR
            status.error_message = msg
            status.save(run_dir)
            return

        # 3. Genera config TOML
        config_path = run_dir / "config.toml"
        try:
            toml_content = generate_toml(
                preset=preset,
                params=self._params,
                base_model_path=base_model_path,
                dataset_dir=report.output_dir.parent,
                run_dir=run_dir,
                activator_tag=self._project.activator_tag,
            )
            atomic_write_text(config_path, toml_content)
            status.config_path = str(config_path)
            status.save(run_dir)
        except Exception as exc:
            self.error.emit(f"Generazione config TOML fallita: {exc}")
            status.status = STATUS_ERROR
            status.error_message = str(exc)
            status.save(run_dir)
            return

        # 4. Costruisce comando accelerate
        cmd = sdscripts_launch_cmd(self._sdscripts_dir, config_path, preset.model_family)
        if not cmd:
            msg = (
                f"sd-scripts non trovato in {self._sdscripts_dir}. "
                "Installa kohya_ss (docs/DEVELOPMENT.md)."
            )
            self.error.emit(msg)
            status.status = STATUS_ERROR
            status.error_message = msg
            status.save(run_dir)
            return

        # 5. Lancia subprocess
        status.status = STATUS_RUNNING
        status.save(run_dir)

        log_file_path = run_dir / "training.log"
        loss_window: list[float] = []

        try:
            with open(log_file_path, "w", encoding="utf-8") as log_fh:
                self._process = subprocess.Popen(
                    cmd,
                    cwd=str(run_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                assert self._process.stdout is not None

                for raw_line in self._process.stdout:
                    line = raw_line.rstrip()
                    log_fh.write(line + "\n")
                    self.log_line.emit(line)

                    if self._aborted:
                        self._process.terminate()
                        break

                    self._parse_line(line, status, run_dir, loss_window)

                self._process.wait()
                ret = self._process.returncode

        except Exception as exc:
            msg = f"Errore avvio subprocess: {exc}"
            logger.exception(msg)
            self.error.emit(msg)
            status.status = STATUS_ERROR
            status.error_message = msg
            status.save(run_dir)
            return
        finally:
            self._process = None

        if self._aborted:
            status.status = STATUS_ABORTED
            status.save(run_dir)
            return

        if ret != 0:
            msg = f"sd-scripts terminato con codice {ret}. Vedi training.log."
            self.error.emit(msg)
            status.status = STATUS_ERROR
            status.error_message = msg
            status.save(run_dir)
            return

        # 6. Finalizza: cerca checkpoint finale
        final = _find_final_checkpoint(run_dir)
        if final is None:
            msg = "Training completato ma nessun checkpoint .safetensors trovato."
            self.error.emit(msg)
            status.status = STATUS_ERROR
            status.error_message = msg
            status.save(run_dir)
            return

        status.status = STATUS_DONE
        status.final_checkpoint_path = str(final)
        status.finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        status.save(run_dir)

        # Aggiorna project.active_lora
        self._project.active_lora = ActiveLora(
            run_id=run_id,
            checkpoint=str(final),
            weight=0.85,
        )
        self._project.stats.training_runs += 1
        self._project.save()

        self.finished_ok.emit(final)

    # --- Parsing stdout ---------------------------------------------------

    def _parse_line(
        self,
        line: str,
        status: RunStatus,
        run_dir: Path,
        loss_window: list[float],
    ) -> None:
        """Estrae progress/loss/checkpoint dalla riga stdout di sd-scripts."""

        # Epoch
        m = _RE_EPOCH.search(line)
        if m:
            epoch, total = int(m.group(1)), int(m.group(2))
            status.current_epoch = epoch
            status.max_epochs = total
            self.epoch_changed.emit(epoch, total)
            status.save(run_dir)

        # Step progress (tqdm format: "  5%|... 50/1000")
        m = _RE_STEP.search(line)
        if m:
            step, total = int(m.group(2)), int(m.group(3))
            status.current_step = step
            status.max_steps = total
            self.progress.emit(step, total)

        # Loss (formato: "step loss: 0.0842  step: 142")
        m = _RE_LOSS_STEP.search(line)
        if not m:
            m = _RE_LOSS.search(line)
        if m:
            try:
                loss_val = float(m.group(1))
                step = status.current_step
                lp = LossPoint(step=step, loss=loss_val)
                # Campiona: una ogni 5 step per non inflazionare il JSON
                if len(status.loss_history) == 0 or step - status.loss_history[-1].step >= 5:
                    status.loss_history.append(lp)
                    status.save(run_dir)
                self.loss_update.emit(loss_val, step)
                # Warning instabilità
                loss_window.append(loss_val)
                if len(loss_window) > _LOSS_WINDOW:
                    loss_window.pop(0)
                if len(loss_window) >= _LOSS_WINDOW:
                    avg = sum(loss_window[:-1]) / (_LOSS_WINDOW - 1)
                    if avg > 0 and loss_val > avg * _LOSS_SPIKE_FACTOR:
                        self.log_line.emit(
                            f"⚠ Loss spike rilevato ({loss_val:.4f} vs media {avg:.4f}) "
                            "— possibile instabilità del training."
                        )
            except (ValueError, IndexError):
                pass

        # Sample image salvata
        m = _RE_SAMPLE.search(line)
        if m:
            sample_path = run_dir / m.group(1).strip()
            if not sample_path.exists():
                # Prova nella sottocartella samples/
                sample_path = run_dir / "samples" / Path(m.group(1)).name
            if sample_path.exists():
                self.sample_ready.emit(sample_path)

        # Checkpoint intermedio salvato
        m = _RE_CHECKPOINT.search(line)
        if m:
            ckpt_path = Path(m.group(1).strip())
            if not ckpt_path.is_absolute():
                ckpt_path = run_dir / "checkpoints" / ckpt_path.name
            if ckpt_path.exists():
                status.last_checkpoint_path = str(ckpt_path)
                status.save(run_dir)
                self.checkpoint_saved.emit(ckpt_path, status.current_epoch)


# --- Helpers ----------------------------------------------------------


def _load_manifest(project: Project) -> dict | None:
    try:
        if project.dataset_manifest_path.exists():
            import json
            return json.loads(project.dataset_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _find_base_model(project: Project, preset: TrainingPreset) -> Path | None:
    """Cerca il checkpoint base nella cartella models/base/.

    Preferisce file .safetensors compatibili con la famiglia del preset."""
    from src.utils.paths import get_models_dir

    base_dir = get_models_dir() / "base"
    if not base_dir.exists():
        return None

    # Cerca prima nella cartella specifica per la famiglia
    family_dir = base_dir / preset.model_family
    if family_dir.exists():
        candidates = sorted(family_dir.glob("*.safetensors"))
        if candidates:
            return candidates[0]

    # Fallback: cerca in tutta la cartella base
    candidates = sorted(base_dir.rglob("*.safetensors"))
    if candidates:
        return candidates[0]

    # Fallback: model_index.json (formato diffusers)
    model_index = next(base_dir.rglob("model_index.json"), None)
    if model_index:
        return model_index.parent

    return None


def _find_final_checkpoint(run_dir: Path) -> Path | None:
    """Trova il checkpoint finale (nome 'lora.safetensors' o l'ultimo creato)."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Cerca prima il file finale canonico
    final = ckpt_dir / "lora.safetensors"
    if final.exists():
        return final

    # Fallback: il .safetensors più recente
    candidates = sorted(
        ckpt_dir.glob("*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
