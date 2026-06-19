"""Schermata Training LoRA.

Layout:
    +----------------------------+----------------------------------+
    |  Pannello configurazione   |  Output                          |
    |  - Preset (Veloce/Std/Max) |  [Progress bar]                  |
    |  - Epoch slider            |  [Step: 142/1000  Epoch: 2/10]   |
    |  - Pannello Advanced       |  [Loss: 0.0842 ↓]                |
    |  - Dataset info            |                                  |
    |  - VRAM check              |  [Sample images grid]            |
    |                            |                                  |
    |  [Inizia Training]         |  [Log live (QPlainTextEdit)]     |
    |  [Interrompi]              |                                  |
    +----------------------------+----------------------------------+
    |  [Banner resume se c'è run interrotto]                        |
    +---------------------------------------------------------------+

Macchina a stati:
    IDLE      → configurazione, Inizia abilitato
    RUNNING   → Inizia disabilitato, Interrompi abilitato, log live
    DONE      → mostra checkpoint, bottone "Usa questo LoRA"
    ERROR     → messaggio errore, Inizia riabilitato
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core.project import Project
from src.training.config import TrainingParams
from src.training.dataset_prep import DatasetReport
from src.training.presets import PRESETS, PresetId, get_preset
from src.training.run_status import RunStatus

logger = logging.getLogger(__name__)

_STATE_IDLE = "idle"
_STATE_RUNNING = "running"
_STATE_DONE = "done"
_STATE_ERROR = "error"

_MAX_LOG_LINES = 1000
_MAX_SAMPLE_COLS = 3


class TrainView(QWidget):
    """View principale del training LoRA."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._worker = None
        self._state = _STATE_IDLE
        self._sdscripts_dir: Optional[Path] = None
        self._loss_history: list[tuple[float, int]] = []  # (loss, step)

        self._build_ui()

    # --- Build UI --------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Banner resume (hidden by default)
        self._resume_banner = self._make_resume_banner()
        root.addWidget(self._resume_banner)
        self._resume_banner.setVisible(False)

        # Splitter: config | output
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Pannello sinistra: configurazione
        config_panel = self._make_config_panel()
        splitter.addWidget(config_panel)

        # Pannello destra: output
        output_panel = self._make_output_panel()
        splitter.addWidget(output_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 900])
        root.addWidget(splitter)

    def _make_resume_banner(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background: #3a3020; border: 1px solid #d9a441; border-radius: 4px; "
            "padding: 6px; }"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        self._resume_label = QLabel()
        self._resume_label.setWordWrap(True)
        layout.addWidget(self._resume_label, 1)
        self._resume_btn = QPushButton("▶ Riprendi")
        self._resume_btn.clicked.connect(self._on_resume)
        layout.addWidget(self._resume_btn)
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedWidth(28)
        dismiss_btn.clicked.connect(lambda: self._resume_banner.setVisible(False))
        layout.addWidget(dismiss_btn)
        return frame

    def _make_config_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(340)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Dataset info
        self._dataset_info_label = QLabel("Nessun progetto selezionato.")
        self._dataset_info_label.setWordWrap(True)
        self._dataset_info_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self._dataset_info_label)

        # Preset
        preset_group = QGroupBox("Preset")
        pg_layout = QVBoxLayout(preset_group)
        self._preset_combo = QComboBox()
        for pid, p in PRESETS.items():
            self._preset_combo.addItem(p.name, pid)
        # Default: Standard
        self._preset_combo.setCurrentIndex(1)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        pg_layout.addWidget(self._preset_combo)
        self._preset_desc = QLabel()
        self._preset_desc.setWordWrap(True)
        self._preset_desc.setStyleSheet("color: #888; font-size: 11px;")
        pg_layout.addWidget(self._preset_desc)
        layout.addWidget(preset_group)

        # Epoch slider
        epoch_group = QGroupBox("Epoch")
        eg_layout = QHBoxLayout(epoch_group)
        self._epoch_slider = QSlider(Qt.Orientation.Horizontal)
        self._epoch_slider.setRange(2, 50)
        self._epoch_slider.setValue(10)
        self._epoch_slider.setTickInterval(5)
        self._epoch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._epoch_slider.valueChanged.connect(self._on_epoch_changed)
        self._epoch_label = QLabel("10")
        self._epoch_label.setFixedWidth(30)
        eg_layout.addWidget(self._epoch_slider)
        eg_layout.addWidget(self._epoch_label)
        layout.addWidget(epoch_group)

        # VRAM check
        self._vram_label = QLabel()
        self._vram_label.setWordWrap(True)
        self._vram_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._vram_label)

        # Pannello Advanced (collassabile)
        self._advanced_group = QGroupBox("Avanzate")
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)
        self._advanced_group.toggled.connect(lambda checked: self._advanced_body.setVisible(checked))
        adv_outer = QVBoxLayout(self._advanced_group)
        self._advanced_body = QWidget()
        adv_layout = QVBoxLayout(self._advanced_body)
        adv_layout.setContentsMargins(0, 0, 0, 0)

        self._dim_spin = QSpinBox()
        self._dim_spin.setRange(4, 128)
        self._dim_spin.setValue(16)
        self._dim_spin.setPrefix("network_dim: ")
        adv_layout.addWidget(self._dim_spin)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(1e-6, 5e-3)
        self._lr_spin.setSingleStep(1e-5)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setValue(1e-4)
        self._lr_spin.setPrefix("learning_rate: ")
        adv_layout.addWidget(self._lr_spin)

        adv_outer.addWidget(self._advanced_body)
        self._advanced_body.setVisible(False)
        layout.addWidget(self._advanced_group)

        layout.addStretch()

        # Bottoni azione
        self._start_btn = QPushButton("▶  Inizia Training")
        self._start_btn.setStyleSheet(
            "QPushButton { background: #2d5a27; font-weight: bold; padding: 8px; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self._start_btn.clicked.connect(self._on_start)
        self._start_btn.setEnabled(False)
        layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏹  Interrompi")
        self._stop_btn.setStyleSheet("QPushButton { background: #5a2727; padding: 8px; }")
        self._stop_btn.clicked.connect(self._on_abort)
        self._stop_btn.setVisible(False)
        layout.addWidget(self._stop_btn)

        scroll.setWidget(container)
        self._on_preset_changed()
        return scroll

    def _make_output_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Status label
        self._status_label = QLabel("In attesa di un progetto...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("font-size: 14px; color: #aaa;")
        layout.addWidget(self._status_label)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # Step / epoch info
        self._step_label = QLabel("")
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_label.setVisible(False)
        layout.addWidget(self._step_label)

        # Loss
        self._loss_label = QLabel("")
        self._loss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loss_label.setStyleSheet("font-size: 13px;")
        self._loss_label.setVisible(False)
        layout.addWidget(self._loss_label)

        # Sample images
        sample_group = QGroupBox("Sample images")
        sg_layout = QVBoxLayout(sample_group)
        sample_scroll = QScrollArea()
        sample_scroll.setWidgetResizable(True)
        sample_scroll.setMinimumHeight(180)
        self._sample_container = QWidget()
        self._sample_grid = QGridLayout(self._sample_container)
        self._sample_grid.setSpacing(4)
        self._sample_col = 0
        self._sample_row = 0
        sample_scroll.setWidget(self._sample_container)
        sg_layout.addWidget(sample_scroll)
        layout.addWidget(sample_group)

        # Log
        log_group = QGroupBox("Log training")
        lg_layout = QVBoxLayout(log_group)
        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumBlockCount(_MAX_LOG_LINES)
        self._log_output.setMinimumHeight(150)
        self._log_output.setStyleSheet("font-family: monospace; font-size: 11px;")
        lg_layout.addWidget(self._log_output)
        layout.addWidget(log_group)

        # Checkpoint finale
        self._final_group = QGroupBox("Training completato")
        fl = QVBoxLayout(self._final_group)
        self._final_label = QLabel("")
        self._final_label.setWordWrap(True)
        fl.addWidget(self._final_label)
        self._use_lora_btn = QPushButton("✓ Usa questo LoRA nel progetto")
        self._use_lora_btn.setStyleSheet(
            "QPushButton { background: #2d5a27; font-weight: bold; padding: 6px; }"
        )
        self._use_lora_btn.clicked.connect(self._on_use_lora)
        fl.addWidget(self._use_lora_btn)
        self._final_group.setVisible(False)
        layout.addWidget(self._final_group)

        return panel

    # --- Progetto --------------------------------------------------------

    def set_project(self, project: Project) -> None:
        self._project = project
        self._refresh_dataset_info()
        self._check_resume()
        self._update_start_enabled()

    def set_sdscripts_dir(self, path: Path) -> None:
        self._sdscripts_dir = Path(path)
        self._update_start_enabled()

    # --- Slot UI ---------------------------------------------------------

    def _on_preset_changed(self) -> None:
        pid = self._preset_combo.currentData()
        if pid is None:
            return
        preset = get_preset(pid)
        self._preset_desc.setText(preset.description)
        self._dim_spin.setValue(preset.network_dim)
        self._lr_spin.setValue(preset.learning_rate)
        self._epoch_slider.setValue(preset.max_train_epochs)
        self._vram_label.setText(
            f"VRAM richiesta: ≥{preset.min_vram_gb} GB · "
            f"Tempo stimato: {preset.estimated_time_per_100} per 100 immagini"
        )

    def _on_epoch_changed(self, value: int) -> None:
        self._epoch_label.setText(str(value))

    def _on_start(self) -> None:
        if self._project is None or self._sdscripts_dir is None:
            return
        if self._state == _STATE_RUNNING:
            return

        preset_id = self._preset_combo.currentData()
        if preset_id is None:
            return

        params = TrainingParams(
            epochs=self._epoch_slider.value(),
            network_dim=self._dim_spin.value() if self._advanced_group.isChecked() else None,
            learning_rate=self._lr_spin.value() if self._advanced_group.isChecked() else None,
        )

        self._start_training(preset_id, params)

    def _on_abort(self) -> None:
        if self._worker is not None:
            self._worker.abort()
        self._set_state(_STATE_IDLE)
        self._status_label.setText("Training interrotto dall'utente.")

    def _on_resume(self) -> None:
        """Riprende il training dall'ultimo checkpoint."""
        if self._project is None:
            return
        runs = RunStatus.load_all(self._project.training_runs_dir)
        resumable = next((r for r in runs if r.is_resumable), None)
        if resumable is None:
            self._resume_banner.setVisible(False)
            return

        preset_id = PresetId(resumable.preset_id) if resumable.preset_id else PresetId.STANDARD
        params = TrainingParams(
            epochs=resumable.max_epochs,
            resume_from=Path(resumable.last_checkpoint_path) if resumable.last_checkpoint_path else None,
        )
        self._resume_banner.setVisible(False)
        self._start_training(preset_id, params)

    def _on_use_lora(self) -> None:
        if self._project is None:
            return
        QMessageBox.information(
            self,
            "LoRA attivo",
            f"Il LoRA addestrato è ora attivo nel progetto «{self._project.name}».\n\n"
            "Puoi generare immagini con il tuo stile nella schermata Generate.",
        )

    # --- Training lifecycle ----------------------------------------------

    def _start_training(self, preset_id: PresetId, params: TrainingParams) -> None:
        from src.workers.training_worker import TrainingWorker

        sdscripts = self._sdscripts_dir
        if sdscripts is None:
            QMessageBox.warning(
                self, "sd-scripts mancante",
                "Percorso sd-scripts non configurato.\n"
                "Imposta VFORGE_SDSCRIPTS_DIR o installa sd-scripts (docs/DEVELOPMENT.md).",
            )
            return

        self._set_state(_STATE_RUNNING)
        self._loss_history.clear()
        self._log_output.clear()
        self._clear_samples()
        self._final_group.setVisible(False)

        self._worker = TrainingWorker(
            project=self._project,
            preset_id=preset_id,
            params=params,
            sdscripts_dir=sdscripts,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.epoch_changed.connect(self._on_epoch_signal)
        self._worker.loss_update.connect(self._on_loss)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.sample_ready.connect(self._on_sample)
        self._worker.dataset_report.connect(self._on_dataset_report)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, step: int, total: int) -> None:
        pct = int(100 * step / total) if total > 0 else 0
        self._progress_bar.setValue(pct)
        self._step_label.setText(f"Step {step}/{total}")

    def _on_epoch_signal(self, epoch: int, total: int) -> None:
        self._status_label.setText(f"Epoch {epoch}/{total} in corso...")

    def _on_loss(self, loss: float, step: int) -> None:
        self._loss_history.append((loss, step))
        arrow = "↓" if len(self._loss_history) < 2 or loss <= self._loss_history[-2][0] else "↑"
        self._loss_label.setText(f"Loss: {loss:.4f} {arrow}")

    def _on_log_line(self, line: str) -> None:
        self._log_output.appendPlainText(line)
        # Scorri in fondo automaticamente
        sb = self._log_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_sample(self, path: Path) -> None:
        px = QPixmap(str(path))
        if px.isNull():
            return
        px = px.scaledToHeight(160, Qt.TransformationMode.SmoothTransformation)
        lbl = QLabel()
        lbl.setPixmap(px)
        lbl.setToolTip(path.name)
        self._sample_grid.addWidget(lbl, self._sample_row, self._sample_col)
        self._sample_col += 1
        if self._sample_col >= _MAX_SAMPLE_COLS:
            self._sample_col = 0
            self._sample_row += 1

    def _on_dataset_report(self, report: DatasetReport) -> None:
        if report.warnings:
            for w in report.warnings:
                self._on_log_line(f"⚠ {w}")

    def _on_finished(self, checkpoint_path: Path) -> None:
        self._set_state(_STATE_DONE)
        self._final_label.setText(
            f"Checkpoint salvato in:\n{checkpoint_path}\n\n"
            "Il LoRA è stato impostato come attivo nel progetto."
        )
        self._final_group.setVisible(True)
        self._status_label.setText("✓ Training completato!")

    def _on_error(self, message: str) -> None:
        self._set_state(_STATE_ERROR)
        self._status_label.setText(f"Errore: {message}")
        self._on_log_line(f"ERRORE: {message}")
        QMessageBox.critical(self, "Training fallito", message)

    # --- Stato UI --------------------------------------------------------

    def _set_state(self, state: str) -> None:
        self._state = state
        is_idle = state in (_STATE_IDLE, _STATE_DONE, _STATE_ERROR)
        self._start_btn.setEnabled(is_idle and self._project is not None)
        self._stop_btn.setVisible(state == _STATE_RUNNING)
        self._progress_bar.setVisible(state == _STATE_RUNNING)
        self._step_label.setVisible(state == _STATE_RUNNING)
        self._loss_label.setVisible(state in (_STATE_RUNNING, _STATE_DONE))
        if state == _STATE_IDLE:
            self._status_label.setText(
                "Pronto. Configura il preset e clicca Inizia Training."
            )
        elif state == _STATE_RUNNING:
            self._status_label.setText("Training in corso...")
            self._progress_bar.setValue(0)
        if is_idle and self._worker is not None:
            self._disconnect_worker()

    def _disconnect_worker(self) -> None:
        if self._worker is None:
            return
        for sig, slot in (
            (self._worker.progress, self._on_progress),
            (self._worker.epoch_changed, self._on_epoch_signal),
            (self._worker.loss_update, self._on_loss),
            (self._worker.log_line, self._on_log_line),
            (self._worker.sample_ready, self._on_sample),
            (self._worker.dataset_report, self._on_dataset_report),
            (self._worker.finished_ok, self._on_finished),
            (self._worker.error, self._on_error),
        ):
            try:
                sig.disconnect(slot)
            except RuntimeError:
                pass
        self._worker = None

    def shutdown(self) -> None:
        """Chiamato da MainWindow.closeEvent — interrompe il worker se attivo."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(8000)
        self._disconnect_worker()

    # --- Helpers ---------------------------------------------------------

    def _refresh_dataset_info(self) -> None:
        if self._project is None:
            self._dataset_info_label.setText("Nessun progetto selezionato.")
            return
        images_dir = self._project.dataset_images_dir
        if not images_dir.exists():
            n = 0
        else:
            n = sum(1 for f in images_dir.iterdir()
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        self._dataset_info_label.setText(
            f"Progetto: {self._project.name}\n"
            f"Dataset: {n} immagini\n"
            f"Tag attivatore: {self._project.activator_tag}"
        )

    def _check_resume(self) -> None:
        if self._project is None:
            return
        runs = RunStatus.load_all(self._project.training_runs_dir)
        resumable = next((r for r in runs if r.is_resumable), None)
        if resumable:
            self._resume_label.setText(
                f"Training interrotto allo step {resumable.current_step}. "
                "Vuoi riprendere da dove eri rimasto?"
            )
            self._resume_banner.setVisible(True)

    def _update_start_enabled(self) -> None:
        self._start_btn.setEnabled(
            self._state != _STATE_RUNNING
            and self._project is not None
            and self._sdscripts_dir is not None
        )

    def _clear_samples(self) -> None:
        while self._sample_grid.count():
            item = self._sample_grid.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._sample_col = 0
        self._sample_row = 0
