"""View principale del training guidato (Pilastro D — modalità Training).

Riferimento: docs/GUIDED_TRAINING.md

Macchina a stati:
    IDLE        → form: inserisci prompt, lora, seed → "Inizia"
    GENERATING  → candidati arrivano uno per uno (spinner + preview parziale)
    CHOOSING    → tutti i candidati pronti, bottoni Approva abilitati
    EXHAUSTED   → K rifiuti consecutivi, ExhaustedDialog
    COMPLETED   → sessione finita, immagine finale, "Salva in Gallery"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.guidance.diary import DiaryEntry, append_entry
from src.core.guidance.exhaustion import DEFAULT_THRESHOLD, check_exhaustion
from src.core.guidance.session import (
    Candidate,
    GuidedSession,
    StepResult,
    default_pipeline,
)
from src.core.guidance.technique_library import TechniqueLibrary
from src.core.project import Project

logger = logging.getLogger(__name__)

# Altezza miniatura in px
_THUMB_H = 220


# ---------------------------------------------------------------------------
# Candidato mini-widget
# ---------------------------------------------------------------------------


class _CandidateThumb(QFrame):
    approved = pyqtSignal(int)  # index

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        self._has_image = False

        self.setFrameShape(QFrame.Shape.Box)
        self.setFixedSize(260, _THUMB_H + 50)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setFixedHeight(_THUMB_H)
        self._img_label.setText("⏳")
        self._img_label.setStyleSheet("font-size: 24px; color: #777;")
        layout.addWidget(self._img_label)

        self._approve_btn = QPushButton("✓  Approva")
        self._approve_btn.setEnabled(False)
        self._approve_btn.clicked.connect(lambda: self.approved.emit(self._index))
        layout.addWidget(self._approve_btn)

    def set_image(self, path: Path) -> None:
        pix = QPixmap(str(path))
        if not pix.isNull():
            self._img_label.setPixmap(
                pix.scaled(
                    self._img_label.width(),
                    _THUMB_H,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._has_image = True
        self._approve_btn.setEnabled(True)
        self.setStyleSheet("")

    def mark_approved(self) -> None:
        self.setStyleSheet("QFrame { border: 3px solid #5cb85c; }")
        self._approve_btn.setText("✓  Approvato")
        self._approve_btn.setEnabled(False)

    def mark_rejected(self) -> None:
        self.setStyleSheet("QFrame { border: 2px solid #555; }")
        self._approve_btn.setEnabled(False)

    def reset(self) -> None:
        self._has_image = False
        self._img_label.setPixmap(QPixmap())
        self._img_label.setText("⏳")
        self._img_label.setStyleSheet("font-size: 24px; color: #777;")
        self._approve_btn.setText("✓  Approva")
        self._approve_btn.setEnabled(False)
        self.setStyleSheet("")


# ---------------------------------------------------------------------------
# Dialog ExhaustedDialog
# ---------------------------------------------------------------------------


class _ExhaustedDialog(QDialog):
    """Mostrato quando K rifiuti consecutivi → 'soluzioni insufficienti'."""

    TECHNIQUE = 0
    DATASET   = 1
    ACCEPT    = 2
    ABORT     = 3

    def __init__(self, step_id: str, total_rejections: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Soluzioni insufficienti")
        self.setMinimumWidth(460)
        self._choice: int = self.ABORT

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        icon = QLabel("⚠")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 36px;")
        layout.addWidget(icon)

        msg = QLabel(
            f"Ho generato candidati per «{step_id}» "
            f"{total_rejections} volte di seguito\nsenza che nessuno ti convincesse.\n\n"
            "Questo può indicare che:\n"
            "→ Il dataset manca di esempi di questa tecnica\n"
            "→ La Libreria Tecniche non ha riferimenti per questo step\n"
            "→ Il LoRA attuale non ha abbastanza dati su questo aspetto"
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        layout.addWidget(_hline())

        label = QLabel("Cosa vuoi fare?")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)

        btn_technique = QPushButton("🎨  Aggiungi esempi alla Libreria Tecniche")
        btn_dataset   = QPushButton("📁  Aggiungi immagini al Dataset e re-train")
        btn_accept    = QPushButton("✓  Accetta il meno peggio tra questi candidati")
        btn_abort     = QPushButton("✕  Annulla questa sessione")

        btn_technique.clicked.connect(lambda: self._pick(self.TECHNIQUE))
        btn_dataset.clicked.connect(lambda: self._pick(self.DATASET))
        btn_accept.clicked.connect(lambda: self._pick(self.ACCEPT))
        btn_abort.clicked.connect(lambda: self._pick(self.ABORT))

        for btn in (btn_technique, btn_dataset, btn_accept, btn_abort):
            btn.setMinimumHeight(36)
            layout.addWidget(btn)

    def _pick(self, choice: int) -> None:
        self._choice = choice
        self.accept()

    def choice(self) -> int:
        return self._choice


# ---------------------------------------------------------------------------
# Pannello avvio sessione
# ---------------------------------------------------------------------------


class _StartPane(QWidget):
    started = pyqtSignal(str, int)  # (prompt, seed)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        layout.addStretch()

        title = QLabel("Training Guidato")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        intro = QLabel(
            "Il sistema genererà 4 candidati per ogni fase del disegno.\n"
            "Tu approvi quello che preferisci — il sistema impara dalla tua scelta."
        )
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addSpacing(12)

        self._project_label = QLabel("Nessun progetto attivo")
        self._project_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._project_label.setStyleSheet("color: #888;")
        layout.addWidget(self._project_label)

        layout.addWidget(_hline())

        layout.addWidget(QLabel("Prompt:"))
        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setFixedHeight(72)
        self._prompt_edit.setPlaceholderText("es. iris, full body, standing, white background")
        layout.addWidget(self._prompt_edit)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed (−1 = casuale):"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 2**31 - 1)
        self._seed_spin.setValue(-1)
        seed_row.addWidget(self._seed_spin)
        layout.addLayout(seed_row)

        layout.addSpacing(12)

        self._start_btn = QPushButton("▶  Inizia sessione")
        self._start_btn.setMinimumHeight(40)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        layout.addWidget(self._start_btn)

        layout.addStretch()

    def set_project(self, project: Optional[Project]) -> None:
        if project is None:
            self._project_label.setText("Nessun progetto attivo")
            self._start_btn.setEnabled(False)
        else:
            lora_info = (
                project.active_lora.checkpoint if project.active_lora else "nessun LoRA"
            )
            self._project_label.setText(
                f"Progetto: {project.name}  |  LoRA: {lora_info}"
            )
            self._start_btn.setEnabled(True)

    def _on_start(self) -> None:
        prompt = self._prompt_edit.toPlainText().strip()
        if not prompt:
            return
        self.started.emit(prompt, self._seed_spin.value())


# ---------------------------------------------------------------------------
# Pannello selezione candidati (durante e dopo generazione)
# ---------------------------------------------------------------------------


class _StepPane(QWidget):
    approved = pyqtSignal(int)         # candidate index approvato
    regenerate = pyqtSignal(str)       # (rejection_reason) — rigenera tutti
    abort_session = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # Header step
        self._step_label = QLabel("Step 1/4 — Composizione")
        self._step_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._step_label)

        self._status_label = QLabel("Generazione in corso…")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #888;")
        layout.addWidget(self._status_label)

        layout.addWidget(_hline())

        # Griglia 2×2 candidati
        grid_widget = QWidget()
        self._grid = QGridLayout(grid_widget)
        self._grid.setSpacing(12)
        self._thumbs: list[_CandidateThumb] = []
        for i in range(4):
            thumb = _CandidateThumb(i)
            thumb.approved.connect(self._on_approved)
            self._thumbs.append(thumb)
            self._grid.addWidget(thumb, i // 2, i % 2)
        layout.addWidget(grid_widget, stretch=1)

        layout.addWidget(_hline())

        # Barra azioni
        actions = QHBoxLayout()

        self._reason_edit = QLineEdit()
        self._reason_edit.setPlaceholderText("Motivo del rifiuto (opzionale)…")
        actions.addWidget(self._reason_edit, stretch=2)

        self._regen_btn = QPushButton("↺  Rigenera tutti")
        self._regen_btn.setEnabled(False)
        self._regen_btn.clicked.connect(self._on_regen)
        actions.addWidget(self._regen_btn)

        self._abort_btn = QPushButton("✕  Annulla")
        self._abort_btn.clicked.connect(self.abort_session)
        actions.addWidget(self._abort_btn)

        layout.addLayout(actions)

    def set_step_label(self, current: int, total: int, label: str) -> None:
        self._step_label.setText(f"Step {current}/{total} — {label}")

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_actions_enabled(self, enabled: bool) -> None:
        self._regen_btn.setEnabled(enabled)

    def reset_thumbs(self) -> None:
        for t in self._thumbs:
            t.reset()
        self._regen_btn.setEnabled(False)
        self._status_label.setText("Generazione in corso…")
        self._reason_edit.clear()

    def show_candidate(self, index: int, path: Path) -> None:
        if 0 <= index < len(self._thumbs):
            self._thumbs[index].set_image(path)

    def mark_approved(self, index: int) -> None:
        for i, t in enumerate(self._thumbs):
            if i == index:
                t.mark_approved()
            else:
                t.mark_rejected()

    def _on_approved(self, index: int) -> None:
        self.mark_approved(index)
        self.set_status("Approvato — passo allo step successivo…")
        self.approved.emit(index)

    def _on_regen(self) -> None:
        reason = self._reason_edit.text().strip()
        self.reset_thumbs()
        self.regenerate.emit(reason)


# ---------------------------------------------------------------------------
# Pannello sessione completata
# ---------------------------------------------------------------------------


class _DonePane(QWidget):
    save_to_gallery = pyqtSignal(Path)
    new_session = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        layout.addStretch()

        self._title = QLabel("Sessione completata!")
        self._title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setFixedHeight(400)
        layout.addWidget(self._img_label)

        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #888;")
        layout.addWidget(self._info_label)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("📁  Salva in Gallery")
        save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        new_btn = QPushButton("↺  Nuova sessione")
        new_btn.setMinimumHeight(36)
        new_btn.clicked.connect(self.new_session)
        btn_row.addWidget(new_btn)

        layout.addLayout(btn_row)
        layout.addStretch()

        self._final_path: Optional[Path] = None

    def set_result(self, image_path: Optional[Path], steps_done: int) -> None:
        self._final_path = image_path
        if image_path and image_path.exists():
            pix = QPixmap(str(image_path))
            if not pix.isNull():
                self._img_label.setPixmap(
                    pix.scaled(
                        self._img_label.width() or 400,
                        400,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        self._info_label.setText(f"Completata in {steps_done} step.")

    def _on_save(self) -> None:
        if self._final_path:
            self.save_to_gallery.emit(self._final_path)


# ---------------------------------------------------------------------------
# Vista principale
# ---------------------------------------------------------------------------


class GuidedTrainingView(QWidget):
    """Vista training guidato. Wiring con main_window via set_project()."""

    session_saved_to_gallery = pyqtSignal(Path)
    switch_to_dataset = pyqtSignal()
    switch_to_technique_library = pyqtSignal()

    _PANE_START = 0
    _PANE_STEP  = 1
    _PANE_DONE  = 2

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._library: Optional[TechniqueLibrary] = None
        self._session: Optional[GuidedSession] = None
        self._worker = None
        self._pending_candidates: list[Candidate] = []
        self._n_candidates = 4  # da step_def, aggiornato ad ogni step
        self._step_attempt = 0  # quante volte lo step corrente è stato (ri)generato

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._start_pane = _StartPane()
        self._step_pane  = _StepPane()
        self._done_pane  = _DonePane()

        self._stack.addWidget(self._start_pane)
        self._stack.addWidget(self._step_pane)
        self._stack.addWidget(self._done_pane)

        layout.addWidget(self._stack)

        # Connessioni
        self._start_pane.started.connect(self._on_session_start)
        self._step_pane.approved.connect(self._on_approved)
        self._step_pane.regenerate.connect(self._on_regenerate)
        self._step_pane.abort_session.connect(self._on_abort)
        self._done_pane.save_to_gallery.connect(self.session_saved_to_gallery)
        self._done_pane.new_session.connect(self._reset_to_start)

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self._start_pane.set_project(project)
        if project:
            self._library = TechniqueLibrary.load(project.technique_library_path)
        else:
            self._library = None

    def set_comfy_client(self, client) -> None:
        self._comfy_client = client

    # --- Avvio sessione --------------------------------------------------

    def _on_session_start(self, prompt: str, seed: int) -> None:
        if self._project is None:
            return
        if not hasattr(self, "_comfy_client") or self._comfy_client is None:
            QMessageBox.warning(
                self, "ComfyUI non pronto",
                "ComfyUI non è ancora disponibile — aspetta il completamento dell'avvio."
            )
            return

        from src.utils.seed import resolve_seed
        actual_seed = resolve_seed(seed)

        self._session = GuidedSession(
            project_slug=self._project.slug,
            prompt=prompt,
            negative_prompt=self._project.default_negative_prompt,
            seed=actual_seed,
            lora_checkpoint=(
                self._project.active_lora.checkpoint
                if self._project.active_lora else ""
            ),
            pipeline=default_pipeline(),
        )
        # Salva sessione su disco subito
        session_dir = self._project.guided_session_dir(self._session.session_id)
        self._session.save(session_dir / "session.json")

        self._stack.setCurrentIndex(self._PANE_STEP)
        self._launch_step()

    # --- Step pipeline ---------------------------------------------------

    def _launch_step(self, *, new_step: bool = True) -> None:
        """Avvia il worker per lo step corrente.

        new_step=True azzera il contatore tentativi (si entra in uno step
        nuovo); False lo incrementa (rigenerazione dello stesso step)."""
        if self._session is None or self._project is None:
            return
        step_def = self._session.current_step_def
        if step_def is None:
            return

        if new_step:
            self._step_attempt = 0
        else:
            self._step_attempt += 1

        total = len(self._session.pipeline)
        current = self._session.current_step_index + 1
        self._step_pane.set_step_label(current, total, step_def.label)
        self._step_pane.reset_thumbs()
        self._pending_candidates = []
        self._n_candidates = step_def.n_candidates

        session_dir = self._project.guided_session_dir(self._session.session_id)
        step_name = f"step_{self._session.current_step_index:02d}"
        if self._step_attempt > 0:
            step_name += f"_r{self._step_attempt}"
        step_dir = session_dir / step_name

        from src.workers.guided_worker import GuidedWorker
        self._worker = GuidedWorker(
            session=self._session,
            project=self._project,
            client=self._comfy_client,
            step_dir=step_dir,
            technique_library=self._library,
            attempt=self._step_attempt,
        )
        self._worker.candidate_ready.connect(self._on_candidate_ready)
        self._worker.step_complete.connect(self._on_step_complete)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_candidate_ready(self, index: int, path: Path) -> None:
        self._step_pane.show_candidate(index, path)
        self._step_pane.set_status(
            f"Generazione {index + 1}/{self._n_candidates}…"
        )

    def _on_step_complete(self, candidates: list[Candidate]) -> None:
        self._pending_candidates = candidates
        self._step_pane.set_actions_enabled(True)
        self._step_pane.set_status("Scegli un candidato da approvare.")

    def _on_approved(self, index: int) -> None:
        if not self._pending_candidates or self._session is None or self._project is None:
            return

        chosen = next((c for c in self._pending_candidates if c.index == index), None)
        rejected = [c for c in self._pending_candidates if c.index != index]
        step_def = self._session.current_step_def

        # Scrivi nel Diary
        entry = DiaryEntry(
            session_id=self._session.session_id,
            project_slug=self._project.slug,
            step_id=step_def.id,
            prompt=self._session.prompt,
            negative_prompt=self._session.negative_prompt,
            seed=self._session.seed,
            chosen_path=chosen.image_path if chosen else None,
            rejected_paths=[c.image_path for c in rejected],
            technique_refs_used=self._active_ref_ids(step_def),
            lora_checkpoint=self._session.lora_checkpoint,
        )
        append_entry(self._project.diary_path, entry)

        # Aggiorna sessione
        result = StepResult(
            step_def=step_def,
            candidates=self._pending_candidates,
            approved_index=index,
        )
        self._session.record_result(result)

        # Salva sessione aggiornata
        session_dir = self._project.guided_session_dir(self._session.session_id)
        self._session.save(session_dir / "session.json")

        if self._session.is_finished:
            self._stack.setCurrentIndex(self._PANE_DONE)
            self._done_pane.set_result(
                self._session.final_image_path,
                len(self._session.results),
            )
        else:
            self._launch_step()

    def _on_regenerate(self, reason: str) -> None:
        if self._session is None or self._project is None:
            return

        step_def = self._session.current_step_def

        # Diary: tutti rifiutati
        entry = DiaryEntry(
            session_id=self._session.session_id,
            project_slug=self._project.slug,
            step_id=step_def.id,
            prompt=self._session.prompt,
            negative_prompt=self._session.negative_prompt,
            seed=self._session.seed,
            chosen_path=None,
            rejected_paths=[c.image_path for c in self._pending_candidates],
            rejection_reason=reason or None,
            technique_refs_used=self._active_ref_ids(step_def),
            lora_checkpoint=self._session.lora_checkpoint,
            was_regeneration=True,
        )
        append_entry(self._project.diary_path, entry)

        # Verifica esaurimento
        from src.core.guidance.diary import load_diary
        diary = load_diary(self._project.diary_path)
        report = check_exhaustion(diary, self._session.session_id, step_def.id, DEFAULT_THRESHOLD)

        if report.exhausted:
            self._show_exhausted_dialog(report)
            return

        # Riparte lo stesso step con seed diversi (tentativo successivo)
        self._launch_step(new_step=False)

    def _show_exhausted_dialog(self, report) -> None:
        dlg = _ExhaustedDialog(
            step_id=report.step_id,
            total_rejections=report.consecutive_rejections,
            parent=self,
        )
        dlg.exec()
        choice = dlg.choice()

        if choice == _ExhaustedDialog.TECHNIQUE:
            self.switch_to_technique_library.emit()
        elif choice == _ExhaustedDialog.DATASET:
            self.switch_to_dataset.emit()
        elif choice == _ExhaustedDialog.ACCEPT:
            self._force_accept_best()
        else:
            self._on_abort()

    def _force_accept_best(self) -> None:
        """Mostra un mini-dialog per scegliere il meno peggio tra i candidati."""
        if not self._pending_candidates:
            return

        from PyQt6.QtWidgets import QInputDialog
        items = [f"Candidato {c.index}" for c in self._pending_candidates]
        choice, ok = QInputDialog.getItem(
            self, "Accetta il meno peggio", "Quale candidato accetti?",
            items, editable=False,
        )
        if ok and choice:
            idx = int(choice.split()[-1])
            self._on_approved(idx)

    def _on_abort(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.abort()
        if self._session is not None and self._project is not None:
            from src.core.guidance.session import STATUS_ABORTED
            self._session.status = STATUS_ABORTED
            session_dir = self._project.guided_session_dir(self._session.session_id)
            self._session.save(session_dir / "session.json")
        self._reset_to_start()

    def _on_worker_error(self, msg: str) -> None:
        self._step_pane.set_status(f"Errore: {msg}")
        logger.error("GuidedWorker error: %s", msg)

    # --- Helper ----------------------------------------------------------

    def _active_ref_ids(self, step_def) -> list[str]:
        if self._library is None:
            return []
        return [r.ref_id for r in self._library.active_for_step(step_def)]

    def _reset_to_start(self) -> None:
        self._session = None
        self._pending_candidates = []
        self._worker = None
        self._stack.setCurrentIndex(self._PANE_START)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #444;")
    return line
