"""Pannello Libreria Tecniche — gestione riferimenti IP-Adapter (Pilastro B).

Accessibile da:
- Sidebar del progetto (tab "Tecniche")
- Pulsante "🎨 Aggiungi esempi alla Libreria Tecniche" in ExhaustedDialog
- Pulsante "→ Libreria" nel detail panel della GalleryView

Permette di:
- Sfogliare i riferimenti per slot (skin, shadow, hair…)
- Aggiungere nuovi riferimenti (via file dialog)
- Rimuovere riferimenti
- Cambiare peso IP-Adapter del singolo riferimento

Riferimento: docs/GUIDED_TRAINING.md (Pilastro B)
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.guidance.technique_library import DEFAULT_SLOTS, TechniqueLibrary, TechniqueRef
from src.core.project import Project

logger = logging.getLogger(__name__)

_THUMB_SIZE = 100


# ---------------------------------------------------------------------------
# Miniatura riferimento
# ---------------------------------------------------------------------------


class _RefThumb(QFrame):
    selected   = pyqtSignal(object)  # TechniqueRef
    removed    = pyqtSignal(object)  # TechniqueRef

    def __init__(self, ref: TechniqueRef, parent=None) -> None:
        super().__init__(parent)
        self._ref = ref
        self.setFrameShape(QFrame.Shape.Box)
        self.setFixedSize(_THUMB_SIZE + 8, _THUMB_SIZE + 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        img = QLabel()
        img.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if ref.image_path.exists():
            pix = QPixmap(str(ref.image_path)).scaled(
                _THUMB_SIZE, _THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            img.setPixmap(pix)
        else:
            img.setText("?")
        layout.addWidget(img)

        lbl = QLabel(ref.label or ref.slot)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

    def mousePressEvent(self, event):
        self.selected.emit(self._ref)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(
            "QFrame { border: 2px solid #5cb85c; }" if selected else ""
        )


# ---------------------------------------------------------------------------
# Pannello dettaglio riferimento selezionato
# ---------------------------------------------------------------------------


class _RefDetail(QWidget):
    saved   = pyqtSignal()
    removed = pyqtSignal(object)  # TechniqueRef

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ref: Optional[TechniqueRef] = None
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._preview = QLabel()
        self._preview.setFixedSize(220, 220)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet("border: 1px solid #444;")
        layout.addWidget(self._preview)

        layout.addWidget(QLabel("Label:"))
        self._label_edit = QLineEdit()
        layout.addWidget(self._label_edit)

        layout.addWidget(QLabel("Note:"))
        self._notes_edit = QLineEdit()
        layout.addWidget(self._notes_edit)

        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("Peso IP-Adapter:"))
        self._weight_spin = QDoubleSpinBox()
        self._weight_spin.setRange(0.0, 2.0)
        self._weight_spin.setSingleStep(0.05)
        self._weight_spin.setValue(1.0)
        weight_row.addWidget(self._weight_spin)
        layout.addLayout(weight_row)

        save_btn = QPushButton("💾  Salva")
        save_btn.clicked.connect(self._on_save)
        layout.addWidget(save_btn)

        remove_btn = QPushButton("🗑  Rimuovi")
        remove_btn.setStyleSheet("color: #d96a6a;")
        remove_btn.clicked.connect(self._on_remove)
        layout.addWidget(remove_btn)

        layout.addStretch()
        self._set_enabled(False)

    def load_ref(self, ref: TechniqueRef) -> None:
        self._ref = ref
        if ref.image_path.exists():
            pix = QPixmap(str(ref.image_path)).scaled(
                220, 220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview.setPixmap(pix)
        else:
            self._preview.setText("Immagine non trovata")
        self._label_edit.setText(ref.label)
        self._notes_edit.setText(ref.notes)
        self._weight_spin.setValue(ref.weight)
        self._set_enabled(True)

    def clear(self) -> None:
        self._ref = None
        self._preview.clear()
        self._preview.setText("")
        self._label_edit.clear()
        self._notes_edit.clear()
        self._weight_spin.setValue(1.0)
        self._set_enabled(False)

    def _set_enabled(self, enabled: bool) -> None:
        for w in (self._label_edit, self._notes_edit, self._weight_spin):
            w.setEnabled(enabled)

    def _on_save(self) -> None:
        if self._ref is None:
            return
        self._ref.label = self._label_edit.text().strip()
        self._ref.notes = self._notes_edit.text().strip()
        self._ref.weight = self._weight_spin.value()
        self.saved.emit()

    def _on_remove(self) -> None:
        if self._ref is not None:
            self.removed.emit(self._ref)


# ---------------------------------------------------------------------------
# Vista griglia (per slot selezionato)
# ---------------------------------------------------------------------------


class _SlotGrid(QWidget):
    ref_selected = pyqtSignal(object)  # TechniqueRef

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thumbs: list[_RefThumb] = []
        self._selected_id: Optional[str] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        self._container = QWidget()
        self._flow = QGridLayout(self._container)
        self._flow.setSpacing(8)
        scroll.setWidget(self._container)

    def load_refs(self, refs: list[TechniqueRef]) -> None:
        # Rimuovi tutti i widget
        for t in self._thumbs:
            t.setParent(None)
        self._thumbs.clear()
        self._selected_id = None

        for i, ref in enumerate(refs):
            thumb = _RefThumb(ref)
            thumb.selected.connect(self._on_ref_selected)
            self._thumbs.append(thumb)
            self._flow.addWidget(thumb, i // 3, i % 3)

        # Riempi spazio vuoto
        if not refs:
            lbl = QLabel("Nessun riferimento per questo slot.\nUsa «+ Aggiungi» per cominciare.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888;")
            self._flow.addWidget(lbl, 0, 0, 1, 3)

    def _on_ref_selected(self, ref: TechniqueRef) -> None:
        self._selected_id = ref.ref_id
        for t in self._thumbs:
            t.set_selected(t._ref.ref_id == ref.ref_id)
        self.ref_selected.emit(ref)


# ---------------------------------------------------------------------------
# Vista principale Libreria Tecniche
# ---------------------------------------------------------------------------


class TechniqueLibraryView(QWidget):
    """Standalone view per gestire i riferimenti tecnici di un progetto."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._library: Optional[TechniqueLibrary] = None
        self._current_slot: str = list(DEFAULT_SLOTS.keys())[0]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Pannello sinistro: slot selector + griglia -----------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Libreria Tecniche")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(title)

        slot_row = QHBoxLayout()
        slot_row.addWidget(QLabel("Slot:"))
        self._slot_combo = QComboBox()
        for key, label in DEFAULT_SLOTS.items():
            self._slot_combo.addItem(f"{key} — {label}", key)
        self._slot_combo.currentIndexChanged.connect(self._on_slot_changed)
        slot_row.addWidget(self._slot_combo, stretch=1)
        left_layout.addLayout(slot_row)

        self._grid = _SlotGrid()
        self._grid.ref_selected.connect(self._on_ref_selected)
        left_layout.addWidget(self._grid, stretch=1)

        add_btn = QPushButton("+ Aggiungi riferimento")
        add_btn.clicked.connect(self._on_add)
        left_layout.addWidget(add_btn)

        layout.addWidget(left, stretch=2)

        # --- Separatore -------------------------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)

        # --- Pannello destro: dettaglio ref --------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.addWidget(QLabel("Dettaglio"))
        self._detail = _RefDetail()
        self._detail.saved.connect(self._on_saved)
        self._detail.removed.connect(self._on_removed)
        right_layout.addWidget(self._detail, stretch=1)
        layout.addWidget(right, stretch=1)

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        if project:
            self._library = TechniqueLibrary.load(project.technique_library_path)
        else:
            self._library = None
        self._reload_grid()

    def _reload_grid(self) -> None:
        if self._library is None:
            self._grid.load_refs([])
            return
        self._grid.load_refs(self._library.by_slot(self._current_slot))
        self._detail.clear()

    def _on_slot_changed(self, _: int) -> None:
        self._current_slot = self._slot_combo.currentData()
        self._reload_grid()

    def _on_ref_selected(self, ref: TechniqueRef) -> None:
        self._detail.load_ref(ref)

    def _on_add(self) -> None:
        if self._project is None or self._library is None:
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleziona immagine riferimento",
            str(Path.home()),
            "Immagini (*.png *.jpg *.jpeg *.webp)",
        )
        if not paths:
            return

        for src_str in paths:
            src = Path(src_str)
            dest_dir = self._project.techniques_refs_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            ref_id = uuid.uuid4().hex
            dest = dest_dir / f"{ref_id}{src.suffix}"
            shutil.copy2(src, dest)

            ref = TechniqueRef(
                slot=self._current_slot,
                image_path=dest,
                label=src.stem,
            )
            ref.ref_id = ref_id
            self._library.add(ref)

        self._library.save(self._project.technique_library_path)
        self._reload_grid()

    def _on_saved(self) -> None:
        if self._project and self._library:
            self._library.save(self._project.technique_library_path)
            self._reload_grid()

    def _on_removed(self, ref: TechniqueRef) -> None:
        if self._project is None or self._library is None:
            return
        self._library.remove(ref.ref_id)
        # Cancella il file immagine della refs dir
        if ref.image_path.is_relative_to(self._project.techniques_refs_dir):
            try:
                ref.image_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Impossibile rimuovere ref image: %s", exc)
        self._library.save(self._project.technique_library_path)
        self._reload_grid()


# ---------------------------------------------------------------------------
# Dialog standalone (per aprirla da GuidedTrainingView)
# ---------------------------------------------------------------------------


class TechniqueLibraryDialog(QDialog):
    """Versione modale della TechniqueLibraryView."""

    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Libreria Tecniche")
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        view = TechniqueLibraryView()
        view.set_project(project)
        layout.addWidget(view)
        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
