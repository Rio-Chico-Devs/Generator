"""Galleria dei risultati — rivedi, filtra e riusa le immagini generate.

Layout:
    +-------------------------------------+--------------------+
    |  [cerca…]            [Aggiorna]      |  Dettagli          |
    |  +-------------------------------+   |  (metadati)        |
    |  | griglia thumbnail (scroll)    |   |                    |
    |  | selezionabili                 |   |  [Usa parametri]   |
    |  |                               |   |  [Apri cartella]   |
    |  +-------------------------------+   |  [Elimina]         |
    +-------------------------------------+--------------------+

La scansione/parsing dei metadati vive in ``src.core.gallery`` ed è testata
senza Qt. Qui c'è solo il wiring dei widget.
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.gallery import GalleryItem, load_gallery, remove_item
from src.core.project import Project

logger = logging.getLogger(__name__)

_THUMB_W = 168
_GALLERY_COLS = 3


class _Thumb(QToolButton):
    """Anteprima cliccabile e selezionabile di un'immagine della galleria."""

    def __init__(self, item: GalleryItem, index: int, parent=None) -> None:
        super().__init__(parent)
        self.item = item
        self.index = index
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setIconSize(QSize(_THUMB_W, _THUMB_W))
        self.setFixedSize(_THUMB_W + 16, _THUMB_W + 34)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setToolTip(item.caption())

        pix = QPixmap(str(item.path))
        if not pix.isNull():
            self.setIcon(
                QIcon(
                    pix.scaled(
                        _THUMB_W, _THUMB_W,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            )
        name = item.name if len(item.name) <= 20 else item.name[:18] + "…"
        self.setText(name)


class GalleryView(QWidget):
    """Vista galleria. Dipendenza iniettata via set_project()."""

    # Emesso con i parametri riusabili dell'immagine selezionata, così
    # MainWindow può pre-compilare la GenerateView.
    reuse_requested = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._items: list[GalleryItem] = []
        self._selected: Optional[GalleryItem] = None
        self._build_ui()

    # --- costruzione UI -------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Cerca per prompt o nome file…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.refresh)
        bar.addWidget(self.search, 1)
        refresh_btn = QPushButton("Aggiorna")
        refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(refresh_btn)
        left.addLayout(bar)

        self._empty_label = QLabel("Nessuna immagine in galleria.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #8a8d96;")
        left.addWidget(self._empty_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._inner = QWidget()
        self._grid = QGridLayout(self._inner)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._inner)
        left.addWidget(self._scroll, 1)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        root.addLayout(left, 1)
        root.addWidget(self._build_detail_panel())

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        v = QVBoxLayout(panel)

        v.addWidget(QLabel("Dettagli"))
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(self.detail, 1)

        self.reuse_btn = QPushButton("Usa questi parametri")
        self.reuse_btn.setToolTip(
            "Pre-compila la schermata Genera con prompt, seed e impostazioni"
        )
        self.reuse_btn.clicked.connect(self._on_reuse)
        v.addWidget(self.reuse_btn)

        self.open_btn = QPushButton("Apri cartella")
        self.open_btn.clicked.connect(self._on_open_folder)
        v.addWidget(self.open_btn)

        self.delete_btn = QPushButton("Elimina")
        self.delete_btn.clicked.connect(self._on_delete)
        v.addWidget(self.delete_btn)

        self._set_actions_enabled(False)
        return panel

    # --- dipendenze / refresh ------------------------------------------

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self.refresh()

    def refresh(self) -> None:
        """Ricarica la griglia dalla gallery_dir del progetto attivo."""
        self._clear_grid()
        self._selected = None
        self.detail.setPlainText("")
        self._set_actions_enabled(False)

        if self._project is None:
            self._items = []
            self._empty_label.setText("Apri un progetto per vedere la galleria.")
            self._empty_label.setVisible(True)
            return

        self._items = load_gallery(self._project.gallery_dir, self.search.text())
        if not self._items:
            self._empty_label.setText(
                "Nessun risultato."
                if self.search.text().strip()
                else "Nessuna immagine in galleria."
            )
            self._empty_label.setVisible(True)
            return

        self._empty_label.setVisible(False)
        for i, item in enumerate(self._items):
            thumb = _Thumb(item, i)
            thumb.toggled.connect(
                lambda checked, it=item: self._on_thumb_toggled(checked, it)
            )
            self._group.addButton(thumb)
            r, c = divmod(i, _GALLERY_COLS)
            self._grid.addWidget(thumb, r, c)

    def _clear_grid(self) -> None:
        for btn in list(self._group.buttons()):
            self._group.removeButton(btn)
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    # --- selezione / azioni --------------------------------------------

    def _on_thumb_toggled(self, checked: bool, item: GalleryItem) -> None:
        if not checked:
            return
        self._selected = item
        self.detail.setPlainText(item.caption())
        self._set_actions_enabled(True)
        self.reuse_btn.setEnabled(item.has_metadata)

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.reuse_btn.setEnabled(enabled)
        self.open_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)

    def _on_reuse(self) -> None:
        if self._selected is None:
            return
        params = self._selected.reuse_params()
        if not params:
            QMessageBox.information(
                self, "Nessun parametro",
                "Questa immagine non ha metadati riusabili.",
            )
            return
        self.reuse_requested.emit(params)

    def _on_open_folder(self) -> None:
        if self._selected is None:
            return
        folder = self._selected.path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_delete(self) -> None:
        if self._selected is None:
            return
        choice = QMessageBox.question(
            self, "Elimina immagine",
            f"Eliminare '{self._selected.name}' e i suoi metadati?\n\n"
            "L'operazione non è reversibile.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        remove_item(self._selected)
        self.refresh()
