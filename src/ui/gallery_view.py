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
    QComboBox,
    QFrame,
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

from src.core.gallery import (
    RATING_KEEP,
    RATING_REJECT,
    RATING_UNTAGGED,
    RATING_VARIANT,
    GalleryItem,
    load_gallery,
    remove_item,
    set_rating,
)
from src.core.project import Project

logger = logging.getLogger(__name__)

_THUMB_W = 168
_GALLERY_COLS = 3

# Colore del bordo del thumbnail per ciascun giudizio (oltre all'emoji nel testo).
_RATING_BORDER = {
    RATING_KEEP: "#3fae5a",
    RATING_VARIANT: "#d8a23a",
    RATING_REJECT: "#c0504d",
}
_RATING_EMOJI = {
    RATING_KEEP: "👍",
    RATING_VARIANT: "🔀",
    RATING_REJECT: "👎",
}


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #2d3344;")
    return line


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
        self._base_name = item.name if len(item.name) <= 20 else item.name[:18] + "…"
        self.apply_rating_style()

    def apply_rating_style(self) -> None:
        """Riflette il giudizio corrente di ``self.item`` su bordo, testo e tooltip."""
        rating = self.item.rating
        emoji = _RATING_EMOJI.get(rating, "")
        self.setText(f"{emoji} {self._base_name}" if emoji else self._base_name)
        self.setToolTip(self.item.caption())
        color = _RATING_BORDER.get(rating)
        border = f"border: 2px solid {color}; border-radius: 6px;" if color else ""
        self.setStyleSheet(
            f"QToolButton {{ {border} }}"
            "QToolButton:checked { background: #2d3344; border-radius: 6px; }"
        )


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
        self._rating_filter: Optional[str] = None
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
        self.filter_combo = QComboBox()
        self.filter_combo.setToolTip("Filtra per giudizio")
        self.filter_combo.addItem("Tutte", None)
        self.filter_combo.addItem("👍 Coerenti", RATING_KEEP)
        self.filter_combo.addItem("🔀 Varianti", RATING_VARIANT)
        self.filter_combo.addItem("👎 Scartate", RATING_REJECT)
        self.filter_combo.addItem("Da valutare", RATING_UNTAGGED)
        self.filter_combo.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.filter_combo)
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

        v.addWidget(_hline())
        v.addWidget(QLabel("Giudizio"))
        rate_help = QLabel(
            "Insegna allo stile: rinforza ciò che è coerente,\n"
            "segna le varianti, scarta gli errori."
        )
        rate_help.setWordWrap(True)
        rate_help.setStyleSheet("color: #8a8d96; font-size: 10px;")
        v.addWidget(rate_help)

        self.rate_keep = QPushButton("👍 Coerente")
        self.rate_keep.setToolTip("È il personaggio: usala per rinforzare lo stile")
        self.rate_keep.setCheckable(True)
        self.rate_keep.clicked.connect(lambda: self._on_rate(RATING_KEEP))
        self.rate_variant = QPushButton("🔀 Variante")
        self.rate_variant.setToolTip("Piace ma è una variante: candidata a un LoRA derivato")
        self.rate_variant.setCheckable(True)
        self.rate_variant.clicked.connect(lambda: self._on_rate(RATING_VARIANT))
        self.rate_reject = QPushButton("👎 Scarta")
        self.rate_reject.setToolTip("Da dimenticare: riferimento negativo")
        self.rate_reject.setCheckable(True)
        self.rate_reject.clicked.connect(lambda: self._on_rate(RATING_REJECT))
        rate_row = QHBoxLayout()
        rate_row.addWidget(self.rate_keep)
        rate_row.addWidget(self.rate_variant)
        rate_row.addWidget(self.rate_reject)
        v.addLayout(rate_row)

        v.addWidget(_hline())

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

        self._rating_filter = self.filter_combo.currentData()
        self._items = load_gallery(
            self._project.gallery_dir, self.search.text(), self._rating_filter
        )
        if not self._items:
            filtering = bool(self.search.text().strip()) or self._rating_filter is not None
            self._empty_label.setText(
                "Nessun risultato." if filtering else "Nessuna immagine in galleria."
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
        self._sync_rating_buttons()

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.reuse_btn.setEnabled(enabled)
        self.open_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
        self.rate_keep.setEnabled(enabled)
        self.rate_variant.setEnabled(enabled)
        self.rate_reject.setEnabled(enabled)
        if not enabled:
            self._sync_rating_buttons()

    def _sync_rating_buttons(self) -> None:
        """Allinea lo stato premuto dei tre pulsanti al giudizio dell'immagine."""
        r = self._selected.rating if self._selected else None
        self.rate_keep.setChecked(r == RATING_KEEP)
        self.rate_variant.setChecked(r == RATING_VARIANT)
        self.rate_reject.setChecked(r == RATING_REJECT)

    def _on_rate(self, value: str) -> None:
        """Applica/azzera il giudizio: ri-cliccare quello attivo lo rimuove."""
        if self._selected is None:
            return
        new = None if self._selected.rating == value else value
        set_rating(self._selected.path, new)
        if new is None:
            self._selected.metadata.pop("rating", None)
        else:
            self._selected.metadata["rating"] = new

        if self._rating_filter is None:
            # Vista "Tutte": ristila in posto e mantieni la selezione.
            btn = self._group.checkedButton()
            if isinstance(btn, _Thumb):
                btn.apply_rating_style()
            self.detail.setPlainText(self._selected.caption())
            self._sync_rating_buttons()
        else:
            # Con un filtro attivo l'item può uscire dalla vista: ricarica.
            self.refresh()

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
