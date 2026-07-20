"""Dataset Inspector — vedi e modifica i dati con cui il modello si addestra.

Il principio: il modello "legge" le caption durante il training. Se vedi un tag
sbagliato o un'immagine che non dovrebbe esserci, lo correggi qui e il prossimo
addestramento produrrà risultati più precisi.

Layout:
    +-tabs:[Dataset | Negativi]------------------------------------------+
    |  [🔍 filtra tag]   N immagini · N tag unici · media M tag/img       |
    |  +-------[tabella tag freq]-------+  +--[dettaglio immagine]------+ |
    |  | tag              N   ████░░░  |  |                             | |
    |  | masterpiece     47   ████████ |  |   [preview immagine]        | |
    |  | 1girl           45   ███████░ |  |                             | |
    |  | red hair        12   ██░░░░░░ |  +-----------------------------+ |
    |  | blurry           3   █░░░░░░░ |  | Caption (testo kohya):      | |
    |  +----------------------------------+  [editor multi-riga]       | |
    |  +-------[griglia thumbnail]------+  | [Salva]   [Rimuovi]       | |
    |  | img img img img img img img   |  +-----------------------------+ |
    |  | img img img img               |                                   |
    |  +---------------------------------+                                 |
    +-----------------------------------------------------------------------+

Clicca un tag nella tabella per filtrare le thumbnail a quelle che lo contengono.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.dataset_inspector import (
    IMAGE_EXTS,
    DatasetItem,
    dataset_stats,
    remove_dataset_item,
    save_caption,
    scan_dataset,
    tag_frequency,
)
from src.core.project import Project

logger = logging.getLogger(__name__)

_THUMB_W = 88
_GRID_COLS = 4
_MAX_TAG_ROWS = 30  # tag da mostrare nella tabella


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #2d3344;")
    return line


class _Thumb(QToolButton):
    """Thumbnail selezionabile di un item del dataset."""

    def __init__(self, item: DatasetItem, index: int, parent=None) -> None:
        super().__init__(parent)
        self.item = item
        self.index = index
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setIconSize(QSize(_THUMB_W, _THUMB_W))
        self.setFixedSize(_THUMB_W + 8, _THUMB_W + 22)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        pix = QPixmap(str(item.image_path))
        if not pix.isNull():
            from PyQt6.QtGui import QIcon
            self.setIcon(
                QIcon(
                    pix.scaled(
                        _THUMB_W, _THUMB_W,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            )
        short = item.name if len(item.name) <= 14 else item.name[:12] + "…"
        self.setText(short)
        self.setToolTip(item.caption or "(nessuna caption)")
        # Bordo ambra se manca la caption
        if not item.has_caption:
            self.setStyleSheet(
                "QToolButton { border: 2px solid #d8a23a; border-radius: 4px; }"
                "QToolButton:checked { background: #2d3344; }"
            )
        else:
            self.setStyleSheet(
                "QToolButton:checked { background: #2d3344; border-radius: 4px; }"
            )


# ---------------------------------------------------------------------------
# Pannello statistiche + tabella tag
# ---------------------------------------------------------------------------


class _StatsPanel(QWidget):
    """Header con stats + tabella tag frequency. Cliccando un tag si filtra la griglia."""

    tag_filter_changed = pyqtSignal(str)  # "" = nessun filtro

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._active_tag: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        # Riga stats
        self._stats_label = QLabel("Nessun dato")
        self._stats_label.setStyleSheet("color: #8a8d96; font-size: 11px;")
        v.addWidget(self._stats_label)

        # Ricerca tag
        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Filtra per tag…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self._search, 1)
        clear_btn = QPushButton("✕ Togli filtro")
        clear_btn.setFixedWidth(100)
        clear_btn.clicked.connect(self._clear_filter)
        search_row.addWidget(clear_btn)
        v.addLayout(search_row)

        # Tabella tag
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Tag", "N immagini", "Frequenza"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setFixedHeight(200)
        self._table.cellClicked.connect(self._on_cell_clicked)
        v.addWidget(self._table)

        self._filter_label = QLabel("")
        self._filter_label.setStyleSheet("color: #d8a23a; font-size: 10px;")
        v.addWidget(self._filter_label)

    def load(self, items: list[DatasetItem]) -> None:
        stats = dataset_stats(items)
        n = stats["total_images"]
        no_cap = stats["images_without_caption"]
        warn = f"  ⚠ {no_cap} senza caption" if no_cap else ""
        self._stats_label.setText(
            f"{n} immagini · {stats['unique_tags']} tag unici · "
            f"media {stats['avg_tags_per_image']} tag/img{warn}"
        )
        freq = tag_frequency(items)
        top = freq.most_common(_MAX_TAG_ROWS)
        max_count = top[0][1] if top else 1

        self._table.setRowCount(len(top))
        for row, (tag, count) in enumerate(top):
            self._table.setItem(row, 0, QTableWidgetItem(tag))
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 1, count_item)
            # Barra proporzionale come label colorata
            bar_frac = count / max_count
            bar_w = int(bar_frac * 120)
            bar_item = QTableWidgetItem(f"{'█' * (bar_w // 8)}{'░' * (15 - bar_w // 8)}")
            bar_item.setForeground(QColor("#3fae5a") if bar_frac > 0.7
                                   else QColor("#d8a23a") if bar_frac > 0.2
                                   else QColor("#c0504d"))
            self._table.setItem(row, 2, bar_item)
        self._table.resizeColumnToContents(0)
        self._table.resizeColumnToContents(1)

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        tag_item = self._table.item(row, 0)
        if tag_item is None:
            return
        tag = tag_item.text()
        if self._active_tag == tag:
            self._clear_filter()
        else:
            self._active_tag = tag
            self._filter_label.setText(f"Filtro attivo: «{tag}» — clicca di nuovo per rimuovere")
            self.tag_filter_changed.emit(tag)

    def _on_search_changed(self, text: str) -> None:
        self._active_tag = text.strip()
        if self._active_tag:
            self._filter_label.setText(f"Filtro testuale: «{self._active_tag}»")
        else:
            self._filter_label.setText("")
        self.tag_filter_changed.emit(self._active_tag)

    def _clear_filter(self) -> None:
        self._active_tag = ""
        self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(False)
        self._filter_label.setText("")
        self.tag_filter_changed.emit("")


# ---------------------------------------------------------------------------
# Pannello griglia thumbnail
# ---------------------------------------------------------------------------


class _ThumbGrid(QWidget):
    """Griglia scrollabile di thumbnail del dataset."""

    item_selected = pyqtSignal(object)  # DatasetItem | None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: list[DatasetItem] = []
        self._tag_filter: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._empty_label = QLabel("Nessuna immagine nel dataset.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #8a8d96;")
        outer.addWidget(self._empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._inner = QWidget()
        self._grid = QGridLayout(self._inner)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._inner)
        outer.addWidget(scroll, 1)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

    def load(self, items: list[DatasetItem]) -> None:
        self._items = items
        self._redraw()

    def apply_tag_filter(self, tag: str) -> None:
        self._tag_filter = tag
        self._redraw()

    def _redraw(self) -> None:
        # Pulisci griglia
        for btn in list(self._group.buttons()):
            self._group.removeButton(btn)
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w:
                w.deleteLater()

        filtered = (
            [it for it in self._items if it.matches_tag(self._tag_filter)]
            if self._tag_filter
            else self._items
        )

        if not filtered:
            msg = (
                f"Nessuna immagine con tag «{self._tag_filter}»."
                if self._tag_filter
                else "Nessuna immagine nel dataset."
            )
            self._empty_label.setText(msg)
            self._empty_label.setVisible(True)
            return
        self._empty_label.setVisible(False)

        for i, item in enumerate(filtered):
            thumb = _Thumb(item, i)
            thumb.toggled.connect(
                lambda checked, it=item: self._on_toggled(checked, it)
            )
            self._group.addButton(thumb)
            r, c = divmod(i, _GRID_COLS)
            self._grid.addWidget(thumb, r, c)

    def _on_toggled(self, checked: bool, item: DatasetItem) -> None:
        if checked:
            self.item_selected.emit(item)

    def deselect(self) -> None:
        checked = self._group.checkedButton()
        if checked:
            self._group.setExclusive(False)
            checked.setChecked(False)
            self._group.setExclusive(True)
        self.item_selected.emit(None)


# ---------------------------------------------------------------------------
# Pannello dettaglio (preview + editor caption)
# ---------------------------------------------------------------------------


class _DetailPanel(QWidget):
    """Preview immagine + editor caption + azioni rimuovi/salva."""

    removed = pyqtSignal()  # dopo rimozione, la griglia deve ricaricarsi

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._item: Optional[DatasetItem] = None
        self._build_ui()

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 0, 0, 0)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(200)
        self._preview.setStyleSheet("background: #1a1d27; border-radius: 6px;")
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self._preview, 1)

        v.addWidget(_hline())

        caption_header = QHBoxLayout()
        caption_lbl = QLabel("Caption (tag comma-separated)")
        caption_lbl.setStyleSheet("font-weight: bold;")
        caption_header.addWidget(caption_lbl)
        caption_help = QLabel("?")
        caption_help.setToolTip(
            "La caption è la 'etichetta' che il modello legge durante il training.\n"
            "Contiene i tag (parole chiave) separati da virgola.\n"
            "Esempio: masterpiece, 1girl, red hair, casual outfit\n\n"
            "Rimuovi i tag che descrivono caratteristiche che NON vuoi\n"
            "che il modello associ a questo personaggio/stile."
        )
        caption_help.setStyleSheet("color: #8a8d96; font-size: 10px;")
        caption_header.addWidget(caption_help)
        caption_header.addStretch()
        v.addLayout(caption_header)

        self._caption_edit = QPlainTextEdit()
        self._caption_edit.setPlaceholderText(
            "Nessuna caption — aggiungi tag separati da virgola…"
        )
        self._caption_edit.setFixedHeight(80)
        self._caption_edit.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(self._caption_edit)

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("💾  Salva caption")
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)
        self._remove_btn = QPushButton("🗑  Rimuovi dal dataset")
        self._remove_btn.setStyleSheet("color: #c0504d;")
        self._remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self._remove_btn)
        v.addLayout(btn_row)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #8a8d96; font-size: 10px;")
        v.addWidget(self._info_label)

        self._set_enabled(False)

    def load(self, item: Optional[DatasetItem]) -> None:
        self._item = item
        if item is None:
            self._preview.clear()
            self._preview.setText("← Seleziona un'immagine")
            self._caption_edit.clear()
            self._info_label.setText("")
            self._set_enabled(False)
            return

        # Preview
        pix = QPixmap(str(item.image_path))
        if not pix.isNull():
            self._preview.setPixmap(
                pix.scaled(
                    self._preview.width() or 260, 260,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._preview.setText(item.name)

        # Caption
        self._caption_edit.setPlainText(item.caption)
        n_tags = len(item.tags)
        self._info_label.setText(
            f"{item.name} · {n_tags} tag" + ("" if item.has_caption else "  ⚠ nessuna caption")
        )
        self._set_enabled(True)

    def _set_enabled(self, v: bool) -> None:
        self._save_btn.setEnabled(v)
        self._remove_btn.setEnabled(v)
        self._caption_edit.setEnabled(v)

    def _on_save(self) -> None:
        if self._item is None:
            return
        text = self._caption_edit.toPlainText()
        try:
            save_caption(self._item, text)
            n = len(self._item.tags)  # tags del vecchio oggetto — aggiorniamo l'info
            saved_tags = [t.strip() for t in text.split(",") if t.strip()]
            self._info_label.setText(
                f"{self._item.name} · {len(saved_tags)} tag  ✓ salvato"
            )
        except OSError as exc:
            QMessageBox.critical(self, "Errore", f"Impossibile salvare:\n{exc}")

    def _on_remove(self) -> None:
        if self._item is None:
            return
        choice = QMessageBox.question(
            self, "Rimuovi dal dataset",
            f"Rimuovere '{self._item.name}' e la sua caption dal dataset?\n\n"
            "Il modello non vedrà più questa immagine nel prossimo addestramento.\n"
            "L'operazione non è reversibile.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        remove_dataset_item(self._item)
        self._item = None
        self._set_enabled(False)
        self._preview.setText("← Seleziona un'immagine")
        self._caption_edit.clear()
        self._info_label.setText("")
        self.removed.emit()


# ---------------------------------------------------------------------------
# Vista principale Dataset
# ---------------------------------------------------------------------------


class DatasetView(QWidget):
    """Vista Dataset Inspector. Dipendenza iniettata via set_project()."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[Project] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # Intro: spiega in linguaggio semplice a cosa serve questa vista
        intro = QLabel(
            "Qui vedi esattamente i dati con cui il modello si allena. "
            "Vuoi che impari qualcosa di diverso? Togli le immagini sbagliate o "
            "correggi le caption — poi riaddestralo."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8a8d96; font-size: 11px; padding: 4px 0;")
        root.addWidget(intro)

        # Tabs Dataset / Negativi
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._dataset_pane = self._build_pane("dataset")
        self._negatives_pane = self._build_pane("negatives")
        self._tabs.addTab(self._dataset_pane["widget"], "🖼  Dataset (training)")
        self._tabs.addTab(self._negatives_pane["widget"], "👎  Negativi (scarti)")

        root.addWidget(self._tabs, 1)

    def _build_pane(self, mode: str) -> dict:
        """Costruisce uno dei due pannelli (dataset o negativi) con lo stesso layout."""
        widget = QWidget()
        v = QVBoxLayout(widget)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # Aggiungi immagini: copia i file scelti direttamente nella cartella
        # giusta del progetto, senza dover andare a cercarla in Explorer.
        add_row = QHBoxLayout()
        add_btn = QPushButton("➕  Aggiungi immagini…")
        add_btn.clicked.connect(lambda _checked=False, m=mode: self._on_add_images(m))
        add_row.addWidget(add_btn)
        add_row.addStretch()
        v.addLayout(add_row)

        # Pannello stats + tabella tag
        stats = _StatsPanel()
        v.addWidget(stats)

        v.addWidget(_hline())

        # Splitter orizzontale: griglia | dettaglio
        splitter = QSplitter(Qt.Orientation.Horizontal)

        grid = _ThumbGrid()
        splitter.addWidget(grid)

        detail = _DetailPanel()
        splitter.addWidget(detail)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 280])
        v.addWidget(splitter, 1)

        # Connessioni interne
        stats.tag_filter_changed.connect(grid.apply_tag_filter)
        grid.item_selected.connect(detail.load)
        detail.removed.connect(lambda: self._reload_pane(mode))

        return {
            "widget": widget,
            "stats": stats,
            "grid": grid,
            "detail": detail,
        }

    # --- dipendenze / refresh ------------------------------------------

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self._reload_all()

    def _reload_all(self) -> None:
        self._reload_pane("dataset")
        self._reload_pane("negatives")

    def _reload_pane(self, mode: str) -> None:
        if self._project is None:
            return
        pane = self._dataset_pane if mode == "dataset" else self._negatives_pane
        directory = (
            self._project.dataset_images_dir
            if mode == "dataset"
            else self._project.negatives_dir
        )
        items = scan_dataset(directory)
        pane["stats"].load(items)
        pane["grid"].load(items)
        pane["detail"].load(None)

    def _on_add_images(self, mode: str) -> None:
        """Copia le immagini scelte dall'utente nella cartella dataset/negativi
        del progetto, poi ricarica il pannello per mostrarle subito."""
        if self._project is None:
            return
        ext_filter = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Scegli immagini da aggiungere", "",
            f"Immagini ({ext_filter})",
        )
        if not paths:
            return

        directory = (
            self._project.dataset_images_dir if mode == "dataset"
            else self._project.negatives_dir
        )
        directory.mkdir(parents=True, exist_ok=True)

        added = 0
        errors: list[str] = []
        for raw in paths:
            src = Path(raw)
            dest = directory / src.name
            if dest.exists():
                # Nome già presente: aggiunge un suffisso invece di sovrascrivere
                # una caption/immagine esistente per errore.
                dest = directory / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix}"
            try:
                shutil.copy2(src, dest)
                added += 1
            except OSError as exc:
                errors.append(f"{src.name}: {exc}")

        self._reload_pane(mode)

        msg = f"{added} immagine/i aggiunta/e." if added else "Nessuna immagine aggiunta."
        if errors:
            msg += "\n\nAlcuni file non sono stati copiati:\n" + "\n".join(errors)
        QMessageBox.information(self, "Aggiunta al dataset", msg)

    def _on_tab_changed(self, index: int) -> None:
        # Ricarica il pannello appena visualizzato per riflettere modifiche recenti
        mode = "dataset" if index == 0 else "negatives"
        self._reload_pane(mode)
