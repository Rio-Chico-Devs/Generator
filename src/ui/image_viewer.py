"""Visore immagini integrato + menu contestuale riusabile.

Usato sia dalla GenerateView (miniature dei risultati appena generati) sia
dalla GalleryView (immagini salvate del progetto). Tiene in un solo posto:

- :class:`ImageViewerDialog` — anteprima a tutto schermo (fit-to-screen) con
  scroll se l'immagine eccede lo schermo. Si chiude con ESC o con un click.
- :func:`build_image_menu` — un ``QMenu`` con Salva con nome / Apri cartella /
  Copia immagine, agganciabile a qualsiasi widget col tasto destro.

Nessuna logica di business: solo presentazione e operazioni su file locali.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class ImageViewerDialog(QDialog):
    """Mostra un'immagine a piena risoluzione, ridimensionata per stare a schermo.

    L'immagine viene scalata per non superare il ~90% dell'area disponibile
    dello schermo; se è comunque più grande, lo scroll permette di esplorarla.
    Click sull'immagine o ESC chiudono il visore.
    """

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle(self.path.name)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet("background: #14161c;")

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._label.customContextMenuRequested.connect(self._on_context_menu)

        pix = QPixmap(str(self.path))
        if pix.isNull():
            self._label.setText(f"Impossibile aprire:\n{self.path}")
            self.resize(480, 200)
        else:
            screen = QGuiApplication.primaryScreen().availableGeometry()
            max_w = int(screen.width() * 0.9)
            max_h = int(screen.height() * 0.9)
            if pix.width() > max_w or pix.height() > max_h:
                pix = pix.scaled(
                    max_w, max_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self._label.setPixmap(pix)
            # +2 per i bordi, così l'immagine non viene tagliata
            self.resize(pix.width() + 2, pix.height() + 2)

        scroll.setWidget(self._label)
        layout.addWidget(scroll)

    def _on_context_menu(self, pos) -> None:
        menu = build_image_menu(self, self.path)
        menu.exec(self._label.mapToGlobal(pos))

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Click sinistro chiude; destro lo lasciamo al menu contestuale del label.
        if event.button() == Qt.MouseButton.LeftButton:
            self.accept()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() == Qt.Key.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)


def show_image(path: Path, parent=None) -> None:
    """Apre il visore a tutto schermo per ``path`` (no-op se il file manca)."""
    p = Path(path)
    if not p.exists():
        QMessageBox.warning(parent, "File assente", f"Immagine non trovata:\n{p}")
        return
    ImageViewerDialog(p, parent=parent).exec()


# --- Menu contestuale -----------------------------------------------------


def build_image_menu(parent, path: Path) -> QMenu:
    """Costruisce un menu con le azioni standard su un'immagine."""
    p = Path(path)
    menu = QMenu(parent)
    menu.addAction("Apri a tutto schermo", lambda: show_image(p, parent))
    menu.addSeparator()
    menu.addAction("Salva con nome…", lambda: save_image_as(parent, p))
    menu.addAction("Copia immagine", lambda: copy_image_to_clipboard(parent, p))
    menu.addAction("Apri cartella", lambda: open_containing_folder(p))
    return menu


def save_image_as(parent, path: Path) -> None:
    """Salva una copia dell'immagine in un percorso scelto dall'utente."""
    p = Path(path)
    if not p.exists():
        QMessageBox.warning(parent, "File assente", f"Immagine non trovata:\n{p}")
        return
    dest, _ = QFileDialog.getSaveFileName(
        parent, "Salva immagine con nome", p.name,
        "Immagini (*.png *.jpg *.jpeg *.webp);;Tutti i file (*.*)",
    )
    if not dest:
        return
    try:
        shutil.copy2(p, dest)
    except OSError as exc:
        QMessageBox.critical(parent, "Errore", f"Salvataggio fallito:\n{exc}")


def copy_image_to_clipboard(parent, path: Path) -> None:
    """Copia l'immagine negli appunti di sistema."""
    pix = QPixmap(str(path))
    if pix.isNull():
        QMessageBox.warning(parent, "Errore", f"Impossibile leggere:\n{path}")
        return
    QApplication.clipboard().setPixmap(pix)


def open_containing_folder(path: Path) -> None:
    """Apre la cartella che contiene il file nel file manager di sistema.

    Su Windows seleziona anche il file; altrove apre la cartella.
    """
    p = Path(path)
    folder = p.parent
    if os.name == "nt" and p.exists():
        # /select evidenzia il file dentro Explorer
        os.system(f'explorer /select,"{p}"')
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
