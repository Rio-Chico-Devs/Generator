"""Dialog di inpainting: disegna una maschera sull'immagine e rigenera solo
quella zona (ricetta ``correct_inpaint``, soli nodi core di ComfyUI).

L'utente dipinge col mouse l'area da correggere; la maschera viene esportata
come PNG in scala di grigi (bianco = zona da rigenerare) alla risoluzione
originale dell'immagine, pronta per il nodo ``LoadImageMask``.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.utils.paths import get_app_data_dir

logger = logging.getLogger(__name__)

_MASK_RED = QColor(220, 40, 40, 120)


class _MaskCanvas(QWidget):
    """Tela su cui dipingere la maschera sopra l'immagine.

    Mantiene due livelli: ``_mask`` (grigio, per l'export verso ComfyUI) e
    ``_overlay`` (rosso semi-trasparente, solo per la visualizzazione)."""

    def __init__(self, display: QPixmap, parent=None) -> None:
        super().__init__(parent)
        self._base = display
        self.setFixedSize(display.size())
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._mask = QImage(display.size(), QImage.Format.Format_Grayscale8)
        self._mask.fill(0)
        self._overlay = QImage(display.size(), QImage.Format.Format_ARGB32)
        self._overlay.fill(0)

        self._brush = 40
        self._last = None
        self._dirty = False

    def set_brush(self, diameter: int) -> None:
        self._brush = max(4, int(diameter))

    def clear(self) -> None:
        self._mask.fill(0)
        self._overlay.fill(0)
        self._dirty = False
        self.update()

    def has_mask(self) -> bool:
        return self._dirty

    def export_mask(self, width: int, height: int) -> QImage:
        """Maschera in scala di grigi scalata alla dimensione originale."""
        return self._mask.scaled(
            width, height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    # --- disegno ---------------------------------------------------------

    def _stroke(self, pos) -> None:
        for img, color in ((self._mask, QColor(255, 255, 255)), (self._overlay, _MASK_RED)):
            p = QPainter(img)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(
                color, self._brush,
                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            p.setPen(pen)
            if self._last is not None:
                p.drawLine(self._last, pos)
            else:
                p.drawPoint(pos)
            p.end()
        self._last = pos
        self._dirty = True
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = None
            self._stroke(event.position().toPoint())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._stroke(event.position().toPoint())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._last = None

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.drawPixmap(0, 0, self._base)
        p.drawImage(0, 0, self._overlay)
        p.end()


class InpaintDialog(QDialog):
    """Disegna la maschera + parametri; espone mask_path / prompt / strength."""

    def __init__(self, image_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Correggi una zona")
        self.setModal(True)
        self._image_path = Path(image_path)
        self._mask_path: Optional[Path] = None

        self._orig = QImage(str(self._image_path))
        ow, oh = self._orig.width(), self._orig.height()

        display = QPixmap.fromImage(self._orig)
        screen = QGuiApplication.primaryScreen().availableGeometry()
        max_w, max_h = int(screen.width() * 0.7), int(screen.height() * 0.7)
        if ow > max_w or oh > max_h:
            display = display.scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Dipingi con il mouse la zona da sistemare (rosso). Verrà "
            "rigenerata solo quella, il resto resta intatto."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a8d96;")
        layout.addWidget(hint)

        self._canvas = _MaskCanvas(display)
        canvas_row = QHBoxLayout()
        canvas_row.addStretch()
        canvas_row.addWidget(self._canvas)
        canvas_row.addStretch()
        layout.addLayout(canvas_row)

        # Pennello
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Pennello:"))
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setMinimum(8)
        self._brush_slider.setMaximum(160)
        self._brush_slider.setValue(40)
        self._brush_slider.valueChanged.connect(self._canvas.set_brush)
        brush_row.addWidget(self._brush_slider, 1)
        clear_btn = QPushButton("Cancella maschera")
        clear_btn.clicked.connect(self._canvas.clear)
        brush_row.addWidget(clear_btn)
        layout.addLayout(brush_row)

        # Intensità (denoise)
        str_row = QHBoxLayout()
        str_row.addWidget(QLabel("Intensità:"))
        self._strength = QSlider(Qt.Orientation.Horizontal)
        self._strength.setMinimum(20)   # 0.20
        self._strength.setMaximum(100)  # 1.00
        self._strength.setValue(50)     # 0.50
        str_row.addWidget(self._strength, 1)
        self._strength_label = QLabel("0.50")
        self._strength.valueChanged.connect(
            lambda v: self._strength_label.setText(f"{v / 100:.2f}")
        )
        str_row.addWidget(self._strength_label)
        layout.addLayout(str_row)
        str_hint = QLabel("Bassa = ritocca leggero · Alta = ridisegna la zona")
        str_hint.setStyleSheet("color: #8a8d96; font-size: 11px;")
        layout.addWidget(str_hint)

        # Prompt opzionale
        layout.addWidget(QLabel("Cosa correggere (opzionale):"))
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText(
            "Lascia vuoto per pulire/ricostruire la zona, oppure descrivi cosa vuoi…"
        )
        self._prompt.setFixedHeight(56)
        layout.addWidget(self._prompt)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Correggi")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self._canvas.has_mask():
            QMessageBox.information(
                self, "Maschera vuota",
                "Dipingi prima la zona da correggere.",
            )
            return
        mask = self._canvas.export_mask(self._orig.width(), self._orig.height())
        tmp_dir = get_app_data_dir() / "inpaint_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self._mask_path = tmp_dir / f"mask_{uuid.uuid4().hex[:12]}.png"
        if not mask.save(str(self._mask_path)):
            QMessageBox.critical(self, "Errore", "Salvataggio maschera fallito.")
            return
        self.accept()

    # --- risultato -------------------------------------------------------

    @property
    def mask_path(self) -> Optional[Path]:
        return self._mask_path

    @property
    def prompt(self) -> str:
        return self._prompt.toPlainText().strip()

    @property
    def strength(self) -> float:
        return self._strength.value() / 100.0
