"""Dialog creazione nuovo progetto.

Sostituisce il QInputDialog.getText stub in main_window.py con un dialog
completo che raccoglie: nome, descrizione, modello base e tag attivatore.

Il tag attivatore viene auto-generato dallo slug del nome (es. "Iris" →
"vf_iris_v1") ma l'utente può modificarlo. Non viene validata la presenza
del modello su disco (avviene alla generazione/training).
"""
from __future__ import annotations

import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
)

from src.core.catalog import CATALOG
from src.core.project import BaseModelRef, Project


class ProjectDialog(QDialog):
    """Dialog modale per creare un nuovo progetto."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nuovo progetto")
        self.setMinimumWidth(460)
        self.setModal(True)

        self._project: Project | None = None
        self._tag_edited_manually = False  # True se l'utente ha toccato il tag

        self._build_ui()

    # --- Build -----------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Nome (obbligatorio)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("es. Iris, RCS BG, Sprite Gothic")
        self._name_edit.textChanged.connect(self._on_name_changed)
        form.addRow("Nome *:", self._name_edit)

        # Descrizione (opzionale)
        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlaceholderText("Descrizione opzionale del progetto...")
        self._desc_edit.setFixedHeight(60)
        form.addRow("Descrizione:", self._desc_edit)

        # Modello base
        self._model_combo = QComboBox()
        self._model_combo.addItem("(nessuno — sceglierò dopo)", None)
        for mid, entry in CATALOG.items():
            self._model_combo.addItem(f"{entry.name} [{entry.family.upper()}]", mid)
        form.addRow("Modello base:", self._model_combo)

        # Tag attivatore
        self._tag_edit = QLineEdit()
        self._tag_edit.setPlaceholderText("Auto-generato dal nome")
        self._tag_edit.textEdited.connect(self._on_tag_edited)
        form.addRow("Tag attivatore:", self._tag_edit)

        tag_hint = QLabel(
            "Il tag attivatore è la parola chiave che attiva il tuo stile "
            "nei prompt (es. <b>vf_iris_v1</b>). Auto-generato dal nome."
        )
        tag_hint.setWordWrap(True)
        tag_hint.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow("", tag_hint)

        layout.addLayout(form)

        # Bottoni
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Crea progetto")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self._ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(False)
        layout.addWidget(buttons)

    # --- Slots -----------------------------------------------------------

    def _on_name_changed(self, text: str) -> None:
        self._ok_btn.setEnabled(bool(text.strip()))
        if not self._tag_edited_manually:
            auto_tag = _auto_tag(text)
            self._tag_edit.setText(auto_tag)

    def _on_tag_edited(self, text: str) -> None:
        self._tag_edited_manually = bool(text.strip())

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return

        description = self._desc_edit.toPlainText().strip()
        model_id = self._model_combo.currentData()

        base_model: BaseModelRef | None = None
        if model_id and model_id in CATALOG:
            entry = CATALOG[model_id]
            base_model = BaseModelRef(id=model_id, name=entry.name)

        try:
            self._project = Project.create(
                name=name,
                description=description,
                base_model=base_model,
            )
            # Override tag attivatore se l'utente ne ha inserito uno custom
            custom_tag = self._tag_edit.text().strip()
            if custom_tag and custom_tag != _auto_tag(name):
                self._project.activator_tag = custom_tag
                self._project.trigger_prompt_prefix = (
                    f"{custom_tag}, masterpiece, best quality,"
                )
                self._project.save()
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Errore", f"Creazione progetto fallita:\n{exc}")
            return

        self.accept()

    # --- Risultato -------------------------------------------------------

    @property
    def project(self) -> Project | None:
        """Il progetto creato, disponibile dopo accept(). None se annullato."""
        return self._project


# --- Helper ----------------------------------------------------------


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "progetto"


def _auto_tag(name: str) -> str:
    slug = _slugify(name)
    return f"vf_{slug}_v1"
