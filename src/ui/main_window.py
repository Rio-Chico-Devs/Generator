"""Finestra principale Vihente Forge.

Layout:
    +-------------------+----------------------------------+
    |                   |  Toolbar / breadcrumb            |
    |  Sidebar          +----------------------------------+
    |  progetti         |                                  |
    |                   |  Workspace (QStackedWidget)      |
    |  - Iris           |    - Dataset view                |
    |  - RCS BG         |    - Train view                  |
    |  - Sprite Gothic  |    - Generate view               |
    |                   |    - Gallery view                |
    |  [+ Nuovo]        |                                  |
    +-------------------+----------------------------------+
    |  Status bar (GPU, VRAM, model loaded, eventi)         |
    +-------------------------------------------------------+

Questo file è uno SCAFFOLD: gli stub delle view sono placeholder, da
espandere in Fase 1+.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.core.project import Project
from src.utils.paths import ensure_user_dirs, get_projects_dir

logger = logging.getLogger(__name__)


class _PlaceholderView(QWidget):
    """Placeholder view per stub Fase 1."""

    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addStretch()
        label = QLabel(f"[ {name} ]\n\nDa implementare in Fase {_phase_of(name)}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addStretch()


def _phase_of(name: str) -> int:
    return {"Dataset": 2, "Train": 3, "Generate": 1, "Gallery": 1}.get(name, 1)


class MainWindow(QMainWindow):
    def __init__(
        self,
        mock: bool = False,
        skip_model_check: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.mock = mock
        self.skip_model_check = skip_model_check
        self._current_project: Optional[Project] = None

        ensure_user_dirs()

        self.setWindowTitle("Vihente Forge")
        self.resize(1280, 800)

        self._build_ui()
        self._refresh_projects_list()

        if not skip_model_check:
            self._maybe_show_first_run_dialog()

    # --- UI build ----------------------------------------------------

    def _build_ui(self) -> None:
        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        action_new = QAction("Nuovo progetto", self)
        action_new.triggered.connect(self._on_new_project)
        toolbar.addAction(action_new)

        toolbar.addSeparator()

        for view_name in ("Dataset", "Train", "Generate", "Gallery"):
            a = QAction(view_name, self)
            a.triggered.connect(lambda checked, n=view_name: self._switch_view(n))
            toolbar.addAction(a)

        # Sidebar progetti
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)

        sidebar_layout.addWidget(QLabel("Progetti"))
        self.projects_list = QListWidget()
        self.projects_list.itemDoubleClicked.connect(self._on_project_activated)
        sidebar_layout.addWidget(self.projects_list)

        new_btn = QPushButton("+ Nuovo")
        new_btn.clicked.connect(self._on_new_project)
        sidebar_layout.addWidget(new_btn)

        # Workspace (stacked views)
        self.workspace = QStackedWidget()
        self._views: dict[str, QWidget] = {}
        for name in ("Dataset", "Train", "Generate", "Gallery"):
            v = _PlaceholderView(name)
            self._views[name] = v
            self.workspace.addWidget(v)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1020])

        self.setCentralWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._start_gpu_monitor()

    # --- Logica progetti ---------------------------------------------

    def _refresh_projects_list(self) -> None:
        self.projects_list.clear()
        projects_root = get_projects_dir()
        if not projects_root.exists():
            return

        for entry in sorted(projects_root.iterdir()):
            if not entry.is_dir():
                continue
            if not (entry / "project.json").exists():
                continue
            try:
                p = Project.load(entry)
                item = QListWidgetItem(p.name)
                item.setData(Qt.ItemDataRole.UserRole, str(entry))
                item.setToolTip(p.description or p.slug)
                self.projects_list.addItem(item)
            except Exception as e:
                logger.warning("Progetto non caricabile %s: %s", entry, e)

    def _on_project_activated(self, item: QListWidgetItem) -> None:
        path_str = item.data(Qt.ItemDataRole.UserRole)
        try:
            self._current_project = Project.load(Path(path_str))
            self.setWindowTitle(f"Vihente Forge — {self._current_project.name}")
            self._update_status()
            # In futuro: notifica view per refresh
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Caricamento progetto fallito:\n{e}")

    def _on_new_project(self) -> None:
        # Stub: in Fase 1 vero aprirà ProjectDialog
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Nuovo progetto", "Nome del progetto:")
        if not ok or not name.strip():
            return

        try:
            project = Project.create(name=name.strip())
            self._refresh_projects_list()
            QMessageBox.information(
                self,
                "Progetto creato",
                f"Progetto '{project.name}' creato in:\n{project.root}\n\n"
                f"Tag attivatore: {project.activator_tag}",
            )
        except Exception as e:
            logger.exception("Creazione progetto fallita")
            QMessageBox.critical(self, "Errore", f"Creazione fallita:\n{e}")

    def _switch_view(self, name: str) -> None:
        view = self._views.get(name)
        if view is None:
            return
        self.workspace.setCurrentWidget(view)

    # --- Status / first run ------------------------------------------

    def _start_gpu_monitor(self) -> None:
        """Avvia aggiornamento periodico della status bar (ogni 2s)."""
        from PyQt6.QtCore import QTimer

        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_status)
        self._gpu_timer.start(2000)
        self._update_status()

    def _update_status(self) -> None:
        from src.utils.gpu_monitor import ThermalState, read_gpu

        parts = []
        if self.mock:
            parts.append("MOCK")
        if self._current_project:
            parts.append(f"Progetto: {self._current_project.name}")
        else:
            parts.append("Nessun progetto attivo")

        snap = read_gpu()
        parts.append(snap.status_line())
        self.status_bar.showMessage(" │ ".join(parts))

        # Colore della status bar in base allo stato termico
        color = {
            ThermalState.COOL: "#8a8d96",
            ThermalState.NORMAL: "#8a8d96",
            ThermalState.WARM: "#d9a441",   # ambra: caldo ma sicuro
            ThermalState.HOT: "#d96a6a",    # rosso: throttling hardware attivo
        }.get(snap.thermal_state, "#8a8d96") if snap.available else "#8a8d96"
        self.status_bar.setStyleSheet(f"QStatusBar {{ color: {color}; }}")

    def _maybe_show_first_run_dialog(self) -> None:
        """Wizard primo avvio: download modelli base."""
        from src.utils.paths import get_models_dir

        models_dir = get_models_dir()
        has_any_model = (
            models_dir.exists()
            and any((models_dir / "base").glob("*/model_index.json"))
            if (models_dir / "base").exists()
            else False
        )
        if has_any_model:
            return

        if self.mock:
            logger.info("Mock mode: skip first run dialog")
            return

        # In Fase 1 vero: aprire wizard con scelta modello + progress download
        QMessageBox.information(
            self,
            "Benvenuto",
            "Sembra il tuo primo avvio.\n\n"
            "Per generare immagini serve scaricare un modello base "
            "(~7 GB per SDXL).\n\n"
            "Funzionalità da implementare in Fase 1: wizard download.",
        )
