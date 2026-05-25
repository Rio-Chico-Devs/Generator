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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.core.app_config import AppConfig

from PyQt6.QtCore import Qt, QThread, pyqtSignal
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


class _GpuPoller(QThread):
    """Polling GPU in un thread separato.

    ``read_gpu()`` lancia ``nvidia-smi`` (subprocess bloccante): se girasse
    nel main thread tramite QTimer, ogni lettura congelerebbe la UI. Qui
    gira fuori dal main thread ed emette lo snapshot via signal, che Qt
    consegna in modo thread-safe allo slot nel main thread.
    """

    snapshot = pyqtSignal(object)  # GpuSnapshot

    def __init__(self, interval_sec: float = 2.0, parent=None) -> None:
        super().__init__(parent)
        self._interval = interval_sec
        self._stop = False

    def run(self) -> None:
        from src.utils.gpu_monitor import read_gpu

        while not self._stop:
            self.snapshot.emit(read_gpu())
            slept = 0.0
            while slept < self._interval and not self._stop:
                self.msleep(100)
                slept += 0.1

    def stop(self) -> None:
        self._stop = True


class _ComfyStarter(QThread):
    """Avvia ComfyUI in background — start() blocca fino a ~60s.

    Il risultato arriva via signal nel main thread (thread-safe).
    """

    ready = pyqtSignal(int)   # porta su cui ComfyUI risponde
    failed = pyqtSignal(str)  # messaggio di errore

    def __init__(self, server, parent=None) -> None:
        super().__init__(parent)
        self._server = server

    def run(self) -> None:
        try:
            port = self._server.start()
            self.ready.emit(port)
        except Exception as exc:
            self.failed.emit(str(exc))


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
        app_config: Optional["AppConfig"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.mock = mock
        self.skip_model_check = skip_model_check
        self._current_project: Optional[Project] = None
        self._last_snap = None  # ultimo GpuSnapshot ricevuto dal poller
        self._comfy_state = ""  # stringa breve per la status bar
        self._comfy_server = None
        self._comfy_starter: Optional[_ComfyStarter] = None
        self._comfy_client = None

        from src.core.app_config import AppConfig as _AppConfig
        self._app_config = app_config or _AppConfig()

        ensure_user_dirs()

        self.setWindowTitle("Vihente Forge")
        self.resize(1280, 800)

        self._build_ui()
        self._refresh_projects_list()
        self._start_comfy()

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
            self._render_status()
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

    def _start_comfy(self) -> None:
        """Avvia ComfyUI in background (o usa MockClient se mock=True)."""
        if self.mock:
            from src.comfy.client import MockComfyClient
            self._comfy_client = MockComfyClient()
            self._comfy_state = "mock"
            self._render_status()
            return

        from src.comfy.server import ComfyServer
        self._comfy_server = ComfyServer(
            vram_mode=self._app_config.comfy_vram_mode,
            preferred_port=self._app_config.comfy_port,
        )
        if not self._comfy_server.is_installed():
            logger.warning("ComfyUI non installato — generazione non disponibile")
            self._comfy_state = "non installato"
            self._render_status()
            return

        self._comfy_starter = _ComfyStarter(self._comfy_server, parent=self)
        self._comfy_starter.ready.connect(self._on_comfy_ready)
        self._comfy_starter.failed.connect(self._on_comfy_failed)
        self._comfy_state = "avvio..."
        self._comfy_starter.start()
        self._render_status()

    def _on_comfy_ready(self, port: int) -> None:
        from src.comfy.client import ComfyClient
        self._comfy_client = ComfyClient(port=port)
        self._comfy_state = f"pronto:{port}"
        logger.info("ComfyUI pronto su porta %d", port)
        self._render_status()

    def _on_comfy_failed(self, msg: str) -> None:
        logger.error("ComfyUI avvio fallito: %s", msg)
        self._comfy_state = "errore"
        self._render_status()

    def _start_gpu_monitor(self) -> None:
        """Avvia il polling GPU in un thread separato (no freeze UI)."""
        self._gpu_poller = _GpuPoller(interval_sec=2.0, parent=self)
        self._gpu_poller.snapshot.connect(self._on_gpu_snapshot)
        self._gpu_poller.start()
        self._render_status()

    def _on_gpu_snapshot(self, snap) -> None:
        """Slot nel main thread: riceve lo snapshot dal poller."""
        self._last_snap = snap
        self._render_status()

    def _render_status(self) -> None:
        from src.utils.gpu_monitor import GpuSnapshot, ThermalState

        snap = self._last_snap or GpuSnapshot(available=False)

        parts = []
        if self.mock:
            parts.append("MOCK")
        if self._current_project:
            parts.append(f"Progetto: {self._current_project.name}")
        else:
            parts.append("Nessun progetto attivo")

        if self._comfy_state:
            parts.append(f"Comfy: {self._comfy_state}")

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

    def closeEvent(self, event) -> None:
        """Ferma thread e processi figli prima di chiudere (no orfani)."""
        poller = getattr(self, "_gpu_poller", None)
        if poller is not None:
            poller.stop()
            poller.wait(2000)

        starter = getattr(self, "_comfy_starter", None)
        if starter is not None and starter.isRunning():
            starter.wait(5000)

        server = getattr(self, "_comfy_server", None)
        if server is not None:
            server.stop()

        super().closeEvent(event)

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
