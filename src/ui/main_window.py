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

        # Wallet crediti condiviso: bonus di benvenuto al primo avvio.
        from src.core.credits import CreditWallet
        self._wallet = CreditWallet.load()
        if not self._wallet.ledger:
            self._wallet.grant(1000.0, reason="bonus di benvenuto")
            self._wallet.save()

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

        for view_name in ("Dataset", "Train", "Guidato", "Generate", "Gallery", "Tecniche", "Diario"):
            a = QAction(view_name, self)
            a.triggered.connect(lambda checked, n=view_name: self._switch_view(n))
            toolbar.addAction(a)

        toolbar.addSeparator()
        action_credits = QAction("Crediti", self)
        action_credits.triggered.connect(self._on_open_credits)
        toolbar.addAction(action_credits)

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
        for name in ("Dataset", "Train", "Guidato", "Generate", "Gallery", "Tecniche", "Diario"):
            if name == "Dataset":
                from src.ui.dataset_view import DatasetView
                v = DatasetView()
                self._dataset_view = v
            elif name == "Generate":
                from src.ui.generate_view import GenerateView
                v = GenerateView()
                v.set_wallet(self._wallet)
                if self.mock:
                    # Mock: niente freno termico, demo veloce
                    from src.utils.gpu_monitor import SafetyConfig
                    v.set_safety(SafetyConfig(enabled=False), None)
                else:
                    v.set_safety(self._app_config.thermal_safety(), None)
                v.balance_changed.connect(self._on_wallet_changed)
                self._generate_view = v
            elif name == "Gallery":
                from src.ui.gallery_view import GalleryView
                v = GalleryView()
                v.reuse_requested.connect(self._on_reuse_params)
                v.projects_changed.connect(self._refresh_projects_list)
                self._gallery_view = v
            elif name == "Guidato":
                from src.ui.guided_training_view import GuidedTrainingView
                v = GuidedTrainingView()
                v.session_saved_to_gallery.connect(self._on_guided_save_to_gallery)
                v.switch_to_dataset.connect(lambda: self._switch_view("Dataset"))
                v.switch_to_technique_library.connect(lambda: self._switch_view("Tecniche"))
                if self.mock:
                    # Mock: niente freno termico → nessuna pausa da 8s tra candidati
                    from src.utils.gpu_monitor import SafetyConfig
                    v.set_safety(SafetyConfig(enabled=False), None)
                else:
                    v.set_safety(self._app_config.thermal_safety(), None)
                self._guided_view = v
            elif name == "Train":
                from src.ui.train_view import TrainView
                import os as _os
                v = TrainView()
                sdscripts_env = _os.environ.get("VFORGE_SDSCRIPTS_DIR", "")
                if sdscripts_env:
                    v.set_sdscripts_dir(Path(sdscripts_env))
                self._train_view = v
            elif name == "Tecniche":
                from src.ui.technique_library_view import TechniqueLibraryView
                v = TechniqueLibraryView()
                self._technique_view = v
            elif name == "Diario":
                from src.ui.diary_stats_view import DiaryStatsView
                v = DiaryStatsView()
                self._diary_view = v
            else:
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
            gv = getattr(self, "_generate_view", None)
            if gv is not None:
                gv.set_project(self._current_project)
            gal = getattr(self, "_gallery_view", None)
            if gal is not None:
                gal.set_project(self._current_project)
            dv = getattr(self, "_dataset_view", None)
            if dv is not None:
                dv.set_project(self._current_project)
            guided = getattr(self, "_guided_view", None)
            if guided is not None:
                guided.set_project(self._current_project)
            tech = getattr(self, "_technique_view", None)
            if tech is not None:
                tech.set_project(self._current_project)
            diary = getattr(self, "_diary_view", None)
            if diary is not None:
                diary.set_project(self._current_project)
            tv = getattr(self, "_train_view", None)
            if tv is not None:
                tv.set_project(self._current_project)
            self._render_status()
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Caricamento progetto fallito:\n{e}")

    def _on_new_project(self) -> None:
        from src.ui.project_dialog import ProjectDialog

        dlg = ProjectDialog(parent=self)
        if dlg.exec() != ProjectDialog.DialogCode.Accepted:
            return

        project = dlg.project
        if project is None:
            return

        self._refresh_projects_list()
        QMessageBox.information(
            self,
            "Progetto creato",
            f"Progetto «{project.name}» creato.\n\n"
            f"Tag attivatore: {project.activator_tag}",
        )

    def _switch_view(self, name: str) -> None:
        view = self._views.get(name)
        if view is None:
            return
        # Entrando in Galleria/Dataset/Diario, ricarica per mostrare i dati più recenti.
        if name == "Gallery":
            gal = getattr(self, "_gallery_view", None)
            if gal is not None:
                gal.refresh()
        elif name == "Dataset":
            dv = getattr(self, "_dataset_view", None)
            if dv is not None and self._current_project is not None:
                dv.set_project(self._current_project)
        elif name == "Diario":
            diary = getattr(self, "_diary_view", None)
            if diary is not None:
                diary.set_project(self._current_project)
        elif name == "Tecniche":
            tech = getattr(self, "_technique_view", None)
            if tech is not None and self._current_project is not None:
                tech.set_project(self._current_project)
        elif name == "Train":
            tv = getattr(self, "_train_view", None)
            if tv is not None and self._current_project is not None:
                tv.set_project(self._current_project)
        self.workspace.setCurrentWidget(view)

    def _on_guided_save_to_gallery(self, image_path: Path) -> None:
        """Copia l'immagine finale della sessione guidata nella gallery del progetto."""
        if self._current_project is None:
            return
        import os as _os
        import shutil as _shutil
        dest_dir = self._current_project.gallery_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / image_path.name
        if not dest.exists():
            tmp = dest.with_suffix(".copy.tmp")
            _shutil.copy2(image_path, tmp)
            _os.replace(tmp, dest)
        # Copia anche sidecar
        sidecar = image_path.with_suffix(".json")
        if sidecar.exists():
            sidecar_dest = dest.with_suffix(".json")
            tmp_s = sidecar_dest.with_suffix(".copy.tmp")
            _shutil.copy2(sidecar, tmp_s)
            _os.replace(tmp_s, sidecar_dest)
        # Passa alla gallery
        self._switch_view("Gallery")

    def _on_reuse_params(self, params: dict) -> None:
        """Riusa i parametri di un'immagine: pre-compila Genera e ci passa."""
        gv = getattr(self, "_generate_view", None)
        if gv is not None:
            gv.apply_metadata(params)
            self._switch_view("Generate")

    # --- Crediti ------------------------------------------------------

    def _on_open_credits(self) -> None:
        from src.ui.credit_dialog import CreditDialog

        dlg = CreditDialog(self._wallet, parent=self)
        dlg.wallet_changed.connect(self._on_wallet_changed)
        dlg.exec()

    def _on_wallet_changed(self) -> None:
        """Persiste il wallet e aggiorna l'etichetta saldo della GenerateView."""
        self._wallet.save()
        gv = getattr(self, "_generate_view", None)
        if gv is not None:
            gv.refresh_balance()

    # --- Status / first run ------------------------------------------

    def _start_comfy(self) -> None:
        """Avvia ComfyUI in background (o usa MockClient se mock=True)."""
        if self.mock:
            from src.comfy.client import MockComfyClient
            self._comfy_client = MockComfyClient()
            self._comfy_state = "mock"
            gv = getattr(self, "_generate_view", None)
            if gv is not None:
                gv.set_comfy_client(self._comfy_client)
            guided = getattr(self, "_guided_view", None)
            if guided is not None:
                guided.set_comfy_client(self._comfy_client)
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
        gv = getattr(self, "_generate_view", None)
        if gv is not None:
            gv.set_comfy_client(self._comfy_client)
        guided = getattr(self, "_guided_view", None)
        if guided is not None:
            guided.set_comfy_client(self._comfy_client)
            if self._comfy_server is not None:
                guided.set_comfy_input_dir(self._comfy_server.input_dir)
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
        wallet = getattr(self, "_wallet", None)
        if wallet is not None:
            try:
                wallet.save()
            except Exception as exc:
                logger.warning("Salvataggio wallet fallito: %s", exc)

        # Ferma i worker PRIMA di spegnere ComfyUI:
        # interrupt/clear_queue hanno bisogno del server ancora vivo.
        generate = getattr(self, "_generate_view", None)
        if generate is not None:
            try:
                generate.shutdown()
            except Exception as exc:
                logger.warning("Shutdown vista generazione fallito: %s", exc)

        guided = getattr(self, "_guided_view", None)
        if guided is not None:
            try:
                guided.shutdown()
            except Exception as exc:
                logger.warning("Shutdown vista guidata fallito: %s", exc)

        train = getattr(self, "_train_view", None)
        if train is not None:
            try:
                train.shutdown()
            except Exception as exc:
                logger.warning("Shutdown vista training fallito: %s", exc)

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
