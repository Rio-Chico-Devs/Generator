"""Bootstrap dell'applicazione PyQt6."""
from __future__ import annotations

import argparse
import logging
import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src.core.app_config import AppConfig
from src.utils.paths import get_app_data_dir, get_assets_dir

logger = logging.getLogger(__name__)


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_dir = get_app_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "vihente-forge.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vihente-forge")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="MockComfyClient (no GPU, no ComfyUI). Per sviluppo UI senza modelli.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (per test isolati).",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Salta verifica modelli al boot.",
    )
    return parser.parse_args(argv[1:])


def _apply_stylesheet(app: QApplication) -> None:
    qss_path = get_assets_dir() / "styles" / "dark.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    else:
        logger.warning("Stylesheet non trovato: %s", qss_path)


def run_app(argv: list[str]) -> int:
    args = _parse_args(argv)
    _setup_logging(args.debug)
    logger.info("Vihente Forge starting up (mock=%s, debug=%s)", args.mock, args.debug)

    app = QApplication(argv)
    app.setApplicationName("Vihente Forge")
    app.setOrganizationName("Bru")
    app.setApplicationDisplayName("Vihente Forge")

    icon_path = get_assets_dir() / "icons" / "app.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    _apply_stylesheet(app)

    cfg = AppConfig.load()
    logger.info(
        "AppConfig: vram_mode=%s port=%d thermal=%s",
        cfg.comfy_vram_mode, cfg.comfy_port, cfg.thermal_safety_enabled,
    )

    # Import qui per non pagare il costo se solo --help
    from src.ui.main_window import MainWindow

    window = MainWindow(
        mock=args.mock,
        skip_model_check=args.skip_model_check,
        app_config=cfg,
    )
    window.show()

    return app.exec()