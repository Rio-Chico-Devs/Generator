"""Entry point Vihente Forge.

IMPORTANTE: questo file setta variabili d'ambiente PRIMA di importare
diffusers/torch. L'ordine conta — non spostare gli import in cima.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_env() -> None:
    """Setta env vars prima di qualsiasi import pesante."""
    # Disabilita telemetria HuggingFace
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Riduce frammentazione VRAM su GPU 8GB
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    # Su Windows xformers a volte ha problemi con triton
    if sys.platform == "win32":
        os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")

    # HF cache dir: la mettiamo nella cartella utente unica per l'app
    from src.utils.paths import get_models_dir

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models_dir))


def main() -> int:
    _bootstrap_env()

    # Import qui, DOPO env setup
    from src.app import run_app

    return run_app(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
