"""Entry point Vihente Forge.

IMPORTANTE: questo file setta variabili d'ambiente PRIMA di avviare
ComfyUI e sd-scripts (i processi figli le ereditano). Non spostare
gli import in cima.
"""
from __future__ import annotations

import os
import sys


def _bootstrap_env() -> None:
    """Setta env vars prima di avviare qualsiasi processo figlio."""
    # Disabilita telemetria HuggingFace (usata da ComfyUI e sd-scripts)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Riduce frammentazione VRAM su GPU 8GB (ereditato da ComfyUI)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    # Su Windows xformers/triton usati da ComfyUI possono dare problemi
    if sys.platform == "win32":
        os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")

    # HF cache dir condivisa tra ComfyUI e sd-scripts
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
