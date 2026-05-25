"""Gestione path utente, multi-piattaforma.

Convenzioni:
- App data: ~/.vihente-forge/ (config, logs, cache app)
- User data: ~/Documents/Vihente Forge/ (progetti, modelli — visibile all'utente)
- Models: ~/Documents/Vihente Forge/models/
- Projects: ~/Documents/Vihente Forge/projects/

Tutto override-abile via env var VFORGE_DATA_DIR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def get_assets_dir() -> Path:
    """Cartella asset bundle (immutabile, distribuita con l'app)."""
    return Path(__file__).resolve().parent.parent.parent / "assets"


def get_app_data_dir() -> Path:
    """Cartella dati app (config, logs, cache). Nascosta."""
    if env := os.environ.get("VFORGE_APP_DIR"):
        return Path(env).expanduser().resolve()

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "VihenteForge"
    return Path.home() / ".vihente-forge"


def get_user_data_dir() -> Path:
    """Root cartella dati utente (progetti + modelli). Visibile."""
    if env := os.environ.get("VFORGE_DATA_DIR"):
        return Path(env).expanduser().resolve()

    docs = Path.home() / "Documents"
    if not docs.exists():
        # fallback su home se Documents non esiste (Linux minimali)
        docs = Path.home()

    return docs / "Vihente Forge"


def get_models_dir() -> Path:
    """Cartella modelli AI."""
    if env := os.environ.get("VFORGE_MODELS_DIR"):
        return Path(env).expanduser().resolve()
    return get_user_data_dir() / "models"


def get_projects_dir() -> Path:
    """Cartella radice progetti."""
    if env := os.environ.get("VFORGE_PROJECTS_DIR"):
        return Path(env).expanduser().resolve()
    return get_user_data_dir() / "projects"


def get_archived_dir() -> Path:
    """Progetti archiviati."""
    return get_user_data_dir() / "archived"


def ensure_user_dirs() -> None:
    """Crea le cartelle utente se non esistono."""
    for d in (
        get_app_data_dir(),
        get_user_data_dir(),
        get_user_data_dir() / "engine",
        get_models_dir(),
        get_projects_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)
