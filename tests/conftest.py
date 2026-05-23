"""Configurazione pytest condivisa.

Isola i path utente in una directory temporanea per ogni test, così la
suite non scrive mai nelle cartelle reali dell'utente
(``~/.vihente-forge``, ``~/Documents/Vihente Forge``).
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("VFORGE_APP_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("VFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VFORGE_MODELS_DIR", str(tmp_path / "data" / "models"))
    monkeypatch.setenv("VFORGE_PROJECTS_DIR", str(tmp_path / "data" / "projects"))
    yield
