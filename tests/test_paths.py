from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from src.utils import paths


@contextmanager
def _env(**overrides: str | None):
    """Imposta/rimuove env var temporaneamente (value None = rimuovi)."""
    prev = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def test_data_dir_drives_derived_paths():
    d = tempfile.mkdtemp(prefix="vf_paths_")
    # Rimuove gli override più specifici per verificare la derivazione.
    with _env(VFORGE_DATA_DIR=d, VFORGE_MODELS_DIR=None, VFORGE_PROJECTS_DIR=None):
        root = Path(d).resolve()
        assert paths.get_user_data_dir() == root
        assert paths.get_models_dir() == root / "models"
        assert paths.get_projects_dir() == root / "projects"


def test_models_dir_explicit_override_wins():
    d = tempfile.mkdtemp(prefix="vf_models_")
    with _env(VFORGE_MODELS_DIR=d):
        assert paths.get_models_dir() == Path(d).resolve()


def test_assets_dir_points_into_repo():
    a = paths.get_assets_dir()
    assert a.name == "assets"
