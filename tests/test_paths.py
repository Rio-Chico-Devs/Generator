"""Test per src/utils/paths — gestione path multi-piattaforma.

Usa env var per isolare ogni test dalle cartelle reali dell'utente.
Il conftest.py isola già VFORGE_*; qui testiamo i comportamenti specifici
di ogni funzione inclusi override e derivazioni.
"""
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


# ---------------------------------------------------------------------------
# get_user_data_dir
# ---------------------------------------------------------------------------


def test_data_dir_drives_derived_paths():
    d = tempfile.mkdtemp(prefix="vf_paths_")
    with _env(VFORGE_DATA_DIR=d, VFORGE_MODELS_DIR=None, VFORGE_PROJECTS_DIR=None):
        root = Path(d).resolve()
        assert paths.get_user_data_dir() == root
        assert paths.get_models_dir() == root / "models"
        assert paths.get_projects_dir() == root / "projects"


def test_user_data_dir_returns_path_instance():
    assert isinstance(paths.get_user_data_dir(), Path)


# ---------------------------------------------------------------------------
# get_app_data_dir
# ---------------------------------------------------------------------------


def test_app_data_dir_explicit_override_wins():
    d = tempfile.mkdtemp(prefix="vf_app_")
    with _env(VFORGE_APP_DIR=d):
        assert paths.get_app_data_dir() == Path(d).resolve()


def test_app_data_dir_returns_path_instance():
    assert isinstance(paths.get_app_data_dir(), Path)


def test_app_data_dir_default_has_expected_name():
    """Senza override deve contenere 'vihente' nel nome (convenzione nascosta)."""
    with _env(VFORGE_APP_DIR=None):
        p = paths.get_app_data_dir()
        # Su Linux: ~/.vihente-forge; su Windows: %APPDATA%/VihenteForge
        assert "vihente" in p.name.lower() or "vihente" in str(p).lower()


# ---------------------------------------------------------------------------
# get_models_dir
# ---------------------------------------------------------------------------


def test_models_dir_explicit_override_wins():
    d = tempfile.mkdtemp(prefix="vf_models_")
    with _env(VFORGE_MODELS_DIR=d):
        assert paths.get_models_dir() == Path(d).resolve()


def test_models_dir_returns_path_instance():
    assert isinstance(paths.get_models_dir(), Path)


def test_models_dir_derives_from_user_data():
    d = tempfile.mkdtemp(prefix="vf_data_")
    with _env(VFORGE_DATA_DIR=d, VFORGE_MODELS_DIR=None):
        assert paths.get_models_dir() == Path(d).resolve() / "models"


# ---------------------------------------------------------------------------
# get_projects_dir
# ---------------------------------------------------------------------------


def test_projects_dir_explicit_override_wins():
    d = tempfile.mkdtemp(prefix="vf_proj_")
    with _env(VFORGE_PROJECTS_DIR=d):
        assert paths.get_projects_dir() == Path(d).resolve()


def test_projects_dir_derives_from_user_data():
    d = tempfile.mkdtemp(prefix="vf_data_")
    with _env(VFORGE_DATA_DIR=d, VFORGE_PROJECTS_DIR=None):
        assert paths.get_projects_dir() == Path(d).resolve() / "projects"


def test_projects_dir_returns_path_instance():
    assert isinstance(paths.get_projects_dir(), Path)


# ---------------------------------------------------------------------------
# get_archived_dir
# ---------------------------------------------------------------------------


def test_archived_dir_derives_from_user_data():
    d = tempfile.mkdtemp(prefix="vf_data_")
    with _env(VFORGE_DATA_DIR=d, VFORGE_MODELS_DIR=None, VFORGE_PROJECTS_DIR=None):
        root = Path(d).resolve()
        assert paths.get_archived_dir() == root / "archived"


def test_archived_dir_returns_path_instance():
    assert isinstance(paths.get_archived_dir(), Path)


# ---------------------------------------------------------------------------
# get_assets_dir
# ---------------------------------------------------------------------------


def test_assets_dir_points_into_repo():
    a = paths.get_assets_dir()
    assert a.name == "assets"


def test_assets_dir_exists():
    assert paths.get_assets_dir().is_dir()


def test_assets_dir_returns_path_instance():
    assert isinstance(paths.get_assets_dir(), Path)


# ---------------------------------------------------------------------------
# Tutte le funzioni ritornano Path (non str)
# ---------------------------------------------------------------------------


def test_all_functions_return_path_instances():
    for fn in [
        paths.get_assets_dir,
        paths.get_app_data_dir,
        paths.get_user_data_dir,
        paths.get_models_dir,
        paths.get_projects_dir,
        paths.get_archived_dir,
    ]:
        result = fn()
        assert isinstance(result, Path), f"{fn.__name__} ha ritornato {type(result)}"


def test_no_path_contains_tilde():
    """Le path devono essere sempre espanse (no ~ letterale)."""
    for fn in [
        paths.get_app_data_dir,
        paths.get_user_data_dir,
        paths.get_models_dir,
        paths.get_projects_dir,
        paths.get_archived_dir,
    ]:
        assert "~" not in str(fn()), f"{fn.__name__} contiene ~ non espanso"


# ---------------------------------------------------------------------------
# ensure_user_dirs
# ---------------------------------------------------------------------------


def test_ensure_user_dirs_creates_all():
    d = tempfile.mkdtemp(prefix="vf_ensure_")
    with _env(
        VFORGE_APP_DIR=str(Path(d) / "app"),
        VFORGE_DATA_DIR=str(Path(d) / "data"),
        VFORGE_MODELS_DIR=None,
        VFORGE_PROJECTS_DIR=None,
    ):
        paths.ensure_user_dirs()
        assert paths.get_app_data_dir().is_dir()
        assert paths.get_user_data_dir().is_dir()
        assert paths.get_models_dir().is_dir()
        assert paths.get_projects_dir().is_dir()


def test_ensure_user_dirs_idempotent():
    d = tempfile.mkdtemp(prefix="vf_ensure2_")
    with _env(
        VFORGE_APP_DIR=str(Path(d) / "app"),
        VFORGE_DATA_DIR=str(Path(d) / "data"),
        VFORGE_MODELS_DIR=None,
        VFORGE_PROJECTS_DIR=None,
    ):
        paths.ensure_user_dirs()
        paths.ensure_user_dirs()  # seconda chiamata non deve fallire
        assert paths.get_user_data_dir().is_dir()


def test_ensure_user_dirs_creates_engine_subdir():
    d = tempfile.mkdtemp(prefix="vf_engine_")
    with _env(
        VFORGE_APP_DIR=str(Path(d) / "app"),
        VFORGE_DATA_DIR=str(Path(d) / "data"),
        VFORGE_MODELS_DIR=None,
        VFORGE_PROJECTS_DIR=None,
    ):
        paths.ensure_user_dirs()
        assert (paths.get_user_data_dir() / "engine").is_dir()
