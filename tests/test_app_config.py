"""Test per src/core/app_config — configurazione applicazione globale.

Testa: default, save/load, resilienza a JSON corrotto, AppMode enum,
resolve_startup_mode, thermal_safety().
"""
from __future__ import annotations

import json

from src.core.app_config import AppConfig, AppMode
from src.utils.paths import get_app_data_dir


# ---------------------------------------------------------------------------
# Valori di default
# ---------------------------------------------------------------------------


def test_default_theme():
    assert AppConfig().theme == "dark"


def test_default_comfy_port():
    assert AppConfig().comfy_port == 8188


def test_default_thermal_enabled():
    assert AppConfig().thermal_safety_enabled is True


def test_default_offline_mode():
    assert AppConfig().offline_after_setup is True


def test_default_schema_version():
    assert AppConfig().schema_version == 1


def test_default_last_mode_is_none():
    assert AppConfig().last_mode is None


def test_default_remember_mode_false():
    assert AppConfig().remember_mode is False


def test_default_thermal_temperatures():
    c = AppConfig()
    assert c.thermal_warn_temp_c == 72
    assert c.thermal_pause_temp_c == 78
    assert c.thermal_resume_temp_c == 68


def test_default_cooldown():
    assert AppConfig().cooldown_between_images_sec == 8.0


def test_default_vram_mode():
    assert AppConfig().comfy_vram_mode == "normalvram"


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_basic():
    c = AppConfig(theme="light", comfy_port=9000)
    c.save()
    loaded = AppConfig.load()
    assert loaded.theme == "light"
    assert loaded.comfy_port == 9000


def test_save_load_roundtrip_thermal():
    c = AppConfig(
        thermal_warn_temp_c=75,
        thermal_pause_temp_c=82,
        thermal_resume_temp_c=65,
        cooldown_between_images_sec=15.0,
    )
    c.save()
    loaded = AppConfig.load()
    assert loaded.thermal_warn_temp_c == 75
    assert loaded.thermal_pause_temp_c == 82
    assert loaded.thermal_resume_temp_c == 65
    assert loaded.cooldown_between_images_sec == 15.0


def test_save_load_roundtrip_flags():
    c = AppConfig(
        thermal_safety_enabled=False,
        offline_after_setup=False,
        remember_mode=True,
        last_mode="studio",
    )
    c.save()
    loaded = AppConfig.load()
    assert loaded.thermal_safety_enabled is False
    assert loaded.offline_after_setup is False
    assert loaded.remember_mode is True
    assert loaded.last_mode == "studio"


def test_save_load_roundtrip_vram_mode():
    c = AppConfig(comfy_vram_mode="lowvram")
    c.save()
    assert AppConfig.load().comfy_vram_mode == "lowvram"


def test_save_creates_config_file():
    AppConfig().save()
    assert (get_app_data_dir() / "app_config.json").exists()


def test_save_creates_valid_json():
    AppConfig(theme="light").save()
    raw = (get_app_data_dir() / "app_config.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["theme"] == "light"


def test_save_twice_last_wins():
    AppConfig(theme="dark").save()
    AppConfig(theme="light").save()
    assert AppConfig.load().theme == "light"


# ---------------------------------------------------------------------------
# Resilienza
# ---------------------------------------------------------------------------


def test_load_missing_file_returns_defaults():
    loaded = AppConfig.load()
    assert loaded == AppConfig()


def test_load_empty_json_returns_defaults():
    p = get_app_data_dir() / "app_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")
    loaded = AppConfig.load()
    assert loaded == AppConfig()


def test_load_corrupted_file_returns_defaults():
    p = get_app_data_dir() / "app_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json", encoding="utf-8")
    loaded = AppConfig.load()
    assert loaded == AppConfig()


def test_load_unknown_keys_ignored():
    """Chiavi future nel JSON non devono far crashare il load."""
    from dataclasses import asdict
    data = asdict(AppConfig())
    data["future_key_v99"] = "whatever"
    p = get_app_data_dir() / "app_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    loaded = AppConfig.load()
    assert loaded.theme == "dark"


def test_load_partial_json_uses_defaults_for_missing():
    p = get_app_data_dir() / "app_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"theme": "light"}), encoding="utf-8")
    loaded = AppConfig.load()
    assert loaded.theme == "light"
    assert loaded.comfy_port == 8188  # default


# ---------------------------------------------------------------------------
# config_path
# ---------------------------------------------------------------------------


def test_config_path_in_app_dir():
    c = AppConfig()
    assert c.config_path() == get_app_data_dir() / "app_config.json"


def test_config_path_is_json():
    assert AppConfig().config_path().suffix == ".json"


# ---------------------------------------------------------------------------
# resolve_startup_mode
# ---------------------------------------------------------------------------


def test_resolve_startup_mode_no_remember():
    c = AppConfig(remember_mode=False, last_mode="forge")
    assert c.resolve_startup_mode() is None


def test_resolve_startup_mode_remember_forge():
    c = AppConfig(remember_mode=True, last_mode="forge")
    assert c.resolve_startup_mode() == AppMode.FORGE


def test_resolve_startup_mode_remember_studio():
    c = AppConfig(remember_mode=True, last_mode="studio")
    assert c.resolve_startup_mode() == AppMode.STUDIO


def test_resolve_startup_mode_remember_but_no_last_mode():
    c = AppConfig(remember_mode=True, last_mode=None)
    assert c.resolve_startup_mode() is None


def test_resolve_startup_mode_defaults_to_none():
    assert AppConfig().resolve_startup_mode() is None


# ---------------------------------------------------------------------------
# thermal_safety() → SafetyConfig
# ---------------------------------------------------------------------------


def test_thermal_safety_enabled():
    sc = AppConfig(thermal_safety_enabled=True).thermal_safety()
    assert sc.enabled is True


def test_thermal_safety_disabled():
    sc = AppConfig(thermal_safety_enabled=False).thermal_safety()
    assert sc.enabled is False


def test_thermal_safety_temperatures():
    c = AppConfig(
        thermal_warn_temp_c=70,
        thermal_pause_temp_c=80,
        thermal_resume_temp_c=62,
    )
    sc = c.thermal_safety()
    assert sc.warn_temp_c == 70
    assert sc.pause_temp_c == 80
    assert sc.resume_temp_c == 62


def test_thermal_safety_cooldown():
    sc = AppConfig(cooldown_between_images_sec=20.0).thermal_safety()
    assert sc.cooldown_between_images_sec == 20.0


# ---------------------------------------------------------------------------
# AppMode enum
# ---------------------------------------------------------------------------


def test_app_mode_forge_value():
    assert AppMode.FORGE == "forge"


def test_app_mode_studio_value():
    assert AppMode.STUDIO == "studio"


def test_app_mode_from_string():
    assert AppMode("forge") == AppMode.FORGE
    assert AppMode("studio") == AppMode.STUDIO
