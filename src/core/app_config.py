"""Configurazione applicazione (preferenze globali, non di progetto).

Salvata in ~/.vihente-forge (o APPDATA) come app_config.json.
Gestisce la scelta del launcher, preferenze UI, path custom.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.utils.paths import get_app_data_dir

if TYPE_CHECKING:
    from src.utils.gpu_monitor import SafetyConfig

logger = logging.getLogger(__name__)


class AppMode(str, Enum):
    FORGE = "forge"
    STUDIO = "studio"


@dataclass
class AppConfig:
    schema_version: int = 1
    # Launcher
    last_mode: Optional[str] = None  # "forge" | "studio" | None
    remember_mode: bool = False
    # UI
    theme: str = "dark"
    # ComfyUI engine
    comfy_vram_mode: str = "normalvram"  # normalvram | lowvram | highvram
    comfy_port: int = 8188
    # Sicurezza termica (default conservativi, pensati per laptop)
    thermal_safety_enabled: bool = True
    thermal_warn_temp_c: int = 72
    thermal_pause_temp_c: int = 78
    thermal_resume_temp_c: int = 68
    cooldown_between_images_sec: float = 8.0
    # Privacy
    offline_after_setup: bool = True

    def config_path(self) -> Path:
        return get_app_data_dir() / "app_config.json"

    def thermal_safety(self) -> "SafetyConfig":
        """Costruisce la SafetyConfig per il freno termico dal config utente."""
        from src.utils.gpu_monitor import SafetyConfig

        return SafetyConfig(
            enabled=self.thermal_safety_enabled,
            warn_temp_c=self.thermal_warn_temp_c,
            pause_temp_c=self.thermal_pause_temp_c,
            resume_temp_c=self.thermal_resume_temp_c,
            cooldown_between_images_sec=self.cooldown_between_images_sec,
        )

    def save(self) -> None:
        p = self.config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> AppConfig:
        p = get_app_data_dir() / "app_config.json"
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        except Exception as e:
            logger.warning("Config corrotta, uso default: %s", e)
            return cls()

    def resolve_startup_mode(self) -> Optional[AppMode]:
        """Se l'utente ha scelto 'ricorda', salta il launcher."""
        if self.remember_mode and self.last_mode:
            return AppMode(self.last_mode)
        return None
