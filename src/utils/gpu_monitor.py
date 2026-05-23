"""Monitoraggio GPU per Vihente Forge.

Legge temperatura, VRAM, utilizzo e consumo via nvidia-smi (sempre
presente con i driver NVIDIA, nessuna dipendenza extra). Fornisce:
- snapshot istantaneo per la status bar
- soglie di sicurezza configurabili (warning + pausa training)

Le GPU hanno già protezioni hardware (throttling, shutdown). Questo
modulo è un layer di visibilità e tranquillità in più, non sostituisce
le protezioni del firmware.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThermalState(str, Enum):
    COOL = "cool"        # < 70°C
    NORMAL = "normal"    # 70-83°C — pieno carico sano
    WARM = "warm"        # 83-87°C — caldo ma sicuro
    HOT = "hot"          # > 87°C — il throttling hardware interviene


@dataclass
class GpuSnapshot:
    available: bool
    name: str = ""
    temperature_c: int = 0
    vram_used_mb: int = 0
    vram_total_mb: int = 0
    utilization_pct: int = 0
    power_w: float = 0.0
    power_limit_w: float = 0.0

    @property
    def vram_used_gb(self) -> float:
        return self.vram_used_mb / 1024

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def vram_pct(self) -> int:
        if self.vram_total_mb == 0:
            return 0
        return round(100 * self.vram_used_mb / self.vram_total_mb)

    @property
    def thermal_state(self) -> ThermalState:
        t = self.temperature_c
        if t < 70:
            return ThermalState.COOL
        if t < 83:
            return ThermalState.NORMAL
        if t < 87:
            return ThermalState.WARM
        return ThermalState.HOT

    def status_line(self) -> str:
        """Stringa compatta per la status bar."""
        if not self.available:
            return "GPU: non disponibile"
        return (
            f"GPU: {self.temperature_c}°C │ "
            f"VRAM: {self.vram_used_gb:.1f}/{self.vram_total_gb:.0f}GB "
            f"({self.vram_pct}%) │ "
            f"Uso: {self.utilization_pct}% │ "
            f"{self.power_w:.0f}W"
        )


def read_gpu() -> GpuSnapshot:
    """Snapshot istantaneo via nvidia-smi. Non solleva eccezioni."""
    query = (
        "name,temperature.gpu,memory.used,memory.total,"
        "utilization.gpu,power.draw,power.limit"
    )
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        # Se più GPU, prende la prima
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        return GpuSnapshot(
            available=True,
            name=parts[0],
            temperature_c=int(float(parts[1])),
            vram_used_mb=int(float(parts[2])),
            vram_total_mb=int(float(parts[3])),
            utilization_pct=int(float(parts[4])),
            power_w=float(parts[5]) if parts[5] not in ("[N/A]", "") else 0.0,
            power_limit_w=float(parts[6]) if parts[6] not in ("[N/A]", "") else 0.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError,
            subprocess.TimeoutExpired, ValueError, IndexError) as e:
        logger.debug("read_gpu fallito: %s", e)
        return GpuSnapshot(available=False)


@dataclass
class SafetyConfig:
    """Soglie di sicurezza opzionali. Ridondanti rispetto all'hardware,
    ma offrono controllo e tranquillità."""

    enabled: bool = True
    warn_temp_c: int = 84          # avviso visivo
    pause_training_temp_c: int = 88  # pausa automatica training
    # max durata sessione training (0 = nessun limite)
    max_session_hours: float = 0.0


def evaluate_safety(snapshot: GpuSnapshot, cfg: SafetyConfig) -> tuple[bool, str]:
    """Valuta se serve un'azione di sicurezza.

    Ritorna (should_pause, message). should_pause=True suggerisce di
    mettere in pausa un training in corso.
    """
    if not cfg.enabled or not snapshot.available:
        return False, ""

    if snapshot.temperature_c >= cfg.pause_training_temp_c:
        return True, (
            f"Temperatura {snapshot.temperature_c}°C oltre la soglia di "
            f"sicurezza ({cfg.pause_training_temp_c}°C). Training in pausa "
            "per raffreddamento."
        )
    if snapshot.temperature_c >= cfg.warn_temp_c:
        return False, (
            f"Temperatura {snapshot.temperature_c}°C: alta ma sicura. "
            "Verifica la ventilazione."
        )
    return False, ""
