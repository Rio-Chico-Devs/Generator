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
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

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
    """Soglie di sicurezza per la generazione e il training.

    I default sono volutamente conservativi (pensati per GPU di portatile,
    dove il calore si propaga al telaio e alla tastiera). Sono regolabili
    dall'utente tramite AppConfig. La priorità è non lasciare mai la GPU a
    pieno carico continuo: meglio una generazione più lenta che surriscaldare.

    Isteresi: si entra in pausa a ``pause_temp_c`` e si riprende solo sotto
    ``resume_temp_c`` (più basso), così non si oscilla attorno alla soglia.
    """

    enabled: bool = True
    warn_temp_c: int = 72        # avviso visivo (status bar ambra)
    pause_temp_c: int = 78       # entra in pausa: attende raffreddamento
    resume_temp_c: int = 68      # riprende solo quando scende qui sotto
    # Pausa fissa tra un'immagine e l'altra, a prescindere dalla temperatura.
    # Dà alla GPU il tempo di dissipare tra un job e il successivo.
    cooldown_between_images_sec: float = 8.0
    poll_interval_sec: float = 3.0   # ogni quanto rileggere la temperatura in attesa
    max_wait_sec: float = 600.0      # cap sull'attesa di raffreddamento (anti-hang)
    # max durata sessione training (0 = nessun limite)
    max_session_hours: float = 0.0

    def __post_init__(self) -> None:
        # Garantisce l'isteresi: resume deve stare sotto pause.
        if self.resume_temp_c >= self.pause_temp_c:
            self.resume_temp_c = max(0, self.pause_temp_c - 5)


class CooldownOutcome(str, Enum):
    """Esito di un'attesa di raffreddamento."""

    PROCEED = "proceed"    # si può generare (mai entrato in pausa o raffreddata)
    ABORTED = "aborted"    # l'utente ha annullato durante l'attesa
    TIMEOUT = "timeout"    # non si è raffreddata entro max_wait_sec


def evaluate_safety(snapshot: GpuSnapshot, cfg: SafetyConfig) -> tuple[bool, str]:
    """Valuta se serve un'azione di sicurezza, per avvisi non bloccanti.

    Ritorna (should_pause, message). should_pause=True suggerisce di
    sospendere il lavoro in corso. Per la pausa attiva con isteresi durante
    la generazione si usa invece :class:`ThermalGovernor`.
    """
    if not cfg.enabled or not snapshot.available:
        return False, ""

    if snapshot.temperature_c >= cfg.pause_temp_c:
        return True, (
            f"Temperatura {snapshot.temperature_c}°C oltre la soglia di "
            f"sicurezza ({cfg.pause_temp_c}°C). In pausa per raffreddamento."
        )
    if snapshot.temperature_c >= cfg.warn_temp_c:
        return False, (
            f"Temperatura {snapshot.temperature_c}°C: alta ma sicura. "
            "Verifica la ventilazione."
        )
    return False, ""


class ThermalGovernor:
    """Freno termico: blocca la generazione finché la GPU non si raffredda.

    Usato dal worker di generazione *prima* di ogni immagine. Se la GPU è
    sopra la soglia di pausa, attende (con polling) che scenda sotto la
    soglia di ripristino, notificando l'attesa via callback per la UI.

    Non dipende da Qt né da hardware specifico: la lettura GPU è iniettabile
    (``read_fn``), il che rende la classe interamente testabile.
    """

    def __init__(
        self,
        cfg: Optional[SafetyConfig] = None,
        read_fn: Callable[[], GpuSnapshot] = read_gpu,
    ) -> None:
        self.cfg = cfg or SafetyConfig()
        self._read = read_fn

    def wait_until_safe(
        self,
        on_wait: Optional[Callable[[GpuSnapshot], None]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> CooldownOutcome:
        """Attende che la GPU sia abbastanza fredda per generare.

        - ``on_wait(snapshot)``: chiamata a ogni ciclo di attesa (per la UI).
        - ``should_abort()``: se ritorna True, l'attesa termina con ABORTED.

        Ritorna :class:`CooldownOutcome`. PROCEED anche se il monitoraggio
        non è disponibile (es. nessuna GPU NVIDIA): in quel caso non si
        blocca, ci si affida alle protezioni hardware.
        """
        if not self.cfg.enabled:
            return CooldownOutcome.PROCEED

        deadline = time.monotonic() + self.cfg.max_wait_sec
        entered_pause = False

        while True:
            snap = self._read()
            if not snap.available:
                # Nessuna lettura affidabile: non bloccare la generazione.
                return CooldownOutcome.PROCEED
            if not entered_pause and snap.temperature_c < self.cfg.pause_temp_c:
                # Sotto la soglia di pausa: via libera senza attendere.
                return CooldownOutcome.PROCEED
            if snap.temperature_c <= self.cfg.resume_temp_c:
                # Raffreddata a sufficienza (isteresi rispettata).
                return CooldownOutcome.PROCEED
            if should_abort is not None and should_abort():
                return CooldownOutcome.ABORTED
            if time.monotonic() >= deadline:
                logger.warning(
                    "GPU ferma a %d°C oltre %ds: non si raffredda sotto %d°C",
                    snap.temperature_c, int(self.cfg.max_wait_sec),
                    self.cfg.resume_temp_c,
                )
                return CooldownOutcome.TIMEOUT

            entered_pause = True
            logger.info(
                "Freno termico: GPU %d°C, attendo scenda a %d°C",
                snap.temperature_c, self.cfg.resume_temp_c,
            )
            if on_wait is not None:
                on_wait(snap)
            time.sleep(self.cfg.poll_interval_sec)
