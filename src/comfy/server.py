"""Gestione del processo ComfyUI headless.

Avvia ComfyUI come subprocess, monitora la salute, lo termina pulito
alla chiusura. Mai lasciare processi orfani.

Riferimento: docs/COMFY_ENGINE.md
"""
from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.utils.paths import get_app_data_dir, get_models_dir, get_user_data_dir

logger = logging.getLogger(__name__)


# Profili VRAM → flag CLI di ComfyUI. "normalvram" (default storico della
# nostra config) e "normal" non hanno più un flag dedicato: il profilo
# bilanciato è semplicemente l'assenza di flag. Mappiamo solo quelli reali.
_VRAM_FLAGS: dict[str, str] = {
    "lowvram": "--lowvram",
    "novram": "--novram",
    "highvram": "--highvram",
    "gpu-only": "--gpu-only",
    "cpu": "--cpu",
}


def find_free_port(start: int = 8188, attempts: int = 20) -> int:
    """Trova una porta libera a partire da `start`."""
    for offset in range(attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"Nessuna porta libera trovata da {start}")


class ComfyServer:
    """Controlla il ciclo di vita del processo ComfyUI."""

    def __init__(
        self,
        vram_mode: str = "normalvram",
        preferred_port: Optional[int] = None,
    ) -> None:
        self.vram_mode = vram_mode
        self.preferred_port = preferred_port
        self.port: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None

    @property
    def comfy_dir(self) -> Path:
        return get_user_data_dir() / "engine" / "ComfyUI"

    @property
    def input_dir(self) -> Path:
        """Cartella input di ComfyUI: i LoadImage leggono i file da qui."""
        return self.comfy_dir / "input"

    def is_installed(self) -> bool:
        return (self.comfy_dir / "main.py").exists()

    def start(self, timeout: float = 60.0) -> int:
        """Avvia ComfyUI, attende che risponda, ritorna la porta."""
        if not self.is_installed():
            raise RuntimeError(
                f"ComfyUI non installato in {self.comfy_dir}. "
                "Eseguire prima il setup engine."
            )

        # Cleanup eventuali processi orfani precedenti
        self._kill_orphans()

        self.port = find_free_port(start=self.preferred_port or 8188)
        log_dir = get_app_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_dir / "comfyui.log", "w", encoding="utf-8")

        extra_paths_config = self._write_extra_model_paths_config()

        cmd = [
            sys.executable,
            "main.py",
            "--listen", "127.0.0.1",
            "--port", str(self.port),
            "--disable-auto-launch",
            "--use-pytorch-cross-attention",
            "--extra-model-paths-config", str(extra_paths_config),
        ]
        # ComfyUI ha rimosso --normalvram: il profilo "normale" è l'assenza
        # di flag. Passiamo un flag SOLO per i profili che ne hanno ancora uno
        # (low/no/high vram, gpu-only, cpu); "normalvram"/"normal"/"" → default.
        vram_flag = _VRAM_FLAGS.get(self.vram_mode)
        if vram_flag:
            cmd.append(vram_flag)
        logger.info("Avvio ComfyUI: %s (porta %d)", " ".join(cmd), self.port)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.comfy_dir),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

        if not self._wait_until_ready(timeout):
            self.stop()
            raise RuntimeError("ComfyUI non risponde entro il timeout")

        logger.info("ComfyUI pronto su porta %d", self.port)
        return self.port

    def stop(self) -> None:
        """Termina ComfyUI pulito (SIGTERM, poi SIGKILL)."""
        if self._proc is None:
            return
        logger.info("Arresto ComfyUI...")
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("ComfyUI non risponde a terminate, kill forzato")
            self._proc.kill()
            self._proc.wait()
        self._proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _wait_until_ready(self, timeout: float) -> bool:
        from src.comfy.client import ComfyClient

        client = ComfyClient(port=self.port)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                logger.error("ComfyUI terminato prematuramente (vedi comfyui.log)")
                return False
            if client.is_alive():
                return True
            time.sleep(1.0)
        return False

    def _write_extra_model_paths_config(self) -> Path:
        """Scrive il config YAML che fa vedere a ComfyUI i modelli "ufficiali"
        di Vihente Forge (in ``VFORGE_DATA_DIR/models/``), in aggiunta alla sua
        cartella nativa ``ComfyUI/models/``. Niente symlink/junction: ComfyUI
        supporta nativamente più root via ``--extra-model-paths-config``, ed
        è l'unico approccio che non richiede privilegi admin su Windows.
        """
        models_dir = get_models_dir()
        for sub in ("checkpoints", "loras", "vae"):
            (models_dir / sub).mkdir(parents=True, exist_ok=True)

        config_path = get_app_data_dir() / "comfy_extra_model_paths.yaml"
        base_path = models_dir.resolve().as_posix()
        config_path.write_text(
            "vihente_forge:\n"
            f"    base_path: {base_path}\n"
            "    checkpoints: checkpoints\n"
            "    loras: loras\n"
            "    vae: vae\n",
            encoding="utf-8",
        )
        return config_path

    def _kill_orphans(self) -> None:
        """Termina processi ComfyUI orfani (da crash precedenti)."""
        try:
            import psutil
        except ImportError:
            return
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if "main.py" in " ".join(cmdline) and any("ComfyUI" in str(c) for c in cmdline):
                    logger.warning("Trovato ComfyUI orfano pid=%s, termino", proc.info["pid"])
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
