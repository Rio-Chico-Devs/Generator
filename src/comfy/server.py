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

from src.utils.paths import get_app_data_dir, get_user_data_dir

logger = logging.getLogger(__name__)


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

    def __init__(self, vram_mode: str = "normalvram") -> None:
        self.vram_mode = vram_mode
        self.port: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None

    @property
    def comfy_dir(self) -> Path:
        return get_user_data_dir() / "engine" / "ComfyUI"

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

        self.port = find_free_port()
        log_dir = get_app_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_dir / "comfyui.log", "w", encoding="utf-8")

        cmd = [
            sys.executable,
            "main.py",
            "--listen", "127.0.0.1",
            "--port", str(self.port),
            "--disable-auto-launch",
            f"--{self.vram_mode}",
            "--use-pytorch-cross-attention",
        ]
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

    def _kill_orphans(self) -> None:
        """Termina processi ComfyUI orfani (da crash precedenti)."""
        try:
            import psutil
        except ImportError:
            return
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if any("ComfyUI" in str(c) and "main.py" in " ".join(cmdline) for c in cmdline):
                    logger.warning("Trovato ComfyUI orfano pid=%s, termino", proc.info["pid"])
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
