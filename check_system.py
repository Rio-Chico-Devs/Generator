"""Diagnostica di sistema per Vihente Forge.

Raccoglie tutte le info tecniche rilevanti in un colpo solo e dice se
l'ambiente è adeguato. Non richiede dipendenze del progetto (usa solo
stdlib), quindi si può lanciare PRIMA di installare requirements.

Uso:
    python check_system.py
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def hr() -> None:
    print("-" * 56)


def check_os() -> None:
    print(f"OS:        {platform.system()} {platform.release()}")
    print(f"Arch:      {platform.machine()}")


def check_python() -> bool:
    v = sys.version_info
    print(f"Python:    {v.major}.{v.minor}.{v.micro}")
    ok = v.major == 3 and v.minor == 11
    if not ok:
        print("           ATTENZIONE: serve Python 3.11.x "
              "(3.12 ha problemi con torch su Windows)")
    return ok


def check_disk() -> bool:
    import os

    home = os.path.expanduser("~")
    total, used, free = shutil.disk_usage(home)
    free_gb = free / (1024 ** 3)
    print(f"Disco lib: {free_gb:.0f} GB liberi su {home}")
    ok = free_gb >= 50
    if not ok:
        print("           ATTENZIONE: consigliati almeno 50 GB liberi")
    return ok


def check_gpu() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
        print(f"GPU:       {out}")

        # Versione CUDA dalla riga header di nvidia-smi
        full = subprocess.check_output(["nvidia-smi"], text=True)
        for line in full.splitlines():
            if "CUDA Version" in line:
                print(f"CUDA:      {line.split('CUDA Version:')[1].strip().split()[0]}")
                break

        # Stima VRAM
        mem = out.split(",")[1].strip()  # es. "12288 MiB"
        mib = int(mem.split()[0])
        gb = mib / 1024
        ok = gb >= 8
        if not ok:
            print(f"           ATTENZIONE: {gb:.1f} GB VRAM, "
                  "consigliati almeno 8 GB")
        return ok
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        print("GPU:       nvidia-smi non trovato")
        print("           STOP: driver NVIDIA mancanti o GPU non NVIDIA.")
        print("           Installa i driver NVIDIA prima di procedere.")
        return False


def check_ram() -> None:
    # senza psutil (non ancora installato): tentativo best-effort
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                text=True,
            )
            n = int("".join(c for c in out if c.isdigit()))
            print(f"RAM:       {n / (1024**3):.0f} GB")
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        print(f"RAM:       {kb / (1024**2):.0f} GB")
                        break
    except Exception:
        print("RAM:       (non rilevabile, verifica manualmente: serve 16+ GB)")


def main() -> int:
    print("=" * 56)
    print("  Vihente Forge — Diagnostica sistema")
    print("=" * 56)
    check_os()
    py_ok = check_python()
    check_ram()
    disk_ok = check_disk()
    hr()
    gpu_ok = check_gpu()
    hr()

    all_ok = py_ok and disk_ok and gpu_ok
    if all_ok:
        print("ESITO: ambiente adeguato. Procedi con docs/DEVELOPMENT.md")
    else:
        print("ESITO: ci sono punti da sistemare (vedi ATTENZIONE/STOP sopra)")
    print("=" * 56)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
