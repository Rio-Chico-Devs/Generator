#!/usr/bin/env python3
"""Runner locale: esegue diagnostica + test suite e raccoglie tutto in un
unico file `local_test_report.txt` da passare facilmente per la review.

Uso:
    python run_local_tests.py

Produce:
    local_test_report.txt        — log leggibile (env, check_system, pytest)
    test-results/report.xml      — JUnit XML (generato da pytest, vedi pyproject)

Nessuna dipendenza esterna: solo stdlib. Non fallisce se manca un pezzo,
annota e prosegue, così il report è sempre completo.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "local_test_report.txt"


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Esegue un comando, ritorna (returncode, output combinato)."""
    try:
        p = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError as e:
        return 127, f"[comando non trovato: {e}]"
    except subprocess.TimeoutExpired:
        return 124, f"[TIMEOUT dopo {timeout}s]"
    except Exception as e:  # pragma: no cover
        return 1, f"[errore esecuzione: {e}]"


def _section(fh, title: str) -> None:
    fh.write("\n" + "=" * 70 + "\n")
    fh.write(f"  {title}\n")
    fh.write("=" * 70 + "\n")


def _probe_imports() -> str:
    """Verifica quali dipendenze chiave sono importabili."""
    mods = [
        "PyQt6", "PyQt6.QtCore", "toml", "pydantic", "psutil",
        "PIL", "numpy", "websocket", "pytest", "pytestqt",
    ]
    lines = []
    for m in mods:
        try:
            __import__(m)
            lines.append(f"  [OK]   {m}")
        except Exception as e:
            lines.append(f"  [MANCA] {m}  ({e.__class__.__name__})")
    return "\n".join(lines)


def main() -> int:
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(f"VIHENTE FORGE — Report test locale\n")
        fh.write(f"Generato: {datetime.now().isoformat(timespec='seconds')}\n")

        _section(fh, "1. Ambiente")
        fh.write(f"Python:   {sys.version}\n")
        fh.write(f"Eseguibile: {sys.executable}\n")
        fh.write(f"Piattaforma: {platform.platform()}\n")

        _section(fh, "2. Dipendenze chiave importabili")
        fh.write(_probe_imports() + "\n")

        _section(fh, "3. check_system.py (GPU / VRAM / engine)")
        if (ROOT / "check_system.py").exists():
            rc, out = _run([sys.executable, "check_system.py"], timeout=120)
            fh.write(out + f"\n[exit code: {rc}]\n")
        else:
            fh.write("[check_system.py non trovato]\n")

        _section(fh, "4. pytest (suite completa)")
        rc, out = _run([sys.executable, "-m", "pytest"], timeout=1800)
        fh.write(out + f"\n[pytest exit code: {rc}]\n")
        pytest_rc = rc

        _section(fh, "5. Artefatti")
        xml = ROOT / "test-results" / "report.xml"
        fh.write(f"JUnit XML: {xml} {'(presente)' if xml.exists() else '(ASSENTE)'}\n")
        fh.write(f"Questo report: {REPORT}\n")

    print(f"\nReport scritto in: {REPORT}")
    print("Allega/incolla quel file (e test-results/report.xml se presente).")
    return pytest_rc


if __name__ == "__main__":
    raise SystemExit(main())
