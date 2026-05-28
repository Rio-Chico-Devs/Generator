"""Configurazione pytest condivisa.

Isola i path utente in una directory temporanea per ogni test, così la
suite non scrive mai nelle cartelle reali dell'utente
(``~/.vihente-forge``, ``~/Documents/Vihente Forge``).

Report automatici scritti in test-results/ ad ogni run:
  report.xml    — JUnit XML (strutturato, tutti i dettagli)
  summary.json  — panoramica rapida leggibile da Claude
  latest.log    — output verboso completo (tb=long)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

_RESULTS_DIR = Path(__file__).parent.parent / "test-results"


# ---------------------------------------------------------------------------
# Isolamento path utente (autouse — ogni test)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_user_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("VFORGE_APP_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("VFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VFORGE_MODELS_DIR", str(tmp_path / "data" / "models"))
    monkeypatch.setenv("VFORGE_PROJECTS_DIR", str(tmp_path / "data" / "projects"))
    yield


# ---------------------------------------------------------------------------
# Raccolta risultati per il summary JSON
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Assicura che test-results/ esista prima che pytest scriva il JUnit XML."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Scrive summary.json e latest.log dopo ogni run completo.

    Tutto deriva dai report già raccolti da pytest: nessun subprocess,
    nessun re-run (eviterebbe ricorsione infinita)."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stats = terminalreporter.stats
    passed  = [r for r in stats.get("passed",  []) if hasattr(r, "nodeid")]
    failed  = [r for r in stats.get("failed",  []) if hasattr(r, "nodeid")]
    errored = [r for r in stats.get("error",   []) if hasattr(r, "nodeid")]
    skipped = [r for r in stats.get("skipped", []) if hasattr(r, "nodeid")]

    def _failure_detail(report) -> dict:
        return {
            "nodeid": report.nodeid,
            "message": _extract_message(report),
            "duration_s": round(getattr(report, "duration", 0.0), 3),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "exit_code": int(exitstatus),
        "totals": {
            "passed":  len(passed),
            "failed":  len(failed),
            "errored": len(errored),
            "skipped": len(skipped),
            "total":   len(passed) + len(failed) + len(errored) + len(skipped),
        },
        "status": "GREEN" if exitstatus == 0 else "RED",
        "failures": [_failure_detail(r) for r in failed + errored],
        "slowest": _slowest(passed + failed, n=10),
    }

    summary_path = _RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_verbose_log(passed, failed, errored, skipped, exitstatus)


def _extract_message(report) -> str:
    """Estrae il messaggio di errore significativo da un report fallito."""
    if hasattr(report, "longreprtext"):
        return report.longreprtext.strip()[:2000]
    if hasattr(report, "longrepr"):
        lr = report.longrepr
        if hasattr(lr, "reprcrash") and lr.reprcrash:
            return str(lr.reprcrash.message)[:2000]
        return str(lr)[:2000]
    return ""


def _slowest(reports, n: int = 10) -> list[dict]:
    timed = [(r.nodeid, getattr(r, "duration", 0.0)) for r in reports]
    timed.sort(key=lambda x: x[1], reverse=True)
    return [{"nodeid": nid, "duration_s": round(d, 3)} for nid, d in timed[:n]]


def _write_verbose_log(passed, failed, errored, skipped, exitstatus) -> None:
    """Scrive latest.log: esito per test + traceback completo dei fallimenti.

    Costruito dai report già in memoria — niente subprocess né ricorsione."""
    lines: list[str] = []
    status = "GREEN" if exitstatus == 0 else "RED"
    lines.append(f"=== Vihente Forge — Test Run [{status}] ===")
    lines.append(
        f"passed={len(passed)} failed={len(failed)} "
        f"errored={len(errored)} skipped={len(skipped)}"
    )
    lines.append("")

    lines.append("--- Esiti ---")
    for r in passed:
        lines.append(f"PASS  {r.nodeid}")
    for r in skipped:
        lines.append(f"SKIP  {r.nodeid}")
    for r in failed:
        lines.append(f"FAIL  {r.nodeid}")
    for r in errored:
        lines.append(f"ERROR {r.nodeid}")

    if failed or errored:
        lines.append("")
        lines.append("--- Dettaglio fallimenti ---")
        for r in failed + errored:
            lines.append("")
            lines.append(f"### {r.nodeid}")
            lines.append(_extract_message(r) or "(nessun traceback disponibile)")

    log_path = _RESULTS_DIR / "latest.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
